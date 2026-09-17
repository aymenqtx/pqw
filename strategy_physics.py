"""Decision-local Formula forecasts priced by the active lap-time solver.

Car specification and commands are held at the decision snapshot. Unknown future
traffic/incidents aren't invented. One currently-known caution lap is forecast,
followed by green running; no hidden safety-car duration is read.
"""
from collections import OrderedDict
from dataclasses import replace
import math

from .track_rubber import rubber_grip_multiplier
from .tyre_model import TyreModel, _safe_float

_STANDARD_TYRE_METHODS = {name: getattr(TyreModel, name) for name in (
    "grip_multipliers", "wear_grip_factors", "wear_rate", "cliff_onset",
    "thermal_rates", "temperature_window_c", "state_for_conditions",
    "get_compound_data", "supplier_pace_rating", "supplier_durability_rating",
    "normalize_compound_name")}


def available(rm):
    return (getattr(rm, "_physics_series_mode", "formula") != "oval"
            and bool(getattr(rm, "use_realistic_physics", False))
            and bool(getattr(getattr(rm, "realistic_physics", None), "enabled", False)))


class PhysicsStrategyScorer:
    STATE_LIMIT = 32768
    LAP_LIMIT = 65536

    # ---- forecast lattice -------------------------------------------------
    # Resolution at which the planner asks the lap solver about a future lap.
    # Set any of these to 0.0 to disable that axis.
    #
    # fuel:  the only axis with a real cost.  0.05 kg is ~0.0008 s of lap time
    #        at the usual ~0.03 s/kg, i.e. ~0.03 s over a 40-lap projection.
    # temp:  grip is exactly 1.0x while the tyre is inside its window, and the
    #        out-of-window slopes are 5e-5 (cold) and 2.5e-4 (hot) per degree,
    #        so 0.5 C is at most 1.25e-4 of grip and usually exactly nothing.
    # wear:  scales tyre_model.wear_quantum, which wear_grip_factors already
    #        rounds to for every compound without a cliff onset.  At 1.0 those
    #        compounds are unaffected; cliff compounds get wear_quantum/10.
    LATTICE_FUEL_KG = 0.05
    LATTICE_TEMP_C = 0.5
    LATTICE_WEAR_SCALE = 1.0
    LATTICE_WEAR_CLIFF_SCALE = 0.1

    def __init__(self, rm):
        self.rm = rm
        self.drivers = {}
        self.rollouts = OrderedDict()
        self.state_count = 0
        # Share lap solver cache across drivers on rm so teammates / identical laps hit instantly
        if not hasattr(rm, "_strategy_lap_cache"):
            rm._strategy_lap_cache = OrderedDict()
        self.lap_times = rm._strategy_lap_cache
        self.evaluations = 0
        self.batches = 0
        self.collecting = False
        self.pending = {}
        self.contexts = {}
        self.totals = OrderedDict()
        self.requests = OrderedDict()
        self.request_states = 0
        self.total_hits = 0
        self.grip_contexts = {}
        self._sector_fractions = tuple(rm._sector_fraction(i) for i in range(len(rm.sector_lengths)))
        # Counts inputs deferred because they were not already cached, and
        # lap times served straight out of the shared solver cache.
        self.deferred_misses = 0
        self.lap_hits = 0
        # Diagnostics only: total batches vs. ones that came from an explicit
        # collecting/flush cycle. (self.batches - self.flush_batches) is the
        # count of eager, unbatched evaluate() calls -- each one pays the
        # solver's fixed per-call overhead alone instead of sharing it.
        self.flush_batches = 0

    def _price_request(self, request):
        indices, inputs, segments, endpoint = request
        values = self.evaluate(inputs)
        from . import strategy as strategy_module
        kernel = getattr(strategy_module._STRATEGY_KERNELS, "price_physics_request", None)
        if kernel is not None:
            return (kernel(indices, values, segments), *endpoint)
        if segments is None:
            total = sum(values)
        else:
            values = dict(zip(indices, values))
            total = values[0]
            for a, b, terms in segments:
                if terms is None:
                    total += values[b]
                else:
                    segment = 0.0
                    for index, weight in terms:
                        segment += weight * values[index]
                    total += segment - values[a]
        return (total, *endpoint)

    def _store_total(self, key, result):
        if not self.collecting:
            self.totals[key] = result
            self.totals.move_to_end(key)
            while len(self.totals) > self.STATE_LIMIT:
                self.totals.popitem(last=False)

    def snapshot(self, driver):
        name = driver.name
        if name in self.drivers:
            return self.drivers[name]
        rm = self.rm
        static = rm._realistic_static_inputs_for_driver(driver)
        base = rm._realistic_lap_inputs_for_driver(driver, apply_consistency=False, apply_aero=False)
        pressure = dict(rm.tyre_pressure_effects(name))
        scale = rm.driver_tyre_management_factor(driver) * driver.tyre_wear_mult()
        scale *= rm._driver_aid_map_value(rm.driver_aid_tyre_wear, name, driver.team, 1.0)
        scale *= rm.driver_pace_modes.tyre_wear_multiplier(name)
        scale *= rm._driver_style_tyre_wear_multiplier(name, driver)
        scale *= rm.track_wear_mult
        caution = bool(getattr(rm, "sc_active", False) or getattr(rm, "vsc_active", False))
        record = {
            "base": base, "pressure": pressure, "scale": scale,
            "shift": rm.driver_tyre_temp_window_shift_c(name),
            "bonus": float(static.get("contract_grip_bonus_mult", 1.0)) * rubber_grip_multiplier(
                rm.track_rubber, rm.track_rubber_grip_gain),
            "caution": caution,
            "wear_caution": bool(getattr(rm, "vsc_active", False) or getattr(rm, "sc_lap_active", {}).get(name, False)),
            "sc_wear": max(0., min(1., rm._safety_car_cfg_float("vsc_tyre_wear_mult", .70))) if getattr(rm, "vsc_active", False) else .5,
            "burn": float(rm.fuel_burn_per_lap.get(name, 0.0)),
            "lap": int(rm.laps.get(name, 0)),
            "weather_lap": rm.weather_lap_index(),
            "fuel": float(rm.fuel_onboard.get(name, 0.0)),
            "ambient": rm._ambient_temperature_for_driver(name),
            "supplier_cache": {},
            "warmup_delta": float(pressure.get("warmup_rate_delta", 0.0)),
            "target_delta": float(pressure.get("target_temp_delta_c", 0.0)),
            "wear_add": float(pressure.get("wear_rate_add", 0.0)),
        }
        self.drivers[name] = record
        return record

    def _supplier_ratings(self, driver_name, compound):
        snap = self.drivers[driver_name]
        cache = snap["supplier_cache"]
        if compound not in cache:
            cache[compound] = self.rm._driver_supplier_ratings(driver_name, compound)
        return cache[compound]

    def wetness(self, driver, lap):
        snap = self.snapshot(driver)
        ahead = min(4, max(0, lap - snap["lap"]))
        return self.rm.lap_wetness(snap["weather_lap"] + ahead)

    def fuel_at(self, driver, lap):
        snap = self.snapshot(driver)
        ahead = max(0, lap - snap["lap"])
        used = snap["burn"] * ahead
        if ahead and snap["caution"] and self.rm.formula_refueling_allowed:
            used -= snap["burn"] * .2
        return max(0.0, snap["fuel"] - used)

    def optimistic_lap(self, driver):
        snap = self.snapshot(driver)
        if "optimistic_lap" not in snap:
            model = self.rm.tyre_model
            lat, longitudinal = 0.0, 0.0
            for compound in model.compound_names():
                pace, _ = self._supplier_ratings(driver.name, compound)
                for index in range(5):
                    a, b = model.grip_multipliers(compound,
                        wetness_mm=self.rm.lap_wetness(snap["weather_lap"]+index), wear=0.,
                        supplier_pace_rating=pace, contract_grip_bonus_mult=snap["bonus"])
                    lat, longitudinal = max(lat, a), max(longitudinal, b)
            inputs = replace(snap["base"], fuel_mass_kg=0., tyre_lat_mult=lat, tyre_long_mult=longitudinal)
            snap["optimistic_lap"] = max(0., self.evaluate([inputs])[0] - .05)
        return snap["optimistic_lap"]

    def _lattice_wear(self, compound, wear):
        """Snap forecast wear.

        TyreModel.wear_grip_factors already rounds wear to wear_quantum for any
        compound without a cliff onset, so for those compounds this is exactly
        a no-op on grip.  Compounds that do define an onset use a continuous
        slope, so they get a much finer step.
        """
        if not self.LATTICE_WEAR_SCALE:
            return wear
        model = self.rm.tyre_model
        step = float(getattr(model, "wear_quantum", 0.02) or 0.02)
        if model.cliff_onset(compound) is not None:
            step *= self.LATTICE_WEAR_CLIFF_SCALE
        step *= self.LATTICE_WEAR_SCALE
        return round(float(wear) / step) * step if step > 0.0 else wear

    def inputs(self, driver, compound, wear, fuel, temperature, lap):
        snap = self.snapshot(driver)
        pace, _ = self._supplier_ratings(driver.name, compound)
        # Snap the forecast state onto a lattice so that two projections of the
        # same future lap made one race lap apart ask the solver the same
        # question, and hit the shared cache instead of re-solving.  This
        # changes the resolution of the planner's forecast only; the lap times
        # the cars actually run never pass through here.
        fuel = max(0.0, float(fuel))
        if self.LATTICE_FUEL_KG > 0.0:
            fuel = round(fuel / self.LATTICE_FUEL_KG) * self.LATTICE_FUEL_KG
        if self.LATTICE_TEMP_C > 0.0:
            temperature = round(float(temperature) / self.LATTICE_TEMP_C) * self.LATTICE_TEMP_C
        wear = self._lattice_wear(compound, wear)
        lat, longitudinal = self.rm.tyre_model.grip_multipliers(
            compound, wetness_mm=self.wetness(driver, lap), wear=wear,
            tyre_temp_c=temperature, temp_window_shift_c=snap["shift"],
            supplier_pace_rating=pace, contract_grip_bonus_mult=snap["bonus"],
        )
        return replace(snap["base"], fuel_mass_kg=fuel,
                       tyre_lat_mult=lat, tyre_long_mult=longitudinal)

    def advance(self, driver, compound, wear, fuel, temperature, lap):
        """Same sector ordering and modifiers as RaceManager._apply_sector_usage."""
        from . import strategy as strategy_module
        kernel = getattr(strategy_module._STRATEGY_KERNELS, "project_physics_stint_states", None)
        if kernel is not None:
            context = self.projection_context(driver, compound)
            if context is not None:
                row = kernel(context[0], wear, fuel, temperature, lap, 1)[0]
                return tuple(row[3:6])
        from .race_manager import (TYRE_TEMP_STEP_C_PER_LAP, TYRE_TEMP_WET_TARGET_THRESHOLD_MM,
                                   TYRE_TEMP_TARGET_WET_OFFSET_C, TYRE_TEMP_TARGET_DRY_OFFSET_C)
        rm, snap = self.rm, self.snapshot(driver)
        wet = self.wetness(driver, lap)
        warm, cool = rm.tyre_model.thermal_rates(compound)
        warm = max(.1, warm + snap["warmup_delta"])
        offset = TYRE_TEMP_TARGET_WET_OFFSET_C if float(wet or 0.0) > TYRE_TEMP_WET_TARGET_THRESHOLD_MM else TYRE_TEMP_TARGET_DRY_OFFSET_C
        target = snap["ambient"] + offset + snap["target_delta"]
        _, durability = self._supplier_ratings(driver.name, compound)
        caution = snap["caution"] and lap == snap["lap"]
        burn_rate = snap["burn"] * (.8 if caution and rm.formula_refueling_allowed else 1.)
        wear_caution = snap["wear_caution"] and lap == snap["lap"]
        wear_scale = snap["scale"] * (snap["sc_wear"] if wear_caution else 1.)
        wear_add = snap["wear_add"]
        shift = snap["shift"]
        wear_rate_func = rm.tyre_model.wear_rate

        for fraction in self._sector_fractions:
            fuel = max(0.0, fuel - burn_rate * fraction)
            if fuel < .0005:
                fuel = 0.0
            step = TYRE_TEMP_STEP_C_PER_LAP * fraction
            if temperature < target:
                temperature = min(target, temperature + step * warm)
            elif temperature > target:
                temperature = max(target, temperature - step * cool)
            temperature = max(20.0, min(180.0, temperature))
            rate = wear_rate_func(compound, wetness_mm=wet,
                                  supplier_durability_rating=durability,
                                  tyre_temp_c=temperature, temp_window_shift_c=shift)
            wear += max(0.0, rate + wear_add) * wear_scale * fraction
        return wear, fuel, temperature

    def _standard_tyre_methods(self):
        model = self.rm.tyre_model
        return type(model) is TyreModel and all(
            getattr(getattr(model, name), "__func__", getattr(model, name))
            is getattr(original, "__func__", original)
            for name, original in _STANDARD_TYRE_METHODS.items())

    def _request_grip_context(self, driver, compound, context):
        key = (driver.name, compound)
        if key in self.grip_contexts:
            return self.grip_contexts[key]
        from .realistic_physics import LapInputs
        snap = self.snapshot(driver)
        if (type(snap["base"]) is not LapInputs or not self._standard_tyre_methods()
                or type(self) is not PhysicsStrategyScorer
                or any(getattr(getattr(self, name), "__func__", getattr(self, name)) is not original
                       for name, original in _STANDARD_SCORER_INPUT_METHODS.items())):
            return None
        model = self.rm.tyre_model
        comp = model.get_compound_data(compound)
        lat, longitudinal = [], []
        for index in range(5):
            state = model.state_for_conditions(compound, self.rm.lap_wetness(snap["weather_lap"]+index))
            lat.append(_safe_float(comp.get("lat_grip_base", 1.), 1.) * _safe_float(state.get("lat_mult", 1.), 1.))
            longitudinal.append(_safe_float(comp.get("long_grip_base", 1.), 1.) * _safe_float(state.get("long_mult", 1.), 1.))
        pace, _ = self._supplier_ratings(driver.name, compound)
        bonus = float(snap["bonus"])
        if bonus <= 0.:
            bonus = 1.
        params = (float(comp.get("wear_cliff_mult_lat", 1.)), float(comp.get("wear_cliff_mult_long", 1.)),
                  float(comp.get("wear_grip_loss_lat", .15)), float(comp.get("wear_grip_loss_long", .18)),
                  float(model.wear_quantum), max(.9, 1.+(model.supplier_pace_rating(pace)-50.)*.0001), bonus)
        if not all(math.isfinite(x) for x in (*lat, *longitudinal, *params, context[2], context[3])) or params[4] <= 0:
            return None
        result = (lat, longitudinal, context[2], context[3], context[4], *params)
        self.grip_contexts[key] = result
        return result

    def projection_context(self, driver, compound):
        key = (driver.name, compound)
        if key in self.contexts:
            return self.contexts[key]
        rm, snap = self.rm, self.snapshot(driver)
        model = rm.tyre_model
        if not self._standard_tyre_methods():
            return None
        from .race_manager import (TYRE_TEMP_WET_TARGET_THRESHOLD_MM,
                                   TYRE_TEMP_TARGET_WET_OFFSET_C, TYRE_TEMP_TARGET_DRY_OFFSET_C)
        warm, cool = model.thermal_rates(compound)
        warm = max(.1, warm + snap["warmup_delta"])
        lo, hi = model.temperature_window_c(compound, window_shift_c=snap["shift"])
        _, durability = self._supplier_ratings(driver.name, compound)
        base = max(0., float(model.get_compound_data(compound).get("wear_rate_base", .02))
                   + (50.-model.supplier_durability_rating(durability))*.0001)
        targets, multipliers, bands = [], [], []
        for index in range(5):
            wet = rm.lap_wetness(snap["weather_lap"]+index)
            state = model.state_for_conditions(compound, wet)
            offset = TYRE_TEMP_TARGET_WET_OFFSET_C if float(wet or 0.) > TYRE_TEMP_WET_TARGET_THRESHOLD_MM else TYRE_TEMP_TARGET_DRY_OFFSET_C
            targets.append(snap["ambient"]+offset+snap["target_delta"])
            multipliers.append(float(state.get("wear_mult", 1.)))
            bands.append((state.get("min_mm"), state.get("max_mm")))
        context = (self._sector_fractions,
                   targets, multipliers, warm, cool, hi, base,
                   snap["wear_add"], snap["scale"],
                   snap["burn"], snap["sc_wear"], snap["lap"],
                   snap["caution"] and rm.formula_refueling_allowed, snap["wear_caution"])
        result = (context, bands, lo, hi, model.cliff_onset(compound))
        self.contexts[key] = result
        return result

    # queue an uncached lap inputs without changing collection ordering
    def _queue_missing(self, inputs, *, already_missing=False):
        values = inputs if already_missing else (value for value in inputs if value not in self.lap_times)
        for value in dict.fromkeys(values):
            self.pending[value] = None
            self.deferred_misses += 1
            if len(self.pending) >= self.LAP_LIMIT:
                self.flush()
                self.collecting = True

    # collect a request while reproducing the exact collecting mode score
    # when self.collecting is true, the request now:
    # 1. queues the missing physics inputs
    # 2. reads whatever physics values are already available
    # 3. calculates the same temporary score that collecting mode previously returned
    # 4. does not invoke the expensive _price_request() path
    # for performance purposes, the collection phase does not need to fully price a candidate through the normal physics path
    # which the old code was doing every time
    def _defer_request(self, request):
        indices, inputs, segments, endpoint = request
        self._queue_missing(inputs)
        values = [self.lap_times.get(value, 0.0) for value in inputs]
        from . import strategy as strategy_module
        kernel = getattr(strategy_module._STRATEGY_KERNELS, "price_physics_request", None)
        if kernel is not None:
            return (kernel(indices, values, segments), *endpoint)
        if segments is None:
            total = sum(values)
        else:
            value_map = dict(zip(indices, values))
            total = value_map[0]
            for a, b, terms in segments:
                if terms is None:
                    total += value_map[b]
                else:
                    segment = 0.0
                    for index, weight in terms:
                        segment += weight * value_map[index]
                    total += segment - value_map[a]
        return (total, *endpoint)

    def evaluate(self, inputs):
        missing = list(dict.fromkeys(value for value in inputs if value not in self.lap_times))
        self.lap_hits += len(inputs) - len(missing)
        if self.collecting:
            self._queue_missing(missing, already_missing=True)
            return [self.lap_times.get(value, 0.0) for value in inputs]
        if missing:
            self.batches += 1
            times = self.rm.realistic_physics.compute_lap_times_only(missing)
            if len(times) != len(missing) or any(not math.isfinite(value) or value <= 0 for value in times):
                raise ValueError("Active physics returned an invalid strategy lap time")
            self.evaluations += len(missing)
            self.lap_times.update(zip(missing, times))
        cache = self.lap_times
        result = []
        promote = cache.move_to_end
        for value in inputs:
            result.append(cache[value])
            # Without this the cap below evicts in insertion order, which
            # throws away the laps the planner keeps asking for.
            promote(value)
        while len(cache) > self.LAP_LIMIT:
            cache.popitem(last=False)
        return result

    def flush(self):
        self.collecting = False
        if self.pending:
            missing = list(self.pending)
            self.pending.clear()
            self.flush_batches += 1
            self.evaluate(missing)

    def stint(self, driver, compound, start_wear, laps, start_lap, *, fuel_kg=None,
              new_set=False, sampled=False, temperature_c=None, with_state=False):
        rm = self.rm
        laps = max(0, int(laps))
        if not laps:
            return 0.0, float(start_wear)
        fuel = self.fuel_at(driver, start_lap) if fuel_kg is None else float(fuel_kg)
        temperature = (rm.tyre_model.initial_temperature_c(compound, pit_out=True) if new_set
                       else float(rm.tyre_temp.get(driver.name, rm.tyre_model.initial_temperature_c(compound))))
        if temperature_c is not None and not new_set:
            temperature = float(temperature_c)
        key = (driver.name, compound, float(start_wear), round(fuel, 2), round(temperature, 1), int(start_lap), bool(new_set))
        result_key = (key, laps, bool(sampled))
        if not self.collecting and result_key in self.totals:
            self.total_hits += 1
            self.totals.move_to_end(result_key)
            result = self.totals[result_key]
            return result if with_state else result[:2]
        request = self.requests.get(result_key)
        if request is not None:
            self.requests.move_to_end(result_key)
            result = self._defer_request(request) if self.collecting else self._price_request(request)
            self._store_total(result_key, result)
            return result if with_state else result[:2]
        records = self.rollouts.pop(key, [])
        self.state_count -= len(records)
        wear = float(start_wear)
        if records:
            wear, fuel, temperature = records[-1][1:4]
        from . import strategy as strategy_module
        kernel = getattr(strategy_module._STRATEGY_KERNELS, "project_physics_stint_states", None)
        context = self.projection_context(driver, compound) if kernel is not None else None
        prepare = getattr(strategy_module._STRATEGY_KERNELS, "prepare_physics_request", None)
        grip_context = self._request_grip_context(driver, compound, context) if context is not None and prepare is not None else None
        if grip_context is not None and all(math.isfinite(x) for x in (wear, fuel, temperature)):
            request = prepare(context, grip_context, self.snapshot(driver)["base"],
                              records, wear, fuel, temperature, start_lap, laps, sampled)
            result = self._defer_request(request) if self.collecting else self._price_request(request)
            self._store_total(result_key, result)
            self._retain_request(result_key, request, key, records)
            return result if with_state else result[:2]
        if context is not None and len(records) < laps:
            numeric, bands, lo, hi, onset = context
            begin = len(records)
            for offset, point in enumerate(kernel(numeric, wear, fuel, temperature, start_lap+begin, laps-begin), start=begin):
                sw, sf, st, wear, fuel, temperature, index = point
                phase = (onset is not None and sw >= onset, sw >= 1., st < lo, st > hi)
                records.append([None, wear, fuel, temperature, bands[index], phase, (sw, sf, st, start_lap+offset)])
        for offset in range(len(records), laps):
            lap = start_lap + offset
            start_state = (wear, fuel, temperature, lap)
            wet = self.wetness(driver, lap)
            state = rm.tyre_model.state_for_conditions(compound, wet)
            band = (state.get("min_mm"), state.get("max_mm"))
            onset = rm.tyre_model.cliff_onset(compound)
            lo, hi = rm.tyre_model.temperature_window_c(compound, window_shift_c=self.snapshot(driver)["shift"])
            cliff = (onset is not None and wear >= onset, wear >= 1., temperature < lo, temperature > hi)
            wear, fuel, temperature = self.advance(driver, compound, wear, fuel, temperature, lap)
            records.append([None, wear, fuel, temperature, band, cliff, start_state])
        chosen = records[:laps]
        def lap_inputs(row):
            if row[0] is None:
                row[0] = self.inputs(driver, compound, *row[6])
            return row[0]
        if sampled and laps > 5:
            knots = {0, laps-1}
            for i in range(1, laps):
                if chosen[i][4:6] != chosen[i-1][4:6]:
                    knots.update((i-1, i))
            knots = sorted(knots)
            indices = set(knots)
            for a, b in zip(knots, knots[1:]):
                indices.add((a+b)//2)
            indices = sorted(indices)
            inputs = [lap_inputs(chosen[i]) for i in indices]
            segments = []
            for a, b in zip(knots, knots[1:]):
                distance = b-a
                if distance == 1:
                    segments.append((a, b, None))
                    continue
                m = (a+b)//2
                nodes = (0, m-a, distance)
                count = distance+1
                s1 = distance*count/2
                s2 = distance*count*(2*distance+1)/6
                terms = []
                for i, x in enumerate(nodes):
                    y, z = [v for j, v in enumerate(nodes) if j != i]
                    weight = (s2-(y+z)*s1+y*z*count)/((x-y)*(x-z))
                    terms.append((a+x, weight))
                segments.append((a, b, terms))
        else:
            indices, segments = None, None
            inputs = [lap_inputs(row) for row in chosen]
        request = (indices, inputs, segments, tuple(chosen[-1][1:4]))
        result = self._defer_request(request) if self.collecting else self._price_request(request)
        self._store_total(result_key, result)
        self._retain_request(result_key, request, key, records)
        return result if with_state else result[:2]

    def _retain_request(self, result_key, request, key, records):
        inputs = request[1]
        if len(inputs) <= self.STATE_LIMIT:
            while self.requests and self.request_states + len(inputs) > self.STATE_LIMIT:
                _, removed = self.requests.popitem(last=False)
                self.request_states -= len(removed[1])
            self.requests[result_key] = request
            self.request_states += len(inputs)
        if len(records) <= self.STATE_LIMIT:
            while self.rollouts and self.state_count + len(records) > self.STATE_LIMIT:
                _, removed = self.rollouts.popitem(last=False)
                self.state_count -= len(removed)
            self.rollouts[key] = records
            self.state_count += len(records)


_STANDARD_SCORER_INPUT_METHODS = {name: getattr(PhysicsStrategyScorer, name)
                                 for name in ("inputs", "advance", "wetness")}
