import copy
import json
import math
import os
import time
from contextlib import nullcontext
from functools import wraps
import random
from collections import OrderedDict
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from .formula_pit_service import tyre_service_mean_seconds
from .tyre_model import TyreModel, _safe_float
from .strategy_physics import PhysicsStrategyScorer, available as physics_scoring_available


_NUMERIC_TYRE_METHODS = {
    name: getattr(TyreModel, name) for name in (
        "grip_multipliers", "wear_grip_factors", "wear_rate", "cliff_onset",
        "state_for_conditions", "thermal_rates", "temperature_window_c",
        "get_compound_data", "supplier_pace_rating", "supplier_durability_rating",
    )
}

try:
    from . import strategy_kernels as _STRATEGY_KERNELS
except Exception:
    _STRATEGY_KERNELS = None
if _STRATEGY_KERNELS is not None and getattr(_STRATEGY_KERNELS, "PHYSICS_STRATEGY_API", 0) != 1:
    # An older installed extension cannot perform the two-phase physics search.
    # Use the equivalent Python path until the matching extension is rebuilt.
    _STRATEGY_KERNELS = None


def _projection_call(method):
    @wraps(method)
    def wrapped(self, rm, *args, **kwargs):
        if getattr(self, "_projection_memo", None) is not None:
            return method(self, rm, *args, **kwargs)
        self._projection_memo = {
            "values": OrderedDict(), "prefixes": OrderedDict(), "prefix_states": 0,
            "numeric_contexts": OrderedDict(), "numeric_rows": 0,
            "compound_cat": {},
        }
        model = getattr(rm, "tyre_model", None)
        self._projection_memo["lookup_model"] = model
        scope = getattr(model, "planning_lookup_cache", None)
        try:
            with scope() if callable(scope) else nullcontext():
                return method(self, rm, *args, **kwargs)
        finally:
            physics = self._projection_memo.get("physics")
            if physics is not None and getattr(self, "_strategy_profile_enabled", False):
                for key, value in (("physics_laps", physics.evaluations), ("physics_batches", physics.batches),
                                   ("lap_hits", physics.lap_hits), ("stint_hits", physics.total_hits),
                                   ("flush_batches", physics.flush_batches)):
                    self._strategy_profile_counts[key] = self._strategy_profile_counts.get(key, 0) + value
            self._projection_memo = None
    return wrapped


def _projection_value(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        memo = getattr(self, "_projection_memo", None)
        if memo is None:
            return method(self, *args, **kwargs)
        # These helpers take rm, driver, then scalar inputs and a flat
        # variation dictionary. Preserve exact values without recursively
        # rebuilding/sorting a generic key on every compiled-loop call.
        if kwargs or len(args) < 2:
            return method(self, *args, **kwargs)
        key = (method.__name__, id(args[0]), id(args[1]),
               tuple(tuple(value.items()) if isinstance(value, dict) else value for value in args[2:]))
        try:
            hash(key)
        except TypeError:
            return method(self, *args, **kwargs)
        values = memo["values"]
        if key in values:
            return values[key]
        result = method(self, *args, **kwargs)
        values[key] = result
        # Evict the oldest entry instead of dropping the whole cache, so a
        # decision that crosses the limit does not recompute from scratch.
        while len(values) > 2048:
            values.popitem(last=False)
        return result
    return wrapped


def _profile_decision(method):
    @wraps(method)
    def wrapped(self, rm, driver, *args, **kwargs):
        if not self._strategy_profile_enabled or getattr(self, "_strategy_profile_active", False):
            return method(self, rm, driver, *args, **kwargs)
        self._strategy_profile_active = True
        started = time.perf_counter_ns()
        profile_laps_before = self._strategy_profile_counts.get("physics_laps", 0)
        profile_batches_before = self._strategy_profile_counts.get("physics_batches", 0)
        profile_hits_before = self._strategy_profile_counts.get("lap_hits", 0)
        profile_stint_before = self._strategy_profile_counts.get("stint_hits", 0)
        profile_flush_before = self._strategy_profile_counts.get("flush_batches", 0)
        try:
            return method(self, rm, driver, *args, **kwargs)
        finally:
            self._strategy_profile_active = False
            elapsed = (time.perf_counter_ns() - started) / 1000.0
            self._strategy_profile_counts["decisions"] += 1
            self._strategy_profile_counts["worst_decision_us"] = max(
                self._strategy_profile_counts["worst_decision_us"], elapsed
            )
            try:
                record = {"event": "pit_decision", "driver": driver.name,
                          "lap": rm.laps.get(driver.name, 0), "duration_us": elapsed,
                          "physics_laps_delta": self._strategy_profile_counts.get("physics_laps", 0) - profile_laps_before,
                          "physics_batches_delta": self._strategy_profile_counts.get("physics_batches", 0) - profile_batches_before,
                          "lap_cache": len(getattr(rm, "_strategy_lap_cache", ()) or ()),
                          "lap_hits_delta": self._strategy_profile_counts.get("lap_hits", 0) - profile_hits_before,
                          "stint_hits_delta": self._strategy_profile_counts.get("stint_hits", 0) - profile_stint_before,
                          "flush_batches_delta": self._strategy_profile_counts.get("flush_batches", 0) - profile_flush_before,
                          "counters": self._strategy_profile_counts}
                with open(self._strategy_profile_path, "a", encoding="utf-8") as output:
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
            except (OSError, TypeError, ValueError):
                pass  # Diagnostic output must not interrupt a race.
    return wrapped


class StrategyManager:
    """Plan and execute AI pit strategy based on tyre wear projections."""

    MAX_PLANNED_STOPS = 3
    WEAR_CLIFF_THRESHOLD = 0.60
    WEAR_CLIFF_PENALTY = 1.5
    GRIP_TO_LAP_SECONDS = 25.0
    TYRE_TEMP_STEP_C_PER_LAP = 5.0
    PLAN_NOISE_RANGE = 0.35
    PLAN_CACHE_LIMIT = 256
    STINT_CACHE_LIMIT = 1024
    SC_PIT_FRESH_WEAR_CUTOFF = 0.18
    SC_PIT_LOSS_MULT_RANGE = (0.45, 0.70)
    SC_PIT_GAIN_MARGIN_RANGE = (0.05, 0.18)
    WEATHER_PAYBACK_HORIZON_LAPS = 12
    WEATHER_WRONG_TYRE_BASE_LAPS = 3
    WEATHER_WRONG_TYRE_BASE_LOSS_S = 10.0
    WEATHER_SEVERE_LOSS_PER_LAP_S = 3.0

    def __init__(self, cfg: Optional[dict]):
        self.cfg = cfg or {}
        self.plans: Dict[str, dict] = {}
        self._sc_state = None
        self._wet_band_defs, self._wet_band_pref = self._build_wet_band_metadata()
        self._plan_cache: Dict[str, OrderedDict] = {}
        self._stint_cache: Dict[str, OrderedDict] = {}
        self._weather_suitability_cache: List[dict] = []
        self._weather_cache_race_id: Optional[int] = None
        self._weather_state: Dict[str, dict] = {}
        self.weather_diagnostics: Dict[str, dict] = {}
        self._strategy_profile_enabled = os.environ.get("RACE_STRATEGY_PROFILE", "").lower() in {"1", "true", "yes", "on"}
        self._strategy_profile_path = os.environ.get("RACE_STRATEGY_PROFILE_PATH", "strategy_profile.jsonl")
        self._strategy_profile_counts = {"decisions": 0, "worst_decision_us": 0.0,
                                         "compiled_laps": 0, "prefix_hits": 0}
        # A safety-car decision is made once, when the SC is deployed, then
        # held until the driver's next reachable pit entry.  This prevents the
        # normal lap-by-lap optimiser from creating delayed SC pit waves.
        self._sc_pit_window_active = False
        self._sc_pit_decisions: Dict[str, dict] = {}

    def _physics_scorer(self, rm):
        if not physics_scoring_available(rm):
            return None
        memo = getattr(self, "_projection_memo", None)
        if memo is None:
            return PhysicsStrategyScorer(rm)
        if "physics" not in memo:
            memo["physics"] = PhysicsStrategyScorer(rm)
        return memo["physics"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @_projection_call
    def plan_initial_strategy(self, rm) -> None:
        """Seed a pit strategy for every driver on the grid."""

        if not self._pit_logic_enabled(rm):
            self.plans.clear()
            self._plan_cache.clear()
            self._stint_cache.clear()
            self._weather_state.clear()
            self.weather_diagnostics.clear()
            return

        self.plans.clear()
        self._plan_cache.clear()
        self._stint_cache.clear()
        self._weather_state.clear()
        self.weather_diagnostics.clear()
        self._sc_pit_window_active = False
        self._sc_pit_decisions.clear()
        self._build_weather_suitability_cache(rm)
        for driver in rm.drivers:
            variation = self._variation_for_driver(driver)
            start_comp = rm.tyre_comp.get(
                driver.name, self._default_compound(rm)
            )
            used = set(rm.used_compounds.get(driver.name, {start_comp}))
            used.add(start_comp)
            current_wear = float(rm.tyre_wear.get(driver.name, 0.0))
            plan, _ = self._generate_plan(
                rm,
                driver,
                start_lap=rm.laps.get(driver.name, 0),
                start_comp=start_comp,
                current_wear=current_wear,
                used_compounds=used,
                pits_done=rm.pit_count.get(driver.name, 0),
                variation=variation,
                restrict_defaults=True,
            )
            self.plans[driver.name] = {
                "variation": variation,
                "stints": plan.get("stints", []),
                "pit_loss": plan.get("pit_loss", self._estimate_pit_loss(rm, driver, variation)),
                "baseline_lap": plan.get("baseline_lap"),
                "mandatory_count": plan.get("mandatory_count", self._mandatory_compounds(rm)),
                "completed_pits": rm.pit_count.get(driver.name, 0),
                "plan_time": plan.get("plan_time"),
                "last_lap_seen": rm.laps.get(driver.name, 0),
                "last_replan_tick": getattr(rm, "tick_count", 0),
                "decision_cache_key": None,
                "decision_cache_value": None,
            }
        self._sc_state = self._safety_car_pit_active(rm)

    def update(self, rm, dt_sim: float) -> None:  # noqa: ARG002 (dt_sim used for interface)
        """Adjust stint targets when the race situation changes."""

        if not self._pit_logic_enabled(rm):
            return

        sc_active = self._safety_car_pit_active(rm)
        sc_toggled = self._sc_state is not None and sc_active != self._sc_state

        for driver in rm.drivers:
            plan = self.plans.get(driver.name)
            if plan is None:
                # Late join (should not happen), seed plan now.
                self.plan_initial_strategy(rm)
                plan = self.plans.get(driver.name)
                if plan is None:
                    continue

            # Refresh plan after a completed pit-stop.
            replanned = False
            completed = rm.pit_count.get(driver.name, 0)
            if (
                completed > plan.get("completed_pits", 0)
                and rm.pit_remaining.get(driver.name, 0.0) <= 0.0
            ):
                self._replan_from_state(rm, driver)
                plan = self.plans.get(driver.name)
                replanned = True

            # Re-plan after safety car toggles or very slow laps.
            lap_no = rm.laps.get(driver.name, 0)
            if sc_toggled:
                decision = self._sc_pit_decisions.get(driver.name, {})
                already_evaluated = (sc_active and (
                    decision.get("first_opportunity_recheck", False)
                    or (decision.get("plan_snapshot") is not None
                        and decision["plan_snapshot"] == self._sc_plan_signature(rm, driver))))
                if not replanned and not already_evaluated:
                    self._replan_from_state(rm, driver)
                    plan = self.plans.get(driver.name)
            else:
                last_seen = plan.get("last_lap_seen", lap_no)
                if lap_no > last_seen:
                    lap_time = rm.last_lap_time.get(driver.name)
                    baseline = plan.get("baseline_lap") or lap_time
                    if (
                        lap_time
                        and baseline
                        and lap_time > baseline * 1.18 + 3.0
                        and rm.pit_remaining.get(driver.name, 0.0) <= 0.0
                    ):
                        self._replan_from_state(rm, driver)
                        plan = self.plans.get(driver.name)
                    plan["last_lap_seen"] = lap_no

            plan["last_replan_tick"] = getattr(rm, "tick_count", 0)

        self._sc_state = sc_active

    def begin_safety_car_period(self, rm) -> None:
        """Make and latch each AI driver's decision for this SC period."""

        drivers = list(getattr(rm, "drivers", []) or [])
        if self._pit_logic_enabled(rm) and any(
            str(getattr(driver, "name", "") or "") not in self.plans
            for driver in drivers
        ):
            self.plan_initial_strategy(rm)
        self._sc_pit_window_active = True
        self._sc_pit_decisions.clear()
        weather_signature = self._safety_car_weather_signature(rm)

        for driver in drivers:
            name = str(getattr(driver, "name", "") or "")
            if not name:
                continue
            manual_control = getattr(rm, "_player_manual_control", None)
            if callable(manual_control) and manual_control(name):
                continue

            inactive = bool(
                name in (getattr(rm, "finished", set()) or set())
                or (getattr(rm, "dnf", {}) or {}).get(name, False)
                or float(
                    (getattr(rm, "pit_remaining", {}) or {}).get(name, 0.0)
                    or 0.0
                )
                > 0.0
            )
            should_pit, compound = (False, None)
            self.plans.get(name, {}).pop("_sc_plan_snapshot", None)
            failed = False
            if not inactive:
                self._sc_deployment_pricing = True
                try:
                    should_pit, compound = self._choose_pit_action_now(rm, driver)
                except Exception as exc:
                    failed = True
                    # RaceManager reports strategy errors at the normal pit
                    # boundary.  A failed pre-decision must never stop the SC.
                    report_error = getattr(rm, "_record_strategy_error", None)
                    if callable(report_error):
                        report_error(name, exc)
                    should_pit, compound = (False, None)
                finally:
                    self._sc_deployment_pricing = False

            plan = self.plans.get(name, {}) or {}
            stints = plan.get("stints", []) or []
            current_stint = stints[0] if stints else {}
            try:
                wear_limit = float(current_stint.get("wear_limit", 0.6) or 0.6)
            except Exception:
                wear_limit = 0.6
            self._sc_pit_decisions[name] = {
                "pit": bool(should_pit),
                "compound": compound,
                "consumed": False,
                "wear_limit": max(0.01, wear_limit),
                "weather_signature": weather_signature,
                "cliff_checked": self._cliff_reassessment_due(rm, driver),
                "first_opportunity_passed": bool(inactive),
                "evaluation_failed": failed,
                "plan_snapshot": plan.pop("_sc_plan_snapshot", None),
                "first_opportunity_recheck": (not inactive and not should_pit
                    and float(rm.tyre_wear.get(name, 0.0)) < self.SC_PIT_FRESH_WEAR_CUTOFF
                    and self._sc_planned_stop_nearby(rm, driver)),
            }

    def end_safety_car_period(self) -> None:
        """Release latched decisions when the field returns to green."""

        self._sc_pit_window_active = False
        self._sc_pit_decisions.clear()

    def consume_safety_car_pit_decision(self, driver_name: str) -> None:
        """Mark a committed SC stop as served so it cannot fire twice."""

        decision = self._sc_pit_decisions.get(str(driver_name or ""))
        if not isinstance(decision, dict):
            return
        decision["pit"] = False
        decision["compound"] = None
        decision["consumed"] = True
        decision["first_opportunity_passed"] = True
        decision["first_opportunity_recheck"] = False

    @_profile_decision
    def choose_pit_action(self, rm, driver) -> Tuple[bool, Optional[str]]:
        """Return the current pit action, honoring any latched SC decision."""

        name = str(getattr(driver, "name", "") or "")
        decision = self._sc_pit_decisions.get(name)
        if (
            self._sc_pit_window_active
            and self._safety_car_pit_active(rm)
            and isinstance(decision, dict)
        ):
            override_reason = self._safety_car_commitment_override_reason(
                rm, driver, decision
            )
            # This public call occurs at this driver's eligible pit boundary;
            # deployment uses _choose_pit_action_now directly. Do not use the
            # leader's lap number to decide whether an opportunity was missed.
            if decision.get("evaluation_failed") and not decision.get("first_opportunity_passed"):
                override_reason = "retry"
            if (override_reason is None and decision.get("first_opportunity_recheck")
                    and not decision.get("first_opportunity_passed")):
                override_reason = "first opportunity"
            if override_reason is None:
                decision["first_opportunity_passed"] = True
                return bool(decision.get("pit", False)), decision.get("compound")

            should_pit, compound = self._choose_pit_action_now(rm, driver)
            decision["first_opportunity_passed"] = True
            decision["evaluation_failed"] = False
            decision["first_opportunity_recheck"] = False
            decision["weather_signature"] = self._safety_car_weather_signature(rm)
            plan = self.plans.get(name, {}) or {}
            stints = plan.get("stints", []) or []
            current_stint = stints[0] if stints else {}
            try:
                decision["wear_limit"] = max(
                    0.01,
                    float(
                        current_stint.get(
                            "wear_limit", decision.get("wear_limit", 0.6)
                        )
                        or 0.6
                    ),
                )
            except Exception:
                pass

            # An emergency may add a stop to a prior stay-out decision or
            # update the tyre choice of an existing commitment.  A transient
            # weather re-evaluation is not allowed to cancel an already
            # committed stop.
            if should_pit:
                decision["pit"] = True
                decision["compound"] = compound
                decision["consumed"] = False
            return bool(decision.get("pit", False)), decision.get("compound")

        return self._choose_pit_action_now(rm, driver)

    @_profile_decision
    @_projection_call
    def _choose_pit_action_now(self, rm, driver) -> Tuple[bool, Optional[str]]:
        """Return whether to pit this lap and the desired next compound."""

        if not self._pit_logic_enabled(rm):
            return False, None

        plan = self.plans.get(driver.name)
        if plan is None:
            self.plan_initial_strategy(rm)
            plan = self.plans.get(driver.name)
            if plan is None:
                return False, None

        laps_done = rm.laps.get(driver.name, 0)
        remaining_laps = rm.total_laps - laps_done
        if remaining_laps <= 0:
            return False, None

        current_comp = rm.tyre_comp.get(driver.name, self._default_compound(rm))
        current_wear = float(rm.tyre_wear.get(driver.name, 0.0))
        used_all = set(rm.used_compounds.get(driver.name, {current_comp}))
        used_all.add(current_comp)
        pits_done = rm.pit_count.get(driver.name, 0)
        variation = plan.get("variation") or self._variation_for_driver(driver)
        stints = plan.get("stints", [])
        current_stint = stints[0] if stints else {}
        wear_limit = float(current_stint.get("wear_limit", 0.6) or 0.6)
        planned_laps = max(0, int(current_stint.get("planned_laps", 0) or 0))
        stint_start = int(current_stint["start_lap"] if current_stint.get("start_lap") is not None else laps_done)
        laps_in_stint = max(0, laps_done - stint_start)
        mandatory_count = self._mandatory_compounds(rm)
        used_mandatory = self._unique_mandatory_count(used_all, rm)
        mandatory_needed = max(0, mandatory_count - used_mandatory)

        preferred_category = None
        lookahead = self._projected_wetness(rm, laps_done, 4)
        if not lookahead:
            immediate = rm.lap_wetness(laps_done)
            if immediate is None:
                immediate = rm.current_wetness()
            if immediate is not None:
                try:
                    lookahead = [float(immediate)]
                except Exception:
                    lookahead = []
        current_cat = self._compound_wet_category(current_comp, rm)
        preferred_category = self._preferred_weather_category(rm, current_cat, lookahead)
        weather_available = None
        available_getter = getattr(rm, "formula_available_compounds", None)
        if preferred_category and callable(available_getter):
            weather_available = list(available_getter(driver.name) or [])
            if not any(self._compound_wet_category(c, rm) == preferred_category for c in weather_available):
                self._projection_memo["unavailable_weather_category"] = driver.name
                entry = self._weather_entry(rm) or {}
                start = self._weather_profile_index(rm)
                entry = next((row for row in self._ensure_weather_suitability_cache(rm)[start:start+4]
                              if row.get("category") == preferred_category), entry)
                scores = entry.get("scores", {})
                best = max(weather_available, key=lambda c: scores.get(c, 0.), default=None)
                preferred_category = self._compound_wet_category(best, rm) if best else None
                if preferred_category == current_cat:
                    preferred_category = None
        wrong_state = self._update_wrong_weather_state(
            rm,
            driver,
            current_comp,
            laps_done,
        )

        sc_opportunity = self._safety_car_pit_opportunity(
            rm,
            driver,
            current_wear,
            remaining_laps,
            mandatory_needed,
            preferred_category,
            wear_limit,
        )
        lookahead_sig = tuple(int(round(float(v) * 10.0)) for v in list(lookahead or [])[:4])
        decision_key = (
            int(laps_done),
            int(pits_done),
            str(current_comp),
            round(float(current_wear), 4),
            int(stint_start),
            int(planned_laps),
            int(mandatory_needed),
            str(preferred_category) if preferred_category is not None else None,
            lookahead_sig,
            bool(self._safety_car_pit_active(rm)),
            bool(sc_opportunity),
            int(wrong_state.get("laps", 0) or 0),
            int(round(float(wrong_state.get("loss_s", 0.0) or 0.0) * 10.0)),
        )
        if plan.get("decision_cache_key") == decision_key:
            cached = plan.get("decision_cache_value")
            if isinstance(cached, tuple) and len(cached) == 2:
                return bool(cached[0]), cached[1]

        evaluated_for_hold = False
        def _cache_and_return(should_pit: bool, compound: Optional[str]):
            plan["decision_cache_key"] = decision_key
            plan["decision_cache_value"] = (bool(should_pit), compound)
            if should_pit:
                plan.pop("physics_follow_plan", None)
            elif evaluated_for_hold and self._physics_scorer(rm) is not None:
                self._schedule_physics_recheck(rm, driver, plan)
            return bool(should_pit), compound

        # Fast-path for obvious no-stop laps: preserve full evaluation for
        # weather category transitions, mandatory-compound pressure, or when
        # nearing planned stint end / wear window.
        near_window = (planned_laps == 0) or (laps_in_stint >= max(0, planned_laps - 2))
        wear_alert = current_wear >= wear_limit * 0.90
        mandatory_alert = mandatory_needed > 0 and remaining_laps <= max(3, mandatory_needed + 1)
        weather_alert = preferred_category is not None
        if (not mandatory_alert and not weather_alert and not sc_opportunity
                and self._following_physics_plan(rm, driver, plan)):
            return _cache_and_return(False, None)
        cliff_alert = self._cliff_reassessment_due(rm, driver)
        if not (near_window or wear_alert or cliff_alert or mandatory_alert or weather_alert or sc_opportunity):
            return _cache_and_return(False, None)

        # zero-behavior-change shortcut: if the ONLY trigger is a weather alert
        # (tyres are fresh, far from pit window, far from cliff, or no safety car),
        # check the 2ms weather spell evaluation first
        # If it decides to stay out we don't need to spend 180ms simulating a full 50-lap race plan
        if (
            weather_alert
            and not (near_window or wear_alert or cliff_alert or mandatory_alert or sc_opportunity)
            and preferred_category
            and current_cat != str(preferred_category).lower()
        ):
            base_pit_loss = plan.get("pit_loss", self._estimate_pit_loss(rm, driver, variation))
            quick_comp_options = self._candidate_compounds(rm, laps_done, remaining_laps)
            quick_spell = self._evaluate_cached_weather_spell(
                rm,
                driver,
                current_comp,
                laps_done,
                remaining_laps,
                base_pit_loss,
                preferred_category,
                quick_comp_options,
                variation,
            )
            quick_emergency = self._weather_incompetence_ceiling(
                rm,
                driver,
                current_comp,
                wrong_state,
                quick_comp_options,
                variation,
            )
            # if weather does not demand an immediate pit stop, stay out immediately
            should_hold = self._should_hold_for_weather(
                rm,
                driver,
                current_comp,
                current_wear,
                laps_done,
                base_pit_loss,
                wear_limit,
                quick_comp_options,
            )
            if not quick_emergency and (should_hold or not quick_spell or quick_spell.get("action") != "pit"):
                return _cache_and_return(False, None)

        projection, time_keep = self._generate_plan(
            rm,
            driver,
            start_lap=laps_done,
            start_comp=current_comp,
            current_wear=current_wear,
            used_compounds=used_all,
            pits_done=pits_done,
            variation=variation,
        )
        if projection is None:
            return _cache_and_return(False, None)
        evaluated_for_hold = True

        plan.update(
            {
                "variation": variation,
                "stints": projection.get("stints", []),
                "pit_loss": projection.get("pit_loss", plan.get("pit_loss")),
                "baseline_lap": projection.get("baseline_lap", plan.get("baseline_lap")),
                "mandatory_count": projection.get("mandatory_count", plan.get("mandatory_count")),
                "plan_time": time_keep,
            }
        )
        if self._safety_car_pit_active(rm):
            plan["_sc_plan_snapshot"] = self._sc_plan_signature(rm, driver)

        stints = plan.get("stints", [])
        current_stint = stints[0] if stints else {}
        if not stints:
            return _cache_and_return(False, None)

        wear_limit = float(current_stint.get("wear_limit", wear_limit) or wear_limit)
        planned_laps = max(0, int(current_stint.get("planned_laps", planned_laps) or planned_laps))
        stint_start = int(current_stint["start_lap"] if current_stint.get("start_lap") is not None else stint_start)
        laps_in_stint = max(0, laps_done - stint_start)
        effective_planned = max(1, planned_laps)
        wear_ratio = current_wear / wear_limit if wear_limit > 1e-6 else 1.0
        sc_opportunity = self._safety_car_pit_opportunity(
            rm,
            driver,
            current_wear,
            remaining_laps,
            mandatory_needed,
            preferred_category,
            wear_limit,
        )
        force_pit = False
        force_due_plan = False
        force_due_wear = False
        if planned_laps == 0 and len(stints) > 1:
            force_pit = True
            force_due_plan = True
        if current_wear >= wear_limit * 0.98:
            force_pit = True
            force_due_wear = True

        best_choice = None
        best_choice_any = None
        best_time = None
        best_time_any = None
        comp_options = self._candidate_compounds(rm, laps_done, remaining_laps)
        available_getter = getattr(rm, "formula_available_compounds", None)
        if callable(available_getter):
            available = set(weather_available if weather_available is not None else (available_getter(driver.name) or []))
            comp_options = [compound for compound in comp_options if compound in available]
            if self._projection_memo.get("unavailable_weather_category") == driver.name:
                comp_options = [c for c in self._tyre_compounds(rm) if c in available]
            if not comp_options:
                return _cache_and_return(False, None)
        if mandatory_alert:
            required = [c for c in comp_options if self._counts_toward_mandatory(c, rm)
                        and str(c).lower() not in {str(used).lower() for used in used_all}]
            if required:
                comp_options = required
                force_pit = True
        current_cat = self._compound_wet_category(current_comp, rm)
        override_fallback = None
        if preferred_category:
            targeted = [
                name
                for name in comp_options
                if self._compound_wet_category(name, rm) == preferred_category
            ]
            if targeted:
                ordered = targeted + [c for c in comp_options if c not in targeted]
                comp_options = ordered
                override_fallback = targeted[0]
        base_pit_loss = plan.get("pit_loss", self._estimate_pit_loss(rm, driver, variation))
        pit_loss = self._effective_immediate_pit_loss(
            rm,
            driver,
            variation,
            base_pit_loss,
            sc_opportunity,
        )
        weather_eval = None
        if preferred_category and current_cat != str(preferred_category).lower():
            weather_eval = self._evaluate_weather_transition(
                rm,
                driver,
                current_comp,
                current_wear,
                laps_done,
                remaining_laps,
                pit_loss,
                wear_limit,
                preferred_category,
                comp_options,
            )
        spell_eval = None
        if preferred_category and current_cat != str(preferred_category).lower():
            spell_eval = self._evaluate_cached_weather_spell(
                rm,
                driver,
                current_comp,
                laps_done,
                remaining_laps,
                pit_loss,
                preferred_category,
                comp_options,
                variation,
            )
        weather_gain = None
        best_weather_comp = None
        if weather_eval is None and comp_options:
            weather_gain, best_weather_comp = self._estimate_weather_gain(
                rm,
                driver,
                current_comp,
                current_wear,
                laps_done,
                remaining_laps,
                pit_loss,
                comp_options,
            )
        force_weather = False
        hold_for_weather = bool(weather_eval and weather_eval.get("action") == "hold")
        if spell_eval and spell_eval.get("action") == "pit" and spell_eval.get("compound"):
            force_weather = True
            best_weather_comp = spell_eval.get("compound")
            override_fallback = best_weather_comp
            hold_for_weather = False
        elif weather_eval and weather_eval.get("action") == "pit" and weather_eval.get("compound"):
            force_weather = True
            best_weather_comp = weather_eval.get("compound")
            override_fallback = override_fallback or best_weather_comp
        elif weather_gain is not None and best_weather_comp:
            threshold = self._weather_gain_threshold(remaining_laps, pit_loss)
            if weather_gain > threshold:
                force_weather = True
            if remaining_laps <= 2 and weather_gain < pit_loss * 0.75:
                force_weather = False
            # Avoid thrashing between compounds after an immediate stop: short
            # showers can otherwise trigger back-to-back forced stops when the
            # projected gain is only marginal. Require the stint to run a few
            # laps unless the benefit is overwhelming.
            if force_weather and laps_in_stint < 3:
                early_gain = pit_loss * 0.6
                if weather_gain < early_gain:
                    force_weather = False
            if override_fallback is None:
                override_fallback = best_weather_comp
        force_pit = force_pit or force_weather
        emergency_weather = self._weather_incompetence_ceiling(
            rm,
            driver,
            current_comp,
            wrong_state,
            comp_options,
            variation,
        )
        if emergency_weather and emergency_weather.get("compound"):
            force_weather = True
            force_pit = True
            hold_for_weather = False
            best_weather_comp = emergency_weather.get("compound")
            override_fallback = best_weather_comp
        if (
            not hold_for_weather
            and
            not force_weather
            and self._should_hold_for_weather(
                rm,
                driver,
                current_comp,
                current_wear,
                laps_done,
                pit_loss,
                wear_limit,
                comp_options,
            )
        ):
            force_pit = False
            force_due_plan = False
            force_due_wear = False
        elif hold_for_weather:
            force_pit = False
            force_due_plan = False
            force_due_wear = False

        # if a weather transition or emergency already
        # locked in override_fallback to a different tyre, downstream code is mathematically
        # guaranteed to select it, so skipping the remaining full-race evaluations saves around 500ms
        if (
            force_pit
            and force_weather
            and override_fallback
            and str(override_fallback).lower() != str(current_comp).lower()
        ):
            self._record_weather_diagnostic(
                driver.name,
                current_cat=current_cat,
                preferred_category=preferred_category,
                action="pit",
                compound=override_fallback,
                spell_eval=spell_eval,
                wrong_state=wrong_state,
                emergency=bool(emergency_weather),
            )
            return _cache_and_return(True, override_fallback)


        for comp in comp_options:
            new_used = set(used_all)
            new_used.add(comp)
            candidate_start_wear = self._next_physical_set_wear(
                rm,
                driver.name,
                comp,
            )
            prepared = projection.get("immediate_physics_plans", {}).get(comp)
            if prepared is not None:
                future_plan, future_time = prepared, prepared["plan_time"]
            else:
                future_plan, future_time = self._generate_plan(
                    rm, driver, start_lap=laps_done, start_comp=comp,
                    current_wear=candidate_start_wear, used_compounds=new_used,
                    pits_done=pits_done + 1, variation=variation, consume_start_set=True)
            if future_plan is None:
                continue
            total_time = pit_loss + future_time
            planned_compounds = [s.get("compound") for s in future_plan.get("stints", [])]
            unique_total = self._unique_mandatory_count(list(new_used) + planned_compounds, rm)
            candidate = (comp, future_plan)
            if best_time_any is None or total_time < best_time_any:
                best_time_any = total_time
                best_choice_any = candidate
            if unique_total >= mandatory_count:
                if best_time is None or total_time < best_time:
                    best_time = total_time
                    best_choice = candidate

        if best_choice is None:
            best_choice = best_choice_any
            best_time = best_time_any
        if best_choice is None and override_fallback:
            best_choice = (override_fallback, {"stints": []})

        if best_choice is None:
            return _cache_and_return(False, None)

        gain = None
        if best_time is not None and time_keep is not None:
            gain = time_keep - best_time
        diff = gain if gain is not None else -1.0

        # Prevent repeated same-tyre wet/inter stops on still-fresh tyres.
        # The mixed-weather planner can otherwise repeatedly choose a fresh
        # copy of the current rain tyre over a very short horizon.
        effective_choice = override_fallback if force_weather and override_fallback else best_choice[0]
        if best_choice and self._same_compound_repit_guard_active(
            rm,
            current_comp,
            effective_choice,
            laps_in_stint,
            current_wear,
            wear_limit,
            gain,
            pit_loss,
            force_due_wear,
        ):
            return _cache_and_return(False, None)

        used_mandatory = self._unique_mandatory_count(used_all, rm)
        final_stint = len(stints) <= 1
        if (self._physics_scorer(rm) is not None
                and remaining_laps <= self._safety_car_late_guard_laps(rm)
                and (not getattr(rm, "formula_refueling_allowed", False)
                     or rm.fuel_onboard.get(driver.name, 0.) >= rm.fuel_burn_per_lap.get(driver.name, 0.) * remaining_laps)):
            # compare a late extra stop with truly staying out, not with a
            # "keep" plan that has already inserted that same stop at lap 0
            final_stint = True
            time_keep, _ = self._simulate_stint(rm, driver, current_comp, current_wear,
                                               remaining_laps, laps_done, new_set=False)
            diff = time_keep - best_time if best_time is not None else -1.
        threshold = 0.25
        if final_stint and not force_weather:
            threshold = max(0.5, pit_loss * 0.45)
            if used_mandatory >= mandatory_count and diff <= threshold:
                if force_due_wear:
                    force_pit = False
                    force_due_wear = False
                if force_due_plan:
                    force_pit = False
                    force_due_plan = False
                force_pit = False
        if sc_opportunity and not force_pit:
            threshold = max(
                threshold,
                self._safety_car_opportunity_gain_threshold(
                    rm,
                    driver,
                    variation,
                    base_pit_loss,
                    current_wear,
                    remaining_laps,
                ),
            )
        if remaining_laps <= 2 and not force_pit and used_mandatory >= mandatory_count:
            return _cache_and_return(False, None)

        if (
            not force_pit
            and best_time is not None
            and time_keep is not None
            and diff > threshold
        ):
            stint_fraction = laps_in_stint / effective_planned if effective_planned else 1.0
            min_progress = max(2, int(round(effective_planned * 0.45)))
            if (
                not sc_opportunity
                and not weather_alert
                and
                remaining_laps > max(5, len(stints))
                and laps_in_stint < min_progress
                and wear_ratio < 0.6
                and stint_fraction < 0.5
            ):
                return _cache_and_return(False, None)

        if force_pit or (best_time is not None and time_keep is not None and diff > threshold):
            chosen = override_fallback if force_weather and override_fallback else best_choice[0]
            self._record_weather_diagnostic(
                driver.name,
                current_cat=current_cat,
                preferred_category=preferred_category,
                action="pit",
                compound=chosen,
                spell_eval=spell_eval,
                wrong_state=wrong_state,
                emergency=bool(emergency_weather),
            )
            return _cache_and_return(True, chosen)
        if force_pit and best_choice:
            return _cache_and_return(True, best_choice[0])
        return _cache_and_return(False, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pit_logic_enabled(self, rm=None) -> bool:
        pcfg = self.cfg.get("pitstops", {})
        if not bool(pcfg.get("enabled", False)):
            return False
        tyre_model = getattr(rm, "tyre_model", None) if rm is not None else None
        if tyre_model is not None:
            try:
                return bool(tyre_model.enabled and tyre_model.compound_names())
            except Exception:
                return bool(tyre_model.compound_names())
        return True

    def _default_compound(self, rm=None) -> str:
        tyre_model = getattr(rm, "tyre_model", None) if rm is not None else None
        if tyre_model is not None:
            try:
                return str(getattr(tyre_model, "default_compound", "medium"))
            except Exception:
                return "medium"
        return self.cfg.get("tyres", {}).get("default_compound", "medium")

    def _mandatory_compounds(self, rm=None) -> int:
        getter = getattr(rm, "formula_compound_requirement", None)
        if callable(getter):
            return getter()
        tyre_model = getattr(rm, "tyre_model", None) if rm is not None else None
        if tyre_model is not None:
            try:
                count = int(getattr(tyre_model, "mandatory_compound_count", 2))
            except Exception:
                count = 2
            return max(1, count)
        tcfg = self.cfg.get("tyres", {})
        try:
            count = int(tcfg.get("mandatory_compound_count", 2))
        except Exception:
            count = 2
        return max(1, count)

    def _tyre_compounds(self, rm=None) -> List[str]:
        tyre_model = getattr(rm, "tyre_model", None) if rm is not None else None
        if tyre_model is not None:
            compounds = list(tyre_model.compound_names())
            if compounds:
                return compounds
        tcfg = self.cfg.get("tyres", {})
        compounds = list(tcfg.get("compounds", {}).keys())
        if not compounds:
            compounds = [self._default_compound(rm)]
        return compounds

    def _default_stop_cap(self, rm) -> int:
        try:
            wear_mult = float(getattr(rm, "track_wear_mult", 1.0))
        except Exception:
            wear_mult = 1.0
        if wear_mult > 1.2:
            return self.MAX_PLANNED_STOPS
        return min(self.MAX_PLANNED_STOPS, 2)

    def _counts_toward_mandatory(self, comp: Optional[str], rm=None) -> bool:
        if not comp:
            return False
        tyre_model = getattr(rm, "tyre_model", None) if rm is not None else None
        if tyre_model is not None:
            return self._compound_wet_category(comp, rm) == "dry"
        tcfg = self.cfg.get("tyres", {})
        comp_def = tcfg.get("compounds", {}).get(comp, {})
        wet_pref = comp_def.get("optimal_wetness")
        if isinstance(wet_pref, (list, tuple)) and len(wet_pref) >= 2:
            try:
                return float(wet_pref[1]) <= 1.0
            except Exception:
                pass
        key = str(comp).lower()
        return key not in {"intermediate", "wet"}

    def _unique_mandatory_count(self, compounds: Iterable[str], rm=None) -> int:
        unique = set()
        for comp in compounds:
            if self._counts_toward_mandatory(comp, rm):
                unique.add(str(comp).lower())
        return len(unique)

    def _compound_wet_category(self, comp: str, rm=None) -> str:
        memo = getattr(self, "_projection_memo", None)
        cache = memo.get("compound_cat") if memo is not None else None
        if cache is not None:
            key = (id(rm), str(comp))
            hit = cache.get(key)
            if hit is not None:
                return hit
        tyre_model = getattr(rm, "tyre_model", None) if rm is not None else None
        if tyre_model is not None:
            result = tyre_model.compound_category(comp)
        else:
            name = str(comp).lower().strip()
            if name in {"wet", "wets", "rain"}:
                result = "wet"
            elif name in {"intermediate", "inter", "inters", "int"}:
                result = "inter"
            else:
                tcfg = self.cfg.get("tyres", {})
                comp_def = tcfg.get("compounds", {}).get(comp, {})
                wet_pref = comp_def.get("optimal_wetness")
                lo = 0.0
                hi = 0.0
                if isinstance(wet_pref, (list, tuple)) and len(wet_pref) >= 2:
                    try:
                        lo = float(wet_pref[0])
                        hi = float(wet_pref[1])
                    except Exception:
                        lo, hi = 0.0, 0.0
                if hi <= 1.0:
                    result = "dry"
                elif hi <= 4.2:
                    result = "inter"
                else:
                    result = "wet"
        if cache is not None:
            if len(cache) >= 4096:
                cache.clear()
            cache[key] = result
        return result

    def _wetness_category(self, wetness_value, rm=None) -> str:
        tyre_model = getattr(rm, "tyre_model", None) if rm is not None else None
        if tyre_model is not None:
            try:
                return str(tyre_model.best_category_for_wetness(wetness_value) or "dry").lower()
            except Exception:
                pass
        return self._band_category(self._wet_band_for_value(wetness_value))

    def _projected_wetness(self, rm, start_lap: int, laps_ahead: Optional[int]) -> List[float]:
        if rm is None:
            return []
        profile = list(getattr(rm, "wetness_profile", []))
        if not profile:
            return []
        # Weather is a race-wide track state. A lapped driver's personal lap
        # must not move their forecast backwards along the weather timeline.
        start = self._weather_profile_index(rm)
        if laps_ahead is None:
            end = len(profile)
        else:
            try:
                span = max(1, int(laps_ahead))
            except Exception:
                span = max(1, len(profile) - start)
            end = start + span
        end = max(start + 1, min(len(profile), end))
        return profile[start:end]

    def _weather_profile_index(self, rm) -> int:
        if rm is None:
            return 0
        func = getattr(rm, "weather_lap_index", None)
        if callable(func):
            try:
                return max(0, int(func()))
            except Exception:
                pass
        laps = getattr(rm, "laps", {}) or {}
        try:
            return max(0, int(max(laps.values()))) if laps else 0
        except Exception:
            return 0

    def _weather_projection_index(self, rm, driver, projected_driver_lap: int) -> int:
        global_now = self._weather_profile_index(rm)
        try:
            driver_now = int((getattr(rm, "laps", {}) or {}).get(driver.name, 0) or 0)
        except Exception:
            driver_now = 0
        try:
            offset = max(0, int(projected_driver_lap) - driver_now)
        except Exception:
            offset = 0
        return max(0, global_now + offset)

    def _build_weather_suitability_cache(self, rm) -> None:
        """Precompute cheap compound/category suitability for the race profile."""
        self._weather_suitability_cache = []
        self._weather_cache_race_id = id(rm) if rm is not None else None
        if rm is None:
            return
        tyre_model = getattr(rm, "tyre_model", None)
        profile = list(getattr(rm, "wetness_profile", []) or [])
        compounds = self._tyre_compounds(rm)
        if tyre_model is None or not profile or not compounds:
            return
        for wetness in profile:
            scores: Dict[str, float] = {}
            best_by_category: Dict[str, Tuple[str, float]] = {}
            best_comp = None
            best_score = None
            for comp in compounds:
                try:
                    lat, longi = tyre_model.grip_multipliers(
                        comp,
                        wetness_mm=wetness,
                        wear=0.0,
                    )
                    score = (0.6 * float(lat)) + (0.4 * float(longi))
                except Exception:
                    score = 1.0
                name = str(comp)
                scores[name] = float(score)
                category = self._compound_wet_category(name, rm)
                previous = best_by_category.get(category)
                if previous is None or score > previous[1]:
                    best_by_category[category] = (name, float(score))
                if best_score is None or score > best_score:
                    best_comp, best_score = name, float(score)
            category = self._compound_wet_category(best_comp, rm) if best_comp else "dry"
            self._weather_suitability_cache.append(
                {
                    "wetness": float(wetness),
                    "category": str(category),
                    "scores": scores,
                    "best_by_category": best_by_category,
                    "best_compound": best_comp,
                    "best_score": float(best_score if best_score is not None else 1.0),
                }
            )

    def _ensure_weather_suitability_cache(self, rm) -> List[dict]:
        if self._weather_cache_race_id != id(rm) or not self._weather_suitability_cache:
            self._build_weather_suitability_cache(rm)
        return self._weather_suitability_cache

    def _weather_entry(self, rm, index: Optional[int] = None) -> Optional[dict]:
        cache = self._ensure_weather_suitability_cache(rm)
        if not cache:
            return None
        idx = self._weather_profile_index(rm) if index is None else int(index)
        idx = max(0, min(len(cache) - 1, idx))
        return cache[idx]

    def _preferred_weather_category(
        self,
        rm,
        current_category: str,
        lookahead: Optional[Sequence[float]] = None,
    ) -> Optional[str]:
        """Choose a direction-aware category with simple crossover hysteresis."""
        current_key = str(current_category or "dry").lower()
        cache = self._ensure_weather_suitability_cache(rm)
        start = self._weather_profile_index(rm)
        categories = [str(entry.get("category", "dry")) for entry in cache[start : start + 4]]
        if not categories and lookahead:
            categories = [self._wetness_category(value, rm) for value in lookahead]
        if not categories:
            return None

        now = categories[0]
        current_rank = self._wet_category_rank(current_key)
        now_rank = self._wet_category_rank(now)
        if now_rank > current_rank:
            return now
        if now_rank < current_rank:
            # Drying crossovers need one confirming forecast lap. This avoids
            # wet->inter->dry double stops through a one-lap transition band.
            if len(categories) >= 2 and categories[1] == now:
                return now
            return None

        # Anticipate a wetter crossover one lap early only when it persists.
        if len(categories) >= 3:
            next_cat = categories[1]
            if (
                self._wet_category_rank(next_cat) > current_rank
                and self._wet_category_rank(categories[2]) >= self._wet_category_rank(next_cat)
            ):
                return next_cat
        return None

    def _evaluate_cached_weather_spell(
        self,
        rm,
        driver,
        current_comp: str,
        laps_done: int,
        remaining_laps: int,
        pit_loss: float,
        preferred_category: str,
        comp_options: Sequence[str],
        variation: Dict[str, float],
    ) -> Optional[dict]:
        """Estimate crossover payback over the actual bounded weather spell."""
        cache = self._ensure_weather_suitability_cache(rm)
        if not cache or remaining_laps <= 0:
            return None
        candidates = self._weather_transition_compounds(
            rm,
            current_comp,
            preferred_category,
            comp_options,
        )
        if not candidates:
            return None
        start = self._weather_profile_index(rm)
        horizon = min(
            max(1, int(remaining_laps)),
            self.WEATHER_PAYBACK_HORIZON_LAPS,
            max(1, len(cache) - start),
        )
        margin = max(0.0, float(variation.get("weather_margin_s", 0.0) or 0.0))
        best = None
        for comp in candidates:
            cumulative = 0.0
            best_cumulative = float("-inf")
            best_laps = 0
            for offset in range(horizon):
                entry = cache[start + offset]
                scores = entry.get("scores", {}) or {}
                current_score = float(scores.get(str(current_comp), 1.0) or 1.0)
                target_score = float(scores.get(str(comp), 1.0) or 1.0)
                if self._physics_scorer(rm) is not None:
                    stay, _ = self._simulate_stint(rm, driver, current_comp,
                        rm.tyre_wear.get(driver.name, 0.), offset+1, laps_done, new_set=False)
                    change, _ = self._simulate_stint(rm, driver, comp,
                        self._next_physical_set_wear(rm, driver.name, comp), offset+1, laps_done, new_set=True)
                    cumulative = stay - change
                else:
                    cumulative += (target_score - current_score) * self.GRIP_TO_LAP_SECONDS
                if cumulative > best_cumulative:
                    best_cumulative = cumulative
                    best_laps = offset + 1
            net_gain = float(best_cumulative) - float(pit_loss)
            candidate = {
                "action": "pit" if net_gain > margin else "stay",
                "compound": str(comp),
                "gain": float(net_gain),
                "gross_gain": float(best_cumulative),
                "horizon": int(best_laps),
                "margin": float(margin),
            }
            if best is None or candidate["gain"] > best["gain"]:
                best = candidate
        return best

    def _update_wrong_weather_state(self, rm, driver, current_comp: str, laps_done: int) -> dict:
        name = str(getattr(driver, "name", "") or "")
        state = self._weather_state.setdefault(
            name,
            {"laps": 0, "loss_s": 0.0, "category": None, "last_driver_lap": None},
        )
        entry = self._weather_entry(rm)
        if not entry:
            return state
        if (getattr(self, "_projection_memo", None) or {}).get("unavailable_weather_category") == name:
            # Never accumulate a forced-weather penalty against a tyre that
            # cannot be fitted. Available alternatives are priced normally.
            state.update({"laps": 0, "loss_s": 0.0, "category": None, "last_driver_lap": int(laps_done)})
            return state
        current_category = self._compound_wet_category(current_comp, rm)
        best_category = str(entry.get("category", "dry"))
        if current_category == best_category:
            state.update({"laps": 0, "loss_s": 0.0, "category": None, "last_driver_lap": int(laps_done)})
            return state
        if state.get("last_driver_lap") == int(laps_done):
            return state
        scores = entry.get("scores", {}) or {}
        current_score = float(scores.get(str(current_comp), 1.0) or 1.0)
        best_score = float(entry.get("best_score", current_score) or current_score)
        if self._physics_scorer(rm) is not None:
            current_time, _ = self._simulate_stint(rm, driver, current_comp,
                rm.tyre_wear.get(name, 0.), 1, laps_done, new_set=False)
            alternatives = [self._simulate_stint(rm, driver, comp,
                self._next_physical_set_wear(rm, name, comp), 1, laps_done, new_set=True)[0]
                for comp in self._candidate_compounds(rm, laps_done, 1)
                if self._compound_wet_category(comp, rm) == best_category]
            lap_loss = max(0., current_time-min(alternatives)) if alternatives else 0.
        else:
            lap_loss = max(0.0, (best_score - current_score) * self.GRIP_TO_LAP_SECONDS)
        if state.get("category") != best_category:
            state.update({"laps": 0, "loss_s": 0.0, "category": best_category})
        state["laps"] = int(state.get("laps", 0) or 0) + 1
        state["loss_s"] = float(state.get("loss_s", 0.0) or 0.0) + lap_loss
        state["last_loss_per_lap_s"] = float(lap_loss)
        state["last_driver_lap"] = int(laps_done)
        return state

    # replacement for copy.deepcopy which is extremely slow
    # a strategy plan is only a dictionary with a list of stint dicts so _clone_plan copies the dicts
    # in a single pass without Python overhead
    @staticmethod
    def _clone_plan(plan: Optional[dict]) -> Optional[dict]:
        if plan is None:
            return None
        cloned = dict(plan)
        if "stints" in cloned and cloned["stints"]:
            cloned["stints"] = [dict(s) for s in cloned["stints"]]
        if "immediate_physics_plans" in cloned and cloned["immediate_physics_plans"]:
            cloned["immediate_physics_plans"] = {
                k: StrategyManager._clone_plan(v)
                for k, v in cloned["immediate_physics_plans"].items()
            }
        return cloned

    def _weather_incompetence_ceiling(
        self,
        rm,
        driver,
        current_comp: str,
        state: dict,
        comp_options: Sequence[str],
        variation: Dict[str, float],
    ) -> Optional[dict]:
        target_category = str(state.get("category") or "")
        if target_category not in {"dry", "inter", "wet"}:
            return None
        cache = self._ensure_weather_suitability_cache(rm)
        start = self._weather_profile_index(rm)
        if not cache or start >= len(cache):
            return None
        # Do not force a correction when the current crossover band disappears
        # on the following lap.
        confirmation = cache[start : min(len(cache), start + 2)]
        if len(confirmation) >= 2 and any(str(e.get("category")) != target_category for e in confirmation):
            return None
        patience = self.WEATHER_WRONG_TYRE_BASE_LAPS + int(
            variation.get("weather_patience_laps", 0) or 0
        )
        loss_limit = self.WEATHER_WRONG_TYRE_BASE_LOSS_S + float(
            variation.get("weather_loss_tolerance_s", 0.0) or 0.0
        )
        severe = float(state.get("last_loss_per_lap_s", 0.0) or 0.0) >= self.WEATHER_SEVERE_LOSS_PER_LAP_S
        over_limit = float(state.get("loss_s", 0.0) or 0.0) >= loss_limit
        over_laps = severe and int(state.get("laps", 0) or 0) >= patience
        if not (over_limit or over_laps):
            return None
        entry = cache[start]
        category_best = (entry.get("best_by_category", {}) or {}).get(target_category)
        target = category_best[0] if category_best else None
        if not target or target not in comp_options:
            target = next(
                (c for c in comp_options if self._compound_wet_category(c, rm) == target_category),
                None,
            )
        if not target or self._compound_wet_category(target, rm) == self._compound_wet_category(current_comp, rm):
            return None
        return {"action": "pit", "compound": str(target), "reason": "wrong_tyre_ceiling"}

    def _record_weather_diagnostic(self, driver_name: str, **payload) -> None:
        self.weather_diagnostics[str(driver_name)] = dict(payload)

    def is_weather_category_change(self, rm, driver, target_compound: Optional[str]) -> bool:
        if not target_compound:
            return False
        current = getattr(rm, "tyre_comp", {}).get(driver.name, self._default_compound(rm))
        return self._compound_wet_category(current, rm) != self._compound_wet_category(target_compound, rm)

    def _candidate_compounds(
        self, rm=None, start_lap: int = 0, laps_ahead: Optional[int] = None
    ) -> List[str]:
        compounds = self._tyre_compounds(rm)
        if rm is None:
            return compounds

        window = self._projected_wetness(rm, start_lap, laps_ahead)
        if not window:
            return compounds

        categories: Dict[str, List[str]] = {"dry": [], "inter": [], "wet": []}
        for comp in compounds:
            cat = self._compound_wet_category(comp, rm)
            categories.setdefault(cat, [])
            if comp not in categories[cat]:
                categories[cat].append(comp)

        present: List[str] = []
        for value in window:
            cat = self._wetness_category(value, rm)
            present.append(cat)

        present_set = {cat for cat in present if cat in categories}
        if not present_set:
            return compounds

        highest = max(self._wet_category_rank(cat) for cat in present_set)

        allowed: List[str] = []

        def add_cat(cat_name: str) -> None:
            for comp in categories.get(cat_name, []):
                if comp not in allowed:
                    allowed.append(comp)

        if highest == self._wet_category_rank("wet"):
            add_cat("wet")
            if "inter" in present_set:
                add_cat("inter")
            if "dry" in present_set:
                add_cat("dry")
        elif highest == self._wet_category_rank("inter"):
            add_cat("inter")
            if "dry" in present_set:
                add_cat("dry")
            if "wet" in present_set:
                add_cat("wet")
        else:
            add_cat("dry")
            if "inter" in present_set:
                add_cat("inter")

        if not allowed:
            return compounds
        return allowed

    def _plan_compound_choices(self, rm, driver_name, options, consume_start_compound=None):
        queues = self._physical_tyre_wear_queues(rm, driver_name, options,
            consume_start_compound=consume_start_compound)
        admitted = getattr(rm, "available_compounds", options)
        filtered = [c for c in options if c in admitted and (queues is None or queues.get(str(c), []))]
        if len(filtered) == len(options) and filtered:
            return filtered, queues
        present = {self._compound_wet_category(c, rm) for c in filtered}
        if filtered and all(self._compound_wet_category(c, rm) in present for c in options):
            return filtered, queues
        # Only the missing-category path asks for any extra stock lookups.
        expanded = [c for c in self._tyre_compounds(rm) if c in admitted]
        if expanded == options:
            return filtered, queues
        queues = self._physical_tyre_wear_queues(rm, driver_name, expanded,
            consume_start_compound=consume_start_compound)
        return [c for c in expanded if queues is None or queues.get(str(c), [])], queues

    def _physical_tyre_wear_queues(
        self,
        rm,
        driver_name: str,
        compounds: Sequence[str],
        *,
        consume_start_compound: Optional[str] = None,
    ) -> Optional[Dict[str, List[float]]]:
        """Return sorted wear for every legal remaining physical race set.

        ``None`` retains the established unlimited-allocation behaviour.  A
        supplied start compound removes its least-worn set for a hypothetical
        pit choice that has not yet been checked out by RaceManager.
        """

        allocation_manager = getattr(rm, "weekend_tyre_manager", None)
        weekend = getattr(rm, "weekend", None)
        if (
            allocation_manager is None
            or not allocation_manager.enabled()
            or weekend is None
        ):
            return None
        queues: Dict[str, List[float]] = {}
        for compound in compounds:
            try:
                rows = allocation_manager.available_sets_for_session(
                    weekend,
                    driver_name,
                    compound,
                    "race",
                )
            except Exception:
                rows = allocation_manager.sets_for_driver(
                    weekend,
                    driver_name,
                    compound,
                    available_only=True,
                )
            queues[str(compound)] = sorted(
                max(0.0, float(row.get("wear", 0.0) or 0.0))
                for row in rows
                if isinstance(row, dict)
            )
        if consume_start_compound is not None:
            queue = queues.get(str(consume_start_compound), [])
            if queue:
                queue.pop(0)
        return queues

    @staticmethod
    def _sequence_physical_start_wears(
        compounds: Sequence[str],
        current_wear: float,
        wear_queues: Optional[Dict[str, List[float]]],
    ) -> Optional[List[float]]:
        """Assign each future stint the next least-worn physical set."""

        if not compounds:
            return []
        if wear_queues is None:
            return [float(current_wear)] + [0.0] * (len(compounds) - 1)
        cursors: Dict[str, int] = {}
        result = [max(0.0, float(current_wear))]
        for compound in list(compounds)[1:]:
            key = str(compound)
            cursor = int(cursors.get(key, 0) or 0)
            queue = wear_queues.get(key, [])
            if cursor >= len(queue):
                return None
            result.append(max(0.0, float(queue[cursor])))
            cursors[key] = cursor + 1
        return result

    def _next_physical_set_wear(self, rm, driver_name: str, compound: str) -> float:
        queues = self._physical_tyre_wear_queues(
            rm,
            driver_name,
            [compound],
        )
        if queues is None:
            return 0.0
        values = queues.get(str(compound), [])
        return max(0.0, float(values[0])) if values else 0.0

    def _wet_category_rank(self, category: Optional[str]) -> int:
        key = str(category).lower() if category is not None else ""
        if key == "wet":
            return 2
        if key == "inter":
            return 1
        return 0

    def _band_category(self, band_name: Optional[str]) -> str:
        if band_name is None:
            return "dry"
        key = str(band_name).lower()
        pref = self._wet_band_pref.get(key)
        if pref:
            return pref
        if "storm" in key or "monsoon" in key or "wet" in key:
            return "wet"
        if "damp" in key or "moist" in key or "rain" in key:
            return "inter"
        return "dry"

    def _wet_band_for_value(self, value) -> str:
        if not self._wet_band_defs:
            return "dry"
        try:
            val = float(value)
        except Exception:
            return self._wet_band_defs[0]["key"]
        for band in self._wet_band_defs:
            lo = band.get("min", self._wet_band_defs[0]["min"])
            hi = band.get("max", self._wet_band_defs[-1]["max"])
            if val < lo:
                continue
            if val <= hi + 1e-6:
                return band.get("key", band.get("name", "dry"))
        return self._wet_band_defs[-1].get("key", self._wet_band_defs[-1].get("name", "dry"))

    def _build_wet_band_metadata(self) -> Tuple[List[dict], Dict[str, str]]:
        tcfg = self.cfg.get("tyres", {})
        wet_range = tcfg.get("wetness_range", [0.0, 5.0])
        try:
            range_lo = float(wet_range[0])
        except Exception:
            range_lo = 0.0
        try:
            range_hi = float(wet_range[1])
        except Exception:
            range_hi = 5.0
        effects = tcfg.get("wetness_effects", {})
        band_defs: List[dict] = []
        band_pref: Dict[str, str] = {}
        for name, data in effects.items():
            if not isinstance(data, dict):
                continue
            band_name = str(name)
            key = band_name.lower()
            lo_raw = data.get("min_mm", range_lo)
            hi_raw = data.get("max_mm", range_hi)
            try:
                lo_val = float(lo_raw)
            except Exception:
                lo_val = range_lo
            try:
                hi_val = float(hi_raw)
            except Exception:
                hi_val = range_hi
            band_defs.append({"name": band_name, "key": key, "min": lo_val, "max": hi_val})
            best_comp = None
            best_pace = None
            for comp_name, comp_stats in data.get("compounds", {}).items():
                if not isinstance(comp_stats, dict):
                    continue
                try:
                    pace = float(comp_stats.get("pace", 0.0))
                except Exception:
                    pace = 0.0
                if best_pace is None or pace < best_pace:
                    best_comp = comp_name
                    best_pace = pace
            if best_comp:
                band_pref[key] = self._compound_wet_category(best_comp, None)
        if not band_defs:
            span = max(1e-6, range_hi - range_lo)
            dry_hi = range_lo + span * 0.20
            inter_hi = range_lo + span * 0.58
            band_defs.extend(
                [
                    {"name": "dry", "key": "dry", "min": range_lo, "max": dry_hi},
                    {"name": "damp", "key": "damp", "min": dry_hi, "max": inter_hi},
                    {"name": "wet", "key": "wet", "min": inter_hi, "max": range_hi},
                ]
            )
            band_pref.setdefault("dry", "dry")
            band_pref.setdefault("damp", "inter")
            band_pref.setdefault("wet", "wet")
        band_defs.sort(key=lambda entry: (entry.get("min", range_lo), entry.get("max", range_hi)))
        return band_defs, band_pref

    def _weather_gain_threshold(self, remaining_laps: int, pit_loss: float) -> float:
        base = max(3.0, float(pit_loss) * 0.18)
        if remaining_laps <= 8:
            base *= 1.05
        if remaining_laps <= 5:
            base *= 1.1
        if remaining_laps <= 3:
            base *= 1.25
        return base

    def _same_compound_repit_guard_active(
        self,
        rm,
        current_comp: str,
        target_comp: Optional[str],
        laps_in_stint: int,
        current_wear: float,
        wear_limit: float,
        projected_gain: Optional[float],
        pit_loss: float,
        force_due_wear: bool,
    ) -> bool:
        """Block low-value rapid re-pits onto the same tyre.

        This is intentionally narrow: it only suppresses very short, very
        low-wear same-compound refresh stops unless the projected gain is
        overwhelming. It exists to stop crossover loops like wet->hard then
        hard->hard again on still-fresh tyres a lap or two later.
        """

        if not target_comp or str(target_comp) != str(current_comp):
            return False
        if force_due_wear:
            return False

        current_cat = self._compound_wet_category(current_comp, rm)
        if current_cat not in {"dry", "inter", "wet"}:
            return False

        if current_cat == "wet":
            stint_lap_guard = 10
            low_wear_guard = min(0.45, max(0.28, float(wear_limit) * 0.80))
            overwhelming_gain = max(float(pit_loss) * 1.10, 18.0)
        elif current_cat == "inter":
            stint_lap_guard = 8
            low_wear_guard = min(0.45, max(0.28, float(wear_limit) * 0.80))
            overwhelming_gain = max(float(pit_loss) * 1.10, 18.0)
        else:
            # Dry same-tyre repits should be even rarer on still-fresh tyres,
            # but keep the guard shorter/tighter so normal dry strategy
            # remains unaffected.
            stint_lap_guard = 6
            low_wear_guard = min(0.22, max(0.12, float(wear_limit) * 0.40))
            overwhelming_gain = max(float(pit_loss) * 1.20, 20.0)

        if int(laps_in_stint) >= stint_lap_guard:
            return False

        if float(current_wear) > low_wear_guard:
            return False

        try:
            if projected_gain is not None and float(projected_gain) >= overwhelming_gain:
                return False
        except Exception:
            pass
        return True

    def _realistic_weather_eval_available(self, rm) -> bool:
        try:
            model = getattr(rm, "realistic_physics", None)
            return bool(
                getattr(rm, "use_realistic_physics", False)
                and model is not None
                and getattr(model, "enabled", False)
                and hasattr(rm, "_realistic_lap_inputs_for_driver")
            )
        except Exception:
            return False

    def _weather_transition_categories(
        self,
        current_category: str,
        preferred_category: Optional[str],
    ) -> set[str]:
        cats = set()
        if preferred_category:
            cats.add(str(preferred_category).lower())
        current_key = str(current_category or "").lower()
        preferred_key = str(preferred_category or "").lower()
        if current_key == "dry" and preferred_key == "wet":
            cats.add("inter")
        elif current_key == "wet" and preferred_key == "dry":
            cats.add("inter")
        return {c for c in cats if c in {"dry", "inter", "wet"} and c != current_key}

    def _weather_transition_compounds(
        self,
        rm,
        current_comp: str,
        preferred_category: Optional[str],
        comp_options: Sequence[str],
    ) -> List[str]:
        current_cat = self._compound_wet_category(current_comp, rm)
        target_cats = self._weather_transition_categories(current_cat, preferred_category)
        if not target_cats:
            return []
        preferred_key = str(preferred_category or "").lower()
        out = []
        for comp in comp_options:
            cat = self._compound_wet_category(comp, rm)
            if cat == current_cat or cat not in target_cats:
                continue
            out.append(str(comp))
        out.sort(key=lambda comp: (self._compound_wet_category(comp, rm) != preferred_key, str(comp)))
        return out

    def _projected_realistic_lap_time(self, rm, driver, comp_name, wear, lap_idx, fuel_kg, tyre_temp_c):
        physics = self._physics_scorer(rm)
        if physics is None:
            return None
        if tyre_temp_c is None:
            tyre_temp_c = rm.tyre_model.initial_temperature_c(comp_name, pit_out=lap_idx > 0)
        inputs = physics.inputs(driver, comp_name, wear, fuel_kg, tyre_temp_c, lap_idx)
        return physics.evaluate([inputs])[0]

    @_projection_call
    def _simulate_weather_horizon(
        self,
        rm,
        driver,
        comp_name: str,
        start_wear: float,
        laps: int,
        start_lap: int,
        *,
        fuel_kg: Optional[float] = None,
    ) -> Tuple[float, float]:
        physics = self._physics_scorer(rm)
        if physics is not None:
            return self._simulate_stint(rm, driver, comp_name, start_wear, laps, start_lap, fuel_kg=fuel_kg)
        if _STRATEGY_KERNELS is not None:
            try:
                return _STRATEGY_KERNELS.simulate_weather_horizon(
                    self,
                    rm,
                    driver,
                    comp_name,
                    float(start_wear),
                    int(laps),
                    int(start_lap),
                    fuel_kg,
                )
            except Exception:
                pass
        return self._simulate_weather_horizon_python(
            rm,
            driver,
            comp_name,
            start_wear,
            laps,
            start_lap,
            fuel_kg=fuel_kg,
        )

    def _simulate_weather_horizon_python(
        self,
        rm,
        driver,
        comp_name: str,
        start_wear: float,
        laps: int,
        start_lap: int,
        *,
        fuel_kg: Optional[float] = None,
    ) -> Tuple[float, float]:
        if self._physics_scorer(rm) is not None:
            return self._simulate_stint(rm, driver, comp_name, start_wear, laps, start_lap, fuel_kg=fuel_kg)
        if not self._realistic_weather_eval_available(rm):
            return self._simulate_stint(rm, driver, comp_name, start_wear, laps, start_lap)
        laps_int = max(0, int(laps))
        if laps_int <= 0:
            return 0.0, float(start_wear)
        wear = float(start_wear)
        total = 0.0
        lap_origin = int(start_lap)
        tyre_temp = self._starting_tyre_temperature(rm, driver, comp_name, wear, lap_origin)
        try:
            fuel = float(fuel_kg if fuel_kg is not None else getattr(rm, "fuel_onboard", {}).get(driver.name, 0.0))
        except Exception:
            fuel = 0.0
        try:
            burn_per_lap = float(getattr(rm, "fuel_burn_per_lap", {}).get(driver.name, 0.0) or 0.0)
        except Exception:
            burn_per_lap = 0.0
        for offset in range(laps_int):
            lap_idx = lap_origin + offset
            _grip_score, deg, tyre_temp = self._projected_tyre_state(
                rm,
                driver,
                comp_name,
                wear,
                lap_idx,
                tyre_temp,
            )
            lap_time = self._projected_realistic_lap_time(
                rm,
                driver,
                comp_name,
                wear,
                lap_idx,
                fuel,
                tyre_temp,
            )
            if lap_time is None:
                return self._simulate_stint(rm, driver, comp_name, start_wear, laps, start_lap)
            total += float(lap_time)
            wear += max(0.0, float(deg))
            fuel = max(0.0, float(fuel) - float(burn_per_lap))
        return float(total), float(wear)

    @_projection_call
    def _evaluate_weather_transition(
        self,
        rm,
        driver,
        current_comp: str,
        current_wear: float,
        laps_done: int,
        remaining_laps: int,
        pit_loss: float,
        wear_limit: float,
        preferred_category: Optional[str],
        comp_options: Sequence[str],
    ) -> Optional[dict]:
        if _STRATEGY_KERNELS is not None:
            try:
                return _STRATEGY_KERNELS.evaluate_weather_transition(
                    self,
                    rm,
                    driver,
                    current_comp,
                    float(current_wear),
                    int(laps_done),
                    int(remaining_laps),
                    float(pit_loss),
                    float(wear_limit),
                    preferred_category,
                    list(comp_options),
                )
            except Exception:
                pass
        return self._evaluate_weather_transition_python(
            rm,
            driver,
            current_comp,
            current_wear,
            laps_done,
            remaining_laps,
            pit_loss,
            wear_limit,
            preferred_category,
            comp_options,
        )

    def _evaluate_weather_transition_python(
        self,
        rm,
        driver,
        current_comp: str,
        current_wear: float,
        laps_done: int,
        remaining_laps: int,
        pit_loss: float,
        wear_limit: float,
        preferred_category: Optional[str],
        comp_options: Sequence[str],
    ) -> Optional[dict]:
        candidates = self._weather_transition_compounds(
            rm,
            current_comp,
            preferred_category,
            comp_options,
        )
        if not candidates:
            return None
        horizon = max(1, min(int(remaining_laps), 4))
        try:
            fuel_now = float(getattr(rm, "fuel_onboard", {}).get(driver.name, 0.0) or 0.0)
        except Exception:
            fuel_now = 0.0
        stay_total, hold_wear = self._simulate_weather_horizon(
            rm,
            driver,
            current_comp,
            current_wear,
            horizon,
            laps_done,
            fuel_kg=fuel_now,
        )
        best_now_total = None
        best_now_comp = None
        best_hold_total = None
        best_hold_comp = None
        for comp in candidates:
            now_total, _ = self._simulate_weather_horizon(
                rm,
                driver,
                comp,
                0.0,
                horizon,
                laps_done,
                fuel_kg=fuel_now,
            )
            now_total += float(pit_loss)
            if best_now_total is None or now_total < best_now_total:
                best_now_total = now_total
                best_now_comp = comp
            if horizon > 1:
                hold_first_total, end_wear = self._simulate_weather_horizon(
                    rm,
                    driver,
                    current_comp,
                    current_wear,
                    1,
                    laps_done,
                    fuel_kg=fuel_now,
                )
                max_hold_wear = min(1.02, max(0.92, float(wear_limit) * 1.08))
                if end_wear <= max_hold_wear:
                    hold_total, _ = self._simulate_weather_horizon(
                        rm,
                        driver,
                        comp,
                        0.0,
                        horizon - 1,
                        laps_done + 1,
                        fuel_kg=max(0.0, fuel_now - float(getattr(rm, "fuel_burn_per_lap", {}).get(driver.name, 0.0) or 0.0)),
                    )
                    hold_total = float(hold_first_total) + float(pit_loss) + float(hold_total)
                    if best_hold_total is None or hold_total < best_hold_total:
                        best_hold_total = hold_total
                        best_hold_comp = comp

        threshold = self._weather_gain_threshold(remaining_laps, pit_loss)
        result = {
            "action": "stay",
            "compound": None,
            "gain": 0.0,
            "stay_total": float(stay_total),
            "pit_total": float(best_now_total) if best_now_total is not None else None,
            "hold_total": float(best_hold_total) if best_hold_total is not None else None,
        }
        if best_now_total is not None:
            gain_now = float(stay_total) - float(best_now_total)
            if gain_now > threshold:
                result.update({"action": "pit", "compound": best_now_comp, "gain": float(gain_now)})
        if best_hold_total is not None:
            gain_hold = float(stay_total) - float(best_hold_total)
            hold_margin = max(0.35, threshold * 0.10)
            if gain_hold > threshold:
                if result["action"] != "pit" or float(best_hold_total) + hold_margin < float(best_now_total):
                    result.update({"action": "hold", "compound": best_hold_comp, "gain": float(gain_hold)})
        return result

    def _estimate_weather_gain(
        self,
        rm,
        driver,
        current_comp: str,
        current_wear: float,
        start_lap: int,
        remaining_laps: int,
        pit_loss: float,
        comp_options: Sequence[str],
    ) -> Tuple[Optional[float], Optional[str]]:
        if remaining_laps <= 0:
            return None, None
        window = min(remaining_laps, 6)
        if window <= 0:
            return None, None
        if window < 3 and remaining_laps >= 3:
            window = 3
        window = max(1, int(window))
        stay_time, _ = self._simulate_stint(
            rm, driver, current_comp, current_wear, window, start_lap
        )
        best_total = None
        best_comp = None
        for comp in comp_options:
            if comp == current_comp:
                continue
            stint_time, _ = self._simulate_stint(rm, driver, comp, 0.0, window, start_lap)
            total = float(pit_loss) + stint_time
            if best_total is None or total < best_total:
                best_total = total
                best_comp = comp
        if best_total is None:
            return None, None
        gain = stay_time - best_total
        return gain, best_comp

    def _variation_for_driver(self, driver) -> Dict[str, float]:
        smoothness = getattr(driver, "smoothness", 10)
        smooth_bias = max(-0.05, min(0.05, (10 - smoothness) / 40.0))
        return {
            "wear_factor": random.uniform(0.95, 1.05),
            "aggression": random.uniform(-0.08, 0.08) + smooth_bias,
            "stop_bias": random.uniform(-1.5, 1.5),
            "pit_offset": random.uniform(-0.5, 0.5),
            "stint_seed": random.randrange(1_000_000),
            # Preserve small, bounded differences in crossover confidence.
            # No driver is allowed an open-ended wrong-tyre delay.
            "weather_margin_s": random.uniform(0.0, 2.0),
            "weather_patience_laps": random.choice((0, 0, 1)),
            "weather_loss_tolerance_s": random.uniform(-1.0, 2.0),
        }

    def _variation_factor(
        self, variation: Dict[str, float], kind: str, index: int, low: float, high: float
    ) -> float:
        seed = int(variation.get("stint_seed", 0))
        rng = random.Random(seed + (index + 1) * (137 if kind == "wear" else 193))
        return rng.uniform(low, high)

    def _safety_car_pit_active(self, rm) -> bool:
        if rm is None:
            return False
        return bool(
            getattr(rm, "sc_active", False)
            or getattr(rm, "vsc_active", False)
            or getattr(rm, "virtual_safety_car_active", False)
        )

    def _safety_car_weather_signature(self, rm) -> str:
        """Return the current physical track-weather category for SC overrides."""

        entry = self._weather_entry(rm)
        if isinstance(entry, dict) and entry.get("category"):
            return str(entry.get("category") or "dry").lower()
        current_wetness = getattr(rm, "current_wetness", None)
        value = current_wetness() if callable(current_wetness) else 0.0
        return str(self._wetness_category(value, rm) or "dry").lower()

    def _physics_command_signature(self, rm, driver):
        name = driver.name
        modes = tuple(getattr(getattr(rm, attr, None), "active_mode", lambda _n: None)(name)
                      for attr in ("driver_pace_modes", "driver_engine_modes", "driver_ers_modes"))
        state = rm.tyre_model.state_for_conditions(rm.tyre_comp.get(name), rm.lap_wetness(rm.weather_lap_index()))
        weather_band = (state.get("lat_mult"), state.get("long_mult"), state.get("wear_mult"))
        return (modes, tuple(sorted(rm.tyre_pressure_effects(name).items())), weather_band,
                rm.front_wing_damage_level(name), rm.pit_count.get(name, 0),
                bool(getattr(rm, "sc_active", False)), bool(getattr(rm, "vsc_active", False)))

    def _schedule_physics_recheck(self, rm, driver, plan):
        """Follow an already evaluated plan until its window or new information.

        This schedules decisions; it does not keep physics-result caches across
        laps. An unexpected wear/fuel change or command change reopens it.
        """
        lap = rm.laps.get(driver.name, 0)
        stints = plan.get("stints", [])
        target = stints[0].get("target_end_lap", lap+1) if stints else lap+1
        late = rm.total_laps-lap <= self._safety_car_late_guard_laps(rm)
        deadline = min(rm.total_laps, lap+3, max(lap+1, target-2)) if not late else min(rm.total_laps, lap+3)
        if deadline <= lap+1:
            plan.pop("physics_follow_plan", None)
            return
        physics = self._physics_scorer(rm)
        comp = rm.tyre_comp[driver.name]
        wear, fuel = rm.tyre_wear[driver.name], rm.fuel_onboard[driver.name]
        temp = rm.tyre_temp.get(driver.name, rm.tyre_model.initial_temperature_c(comp))
        expected = {}
        for future in range(lap+1, deadline):
            wear, fuel, temp = physics.advance(driver, comp, wear, fuel, temp, future-1)
            # _following_physics_plan reads only indices 0 and 1; the third
            # element is kept for diagnostics.
            expected[future] = (wear, fuel, temp)
        plan["physics_follow_plan"] = {"until": deadline, "expected": expected,
            "compound": comp, "commands": self._physics_command_signature(rm, driver)}

    def _following_physics_plan(self, rm, driver, plan):
        hold = plan.get("physics_follow_plan")
        if not hold or not physics_scoring_available(rm):
            return False
        lap, name = rm.laps.get(driver.name, 0), driver.name
        expected = hold["expected"].get(lap)
        if (lap >= hold["until"] or expected is None or rm.tyre_comp[name] != hold["compound"]
                or self._physics_command_signature(rm, driver) != hold["commands"]):
            return False
        return (abs(rm.tyre_wear[name]-expected[0]) <= .005
                and abs(rm.fuel_onboard[name]-expected[1]) <= .01)

    def _cliff_reassessment_due(self, rm, driver):
        name = driver.name
        compound = rm.tyre_comp.get(name)
        onset = getattr(getattr(rm, "tyre_model", None), "cliff_onset", lambda _c: None)(compound)
        if onset is None:
            return False
        wear = float(rm.tyre_wear.get(name, 0.0))
        return wear >= onset or wear + 2 * self._recommended_wear_delta(
            rm, driver, compound, wear, rm.laps.get(name, 0)
        ) >= onset

    def _safety_car_commitment_override_reason(
        self, rm, driver, decision: dict
    ) -> Optional[str]:
        """Identify changes serious enough to reopen a latched SC decision."""

        if self._safety_car_weather_signature(rm) != str(
            decision.get("weather_signature", "dry") or "dry"
        ).lower():
            return "weather"

        # A served commitment stays closed unless the weather changes.  This
        # prevents a second ordinary stop during the same SC period.
        if decision.get("consumed", False):
            return None

        name = str(getattr(driver, "name", "") or "")
        try:
            wear = max(
                0.0,
                float(
                    (getattr(rm, "tyre_wear", {}) or {}).get(name, 0.0)
                    or 0.0
                ),
            )
            wear_limit = max(0.01, float(decision.get("wear_limit", 0.6) or 0.6))
        except Exception:
            wear, wear_limit = 0.0, 0.6
        if wear >= wear_limit * 0.98:
            return "critical_wear"

        if (not decision.get("first_opportunity_passed") and not decision.get("cliff_checked")
                and self._cliff_reassessment_due(rm, driver)):
            decision["cliff_checked"] = True
            return "tyre_cliff"

        try:
            laps_done = int((getattr(rm, "laps", {}) or {}).get(name, 0) or 0)
            remaining_laps = max(0, int(getattr(rm, "total_laps", 0) or 0) - laps_done)
        except Exception:
            remaining_laps = 999
        current_comp = (getattr(rm, "tyre_comp", {}) or {}).get(
            name, self._default_compound(rm)
        )
        used = set(
            (getattr(rm, "used_compounds", {}) or {}).get(name, {current_comp})
            or {current_comp}
        )
        used.add(current_comp)
        mandatory_needed = max(
            0,
            self._mandatory_compounds(rm) - self._unique_mandatory_count(used, rm),
        )
        if mandatory_needed > 0 and remaining_laps <= max(3, mandatory_needed + 1):
            return "mandatory_stop"
        return None

    def _sc_plan_signature(self, rm, driver):
        """Inputs which can invalidate deployment pricing before its update.

        Ordinary position progress cannot alter the latched SC action; changed
        lap/forecast, service, car commands, wear or fuel must still replan.
        """
        name = driver.name
        try:
            return (rm.laps.get(name, 0), self._weather_profile_index(rm),
                    rm.tyre_comp.get(name), rm.tyre_wear.get(name), rm.tyre_temp.get(name),
                    rm.fuel_onboard.get(name), rm.last_lap_time.get(name),
                    tuple(sorted(rm.used_compounds.get(name, ()))),
                    self._physics_command_signature(rm, driver), rm.current_temperature(),
                    tuple(self._projected_wetness(rm, rm.laps.get(name, 0), 4)))
        except (AttributeError, TypeError, ValueError):
            return None  # Custom managers retain the existing replan fallback.

    def _sc_planned_stop_nearby(self, rm, driver):
        stints = self.plans.get(driver.name, {}).get("stints", [])
        if len(stints) < 2:
            return False
        remaining = rm.total_laps - rm.laps.get(driver.name, 0)
        until = stints[0].get("target_end_lap", rm.total_laps) - rm.laps.get(driver.name, 0)
        return remaining > self._safety_car_late_guard_laps(rm) and 0 <= until <= 6

    def _safety_car_late_guard_laps(self, rm) -> int:
        try:
            total = max(1, int(getattr(rm, "total_laps", 0) or 0))
        except Exception:
            total = 0
        if total <= 0:
            return 4
        return max(3, min(6, int(round(total * 0.10))))

    def _safety_car_pit_opportunity(
        self,
        rm,
        driver,
        current_wear: float,
        remaining_laps: int,
        mandatory_needed: int,
        preferred_category: Optional[str],
        wear_limit: float,
    ) -> bool:
        if not self._safety_car_pit_active(rm):
            return False
        try:
            wear = max(0.0, float(current_wear))
        except Exception:
            wear = 0.0
        if wear < self.SC_PIT_FRESH_WEAR_CUTOFF and (
                getattr(self, "_sc_deployment_pricing", False) or not self._sc_planned_stop_nearby(rm, driver)):
            return False
        try:
            remaining = int(remaining_laps)
        except Exception:
            remaining = 0
        try:
            critical_wear = wear >= float(wear_limit) * 0.98
        except Exception:
            critical_wear = False
        if remaining <= self._safety_car_late_guard_laps(rm):
            if not (mandatory_needed > 0 or preferred_category is not None or critical_wear):
                return False
        return True

    def _safety_car_pit_loss_multiplier(
        self,
        rm,
        driver,
        variation: Dict[str, float],
    ) -> float:
        sc_cfg = self.cfg.get("safety_car", {}) if isinstance(self.cfg, dict) else {}
        raw = None
        default_range = self.SC_PIT_LOSS_MULT_RANGE
        if isinstance(sc_cfg, dict):
            if bool(getattr(rm, "vsc_active", False)):
                default_range = (0.60, 0.75)
                raw = sc_cfg.get(
                    "vsc_ai_pit_loss_mult",
                    sc_cfg.get("vsc_ai_pit_loss_mult_range"),
                )
            else:
                raw = sc_cfg.get(
                    "ai_pit_loss_mult",
                    sc_cfg.get("ai_pit_loss_mult_range"),
                )
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                lo, hi = float(raw[0]), float(raw[1])
            except Exception:
                lo, hi = default_range
        elif raw is not None:
            try:
                value = float(raw)
            except Exception:
                value = sum(default_range) * 0.5
            return max(0.1, min(1.0, value))
        else:
            lo, hi = default_range
        if hi < lo:
            lo, hi = hi, lo
        lo = max(0.1, min(1.0, lo))
        hi = max(lo, min(1.0, hi))
        seed = int(variation.get("stint_seed", 0) or 0)
        name_sig = sum(ord(ch) for ch in str(getattr(driver, "name", "")))
        rng = random.Random(seed + name_sig + 27183)
        return rng.uniform(lo, hi)

    def _effective_immediate_pit_loss(
        self,
        rm,
        driver,
        variation: Dict[str, float],
        base_pit_loss: float,
        sc_opportunity: bool,
    ) -> float:
        try:
            base = float(base_pit_loss)
        except Exception:
            base = self._estimate_pit_loss(rm, driver, variation)
        if not sc_opportunity:
            return base
        multiplier = self._safety_car_pit_loss_multiplier(rm, driver, variation)
        return max(4.0, base * multiplier)

    def _safety_car_opportunity_gain_threshold(
        self,
        rm,
        driver,
        variation: Dict[str, float],
        base_pit_loss: float,
        current_wear: float,
        remaining_laps: int,
    ) -> float:
        try:
            base = float(base_pit_loss)
        except Exception:
            base = 18.5
        lo, hi = self.SC_PIT_GAIN_MARGIN_RANGE
        seed = int(variation.get("stint_seed", 0) or 0)
        name_sig = sum(ord(ch) for ch in str(getattr(driver, "name", "")))
        rng = random.Random(seed + name_sig + 31415)
        margin = base * rng.uniform(lo, hi)
        try:
            wear = float(current_wear)
        except Exception:
            wear = 0.0
        if wear < 0.30:
            margin += base * 0.10
        try:
            remaining = int(remaining_laps)
        except Exception:
            remaining = 0
        if remaining <= self._safety_car_late_guard_laps(rm) + 2:
            margin += base * 0.12
        return max(0.5, margin)

    def _estimate_pit_loss(self, rm, driver, variation: Dict[str, float]) -> float:
        pcfg = self.cfg.get("pitstops", {})
        loss = float(pcfg.get("pit_lane_loss_s", 18.5))
        team = getattr(driver, "team", None)
        skill = rm.mechanic_skill.get(team, 10) if team else 10
        if str(getattr(rm, "_physics_series_mode", "formula")) != "oval":
            skill = getattr(rm, "formula_pit_skill", {}).get(team, skill)
            return max(0.0, loss + tyre_service_mean_seconds(skill) + variation.get("pit_offset", 0.0))
        try:
            skill = max(1, min(20, int(skill)))
        except Exception:
            skill = 10
        delta = ((10 - skill) / 9.0) * 2.0
        delta = max(-2.0, min(2.0, delta))
        loss += delta
        loss += variation.get("pit_offset", 0.0)
        return max(12.0, loss)

    def _tyre_contract_grip_bonus_mult(self, rm, driver) -> float:
        contract_meta = {}
        if hasattr(rm, "tyre_contracts"):
            contract_meta = rm.tyre_contracts.get(driver.team, {}) or {}
        contract_type = str(
            getattr(rm, "tyre_contract_type_by_driver", {}).get(driver.name)
            or contract_meta.get("type")
            or ""
        ).lower()
        raw_bonus = contract_meta.get("bonus_grip_mult")
        if raw_bonus is None:
            if contract_type == "partner" and contract_meta.get("bonus_pace") is not None:
                raw_bonus = 1.0005
            elif contract_type == "works" and float(contract_meta.get("funding_weekly_m", 0.0) or 0.0) > 0.0:
                raw_bonus = 1.0005
        try:
            bonus = float(raw_bonus) if raw_bonus is not None else 1.0
        except Exception:
            bonus = 1.0
        if bonus <= 0.0:
            bonus = 1.0
        return float(bonus)

    def _supplier_ratings_for_compound(self, rm, driver, comp_name: str) -> Tuple[float, float]:
        tyre_model = getattr(rm, "tyre_model", None)
        supplier = rm.tyre_supplier_by_driver.get(driver.name)
        if tyre_model is None or supplier is None:
            return 50.0, 50.0
        try:
            comp_key = tyre_model.normalize_compound_name(comp_name)
        except Exception:
            comp_key = str(comp_name).lower().strip()
        try:
            pace_raw = float((getattr(supplier, "pace", {}) or {}).get(comp_key, 50.0))
        except Exception:
            pace_raw = 50.0
        try:
            dur_raw = float((getattr(supplier, "durability", {}) or {}).get(comp_key, 50.0))
        except Exception:
            dur_raw = 50.0
        return (
            float(tyre_model.supplier_pace_rating(pace_raw)),
            float(tyre_model.supplier_durability_rating(dur_raw)),
        )

    def _baseline_reference_lap(self, rm, driver) -> float:
        last_lap = getattr(rm, "last_lap_time", {}).get(driver.name)
        try:
            if last_lap is not None:
                return max(20.0, float(last_lap))
        except Exception:
            pass
        realistic = getattr(rm, "realistic_base_lap_by_driver", {}).get(driver.name)
        try:
            if realistic is not None:
                return max(20.0, float(realistic))
        except Exception:
            pass
        return 90.0

    def _starting_tyre_temperature(
        self,
        rm,
        driver,
        comp_name: str,
        start_wear: float,
        start_lap: int,
    ) -> Optional[float]:
        tyre_model = getattr(rm, "tyre_model", None)
        if tyre_model is None:
            return None
        if start_wear > 1e-6:
            try:
                live_temp = getattr(rm, "tyre_temp", {}).get(driver.name)
                if live_temp is not None:
                    return float(live_temp)
            except Exception:
                pass
        try:
            return float(
                tyre_model.initial_temperature_c(
                    comp_name,
                    pit_out=bool(start_lap > 0),
                )
            )
        except Exception:
            return None

    def _advance_tyre_temperature(
        self,
        rm,
        driver,
        comp_name: str,
        wetness_mm: Optional[float],
        current_temp_c: Optional[float],
    ) -> Optional[float]:
        tyre_model = getattr(rm, "tyre_model", None)
        if tyre_model is None:
            return None
        if current_temp_c is None:
            return None
        try:
            warmup_rate, cool_rate = tyre_model.thermal_rates(comp_name)
        except Exception:
            warmup_rate, cool_rate = 1.0, 1.0
        target_func = getattr(rm, "tyre_temperature_target", None)
        if callable(target_func):
            try:
                target_temp = float(target_func(driver.name, wetness_mm))
            except Exception:
                target_temp = float(current_temp_c)
        else:
            target_temp = float(current_temp_c)
        next_temp = float(current_temp_c)
        lap_step = float(self.TYRE_TEMP_STEP_C_PER_LAP)
        if next_temp < target_temp:
            next_temp = min(target_temp, next_temp + (lap_step * float(warmup_rate)))
        elif next_temp > target_temp:
            next_temp = max(target_temp, next_temp - (lap_step * float(cool_rate)))
        return max(20.0, min(180.0, float(next_temp)))

    @_projection_value
    def _projected_tyre_invariants(self, rm, driver, comp_name):
        try:
            if hasattr(rm, "driver_tyre_management_factor"):
                management = float(rm.driver_tyre_management_factor(driver))
            else:
                management = float(rm.team_meta.get(driver.team, {}).get("tyre_management", 1.0))
        except Exception:
            management = 1.0
        wear_scale = management * driver.tyre_wear_mult()
        pace, durability = self._supplier_ratings_for_compound(rm, driver, comp_name)
        bonus = self._tyre_contract_grip_bonus_mult(rm, driver)
        try:
            shift = float(getattr(rm, "driver_tyre_temp_window_shift_c")(driver))
        except Exception:
            shift = 0.0
        return wear_scale, pace, durability, bonus, shift, float(getattr(rm, "track_wear_mult", 1.0))

    def _projected_tyre_state(
        self,
        rm,
        driver,
        comp_name: str,
        wear: float,
        lap_idx: int,
        tyre_temp_c: Optional[float],
        _invariants=None,
    ) -> Tuple[float, float, Optional[float]]:
        tyre_model = getattr(rm, "tyre_model", None)
        if tyre_model is None:
            return 1.0, 0.0, tyre_temp_c
        physics = self._physics_scorer(rm)
        if physics is not None:
            temperature = tyre_temp_c if tyre_temp_c is not None else tyre_model.initial_temperature_c(comp_name)
            end_wear, _, end_temp = physics.advance(
                driver, comp_name, wear, physics.fuel_at(driver, lap_idx), temperature, lap_idx
            )
            inputs = physics.inputs(driver, comp_name, wear, physics.fuel_at(driver, lap_idx), temperature, lap_idx)
            return .6 * inputs.tyre_lat_mult + .4 * inputs.tyre_long_mult, max(0., end_wear-wear), end_temp
        if _invariants is None:
            _invariants = self._projected_tyre_invariants(rm, driver, comp_name)
        wear_scale, supplier_pace_rating, supplier_durability_rating, contract_grip_bonus_mult, temp_window_shift_c, track_wear = _invariants
        try:
            wet = rm.lap_wetness(self._weather_projection_index(rm, driver, lap_idx))
        except Exception:
            wet = None
        tyre_temp_next = self._advance_tyre_temperature(
            rm,
            driver,
            comp_name,
            wet,
            tyre_temp_c,
        )
        lat_mult, long_mult = tyre_model.grip_multipliers(
            comp_name,
            wetness_mm=wet,
            wear=wear,
            tyre_temp_c=tyre_temp_next,
            temp_window_shift_c=temp_window_shift_c,
            supplier_pace_rating=supplier_pace_rating,
            contract_grip_bonus_mult=contract_grip_bonus_mult,
        )
        grip_score = (0.6 * float(lat_mult)) + (0.4 * float(long_mult))
        deg_per_lap = float(
            tyre_model.wear_rate(
                comp_name,
                wetness_mm=wet,
                supplier_durability_rating=supplier_durability_rating,
                tyre_temp_c=tyre_temp_next,
                temp_window_shift_c=temp_window_shift_c,
            )
        )
        deg_per_lap *= wear_scale
        deg_per_lap *= track_wear
        deg_per_lap = max(0.0, deg_per_lap)
        return grip_score, deg_per_lap, tyre_temp_next

    @_projection_value
    def _recommended_wear_delta(
        self,
        rm,
        driver,
        comp_name: str,
        start_wear: float,
        start_lap: int,
    ) -> float:
        tyre_temp = self._starting_tyre_temperature(rm, driver, comp_name, start_wear, start_lap)
        physics = self._physics_scorer(rm)
        if physics is not None:
            temperature = tyre_temp if tyre_temp is not None else rm.tyre_model.initial_temperature_c(comp_name)
            end_wear, _, _ = physics.advance(driver, comp_name, start_wear,
                physics.fuel_at(driver, start_lap), temperature, start_lap)
            return max(0.0, end_wear - start_wear)
        _, deg, _ = self._projected_tyre_state(
            rm,
            driver,
            comp_name,
            start_wear,
            start_lap,
            tyre_temp,
        )
        return max(0.0, float(deg))

    def _plan_noise(
        self,
        variation: Dict[str, float],
        compounds: Sequence[str],
        lengths: Sequence[int],
    ) -> float:
        seed = int(variation.get("stint_seed", 0))
        signature = 0
        for idx, (comp, length) in enumerate(zip(compounds, lengths), start=1):
            comp_key = str(comp).lower()
            signature += idx * (sum(ord(ch) for ch in comp_key) + (31 * int(length)))
        rng = random.Random(seed + signature + 4099)
        return rng.uniform(-self.PLAN_NOISE_RANGE, self.PLAN_NOISE_RANGE)

    def _should_hold_for_weather(
        self,
        rm,
        driver,
        current_comp: str,
        current_wear: float,
        laps_done: int,
        pit_loss: float,
        wear_limit: float,
        comp_options: Sequence[str],
    ) -> bool:
        current_cat = self._compound_wet_category(current_comp, rm)
        if current_cat != "dry":
            return False
        lookahead = self._projected_wetness(rm, laps_done, 4)
        if len(lookahead) < 2:
            return False
        first_val = None
        try:
            first_val = float(lookahead[0])
        except Exception:
            first_val = None
        target_cat = None
        eta = None
        for idx, value in enumerate(lookahead[1:], start=1):
            try:
                wet_val = float(value)
            except Exception:
                continue
            cat = self._wetness_category(wet_val, rm)
            if cat in {"inter", "wet"} and (first_val is None or wet_val > first_val + 0.10):
                target_cat = cat
                eta = idx
                break
        if target_cat is None or eta is None or eta > 2:
            return False
        stay_score, end_wear = self._simulate_stint(
            rm,
            driver,
            current_comp,
            current_wear,
            eta,
            laps_done,
        )
        max_hold_wear = min(1.02, max(0.92, wear_limit * 1.08))
        if end_wear > max_hold_wear:
            return False
        best_switch = None
        for comp in comp_options:
            if self._compound_wet_category(comp, rm) != target_cat:
                continue
            switch_score, _ = self._simulate_stint(rm, driver, comp, 0.0, eta, laps_done)
            total_switch = float(pit_loss) + float(switch_score)
            if best_switch is None or total_switch < best_switch:
                best_switch = total_switch
        if best_switch is None:
            return False
        return float(stay_score) <= float(best_switch)

    def _numeric_stint_context(self, rm, driver, compound, invariants):
        """Snapshot the standard tyre model for this synchronous decision only.

        Keep model lookups in Python, including wetness boundary handling. The
        kernel receives their results, never a second set of weather rules.
        Overrides and non-finite/custom models use the reference evaluator.
        """
        memo = getattr(self, "_projection_memo", None)
        model = getattr(rm, "tyre_model", None)
        if (memo is None or type(model) is not TyreModel
                or not callable(getattr(_STRATEGY_KERNELS, "simulate_stint_exact", None))):
            return None
        key = (id(rm), driver.name, compound, invariants)
        contexts = memo["numeric_contexts"]
        if key in contexts:
            return contexts[key]
        for name, original in _NUMERIC_TYRE_METHODS.items():
            actual = getattr(model, name, None)
            if getattr(actual, "__func__", actual) is not getattr(original, "__func__", original):
                return None
        if (getattr(self._advance_tyre_temperature, "__func__", None) is not StrategyManager._advance_tyre_temperature
                or getattr(self._weather_projection_index, "__func__", None) is not StrategyManager._weather_projection_index):
            return None
        try:
            # The standard race lookup clamps to the final profile entry.
            # Custom weather accessors may have different extrapolation rules.
            from .race_manager import RaceManager
            if getattr(rm.lap_wetness, "__func__", None) is not RaceManager.lap_wetness:
                return None
            count = len(rm.wetness_profile)
            if not 0 < count <= 512:
                return None
            comp = model.get_compound_data(compound)
            if not comp:
                return None
            wear_scale, pace, durability, bonus, shift, track_wear = invariants
            warmup, cool = model.thermal_rates(compound)
            lo, hi = model.temperature_window_c(compound, window_shift_c=shift)
            wear_base = max(0.0, _safe_float(comp.get("wear_rate_base", .02), .02)
                            + (50.0 - model.supplier_durability_rating(durability)) * .0001)
            pace_mult = max(.9, 1.0 + (model.supplier_pace_rating(pace) - 50.0) * .0001)
            onset = model.cliff_onset(compound)
            params = (float(model.wear_quantum),
                      _safe_float(comp.get("lat_grip_base", 1), 1),
                      _safe_float(comp.get("long_grip_base", 1), 1),
                      float(comp.get("wear_grip_loss_lat", .15)),
                      float(comp.get("wear_grip_loss_long", .18)),
                      wear_base, pace_mult, bonus, wear_scale, track_wear,
                      warmup, cool, lo, hi, 0.0, 0.0,
                      float(callable(getattr(rm, "tyre_temperature_target", None))),
                      float(self.TYRE_TEMP_STEP_C_PER_LAP), float(self.WEAR_CLIFF_THRESHOLD),
                      float(self.WEAR_CLIFF_PENALTY), float(self.GRIP_TO_LAP_SECONDS),
                      -1.0 if onset is None else float(onset),
                      float(comp.get("wear_cliff_mult_lat", 1)),
                      float(comp.get("wear_cliff_mult_long", 1)))
            if not all(math.isfinite(value) for value in params) or params[0] <= 0:
                return None
            rows = []
            by_wetness = {}
            target = getattr(rm, "tyre_temperature_target", None)
            for index in range(count):
                wet = rm.lap_wetness(index)
                if wet is not None and not math.isfinite(float(wet)):
                    return None
                if wet not in by_wetness:
                    state = model.state_for_conditions(compound, wet)
                    row = (_safe_float(state.get("lat_mult", 1), 1),
                           _safe_float(state.get("long_mult", 1), 1),
                           _safe_float(state.get("wear_mult", 1), 1),
                           0.0 if wet is None else float(wet),
                           float(target(driver.name, wet)) if callable(target) else 0.0)
                    if not all(math.isfinite(value) for value in row):
                        return None
                    by_wetness[wet] = row
                rows.append(by_wetness[wet])
            context = tuple(tuple(row[column] for row in rows) for column in range(5)) + params
            # Every entry holds exactly len(rm.wetness_profile) rows for this
            # race, so dropping one entry frees exactly "count" rows.  Evict
            # oldest-first rather than clearing the whole table.
            while contexts and memo["numeric_rows"] + count > 2048:
                contexts.popitem(last=False)
                memo["numeric_rows"] = max(0, memo["numeric_rows"] - count)
            contexts[key] = context
            memo["numeric_rows"] += count
            return context
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return None

    @_projection_call
    def _simulate_stint(
        self,
        rm,
        driver,
        comp_name: str,
        start_wear: float,
        laps: int,
        start_lap: int,
        *, fuel_kg=None, new_set=None,
    ) -> Tuple[float, float]:
        physics = self._physics_scorer(rm)
        if physics is not None:
            memo = self._projection_memo
            if new_set is None:
                new_set = (bool(memo.get("new_start_set"))
                           or start_lap > memo.get("plan_origin", rm.laps.get(driver.name, 0))
                           or comp_name != rm.tyre_comp.get(driver.name)
                           or start_wear < rm.tyre_wear.get(driver.name, 0.0))
            return physics.stint(driver, comp_name, start_wear, laps, start_lap,
                                 fuel_kg=fuel_kg, new_set=new_set,
                                 sampled=bool(memo.get("physics_screening", False)))
        wear = float(start_wear)
        total = 0.0
        laps_int = max(0, int(laps))
        lap_origin = int(start_lap)
        cache_key = self._stint_cache_key(
            rm,
            driver,
            comp_name,
            lap_origin,
            laps_int,
            wear,
        )
        tyre_temp = self._starting_tyre_temperature(
            rm,
            driver,
            comp_name,
            wear,
            lap_origin,
        )
        configured_cliff = getattr(getattr(rm, "tyre_model", None), "cliff_onset", lambda _c: None)(comp_name)
        evaluator = self._projected_tyre_state
        standard_evaluator = getattr(evaluator, "__func__", None) is StrategyManager._projected_tyre_state
        invariants = self._projected_tyre_invariants(rm, driver, comp_name) if standard_evaluator and getattr(rm, "tyre_model", None) is not None and laps_int else None
        # Keep the established cache, but never alias exact starting conditions
        # when sharing compiled/Python projections across candidate plans.
        cache_key += (wear, tyre_temp, invariants, id(getattr(rm, "tyre_model", None)),
                      int(getattr(rm, "laps", {}).get(driver.name, 0)))
        cached = self._lookup_stint_cache(driver.name, cache_key)
        if cached is not None:
            return cached
        memo = getattr(self, "_projection_memo", None)
        prefix_key = None
        points = {}
        offset_start = 0
        if memo is not None and standard_evaluator and 0 < laps_int <= 256:
            prefix_key = (id(rm), id(getattr(rm, "tyre_model", None)),
                          cache_key[:3] + cache_key[4:], wear, tyre_temp, invariants,
                          int(getattr(rm, "laps", {}).get(driver.name, 0)))
            points = dict(memo["prefixes"].get(prefix_key, {}))
            if laps_int in points:
                if self._strategy_profile_enabled:
                    self._strategy_profile_counts["prefix_hits"] += 1
                total, wear, _temperature = points[laps_int]
                result = (total, wear)
                self._store_stint_cache(driver.name, cache_key, result)
                return result
            if points:
                offset_start = max(points)
                total, wear, tyre_temp = points[offset_start]
        context = self._numeric_stint_context(rm, driver, comp_name, invariants) if (
            standard_evaluator and invariants is not None and 0 < laps_int <= 256
            and math.isfinite(wear)
            and (tyre_temp is None or (math.isfinite(tyre_temp) and tyre_temp >= 0))
        ) else None
        if context is not None:
            # The Cython scorer still enters through this method, so it cannot
            # bypass cliff selection, exact prefix keys or cache invalidation.
            total, wear, tyre_temp, extension = _STRATEGY_KERNELS.simulate_stint_exact(
                context, wear, laps_int, lap_origin, self._weather_profile_index(rm),
                int(getattr(rm, "laps", {}).get(driver.name, 0)), offset_start,
                total, -1.0 if tyre_temp is None else tyre_temp,
            )
            if self._strategy_profile_enabled:
                self._strategy_profile_counts["compiled_laps"] += laps_int - offset_start
            if prefix_key is not None:
                points.update(extension)
            offset_start = laps_int
        for offset in range(offset_start, laps_int):
            lap_idx = lap_origin + offset
            cliff_penalty = 0.0
            if wear >= self.WEAR_CLIFF_THRESHOLD and configured_cliff is None:
                cliff_penalty = self.WEAR_CLIFF_PENALTY
            if standard_evaluator:
                grip_score, deg, tyre_temp = evaluator(rm, driver, comp_name, wear, lap_idx, tyre_temp, _invariants=invariants)
            else:
                grip_score, deg, tyre_temp = evaluator(rm, driver, comp_name, wear, lap_idx, tyre_temp)
            pace_delta = (1.0 - float(grip_score)) * self.GRIP_TO_LAP_SECONDS
            total += pace_delta + cliff_penalty
            wear += max(0.0, float(deg))
            if prefix_key is not None:
                points[offset + 1] = (total, wear, tyre_temp)
        if prefix_key is not None:
            prefixes = memo["prefixes"]
            previous = prefixes.pop(prefix_key, {})
            memo["prefix_states"] -= len(previous)
            while prefixes and memo["prefix_states"] + len(points) > 2048:
                _key, removed = prefixes.popitem(last=False)
                memo["prefix_states"] -= len(removed)
            prefixes[prefix_key] = points
            memo["prefix_states"] += len(points)
        result = (total, wear)
        self._store_stint_cache(driver.name, cache_key, result)
        return result

    @_projection_value
    def _wear_limit(
        self, rm, driver, comp_name: str, variation: Dict[str, float], stint_idx: int
    ) -> float:
        base = float(self.cfg.get("pitstops", {}).get("wear_threshold", 0.55))
        contract = rm.tyre_contract_type_by_driver.get(driver.name)
        if contract == "works":
            base *= 1.08
        elif contract == "partner":
            base *= 1.03

        base *= variation.get("wear_factor", 1.0)
        base *= self._variation_factor(variation, "wear", stint_idx, 0.92, 1.08)
        return max(0.35, min(0.9, base))

    @_projection_value
    def _recommended_length(
        self,
        rm,
        driver,
        comp_name: str,
        wear_limit: float,
        start_wear: float,
        start_lap: int,
        variation: Dict[str, float],
        stint_idx: int,
        honor_cliff: bool = True,
    ) -> float:
        deg = self._recommended_wear_delta(
            rm,
            driver,
            comp_name,
            start_wear,
            start_lap,
        )
        # Seed stint splits near the configured cliff; the race-time scorer
        # still decides whether an extra stop pays for itself.
        onset = getattr(getattr(rm, "tyre_model", None), "cliff_onset", lambda _c: None)(comp_name)
        if honor_cliff and onset is not None:
            wear_limit = min(wear_limit, onset)
        if deg <= 1e-6:
            return float(rm.total_laps)
        if start_wear >= wear_limit:
            return 0.0
        laps = (wear_limit - start_wear) / deg
        aggression = variation.get("aggression", 0.0)
        laps *= max(0.6, 1.0 - aggression)
        laps *= self._variation_factor(variation, "len", stint_idx, 0.95, 1.05)
        return max(0.0, laps)

    def _allocate_lengths(
        self, recommendations: Sequence[float], remaining_laps: int, allow_zero_first: bool
    ) -> Optional[List[int]]:
        if _STRATEGY_KERNELS is not None:
            return _STRATEGY_KERNELS.allocate_lengths(
                self, recommendations, remaining_laps, allow_zero_first
            )
        return self._allocate_lengths_python(
            recommendations, remaining_laps, allow_zero_first
        )

    def _allocate_lengths_python(
        self, recommendations: Sequence[float], remaining_laps: int, allow_zero_first: bool
    ) -> Optional[List[int]]:
        count = len(recommendations)
        if count == 0:
            return []
        if remaining_laps < max(0, count - 1):
            return None

        lengths: List[int] = []
        rem = int(remaining_laps)

        # Current stint
        max_for_first = rem - max(0, count - 1)
        first_rec = recommendations[0]
        if allow_zero_first and count > 1:
            if max_for_first <= 0:
                first_laps = 0
            else:
                desired = min(max_for_first, first_rec)
                first_laps = int(round(max(0.0, desired)))
                if first_rec < 0.75:
                    first_laps = 0
                first_laps = min(max_for_first, max(0, first_laps))
        else:
            desired = min(max_for_first if max_for_first > 0 else rem, first_rec)
            first_laps = int(round(max(desired, 0.0)))
            if max_for_first > 0:
                first_laps = max(1, min(max_for_first, first_laps))
            else:
                first_laps = max(0, first_laps)
        if count == 1:
            return [rem]
        lengths.append(first_laps)
        rem -= first_laps
        if rem < count - 1:
            deficit = (count - 1) - rem
            reduce = min(deficit, lengths[0])
            lengths[0] -= reduce
            rem += reduce
            if rem < count - 1:
                return None

        future_recs = [max(0.2, r) for r in recommendations[1:]]
        total_future = sum(future_recs)
        for idx, rec in enumerate(future_recs):
            stints_left = len(future_recs) - idx
            min_for_rest = stints_left - 1
            max_for_this = rem - min_for_rest
            if max_for_this < 0:
                return None
            if stints_left == 1:
                laps = rem
            else:
                share = rec / total_future if total_future > 0 else 1.0 / stints_left
                desired = share * rem
                laps = int(round(max(1.0, desired)))
                if laps > max_for_this:
                    laps = max_for_this
                laps = max(1, laps)
            lengths.append(laps)
            rem -= laps
            total_future -= rec
        if rem != 0:
            lengths[-1] += rem
        return self._rebalance_final_stint_python(
            lengths, recommendations, allow_zero_first
        )

    def _rebalance_final_stint(
        self,
        lengths: Sequence[int],
        recommendations: Sequence[float],
        allow_zero_first: bool,
    ) -> List[int]:
        if _STRATEGY_KERNELS is not None:
            return _STRATEGY_KERNELS.rebalance_final_stint(
                self, lengths, recommendations, allow_zero_first
            )
        return self._rebalance_final_stint_python(
            lengths, recommendations, allow_zero_first
        )

    def _rebalance_final_stint_python(
        self,
        lengths: Sequence[int],
        recommendations: Sequence[float],
        allow_zero_first: bool,
    ) -> List[int]:
        if not lengths:
            return []
        new_lengths = [max(0, int(l)) for l in lengths]
        if len(new_lengths) <= 1:
            return new_lengths
        total_remaining = sum(new_lengths)
        if total_remaining <= 0:
            return new_lengths
        final_rec = recommendations[-1] if recommendations else None
        target = self._final_target_length_python(final_rec, total_remaining)
        if target <= 0:
            return new_lengths
        current = new_lengths[-1]
        if current >= target:
            return new_lengths
        needed = target - current
        moved = 0
        for idx in range(len(new_lengths) - 1):
            if moved >= needed:
                break
            min_len = 0 if idx == 0 and allow_zero_first and len(new_lengths) > 1 else 1
            available = max(0, new_lengths[idx] - min_len)
            if available <= 0:
                continue
            take = min(available, needed - moved)
            if take <= 0:
                continue
            new_lengths[idx] -= take
            moved += take
        if moved <= 0:
            return new_lengths
        new_lengths[-1] = current + moved
        return new_lengths

    def _final_target_length(
        self, final_recommendation: Optional[float], total_remaining: int
    ) -> int:
        if _STRATEGY_KERNELS is not None:
            return _STRATEGY_KERNELS.final_target_length(
                self, final_recommendation, total_remaining
            )
        return self._final_target_length_python(
            final_recommendation, total_remaining
        )

    def _final_target_length_python(
        self, final_recommendation: Optional[float], total_remaining: int
    ) -> int:
        if total_remaining <= 0:
            return 0
        target: float
        if final_recommendation is not None and final_recommendation > 0:
            if final_recommendation < 4.0:
                target = max(1.0, final_recommendation)
            else:
                target = max(3.0, final_recommendation * 0.7)
        else:
            target = max(3.0, total_remaining * 0.2)
        target_int = int(round(target))
        target_int = max(1, target_int)
        target_int = min(total_remaining, target_int)
        return target_int

    def _score_plan(
        self,
        rm,
        driver,
        start_lap: int,
        compounds: Sequence[str],
        lengths: Sequence[int],
        wear_limits: Sequence[float],
        recommendations: Sequence[float],
        current_wear: float,
        pit_loss: float,
        mandatory_count: int,
        used_compounds: Iterable[str],
        stint_start_wears: Optional[Sequence[float]] = None,
    ) -> Tuple[float, Optional[List[dict]], Optional[float]]:
        if _STRATEGY_KERNELS is not None:
            return _STRATEGY_KERNELS.score_plan(
                self,
                rm,
                driver,
                start_lap,
                compounds,
                lengths,
                wear_limits,
                recommendations,
                current_wear,
                pit_loss,
                mandatory_count,
                used_compounds,
                stint_start_wears,
            )
        return self._score_plan_python(
            rm,
            driver,
            start_lap,
            compounds,
            lengths,
            wear_limits,
            recommendations,
            current_wear,
            pit_loss,
            mandatory_count,
            used_compounds,
            stint_start_wears,
        )

    def _score_plan_python(
        self,
        rm,
        driver,
        start_lap: int,
        compounds: Sequence[str],
        lengths: Sequence[int],
        wear_limits: Sequence[float],
        recommendations: Sequence[float],
        current_wear: float,
        pit_loss: float,
        mandatory_count: int,
        used_compounds: Iterable[str],
        stint_start_wears: Optional[Sequence[float]] = None,
    ) -> Tuple[float, Optional[List[dict]], Optional[float]]:
        stints: List[dict] = []
        lap_cursor = int(start_lap)
        wear = float(current_wear)
        total = 0.0
        baseline = self._baseline_reference_lap(rm, driver)
        used_norm = {
            str(comp).lower() for comp in used_compounds if self._counts_toward_mandatory(comp, rm)
        }
        total_remaining = sum(max(0, int(l)) for l in lengths)

        for idx, (comp, laps, w_limit) in enumerate(zip(compounds, lengths, wear_limits)):
            start = lap_cursor
            start_wear = (
                max(0.0, float(stint_start_wears[idx]))
                if stint_start_wears is not None and idx < len(stint_start_wears)
                else wear
                if idx == 0
                else 0.0
            )
            if laps <= 0:
                stint_time = 0.0
                end_wear = start_wear
            else:
                stint_time, end_wear = self._simulate_stint(
                    rm, driver, comp, start_wear, laps, start,
                    new_set=idx > 0 or bool((getattr(self, "_projection_memo", None) or {}).get("new_start_set"))
                )
                total += stint_time
            comp_norm = str(comp).lower()
            is_new = comp_norm not in used_norm
            counts_for_rule = (
                self._counts_toward_mandatory(comp, rm)
                and is_new
                and len(used_norm) < mandatory_count
            )
            stints.append(
                {
                    "compound": comp,
                    "planned_laps": int(laps),
                    "start_lap": start,
                    "target_end_lap": start + int(laps),
                    "start_wear": float(start_wear),
                    "wear_limit": w_limit,
                    "mandatory": counts_for_rule,
                }
            )
            if self._counts_toward_mandatory(comp, rm):
                used_norm.add(comp_norm)
            lap_cursor += int(laps)
            wear = 0.0
            if idx < len(lengths) - 1:
                total += pit_loss

        if len(used_norm) < mandatory_count:
            return float("inf"), None, baseline

        if len(lengths) > 1 and total_remaining > 0:
            final_len = max(0, int(lengths[-1]))
            target = self._final_target_length_python(
                recommendations[-1] if recommendations else None, total_remaining
            )
            shortfall = max(0, target - final_len)
            if shortfall > 0:
                    lap_penalty = max(pit_loss * 0.45, baseline * 0.08)
                    total += lap_penalty * shortfall

        return total, stints, baseline

    @_projection_call
    def _generate_plan(
        self,
        rm,
        driver,
        start_lap: int,
        start_comp: str,
        current_wear: float,
        used_compounds: Iterable[str],
        pits_done: int,
        variation: Optional[Dict[str, float]] = None,
        restrict_defaults: bool = False,
        consume_start_set: bool = False,
    ) -> Tuple[Optional[dict], Optional[float]]:
        memo = self._projection_memo
        prior = (memo.get("plan_origin"), memo.get("new_start_set"), memo.get("physics_screening"))
        memo["plan_origin"], memo["new_start_set"] = start_lap, consume_start_set
        physics = self._physics_scorer(rm)
        memo["physics_screening"] = physics is not None
        if physics is not None:
            physics.collecting = True
        try:
            if _STRATEGY_KERNELS is not None:
                return _STRATEGY_KERNELS.generate_plan(
                    self,
                    rm,
                    driver,
                    start_lap,
                    start_comp,
                    current_wear,
                    used_compounds,
                    pits_done,
                    variation,
                    restrict_defaults,
                    consume_start_set,
                )
            return self._generate_plan_python(
                rm,
                driver,
                start_lap,
                start_comp,
                current_wear,
                used_compounds,
                pits_done,
                variation,
                restrict_defaults,
                consume_start_set,
            )

        finally:
            if physics is not None:
                physics.collecting = False
                physics.pending.clear()
            for key, value in zip(("plan_origin", "new_start_set", "physics_screening"), prior):
                if value is None:
                    memo.pop(key, None)
                else:
                    memo[key] = value

    @_projection_call
    def _generate_plan_python(
        self,
        rm,
        driver,
        start_lap: int,
        start_comp: str,
        current_wear: float,
        used_compounds: Iterable[str],
        pits_done: int,
        variation: Optional[Dict[str, float]] = None,
        restrict_defaults: bool = False,
        consume_start_set: bool = False,
    ) -> Tuple[Optional[dict], Optional[float]]:
        if not self._pit_logic_enabled(rm):
            return {
                "stints": [],
                "pit_loss": 0.0,
                "baseline_lap": None,
                "mandatory_count": self._mandatory_compounds(rm),
            }, 0.0

        variation = variation or self._variation_for_driver(driver)
        pit_loss = self._estimate_pit_loss(rm, driver, variation)
        remaining_laps = rm.total_laps - int(start_lap)
        mandatory_base = self._mandatory_compounds(rm)
        used_set = set(used_compounds)
        used_set.add(start_comp)
        comp_options = self._candidate_compounds(rm, start_lap, remaining_laps)
        comp_options, physical_wear_queues = self._plan_compound_choices(
            rm, driver.name, comp_options, start_comp if consume_start_set else None)
        if not comp_options:
            # No replacement set remains.  The only legal plan is to stay on
            # the currently fitted physical set to the finish.
            comp_options = []
        cache_key = self._plan_cache_key(
            rm,
            driver.name,
            start_lap,
            start_comp,
            current_wear,
            used_set,
            comp_options,
            pits_done,
            variation,
            restrict_defaults,
            consume_start_set,
        )
        cached = self._lookup_plan_cache(driver.name, cache_key)
        if cached is not None:
            cached_plan, cached_time = cached
            return (self._clone_plan(cached_plan), cached_time)

        used_dry_norm = {
            str(c).lower() for c in used_set if self._counts_toward_mandatory(c, rm)
        }
        option_dry_norm = {
            str(c).lower() for c in comp_options if self._counts_toward_mandatory(c, rm)
        }
        effective_mandatory = min(mandatory_base, len(option_dry_norm | used_dry_norm))
        if remaining_laps <= 0:
            plan = {
                "stints": [],
                "pit_loss": pit_loss,
                "baseline_lap": None,
                "mandatory_count": effective_mandatory,
                "plan_time": 0.0,
            }
            stored = self._clone_plan(plan)
            self._store_plan_cache(driver.name, cache_key, stored, 0.0)
            return self._clone_plan(stored), 0.0

        mandatory_needed = max(0, effective_mandatory - len(used_dry_norm))
        if restrict_defaults:
            base_cap = self._default_stop_cap(rm)
        else:
            base_cap = self.MAX_PLANNED_STOPS
        base_cap = max(base_cap, pits_done + mandatory_needed)
        base_cap = min(self.MAX_PLANNED_STOPS, base_cap)
        max_additional = max(mandatory_needed, max(0, base_cap - pits_done))
        min_additional = mandatory_needed
        if min_additional > max_additional:
            max_additional = min_additional

        physics_candidates = []
        best_plan = None
        best_score = None
        best_time = None

        for stops in range(min_additional, max_additional + 1):
            stint_count = stops + 1
            if stint_count <= 0:
                continue
            if remaining_laps < max(0, stint_count - 1):
                continue

            # Hoist initial stint invariants: they do not vary across candidate sequences
            limit0 = self._wear_limit(rm, driver, start_comp, variation, 0)
            rec0 = self._recommended_length(
                rm,
                driver,
                start_comp,
                limit0,
                current_wear,
                start_lap,
                variation,
                0,
            )
            allow_zero_first = False
            if stops > 0:
                near_limit = limit0 <= 1e-6 or current_wear >= limit0 * 0.95
                low_distance = remaining_laps <= max(1, stops)
                onset = getattr(getattr(rm, "tyre_model", None), "cliff_onset", lambda _c: None)(start_comp)
                if near_limit or low_distance or (onset is not None and current_wear >= onset):
                    allow_zero_first = True

            for seq in self._sequence_iter(comp_options, stint_count - 1):
                compounds = [start_comp] + list(seq)
                stint_start_wears = self._sequence_physical_start_wears(
                    compounds,
                    current_wear,
                    physical_wear_queues,
                )
                if stint_start_wears is None:
                    continue
                total_unique = self._unique_mandatory_count(list(compounds) + list(used_set), rm)
                if total_unique < effective_mandatory:
                    continue

                wear_limits = [limit0]
                recommendations = [rec0]
                projected_lap_cursor = int(start_lap) + max(0, int(round(rec0)))
                for idx, comp in enumerate(seq, start=1):
                    limit = self._wear_limit(rm, driver, comp, variation, idx)
                    wear_limits.append(limit)
                    rec = self._recommended_length(
                        rm,
                        driver,
                        comp,
                        limit,
                        stint_start_wears[idx],
                        projected_lap_cursor,
                        variation,
                        idx,
                    )
                    recommendations.append(rec)
                    projected_lap_cursor += max(0, int(round(rec)))

                lengths = self._allocate_lengths_python(
                    recommendations, remaining_laps, allow_zero_first
                )
                if lengths is None:
                    continue
                score_time, stints, baseline = self._score_plan_python(
                    rm,
                    driver,
                    start_lap,
                    compounds,
                    lengths,
                    wear_limits,
                    recommendations,
                    current_wear,
                    pit_loss,
                    effective_mandatory,
                    used_set,
                    stint_start_wears,
                )
                if stints is None:
                    continue
                score = score_time
                score += (stint_count - 1) * variation.get("stop_bias", 0.0)
                score += self._plan_noise(variation, compounds, lengths)
                physics_candidates.append((score, {"stints": stints, "pit_loss": pit_loss,
                    "baseline_lap": baseline, "mandatory_count": effective_mandatory, "plan_time": score_time}))
                if best_score is None or score < best_score:
                    best_score = score
                    best_time = score_time
                    best_plan = {
                        "stints": stints,
                        "pit_loss": pit_loss,
                        "baseline_lap": baseline,
                        "mandatory_count": effective_mandatory,
                        "plan_time": score_time,
                    }

        if self._physics_scorer(rm) is not None:
            best_plan, best_time = self._finalize_physics_candidates(rm, driver, physics_candidates, used_set, variation)
        if best_plan is None:
            # Fallback: ensure a legal plan exists even if optimisation fails.
            dry_candidates = [
                c
                for c in comp_options
                if self._counts_toward_mandatory(c, rm)
                and str(c).lower() not in used_dry_norm
            ]
            fallback_compounds: List[str] = [start_comp]
            needed = max(0, effective_mandatory - len(used_dry_norm))
            idx = 0
            while needed > 0 and idx < len(dry_candidates):
                fallback_compounds.append(dry_candidates[idx])
                idx += 1
                needed -= 1
            if self._unique_mandatory_count(fallback_compounds + list(used_set), rm) < effective_mandatory and comp_options:
                for comp in comp_options:
                    fallback_compounds.append(comp)
                    if (
                        self._unique_mandatory_count(fallback_compounds + list(used_set), rm)
                        >= effective_mandatory
                    ):
                        break

            fallback_start_wears = self._sequence_physical_start_wears(
                fallback_compounds,
                current_wear,
                physical_wear_queues,
            )
            if fallback_start_wears is None:
                fallback_compounds = [start_comp]
                fallback_start_wears = [float(current_wear)]

            stint_lengths: List[int] = []
            laps_left = remaining_laps
            future_stints = len(fallback_compounds) - 1
            first_len = max(0, laps_left - max(1, future_stints)) if future_stints > 0 else laps_left
            if future_stints == 0:
                first_len = laps_left
            stint_lengths.append(first_len)
            laps_left -= first_len
            for idx in range(1, len(fallback_compounds)):
                remaining_slots = len(fallback_compounds) - idx
                min_needed = remaining_slots - 1
                length = 1 if laps_left > min_needed else max(0, laps_left - min_needed)
                stint_lengths.append(length)
                laps_left -= length
            if laps_left > 0:
                stint_lengths[-1] += laps_left

            fallback_recs = [float(max(1, l)) for l in stint_lengths]
            stint_lengths = self._rebalance_final_stint_python(
                stint_lengths, fallback_recs, allow_zero_first=False
            )
            fallback_recs = [float(max(1, l)) for l in stint_lengths]

            fallback_stints: List[dict] = []
            lap_cursor = int(start_lap)
            total_time = 0.0
            baseline = None
            unique_tracker = {
                str(comp).lower() for comp in used_set if self._counts_toward_mandatory(comp, rm)
            }
            for idx, (comp, length) in enumerate(zip(fallback_compounds, stint_lengths)):
                limit = self._wear_limit(rm, driver, comp, variation, idx)
                start_wear = float(fallback_start_wears[idx])
                stint_time, _ = self._simulate_stint(
                    rm,
                    driver,
                    comp,
                    start_wear,
                    length,
                    lap_cursor,
                )
                total_time += stint_time
                fallback_stints.append(
                    {
                        "compound": comp,
                        "planned_laps": int(length),
                        "start_lap": lap_cursor,
                        "target_end_lap": lap_cursor + int(length),
                        "start_wear": start_wear,
                        "wear_limit": limit,
                        "mandatory": (
                            self._counts_toward_mandatory(comp, rm)
                            and str(comp).lower() not in unique_tracker
                            and len(unique_tracker) < effective_mandatory
                        ),
                    }
                )
                if self._counts_toward_mandatory(comp, rm):
                    unique_tracker.add(str(comp).lower())
                lap_cursor += int(length)
                if idx < len(fallback_compounds) - 1:
                    total_time += pit_loss

            if baseline is None:
                baseline = self._baseline_reference_lap(rm, driver)

            total_remaining_fb = sum(max(0, int(l)) for l in stint_lengths)
            if len(stint_lengths) > 1 and total_remaining_fb > 0:
                final_len_fb = max(0, int(stint_lengths[-1]))
                target_fb = self._final_target_length_python(
                    fallback_recs[-1] if fallback_recs else None, total_remaining_fb
                )
                shortfall_fb = max(0, target_fb - final_len_fb)
                if shortfall_fb > 0:
                    lap_penalty_fb = max(pit_loss * 0.45, baseline * 0.08)
                    total_time += lap_penalty_fb * shortfall_fb

            plan = {
                "stints": fallback_stints,
                "pit_loss": pit_loss,
                "baseline_lap": baseline,
                "mandatory_count": effective_mandatory,
                "plan_time": total_time,
            }
            stored = self._clone_plan(plan)
            self._store_plan_cache(driver.name, cache_key, stored, total_time)
            return self._clone_plan(stored), total_time

        if not consume_start_set and self._physics_scorer(rm) is None:
            best_plan, best_time = self._refine_cliff_plan(
                rm, driver, best_plan, best_time, used_set, variation
            )
        stored_plan = self._clone_plan(best_plan)
        self._store_plan_cache(driver.name, cache_key, stored_plan, best_time)
        return (self._clone_plan(stored_plan), best_time)

    @staticmethod
    def _physics_shortlist(candidates):
        ordered = sorted(candidates, key=lambda row: row[0])
        selected = ordered[:1]
        # Keep the best screened plan for each stop count, especially staying
        # out. A sampled estimate must not silently delete that alternative.
        counts = {len(row[1]["stints"]) for row in selected}
        for row in ordered[1:]:
            count = len(row[1]["stints"])
            if count not in counts:
                selected.append(row)
                counts.add(count)
        return selected

    def _finalize_physics_candidates(self, rm, driver, candidates, used, variation):
        physics = self._physics_scorer(rm)
        physics.flush()

        def price(plan):
            stints = plan["stints"]
            compounds = [s["compound"] for s in stints]
            lengths = [s["planned_laps"] for s in stints]
            wears = [s["start_wear"] for s in stints]
            limits = [s["wear_limit"] for s in stints]
            recs = [self._recommended_length(rm, driver, c, limits[i], wears[i],
                                            stints[i]["start_lap"], variation, i)
                    for i, c in enumerate(compounds)]
            cost, revised, baseline = self._score_plan(
                rm, driver, stints[0]["start_lap"], compounds, lengths, limits,
                recs, wears[0], plan["pit_loss"], plan["mandatory_count"], used, wears)
            score = cost + (len(stints)-1)*variation.get("stop_bias", 0.) + self._plan_noise(variation, compounds, lengths)
            return score, dict(plan, stints=revised, plan_time=cost, baseline_lap=baseline)

        # Price all screening samples in one batch, then all shortlisted laps
        # in a second batch. This shares the existing parallel physics solver.
        # performance fix: wrapping the candidate loop in physics.collecting = True queues all
        # missing sample knots from all 24 candidates into pending first and evaluating them
        # in a single batch via flush()
        physics.collecting = True
        physics.deferred_misses = 0
        first_pass = [price(plan) for _, plan in candidates]
        deferred = physics.deferred_misses > 0
        physics.flush()
        # When nothing was missing the collecting pass already read real lap
        # times through the same pricing kernel, so it produced identical
        # numbers and the second pass is pure duplicate work.
        screened = [price(plan) for _, plan in candidates] if deferred else first_pass
        if screened and not self._projection_memo.get("new_start_set"):
            _, seed = min(screened, key=lambda row: row[0])
            refined, cost = self._refine_cliff_plan(rm, driver, seed, seed["plan_time"], used, variation)
            if refined != seed:
                compounds = [row["compound"] for row in refined["stints"]]
                lengths = [row["planned_laps"] for row in refined["stints"]]
                score = cost + (len(lengths)-1)*variation.get("stop_bias", 0.) + self._plan_noise(variation, compounds, lengths)
                screened.append((score, refined))
        chosen = self._physics_shortlist(screened)
        immediate = {}
        if not self._projection_memo.get("new_start_set"):
            for row in sorted(screened, key=lambda row: row[0]):
                stints = row[1]["stints"]
                if len(stints) > 1 and stints[0]["planned_laps"] == 0:
                    comp = stints[1]["compound"]
                    if comp not in immediate:
                        immediate[comp] = row[1]
                        if not any(row[1] == entry[1] for entry in chosen):
                            chosen.append(row)
        # price(chosen[0][1]) is executed with physics.collecting = False which is
        # causing all 24 unsampled laps of chosen[0] to be evaluated individually and unbatched
        # it also reevaluated chosen[0] a second time
        self._projection_memo["physics_screening"] = False
        physics.collecting = True
        physics.deferred_misses = 0
        first_exact = [price(plan) for _, plan in chosen]
        deferred = physics.deferred_misses > 0
        physics.flush()
        exact = [price(plan) for _, plan in chosen] if deferred else first_exact

        if not exact:
            return None, None
        _, best = min(exact, key=lambda row: row[0])
        prepared = {}
        for _, plan in exact:
            stints = plan["stints"]
            if len(stints) > 1 and stints[0]["planned_laps"] == 0:
                comp = stints[1]["compound"]
                after = dict(plan, stints=stints[1:], plan_time=plan["plan_time"]-plan["pit_loss"])
                if comp not in prepared or after["plan_time"] < prepared[comp]["plan_time"]:
                    prepared[comp] = after
        best["immediate_physics_plans"] = prepared
        return best, best["plan_time"]

    def _refine_cliff_plan(self, rm, driver, plan, plan_time, used, variation):
        """Price a small neighbourhood of the selected tyre sequence only.

        Keep the same physical sets, compounds and stop count. Include a stretch
        to the ordinary wear window; a cliff is a pace cost, not a hard limit.
        """
        stints = plan.get("stints", []) if plan else []
        if len(stints) < 2 or getattr(getattr(rm, "tyre_model", None), "cliff_onset", lambda _c: None)(stints[0]["compound"]) is None:
            return plan, plan_time
        compounds = [s["compound"] for s in stints]
        lengths = [s["planned_laps"] for s in stints]
        wears = [s["start_wear"] for s in stints]
        limits = [s["wear_limit"] for s in stints]
        recs = [self._recommended_length(rm, driver, c, limits[i], wears[i],
                                        stints[i]["start_lap"], variation, i)
                for i, c in enumerate(compounds)]
        stretch = int(round(self._recommended_length(
            rm, driver, compounds[0], limits[0], wears[0], stints[0]["start_lap"],
            variation, 0, honor_cliff=False
        )))
        best_score = plan_time + self._plan_noise(variation, compounds, lengths)
        options = sorted({max(0, lengths[0] - 2), lengths[0] + 2, stretch} - {lengths[0]})

        def try_option(first):
            candidate = list(lengths)
            candidate[0] = first
            candidate[1] += lengths[0] - first
            if candidate[1] < 1:
                return None
            time_cost, revised, baseline = self._score_plan(
                rm, driver, stints[0]["start_lap"], compounds, candidate, limits,
                recs, wears[0], plan["pit_loss"], plan["mandatory_count"], used, wears
            )
            score = time_cost + self._plan_noise(variation, compounds, candidate)
            return (time_cost, revised, baseline, score)

        # Each candidate here used to price through its own uncollected call,
        # forcing one extra unbatched physics solve per option (up to three
        # here, one to three stints each) on top of the two batches
        # _finalize_physics_candidates already issues. Queue them the same
        # way: a collecting pass to discover what is missing, then one batch,
        # then re-price only if anything was actually missing.
        physics = self._physics_scorer(rm)
        if physics is not None:
            physics.collecting = True
            physics.deferred_misses = 0
        first_pass = [try_option(first) for first in options]
        deferred = physics is not None and physics.deferred_misses > 0
        if physics is not None:
            physics.flush()
        results = [try_option(first) for first in options] if deferred else first_pass

        for entry in results:
            if entry is None:
                continue
            time_cost, revised, baseline, score = entry
            if revised is not None and score < best_score:
                best_score = score
                plan = dict(plan, stints=revised, plan_time=time_cost, baseline_lap=baseline)
                plan_time = time_cost
        return plan, plan_time

    def _sequence_iter(self, compounds: Sequence[str], length: int) -> Iterable[Tuple[str, ...]]:
        if length <= 0:
            yield tuple()
            return
        for seq in product(compounds, repeat=length):
            yield seq

    def _replan_from_state(self, rm, driver) -> None:
        plan = self.plans.get(driver.name)
        variation = plan.get("variation") if plan else self._variation_for_driver(driver)
        start_comp = rm.tyre_comp.get(driver.name, self._default_compound(rm))
        used = set(rm.used_compounds.get(driver.name, {start_comp}))
        used.add(start_comp)
        laps_done = rm.laps.get(driver.name, 0)
        current_wear = float(rm.tyre_wear.get(driver.name, 0.0))
        pits_done = rm.pit_count.get(driver.name, 0)
        self._invalidate_plan_cache(driver.name)
        projection, time_keep = self._generate_plan(
            rm,
            driver,
            start_lap=laps_done,
            start_comp=start_comp,
            current_wear=current_wear,
            used_compounds=used,
            pits_done=pits_done,
            variation=variation,
            restrict_defaults=True,
        )
        if projection is None:
            return
        self.plans[driver.name] = {
            "variation": variation,
            "stints": projection.get("stints", []),
            "pit_loss": projection.get("pit_loss", plan.get("pit_loss") if plan else self._estimate_pit_loss(rm, driver, variation)),
            "baseline_lap": projection.get("baseline_lap"),
            "mandatory_count": projection.get("mandatory_count", self._mandatory_compounds(rm)),
            "completed_pits": rm.pit_count.get(driver.name, 0),
            "plan_time": time_keep,
            "last_lap_seen": rm.laps.get(driver.name, 0),
            "last_replan_tick": getattr(rm, "tick_count", 0),
            "decision_cache_key": None,
            "decision_cache_value": None,
        }

    def _profile_signature(self, profile: Optional[Sequence]) -> Optional[Tuple[int, ...]]:
        if not profile:
            return None
        signature: List[int] = []
        for value in profile:
            try:
                signature.append(int(round(float(value) * 100)))
            except Exception:
                signature.append(0)
        return tuple(signature)

    def _variation_signature(self, variation: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
        items: List[Tuple[str, float]] = []
        for key, value in sorted(variation.items()):
            try:
                items.append((key, round(float(value), 5)))
            except Exception:
                items.append((key, 0.0))
        return tuple(items)

    def _plan_cache_key(
        self,
        rm,
        driver_name: str,
        start_lap: int,
        start_comp: str,
        current_wear: float,
        used_set: Iterable[str],
        comp_options: Sequence[str],
        pits_done: int,
        variation: Dict[str, float],
        restrict_defaults: bool,
        consume_start_set: bool = False,
    ) -> Tuple:
        used_norm = tuple(sorted(str(c).lower() for c in used_set))
        comp_tuple = tuple(str(c) for c in comp_options)
        weather_sig = self._profile_signature(getattr(rm, "wetness_profile", None))
        temp_sig = self._profile_signature(getattr(rm, "temperature_profile", None))
        weather_now_index = self._weather_profile_index(rm)
        variation_sig = self._variation_signature(variation)
        track_wear = getattr(rm, "track_wear_mult", 1.0)
        physical_inventory_sig = None
        allocation_manager = getattr(rm, "weekend_tyre_manager", None)
        weekend = getattr(rm, "weekend", None)
        if (
            allocation_manager is not None
            and allocation_manager.enabled()
            and weekend is not None
        ):
            physical_inventory_sig = tuple(
                (
                    str(comp),
                    tuple(
                        round(max(0.0, float(row.get("wear", 0.0) or 0.0)), 3)
                        for row in sorted(
                            allocation_manager.available_sets_for_session(
                                weekend,
                                driver_name,
                                comp,
                                "race",
                            ),
                            key=lambda item: (
                                float(item.get("wear", 0.0) or 0.0),
                                int(item.get("laps", 0) or 0),
                                str(item.get("set_id", "")),
                            ),
                        )
                    ),
                )
                for comp in comp_tuple
            )
        return (
            driver_name,
            int(start_lap),
            str(start_comp),
            round(float(current_wear), 3),
            used_norm,
            comp_tuple,
            int(pits_done),
            variation_sig,
            bool(restrict_defaults),
            bool(consume_start_set),
            int(getattr(rm, "total_laps", 0)),
            round(float(track_wear), 3),
            weather_sig,
            temp_sig,
            int(weather_now_index),
            physical_inventory_sig,
            self._mandatory_compounds(rm),
        )

    def _lookup_plan_cache(self, driver_name: str, key: Tuple):
        memo = getattr(self, "_projection_memo", None)
        if memo is not None and memo.get("physics") is not None:
            return memo.setdefault("physics_plans", {}).get((driver_name, key))
        bucket = self._plan_cache.get(driver_name)
        if not bucket:
            return None
        entry = bucket.get(key)
        if entry is not None:
            bucket.move_to_end(key)
        return entry

    def _store_plan_cache(
        self,
        driver_name: str,
        key: Tuple,
        plan: Optional[dict],
        plan_time: Optional[float],
    ) -> None:
        memo = getattr(self, "_projection_memo", None)
        if memo is not None and memo.get("physics") is not None:
            cache = memo.setdefault("physics_plans", {})
            if len(cache) >= 256:
                cache.clear()
            cache[(driver_name, key)] = (plan, plan_time)
            return
        bucket = self._plan_cache.setdefault(driver_name, OrderedDict())
        bucket[key] = (plan, plan_time)
        bucket.move_to_end(key)
        while len(bucket) > self.PLAN_CACHE_LIMIT:
            bucket.popitem(last=False)

    def _invalidate_plan_cache(self, driver_name: Optional[str] = None) -> None:
        memo = getattr(self, "_projection_memo", None)
        if memo is not None:
            memo["prefixes"].clear()
            memo["prefix_states"] = 0
            memo["values"].clear()
            memo["numeric_contexts"].clear()
            memo["numeric_rows"] = 0
            memo.pop("physics", None)
            memo.pop("physics_plans", None)
            caches = getattr(memo.get("lookup_model"), "_lookup_caches", None)
            if caches is not None:
                for cache in caches.values():
                    cache.clear()
        if driver_name is None:
            self._plan_cache.clear()
            self._stint_cache.clear()
            return
        bucket = self._plan_cache.get(driver_name)
        if bucket is not None:
            bucket.clear()
        self._invalidate_stint_cache(driver_name)

    def _stint_cache_key(
        self,
        rm,
        driver,
        comp_name: str,
        start_lap: int,
        laps: int,
        start_wear: float,
    ) -> Tuple:
        weather_sig = self._profile_signature(getattr(rm, "wetness_profile", None))
        temp_sig = self._profile_signature(getattr(rm, "temperature_profile", None))
        weather_now_index = self._weather_profile_index(rm)
        track_wear = getattr(rm, "track_wear_mult", 1.0)
        contract_type = getattr(rm, "tyre_contract_type_by_driver", {}).get(driver.name)
        supplier_obj = getattr(rm, "tyre_supplier_by_driver", {}).get(driver.name)
        supplier_name = None
        supplier_pace_rating = 50.0
        supplier_durability_rating = 50.0
        if supplier_obj is not None:
            supplier_name = getattr(supplier_obj, "name", str(supplier_obj))
            tyre_model = getattr(rm, "tyre_model", None)
            if tyre_model is not None:
                try:
                    comp_key = tyre_model.normalize_compound_name(comp_name)
                except Exception:
                    comp_key = str(comp_name).lower().strip()
                try:
                    raw_pace = (getattr(supplier_obj, "pace", {}) or {}).get(comp_key, 50.0)
                except Exception:
                    raw_pace = 50.0
                try:
                    raw_dur = (getattr(supplier_obj, "durability", {}) or {}).get(comp_key, 50.0)
                except Exception:
                    raw_dur = 50.0
                try:
                    supplier_pace_rating = float(tyre_model.supplier_pace_rating(raw_pace))
                except Exception:
                    supplier_pace_rating = 50.0
                try:
                    supplier_durability_rating = float(tyre_model.supplier_durability_rating(raw_dur))
                except Exception:
                    supplier_durability_rating = 50.0
        contract_meta = getattr(rm, "tyre_contracts", {}).get(driver.team, {}) or {}
        raw_bonus = contract_meta.get("bonus_grip_mult")
        if raw_bonus is None:
            if str(contract_type or "").lower() == "partner" and contract_meta.get("bonus_pace") is not None:
                raw_bonus = 1.0005
            elif str(contract_type or "").lower() == "works" and float(contract_meta.get("funding_weekly_m", 0.0) or 0.0) > 0.0:
                raw_bonus = 1.0005
        try:
            grip_bonus_mult = float(raw_bonus) if raw_bonus is not None else 1.0
        except Exception:
            grip_bonus_mult = 1.0
        if grip_bonus_mult <= 0.0:
            grip_bonus_mult = 1.0
        try:
            if hasattr(rm, "driver_tyre_management_factor"):
                management = float(rm.driver_tyre_management_factor(driver))
            else:
                management = float(
                    getattr(rm, "team_meta", {}).get(driver.team, {}).get("tyre_management", 1.0)
                )
        except Exception:
            management = 1.0
        wear_quantum = 0.02
        tyre_model = getattr(rm, "tyre_model", None)
        if tyre_model is not None:
            try:
                wear_quantum = max(0.001, float(getattr(tyre_model, "wear_quantum", 0.02)))
            except Exception:
                wear_quantum = 0.02
        try:
            wear_for_key = max(0.0, min(1.0, float(start_wear)))
        except Exception:
            wear_for_key = 0.0
        if wear_quantum > 1e-9:
            wear_for_key = round(wear_for_key / wear_quantum) * wear_quantum
        return (
            driver.name,
            str(comp_name),
            int(start_lap),
            int(laps),
            round(float(wear_for_key), 4),
            round(float(wear_quantum), 4),
            round(float(track_wear), 3),
            str(contract_type) if contract_type is not None else None,
            supplier_name,
            round(float(supplier_pace_rating), 3),
            round(float(supplier_durability_rating), 3),
            round(float(grip_bonus_mult), 6),
            round(management, 4),
            weather_sig,
            temp_sig,
            int(weather_now_index),
        )

    def _lookup_stint_cache(self, driver_name: str, key: Tuple):
        bucket = self._stint_cache.get(driver_name)
        if not bucket:
            return None
        entry = bucket.get(key)
        if entry is not None:
            bucket.move_to_end(key)
        return entry

    def _store_stint_cache(self, driver_name: str, key: Tuple, value: Tuple[float, float]) -> None:
        bucket = self._stint_cache.setdefault(driver_name, OrderedDict())
        bucket[key] = value
        bucket.move_to_end(key)
        while len(bucket) > self.STINT_CACHE_LIMIT:
            bucket.popitem(last=False)

    def _invalidate_stint_cache(self, driver_name: Optional[str] = None) -> None:
        memo = getattr(self, "_projection_memo", None)
        if memo is not None:
            memo["prefixes"].clear()
            memo["prefix_states"] = 0
            memo["values"].clear()
            memo["numeric_contexts"].clear()
            memo["numeric_rows"] = 0
            memo.pop("physics", None)
            memo.pop("physics_plans", None)
            caches = getattr(memo.get("lookup_model"), "_lookup_caches", None)
            if caches is not None:
                for cache in caches.values():
                    cache.clear()
        if driver_name is None:
            self._stint_cache.clear()
            return
        bucket = self._stint_cache.get(driver_name)
        if bucket is not None:
            bucket.clear()
