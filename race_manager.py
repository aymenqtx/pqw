
import random
import json
import math
import time
import gc
import inspect
from array import array
from dataclasses import replace
from collections import defaultdict, deque

from car_ratings import rating_to_dirty_air, rating_to_reliability, rating_to_tyre_wear
from utils import data_path
from typing import Any, Dict, Optional, Tuple
from constants import (
    TYRE_MANAGEMENT_MIN,
    TYRE_MANAGEMENT_MAX,
    PART_WEAR_PER_LAP,
    PART_CRASH_DAMAGE,
    DRIVER_AID_WETNESS_THRESHOLD,
    FUEL_LAPTIME_PENALTY_PER_KG,
    GRID_START_SPACING_KM,
    RACE_START_LAUNCH_DURATION_S,
)

from .track import PolylineTrack
from .strategy import StrategyManager
from .formula_pit_service import chief_mechanic_pit_skill, roll_formula_pit_service, track_pit_lane_seconds
from .formula_refueling import normalize_service, target_for_laps, finite_fuel
from .driver_traits import apply_spin_prob, apply_dirty_air
from .realistic_physics import (
    LapInputs,
    RealisticPhysicsModel,
    ACCEL_MS2_PER_RATING,
    ATTR_KMH_PER_RATING,
    BASE_ACCEL_MS2,
    BASE_BRAKE_MS2,
    DELTA_SECONDS_PER_RATING,
    MASS_CACHE_BIN_KG,
    UNIVERSAL_KMH_PER_RATING,
    braking_zone_markers,
)
from .tyre_model import TyreModel
from .track_rubber import (
    RUBBER_RACE_BUILDUP_MULTIPLIER,
    begin_session_surface,
    commit_session_surface,
    deposit_completed_car_lap,
    get_or_seed_session_rubber_target,
    grip_gain_from_target_seconds,
    rubber_grip_multiplier,
    wash_rubber_for_weather,
)
from .driver_pace_mode_manager import (
    DriverPaceModeController,
    DriverEngineModeController,
    ENGINE_MODE_MEDIUM,
    ENGINE_MODE_ORDER,
    PACE_MODE_ORDER,
    PACE_MODE_PUSH,
)
from .driver_style import (
    BALANCE_MISMATCH_CORNER_LOSS_KMH_PER_POINT,
    TRACTION_MISMATCH_SPIN_RISK_PER_POINT,
    TRACTION_MISMATCH_TYRE_WEAR_PER_POINT,
    car_concept_feel,
    clamp_style_value,
    driver_style_mismatch_effects,
    fitted_chassis_concept_key,
)
from .ers_manager import (
    DriverErsModeController,
    ERS_BATTERY_CAPACITY_MJ,
    ERS_DEPLOY_CAP_MJ_PER_LAP,
    ERS_HARVEST_CAP_MJ_PER_LAP,
    ERS_MODE_BALANCED,
    ERS_MODE_CONSERVATIVE,
    ERS_MODE_NO_DEPLOY,
    ERS_MODE_ORDER,
    ERS_MODE_SPECS,
    apply_ers_progress_segment,
    build_ers_mode_energy_plan,
    build_track_ers_map,
    ers_deploy_windows_for_coverage,
    overlap_fraction,
    split_progress_segments,
)
from .driver import canonical_driver_status, canonical_driver_personalities
from .practice_acclimatization import driver_confidence_value
from .session_log_exporter import (
    capture_session_telemetry,
    clear_session_export,
    finalize_session_telemetry,
    synchronize_session_event_cursor,
)
try:
    from . import realistic_physics_kernels as _REALISTIC_KERNELS
except Exception:
    _REALISTIC_KERNELS = None
from utils import (
    normalize_mix_dict,
    normalize_sector_mixes,
    normalize_sector_splits as normalize_track_sector_splits,
    sector_lengths_from_splits,
)

EPSILON = 1e-6
MIN_VALID_LAP_TIME = 0.5
DEFERRED_EVENT_FLUSH_BUDGET_S = 0.0015
DEFERRED_MAX_EVENTS_PER_UPDATE = 96
DEFERRED_MAX_LAP_HISTORY_PER_UPDATE = 48
DEFERRED_MAX_PART_WEAR_PER_UPDATE = 96
MAX_RACE_EVENTS = 2000
SAFETY_CAR_QUEUE_SPACING_MULTIPLIER = 2.0
SAFETY_CAR_MIN_FOLLOW_GAP_M = 4.0
MOVE_OVER_HANDOVER_PROGRESS_MULTIPLIER = 0.74
MOVE_OVER_SETTLE_PROGRESS_MULTIPLIER = 0.90
MOVE_OVER_SETTLE_TARGET_GAP_S = 0.60
MOVE_OVER_SETTLE_MAX_DURATION_S = 5.0
MOVE_OVER_SETTLE_THREAT_GAP_S = 0.75
# Driver-error incidents remain possible during a neutralization, but at five
# percent of their otherwise fully calculated chance. This applies only while
# a physical or virtual safety car is active.
CAUTION_DRIVER_ERROR_CHANCE_MULTIPLIER = 0.05
REALISTIC_PREWARM_DRIVERS_PER_STEP = 0
REALISTIC_PREWARM_LEVELS = 0
GAP_HISTORY_MAX_SAMPLES = 180
DEFAULT_PART_CORNERING_MAX_LOSS_KMH = {
    "front_wing": 0.5,
    "rear_wing": 0.5,
    "underfloor": 0.5,
    "diffuser": 0.5,
    "sidepods": 0.5,
    "chassis": 2.5,
}
TYRE_TEMP_TARGET_DRY_OFFSET_C = 70.0
TYRE_TEMP_TARGET_WET_OFFSET_C = 60.0
TYRE_TEMP_WET_TARGET_THRESHOLD_MM = 1.0
TYRE_TEMP_STEP_C_PER_LAP = 5.0
TYRE_TEMP_MIN_C = 20.0
TYRE_TEMP_MAX_C = 180.0
DEFAULT_DIRTY_AIR_CURVE_KMH = (
    (0.0, 1.5),
    (0.3, 1.5),
    (0.4, 1.0),
    (0.6, 0.75),
    (0.8, 0.5),
    (1.0, 0.25),
    (2.0, 0.0),
)
DEFAULT_SLIPSTREAM_CURVE_KMH = (
    (0.0, 10.0),
    (0.4, 5.0),
    (0.6, 4.0),
    (0.8, 2.0),
    (1.0, 0.0),
    (2.0, 0.0),
)
DEFAULT_DRS_GAP_THRESHOLD_S = 1.0
DEFAULT_DRS_BOOST_KMH = 6.0
DEFAULT_DRS_ACTIVATION_LAP = 3
FRONT_WING_DAMAGE_CHANCE = 0.50
FRONT_WING_PIT_TIME_RANGE_S = (5.0, 10.0)
FRONT_WING_DAMAGE_LEVELS = {
    "minor": {"high": 5.0, "med": 3.75, "slow": 2.5, "stay_out_laps": 10},
    "moderate": {"high": 10.0, "med": 7.5, "slow": 5.0, "stay_out_laps": 5},
    "severe": {"high": 20.0, "med": 15.0, "slow": 10.0, "stay_out_laps": 3},
    "destroyed": {"high": 40.0, "med": 30.0, "slow": 20.0, "stay_out_laps": 2},
}
FRONT_WING_DAMAGE_ORDER = {"minor": 1, "moderate": 2, "severe": 3, "destroyed": 4}
OVAL_TYRE_KEYS = (
    "left_front",
    "right_front",
    "left_rear",
    "right_rear",
)

from .formula_compound_rule import FormulaCompoundRuleMixin, validate as validate_compound_rule


class RaceManager(FormulaCompoundRuleMixin):
    DEFAULT_WING_SETUP_CFG = {
        "min_level": 1,
        "max_level": 20,
        "neutral_level": 10,
        "front_per_click_kmh": {"slow": 0.22, "med": 0.12, "high": 0.03, "straight": -0.10},
        "rear_per_click_kmh": {"slow": 0.05, "med": 0.16, "high": 0.24, "straight": -0.20},
        "diminishing_returns": {
            "enabled": True,
            "coefficient": 0.025,
            "exponent": 1.35,
            "min_efficiency": 0.70,
            "max_efficiency": 1.0,
        },
    }
    DEFAULT_TYRE_PRESSURE_CFG = {
        "front": {
            "min_psi": 20.0,
            "max_psi": 28.0,
            "neutral_psi": 24.0,
            "step_psi": 0.1,
            "min_effects": {"slow": 1.0, "med": 1.0, "high": 0.0, "straight": 0.0, "accel_ms2": 0.0, "warmup_rate_delta": 0.2, "target_temp_delta_c": 5.0, "wear_rate_add": 0.001},
            "max_effects": {"slow": 0.0, "med": -1.0, "high": -1.0, "straight": 1.0, "accel_ms2": 0.0, "warmup_rate_delta": -0.2, "target_temp_delta_c": -5.0, "wear_rate_add": -0.001},
        },
        "rear": {
            "min_psi": 20.0,
            "max_psi": 25.0,
            "neutral_psi": 22.5,
            "step_psi": 0.1,
            "min_effects": {"slow": 0.0, "med": 0.0, "high": 0.0, "straight": 0.0, "accel_ms2": 0.03, "warmup_rate_delta": 0.2, "target_temp_delta_c": 5.0, "wear_rate_add": 0.001},
            "max_effects": {"slow": 0.0, "med": 0.0, "high": 0.0, "straight": 1.0, "accel_ms2": 0.0, "warmup_rate_delta": -0.2, "target_temp_delta_c": -5.0, "wear_rate_add": -0.001},
        },
    }
    DEFAULT_SUSPENSION_SETUP_CFG = {
        "min_level": 1,
        "max_level": 11,
        "neutral_level": 6,
        "min_effects": {"slow": 2.0, "med": -1.5, "high": -1.5, "straight": 0.0},
        "max_effects": {"slow": -2.0, "med": 1.5, "high": 1.5, "straight": 0.0},
    }

    def __init__(
        self,
        drivers,
        track_points,
        team_paces=None,
        grid_order=None,
        total_laps=12,
        form_scale=1.0,
        track_mix=None,
        track_wear_mult=None,
        track_dnf_mult=None,
        track_dirty_air_mult=None,
        team_attrs=None,
        team_meta=None,
        driver_attrs_by_driver=None,
        driver_dirty_air_sensitivity_by_driver=None,
        driver_tyre_management=None,
        engine_power_rating_by_driver=None,
        engine_mass_kg_by_driver=None,
        head_of_dynamics_skill=None,
        head_of_dynamics_tire_temp_skill=None,
        chief_mechanic_skill=None,
        tyre_suppliers=None,
        tyre_contract_types=None,
        tyre_contracts=None,
        fuel_loads=None,
        fuel_burn_per_lap=None,
        fuel_penalty_per_kg=None,
        email_manager: Optional[object] = None,
        temperature_profile=None,
        wetness_profile=None,
        player_team=None,
        player_auto_pit=True,
        part_reliability=None,
        chassis_reliability_by_driver=None,
        part_cornering_profile=None,
        part_wear_per_lap=None,
        supplier_part_specs=None,
        engine_unit_specs=None,
        event_name: Optional[str] = None,
        mechanical_override=None,
        driver_consistency=None,
        tyre_wear_bonus=None,
        crash_multiplier=None,
        allow_start_stalls: bool = True,
        sector_splits=None,
        sector_mixes=None,
        track_length_km=None,
        micro_sectors=None,
        weekend_cornering_variability_kmh_by_driver=None,
        driver_cornering_kmh_per_point=0.1,
        driver_braking_ms2_per_point=0.03,
        wing_setup_by_driver=None,
        tyre_pressure_setup_by_driver=None,
        suspension_setup_by_driver=None,
        driver_team_pace_rating_points_by_driver=None,
        driver_practice_comfort=None,
        drs_zones=None,
        regulation_drs_enabled=True,
        regulation_ers_enabled=True,
        regulation_safety_car_policy="physical_only",
        parts_inventory_manager=None,
        relationship_manager=None,
        game_state=None,
        physics_track_meta=None,
        headless_simulation: bool = False,
    ):
        # Drivers in grid order if provided
        self.drivers = list(drivers)
        if grid_order:
            name_to_driver = {d.name: d for d in self.drivers}
            ordered = [name_to_driver[n] for n in grid_order if n in name_to_driver]
            for d in self.drivers:
                if d not in ordered:
                    ordered.append(d)
            self.drivers = ordered
        else:
            random.shuffle(self.drivers)
        self.starting_grid = [str(getattr(driver, "name", "") or "") for driver in self.drivers]
        self.driver_by_name = {d.name: d for d in self.drivers}
        self.state = game_state
        self.weekend = (
            game_state.active_weekend_for_current_event()
            if game_state is not None and callable(getattr(game_state, "active_weekend_for_current_event", None))
            else None
        )
        self._session_export_recorder = None
        clear_session_export(self.weekend, "race")
        self.weekend_tyre_manager = getattr(game_state, "formula_weekend_tyre_manager", None) if game_state is not None else None
        self.active_tyre_set_id = {}
        self.active_tyre_set_start_lap = {}
        # Presentation-only stint telemetry for the Formula strategy graph.
        # Race physics and strategy decisions never read this structure.
        self.tyre_stint_history = {}
        self._formula_tyre_inventory_committed = False
        if self.weekend_tyre_manager is not None and self.weekend is not None:
            self.weekend_tyre_manager.ensure_weekend(self.weekend, self.drivers)
            self.weekend_tyre_manager.snapshot_session(self.weekend, "race")
        # ERS development is fitted per driver, so establish this mapping
        # before initial battery capacity is read below.
        self.driver_team = {d.name: getattr(d, "team", None) for d in self.drivers}
        self.headless_simulation = bool(headless_simulation)
        if self.weekend is not None:
            self.weekend.race_was_interactive = not self.headless_simulation
        self._race_update_max_step = 0.35
        self.parts_inventory_manager = parts_inventory_manager
        self.relationship_manager = relationship_manager
        self.collision_relationship_values = self._build_collision_relationship_values()
        self._driver_trait_flags = {}
        self._driver_dirty_air_trait_factor = {}
        self._build_driver_trait_runtime_cache()
        self._aero_kernel_supports_trait_factor = False
        if _REALISTIC_KERNELS is not None:
            try:
                _sig = inspect.signature(
                    _REALISTIC_KERNELS.compute_aero_effect_bases_from_progress
                )
                self._aero_kernel_supports_trait_factor = (
                    "dirty_trait_factors_obj" in _sig.parameters
                )
            except Exception:
                self._aero_kernel_supports_trait_factor = False

        # Track & basic state
        self.track = PolylineTrack(track_points)
        self.sector_splits = self._normalize_sector_splits(sector_splits)
        self._sector_splits_arr = array("d", [float(s) for s in self.sector_splits])
        self.sector_lengths = sector_lengths_from_splits(self.sector_splits)
        try:
            self.grid_start_spacing_km = max(0.0, float(GRID_START_SPACING_KM))
        except Exception:
            self.grid_start_spacing_km = 0.05
        self.progress = {d.name: 0.0 for d in self.drivers}
        self.live_speed_kmh = {d.name: 0.0 for d in self.drivers}
        self.laps = {d.name: 0 for d in self.drivers}
        self.laps_led = {d.name: 0 for d in self.drivers}
        self._laps_led_scored_through = 0
        self.grid_start_pending = {d.name: True for d in self.drivers}
        self.total_sim_time = defaultdict(float)
        self.distance_along_track_m = {d.name: 0.0 for d in self.drivers}
        self.distance_time_history = {
            d.name: deque(maxlen=GAP_HISTORY_MAX_SAMPLES) for d in self.drivers
        }
        self.finished = set()
        self.finish_time = {}
        self.last_line_crossing_time = {d.name: 0.0 for d in self.drivers}
        self.checkered_flag_time = None
        self.checkered_flag_winner = None
        self.events = []
        # Structured, RNG-free presentation alerts. Interactive race screens
        # consume these at simulation-chunk boundaries; headless races ignore
        # them completely.
        self.race_alerts = []
        self._race_alert_serial = 0
        self._race_incident_serial = 0
        self._active_race_incident_id = None
        # Structured presentation facts consumed only by interactive-session
        # UI. No quote selection or simulation RNG is used here.
        self.radio_events = []
        self._radio_event_sequence = 0
        self.tick_count = 0
        self._gap_snapshot_tick = None
        self._gap_snapshot_mode = None
        self._gap_snapshot = None

        def _normalize_reliability(value: float) -> float:
            try:
                out = float(value)
            except Exception:
                out = 1.0
            if out > 5.0:
                out = float(rating_to_reliability(out))
            return max(0.0, out)

        self.part_reliability_mult = {
            d.name: float((part_reliability or {}).get(d.name, 1.0))
            for d in self.drivers
        }
        self.chassis_reliability_mult = {
            d.name: max(
                0.0,
                _normalize_reliability((chassis_reliability_by_driver or {}).get(d.name, 1.0)),
            )
            for d in self.drivers
        }
        wear_src = part_wear_per_lap if isinstance(part_wear_per_lap, dict) else PART_WEAR_PER_LAP
        wear_map = {}
        if isinstance(wear_src, dict):
            for key, val in wear_src.items():
                try:
                    wear_map[str(key)] = max(0.0, float(val))
                except Exception:
                    continue
        if not wear_map:
            wear_map = {k: float(v) for k, v in PART_WEAR_PER_LAP.items()}
        self.part_wear_per_lap = wear_map
        self.part_wear_delta = {
            d.name: {key: 0.0 for key in self.part_wear_per_lap.keys()}
            for d in self.drivers
        }
        self._runtime_part_wear_delta = {
            d.name: {key: 0.0 for key in self.part_wear_per_lap.keys()}
            for d in self.drivers
        }
        self.part_cornering_profile = {}
        self.part_cornering_wear_nerf_kmh = {d.name: 0.0 for d in self.drivers}
        self._initialize_part_cornering_profile(part_cornering_profile)
        self.front_wing_damage = {d.name: None for d in self.drivers}
        self.pending_front_wing_change = {d.name: False for d in self.drivers}
        self.front_wing_damage_alerts = []
        self._front_wing_pit_extra_s = {d.name: 0.0 for d in self.drivers}
        self.supplier_part_specs = supplier_part_specs if isinstance(supplier_part_specs, dict) else {}
        self.supplier_part_wear_delta = {
            d.name: {key: 0.0 for key in (self.supplier_part_specs.get(d.name, {}) or {}).keys()}
            for d in self.drivers
        }
        self.engine_unit_specs = engine_unit_specs if isinstance(engine_unit_specs, dict) else {}
        self.engine_wear_delta = {d.name: 0.0 for d in self.drivers}
        self.dnf_reason = {}
        self.relationship_events = []
        self.fuel_penalty_per_kg = float(
            FUEL_LAPTIME_PENALTY_PER_KG if fuel_penalty_per_kg is None else fuel_penalty_per_kg
        )
        explicit_fuel_load_names = (
            {str(name) for name in fuel_loads.keys()}
            if isinstance(fuel_loads, dict)
            else set()
        )
        fuel_loads = fuel_loads or {}
        self._explicit_fuel_load_names = explicit_fuel_load_names
        self.fuel_onboard = {
            d.name: float(fuel_loads.get(d.name, 0.0)) for d in self.drivers
        }
        self.initial_fuel = dict(self.fuel_onboard)
        fuel_burn_per_lap = fuel_burn_per_lap or {}
        self.fuel_burn_per_lap = {
            d.name: float(fuel_burn_per_lap.get(d.name, 0.0)) for d in self.drivers
        }

        # Lap timing
        self.current_lap_start = {d.name: 0.0 for d in self.drivers}
        self.last_lap_time = {d.name: None for d in self.drivers}
        self.best_lap_time = {d.name: None for d in self.drivers}
        self.current_sector_index = {d.name: 0 for d in self.drivers}
        self.current_sector_start_time = {d.name: 0.0 for d in self.drivers}
        self.current_lap_sector_times = {d.name: [] for d in self.drivers}
        self.current_lap_peak_speed_kmh = {d.name: 0.0 for d in self.drivers}
        self.last_sector_times = {d.name: [] for d in self.drivers}
        self.speed_trap_best_kmh = {d.name: None for d in self.drivers}
        self.cornering_speed_best_kmh = {d.name: None for d in self.drivers}
        # Global fastest lap tracking
        self.fastest_lap_time = None
        self.fastest_lap_driver = None
        self.fastest_lap_lap = None

        from collections import defaultdict as _dd
        self.lap_history = _dd(list)
        self._deferred_events = deque()
        self._deferred_lap_history = deque()
        self._deferred_part_wear = deque()
        self._realistic_prewarm_cursor = 0

        src_weekend_corner_var = (
            weekend_cornering_variability_kmh_by_driver
            if isinstance(weekend_cornering_variability_kmh_by_driver, dict)
            else {}
        )
        self.weekend_cornering_variability_kmh_by_driver = {}
        for d in self.drivers:
            name = d.name
            try:
                self.weekend_cornering_variability_kmh_by_driver[name] = float(
                    src_weekend_corner_var.get(name, 0.0) or 0.0
                )
            except Exception:
                self.weekend_cornering_variability_kmh_by_driver[name] = 0.0

        # Options
        self.total_laps = total_laps
        self.form_bonus = {d.name: random.uniform(-0.01, 0.02) * form_scale for d in self.drivers}
        self.order = list(self.drivers)
        self.track_rubber = begin_session_surface(self.weekend, "race")
        self.track_rubber_target_gain_s = get_or_seed_session_rubber_target(
            self.weekend,
            "weekend",
        )
        self.track_rubber_grip_gain = grip_gain_from_target_seconds(
            self.track_rubber_target_gain_s
        )
        self.team_paces = team_paces or {}
        self.driver_team_pace_rating_points_by_driver = (
            driver_team_pace_rating_points_by_driver
            if isinstance(driver_team_pace_rating_points_by_driver, dict)
            else {}
        )
        self.track_mix = normalize_mix_dict(track_mix)
        self.sector_mixes = normalize_sector_mixes(
            sector_mixes,
            self.sector_splits,
            self.track_mix,
        )
        physics_payload = (
            dict(physics_track_meta)
            if isinstance(physics_track_meta, dict)
            else {}
        )
        physics_payload.setdefault(
            "length_km", track_length_km if track_length_km is not None else 0.0
        )
        physics_payload.setdefault(
            "micro_sectors", micro_sectors if micro_sectors is not None else []
        )
        physics_payload.setdefault("sector_splits", self.sector_splits)
        self._physics_series_mode = str(
            physics_payload.get("series_mode", "") or ""
        ).strip().lower()
        self._oval_track_type = str(
            physics_payload.get("oval_track_type", "") or ""
        ).strip().lower()
        formula_surface = physics_payload.get("racing_surface")
        formula_grid = physics_payload.get("starting_grid")
        # Keep the authored grid geometry available to the dedicated headless
        # race resolver.  Formula's lateral grid/racecraft layer remains
        # interactive-only, but simulated races still need the row spacing to
        # translate grid position into a realistic initial distance deficit.
        self.simulated_starting_grid_meta = (
            dict(formula_grid) if isinstance(formula_grid, dict) else {}
        )
        self.formula_multiline_grid_enabled = bool(
            self._physics_series_mode != "oval"
            and not self.headless_simulation
            and isinstance(formula_surface, dict)
            and bool(formula_surface.get("enabled", False))
            and str(formula_surface.get("mode") or "").strip().lower() == "road_course"
            and isinstance(formula_grid, dict)
        )
        self.formula_starting_grid = (
            dict(formula_grid)
            if self.formula_multiline_grid_enabled
            else {}
        )
        requested_start_procedure = str(
            getattr(game_state, "regulation_oval_start_procedure", "") or ""
        ).strip().lower()
        if getattr(self, "_physics_series_mode", "") == "oval":
            if requested_start_procedure not in {"double_file_pace_lap", "direct_start"}:
                requested_start_procedure = "double_file_pace_lap"
        else:
            requested_start_procedure = "direct_start"
        self.oval_start_procedure = requested_start_procedure
        self.oval_pace_lap_start_enabled = bool(
            self._physics_series_mode == "oval"
            and self.oval_start_procedure == "double_file_pace_lap"
        )
        physics_factory = getattr(self.state, "create_physics_model", None)
        if callable(physics_factory):
            self.realistic_physics = physics_factory(
                physics_payload,
                self.sector_splits,
            )
        else:
            self.realistic_physics = RealisticPhysicsModel(
                physics_payload.get("length_km", 0.0),
                physics_payload.get("micro_sectors", []),
                sector_splits=self.sector_splits,
            )
        if self.realistic_physics is None:
            self.realistic_physics = RealisticPhysicsModel(0.0, [])
        static_bin_path = physics_payload.get("static_bin_path")
        if static_bin_path and hasattr(self.realistic_physics, "load_static_track_bin"):
            try:
                self.realistic_physics.load_static_track_bin(static_bin_path)
            except Exception:
                pass
        if self.headless_simulation:
            try:
                self.realistic_physics.max_profile_cache_entries = max(
                    int(getattr(self.realistic_physics, "max_profile_cache_entries", 1024)),
                    4096,
                )
                self.realistic_physics.max_input_profile_identity_cache_entries = max(
                    int(getattr(self.realistic_physics, "max_input_profile_identity_cache_entries", 512)),
                    2048,
                )
                self.realistic_physics.mass_cache_bin_kg = max(
                    float(getattr(self.realistic_physics, "mass_cache_bin_kg", 10.0) or 10.0),
                    20.0,
                )
                self.realistic_physics.profile_delta_round_digits = 2
                self.realistic_physics.profile_engine_round_digits = 2
                self.realistic_physics.profile_driver_bonus_round_digits = 2
                self.realistic_physics.profile_tyre_mult_round_digits = 3
            except Exception:
                pass
        self.use_realistic_physics = bool(self.realistic_physics.enabled)
        self._realistic_static_input_cache = {}
        self._driver_style_mismatch_cache = {}
        if not self.use_realistic_physics:
            raise ValueError(
                "Realistic race physics requires track_length_km and micro_sectors."
            )
        self.seed_grid_progress(grid_order)
        self.realistic_base_lap_by_driver = {}
        self._realistic_lap_inputs_cache = {}
        self._realistic_step_base_lap_inputs_cache = None
        self._realistic_last_step_data = {}
        self._realistic_consistency_state = {}
        self._realistic_consistency_sensitivity = {}
        self._realistic_aero_state = {}
        self._realistic_last_local_rate = {}
        self._aero_slip_delta_cached_s = {d.name: 0.0 for d in self.drivers}
        self._aero_slip_delta_next_update_s = {d.name: 0.0 for d in self.drivers}
        self._aero_dirty_brake_cached_ms2 = {d.name: 0.0 for d in self.drivers}
        self._aero_dirty_brake_next_update_s = {d.name: 0.0 for d in self.drivers}
        self.drs_zones = self._normalize_drs_zones(drs_zones)
        self._drs_zone_detection_lap = {d.name: {} for d in self.drivers}
        self._drs_zone_eligible = {d.name: {} for d in self.drivers}
        self._drs_last_detection_crossing = {}
        self._drs_pending_detection_crossings = []
        self._drs_active_now = {d.name: False for d in self.drivers}
        self._drs_regulation_enabled = bool(regulation_drs_enabled)
        self._ers_regulation_enabled = bool(regulation_ers_enabled)
        self.ers_failed = {d.name: False for d in self.drivers}
        self.ers_failure_checked_lap = {d.name: 0 for d in self.drivers}
        self.ers_reliability_by_driver = {d.name: 50.0 for d in self.drivers}
        self.ers_failure_probability_by_driver = {d.name: 0.0 for d in self.drivers}
        self._ers_track_map = build_track_ers_map(micro_sectors if micro_sectors is not None else [])
        self._ers_deploy_windows = (self._ers_track_map or {}).get("deploy_windows", []) or []
        self._ers_extended_deploy_windows = (self._ers_track_map or {}).get("extended_deploy_windows", []) or self._ers_deploy_windows
        self._ers_harvest_windows = (self._ers_track_map or {}).get("harvest_windows", []) or []
        self._ers_mode_target_cache = {}
        self._ers_mode_coverage_cache = {}
        self._ers_deploy_window_cache = {}
        # Capacity and efficiency are immutable for the duration of a race.
        # Cache the fitted ERS snapshot so headless and interactive races avoid
        # repeated inventory resolution without changing sporting logic.
        self._session_ers_effect_cache = {}
        self.ers_capacity_mj = float(ERS_BATTERY_CAPACITY_MJ)
        self.ers_deploy_cap_mj = float(ERS_DEPLOY_CAP_MJ_PER_LAP)
        self.ers_harvest_cap_mj = float(ERS_HARVEST_CAP_MJ_PER_LAP)
        self.energy_store_by_driver = {d.name: float(self._ers_capacity_for_driver(d.name)) for d in self.drivers}
        self.lap_deploy_used_by_driver = {d.name: 0.0 for d in self.drivers}
        self.lap_harvest_used_by_driver = {d.name: 0.0 for d in self.drivers}
        try:
            self.track_wear_mult = float(track_wear_mult) if track_wear_mult is not None else 1.0
        except Exception:
            self.track_wear_mult = 1.0
        try:
            self.track_dnf_mult = float(track_dnf_mult) if track_dnf_mult is not None else 1.0
        except Exception:
            self.track_dnf_mult = 1.0
        if self.track_dnf_mult <= 0:
            self.track_dnf_mult = 1.0
        try:
            self.track_dirty_air_mult = float(track_dirty_air_mult) if track_dirty_air_mult is not None else 1.0
        except Exception:
            self.track_dirty_air_mult = 1.0
        if self.track_dirty_air_mult <= 0:
            self.track_dirty_air_mult = 1.0
        self.team_attrs = team_attrs or {}
        try:
            self.driver_cornering_kmh_per_point = max(0.0, float(driver_cornering_kmh_per_point))
        except Exception:
            self.driver_cornering_kmh_per_point = 0.1
        try:
            self.driver_braking_ms2_per_point = max(0.0, float(driver_braking_ms2_per_point))
        except Exception:
            self.driver_braking_ms2_per_point = 0.03
        self.engine_power_rating_by_driver = (
            engine_power_rating_by_driver if isinstance(engine_power_rating_by_driver, dict) else {}
        )
        self.engine_mass_kg_by_driver = (
            engine_mass_kg_by_driver if isinstance(engine_mass_kg_by_driver, dict) else {}
        )
        self.mass_ref_kg = float(getattr(self.state, "physics_mass_ref_kg", 800.0) or 800.0)
        hod_skill = head_of_dynamics_skill or {}
        hod_tire_temp_skill = (
            head_of_dynamics_tire_temp_skill
            if isinstance(head_of_dynamics_tire_temp_skill, dict)
            else {}
        )
        self.mechanic_skill = chief_mechanic_skill or {}
        self.formula_pit_skill = {
            team: chief_mechanic_pit_skill(self.state, team, self.mechanic_skill.get(team, 10))
            for team in {d.team for d in self.drivers}
        } if str(getattr(self.state, "game_mode", "formula")) != "oval" else {}
        self.tyre_contracts = tyre_contracts or {}

        # Load team meta (tyre_management, dirty_air_sensitivity) from data/teams.json
        def _clamp_mgmt(val: float) -> float:
            try:
                value = float(val)
            except Exception:
                value = 1.0
            if value > 5.0:
                value = float(rating_to_tyre_wear(value))
            return max(TYRE_MANAGEMENT_MIN, min(TYRE_MANAGEMENT_MAX, value))

        def _clamp_dirty(val: float) -> float:
            try:
                value = float(val)
            except Exception:
                value = 1.0
            if abs(value) > 5.0:
                value = float(rating_to_dirty_air(value))
            return max(0.1, value)

        self.team_meta = {}
        if isinstance(team_meta, dict) and team_meta:
            for team, meta in team_meta.items():
                if not isinstance(meta, dict):
                    meta = {}
                self.team_meta[team] = {
                    "tyre_management": _clamp_mgmt(meta.get("tyre_management", 1.0)),
                    "dirty_air_sensitivity": _clamp_dirty(meta.get("dirty_air_sensitivity", 1.0)),
                }
        else:
            try:
                with open(data_path("teams.json"), "r", encoding="utf-8") as _tf:
                    _teams = json.load(_tf)
                for _t in _teams:
                    self.team_meta[_t.get("name")] = {
                        "tyre_management": _clamp_mgmt(_t.get("tyre_management", 1.0)),
                        "dirty_air_sensitivity": _clamp_dirty(_t.get("dirty_air_sensitivity", 1.0)),
                    }
            except Exception:
                self.team_meta = {
                    t: {"tyre_management": 1.0, "dirty_air_sensitivity": 1.0}
                    for t in (self.team_paces or {}).keys()
                }

        # Apply Head of Dynamics impact on tyre management
        for team, skill in hod_skill.items():
            if skill <= 0:
                continue
            factor = 1.1 - ((skill - 1) / 19.0) * 0.2
            meta = self.team_meta.setdefault(team, {"tyre_management": 1.0, "dirty_air_sensitivity": 1.0})
            meta["tyre_management"] = _clamp_mgmt(meta.get("tyre_management", 1.0) * factor)

        for team, meta in self.team_meta.items():
            tyre_attr = float(self.team_attrs.get(team, {}).get("tyre_wear", 0.0))
            if tyre_attr:
                meta["tyre_management"] = _clamp_mgmt(
                    meta.get("tyre_management", 1.0) * max(0.1, 1.0 + tyre_attr)
                )

        self.move_over_orders = {}
        self.ai_move_over_pair_counts = {}
        self._ai_move_over_last_eval_key = None
        self.driver_attrs_by_driver = {}
        raw_driver_attrs = (
            driver_attrs_by_driver if isinstance(driver_attrs_by_driver, dict) else {}
        )
        for d in self.drivers:
            src = raw_driver_attrs.get(d.name)
            if isinstance(src, dict):
                self.driver_attrs_by_driver[d.name] = dict(src)
        self.driver_dirty_air_sensitivity = {}
        raw_driver_dirty = (
            driver_dirty_air_sensitivity_by_driver
            if isinstance(driver_dirty_air_sensitivity_by_driver, dict)
            else {}
        )
        for d in self.drivers:
            team = self.driver_team.get(d.name)
            fallback = self.team_meta.get(team, {}).get("dirty_air_sensitivity", 1.0)
            try:
                value = float(raw_driver_dirty.get(d.name, fallback))
            except Exception:
                value = float(fallback)
            self.driver_dirty_air_sensitivity[d.name] = _clamp_dirty(value)
        self.driver_tyre_management = {}
        raw_driver_tyre_management = (
            driver_tyre_management if isinstance(driver_tyre_management, dict) else {}
        )
        for d in self.drivers:
            team = self.driver_team.get(d.name)
            fallback = self.team_meta.get(team, {}).get("tyre_management", 1.0)
            try:
                raw_value = float(raw_driver_tyre_management.get(d.name, fallback))
            except Exception:
                raw_value = float(fallback)
            self.driver_tyre_management[d.name] = _clamp_mgmt(raw_value)
        raw_driver_practice_comfort = (
            driver_practice_comfort if isinstance(driver_practice_comfort, dict) else {}
        )
        self.driver_practice_comfort = {}
        for d in self.drivers:
            try:
                value = float(raw_driver_practice_comfort.get(d.name, 0.0) or 0.0)
            except Exception:
                value = 0.0
            self.driver_practice_comfort[d.name] = max(0.0, min(100.0, value))
        self.team_tire_temp_skill = {}
        team_names = set(self.team_meta.keys()) | set(self.driver_team.values())
        for team in team_names:
            if not team:
                continue
            raw_skill = hod_tire_temp_skill.get(team)
            if raw_skill is None:
                raw_skill = hod_skill.get(team, 1.0)
            try:
                skill_f = float(raw_skill)
            except Exception:
                skill_f = 1.0
            self.team_tire_temp_skill[team] = max(1.0, min(20.0, skill_f if skill_f > 0.0 else 1.0))
        self.email_manager = email_manager
        self.event_name = event_name
        self.tyre_suppliers = tyre_suppliers or {}
        self.tyre_contract_types = tyre_contract_types or {}
        self.tyre_supplier_by_driver = {
            d.name: self.tyre_suppliers.get(self.driver_team[d.name]) for d in self.drivers
        }
        self.tyre_contract_type_by_driver = {
            d.name: self.tyre_contract_types.get(self.driver_team[d.name]) for d in self.drivers
        }
        self.player_team = player_team
        self.player_auto_pit = bool(player_auto_pit)
        self.driver_pace_modes = DriverPaceModeController(
            self.drivers,
            player_team=self.player_team,
            player_default_mode="push",
            ai_default_mode="neutral",
        )
        self.driver_engine_modes = DriverEngineModeController(
            self.drivers,
            player_team=self.player_team,
            player_default_mode=ENGINE_MODE_MEDIUM,
            ai_default_mode=ENGINE_MODE_MEDIUM,
        )
        self.driver_ers_modes = DriverErsModeController(
            self.drivers,
            player_team=self.player_team,
            player_default_mode=ERS_MODE_BALANCED,
            ai_default_mode=ERS_MODE_BALANCED,
        )
        edm = self._ers_development_manager()
        for d in self.drivers:
            name = d.name
            team_name = self.driver_team.get(name)
            if not self._ers_regulation_enabled:
                continue
            try:
                if edm and hasattr(edm, "reliability_for_driver"):
                    self.ers_reliability_by_driver[name] = float(
                        edm.reliability_for_driver(name, team_name)
                    )
                if edm and hasattr(edm, "failure_probability_per_lap"):
                    self.ers_failure_probability_by_driver[name] = max(
                        0.0,
                        min(1.0, float(edm.failure_probability_per_lap(name, team_name))),
                    )
                else:
                    self.ers_failure_probability_by_driver[name] = 0.001
            except Exception:
                self.ers_reliability_by_driver[name] = 50.0
                self.ers_failure_probability_by_driver[name] = 0.001
        self._ai_race_sector_tokens = {d.name: 0 for d in self.drivers}
        self._ai_race_push_window = {d.name: 0 for d in self.drivers}
        self._ai_race_ers_attack_window = {d.name: 0 for d in self.drivers}
        for d in self.drivers:
            if self.driver_pace_modes.is_ai_controlled(d.name):
                self.driver_pace_modes.set_active_mode(d.name, PACE_MODE_PUSH)
                self._ai_race_push_window[d.name] = 2
        self.driver_consistency_override = driver_consistency or {}
        self.driver_aid_tyre_wear = tyre_wear_bonus or {}
        self.driver_aid_crash_multiplier = crash_multiplier or {}


        # Load modular config
        cfg_path = data_path("config.json")
        state_data_file = getattr(game_state, "data_file_path", None)
        if callable(state_data_file):
            try:
                cfg_path = state_data_file("config.json")
            except Exception:
                cfg_path = data_path("config.json")
        try:
            with open(cfg_path, "r", encoding="utf-8") as _f:
                self.cfg = json.load(_f)
        except Exception:
            self.cfg = {}
        if self._physics_series_mode != "oval":
            # This race-local config is also passed to the strategy manager.
            # Resolve once so actual stops and AI estimates use the same lane.
            self.cfg["pitstops"] = dict(self.cfg.get("pitstops") or {})
            self.cfg["pitstops"]["pit_lane_loss_s"] = track_pit_lane_seconds(physics_payload)
        policy = str(regulation_safety_car_policy or "physical_only").strip().lower()
        self.formula_refueling_allowed = self._physics_series_mode != "oval" and bool(getattr(game_state, "regulation_formula_refueling_allowed", False))
        capacity = getattr(game_state, "regulation_formula_fuel_capacity_kg", 80)
        self.formula_fuel_capacity_kg = float(capacity if capacity in (60, 80, 100) else 80)
        self.formula_fuel_overtaking_difficulty = finite_fuel(physics_payload.get("overtaking_difficulty"), 1.0)
        if getattr(self, "formula_refueling_allowed", False):
            for name in self.fuel_onboard:
                self.fuel_onboard[name] = min(self.formula_fuel_capacity_kg, max(0.0, self.fuel_onboard[name]))
            self.initial_fuel = dict(self.fuel_onboard)
        self._safety_car_policy = (
            policy
            if policy in {"none", "physical_only", "all_types"}
            else "physical_only"
        )
        self.oval_refueling_allowed = bool(
            self._physics_series_mode == "oval"
            and getattr(game_state, "regulation_oval_refueling_allowed", True)
        )
        raw_refuel_cfg = self.cfg.get("oval_refueling", {})
        self.oval_refueling_cfg = (
            dict(raw_refuel_cfg) if isinstance(raw_refuel_cfg, dict) else {}
        )
        try:
            self.oval_fuel_capacity_kg = max(
                1.0,
                float(self.oval_refueling_cfg.get("tank_capacity_kg", 100.0) or 100.0),
            )
        except Exception:
            self.oval_fuel_capacity_kg = 100.0
        try:
            self.oval_caution_fuel_burn_multiplier = max(
                0.0,
                min(
                    1.0,
                    float(
                        self.oval_refueling_cfg.get(
                            "caution_burn_multiplier", 0.45
                        )
                        or 0.45
                    ),
                ),
            )
        except Exception:
            self.oval_caution_fuel_burn_multiplier = 0.45
        try:
            self.oval_empty_fuel_speed_cap_kmh = max(
                20.0,
                float(
                    self.oval_refueling_cfg.get(
                        "empty_fuel_speed_cap_kmh", 80.0
                    )
                    or 80.0
                ),
            )
        except Exception:
            self.oval_empty_fuel_speed_cap_kmh = 80.0
        if self.oval_refueling_allowed:
            for name in tuple(self.fuel_onboard.keys()):
                if name not in self._explicit_fuel_load_names:
                    # Lightweight tests and secondary callers that predate
                    # refueling may omit fuel plans altogether. Treat omission
                    # as the same full-tank grid load used by the Oval session
                    # factories; an explicitly supplied zero remains an empty
                    # tank for diagnostics and gameplay.
                    self.fuel_onboard[name] = self.oval_fuel_capacity_kg
                else:
                    self.fuel_onboard[name] = max(
                        0.0,
                        min(
                            self.oval_fuel_capacity_kg,
                            float(self.fuel_onboard.get(name, 0.0) or 0.0),
                        ),
                    )
            self.initial_fuel = dict(self.fuel_onboard)

        raw_four_tyre_cfg = self.cfg.get("oval_four_tyre_model", {})
        self.oval_four_tyre_cfg = (
            dict(raw_four_tyre_cfg)
            if isinstance(raw_four_tyre_cfg, dict)
            else {}
        )
        self.oval_four_tyre_enabled = bool(
            self._physics_series_mode == "oval"
            and self.oval_four_tyre_cfg.get("enabled", True)
        )
        configured_loads = physics_payload.get("tyre_load_factors")
        if not isinstance(configured_loads, dict):
            defaults = self.oval_four_tyre_cfg.get("default_load_factors", {})
            configured_loads = defaults if isinstance(defaults, dict) else {}
        load_factors = {}
        for key in OVAL_TYRE_KEYS:
            try:
                load_factors[key] = max(
                    0.05, float(configured_loads.get(key, 1.0) or 1.0)
                )
            except Exception:
                load_factors[key] = 1.0
        load_mean = sum(load_factors.values()) / float(len(OVAL_TYRE_KEYS))
        if load_mean <= 1e-9:
            load_mean = 1.0
        self.oval_tyre_load_factors = {
            key: float(value) / load_mean for key, value in load_factors.items()
        }
        pace_lap_cfg = {}
        if self.oval_pace_lap_start_enabled:
            try:
                raw_pace_lap_cfg = (self.cfg.get("race_start", {}) or {}).get("pace_lap", {})
                if isinstance(raw_pace_lap_cfg, dict):
                    pace_lap_cfg = raw_pace_lap_cfg
            except Exception:
                pace_lap_cfg = {}
        try:
            self.oval_pace_laps = max(1, int(pace_lap_cfg.get("laps", 1) or 1))
        except Exception:
            self.oval_pace_laps = 1
        try:
            self.oval_pace_speed_kmh = max(
                40.0,
                min(130.0, float(pace_lap_cfg.get("speed_kmh", 80.4672) or 80.4672)),
            )
        except Exception:
            self.oval_pace_speed_kmh = 80.4672
        try:
            self.oval_pace_car_leader_gap_m = max(
                8.0,
                min(60.0, float(pace_lap_cfg.get("pace_car_leader_gap_m", 18.0) or 18.0)),
            )
        except Exception:
            self.oval_pace_car_leader_gap_m = 18.0
        try:
            self.oval_pace_row_gap_m = max(
                5.5,
                min(30.0, float(pace_lap_cfg.get("row_gap_m", 8.0) or 8.0)),
            )
        except Exception:
            self.oval_pace_row_gap_m = 8.0
        try:
            self.oval_release_acceleration_s = max(
                0.5,
                min(15.0, float(pace_lap_cfg.get("release_acceleration_s", 4.5) or 4.5)),
            )
        except Exception:
            self.oval_release_acceleration_s = 4.5
        try:
            self.oval_release_min_speed_factor = max(
                0.1,
                min(0.8, float(pace_lap_cfg.get("release_min_speed_factor", 0.32) or 0.32)),
            )
        except Exception:
            self.oval_release_min_speed_factor = 0.32
        self._oval_track_type_override = {}
        if (
            self._physics_series_mode == "oval"
            and self._oval_track_type == "superspeedway"
        ):
            try:
                track_overrides = self.cfg.get("track_type_overrides", {}) or {}
                superspeedway_override = track_overrides.get("superspeedway", {}) or {}
                if isinstance(superspeedway_override, dict):
                    self._oval_track_type_override = superspeedway_override
            except Exception:
                self._oval_track_type_override = {}
        self._load_wing_setup_config()
        self._load_tyre_pressure_config()
        self._load_suspension_setup_config()
        self._init_driver_wing_setup(wing_setup_by_driver)
        self._init_driver_tyre_pressure_setup(tyre_pressure_setup_by_driver)
        self._init_driver_suspension_setup(suspension_setup_by_driver)
        try:
            dirty_cfg = self.cfg.get("dirty_air", {}) or {}
            override_dirty_cfg = self._oval_track_type_override.get("dirty_air", {})
            if isinstance(override_dirty_cfg, dict) and override_dirty_cfg:
                dirty_cfg = override_dirty_cfg
        except Exception:
            dirty_cfg = {}
        try:
            slip_cfg = self.cfg.get("slipstream", {}) or {}
            override_slip_cfg = self._oval_track_type_override.get("slipstream", {})
            if isinstance(override_slip_cfg, dict) and override_slip_cfg:
                slip_cfg = override_slip_cfg
        except Exception:
            slip_cfg = {}
        try:
            drs_cfg = self.cfg.get("drs", {}) or {}
        except Exception:
            drs_cfg = {}
        q_raw = dirty_cfg.get("quantum_kmh", slip_cfg.get("quantum_kmh", 0.25))
        try:
            self._aero_quantum_kmh = max(0.01, float(q_raw))
        except Exception:
            self._aero_quantum_kmh = 0.25
        try:
            self._dirty_brake_quantum_ms2 = max(
                0.0005,
                float(dirty_cfg.get("brake_quantum_ms2", 0.01)),
            )
        except Exception:
            self._dirty_brake_quantum_ms2 = 0.01
        try:
            self._dirty_brake_deadband_ms2 = max(
                0.0,
                float(dirty_cfg.get("brake_deadband_ms2", 0.005)),
            )
        except Exception:
            self._dirty_brake_deadband_ms2 = 0.005
        try:
            self._dirty_brake_update_interval_s = max(
                0.0,
                float(dirty_cfg.get("brake_update_interval_s", 0.12)),
            )
        except Exception:
            self._dirty_brake_update_interval_s = 0.12
        try:
            self._slip_delta_quantum_s = max(
                0.001,
                float(
                    slip_cfg.get(
                        "delta_quantum_s",
                        0.01,
                    )
                ),
            )
        except Exception:
            self._slip_delta_quantum_s = 0.01
        try:
            self._slip_delta_deadband_s = max(
                0.0,
                float(
                    slip_cfg.get(
                        "delta_deadband_s",
                        0.005,
                    )
                ),
            )
        except Exception:
            self._slip_delta_deadband_s = 0.005
        try:
            self._slip_delta_update_interval_s = max(
                0.0,
                float(
                    slip_cfg.get(
                        "delta_update_interval_s",
                        0.10,
                    )
                ),
            )
        except Exception:
            self._slip_delta_update_interval_s = 0.10
        self._aero_use_exact_gap = bool(
            dirty_cfg.get("use_exact_gap", False) or slip_cfg.get("use_exact_gap", False)
        )
        self._dirty_air_curve_kmh = self._load_gap_curve_kmh(
            dirty_cfg.get("curve"),
            value_keys=("kmh", "corner_kmh", "value"),
            fallback=DEFAULT_DIRTY_AIR_CURVE_KMH,
        )
        self._dirty_air_brake_curve_ms2 = self._load_gap_curve_kmh(
            dirty_cfg.get("curve"),
            value_keys=("brake_ms2", "brake_nerf_ms2", "brake"),
            fallback=tuple((float(g), 0.0) for g, _ in self._dirty_air_curve_kmh),
        )
        self._dirty_air_enabled = bool(dirty_cfg.get("enabled", True))
        stack_cfg = dirty_cfg.get("stacking", {}) if isinstance(dirty_cfg, dict) else {}
        self._dirty_air_stacking_enabled = False
        self._dirty_air_stack_base_max_gap_s = 1.0
        self._dirty_air_stack_layer2_max_gap_s = 1.8
        self._dirty_air_stack_layer2_mult = 0.35
        self._dirty_air_stack_layer3_max_gap_s = 2.5
        self._dirty_air_stack_layer3_mult = 0.15
        if isinstance(stack_cfg, dict):
            self._dirty_air_stacking_enabled = bool(stack_cfg.get("enabled", False))
            try:
                self._dirty_air_stack_base_max_gap_s = max(
                    0.0,
                    float(stack_cfg.get("base_max_gap_s", self._dirty_air_stack_base_max_gap_s)),
                )
            except Exception:
                pass
            layers = stack_cfg.get("layers", [])
            if isinstance(layers, list):
                for layer in layers:
                    if not isinstance(layer, dict):
                        continue
                    try:
                        offset = int(layer.get("position_offset", layer.get("offset", 0)) or 0)
                    except Exception:
                        offset = 0
                    try:
                        max_gap = max(0.0, float(layer.get("max_gap", layer.get("max_gap_s", 0.0)) or 0.0))
                    except Exception:
                        max_gap = 0.0
                    try:
                        mult = max(0.0, float(layer.get("mult", layer.get("multiplier", 0.0)) or 0.0))
                    except Exception:
                        mult = 0.0
                    if offset == 2:
                        if max_gap > 0.0:
                            self._dirty_air_stack_layer2_max_gap_s = max_gap
                        self._dirty_air_stack_layer2_mult = mult
                    elif offset == 3:
                        if max_gap > 0.0:
                            self._dirty_air_stack_layer3_max_gap_s = max_gap
                        self._dirty_air_stack_layer3_mult = mult
        else:
            try:
                legacy_max_stack = int(dirty_cfg.get("max_stack", 1) or 1)
                legacy_decay = max(0.0, float(dirty_cfg.get("stack_decay", 0.0) or 0.0))
                if legacy_max_stack > 1 and legacy_decay > 0.0:
                    self._dirty_air_stacking_enabled = True
                    self._dirty_air_stack_layer2_mult = legacy_decay
                    self._dirty_air_stack_layer3_mult = legacy_decay * legacy_decay
            except Exception:
                pass
        self._slipstream_curve_kmh = self._load_gap_curve_kmh(
            slip_cfg.get("curve"),
            value_keys=("kmh", "straight_kmh", "value"),
            fallback=DEFAULT_SLIPSTREAM_CURVE_KMH,
        )
        self._slipstream_enabled = bool(slip_cfg.get("enabled", True))
        self._drs_cfg_enabled = bool(drs_cfg.get("enabled", True))
        try:
            self._drs_gap_threshold_s = max(
                0.0,
                float(drs_cfg.get("gap_threshold_s", DEFAULT_DRS_GAP_THRESHOLD_S)),
            )
        except Exception:
            self._drs_gap_threshold_s = DEFAULT_DRS_GAP_THRESHOLD_S
        try:
            self._drs_boost_kmh = max(
                0.0,
                float(drs_cfg.get("boost_kmh", DEFAULT_DRS_BOOST_KMH)),
            )
        except Exception:
            self._drs_boost_kmh = DEFAULT_DRS_BOOST_KMH
        try:
            self._drs_accel_mult_base = max(
                1.0,
                float(drs_cfg.get("accel_mult_base", 1.0)),
            )
        except Exception:
            self._drs_accel_mult_base = 1.0
        try:
            self._drs_accel_mult_per_kmh = max(
                0.0,
                float(drs_cfg.get("accel_mult_per_kmh", 0.003)),
            )
        except Exception:
            self._drs_accel_mult_per_kmh = 0.003
        try:
            self._drs_accel_mult_cap = max(
                1.0,
                float(drs_cfg.get("accel_mult_cap", 1.2)),
            )
        except Exception:
            self._drs_accel_mult_cap = 1.2
        try:
            self._drs_activation_lap = max(
                1,
                int(drs_cfg.get("activation_lap", DEFAULT_DRS_ACTIVATION_LAP)),
            )
        except Exception:
            self._drs_activation_lap = DEFAULT_DRS_ACTIVATION_LAP
        try:
            self._drs_wet_disable_threshold_mm = max(
                0.0,
                float(drs_cfg.get("wet_disable_threshold_mm", 1.0)),
            )
        except Exception:
            self._drs_wet_disable_threshold_mm = 1.0
        self._drs_enabled = bool(
            self._drs_cfg_enabled
            and self._drs_regulation_enabled
            and bool(self.drs_zones)
            and self._drs_gap_threshold_s > 0.0
            and (
                self._drs_boost_kmh > 0.0
                or self._drs_accel_mult_base > 1.0
                or self._drs_accel_mult_per_kmh > 0.0
            )
        )
        self._dirty_air_curve_gaps_arr = array(
            "d",
            [float(g) for g, _ in self._dirty_air_curve_kmh],
        )
        self._dirty_air_curve_vals_arr = array(
            "d",
            [float(v) for _, v in self._dirty_air_curve_kmh],
        )
        self._dirty_air_brake_curve_gaps_arr = array(
            "d",
            [float(g) for g, _ in self._dirty_air_brake_curve_ms2],
        )
        self._dirty_air_brake_curve_vals_arr = array(
            "d",
            [float(v) for _, v in self._dirty_air_brake_curve_ms2],
        )
        self._slipstream_curve_gaps_arr = array(
            "d",
            [float(g) for g, _ in self._slipstream_curve_kmh],
        )
        self._slipstream_curve_vals_arr = array(
            "d",
            [float(v) for _, v in self._slipstream_curve_kmh],
        )
        self._aero_max_gap_s = max(
            float(self._dirty_air_curve_gaps_arr[-1] if self._dirty_air_curve_gaps_arr else 0.0),
            float(self._slipstream_curve_gaps_arr[-1] if self._slipstream_curve_gaps_arr else 0.0),
        )
        self.tyre_model = TyreModel.load_for_state(
            game_state,
            fallback_path=data_path("tyremodel.json"),
        )

        # Race start stall delays (optional)
        self.start_delay_remaining = {d.name: 0.0 for d in self.drivers}
        self.start_stall_announced = set()
        if allow_start_stalls:
            stall_cfg = self.cfg.get("race_start", {})
            try:
                stall_chance = float(stall_cfg.get("stall_chance", 0.0) or 0.0)
            except Exception:
                stall_chance = 0.0
            delay_cfg = stall_cfg.get("stall_delay_s", [3.0, 4.0])
            lo = hi = 0.0
            if isinstance(delay_cfg, (list, tuple)) and len(delay_cfg) >= 2:
                try:
                    lo, hi = float(delay_cfg[0]), float(delay_cfg[1])
                except Exception:
                    lo = hi = 0.0
            else:
                try:
                    lo = hi = float(delay_cfg)
                except Exception:
                    lo = hi = 0.0
            if hi < lo:
                lo, hi = hi, lo
            lo = max(0.0, lo)
            hi = max(0.0, hi)
            if stall_chance > 0.0 and hi > 0.0:
                for d in self.drivers:
                    if random.random() < stall_chance:
                        self.start_delay_remaining[d.name] = random.uniform(lo, hi)
        else:
            stall_cfg = self.cfg.get("race_start", {})
        self.start_quality_score = {}
        self.start_rating = {}
        self.start_quality_breakdown = {}
        self.start_reaction_delay_remaining = {}
        self.launch_accel_mult = {}
        self._roll_race_start_qualities()
        self.launch_elapsed_s = {d.name: 0.0 for d in self.drivers}
        self.launch_duration_s = {}
        for d in self.drivers:
            self.launch_duration_s[d.name] = self._launch_duration_for_driver(
                d,
                stall_cfg,
                start_rating=self.start_rating.get(d.name),
            )
        self._oval_release_elapsed_s = {d.name: 0.0 for d in self.drivers}
        if self.oval_pace_lap_start_enabled:
            # Rolling starts cannot stall or wait for a standing-start reaction.
            # Start quality remains available to the legacy direct-start option.
            for d in self.drivers:
                self.start_delay_remaining[d.name] = 0.0
                self.start_reaction_delay_remaining[d.name] = 0.0
                self.launch_accel_mult[d.name] = 1.0

        incidents_cfg = self.cfg.setdefault("incidents", {})
        if mechanical_override is not None:
            try:
                incidents_cfg["mechanical_dnf_prob_per_lap"] = float(mechanical_override)
            except Exception:
                pass
        self._load_lockup_config(incidents_cfg)

        # Tyre state
        tyres_cfg = self.cfg.get("tyres", {})
        rule = getattr(self.state, "regulation_formula_compound_usage", None)
        if self._physics_series_mode != "oval":
            validate_compound_rule(self.tyre_model, rule)
        self.formula_compound_required_count = (2 if rule == "enabled" else 0 if rule == "disabled" else self.tyre_model.mandatory_compound_count)
        self.formula_compound_wet_exemption = False
        self.formula_compound_warned = set()
        self.formula_tyre_disqualifications = set()

        default_comp = getattr(self.tyre_model, "default_compound", "medium")
        self.tyre_comp = {d.name: default_comp for d in self.drivers}

        # Temperature configuration
        self.temperature_range = tuple(tyres_cfg.get("temperature_range", (10.0, 40.0)))
        effects_cfg = tyres_cfg.get("temperature_effects", {})
        cold_cfg = effects_cfg.get("cold", {}) if isinstance(effects_cfg, dict) else {}
        hot_cfg = effects_cfg.get("hot", {}) if isinstance(effects_cfg, dict) else {}
        self._temp_effects = {"cold": {}, "hot": {}}
        for band_name, cfg in (("cold", cold_cfg), ("hot", hot_cfg)):
            comp_map = cfg.get("compounds", {}) if isinstance(cfg, dict) else {}
            out = {}
            for comp, vals in comp_map.items():
                if not isinstance(vals, dict):
                    continue
                pace = float(vals.get("pace", 0.0))
                wear_mult = float(vals.get("wear_mult", 1.0))
                out[comp] = {"pace": pace, "wear": wear_mult}
            self._temp_effects[band_name] = out
        try:
            self._cold_threshold = float(cold_cfg.get("max_c", 20.0))
        except Exception:
            self._cold_threshold = 20.0
        try:
            self._hot_threshold = float(hot_cfg.get("min_c", 30.0))
        except Exception:
            self._hot_threshold = 30.0
        self.temperature_profile = []
        if temperature_profile:
            try:
                self.temperature_profile = [float(t) for t in temperature_profile]
            except Exception:
                self.temperature_profile = []
        if not self.temperature_profile:
            base = sum(self.temperature_range) / 2.0 if len(self.temperature_range) == 2 else 25.0
            self.temperature_profile = [base for _ in range(max(1, self.total_laps))]
        if len(self.temperature_profile) < self.total_laps:
            pad_val = self.temperature_profile[-1] if self.temperature_profile else 25.0
            self.temperature_profile.extend([pad_val] * (self.total_laps - len(self.temperature_profile)))
        elif len(self.temperature_profile) > self.total_laps:
            self.temperature_profile = self.temperature_profile[: self.total_laps]

        # Wetness configuration
        wet_range = tyres_cfg.get("wetness_range", (0.0, 5.0))
        try:
            low, high = float(wet_range[0]), float(wet_range[1])
            if high < low:
                low, high = high, low
            self.wetness_range = (low, high)
        except Exception:
            self.wetness_range = (0.0, 5.0)
        wet_cfg = tyres_cfg.get("wetness_effects", {}) if isinstance(tyres_cfg, dict) else {}
        self._wet_bands = []
        self._wet_effects = {}
        if isinstance(wet_cfg, dict):
            for band_name, band_cfg in wet_cfg.items():
                if not isinstance(band_cfg, dict):
                    continue
                comps = {}
                comp_map = band_cfg.get("compounds", {}) if isinstance(band_cfg, dict) else {}
                for comp, vals in comp_map.items():
                    if not isinstance(vals, dict):
                        continue
                    pace = float(vals.get("pace", 0.0))
                    wear = float(vals.get("wear_mult", 1.0))
                    comps[comp] = {"pace": pace, "wear": wear}
                try:
                    band_min = float(band_cfg.get("min_mm", self.wetness_range[0]))
                except Exception:
                    band_min = float(self.wetness_range[0])
                try:
                    band_max = float(band_cfg.get("max_mm", self.wetness_range[1]))
                except Exception:
                    band_max = float(self.wetness_range[1])
                if band_max < band_min:
                    band_min, band_max = band_max, band_min
                self._wet_bands.append({
                    "name": band_name,
                    "min": band_min,
                    "max": band_max,
                    "effects": comps,
                })
                self._wet_effects[band_name] = comps
        if not self._wet_bands:
            # Build band metadata from the physics tyre model when config tyres
            # are absent. This keeps weather-category logic available.
            band_map = {}
            for comp_name in getattr(self.tyre_model, "compound_names", lambda: [])():
                comp_data = self.tyre_model.get_compound_data(comp_name) or {}
                for state in list(comp_data.get("wetness_states", []) or []):
                    key = str(state.get("name", "state"))
                    entry = band_map.setdefault(
                        key,
                        {
                            "name": key,
                            "min": float(state.get("min_mm", self.wetness_range[0])),
                            "max": float(state.get("max_mm", self.wetness_range[1])),
                            "effects": {},
                        },
                    )
                    entry["min"] = min(entry["min"], float(state.get("min_mm", entry["min"])))
                    entry["max"] = max(entry["max"], float(state.get("max_mm", entry["max"])))
                    entry["effects"][comp_name] = {
                        "pace": 0.0,
                        "wear": float(state.get("wear_mult", 1.0)),
                    }
            self._wet_bands = list(band_map.values())
            self._wet_effects = {item["name"]: dict(item.get("effects", {})) for item in self._wet_bands}
        if not self._wet_bands:
            self._wet_bands.append({
                "name": "dry",
                "min": self.wetness_range[0],
                "max": self.wetness_range[1],
                "effects": {},
            })
        self._wet_bands.sort(key=lambda item: item.get("min", 0.0))

        self.wetness_profile = []
        if wetness_profile:
            try:
                raw = [float(w) for w in wetness_profile]
            except Exception:
                raw = []
            for val in raw:
                lo, hi = self.wetness_range
                self.wetness_profile.append(max(lo, min(hi, val)))
        if not self.wetness_profile:
            base_wet = self.wetness_range[0]
            self.wetness_profile = [base_wet for _ in range(max(1, self.total_laps))]
        if len(self.wetness_profile) < self.total_laps:
            pad = self.wetness_profile[-1] if self.wetness_profile else self.wetness_range[0]
            self.wetness_profile.extend([pad] * (self.total_laps - len(self.wetness_profile)))
        elif len(self.wetness_profile) > self.total_laps:
            self.wetness_profile = self.wetness_profile[: self.total_laps]

        self.rain_intensity_profile = []
        if self.weekend is not None:
            try:
                self.rain_intensity_profile = [
                    max(0.0, min(1.0, float(value)))
                    for value in list(getattr(self.weekend, "rain_intensity_profile", []) or [])
                ]
            except Exception:
                self.rain_intensity_profile = []
        if len(self.rain_intensity_profile) != len(self.wetness_profile):
            self.rain_intensity_profile = [
                max(0.0, min(1.0, float(value or 0.0) / 5.0))
                for value in self.wetness_profile
            ]

        self._last_weather_check_lap = -1
        self._last_logged_wet_band = self.wetness_band(self.current_wetness())
        self._drs_wet_disabled = False

        # Default available compound list for UI/strategy consumers
        self.available_compounds = list(self.tyre_model.compound_names()) or ["soft", "medium", "hard"]

        # Randomize starting compounds using the physics tyre model. Under a
        # Formula allocation rule, the choice is then resolved to an actual
        # available physical set from the shared weekend inventory.
        starting_set_wear = {}
        try:
            compounds = list(self.tyre_model.compound_names()) or ["soft", "medium", "hard"]
            self.available_compounds = list(compounds)

            start_wet = 0.0
            if self.wetness_profile:
                try:
                    start_wet = float(self.wetness_profile[0])
                except Exception:
                    start_wet = 0.0
            for _d in self.drivers:
                if (
                    self._physics_series_mode != "oval"
                    and self.weekend_tyre_manager is not None
                    and self.weekend_tyre_manager.enabled()
                    and self.weekend is not None
                ):
                    chosen_compound, chosen_set_id = self._formula_starting_tyre_choice(
                        _d.name,
                        start_wet,
                        compounds,
                    )
                    self.tyre_comp[_d.name] = chosen_compound
                    physical, _reason = self.weekend_tyre_manager.checkout(
                        self.weekend,
                        _d.name,
                        chosen_compound,
                        "race",
                        set_id=chosen_set_id,
                    )
                    if physical is None:
                        fallback_sets = []
                        for fallback in compounds:
                            candidate = self.weekend_tyre_manager.choose_set(
                                self.weekend,
                                _d.name,
                                fallback,
                            )
                            if candidate is not None:
                                fallback_sets.append((
                                    float(candidate.get("wear", 0.0) or 0.0),
                                    int(candidate.get("laps", 0) or 0),
                                    str(fallback),
                                    candidate,
                                ))
                        fallback_sets.sort(key=lambda row: (row[0], row[1], row[2]))
                        for _wear, _laps, fallback, candidate in fallback_sets:
                            physical, _reason = self.weekend_tyre_manager.checkout(
                                self.weekend,
                                _d.name,
                                fallback,
                                "race",
                                set_id=candidate.get("set_id"),
                            )
                            if physical is not None:
                                chosen_compound = fallback
                                break
                    if physical is not None:
                        self.tyre_comp[_d.name] = chosen_compound
                        self.active_tyre_set_id[_d.name] = physical.get("set_id")
                        self.active_tyre_set_start_lap[_d.name] = 0
                        starting_set_wear[_d.name] = float(physical.get("wear", 0.0) or 0.0)
                else:
                    self.tyre_comp[_d.name] = self.tyre_model.choose_start_compound(
                        start_wet,
                        random,
                    )
        except Exception:
            pass
        self._strategy_wear_rate_by_compound = {}
        self._build_strategy_wear_rate_cache()

        self.tyre_wear = defaultdict(float, starting_set_wear)
        if self._physics_series_mode != "oval":
            for driver in self.drivers:
                name = driver.name
                self._start_tyre_stint_history(
                    name,
                    start_lap=0.0,
                    compound=self.tyre_comp.get(name),
                    start_wear=float(self.tyre_wear.get(name, 0.0) or 0.0),
                )
        self.tyre_wear_by_corner = {
            d.name: {key: 0.0 for key in OVAL_TYRE_KEYS}
            for d in self.drivers
        }
        self.tyre_temp = {
            d.name: float(self.tyre_model.initial_temperature_c(self.tyre_comp.get(d.name), pit_out=False))
            for d in self.drivers
        }
        if self.use_realistic_physics:
            self._build_realistic_static_input_cache()
            self._rebuild_realistic_base_laps()
            self._initialize_realistic_consistency_execution()
        self._lap_start_fuel = {
            d.name: float(self.fuel_onboard.get(d.name, 0.0)) for d in self.drivers
        }
        self._lap_start_wear = {d.name: 0.0 for d in self.drivers}
        # Track compounds used and precompute pit windows (2 stops, all compounds)
        self.used_compounds = {d.name: {self.tyre_comp[d.name]} for d in self.drivers}
        third = max(2, self.total_laps // 3)
        var = max(1, self.total_laps // 10)
        self.pit_windows = {}
        for _d in self.drivers:
            p1 = max(1, third + random.randint(-var, var))
            p2_base = 2 * third + random.randint(-var, var)
            p2 = max(p1 + 2, p2_base)
            self.pit_windows[_d.name] = [p1, p2]

        # Subsystems. Oval supplies its own fuel/four-tyre strategy manager;
        # Formula continues to instantiate the existing compound planner.
        strategy_factory = getattr(self.state, "create_race_strategy_manager", None)
        if getattr(self, "formula_refueling_allowed", False):
            from .formula_refueling_strategy import FormulaRefuelingStrategy
            strategy_factory = FormulaRefuelingStrategy
        if callable(strategy_factory):
            try:
                self.strategy = strategy_factory(self.cfg)
            except Exception:
                self.strategy = StrategyManager(self.cfg)
        else:
            self.strategy = StrategyManager(self.cfg)

        # Pit-stop state
        self.pit_remaining = {d.name: 0.0 for d in self.drivers}
        self.pitted_last_lap = {d.name: False for d in self.drivers}
        self.pending_compound = {d.name: None for d in self.drivers}
        self.manual_pit_requests = {d.name: None for d in self.drivers}
        # Formula-only player requests from the race HUD's urgent Pit Now
        # control.  This tag is deliberately separate from ordinary manual
        # strategy requests so a one-off player command can be honoured even
        # when automatic pit strategy is enabled, without changing any other
        # player or AI pit decisions.
        self.formula_pit_now_requests = {d.name: None for d in self.drivers}
        self.pending_pit_service = {d.name: None for d in self.drivers}
        self.last_pit_service = {d.name: None for d in self.drivers}
        # Player-authored Oval stop plans live with the race runtime rather
        # than the UI so they survive closing the strategy panel and save/load.
        # Each item is {lap, tyres, fuel_mode, custom_target_kg}.
        self.oval_player_strategy_plans = {}
        self.formula_player_strategy_plans = {}
        self.formula_start_fuel_overrides = {}
        self.formula_fuel_stints = {}
        self.formula_fuel_exhausted_early = set()
        self._pit_flag_decay = {d.name: 0 for d in self.drivers}
        self.pit_count = {d.name: 0 for d in self.drivers}
        self.pit_stop_total_time = {d.name: 0.0 for d in self.drivers}
        self.pit_stop_elapsed = {d.name: 0.0 for d in self.drivers}
        self.pit_stop_last_time = {d.name: None for d in self.drivers}
        self.pit_stop_timer_serial = {d.name: 0 for d in self.drivers}
        self.pit_stop_completed_at_wall_time = {d.name: None for d in self.drivers}
        self._active_pit_stop_details = {}
        self.pit_service_last_time = {}
        self.pit_service_best_time = {}
        self.pit_service_completed_lap = {}
        self.pit_service_history = defaultdict(list)
        # A pit stop is simulated at the lap boundary rather than along a
        # spatial pit lane.  During a safety-car mass stop that gives several
        # cars the exact same race distance, so retain their real entry time
        # and scheduled exit time as the authoritative tie-break instead of
        # relying on the mutable race-order list.
        self._pit_entry_serial_counter = 0
        self._sc_pit_entry_time_s = {}
        self._sc_pit_exit_due_time_s = {}
        self._sc_pit_entry_serial = {}
        self._sc_pit_period_by_driver = {}

        # Seed AI strategy plans now that tyre state exists
        try:
            self.strategy.plan_initial_strategy(self)
        except Exception:
            pass

        # Incident & DNF state
        self.freeze_remaining = {d.name: 0.0 for d in self.drivers}
        self.incident_flag_decay = {d.name: 0 for d in self.drivers}
        self._pending_spin_marker = {}
        self._pending_spin_loss_s = {}
        self._pending_spin_lap = {}
        self.dnf = defaultdict(bool)
        self.dnf_laps = {}
        self.dnf_prog = {}

        # Safety car state
        self.sc_active = False
        self.vsc_active = False
        self.vsc_duration_s = 0.0
        self.vsc_remaining_s = 0.0
        self.vsc_elapsed_s = 0.0
        self._safety_car_period_serial = 0
        self.sc_phase = None
        self.sc_laps_remaining = 0
        self.sc_leader = None
        self.sc_train = []
        self.sc_progress = 0.0
        self.sc_distance_m = 0.0
        self.sc_live_speed_kmh = 0.0
        self._sc_pickup_line_distance_m = 0.0
        self._sc_pickup_offset_m = 0.0
        self._sc_pickup_candidate = None
        self._sc_collection_countdown_armed = False
        self.sc_collect_gap_s = max(
            0.2,
            self._safety_car_cfg_float("queue_gap_s", 0.3)
            * SAFETY_CAR_QUEUE_SPACING_MULTIPLIER,
        )
        self.sc_leader_gap_s = 1.0
        self._sc_clear_on_line = False
        self._sc_profile_inputs = LapInputs()
        self.sc_lap_active = {d.name: False for d in self.drivers}
        self._strategy_update_pending = False
        self._strategy_error_reports = set()
        self._strategy_last_sc_active = bool(self.caution_active)

        # A series may provide an optional lateral racecraft layer through its
        # game-state factory.  Formula only enables its model for explicitly
        # converted tracks in interactive races; Oval supplies its own model.
        # ``oval_racecraft`` remains as a compatibility alias for existing Oval
        # UI and tests while new code can use the series-neutral name.
        physics_payload["_interactive_race"] = not self.headless_simulation
        self.racecraft_model = None
        racecraft_factory = getattr(self.state, "create_racecraft_model", None)
        if callable(racecraft_factory):
            try:
                self.racecraft_model = racecraft_factory(
                    physics_payload,
                    self.track,
                    self.drivers,
                    self.cfg,
                )
            except Exception:
                self.racecraft_model = None
        self.oval_racecraft = self.racecraft_model
        self.oval_formation_active = False
        self.oval_formation_complete = not self.oval_pace_lap_start_enabled
        self._oval_formation_elapsed_s = 0.0
        self._oval_formation_distance_m_by_driver = {}
        self._oval_formation_target_sc_distance_m = 0.0
        if self.oval_pace_lap_start_enabled:
            self._initialize_oval_formation_start(self.starting_grid)

    def _formula_starting_tyre_choice(
        self,
        driver_name: str,
        start_wetness: float,
        compounds,
    ):
        """Choose a race-start compound and exact physical set together.

        Existing compound probabilities are retained among comparably worn
        sets. A heavily used set cannot win the random choice while a much
        healthier weather-appropriate alternative exists.
        """

        manager = self.weekend_tyre_manager
        if manager is None or self.weekend is None:
            return self.tyre_model.choose_start_compound(start_wetness, random), None

        try:
            desired_category = self.tyre_model.best_category_for_wetness(
                start_wetness
            )
        except Exception:
            desired_category = "dry"

        candidates = []
        for compound in list(compounds or []):
            try:
                category = self.tyre_model.compound_category(compound)
            except Exception:
                category = "dry"
            if category != desired_category:
                continue
            physical = manager.choose_set(self.weekend, driver_name, compound)
            if physical is None:
                continue
            wear = max(0.0, float(physical.get("wear", 0.0) or 0.0))
            candidates.append(
                {
                    "compound": str(compound),
                    "set": physical,
                    "wear": wear,
                }
            )

        if not candidates:
            # A category can be exhausted after an unusual weekend. Compare
            # every legal remaining set rather than using compound file order.
            for compound in list(compounds or []):
                physical = manager.choose_set(self.weekend, driver_name, compound)
                if physical is None:
                    continue
                candidates.append(
                    {
                        "compound": str(compound),
                        "set": physical,
                        "wear": max(
                            0.0,
                            float(physical.get("wear", 0.0) or 0.0),
                        ),
                    }
                )

        if not candidates:
            return self.tyre_model.choose_start_compound(start_wetness, random), None

        lowest_wear = min(float(row["wear"]) for row in candidates)
        wear_ceiling = min(0.25, lowest_wear + 0.10)
        eligible = [
            row for row in candidates if float(row["wear"]) <= wear_ceiling + 1e-12
        ]
        if not eligible:
            eligible = [
                min(
                    candidates,
                    key=lambda row: (
                        float(row["wear"]),
                        int(row["set"].get("laps", 0) or 0),
                        str(row["compound"]),
                    ),
                )
            ]

        chosen = None
        if desired_category == "dry":
            probabilities = dict(
                getattr(self.tyre_model, "start_probabilities", {}) or {}
            )
            weights = [
                max(0.0, float(probabilities.get(row["compound"], 0.0) or 0.0))
                for row in eligible
            ]
            total_weight = sum(weights)
            if total_weight > 1e-12:
                pick = random.random() * total_weight
                accumulated = 0.0
                for row, weight in zip(eligible, weights):
                    accumulated += weight
                    if pick <= accumulated:
                        chosen = row
                        break
                if chosen is None:
                    chosen = eligible[-1]

        if chosen is None:
            def _actual_grip_score(row):
                try:
                    lateral, longitudinal = self.tyre_model.grip_multipliers(
                        row["compound"],
                        wetness_mm=start_wetness,
                        wear=float(row["wear"]),
                    )
                    return (0.6 * float(lateral)) + (0.4 * float(longitudinal))
                except Exception:
                    return 0.0

            chosen = max(
                eligible,
                key=lambda row: (
                    _actual_grip_score(row),
                    -float(row["wear"]),
                    str(row["compound"]),
                ),
            )

        return str(chosen["compound"]), chosen["set"].get("set_id")

    def _consistency_rating_for_driver(self, driver) -> float:
        """Return effective 1..20 consistency for race execution modeling."""

        name = getattr(driver, "name", None)
        base = getattr(driver, "consistency", 10.0)
        try:
            if self.driver_consistency_override and name in self.driver_consistency_override:
                base = self.driver_consistency_override.get(name, base)
        except Exception:
            pass
        try:
            value = float(base)
        except Exception:
            value = 10.0
        return max(1.0, min(20.0, value))

    def _consistency_spread_seconds(self, consistency_rating: float) -> float:
        """Map consistency 20->~0.30s spread and 1->~2.00s spread."""

        c = max(1.0, min(20.0, float(consistency_rating)))
        # Linear mapping over inclusive [1, 20].
        return 0.30 + ((20.0 - c) / 19.0) * 1.70

    def _ensure_realistic_consistency_sensitivity(self, driver, neutral_inputs: LapInputs):
        """Estimate how cornering/braking nerfs translate into lap-time loss."""

        if not self.use_realistic_physics:
            return 0.5, 0.3
        name = getattr(driver, "name", None)
        if not name:
            return 0.5, 0.3
        cached = self._realistic_consistency_sensitivity.get(name)
        if isinstance(cached, tuple) and len(cached) == 2:
            return cached

        sec_per_kmh = 0.5
        sec_per_ms2 = 0.3
        try:
            base_lap, _ = self.realistic_physics.compute_lap(neutral_inputs)
            corner_step = 0.5
            brake_step = 0.25
            corner_inputs = replace(
                neutral_inputs,
                driver_cornering_kmh_bonus=float(neutral_inputs.driver_cornering_kmh_bonus) - corner_step,
            )
            brake_inputs = replace(
                neutral_inputs,
                driver_braking_ms2_bonus=float(neutral_inputs.driver_braking_ms2_bonus) - brake_step,
            )
            corner_lap, _ = self.realistic_physics.compute_lap(corner_inputs)
            brake_lap, _ = self.realistic_physics.compute_lap(brake_inputs)
            sec_per_kmh = max(0.05, float(corner_lap - base_lap) / corner_step)
            sec_per_ms2 = max(0.05, float(brake_lap - base_lap) / brake_step)
        except Exception:
            pass

        out = (float(sec_per_kmh), float(sec_per_ms2))
        self._realistic_consistency_sensitivity[name] = out
        return out

    def _roll_realistic_consistency_execution(self, driver, *, seed_only: bool = False) -> None:
        """Advance mean-reverting execution noise for realistic physics.

        Noise is negative-only and represented as cornering/braking nerfs so
        inconsistency degrades lap execution without direct time offsets.
        """

        if not self.use_realistic_physics:
            return
        name = getattr(driver, "name", None)
        if not name:
            return
        rating = self._consistency_rating_for_driver(driver)
        spread_s = self._consistency_spread_seconds(rating)
        c_norm = (rating - 1.0) / 19.0

        # Execution variance/mistakes are amplified by adverse conditions.
        wet_mult = 1.0
        wear_mult = 1.0
        traffic_mult = 1.0
        try:
            wet_mm = max(0.0, float(self.wetness_for_driver(name)))
        except Exception:
            wet_mm = 0.0
        wet_mult += 0.35 * min(1.0, wet_mm / 3.0)
        try:
            tyre_wear = max(0.0, float(self.tyre_wear.get(name, 0.0)))
        except Exception:
            tyre_wear = 0.0
        wear_mult += 0.25 * min(1.0, tyre_wear / 0.60)
        try:
            idx = self.order.index(driver)
        except Exception:
            idx = -1
        if idx > 0:
            ahead = self.order[idx - 1]
            try:
                _, gap_s = self.distance_reference_gap(name, ahead.name)
            except Exception:
                gap_s = None
            if gap_s is not None:
                traffic_mult += 0.25 * max(0.0, min(1.0, (2.5 - float(gap_s)) / 2.5))
        execution_mult = max(1.0, min(2.0, wet_mult * wear_mult * traffic_mult))
        spread_eff_s = spread_s * execution_mult

        state = self._realistic_consistency_state.get(name)
        if not isinstance(state, dict):
            state = {}
        prev_loss = float(state.get("loss_s", 0.0) or 0.0)
        prev_drift = float(state.get("drift_s", 0.0) or 0.0)

        if seed_only:
            loss_s = random.uniform(0.0, spread_eff_s * 0.20)
            drift_s = random.uniform(-0.05 * spread_eff_s, 0.05 * spread_eff_s)
            spike = False
        else:
            # Low-frequency drift plus mean-reverting lap execution noise.
            drift_amp = spread_eff_s * (0.05 + (1.0 - c_norm) * 0.10)
            drift_s = (prev_drift * 0.88) + random.uniform(-drift_amp, drift_amp)
            drift_s = max(-0.50 * spread_eff_s, min(0.75 * spread_eff_s, drift_s))

            reversion = 0.45 + 0.25 * c_norm
            sigma = max(0.12, spread_eff_s * (0.20 + (1.0 - c_norm) * 0.30))
            candidate = (reversion * prev_loss) + drift_s + random.gauss(0.0, sigma)
            loss_s = max(0.0, candidate)

            spike_prob = (0.03 + ((1.0 - c_norm) ** 2) * 0.10) * execution_mult
            spike_prob = max(0.0, min(0.35, spike_prob))
            spike = random.random() < spike_prob
            if spike:
                loss_s += random.uniform(0.25, 0.85) * spread_eff_s
            loss_s = min(spread_eff_s * 1.30, loss_s)

        corner_rating = driver.effective_rating("cornering") if hasattr(driver, "effective_rating") else getattr(driver, "cornering", 0.0)
        brake_rating = driver.effective_rating("braking") if hasattr(driver, "effective_rating") else getattr(driver, "braking", 0.0)
        base_corner_bonus = float(self.driver_cornering_kmh_per_point) * max(0.0, float(corner_rating or 0.0))
        base_brake_bonus = float(self.driver_braking_ms2_per_point) * max(0.0, float(brake_rating or 0.0))
        neutral_inputs = self._realistic_lap_inputs_for_driver(
            driver,
            apply_consistency=False,
            apply_aero=False,
        )
        sec_per_kmh, sec_per_ms2 = self._ensure_realistic_consistency_sensitivity(
            driver, neutral_inputs
        )

        # Allocate target loss across cornering and braking nerfs.
        corner_loss = loss_s * 0.62
        brake_loss = loss_s * 0.38
        corner_nerf = corner_loss / max(0.05, sec_per_kmh)
        brake_nerf = brake_loss / max(0.05, sec_per_ms2)

        # Allow inconsistent laps to drop below baseline (negative bonus).
        corner_cap = max(0.25, base_corner_bonus + 3.5)
        brake_cap = max(0.08, base_brake_bonus + 1.5)
        corner_nerf = max(0.0, min(corner_cap, corner_nerf))
        brake_nerf = max(0.0, min(brake_cap, brake_nerf))

        est_loss = (corner_nerf * sec_per_kmh) + (brake_nerf * sec_per_ms2)
        self._realistic_consistency_state[name] = {
            "loss_s": float(loss_s),
            "drift_s": float(drift_s),
            "spread_s": float(spread_s),
            "execution_mult": float(execution_mult),
            "corner_nerf_kmh": float(corner_nerf),
            "brake_nerf_ms2": float(brake_nerf),
            "estimated_loss_s": float(est_loss),
            "spike": bool(spike),
        }

    def _initialize_realistic_consistency_execution(self) -> None:
        if not self.use_realistic_physics:
            return
        self._realistic_consistency_state = {}
        self._realistic_consistency_sensitivity = {}
        for driver in self.drivers:
            try:
                self._roll_realistic_consistency_execution(driver, seed_only=True)
            except Exception:
                continue

    def _build_driver_trait_runtime_cache(self) -> None:
        """Precompute trait flags/factors used in hot realistic-race loops."""
        self._driver_trait_flags = {}
        self._driver_dirty_air_trait_factor = {}
        for driver in self.drivers:
            name = getattr(driver, "name", None)
            if not name:
                continue
            traits = set(getattr(driver, "traits", []) or [])
            self._driver_trait_flags[name] = {
                "clean_air_merchant": ("clean_air_merchant" in traits),
                "nervous": ("nervous" in traits),
                "rainmaster": ("rainmaster" in traits),
            }
            try:
                # Probe dirty-air trait function once: 2.0 => (1 + 1 * factor).
                dirty_factor = max(
                    0.0,
                    float(apply_dirty_air(driver, 2.0)) - 1.0,
                )
            except Exception:
                dirty_factor = 1.0
            self._driver_dirty_air_trait_factor[name] = max(0.0, min(3.0, float(dirty_factor)))

    @staticmethod
    def _safe_part_cornering_loss(raw_spec) -> float:
        if not isinstance(raw_spec, dict):
            return 0.0
        value = raw_spec.get("max_cornering_kmh_loss", raw_spec.get("max_pace_penalty", 0.0))
        try:
            return max(0.0, float(value or 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_condition_pct(value, default: float = 100.0) -> float:
        try:
            cond = float(value)
        except Exception:
            cond = float(default)
        return max(0.0, min(100.0, cond))

    def _initialize_part_cornering_profile(self, raw_profile) -> None:
        self.part_cornering_profile = {}
        for driver in self.drivers:
            name = driver.name
            src_by_driver = raw_profile.get(name, {}) if isinstance(raw_profile, dict) else {}
            dst = {}
            for part_key in self.part_wear_per_lap.keys():
                spec = src_by_driver.get(part_key, {}) if isinstance(src_by_driver, dict) else {}
                max_loss = self._safe_part_cornering_loss(spec)
                if max_loss <= 0.0:
                    max_loss = float(DEFAULT_PART_CORNERING_MAX_LOSS_KMH.get(str(part_key), 0.0))
                if max_loss <= 0.0:
                    continue
                condition = self._safe_condition_pct(
                    spec.get("condition", 100.0) if isinstance(spec, dict) else 100.0,
                    default=100.0,
                )
                dst[str(part_key)] = {
                    "condition": float(condition),
                    "max_cornering_kmh_loss": float(max_loss),
                }
            self.part_cornering_profile[name] = dst
            self._recompute_driver_part_cornering_nerf(name)

    def _recompute_driver_part_cornering_nerf(self, driver_name: str) -> None:
        profile = self.part_cornering_profile.get(driver_name, {})
        wear_map = self._runtime_part_wear_delta.get(driver_name, {})
        total_loss = 0.0
        for part_key, spec in (profile.items() if isinstance(profile, dict) else []):
            if not isinstance(spec, dict):
                continue
            start_cond = self._safe_condition_pct(spec.get("condition", 100.0), default=100.0)
            max_loss = self._safe_part_cornering_loss(spec)
            if max_loss <= 0.0:
                continue
            try:
                wear_delta = float((wear_map or {}).get(part_key, 0.0) or 0.0)
            except Exception:
                wear_delta = 0.0
            current_cond = max(0.0, start_cond - max(0.0, wear_delta))
            wear_frac = 1.0 - (current_cond / 100.0)
            total_loss += max(0.0, wear_frac) * max_loss
        previous = float(self.part_cornering_wear_nerf_kmh.get(driver_name, 0.0) or 0.0)
        total_loss = float(max(0.0, total_loss))
        self.part_cornering_wear_nerf_kmh[driver_name] = total_loss
        if abs(total_loss - previous) > 1e-9 and hasattr(self, "_realistic_lap_inputs_cache"):
            self._realistic_lap_inputs_cache.pop(driver_name, None)

    def _apply_runtime_part_wear_increment(self, driver_name: str, part_pairs) -> None:
        wear_map = self._runtime_part_wear_delta.setdefault(
            driver_name,
            {key: 0.0 for key in self.part_wear_per_lap.keys()},
        )
        changed = False
        for part_key, rate_per_lap in part_pairs:
            try:
                delta = float(rate_per_lap)
            except Exception:
                delta = 0.0
            if delta <= 0.0:
                continue
            key = str(part_key)
            wear_map[key] = float(wear_map.get(key, 0.0)) + delta
            changed = True
        if changed:
            self._recompute_driver_part_cornering_nerf(driver_name)

    def _invalidate_driver_lap_inputs(self, driver_name: str) -> None:
        try:
            self._realistic_lap_inputs_cache.pop(driver_name, None)
        except Exception:
            pass
        step_cache = getattr(self, "_realistic_step_base_lap_inputs_cache", None)
        if isinstance(step_cache, dict):
            for key in list(step_cache.keys()):
                try:
                    if key and key[0] == driver_name:
                        step_cache.pop(key, None)
                except Exception:
                    continue

    def _front_wing_damage_record(self, driver_name: str) -> Optional[dict]:
        rec = self.front_wing_damage.get(driver_name)
        return rec if isinstance(rec, dict) else None

    def front_wing_damage_level(self, driver_name: str) -> Optional[str]:
        rec = self._front_wing_damage_record(driver_name)
        if not rec:
            return None
        level = str(rec.get("level") or "").strip().lower()
        return level or None

    def front_wing_damage_losses(self, driver_name: str) -> dict:
        rec = self._front_wing_damage_record(driver_name)
        if not rec:
            return {}
        return {
            "slow": float(rec.get("slow", 0.0) or 0.0),
            "med": float(rec.get("med", 0.0) or 0.0),
            "high": float(rec.get("high", 0.0) or 0.0),
        }

    def _player_driver(self, driver_name: str) -> bool:
        return bool(self.player_team and self.driver_team.get(driver_name) == self.player_team)

    def _radio_sim_time(self) -> float:
        try:
            return max(
                [0.0]
                + [float(value or 0.0) for value in self.total_sim_time.values()]
            )
        except Exception:
            return 0.0

    def _emit_radio_event(self, category: str, driver_name: str, **context) -> None:
        """Publish a player-team presentation event without affecting race logic."""
        name = str(driver_name or "")
        if not category or not self._player_driver(name):
            return
        self._radio_event_sequence = int(getattr(self, "_radio_event_sequence", 0) or 0) + 1
        driver_id = ""
        for driver in self.drivers:
            if getattr(driver, "name", None) == name:
                driver_id = str(getattr(driver, "series_driver_id", "") or "")
                break
        self.radio_events.append(
            {
                "sequence": self._radio_event_sequence,
                "category": str(category),
                "driver": name,
                "driver_id": driver_id,
                "sim_time": self._radio_sim_time(),
                "context": dict(context or {}),
            }
        )
        if len(self.radio_events) > 100:
            del self.radio_events[:-100]

    def _emit_player_team_radio(self, category: str, **context) -> None:
        for driver in self.drivers:
            name = str(getattr(driver, "name", "") or "")
            if name and name not in self.finished and not self.dnf.get(name, False):
                self._emit_radio_event(category, name, **context)

    def _next_race_incident_id(self) -> int:
        self._race_incident_serial = int(getattr(self, "_race_incident_serial", 0) or 0) + 1
        return int(self._race_incident_serial)

    def _race_alert_lap(self, drivers=None) -> int:
        names = [str(name) for name in (drivers or []) if str(name or "")]
        laps = getattr(self, "laps", {}) or {}
        order = getattr(self, "order", []) or []
        completed = [int(laps.get(name, 0) or 0) for name in names]
        if not completed and order:
            completed = [int(laps.get(getattr(order[0], "name", ""), 0) or 0)]
        total_laps = max(1, int(getattr(self, "total_laps", 1) or 1))
        return max(1, min(total_laps, max(completed or [0]) or 1))

    def _emit_race_alert(
        self,
        alert_type: str,
        severity: str,
        summary: str,
        *,
        drivers=None,
        outcomes=None,
        incident_id=None,
        player_involved: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Emit or merge one structured interactive-race alert."""

        if bool(getattr(self, "headless_simulation", False)):
            return None
        if incident_id is None:
            incident_id = getattr(self, "_active_race_incident_id", None)
        if incident_id is None:
            incident_id = self._next_race_incident_id()
        alerts = getattr(self, "race_alerts", None)
        if not isinstance(alerts, list):
            alerts = []
            self.race_alerts = alerts
        names = []
        for name in drivers or []:
            name = str(name or "")
            if name and name not in names:
                names.append(name)
        if player_involved is None:
            player_involved = any(self._player_driver(name) for name in names)

        type_key = str(alert_type or "incident").strip().lower()
        severity_key = str(severity or "minor").strip().lower()
        title_map = {
            "safety_car": "SAFETY CAR DEPLOYED",
            "vsc": "VIRTUAL SAFETY CAR DEPLOYED",
            "crash": "CRASH",
            "collision": "COLLISION",
            "mechanical_retirement": "MECHANICAL RETIREMENT",
            "retirement": "RETIREMENT",
            "player_damage": "CAR DAMAGE",
            "contact": "CONTACT",
            "spin": "SPIN",
        }
        type_priority = {
            "spin": 1,
            "contact": 2,
            "retirement": 3,
            "mechanical_retirement": 4,
            "player_damage": 5,
            "collision": 6,
            "crash": 7,
            "vsc": 8,
            "safety_car": 9,
        }
        severity_priority = {"minor": 1, "major": 2, "critical": 3}
        existing = next(
            (
                alert
                for alert in reversed(alerts)
                if int(alert.get("incident_id", -1) or -1) == int(incident_id)
            ),
            None,
        )
        if existing is None:
            self._race_alert_serial = int(getattr(self, "_race_alert_serial", 0) or 0) + 1
            existing = {
                "id": int(self._race_alert_serial),
                "incident_id": int(incident_id),
                "type": type_key,
                "types": [type_key],
                "severity": severity_key,
                "title": title_map.get(type_key, "RACE INCIDENT"),
                "lap": self._race_alert_lap(names),
                "total_laps": int(getattr(self, "total_laps", 0) or 0),
                "drivers": [],
                "teams": {},
                "summary": str(summary or "Race incident"),
                "outcomes": [],
                "player_involved": bool(player_involved),
                "race_status": "Safety Car" if getattr(self, "sc_active", False) else "Virtual Safety Car" if getattr(self, "vsc_active", False) else "Green Flag",
            }
            alerts.append(existing)
        else:
            current_type = str(existing.get("type") or "incident")
            if type_priority.get(type_key, 0) > type_priority.get(current_type, 0):
                existing["type"] = type_key
                existing["title"] = title_map.get(type_key, "RACE INCIDENT")
            types = existing.setdefault("types", [])
            if type_key not in types:
                types.append(type_key)
            if severity_priority.get(severity_key, 0) > severity_priority.get(str(existing.get("severity") or "minor"), 0):
                existing["severity"] = severity_key
            existing["player_involved"] = bool(existing.get("player_involved", False) or player_involved)
            existing["race_status"] = "Safety Car" if getattr(self, "sc_active", False) else "Virtual Safety Car" if getattr(self, "vsc_active", False) else existing.get("race_status", "Green Flag")

        for name in names:
            if name not in existing["drivers"]:
                existing["drivers"].append(name)
            team = (getattr(self, "driver_team", {}) or {}).get(name)
            if team:
                existing.setdefault("teams", {})[name] = str(team)
        for outcome in outcomes or []:
            outcome = str(outcome or "").strip()
            if outcome and outcome not in existing["outcomes"]:
                existing["outcomes"].append(outcome)
        if type_key in {"safety_car", "vsc"}:
            caution_outcome = "Safety Car deployed" if type_key == "safety_car" else "Virtual Safety Car deployed"
            if caution_outcome not in existing["outcomes"]:
                existing["outcomes"].append(caution_outcome)
        return existing

    def _emit_overtake_radio(self, overtaker: str, passed_driver: str) -> None:
        self._emit_radio_event("race_overtake", overtaker, opponent=str(passed_driver or ""))
        self._emit_radio_event("race_overtaken", passed_driver, opponent=str(overtaker or ""))

    def driver_front_wing_change_available(self, driver_name: str) -> bool:
        pim = getattr(self, "parts_inventory_manager", None)
        team = self.driver_team.get(driver_name)
        if not pim or not team:
            return False
        try:
            return bool(pim.has_matching_usable_part_in_stock(team, driver_name, "front_wing"))
        except Exception:
            return False

    def request_front_wing_change(self, driver_name: str, enabled: bool = True) -> bool:
        if driver_name not in self.pending_front_wing_change:
            return False
        if not enabled:
            self.pending_front_wing_change[driver_name] = False
            return True
        if not self.front_wing_damage_level(driver_name):
            return False
        if not self.driver_front_wing_change_available(driver_name):
            return False
        self.pending_front_wing_change[driver_name] = True
        return True

    def consume_front_wing_damage_alerts(self) -> list[dict]:
        alerts = list(getattr(self, "front_wing_damage_alerts", []) or [])
        self.front_wing_damage_alerts = []
        return alerts

    def _roll_front_wing_damage(self, driver_name: str, source: str = "contact") -> Optional[str]:
        if str(getattr(self.game_state, "game_mode", "formula") or "formula").lower() == "oval":
            return None
        if driver_name in self.finished or self.dnf.get(driver_name, False):
            return None
        if random.random() >= FRONT_WING_DAMAGE_CHANCE:
            return None
        level = random.choice(list(FRONT_WING_DAMAGE_LEVELS.keys()))
        current = self.front_wing_damage_level(driver_name)
        if current and FRONT_WING_DAMAGE_ORDER.get(current, 0) >= FRONT_WING_DAMAGE_ORDER.get(level, 0):
            return current
        spec = FRONT_WING_DAMAGE_LEVELS.get(level, FRONT_WING_DAMAGE_LEVELS["minor"])
        self.front_wing_damage[driver_name] = {
            "level": level,
            "slow": float(spec.get("slow", 0.0)),
            "med": float(spec.get("med", 0.0)),
            "high": float(spec.get("high", 0.0)),
            "source": str(source or "contact"),
        }
        try:
            parts = self.part_wear_delta.setdefault(
                driver_name,
                {key: 0.0 for key in self.part_wear_per_lap.keys()},
            )
            parts["front_wing"] = max(float(parts.get("front_wing", 0.0) or 0.0), 100.0)
        except Exception:
            pass
        self.pending_front_wing_change[driver_name] = False
        self._invalidate_driver_lap_inputs(driver_name)
        team = self.driver_team.get(driver_name)
        pim = getattr(self, "parts_inventory_manager", None)
        if pim and team:
            try:
                pim.mark_installed_part_destroyed(team, driver_name, "front_wing")
            except Exception:
                pass
        msg = f"FRONT WING DAMAGE: {driver_name} has {level} front wing damage"
        self.events.append(msg)
        damage_rank = int(FRONT_WING_DAMAGE_ORDER.get(level, 0) or 0)
        player_damage = bool(self._player_driver(driver_name))
        self._emit_race_alert(
            "player_damage",
            "major" if player_damage and damage_rank >= FRONT_WING_DAMAGE_ORDER.get("moderate", 2) else "minor",
            f"{driver_name} has sustained front-wing damage",
            drivers=[driver_name],
            outcomes=[f"Damage level: {str(level).title()}"],
            player_involved=player_damage,
        )
        self._emit_radio_event(
            "race_wing_damage",
            driver_name,
            damage_level=str(level),
        )
        if self._player_driver(driver_name):
            self.front_wing_damage_alerts.append(
                {
                    "driver": driver_name,
                    "level": level,
                    "message": (
                        f"Oh no! {driver_name} has {level} wing damage! "
                        "Consider toggling a wing change within the car setup UI, "
                        "and moving up the next pitstop."
                    ),
                }
            )
        return level

    def _replace_front_wing_at_pit_stop(self, driver_name: str) -> bool:
        if not self.pending_front_wing_change.get(driver_name, False):
            return False
        team = self.driver_team.get(driver_name)
        pim = getattr(self, "parts_inventory_manager", None)
        if not pim or not team:
            self.pending_front_wing_change[driver_name] = False
            return False
        try:
            ok, _msg = pim.replace_installed_part_from_matching_stock(team, driver_name, "front_wing")
        except Exception:
            ok = False
        if not ok:
            self.events.append(f"FRONT WING: {driver_name} could not change front wing - no spare available")
            self.pending_front_wing_change[driver_name] = False
            return False
        try:
            self.part_wear_delta.setdefault(driver_name, {})["front_wing"] = 0.0
            self._runtime_part_wear_delta.setdefault(driver_name, {})["front_wing"] = 0.0
            parts = pim.get_driver_parts(team, driver_name)
            wing = parts.get("front_wing") if isinstance(parts, dict) else None
            if isinstance(wing, dict):
                profile = self.part_cornering_profile.setdefault(driver_name, {})
                spec = profile.setdefault("front_wing", {})
                spec["condition"] = self._safe_condition_pct(wing.get("condition", 100.0), default=100.0)
                spec["max_cornering_kmh_loss"] = self._safe_part_cornering_loss(spec)
                if spec["max_cornering_kmh_loss"] <= 0.0:
                    spec["max_cornering_kmh_loss"] = float(DEFAULT_PART_CORNERING_MAX_LOSS_KMH.get("front_wing", 0.5))
            self._recompute_driver_part_cornering_nerf(driver_name)
        except Exception:
            pass
        self.front_wing_damage[driver_name] = None
        self.pending_front_wing_change[driver_name] = False
        self._front_wing_pit_extra_s[driver_name] = 0.0
        self._invalidate_driver_lap_inputs(driver_name)
        self.events.append(f"FRONT WING: {driver_name} changed front wing")
        return True

    def _should_ai_pit_for_front_wing(self, driver_name: str) -> bool:
        level = self.front_wing_damage_level(driver_name)
        if not level or self.pending_front_wing_change.get(driver_name, False):
            return False
        if self.pit_remaining.get(driver_name, 0.0) > 0.0:
            return False
        if not self.driver_front_wing_change_available(driver_name):
            return False
        if self.caution_active:
            return True
        try:
            remaining = max(0, int(self.total_laps) - int(self.laps.get(driver_name, 0) or 0))
        except Exception:
            remaining = 999
        threshold = int(FRONT_WING_DAMAGE_LEVELS.get(level, {}).get("stay_out_laps", 0) or 0)
        return remaining > threshold

    def _ai_front_wing_pit_compound(self, driver_name: str):
        try:
            current = self.tyre_comp.get(driver_name, self.tyre_model.default_compound)
        except Exception:
            current = getattr(self.tyre_model, "default_compound", "medium")
        return current

    def _defer_event(self, text: str) -> None:
        if not text:
            return
        try:
            self._deferred_events.append(str(text))
        except Exception:
            pass

    def _defer_lap_history_entry(
        self,
        driver_name: str,
        lap_no: int,
        lap_time: float,
        sector_times,
        compound: str,
        wear_before: float,
        fuel_before: float,
        top_speed_kmh=None,
        cornering_speed_kmh=None,
    ) -> None:
        try:
            sectors = [float(x) for x in list(sector_times or [])]
            payload = (
                str(driver_name),
                int(lap_no),
                float(lap_time),
                float(sectors[0]) if len(sectors) >= 1 else None,
                float(sectors[1]) if len(sectors) >= 2 else None,
                float(sectors[2]) if len(sectors) >= 3 else None,
                sectors,
                compound,
                float(wear_before),
                float(fuel_before),
                (max(0.0, float(top_speed_kmh)) if top_speed_kmh is not None else None),
                (
                    max(0.0, float(cornering_speed_kmh))
                    if cornering_speed_kmh is not None
                    else None
                ),
            )
            self._deferred_lap_history.append(payload)
        except Exception:
            pass

    def _defer_part_wear_increment(self, driver_name: str) -> None:
        try:
            part_pairs = tuple(self.part_wear_per_lap.items())
            self._apply_runtime_part_wear_increment(driver_name, part_pairs)
            payload = (
                str(driver_name),
                part_pairs,
            )
            self._deferred_part_wear.append(payload)
        except Exception:
            pass

    def _flush_deferred_nonphysics(self, *, force: bool = False) -> None:
        if (
            not self._deferred_events
            and not self._deferred_lap_history
            and not self._deferred_part_wear
        ):
            return
        if force:
            max_events = len(self._deferred_events)
            max_hist = len(self._deferred_lap_history)
            max_part_wear = len(self._deferred_part_wear)
            deadline = None
        else:
            max_events = int(DEFERRED_MAX_EVENTS_PER_UPDATE)
            max_hist = int(DEFERRED_MAX_LAP_HISTORY_PER_UPDATE)
            max_part_wear = int(DEFERRED_MAX_PART_WEAR_PER_UPDATE)
            deadline = time.perf_counter() + float(DEFERRED_EVENT_FLUSH_BUDGET_S)

        while self._deferred_events and max_events > 0:
            if deadline is not None and time.perf_counter() > deadline:
                break
            try:
                self.events.append(self._deferred_events.popleft())
            except Exception:
                break
            max_events -= 1

        while self._deferred_lap_history and max_hist > 0:
            if deadline is not None and time.perf_counter() > deadline:
                break
            try:
                (
                    name,
                    lap_no,
                    lap_time,
                    s1,
                    s2,
                    s3,
                    sectors,
                    compound,
                    wear,
                    fuel,
                    top_speed_kmh,
                    cornering_speed_kmh,
                ) = self._deferred_lap_history.popleft()
                self.lap_history[name].append(
                    {
                        "lap": int(lap_no),
                        "time": float(lap_time),
                        "s1": s1,
                        "s2": s2,
                        "s3": s3,
                        "sectors": sectors,
                        "compound": compound,
                        "wear": float(wear),
                        "fuel": float(fuel),
                        "top_speed_kmh": (
                            max(0.0, float(top_speed_kmh))
                            if top_speed_kmh is not None
                            else None
                        ),
                        "cornering_speed_kmh": (
                            max(0.0, float(cornering_speed_kmh))
                            if cornering_speed_kmh is not None
                            else None
                        ),
                    }
                )
            except Exception:
                break
            max_hist -= 1

        while self._deferred_part_wear and max_part_wear > 0:
            if deadline is not None and time.perf_counter() > deadline:
                break
            try:
                name, part_pairs = self._deferred_part_wear.popleft()
                parts_map = self.part_wear_delta.setdefault(
                    name,
                    {key: 0.0 for key in self.part_wear_per_lap.keys()},
                )
                for part_key, rate_per_lap in part_pairs:
                    parts_map[part_key] = parts_map.get(part_key, 0.0) + float(rate_per_lap)
            except Exception:
                break
            max_part_wear -= 1

    def _cap_event_log(self) -> None:
        max_events = int(MAX_RACE_EVENTS)
        if max_events <= 0:
            return
        overflow = len(self.events) - max_events
        if overflow <= 0:
            return
        try:
            del self.events[:overflow]
        except Exception:
            pass

    def _record_strategy_error(self, driver_name: str, exc: Exception) -> None:
        """Report each distinct per-driver strategy failure once per race."""
        key = (str(driver_name), type(exc).__name__, str(exc))
        reports = getattr(self, "_strategy_error_reports", None)
        if not isinstance(reports, set):
            reports = set()
            self._strategy_error_reports = reports
        if key in reports:
            return
        reports.add(key)
        self.events.append(
            f"STRATEGY ERROR: {driver_name} ({type(exc).__name__}: {exc})"
        )

    @staticmethod
    def _normalize_oval_tyre_service(value) -> str:
        key = str(value or "four").strip().lower()
        aliases = {
            "2_left": "two_left",
            "two left": "two_left",
            "left": "two_left",
            "2_right": "two_right",
            "two right": "two_right",
            "right": "two_right",
            "both": "four",
            "four_tyres": "four",
            "four_tires": "four",
        }
        key = aliases.get(key, key)
        return key if key in {"two_left", "two_right", "four"} else "four"

    def driver_tyre_wear_by_corner(self, driver_name: str):
        name = str(driver_name or "")
        values = self.tyre_wear_by_corner.get(name)
        if not self.oval_four_tyre_enabled or not isinstance(values, dict):
            scalar = max(0.0, float(self.tyre_wear.get(name, 0.0) or 0.0))
            return {key: scalar for key in OVAL_TYRE_KEYS}
        return {
            key: max(0.0, float(values.get(key, 0.0) or 0.0))
            for key in OVAL_TYRE_KEYS
        }

    def _sync_oval_aggregate_tyre_wear(self, driver_name: str) -> float:
        name = str(driver_name or "")
        values = self.driver_tyre_wear_by_corner(name)
        average = sum(values.values()) / float(len(OVAL_TYRE_KEYS))
        self.tyre_wear[name] = max(0.0, float(average))
        self._realistic_lap_inputs_cache.pop(name, None)
        return float(self.tyre_wear[name])

    def _oval_effective_tyre_wear(self, driver_name: str):
        values = self.driver_tyre_wear_by_corner(driver_name)
        raw_weights = self.oval_four_tyre_cfg.get("grip_wear_weights", {})
        weights = raw_weights if isinstance(raw_weights, dict) else {}

        def _weighted(group: str, defaults):
            raw = weights.get(group, {})
            configured = raw if isinstance(raw, dict) else {}
            local = {}
            for key in OVAL_TYRE_KEYS:
                try:
                    local[key] = max(
                        0.0, float(configured.get(key, defaults[key]) or 0.0)
                    )
                except Exception:
                    local[key] = float(defaults[key])
            total = sum(local.values())
            if total <= 1e-9:
                total = 1.0
            return sum(values[key] * local[key] for key in OVAL_TYRE_KEYS) / total

        lat = _weighted(
            "lateral",
            {
                "left_front": 0.40,
                "right_front": 0.40,
                "left_rear": 0.10,
                "right_rear": 0.10,
            },
        )
        longi = _weighted(
            "longitudinal",
            {
                "left_front": 0.15,
                "right_front": 0.15,
                "left_rear": 0.35,
                "right_rear": 0.35,
            },
        )
        return max(0.0, lat), max(0.0, longi)

    def oval_projected_tyre_wear_rates(self, driver):
        name = str(getattr(driver, "name", driver) or "")
        driver_obj = (
            driver
            if getattr(driver, "name", None)
            else self.driver_by_name.get(name)
        )
        comp = self.tyre_comp.get(name, self.tyre_model.default_compound)
        lap_index = max(0, int(self.laps.get(name, 0) or 0))
        base = max(0.0, float(self.strategy_wear_rate(comp, lap_index)))
        try:
            wear_scale = float(self.driver_tyre_management_factor(driver_obj))
        except Exception:
            wear_scale = 1.0
        try:
            wear_scale *= float(driver_obj.tyre_wear_mult())
        except Exception:
            pass
        try:
            wear_scale *= float(self.driver_pace_modes.tyre_wear_multiplier(name))
        except Exception:
            pass
        try:
            wear_scale *= self._driver_style_tyre_wear_multiplier(name, driver_obj)
        except Exception:
            pass
        total = max(0.0, base * wear_scale * float(self.track_wear_mult))
        return {
            key: total * float(self.oval_tyre_load_factors.get(key, 1.0))
            for key in OVAL_TYRE_KEYS
        }

    def oval_tyre_imbalance_spin_multiplier(self, driver_name: str) -> float:
        if not self.oval_four_tyre_enabled:
            return 1.0
        values = self.driver_tyre_wear_by_corner(driver_name)
        side_delta = abs(
            (values["left_front"] + values["left_rear"])
            - (values["right_front"] + values["right_rear"])
        ) / 2.0
        axle_delta = abs(
            (values["left_front"] + values["right_front"])
            - (values["left_rear"] + values["right_rear"])
        ) / 2.0
        try:
            per_full_delta = max(
                0.0,
                float(
                    self.oval_four_tyre_cfg.get(
                        "imbalance_spin_risk_at_full_delta", 0.75
                    )
                    or 0.75
                ),
            )
        except Exception:
            per_full_delta = 0.75
        return 1.0 + min(1.0, max(side_delta, axle_delta)) * per_full_delta

    def oval_fuel_laps_remaining(self, driver_name: str) -> float:
        name = str(driver_name or "")
        burn = max(0.0, float(self.fuel_burn_per_lap.get(name, 0.0) or 0.0))
        if burn <= 1e-9:
            return float("inf")
        return max(0.0, float(self.fuel_onboard.get(name, 0.0) or 0.0)) / burn

    def _oval_fuel_burn_multiplier(self, driver_name: str) -> float:
        if self._physics_series_mode != "oval":
            return 1.0
        name = str(driver_name or "")
        if self.vsc_active:
            return max(
                0.0,
                min(
                    1.0,
                    self._safety_car_cfg_float("vsc_fuel_burn_mult", 0.65),
                ),
            )
        if self.sc_active or bool(self.sc_lap_active.get(name, False)):
            return float(self.oval_caution_fuel_burn_multiplier)
        pace_map = self.oval_refueling_cfg.get(
            "pace_mode_burn_multipliers",
            {"conserve": 0.94, "neutral": 1.0, "push": 1.05},
        )
        engine_map = self.oval_refueling_cfg.get(
            "engine_mode_burn_multipliers",
            {"low": 0.94, "medium": 1.0, "high": 1.06},
        )
        try:
            pace_mode = str(self.driver_pace_modes.active_mode(name) or "neutral")
        except Exception:
            pace_mode = "neutral"
        try:
            engine_mode = str(self.driver_engine_modes.active_mode(name) or "medium")
        except Exception:
            engine_mode = "medium"
        try:
            pace_mult = float(
                pace_map.get(pace_mode, 1.0) if isinstance(pace_map, dict) else 1.0
            )
        except Exception:
            pace_mult = 1.0
        try:
            engine_mult = float(
                engine_map.get(engine_mode, 1.0)
                if isinstance(engine_map, dict)
                else 1.0
            )
        except Exception:
            engine_mult = 1.0
        return max(0.25, min(2.0, pace_mult * engine_mult))

    def _normalize_oval_pit_service_plan(self, driver_name: str, plan=None):
        name = str(driver_name or "")
        raw = dict(plan) if isinstance(plan, dict) else {}
        tyre_service = self._normalize_oval_tyre_service(raw.get("tyres", "four"))
        current_fuel = max(0.0, float(self.fuel_onboard.get(name, 0.0) or 0.0))
        if self.oval_refueling_allowed:
            try:
                target = float(
                    raw.get("fuel_target_kg", self.oval_fuel_capacity_kg)
                    or self.oval_fuel_capacity_kg
                )
            except Exception:
                target = self.oval_fuel_capacity_kg
            target = max(current_fuel, min(self.oval_fuel_capacity_kg, target))
        else:
            target = current_fuel
        return {
            "tyres": tyre_service,
            "fuel_target_kg": float(target),
            "fuel_add_kg": max(0.0, float(target) - current_fuel),
            "reason": str(raw.get("reason") or "scheduled_stop"),
            "automatic": bool(raw.get("automatic", False)),
        }

    def request_oval_pit_service(
        self,
        driver_name: str,
        tyre_service: str = "four",
        fuel_target_kg=None,
        reason: str = "manual_request",
    ) -> bool:
        name = str(driver_name or "")
        if self._physics_series_mode != "oval":
            return False
        if self.pit_remaining.get(name, 0.0) > 0.0:
            return False
        if name not in self.manual_pit_requests:
            return False
        raw = {
            "tyres": tyre_service,
            "reason": reason,
            "automatic": False,
        }
        if fuel_target_kg is not None:
            raw["fuel_target_kg"] = fuel_target_kg
        plan = self._normalize_oval_pit_service_plan(name, raw)
        self.pending_pit_service[name] = dict(plan)
        self.pending_compound[name] = self.tyre_comp.get(
            name, self.tyre_model.default_compound
        )
        self.manual_pit_requests[name] = dict(plan)
        return True

    def oval_service_state_to_save_data(self):
        """Return JSON-safe Oval fuel, tire and queued-service runtime state."""

        if self._physics_series_mode != "oval":
            return {}
        drivers = {}
        for driver in self.drivers:
            name = str(getattr(driver, "name", "") or "")
            if not name:
                continue
            pending = self.pending_pit_service.get(name)
            manual = self.manual_pit_requests.get(name)
            last = self.last_pit_service.get(name)
            drivers[name] = {
                "fuel_onboard_kg": max(
                    0.0, float(self.fuel_onboard.get(name, 0.0) or 0.0)
                ),
                "tyre_wear": self.driver_tyre_wear_by_corner(name),
                "pending_service": (
                    dict(pending) if isinstance(pending, dict) else None
                ),
                "manual_request": (
                    dict(manual) if isinstance(manual, dict) else None
                ),
                "last_service": dict(last) if isinstance(last, dict) else None,
            }
        return {
            "schema": 2,
            "refueling_allowed": bool(self.oval_refueling_allowed),
            "tank_capacity_kg": float(self.oval_fuel_capacity_kg),
            "player_strategy_plans": {
                str(name): [
                    dict(stop)
                    for stop in stops
                    if isinstance(stop, dict)
                ]
                for name, stops in (
                    self.oval_player_strategy_plans.items()
                    if isinstance(self.oval_player_strategy_plans, dict)
                    else []
                )
                if isinstance(stops, list)
            },
            "drivers": drivers,
        }

    def restore_oval_service_state(self, payload) -> bool:
        """Restore a state produced by :meth:`oval_service_state_to_save_data`."""

        if self._physics_series_mode != "oval" or not isinstance(payload, dict):
            return False
        records = payload.get("drivers")
        if not isinstance(records, dict):
            return False
        raw_strategy_plans = payload.get("player_strategy_plans")
        if isinstance(raw_strategy_plans, dict):
            restored_plans = {}
            total_laps = max(1, int(getattr(self, "total_laps", 1) or 1))
            for name, raw_stops in raw_strategy_plans.items():
                name = str(name or "")
                if name not in self.driver_by_name or not isinstance(raw_stops, list):
                    continue
                clean = []
                for raw_stop in raw_stops:
                    if not isinstance(raw_stop, dict):
                        continue
                    try:
                        lap = max(1, min(total_laps - 1, int(raw_stop.get("lap", 1) or 1)))
                    except Exception:
                        continue
                    service = self._normalize_oval_tyre_service(
                        raw_stop.get("tyres", "four")
                    )
                    fuel_mode = str(raw_stop.get("fuel_mode") or "full").lower()
                    if fuel_mode not in {"full", "finish", "custom"}:
                        fuel_mode = "full"
                    try:
                        custom_target = max(
                            0.0,
                            min(
                                self.oval_fuel_capacity_kg,
                                float(
                                    raw_stop.get(
                                        "custom_target_kg",
                                        self.oval_fuel_capacity_kg,
                                    )
                                    or self.oval_fuel_capacity_kg
                                ),
                            ),
                        )
                    except Exception:
                        custom_target = self.oval_fuel_capacity_kg
                    clean.append(
                        {
                            "lap": int(lap),
                            "tyres": service,
                            "fuel_mode": fuel_mode,
                            "custom_target_kg": float(custom_target),
                        }
                    )
                restored_plans[name] = sorted(clean, key=lambda row: int(row["lap"]))
            self.oval_player_strategy_plans = restored_plans
        restored = False
        for name, raw in records.items():
            name = str(name or "")
            if name not in self.driver_by_name or not isinstance(raw, dict):
                continue
            try:
                fuel = max(0.0, float(raw.get("fuel_onboard_kg", 0.0) or 0.0))
            except Exception:
                fuel = 0.0
            self.fuel_onboard[name] = min(self.oval_fuel_capacity_kg, fuel)
            tyre_values = raw.get("tyre_wear")
            if isinstance(tyre_values, dict):
                self.tyre_wear_by_corner[name] = {
                    key: max(
                        0.0,
                        min(
                            1.5,
                            float(tyre_values.get(key, 0.0) or 0.0),
                        ),
                    )
                    for key in OVAL_TYRE_KEYS
                }
                self._sync_oval_aggregate_tyre_wear(name)
            pending = raw.get("pending_service")
            manual = raw.get("manual_request")
            last = raw.get("last_service")
            self.pending_pit_service[name] = (
                self._normalize_oval_pit_service_plan(name, pending)
                if isinstance(pending, dict)
                else None
            )
            self.manual_pit_requests[name] = (
                self._normalize_oval_pit_service_plan(name, manual)
                if isinstance(manual, dict)
                else None
            )
            self.last_pit_service[name] = (
                dict(last) if isinstance(last, dict) else None
            )
            restored = True
        if restored:
            self.initial_fuel = {
                name: float(self.fuel_onboard.get(name, 0.0) or 0.0)
                for name in self.fuel_onboard
            }
            self._realistic_lap_inputs_cache.clear()
        return restored

    def _record_safety_car_pit_entry_order(
        self,
        driver_name: str,
        stop_duration_s: float,
    ) -> None:
        """Record a caution pit-exit schedule without changing stop timing."""

        if not self.caution_active:
            return
        name = str(driver_name or "")
        if not name:
            return
        try:
            entry_time_s = max(
                0.0,
                float(self.total_sim_time.get(name, 0.0) or 0.0),
            )
        except Exception:
            entry_time_s = 0.0
        try:
            duration_s = max(0.0, float(stop_duration_s or 0.0))
        except Exception:
            duration_s = 0.0
        self._pit_entry_serial_counter = (
            int(getattr(self, "_pit_entry_serial_counter", 0) or 0) + 1
        )
        self._sc_pit_entry_time_s[name] = float(entry_time_s)
        self._sc_pit_exit_due_time_s[name] = float(entry_time_s + duration_s)
        self._sc_pit_entry_serial[name] = int(self._pit_entry_serial_counter)
        self._sc_pit_period_by_driver[name] = int(
            getattr(self, "_safety_car_period_serial", 0) or 0
        )

    def _safety_car_pit_exit_order_key(
        self,
        driver_name: str,
    ) -> Optional[Tuple[float, int]]:
        """Return this caution period's deterministic pit-exit ordering key."""

        if not self.caution_active:
            return None
        name = str(driver_name or "")
        try:
            period = int(self._sc_pit_period_by_driver.get(name, -1))
            active_period = int(
                getattr(self, "_safety_car_period_serial", 0) or 0
            )
        except Exception:
            return None
        if period != active_period:
            return None
        try:
            exit_due = float(self._sc_pit_exit_due_time_s[name])
            entry_serial = int(self._sc_pit_entry_serial[name])
        except Exception:
            return None
        if not math.isfinite(exit_due):
            return None
        return float(exit_due), int(entry_serial)

    def _clear_safety_car_pit_order(self) -> None:
        self._sc_pit_entry_time_s.clear()
        self._sc_pit_exit_due_time_s.clear()
        self._sc_pit_entry_serial.clear()
        self._sc_pit_period_by_driver.clear()

    def _trigger_pit(self, driver_name, *, evaluated_service=None):
        pcfg = self.cfg.get("pitstops", {})
        service_plan = None
        if getattr(self, "formula_refueling_allowed", False):
            # Only the synchronous lap-boundary caller can supply a service it
            # has just evaluated. Queued/manual requests still revalidate here.
            raw = evaluated_service if evaluated_service is not None else self.pending_pit_service.get(driver_name)
            if evaluated_service is None and (raw is None or (isinstance(raw, dict) and raw.get("ai_fuel_plan"))):
                reason = raw.get("stop_reason", "automatic") if isinstance(raw, dict) else "automatic"
                driver = next(d for d in self.drivers if d.name == driver_name)
                raw = self.strategy.service_for_stop(self, driver, self.pending_compound.get(driver_name))
                raw["ai_fuel_plan"] = True
                raw["stop_reason"] = reason
                self.pending_compound[driver_name] = raw["compound"]
            service_plan = normalize_service(self, driver_name, raw)
            self.pending_pit_service[driver_name] = dict(service_plan)
            if service_plan.get("ai_fuel_plan"):
                self.events.append(
                    f"PIT PLAN: {driver_name} | {service_plan.get('stop_reason', 'automatic')} | "
                    f"{service_plan.get('compound') or 'keep tyres'} | {service_plan.get('planned_laps', 0)} laps | "
                    f"end wear {100 * service_plan.get('predicted_end_wear', 0):.1f}% | "
                    f"fuel {self.fuel_onboard.get(driver_name, 0):.1f} -> {service_plan['fuel_target_kg']:.1f} kg")
        if getattr(self, "_physics_series_mode", "") == "oval":
            service_plan = self._normalize_oval_pit_service_plan(
                driver_name,
                self.pending_pit_service.get(driver_name),
            )
            self.pending_pit_service[driver_name] = dict(service_plan)
        pit_result = None
        oval_pit_hook = getattr(self.state, "race_pit_stop_result", None)
        if callable(oval_pit_hook):
            try:
                candidate = oval_pit_hook(
                    driver_name,
                    rng=random,
                    service_plan=service_plan,
                )
            except TypeError:
                candidate = oval_pit_hook(driver_name, rng=random)
            except Exception:
                candidate = None
            if isinstance(candidate, dict):
                try:
                    candidate_time = max(0.0, float(candidate.get("time_s", 0.0)))
                except Exception:
                    candidate_time = 0.0
                if candidate_time > 0.0:
                    pit_result = dict(candidate)
                    loss = candidate_time
        if pit_result is None and self._physics_series_mode != "oval":
            team = self.driver_team.get(driver_name)
            skill = self.formula_pit_skill.get(team, self.mechanic_skill.get(team, 10))
            pit_result = roll_formula_pit_service(pcfg, skill, random, change_tyres=service_plan.get("change_tyres", True) if self.formula_refueling_allowed else True)
            if getattr(self, "formula_refueling_allowed", False):
                pit_result["refuel_time_s"] = service_plan["refuel_time_s"]
                pit_result["time_s"] += service_plan["refuel_time_s"]
                pit_result["stationary_time_s"] += service_plan["refuel_time_s"]
            loss = pit_result["time_s"]
        if pit_result is None:
            loss = float(pcfg.get("pit_lane_loss_s", 19.5))
            team = self.driver_team.get(driver_name)
            skill = self.mechanic_skill.get(team, 10)
            try:
                skill = max(1, min(20, int(skill)))
            except Exception:
                skill = 10
            delta = ((10 - skill) / 9.0) * 2.0
            if delta > 2.0:
                delta = 2.0
            if delta < -2.0:
                delta = -2.0
            loss += delta
            if random.random() < float(pcfg.get("fail_prob", 0.0)):
                extra = pcfg.get("fail_extra_s", [1.0, 3.0])
                try:
                    lo, hi = float(extra[0]), float(extra[1])
                except Exception:
                    lo, hi = 1.0, 3.0
                loss += random.uniform(lo, hi)
            pit_result = {
                "time_s": float(loss),
                "base_time_s": float(loss),
                "mistake_time_s": 0.0,
                "mistake": False,
            }
        self._front_wing_pit_extra_s[driver_name] = 0.0
        if self.pending_front_wing_change.get(driver_name, False):
            if self.driver_front_wing_change_available(driver_name):
                lo, hi = FRONT_WING_PIT_TIME_RANGE_S
                try:
                    extra_wing = random.uniform(float(lo), float(hi))
                except Exception:
                    extra_wing = 7.5
                self._front_wing_pit_extra_s[driver_name] = float(extra_wing)
                loss += float(extra_wing)
                if self._physics_series_mode != "oval":
                    pit_result["front_wing_time_s"] = float(extra_wing)
                    pit_result["stationary_time_s"] = float(pit_result.get("stationary_time_s", 0.0)) + float(extra_wing)
            else:
                self.pending_front_wing_change[driver_name] = False
        self.manual_pit_requests[driver_name] = None
        if self._physics_series_mode != "oval":
            self.formula_pit_now_requests[driver_name] = None
        if isinstance(service_plan, dict):
            pit_result.update(service_plan)
        pit_result["time_s"] = float(loss)
        self._record_safety_car_pit_entry_order(driver_name, loss)
        self._active_pit_stop_details[driver_name] = pit_result
        self.pit_stop_total_time[driver_name] = float(loss)
        self.pit_stop_elapsed[driver_name] = 0.0
        self.pit_remaining[driver_name] = loss
        self.pitted_last_lap[driver_name] = True
        self._pit_flag_decay[driver_name] = 6
        if self.pending_compound.get(driver_name) is None and not (self.formula_refueling_allowed and not service_plan.get("change_tyres", True)):
            self.pending_compound[driver_name] = self.tyre_comp.get(
                driver_name,
                self.tyre_model.default_compound,
            )
        # log pit entry
        self.events.append(f"PIT IN: {driver_name}")
        self._emit_radio_event(
            "race_pit_entry",
            driver_name,
            compound=str(self.pending_compound.get(driver_name) or self.tyre_comp.get(driver_name, "tyres")),
        )
        if bool(pit_result.get("mistake")):
            self.events.append(
                f"PIT CREW MISTAKE: {driver_name} (+{float(pit_result.get('mistake_time_s', 0.0)):.1f}s)"
            )

    def set_formula_start_fuel(self, name, amount, *, manual=True):
        if not self.formula_refueling_allowed or any(float(v) > 1e-6 for v in self.total_sim_time.values()):
            return False
        if name not in self.fuel_onboard:
            return False
        value = min(self.formula_fuel_capacity_kg, finite_fuel(amount))
        self.fuel_onboard[name] = value
        self.initial_fuel[name] = value
        if manual:
            self.formula_start_fuel_overrides[name] = value
        self._lap_start_fuel[name] = value
        if self.formula_fuel_stints.get(name):
            self.formula_fuel_stints[name][0]["fuel_target_kg"] = value
        self._realistic_lap_inputs_cache.pop(name, None)
        return True

    def manual_pit(self, driver_name, compound):
        if getattr(self, "formula_refueling_allowed", False):
            if self.pit_remaining.get(driver_name, 0) > 0 or driver_name not in self.manual_pit_requests:
                return False
            plan = normalize_service(self, driver_name, compound)
            if plan["change_tyres"] and plan["compound"] not in self.formula_available_compounds(driver_name):
                return False
            if not plan["change_tyres"] and plan["fuel_add_kg"] <= 0 and not self.pending_front_wing_change.get(driver_name):
                return False
            self.pending_pit_service[driver_name] = plan
            self.pending_compound[driver_name] = plan["compound"]
            self.manual_pit_requests[driver_name] = dict(plan)
            return True
        if self.pit_remaining.get(driver_name, 0.0) > 0.0:
            return False
        if driver_name not in self.manual_pit_requests:
            return False
        if (
            self._physics_series_mode != "oval"
            and self.weekend_tyre_manager is not None
            and self.weekend_tyre_manager.enabled()
            and compound not in self.formula_available_compounds(driver_name)
        ):
            return False
        if self._physics_series_mode == "oval" and isinstance(compound, dict):
            plan = self._normalize_oval_pit_service_plan(driver_name, compound)
            self.pending_pit_service[driver_name] = dict(plan)
            self.pending_compound[driver_name] = self.tyre_comp.get(
                driver_name, self.tyre_model.default_compound
            )
            self.manual_pit_requests[driver_name] = dict(plan)
            return True
        self.pending_compound[driver_name] = compound
        self.manual_pit_requests[driver_name] = compound
        return True

    def formula_pit_now_pending(self, driver_name: str) -> bool:
        """Return whether a player urgent-stop request is awaiting pit entry."""

        if self._physics_series_mode == "oval":
            return False
        return self.formula_pit_now_requests.get(str(driver_name or "")) is not None

    def request_formula_pit_now(self, driver_name: str, compound) -> bool:
        """Queue one Formula stop for the driver's next available pit entry."""

        name = str(driver_name or "")
        if self._physics_series_mode == "oval":
            return False
        if not self.player_team or self.driver_team.get(name) != self.player_team:
            return False
        if name not in self.formula_pit_now_requests:
            return False
        if self.dnf.get(name, False) or name in self.finished:
            return False
        # Once the final lap is underway, the next line crossing is the finish
        # rather than another available pit entry.
        if int(self.laps.get(name, 0) or 0) >= max(0, int(self.total_laps or 0) - 1):
            return False
        if self.pit_remaining.get(name, 0.0) > 0.0:
            return False
        if self.formula_pit_now_pending(name):
            return False
        if not self.manual_pit(name, compound):
            return False
        self.formula_pit_now_requests[name] = {
            "compound": self.pending_compound.get(name),
            "pit_count_before": int(self.pit_count.get(name, 0) or 0),
            "requested_lap": int(self.laps.get(name, 0) or 0),
        }
        self._emit_radio_event(
            "race_pit_now",
            name,
            compound=str(self.pending_compound.get(name) or compound or "tyres"),
        )
        return True

    def cancel_formula_pit_now(self, driver_name: str) -> bool:
        """Cancel an urgent Formula request before the car reaches pit entry."""

        name = str(driver_name or "")
        if not self.formula_pit_now_pending(name):
            return False
        if self.pit_remaining.get(name, 0.0) > 0.0:
            return False
        self.formula_pit_now_requests[name] = None
        self.manual_pit_requests[name] = None
        self.pending_compound[name] = None
        return True

    def formula_available_compounds(self, driver_name: str) -> list:
        if (
            self._physics_series_mode == "oval"
            or self.weekend_tyre_manager is None
            or not self.weekend_tyre_manager.enabled()
            or self.weekend is None
        ):
            return list(self.available_compounds)
        return [
            compound
            for compound in self.available_compounds
            if self.weekend_tyre_manager.sets_for_driver(
                self.weekend, driver_name, compound, available_only=True
            )
        ]

    def _driver_tyre_grip_bonus_mult(self, driver_name: str) -> float:
        team_name = self.driver_team.get(driver_name)
        if not team_name:
            return 1.0
        contract = self.tyre_contracts.get(team_name, {}) if isinstance(self.tyre_contracts, dict) else {}
        if not isinstance(contract, dict):
            return 1.0
        contract_type = str(contract.get("type") or self.tyre_contract_type_by_driver.get(driver_name) or "").lower()
        raw_bonus = contract.get("bonus_grip_mult")
        if raw_bonus is None:
            if contract_type == "partner" and contract.get("bonus_pace") is not None:
                raw_bonus = 1.0005
            elif contract_type == "works" and float(contract.get("funding_weekly_m", 0.0) or 0.0) > 0.0:
                raw_bonus = 1.0005
        try:
            bonus = float(raw_bonus) if raw_bonus is not None else 1.0
        except Exception:
            bonus = 1.0
        if bonus <= 0.0:
            bonus = 1.0
        return float(bonus)

    def _driver_supplier_ratings(self, driver_name: str, compound_key: str) -> tuple[float, float]:
        supplier = self.tyre_supplier_by_driver.get(driver_name)
        if supplier is None:
            return 50.0, 50.0
        try:
            pace_raw = (getattr(supplier, "pace", {}) or {}).get(compound_key, 50.0)
        except Exception:
            pace_raw = 50.0
        try:
            dur_raw = (getattr(supplier, "durability", {}) or {}).get(compound_key, 50.0)
        except Exception:
            dur_raw = 50.0
        return (
            float(self.tyre_model.supplier_pace_rating(pace_raw)),
            float(self.tyre_model.supplier_durability_rating(dur_raw)),
        )

    def _effective_engine_power_rating(self, driver_name: str, base_rating: float) -> float:
        try:
            rating = float(base_rating)
        except Exception:
            rating = 50.0
        rating = max(0.0, min(100.0, rating))

        specs = self.engine_unit_specs if isinstance(self.engine_unit_specs, dict) else {}
        spec = specs.get(driver_name)
        if not isinstance(spec, dict):
            return float(rating)

        start_cond = self._safe_condition_pct(spec.get("condition", 100.0), default=100.0)
        try:
            wear_delta = float(self.engine_wear_delta.get(driver_name, 0.0) or 0.0)
        except Exception:
            wear_delta = 0.0
        current_cond = max(0.0, min(100.0, float(start_cond) - max(0.0, wear_delta)))
        wear_frac = 1.0 - (current_cond / 100.0)

        try:
            max_loss = float(spec.get("max_power_rating_loss", 0.0) or 0.0)
        except Exception:
            max_loss = 0.0
        if max_loss <= 1e-9:
            try:
                rating += float(self.driver_engine_modes.power_rating_delta(driver_name))
            except Exception:
                pass
            return float(max(0.0, min(100.0, rating)))

        try:
            quantum = max(0.01, float(spec.get("power_rating_wear_quantum", 0.25) or 0.25))
        except Exception:
            quantum = 0.25

        effective = rating - (wear_frac * max_loss)
        effective = round(effective / quantum) * quantum
        try:
            effective += float(self.driver_engine_modes.power_rating_delta(driver_name))
        except Exception:
            pass
        return float(max(0.0, min(100.0, effective)))

    @staticmethod
    def _rating_to_points(value: float, default: float = 50.0) -> float:
        try:
            rating = float(value)
        except Exception:
            rating = float(default)
        if rating < 0.0:
            rating = 0.0
        elif rating > 100.0:
            rating = 100.0
        return rating - 50.0

    @staticmethod
    def _points_to_delta_seconds(points: float) -> float:
        try:
            pts = float(points)
        except Exception:
            pts = 0.0
        return -pts * 0.10

    @staticmethod
    def _delta_seconds_to_points(delta_seconds: float) -> float:
        try:
            delta = float(delta_seconds)
        except Exception:
            delta = 0.0
        return -delta / 0.10

    @staticmethod
    def _kmh_to_rating_points(zone: str, kmh_delta: float) -> float:
        try:
            kmh = float(kmh_delta)
        except Exception:
            kmh = 0.0
        if abs(kmh) <= 1e-12:
            return 0.0
        zone_key = str(zone)
        if zone_key == "slow":
            zone_key = "low"
        if zone_key == "straight":
            per_point = float(UNIVERSAL_KMH_PER_RATING.get("straight", 0.0))
        else:
            per_point = float(ATTR_KMH_PER_RATING.get(zone_key, 0.0))
        if per_point <= 1e-12:
            return 0.0
        return kmh / per_point

    def _grid_spacing_progress(self) -> float:
        try:
            track_length_m = float(getattr(self.realistic_physics, "track_length_m", 0.0) or 0.0)
        except Exception:
            track_length_m = 0.0
        if bool(getattr(self, "formula_multiline_grid_enabled", False)):
            try:
                spacing_m = max(
                    4.0,
                    min(
                        30.0,
                        float(
                            (getattr(self, "formula_starting_grid", {}) or {}).get(
                                "row_spacing_m",
                                8.0,
                            )
                            or 8.0
                        ),
                    ),
                )
            except Exception:
                spacing_m = 8.0
        else:
            try:
                spacing_m = max(0.0, float(getattr(self, "grid_start_spacing_km", 0.05) or 0.05) * 1000.0)
            except Exception:
                spacing_m = 50.0
        if track_length_m <= 1e-9 or spacing_m <= 1e-9:
            return 0.0
        return max(0.0, min(0.5, spacing_m / track_length_m))

    def seed_grid_progress(self, grid_order=None) -> None:
        ordered_names = []
        if isinstance(grid_order, list) and grid_order:
            ordered_names.extend(str(name) for name in grid_order if name in self.driver_by_name)
        if not ordered_names:
            ordered_names.extend(d.name for d in self.drivers)
        seen = set(ordered_names)
        for driver in self.drivers:
            if driver.name not in seen:
                ordered_names.append(driver.name)
                seen.add(driver.name)

        step = self._grid_spacing_progress()
        for name in list(self.progress.keys()):
            self.progress[name] = 0.0
        total = len(ordered_names)
        for idx, name in enumerate(ordered_names):
            if name not in self.progress:
                continue
            if step > 0.0:
                two_column_grid = bool(
                    self.oval_pace_lap_start_enabled
                    or getattr(self, "formula_multiline_grid_enabled", False)
                )
                slot_index = (idx // 2) if two_column_grid else idx
                slot_offset = max(EPSILON, step * float(slot_index + 1))
                if (
                    bool(getattr(self, "formula_multiline_grid_enabled", False))
                    and idx % 2 == 1
                ):
                    try:
                        lap_length_m = max(
                            1.0,
                            float(getattr(self.realistic_physics, "track_length_m", 0.0) or 0.0),
                        )
                        stagger_m = max(
                            0.0,
                            min(
                                10.0,
                                float(
                                    (getattr(self, "formula_starting_grid", {}) or {}).get(
                                        "stagger_m",
                                        2.0,
                                    )
                                    or 0.0
                                ),
                            ),
                        )
                        slot_offset += stagger_m / lap_length_m
                    except Exception:
                        pass
                self.progress[name] = max(0.0, min(0.999999, 1.0 - slot_offset))
            else:
                self.progress[name] = 0.999999

        if hasattr(self, "order") and isinstance(self.order, list):
            try:
                self.order.sort(
                    key=lambda d: (
                        -self.laps.get(d.name, 0),
                        -self.progress.get(d.name, 0.0),
                    )
                )
            except Exception:
                pass
        if hasattr(self, "distance_time_history") and hasattr(self, "_initialize_distance_gap_tracking"):
            try:
                self._initialize_distance_gap_tracking()
            except Exception:
                pass
        if bool(getattr(self, "oval_formation_active", False)):
            self._reset_oval_formation_grid(ordered_names)
        racecraft = getattr(self, "racecraft_model", None)
        seed_lanes = getattr(racecraft, "seed_grid_lanes", None)
        if callable(seed_lanes):
            try:
                seed_lanes(ordered_names)
            except Exception:
                pass

    def _arm_driver_race_start(self, driver_name: str, crossing_time: float) -> None:
        self.grid_start_pending[driver_name] = False
        self.current_lap_start[driver_name] = float(crossing_time)
        self.current_sector_index[driver_name] = 0
        self.current_sector_start_time[driver_name] = float(crossing_time)
        self.current_lap_sector_times[driver_name] = []
        self.current_lap_peak_speed_kmh[driver_name] = 0.0
        self.last_sector_times[driver_name] = []
        self._sync_lap_start_snapshot(driver_name)
        self.sc_lap_active[driver_name] = bool(self.sc_phase in ("collecting", "returning"))

    @staticmethod
    def _clamp01(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    @staticmethod
    def _lerp(start: float, end: float, t: float) -> float:
        return float(start) + ((float(end) - float(start)) * max(0.0, min(1.0, float(t))))

    def _driver_start_attr_score(self, driver, attr: str, max_points: float) -> float:
        try:
            if hasattr(driver, "effective_rating"):
                rating = float(driver.effective_rating(attr))
            else:
                rating = float(getattr(driver, attr, 10.0) or 10.0)
        except Exception:
            rating = 10.0
        rating = max(0.0, min(20.0, rating))
        return max(0.0, min(float(max_points), (rating / 20.0) * float(max_points)))

    def _team_start_acceleration_bonus(self) -> dict:
        team_values = {}
        team_counts = {}
        for d in self.drivers:
            team = self.driver_team.get(d.name)
            if not team:
                continue
            attrs = self.driver_attrs_by_driver.get(d.name, {}) or {}
            try:
                value = float(attrs.get("acceleration", 0.0) or 0.0)
            except Exception:
                value = 0.0
            team_values[team] = float(team_values.get(team, 0.0)) + value
            team_counts[team] = int(team_counts.get(team, 0) or 0) + 1
        averages = []
        for team, total in team_values.items():
            count = max(1, int(team_counts.get(team, 1) or 1))
            averages.append((team, float(total) / float(count)))
        averages.sort(key=lambda item: (-item[1], item[0]))
        if not averages:
            return {}
        if len(averages) == 1:
            return {averages[0][0]: 15.0}
        last_rank = float(len(averages) - 1)
        return {
            team: 15.0 * (1.0 - (float(rank) / last_rank))
            for rank, (team, _value) in enumerate(averages)
        }

    def _roll_start_quality(self, driver, team_accel_bonus: Optional[dict] = None) -> dict:
        name = str(getattr(driver, "name", "") or "")
        team = self.driver_team.get(name)
        control_score = self._driver_start_attr_score(driver, "control", 20.0)
        smooth_score = self._driver_start_attr_score(driver, "smoothness", 20.0)
        try:
            confidence_score = (driver_confidence_value(driver) / 100.0) * 10.0
        except Exception:
            confidence_score = 5.0
        try:
            comfort_score = (float(self.driver_practice_comfort.get(name, 0.0) or 0.0) / 100.0) * 15.0
        except Exception:
            comfort_score = 0.0
        try:
            accel_score = float((team_accel_bonus or {}).get(team, 0.0) or 0.0)
        except Exception:
            accel_score = 0.0
        random_score = random.uniform(-40.0, 40.0)
        raw_score = control_score + smooth_score + confidence_score + accel_score + comfort_score + random_score
        score = max(1.0, min(100.0, float(raw_score)))
        rating = max(1, min(10, int((score - 1.0) // 10.0) + 1))
        return {
            "score": float(score),
            "rating": int(rating),
            "control": float(control_score),
            "smoothness": float(smooth_score),
            "confidence": float(confidence_score),
            "car_acceleration": float(accel_score),
            "comfort": float(comfort_score),
            "random": float(random_score),
        }

    def _roll_race_start_qualities(self) -> None:
        accel_bonus = self._team_start_acceleration_bonus()
        for d in self.drivers:
            name = str(getattr(d, "name", "") or "")
            if not name:
                continue
            result = self._roll_start_quality(d, accel_bonus)
            self.start_quality_score[name] = float(result.get("score", 50.0))
            self.start_rating[name] = int(result.get("rating", 5))
            self.start_quality_breakdown[name] = dict(result)
            rating_norm = self._clamp01((float(self.start_rating[name]) - 1.0) / 9.0)
            self.start_reaction_delay_remaining[name] = self._lerp(0.1075, 0.02, rating_norm)
            self.launch_accel_mult[name] = self._lerp(0.9, 1.1, rating_norm)

    def _launch_duration_for_driver(self, driver, race_start_cfg=None, start_rating: Optional[int] = None) -> float:
        cfg = race_start_cfg if isinstance(race_start_cfg, dict) else (self.cfg.get("race_start", {}) or {})
        try:
            base_duration = float(cfg.get("launch_duration_s", RACE_START_LAUNCH_DURATION_S) or RACE_START_LAUNCH_DURATION_S)
        except Exception:
            base_duration = float(RACE_START_LAUNCH_DURATION_S)
        base_duration = max(2.5, base_duration)
        if start_rating is not None:
            try:
                rating_norm = self._clamp01((float(start_rating) - 1.0) / 9.0)
            except Exception:
                rating_norm = 4.0 / 9.0
            duration_mult = self._lerp(1.1425, 0.9275, rating_norm)
            return max(2.2, min(6.5, base_duration * duration_mult))
        ratings = []
        for attr in ("smoothness", "consistency", "control", "braking"):
            try:
                ratings.append(max(1.0, min(20.0, float(getattr(driver, attr, 10.0) or 10.0))))
            except Exception:
                ratings.append(10.0)
        skill = sum(ratings) / max(1, len(ratings))
        normalized = (skill - 1.0) / 19.0
        duration_mult = 1.10 - (0.20 * normalized)
        return max(2.8, min(5.5, base_duration * duration_mult))

    def _launch_accel_multiplier(self, driver_name: str) -> float:
        if self.oval_pace_lap_start_enabled:
            return 1.0
        if self.laps.get(driver_name, 0) > 0:
            return 1.0
        try:
            duration = float((self.launch_duration_s or {}).get(driver_name, RACE_START_LAUNCH_DURATION_S))
            elapsed = float((self.launch_elapsed_s or {}).get(driver_name, 0.0) or 0.0)
        except Exception:
            return 1.0
        if duration <= 1e-9 or elapsed >= duration:
            return 1.0
        try:
            return max(0.75, min(1.25, float(self.launch_accel_mult.get(driver_name, 1.0) or 1.0)))
        except Exception:
            return 1.0

    def _launch_speed_factor(
        self,
        driver_name: str,
        dt_sim: float,
        base_speed_kmh: Optional[float] = None,
    ) -> float:
        if self.oval_pace_lap_start_enabled:
            if not bool(getattr(self, "oval_formation_complete", False)):
                return 1.0
            try:
                duration = max(1e-9, float(self.oval_release_acceleration_s))
                elapsed_before = max(
                    0.0,
                    float(self._oval_release_elapsed_s.get(driver_name, 0.0) or 0.0),
                )
                elapsed_after = min(duration, elapsed_before + max(0.0, float(dt_sim)))
                self._oval_release_elapsed_s[driver_name] = elapsed_after
                t0 = self._clamp01(elapsed_before / duration)
                t1 = self._clamp01(elapsed_after / duration)
                smooth0 = t0 * t0 * (3.0 - (2.0 * t0))
                smooth1 = t1 * t1 * (3.0 - (2.0 * t1))
                smooth = 0.5 * (smooth0 + smooth1)
                minimum = float(self.oval_release_min_speed_factor)
                if base_speed_kmh is not None and float(base_speed_kmh) > 1e-9:
                    # The green-flag ramp begins at pacing speed. This prevents
                    # a low-speed track profile from making the field visibly
                    # slow down at the instant the pace car pulls in.
                    minimum = max(
                        minimum,
                        min(1.0, float(self.oval_pace_speed_kmh) / float(base_speed_kmh)),
                    )
                return max(minimum, min(1.0, minimum + ((1.0 - minimum) * smooth)))
            except Exception:
                return 1.0
        if self.laps.get(driver_name, 0) > 0:
            return 1.0
        try:
            duration = float((self.launch_duration_s or {}).get(driver_name, RACE_START_LAUNCH_DURATION_S))
        except Exception:
            duration = float(RACE_START_LAUNCH_DURATION_S)
        if duration <= 1e-9:
            return 1.0
        elapsed_before = max(0.0, float((self.launch_elapsed_s or {}).get(driver_name, 0.0) or 0.0))
        elapsed_after = min(duration, elapsed_before + max(0.0, float(dt_sim)))
        self.launch_elapsed_s[driver_name] = elapsed_after
        start = self._clamp01(elapsed_before / duration)
        end = self._clamp01(elapsed_after / duration)
        # Smoothstep produces a clean standing start without a harsh speed jump.
        start_factor = start * start * (3.0 - (2.0 * start))
        end_factor = end * end * (3.0 - (2.0 * end))
        avg_factor = 0.5 * (start_factor + end_factor)
        return max(0.0, min(1.0, avg_factor))

    def _load_wing_setup_config(self):
        cfg = (getattr(self, "cfg", {}) or {}).get("car_setup", {}) or {}
        wing = cfg.get("wing_levels", {}) if isinstance(cfg, dict) else {}
        defaults = self.DEFAULT_WING_SETUP_CFG
        try:
            min_level = int(wing.get("min_level", defaults["min_level"]))
        except Exception:
            min_level = int(defaults["min_level"])
        try:
            max_level = int(wing.get("max_level", defaults["max_level"]))
        except Exception:
            max_level = int(defaults["max_level"])
        if max_level < min_level:
            min_level, max_level = max_level, min_level
        try:
            neutral = int(wing.get("neutral_level", defaults["neutral_level"]))
        except Exception:
            neutral = int(defaults["neutral_level"])
        neutral = max(min_level, min(max_level, neutral))

        def _read_map(key: str, legacy_key: str):
            src = {}
            if isinstance(wing, dict):
                src = wing.get(key, {})
                if not isinstance(src, dict):
                    src = wing.get(legacy_key, {})
            base = dict(defaults[key])
            if isinstance(src, dict):
                for field in ("slow", "med", "high", "straight"):
                    if field in src:
                        try:
                            base[field] = float(src.get(field))
                        except Exception:
                            pass
            return base

        def _read_diminishing():
            src = wing.get("diminishing_returns", {}) if isinstance(wing, dict) else {}
            base = dict(defaults.get("diminishing_returns", {}))
            if not isinstance(src, dict):
                src = {}
            try:
                base["enabled"] = bool(src.get("enabled", base.get("enabled", True)))
            except Exception:
                pass
            for key in ("coefficient", "exponent", "min_efficiency", "max_efficiency"):
                if key in src:
                    try:
                        base[key] = float(src.get(key))
                    except Exception:
                        pass
            try:
                min_eff = float(base.get("min_efficiency", 0.70))
            except Exception:
                min_eff = 0.70
            try:
                max_eff = float(base.get("max_efficiency", 1.0))
            except Exception:
                max_eff = 1.0
            if max_eff < min_eff:
                min_eff, max_eff = max_eff, min_eff
            base["min_efficiency"] = max(0.0, min(1.0, min_eff))
            base["max_efficiency"] = max(0.0, min(1.0, max_eff))
            try:
                base["coefficient"] = max(0.0, float(base.get("coefficient", 0.025)))
            except Exception:
                base["coefficient"] = 0.025
            try:
                base["exponent"] = max(0.1, float(base.get("exponent", 1.35)))
            except Exception:
                base["exponent"] = 1.35
            return base

        self.wing_setup_cfg = {
            "min_level": min_level,
            "max_level": max_level,
            "neutral_level": neutral,
            "front_per_click_kmh": _read_map("front_per_click_kmh", "front_per_click"),
            "rear_per_click_kmh": _read_map("rear_per_click_kmh", "rear_per_click"),
            "diminishing_returns": _read_diminishing(),
        }

    def _init_driver_wing_setup(self, initial_setup) -> None:
        cfg = self.wing_setup_cfg
        default_front = int(cfg["neutral_level"])
        default_rear = int(cfg["neutral_level"])
        if isinstance(initial_setup, dict):
            store = initial_setup
        else:
            store = {}
        self.wing_setup_by_driver = store
        self._wing_setup_kmh_delta_by_driver = {}
        for d in self.drivers:
            rec = store.get(d.name)
            if not isinstance(rec, dict):
                rec = {}
            try:
                front = int(rec.get("front", default_front))
            except Exception:
                front = default_front
            try:
                rear = int(rec.get("rear", default_rear))
            except Exception:
                rear = default_rear
            front = max(int(cfg["min_level"]), min(int(cfg["max_level"]), front))
            rear = max(int(cfg["min_level"]), min(int(cfg["max_level"]), rear))
            store[d.name] = {"front": front, "rear": rear}
            self._wing_setup_kmh_delta_by_driver[d.name] = self._compute_wing_setup_attr_delta(front, rear)

    def get_wing_setup(self, driver_name: str) -> tuple[int, int]:
        cfg = self.wing_setup_cfg
        neutral = int(cfg["neutral_level"])
        store = getattr(self, "wing_setup_by_driver", {}) or {}
        rec = store.get(driver_name, {}) if isinstance(store, dict) else {}
        try:
            front = int(rec.get("front", neutral))
        except Exception:
            front = neutral
        try:
            rear = int(rec.get("rear", neutral))
        except Exception:
            rear = neutral
        front = max(int(cfg["min_level"]), min(int(cfg["max_level"]), front))
        rear = max(int(cfg["min_level"]), min(int(cfg["max_level"]), rear))
        return front, rear

    def _refresh_driver_setup_caches(self, driver_name: str) -> None:
        self._realistic_lap_inputs_cache.pop(driver_name, None)
        self._realistic_consistency_sensitivity.pop(driver_name, None)
        try:
            cache = getattr(self, "_driver_style_mismatch_cache", None)
            if isinstance(cache, dict):
                for key in list(cache.keys()):
                    if isinstance(key, tuple) and key and key[0] == driver_name:
                        cache.pop(key, None)
        except Exception:
            pass
        try:
            self._realistic_static_input_cache.pop(driver_name, None)
        except Exception:
            pass
        if not self.use_realistic_physics:
            return
        driver = self.driver_by_name.get(driver_name)
        if driver is None:
            return
        try:
            lap, _ = self.realistic_physics.compute_lap(
                self._realistic_lap_inputs_for_driver(
                    driver,
                    apply_consistency=False,
                    apply_aero=False,
                )
            )
            self.realistic_base_lap_by_driver[driver_name] = max(MIN_VALID_LAP_TIME, float(lap))
        except Exception:
            pass

    def set_wing_setup(self, driver_name: str, *, front: Optional[int] = None, rear: Optional[int] = None) -> bool:
        # Race setups are editable on the grid only; locked once race starts.
        if self.tick_count > 0 or any(self.laps.values()):
            return False
        cfg = self.wing_setup_cfg
        store = getattr(self, "wing_setup_by_driver", None)
        if not isinstance(store, dict):
            store = {}
            self.wing_setup_by_driver = store
        cur_front, cur_rear = self.get_wing_setup(driver_name)
        if front is None:
            front = cur_front
        if rear is None:
            rear = cur_rear
        try:
            front = int(front)
        except Exception:
            front = cur_front
        try:
            rear = int(rear)
        except Exception:
            rear = cur_rear
        front = max(int(cfg["min_level"]), min(int(cfg["max_level"]), front))
        rear = max(int(cfg["min_level"]), min(int(cfg["max_level"]), rear))
        if front == cur_front and rear == cur_rear:
            return True
        store[driver_name] = {"front": front, "rear": rear}
        self._wing_setup_kmh_delta_by_driver[driver_name] = self._compute_wing_setup_attr_delta(front, rear)
        self._refresh_driver_setup_caches(driver_name)
        return True

    def _load_tyre_pressure_config(self):
        cfg = (getattr(self, "cfg", {}) or {}).get("car_setup", {}) or {}
        raw = cfg.get("tyre_pressures", cfg.get("tire_pressures", {}))
        if not isinstance(raw, dict):
            raw = {}

        def _read_axis(key: str):
            src = raw.get(key, {})
            if not isinstance(src, dict):
                src = {}
            base = dict(self.DEFAULT_TYRE_PRESSURE_CFG[key])
            for num_key in ("min_psi", "max_psi", "neutral_psi", "step_psi"):
                if num_key in src:
                    try:
                        base[num_key] = float(src.get(num_key))
                    except Exception:
                        pass
            for effect_key in ("min_effects", "max_effects"):
                merged = dict(base[effect_key])
                eff_src = src.get(effect_key, {})
                if isinstance(eff_src, dict):
                    for name in merged.keys():
                        if name in eff_src:
                            try:
                                merged[name] = float(eff_src.get(name))
                            except Exception:
                                pass
                base[effect_key] = merged
            min_psi = float(base["min_psi"])
            max_psi = float(base["max_psi"])
            if max_psi < min_psi:
                min_psi, max_psi = max_psi, min_psi
            neutral_psi = max(min_psi, min(max_psi, float(base["neutral_psi"])))
            step_psi = max(0.1, float(base.get("step_psi", 0.1) or 0.1))
            base["min_psi"] = min_psi
            base["max_psi"] = max_psi
            base["neutral_psi"] = neutral_psi
            base["step_psi"] = step_psi
            return base

        self.tyre_pressure_cfg = {
            "front": _read_axis("front"),
            "rear": _read_axis("rear"),
        }

    def _round_pressure_value(self, value: float) -> float:
        return round(float(value) * 10.0) / 10.0

    def _clamp_pressure_value(self, axis: str, value: float) -> float:
        cfg = (self.tyre_pressure_cfg or {}).get(axis, {})
        lo = float(cfg.get("min_psi", 20.0))
        hi = float(cfg.get("max_psi", 28.0))
        step = max(0.1, float(cfg.get("step_psi", 0.1) or 0.1))
        if hi < lo:
            lo, hi = hi, lo
        try:
            v = float(value)
        except Exception:
            v = float(cfg.get("neutral_psi", lo))
        v = max(lo, min(hi, v))
        clicks = round((v - lo) / step)
        return self._round_pressure_value(lo + (clicks * step))

    def _init_driver_tyre_pressure_setup(self, initial_setup) -> None:
        cfg = self.tyre_pressure_cfg
        default_front = float(cfg["front"]["neutral_psi"])
        default_rear = float(cfg["rear"]["neutral_psi"])
        if isinstance(initial_setup, dict):
            store = initial_setup
        else:
            store = {}
        self.tyre_pressure_setup_by_driver = store
        self._tyre_pressure_effects_by_driver = {}
        for d in self.drivers:
            rec = store.get(d.name)
            if not isinstance(rec, dict):
                rec = {}
            try:
                front = float(rec.get("front_psi", default_front))
            except Exception:
                front = default_front
            try:
                rear = float(rec.get("rear_psi", default_rear))
            except Exception:
                rear = default_rear
            front = self._clamp_pressure_value("front", front)
            rear = self._clamp_pressure_value("rear", rear)
            store[d.name] = {"front_psi": front, "rear_psi": rear}
            self._tyre_pressure_effects_by_driver[d.name] = self._compute_tyre_pressure_effects(front, rear)

    def get_tyre_pressures(self, driver_name: str) -> tuple[float, float]:
        cfg = self.tyre_pressure_cfg
        default_front = float(cfg["front"]["neutral_psi"])
        default_rear = float(cfg["rear"]["neutral_psi"])
        store = getattr(self, "tyre_pressure_setup_by_driver", {}) or {}
        rec = store.get(driver_name, {}) if isinstance(store, dict) else {}
        try:
            front = float(rec.get("front_psi", default_front))
        except Exception:
            front = default_front
        try:
            rear = float(rec.get("rear_psi", default_rear))
        except Exception:
            rear = default_rear
        return (
            self._clamp_pressure_value("front", front),
            self._clamp_pressure_value("rear", rear),
        )

    def set_tyre_pressures(
        self,
        driver_name: str,
        *,
        front_psi: Optional[float] = None,
        rear_psi: Optional[float] = None,
    ) -> bool:
        if self.tick_count > 0 or any(self.laps.values()):
            return False
        store = getattr(self, "tyre_pressure_setup_by_driver", None)
        if not isinstance(store, dict):
            store = {}
            self.tyre_pressure_setup_by_driver = store
        cur_front, cur_rear = self.get_tyre_pressures(driver_name)
        if front_psi is None:
            front_psi = cur_front
        if rear_psi is None:
            rear_psi = cur_rear
        front_psi = self._clamp_pressure_value("front", front_psi)
        rear_psi = self._clamp_pressure_value("rear", rear_psi)
        if abs(front_psi - cur_front) <= 1e-9 and abs(rear_psi - cur_rear) <= 1e-9:
            return True
        store[driver_name] = {"front_psi": front_psi, "rear_psi": rear_psi}
        self._tyre_pressure_effects_by_driver[driver_name] = self._compute_tyre_pressure_effects(front_psi, rear_psi)
        self._refresh_driver_setup_caches(driver_name)
        return True

    def _pressure_axis_effects(self, axis: str, psi_value: float) -> dict:
        cfg = (self.tyre_pressure_cfg or {}).get(axis, {})
        neutral = float(cfg.get("neutral_psi", 0.0))
        low = float(cfg.get("min_psi", neutral))
        high = float(cfg.get("max_psi", neutral))
        if psi_value < neutral - 1e-9 and neutral > low + 1e-9:
            ratio = (neutral - float(psi_value)) / (neutral - low)
            src = cfg.get("min_effects", {})
        elif psi_value > neutral + 1e-9 and high > neutral + 1e-9:
            ratio = (float(psi_value) - neutral) / (high - neutral)
            src = cfg.get("max_effects", {})
        else:
            ratio = 0.0
            src = {}
        ratio = max(0.0, min(1.0, float(ratio)))
        out = {
            "slow": 0.0,
            "med": 0.0,
            "high": 0.0,
            "straight": 0.0,
            "accel_ms2": 0.0,
            "warmup_rate_delta": 0.0,
            "target_temp_delta_c": 0.0,
            "wear_rate_add": 0.0,
        }
        for key in out.keys():
            try:
                out[key] = float(src.get(key, 0.0) or 0.0) * ratio
            except Exception:
                out[key] = 0.0
        return out

    def _compute_tyre_pressure_effects(self, front_psi: float, rear_psi: float) -> dict:
        out = {
            "slow": 0.0,
            "med": 0.0,
            "high": 0.0,
            "straight": 0.0,
            "accel_ms2": 0.0,
            "warmup_rate_delta": 0.0,
            "target_temp_delta_c": 0.0,
            "wear_rate_add": 0.0,
            "front_psi": float(front_psi),
            "rear_psi": float(rear_psi),
        }
        for axis, psi_value in (("front", front_psi), ("rear", rear_psi)):
            axis_effects = self._pressure_axis_effects(axis, psi_value)
            for key in ("slow", "med", "high", "straight", "accel_ms2", "warmup_rate_delta", "target_temp_delta_c", "wear_rate_add"):
                out[key] += float(axis_effects.get(key, 0.0) or 0.0)
        return out

    def tyre_pressure_effects(
        self,
        driver_name: str,
        *,
        front_override: Optional[float] = None,
        rear_override: Optional[float] = None,
    ) -> dict:
        if front_override is None and rear_override is None:
            cached = (getattr(self, "_tyre_pressure_effects_by_driver", {}) or {}).get(driver_name)
            if isinstance(cached, dict):
                return cached
        front, rear = self.get_tyre_pressures(driver_name)
        if front_override is not None:
            front = self._clamp_pressure_value("front", front_override)
        if rear_override is not None:
            rear = self._clamp_pressure_value("rear", rear_override)
        return self._compute_tyre_pressure_effects(front, rear)

    def _load_suspension_setup_config(self):
        cfg = (getattr(self, "cfg", {}) or {}).get("car_setup", {}) or {}
        raw = cfg.get("suspension_setup", cfg.get("suspension", {}))
        if not isinstance(raw, dict):
            raw = {}
        base = dict(self.DEFAULT_SUSPENSION_SETUP_CFG)
        for key in ("min_level", "max_level", "neutral_level"):
            if key in raw:
                try:
                    base[key] = int(raw.get(key))
                except Exception:
                    pass
        for effect_key in ("min_effects", "max_effects"):
            merged = dict(base[effect_key])
            eff_src = raw.get(effect_key, {})
            if isinstance(eff_src, dict):
                for name in merged.keys():
                    if name in eff_src:
                        try:
                            merged[name] = float(eff_src.get(name))
                        except Exception:
                            pass
            base[effect_key] = merged
        min_level = int(base.get("min_level", 1))
        max_level = int(base.get("max_level", 11))
        if max_level < min_level:
            min_level, max_level = max_level, min_level
        neutral = max(min_level, min(max_level, int(base.get("neutral_level", 6))))
        base["min_level"] = min_level
        base["max_level"] = max_level
        base["neutral_level"] = neutral
        self.suspension_setup_cfg = base

    def _clamp_suspension_level(self, value: int) -> int:
        cfg = getattr(self, "suspension_setup_cfg", self.DEFAULT_SUSPENSION_SETUP_CFG)
        lo = int(cfg.get("min_level", 1))
        hi = int(cfg.get("max_level", 11))
        if hi < lo:
            lo, hi = hi, lo
        try:
            out = int(round(float(value)))
        except Exception:
            out = int(cfg.get("neutral_level", lo))
        return max(lo, min(hi, out))

    def _init_driver_suspension_setup(self, initial_setup) -> None:
        cfg = self.suspension_setup_cfg
        default_level = int(cfg["neutral_level"])
        if isinstance(initial_setup, dict):
            store = initial_setup
        else:
            store = {}
        self.suspension_setup_by_driver = store
        self._suspension_setup_effects_by_driver = {}
        for d in self.drivers:
            rec = store.get(d.name)
            if not isinstance(rec, dict):
                rec = {}
            try:
                level = int(rec.get("level", default_level))
            except Exception:
                level = default_level
            level = self._clamp_suspension_level(level)
            store[d.name] = {"level": level}
            self._suspension_setup_effects_by_driver[d.name] = self._compute_suspension_setup_effects(level)

    def get_suspension_setup(self, driver_name: str) -> int:
        cfg = self.suspension_setup_cfg
        default_level = int(cfg["neutral_level"])
        store = getattr(self, "suspension_setup_by_driver", {}) or {}
        rec = store.get(driver_name, {}) if isinstance(store, dict) else {}
        try:
            level = int(rec.get("level", default_level))
        except Exception:
            level = default_level
        return self._clamp_suspension_level(level)

    def set_suspension_setup(self, driver_name: str, level: Optional[int] = None) -> bool:
        if self.tick_count > 0 or any(self.laps.values()):
            return False
        store = getattr(self, "suspension_setup_by_driver", None)
        if not isinstance(store, dict):
            store = {}
            self.suspension_setup_by_driver = store
        cur_level = self.get_suspension_setup(driver_name)
        if level is None:
            level = cur_level
        level = self._clamp_suspension_level(level)
        if level == cur_level:
            return True
        store[driver_name] = {"level": level}
        self._suspension_setup_effects_by_driver[driver_name] = self._compute_suspension_setup_effects(level)
        self._refresh_driver_setup_caches(driver_name)
        return True

    def _compute_suspension_setup_effects(self, level: int) -> dict:
        cfg = self.suspension_setup_cfg
        neutral = int(cfg.get("neutral_level", 6))
        low = int(cfg.get("min_level", neutral))
        high = int(cfg.get("max_level", neutral))
        if level < neutral and neutral > low:
            ratio = (neutral - int(level)) / float(neutral - low)
            src = cfg.get("min_effects", {})
        elif level > neutral and high > neutral:
            ratio = (int(level) - neutral) / float(high - neutral)
            src = cfg.get("max_effects", {})
        else:
            ratio = 0.0
            src = {}
        ratio = max(0.0, min(1.0, float(ratio)))
        out = {"slow": 0.0, "med": 0.0, "high": 0.0, "straight": 0.0, "level": int(level)}
        for key in ("slow", "med", "high", "straight"):
            try:
                out[key] = float(src.get(key, 0.0) or 0.0) * ratio
            except Exception:
                out[key] = 0.0
        return out

    def suspension_setup_effects(self, driver_name: str, *, level_override: Optional[int] = None) -> dict:
        if level_override is None:
            cached = (getattr(self, "_suspension_setup_effects_by_driver", {}) or {}).get(driver_name)
            if isinstance(cached, dict):
                return cached
        level = self.get_suspension_setup(driver_name)
        if level_override is not None:
            level = self._clamp_suspension_level(level_override)
        return self._compute_suspension_setup_effects(level)

    def wing_setup_attr_delta(
        self,
        driver_name: str,
        *,
        front_override: Optional[int] = None,
        rear_override: Optional[int] = None,
    ) -> dict:
        # Returns explicit km/h setup effects by zone.
        if front_override is None and rear_override is None:
            cached = (getattr(self, "_wing_setup_kmh_delta_by_driver", {}) or {}).get(driver_name)
            if isinstance(cached, dict):
                return cached
        cfg = self.wing_setup_cfg
        front, rear = self.get_wing_setup(driver_name)
        if front_override is not None:
            try:
                front = int(front_override)
            except Exception:
                pass
        if rear_override is not None:
            try:
                rear = int(rear_override)
            except Exception:
                pass
        front = max(int(cfg["min_level"]), min(int(cfg["max_level"]), int(front)))
        rear = max(int(cfg["min_level"]), min(int(cfg["max_level"]), int(rear)))
        return self._compute_wing_setup_attr_delta(front, rear)

    def _compute_wing_setup_attr_delta(self, front: int, rear: int) -> dict:
        cfg = self.wing_setup_cfg
        neutral = int(cfg["neutral_level"])
        front_clicks = int(front - neutral)
        rear_clicks = int(rear - neutral)
        front_map = cfg.get("front_per_click_kmh", {}) or {}
        rear_map = cfg.get("rear_per_click_kmh", {}) or {}
        out = {"slow": 0.0, "med": 0.0, "high": 0.0, "straight": 0.0}
        for key in out.keys():
            out[key] = (
                float(front_map.get(key, 0.0)) * float(front_clicks)
                + float(rear_map.get(key, 0.0)) * float(rear_clicks)
            )
        dim_cfg = cfg.get("diminishing_returns", {}) or {}
        if bool(dim_cfg.get("enabled", True)):
            df = float(front_clicks)
            dr = float(rear_clicks)
            dist = math.sqrt((df * df) + (dr * dr))
            coeff = float(dim_cfg.get("coefficient", 0.025) or 0.0)
            expo = float(dim_cfg.get("exponent", 1.35) or 1.35)
            min_eff = float(dim_cfg.get("min_efficiency", 0.70) or 0.70)
            max_eff = float(dim_cfg.get("max_efficiency", 1.0) or 1.0)
            if max_eff < min_eff:
                min_eff, max_eff = max_eff, min_eff
            raw_eff = 1.0 - (coeff * (dist ** expo if dist > 0.0 else 0.0))
            eff = max(min_eff, min(max_eff, raw_eff))
            for key in out.keys():
                out[key] = float(out[key]) * float(eff)
        return out

    def _assemble_realistic_lap_inputs_numeric(
        self,
        *,
        slow_points: float,
        med_points: float,
        high_points: float,
        straight_points: float,
        team_pace_points: float,
        braking_points: float,
        acceleration_points: float,
        setup_kmh: dict,
        pressure_fx: dict,
        suspension_fx: dict,
        driver_cornering: float,
        driver_braking: float,
        weekend_cornering_variability_kmh: float,
        corner_nerf: float,
        brake_nerf: float,
        part_corner_nerf: float,
        aero_corner_nerf: float,
        aero_brake_nerf: float,
        aero_corner_trait_delta: float,
        aero_brake_trait_mult: float,
        aero_straight_delta_s: float,
    ) -> tuple[float, ...]:
        delta_seconds_per_rating = float(DELTA_SECONDS_PER_RATING)
        accel_ms2_per_rating = float(ACCEL_MS2_PER_RATING)
        if _REALISTIC_KERNELS is not None:
            try:
                return tuple(
                    _REALISTIC_KERNELS.assemble_lap_inputs_numeric(
                        float(slow_points),
                        float(med_points),
                        float(high_points),
                        float(straight_points),
                        float(team_pace_points),
                        float(braking_points),
                        float(acceleration_points),
                        float(setup_kmh.get("slow", 0.0) or 0.0),
                        float(setup_kmh.get("med", 0.0) or 0.0),
                        float(setup_kmh.get("high", 0.0) or 0.0),
                        float(setup_kmh.get("straight", 0.0) or 0.0),
                        float(pressure_fx.get("slow", 0.0) or 0.0),
                        float(pressure_fx.get("med", 0.0) or 0.0),
                        float(pressure_fx.get("high", 0.0) or 0.0),
                        float(pressure_fx.get("straight", 0.0) or 0.0),
                        float(pressure_fx.get("accel_ms2", 0.0) or 0.0),
                        float(suspension_fx.get("slow", 0.0) or 0.0),
                        float(suspension_fx.get("med", 0.0) or 0.0),
                        float(suspension_fx.get("high", 0.0) or 0.0),
                        float(suspension_fx.get("straight", 0.0) or 0.0),
                        float(ATTR_KMH_PER_RATING.get("low", 0.0)),
                        float(ATTR_KMH_PER_RATING.get("med", 0.0)),
                        float(ATTR_KMH_PER_RATING.get("high", 0.0)),
                        float(UNIVERSAL_KMH_PER_RATING.get("straight", 0.0)),
                        float(self.driver_cornering_kmh_per_point),
                        float(driver_cornering),
                        float(weekend_cornering_variability_kmh),
                        float(self.driver_braking_ms2_per_point),
                        float(driver_braking),
                        float(corner_nerf),
                        float(brake_nerf),
                        float(part_corner_nerf),
                        float(aero_corner_nerf),
                        float(aero_brake_nerf),
                        float(aero_corner_trait_delta),
                        float(aero_brake_trait_mult),
                        float(aero_straight_delta_s),
                        float(delta_seconds_per_rating),
                        float(accel_ms2_per_rating),
                    )
                )
            except Exception:
                pass

        slow_points += self._kmh_to_rating_points("slow", setup_kmh.get("slow", 0.0))
        med_points += self._kmh_to_rating_points("med", setup_kmh.get("med", 0.0))
        high_points += self._kmh_to_rating_points("high", setup_kmh.get("high", 0.0))
        straight_points += self._kmh_to_rating_points("straight", setup_kmh.get("straight", 0.0))
        slow_points += self._kmh_to_rating_points("slow", pressure_fx.get("slow", 0.0))
        med_points += self._kmh_to_rating_points("med", pressure_fx.get("med", 0.0))
        high_points += self._kmh_to_rating_points("high", pressure_fx.get("high", 0.0))
        straight_points += self._kmh_to_rating_points("straight", pressure_fx.get("straight", 0.0))
        slow_points += self._kmh_to_rating_points("slow", suspension_fx.get("slow", 0.0))
        med_points += self._kmh_to_rating_points("med", suspension_fx.get("med", 0.0))
        high_points += self._kmh_to_rating_points("high", suspension_fx.get("high", 0.0))
        straight_points += self._kmh_to_rating_points("straight", suspension_fx.get("straight", 0.0))

        base_corner_bonus = float(self.driver_cornering_kmh_per_point) * driver_cornering
        base_corner_bonus += float(weekend_cornering_variability_kmh)
        base_brake_bonus = float(self.driver_braking_ms2_per_point) * driver_braking
        effective_corner_bonus = base_corner_bonus - corner_nerf - part_corner_nerf
        effective_corner_bonus -= aero_corner_nerf
        effective_corner_bonus += aero_corner_trait_delta
        effective_brake_bonus = (
            base_brake_bonus * aero_brake_trait_mult
        ) - brake_nerf - aero_brake_nerf
        aero_straight_points = self._delta_seconds_to_points(aero_straight_delta_s)
        effective_straight_points = float(straight_points) + float(aero_straight_points)
        slow_delta_equiv = self._points_to_delta_seconds(slow_points)
        med_delta_equiv = self._points_to_delta_seconds(med_points)
        high_delta_equiv = self._points_to_delta_seconds(high_points)
        straight_delta_equiv = self._points_to_delta_seconds(effective_straight_points)
        team_pace_delta_equiv = self._points_to_delta_seconds(team_pace_points)
        braking_delta_equiv = self._points_to_delta_seconds(braking_points)
        acceleration_points += (
            float(pressure_fx.get("accel_ms2", 0.0) or 0.0) / float(ACCEL_MS2_PER_RATING)
        )
        acceleration_delta_equiv = self._points_to_delta_seconds(acceleration_points)
        return (
            float(slow_points),
            float(med_points),
            float(high_points),
            float(effective_straight_points),
            float(team_pace_points),
            float(braking_points),
            float(acceleration_points),
            float(effective_corner_bonus),
            float(effective_brake_bonus),
            float(slow_delta_equiv),
            float(med_delta_equiv),
            float(high_delta_equiv),
            float(straight_delta_equiv),
            float(team_pace_delta_equiv),
            float(braking_delta_equiv),
            float(acceleration_delta_equiv),
        )

    def _begin_step_lap_input_cache(self) -> None:
        self._realistic_step_base_lap_inputs_cache = {}

    def _clear_step_lap_input_cache(self) -> None:
        self._realistic_step_base_lap_inputs_cache = None

    def _build_realistic_static_input_cache(self) -> None:
        self._realistic_static_input_cache = {}
        self._driver_style_mismatch_cache = {}
        for driver in self.drivers:
            try:
                self._realistic_static_inputs_for_driver(driver)
            except Exception:
                continue

    def _realistic_static_inputs_for_driver(self, driver) -> dict:
        driver_name = getattr(driver, "name", "")
        cached = (getattr(self, "_realistic_static_input_cache", {}) or {}).get(driver_name)
        if isinstance(cached, dict):
            return cached

        team_name = getattr(driver, "team", None)
        team_attr = self.team_attrs.get(team_name, {}) if team_name is not None else {}
        driver_attr = self.driver_attrs_by_driver.get(driver_name)
        if isinstance(driver_attr, dict):
            team_attr = driver_attr

        def _float_from(mapping, key, default=0.0):
            try:
                return float((mapping or {}).get(key, default))
            except Exception:
                return float(default)

        slow_points = _float_from(team_attr, "slow")
        med_points = _float_from(team_attr, "med")
        high_points = _float_from(team_attr, "high")
        corner_points = _float_from(team_attr, "corner")
        straight_points = _float_from(team_attr, "straight")
        braking_points = _float_from(team_attr, "braking")
        acceleration_points = _float_from(team_attr, "acceleration")
        try:
            team_pace_points = float(self.team_paces.get(team_name, 0.0))
        except Exception:
            team_pace_points = 0.0
        try:
            team_pace_points += float(
                self.driver_team_pace_rating_points_by_driver.get(driver_name, 0.0) or 0.0
            )
        except Exception:
            pass
        try:
            engine_power_rating = float(self.engine_power_rating_by_driver.get(driver_name, 50.0))
        except Exception:
            engine_power_rating = 50.0
        try:
            car_mass_kg = float(team_attr.get("car_mass_kg", 650.0))
        except Exception:
            car_mass_kg = 650.0
        try:
            engine_mass_kg = float(self.engine_mass_kg_by_driver.get(driver_name, 150.0))
        except Exception:
            engine_mass_kg = 150.0
        try:
            if hasattr(driver, "effective_rating"):
                driver_cornering = max(0.0, float(driver.effective_rating("cornering")))
            else:
                driver_cornering = max(0.0, float(getattr(driver, "cornering", 0.0)))
        except Exception:
            driver_cornering = 0.0
        try:
            if hasattr(driver, "effective_rating"):
                driver_braking = max(0.0, float(driver.effective_rating("braking")))
            else:
                driver_braking = max(0.0, float(getattr(driver, "braking", 0.0)))
        except Exception:
            driver_braking = 0.0
        try:
            oval_discipline_rating = float(
                team_attr.get(
                    "oval_driver_discipline_rating",
                    driver.active_discipline_rating()
                    if callable(getattr(driver, "active_discipline_rating", None))
                    else 10.0,
                )
            )
        except Exception:
            oval_discipline_rating = 10.0
        if "oval_driver_discipline_rating" in team_attr:
            # Oval track-type spread compression is carried with the entrant
            # inputs. Formula dictionaries do not contain this key.
            driver_cornering = max(0.0, oval_discipline_rating)
            driver_braking = max(0.0, oval_discipline_rating)

        normalized_comp_by_comp = {}
        supplier_pace_by_comp = {}
        default_tyre_temp_by_comp = {}
        compounds = []
        try:
            compounds = list(getattr(self, "available_compounds", []) or [])
        except Exception:
            compounds = []
        if not compounds:
            try:
                compounds = list(self.tyre_model.compound_names() or [])
            except Exception:
                compounds = []
        current_comp = None
        try:
            current_comp = self.tyre_comp.get(driver_name, self.tyre_model.default_compound)
        except Exception:
            current_comp = None
        if current_comp and current_comp not in compounds:
            compounds.append(current_comp)
        for comp in compounds:
            try:
                normalized = self.tyre_model.normalize_compound_name(comp)
            except Exception:
                normalized = str(comp).lower().strip()
            normalized_comp_by_comp[comp] = normalized
            try:
                supplier_pace_by_comp[normalized] = self._driver_supplier_ratings(driver_name, normalized)[0]
            except Exception:
                supplier_pace_by_comp[normalized] = 50.0
            try:
                default_tyre_temp_by_comp[comp] = float(self.tyre_model.initial_temperature_c(comp, pit_out=False))
            except Exception:
                default_tyre_temp_by_comp[comp] = 0.0

        try:
            temp_window_shift_c = float(self.driver_tyre_temp_window_shift_c(driver_name))
        except Exception:
            temp_window_shift_c = 0.0
        try:
            contract_grip_bonus_mult = float(self._driver_tyre_grip_bonus_mult(driver_name))
        except Exception:
            contract_grip_bonus_mult = 1.0
        style_concept_key = None
        style_base_feel = None
        try:
            style_concept_key = fitted_chassis_concept_key(getattr(self, "state", None), team_name, driver_name)
            style_base_feel = car_concept_feel(getattr(self, "state", None), style_concept_key)
        except Exception:
            style_concept_key = None
            style_base_feel = None

        out = {
            "team_name": team_name,
            "slow_points": float(slow_points),
            "med_points": float(med_points),
            "high_points": float(high_points),
            "corner_points": float(corner_points),
            "straight_points": float(straight_points),
            "braking_points": float(braking_points),
            "acceleration_points": float(acceleration_points),
            "team_pace_points": float(team_pace_points),
            "engine_power_rating": float(engine_power_rating),
            "car_mass_kg": float(car_mass_kg),
            "engine_mass_kg": float(engine_mass_kg),
            "driver_cornering": float(driver_cornering),
            "driver_braking": float(driver_braking),
            "oval_driver_discipline_rating": float(oval_discipline_rating),
            "weekend_cornering_variability_kmh": float(
                self.weekend_cornering_variability_kmh_by_driver.get(driver_name, 0.0) or 0.0
            ),
            "setup_kmh": self.wing_setup_attr_delta(driver_name),
            "pressure_fx": self.tyre_pressure_effects(driver_name),
            "suspension_fx": self.suspension_setup_effects(driver_name),
            "normalized_comp_by_comp": normalized_comp_by_comp,
            "supplier_pace_by_comp": supplier_pace_by_comp,
            "default_tyre_temp_by_comp": default_tyre_temp_by_comp,
            "temp_window_shift_c": float(temp_window_shift_c),
            "contract_grip_bonus_mult": float(contract_grip_bonus_mult),
            "style_fitted_chassis_concept": style_concept_key,
            "style_base_feel": dict(style_base_feel) if isinstance(style_base_feel, dict) else None,
        }
        self._realistic_static_input_cache[driver_name] = out
        return out

    def _realistic_base_lap_inputs_for_driver(
        self,
        driver,
        apply_consistency: bool = True,
        apply_aero: bool = True,
    ) -> LapInputs:
        driver_name = getattr(driver, "name", "")
        step_cache = getattr(self, "_realistic_step_base_lap_inputs_cache", None)
        step_key = (driver_name, bool(apply_consistency), bool(apply_aero))
        if isinstance(step_cache, dict):
            cached_step = step_cache.get(step_key)
            if isinstance(cached_step, LapInputs):
                return cached_step
        static_inputs = self._realistic_static_inputs_for_driver(driver)
        team_name = static_inputs.get("team_name", getattr(driver, "team", None))
        slow_points = float(static_inputs.get("slow_points", 0.0))
        med_points = float(static_inputs.get("med_points", 0.0))
        high_points = float(static_inputs.get("high_points", 0.0))
        straight_points = float(static_inputs.get("straight_points", 0.0))
        setup_kmh = static_inputs.get("setup_kmh", {}) or {}
        wing_damage_kmh = self.front_wing_damage_losses(driver_name)
        if wing_damage_kmh:
            setup_kmh = dict(setup_kmh)
            setup_kmh["slow"] = float(setup_kmh.get("slow", 0.0) or 0.0) - float(wing_damage_kmh.get("slow", 0.0) or 0.0)
            setup_kmh["med"] = float(setup_kmh.get("med", 0.0) or 0.0) - float(wing_damage_kmh.get("med", 0.0) or 0.0)
            setup_kmh["high"] = float(setup_kmh.get("high", 0.0) or 0.0) - float(wing_damage_kmh.get("high", 0.0) or 0.0)
        pressure_fx = static_inputs.get("pressure_fx", {}) or {}
        suspension_fx = static_inputs.get("suspension_fx", {}) or {}
        style_fx = self._driver_style_mismatch_effects(driver_name, driver)
        style_corner_loss = float(style_fx.get("corner_kmh_loss", 0.0) or 0.0)
        if style_corner_loss > 0.0:
            setup_kmh = dict(setup_kmh)
            setup_kmh["slow"] = float(setup_kmh.get("slow", 0.0) or 0.0) - style_corner_loss
            setup_kmh["med"] = float(setup_kmh.get("med", 0.0) or 0.0) - style_corner_loss
            setup_kmh["high"] = float(setup_kmh.get("high", 0.0) or 0.0) - style_corner_loss
        braking_points = float(static_inputs.get("braking_points", 0.0))
        acceleration_points = float(static_inputs.get("acceleration_points", 0.0))
        team_pace_points = float(static_inputs.get("team_pace_points", 0.0))
        engine_power_rating = float(static_inputs.get("engine_power_rating", 50.0))
        engine_power_rating = self._effective_engine_power_rating(driver_name, engine_power_rating)
        car_mass_kg = float(static_inputs.get("car_mass_kg", 650.0))
        engine_mass_kg = float(static_inputs.get("engine_mass_kg", 150.0))
        try:
            fuel_mass_kg = float(self.fuel_onboard.get(driver_name, 0.0))
        except Exception:
            fuel_mass_kg = 0.0
        driver_cornering = float(static_inputs.get("driver_cornering", 0.0))
        driver_braking = float(static_inputs.get("driver_braking", 0.0))
        corner_points = float(static_inputs.get("corner_points", 0.0))
        weekend_cornering_variability_kmh = float(
            static_inputs.get("weekend_cornering_variability_kmh", 0.0)
        )
        pace_mode_corner_bonus = float(self.driver_pace_modes.cornering_bonus_kmh(driver_name))
        corner_nerf = 0.0
        brake_nerf = 0.0
        aero_corner_nerf = 0.0
        aero_brake_nerf = 0.0
        aero_corner_trait_delta = 0.0
        aero_brake_trait_mult = 1.0
        aero_straight_delta_s = 0.0
        part_corner_nerf = max(
            0.0,
            float(self.part_cornering_wear_nerf_kmh.get(driver_name, 0.0) or 0.0),
        )
        if apply_consistency and self.use_realistic_physics:
            info = self._realistic_consistency_state.get(driver_name, {})
            try:
                corner_nerf = max(0.0, float(info.get("corner_nerf_kmh", 0.0) or 0.0))
            except Exception:
                corner_nerf = 0.0
            try:
                brake_nerf = max(0.0, float(info.get("brake_nerf_ms2", 0.0) or 0.0))
            except Exception:
                brake_nerf = 0.0
        if apply_aero and self.use_realistic_physics:
            aero_info = self._realistic_aero_state.get(driver_name, {})
            try:
                aero_corner_nerf = max(
                    0.0, float(aero_info.get("dirty_corner_nerf_kmh", 0.0) or 0.0)
                )
            except Exception:
                aero_corner_nerf = 0.0
            try:
                aero_brake_nerf = max(
                    0.0, float(aero_info.get("dirty_brake_nerf_ms2", 0.0) or 0.0)
                )
            except Exception:
                aero_brake_nerf = 0.0
            try:
                aero_corner_trait_delta = float(
                    aero_info.get("trait_corner_kmh_delta", 0.0) or 0.0
                )
            except Exception:
                aero_corner_trait_delta = 0.0
            try:
                aero_brake_trait_mult = max(
                    0.1,
                    float(aero_info.get("trait_brake_mult", 1.0) or 1.0),
                )
            except Exception:
                aero_brake_trait_mult = 1.0
            try:
                raw_slip_delta = float(
                    aero_info.get("slip_straight_delta_s", 0.0) or 0.0
                )
            except Exception:
                raw_slip_delta = 0.0
            aero_straight_delta_s = self._smoothed_slip_delta_seconds(
                driver_name,
                raw_slip_delta,
            )
        native_superspeedway_aero = bool(
            self._physics_series_mode == "oval"
            and self._oval_track_type == "superspeedway"
            and bool(
                getattr(
                    getattr(self, "oval_racecraft", None),
                    "superspeedway_profile_active",
                    False,
                )
            )
        )
        if native_superspeedway_aero:
            # Superspeedway slipstream and dirty air are transient physical
            # speed deltas in OvalRacecraftModel.  Do not also feed them
            # through Formula's rating-point compatibility path.
            aero_corner_nerf = 0.0
            aero_brake_nerf = 0.0
            aero_straight_delta_s = 0.0
        (
            slow_points,
            med_points,
            high_points,
            effective_straight_points,
            team_pace_points,
            braking_points,
            acceleration_points,
            effective_corner_bonus,
            effective_brake_bonus,
            slow_delta_equiv,
            med_delta_equiv,
            high_delta_equiv,
            straight_delta_equiv,
            team_pace_delta_equiv,
            braking_delta_equiv,
            acceleration_delta_equiv,
        ) = self._assemble_realistic_lap_inputs_numeric(
            slow_points=slow_points,
            med_points=med_points,
            high_points=high_points,
            straight_points=straight_points,
            team_pace_points=team_pace_points,
            braking_points=braking_points,
            acceleration_points=acceleration_points,
            setup_kmh=setup_kmh,
            pressure_fx=pressure_fx,
            suspension_fx=suspension_fx,
            driver_cornering=driver_cornering,
            driver_braking=driver_braking,
            weekend_cornering_variability_kmh=weekend_cornering_variability_kmh,
            corner_nerf=corner_nerf,
            brake_nerf=brake_nerf,
            part_corner_nerf=part_corner_nerf,
            aero_corner_nerf=aero_corner_nerf,
            aero_brake_nerf=aero_brake_nerf,
            aero_corner_trait_delta=aero_corner_trait_delta,
            aero_brake_trait_mult=aero_brake_trait_mult,
            aero_straight_delta_s=aero_straight_delta_s,
        )
        effective_corner_bonus = float(effective_corner_bonus) + float(pace_mode_corner_bonus)
        tyre_comp_map = getattr(self, "tyre_comp", {})
        comp = tyre_comp_map.get(driver_name, self.tyre_model.default_compound)
        tyre_wear_map = getattr(self, "tyre_wear", {})
        wear = float(tyre_wear_map.get(driver_name, 0.0))
        tyre_lat_wear = wear
        tyre_long_wear = wear
        if self.oval_four_tyre_enabled:
            tyre_lat_wear, tyre_long_wear = self._oval_effective_tyre_wear(
                driver_name
            )
        wetness = self.wetness_for_driver(driver_name)
        tyre_temp_map = getattr(self, "tyre_temp", {})
        temp_value = tyre_temp_map.get(driver_name)
        if temp_value is None:
            temp_value = (static_inputs.get("default_tyre_temp_by_comp", {}) or {}).get(comp)
        if temp_value is None:
            temp_value = self.tyre_model.initial_temperature_c(comp, pit_out=False)
        tyre_temp = float(temp_value)
        normalized_comp = (static_inputs.get("normalized_comp_by_comp", {}) or {}).get(comp)
        if not normalized_comp:
            try:
                normalized_comp = self.tyre_model.normalize_compound_name(comp)
            except Exception:
                normalized_comp = str(comp).lower().strip()
        supplier_pace_rating = (static_inputs.get("supplier_pace_by_comp", {}) or {}).get(normalized_comp)
        if supplier_pace_rating is None:
            supplier_pace_rating, _ = self._driver_supplier_ratings(driver_name, normalized_comp)
        contract_grip_bonus_mult = float(static_inputs.get("contract_grip_bonus_mult", 1.0))
        contract_grip_bonus_mult *= rubber_grip_multiplier(
            getattr(self, "track_rubber", 0.0),
            getattr(self, "track_rubber_grip_gain", 0.0),
        )
        mass_bin = max(0.5, float(MASS_CACHE_BIN_KG))
        wear_quantum = max(0.001, float(getattr(self.tyre_model, "wear_quantum", 0.01)))
        cornering_cache_value = float(driver_cornering)
        if float(self.driver_cornering_kmh_per_point) > 1e-9:
            cornering_cache_value += float(pace_mode_corner_bonus) / float(self.driver_cornering_kmh_per_point)
        if _REALISTIC_KERNELS is not None:
            try:
                cache_key = _REALISTIC_KERNELS.build_lap_inputs_cache_key(
                    team_name,
                    float(slow_delta_equiv),
                    float(med_delta_equiv),
                    float(high_delta_equiv),
                    float(straight_delta_equiv),
                    float(team_pace_delta_equiv),
                    float(braking_delta_equiv),
                    float(acceleration_delta_equiv),
                    float(engine_power_rating),
                    float(car_mass_kg),
                    float(engine_mass_kg),
                    float(fuel_mass_kg),
                    float(cornering_cache_value),
                    float(driver_braking),
                    normalized_comp,
                    float(wear),
                    float(wetness),
                    float(tyre_temp),
                    float(self.mass_ref_kg),
                    float(self.driver_cornering_kmh_per_point),
                    float(self.driver_braking_ms2_per_point),
                    bool(apply_consistency),
                    float(corner_nerf),
                    float(brake_nerf),
                    float(part_corner_nerf),
                    float(aero_corner_nerf),
                    float(aero_brake_nerf),
                    float(aero_corner_trait_delta),
                    float(aero_brake_trait_mult),
                    float(aero_straight_delta_s),
                    float(supplier_pace_rating),
                    float(contract_grip_bonus_mult),
                    float(mass_bin),
                    float(wear_quantum),
                )
            except Exception:
                fuel_bucket = round(max(0.0, fuel_mass_kg) / mass_bin) * mass_bin
                wear_bucket = round(max(0.0, wear) / wear_quantum) * wear_quantum
                cache_key = (
                    team_name,
                    round(slow_delta_equiv, 3),
                    round(med_delta_equiv, 3),
                    round(high_delta_equiv, 3),
                    round(straight_delta_equiv, 3),
                    round(team_pace_delta_equiv, 3),
                    round(braking_delta_equiv, 3),
                    round(acceleration_delta_equiv, 3),
                    round(engine_power_rating, 3),
                    round(max(300.0, car_mass_kg), 3),
                    round(max(50.0, engine_mass_kg), 3),
                    round(fuel_bucket, 3),
                    round(cornering_cache_value, 3),
                    round(driver_braking, 3),
                    normalized_comp,
                    round(wear_bucket, 3),
                    round(wetness, 3),
                    round(tyre_temp, 3),
                    round(float(self.mass_ref_kg), 3),
                    round(float(self.driver_cornering_kmh_per_point), 3),
                    round(float(self.driver_braking_ms2_per_point), 3),
                    bool(apply_consistency),
                    round(float(corner_nerf), 3),
                    round(float(brake_nerf), 3),
                    round(float(part_corner_nerf), 3),
                    round(float(aero_corner_nerf), 2),
                    round(float(aero_brake_nerf), 3),
                    round(float(aero_corner_trait_delta), 3),
                    round(float(aero_brake_trait_mult), 3),
                    round(float(aero_straight_delta_s), 3),
                    round(float(supplier_pace_rating), 3),
                    round(float(contract_grip_bonus_mult), 6),
                )
        else:
            fuel_bucket = round(max(0.0, fuel_mass_kg) / mass_bin) * mass_bin
            wear_bucket = round(max(0.0, wear) / wear_quantum) * wear_quantum
            cache_key = (
                team_name,
                round(slow_delta_equiv, 3),
                round(med_delta_equiv, 3),
                round(high_delta_equiv, 3),
                round(straight_delta_equiv, 3),
                round(team_pace_delta_equiv, 3),
                round(braking_delta_equiv, 3),
                round(acceleration_delta_equiv, 3),
                round(engine_power_rating, 3),
                round(max(300.0, car_mass_kg), 3),
                round(max(50.0, engine_mass_kg), 3),
                round(fuel_bucket, 3),
                round(driver_cornering, 3),
                round(driver_braking, 3),
                normalized_comp,
                round(wear_bucket, 3),
                round(wetness, 3),
                round(tyre_temp, 3),
                round(float(self.mass_ref_kg), 3),
                round(float(self.driver_cornering_kmh_per_point), 3),
                round(float(self.driver_braking_ms2_per_point), 3),
                bool(apply_consistency),
                round(float(corner_nerf), 3),
                round(float(brake_nerf), 3),
                round(float(part_corner_nerf), 3),
                round(float(aero_corner_nerf), 2),
                round(float(aero_brake_nerf), 3),
                round(float(aero_corner_trait_delta), 3),
                round(float(aero_brake_trait_mult), 3),
                round(float(aero_straight_delta_s), 3),
                round(float(supplier_pace_rating), 3),
                round(float(contract_grip_bonus_mult), 6),
            )
        if self.oval_four_tyre_enabled:
            cache_key = (
                cache_key,
                round(float(tyre_lat_wear), 4),
                round(float(tyre_long_wear), 4),
            )
        cached_entry = self._realistic_lap_inputs_cache.get(driver_name)
        if (
            isinstance(cached_entry, tuple)
            and len(cached_entry) == 2
            and cached_entry[0] == cache_key
            and isinstance(cached_entry[1], LapInputs)
        ):
            return cached_entry[1]
        if self.oval_four_tyre_enabled:
            tyre_lat_mult, _unused_long = self.tyre_model.grip_multipliers(
                comp,
                wetness_mm=wetness,
                wear=tyre_lat_wear,
                tyre_temp_c=tyre_temp,
                temp_window_shift_c=float(
                    static_inputs.get("temp_window_shift_c", 0.0)
                ),
                supplier_pace_rating=supplier_pace_rating,
                contract_grip_bonus_mult=contract_grip_bonus_mult,
            )
            _unused_lat, tyre_long_mult = self.tyre_model.grip_multipliers(
                comp,
                wetness_mm=wetness,
                wear=tyre_long_wear,
                tyre_temp_c=tyre_temp,
                temp_window_shift_c=float(
                    static_inputs.get("temp_window_shift_c", 0.0)
                ),
                supplier_pace_rating=supplier_pace_rating,
                contract_grip_bonus_mult=contract_grip_bonus_mult,
            )
        else:
            tyre_lat_mult, tyre_long_mult = self.tyre_model.grip_multipliers(
                comp,
                wetness_mm=wetness,
                wear=wear,
                tyre_temp_c=tyre_temp,
                temp_window_shift_c=float(
                    static_inputs.get("temp_window_shift_c", 0.0)
                ),
                supplier_pace_rating=supplier_pace_rating,
                contract_grip_bonus_mult=contract_grip_bonus_mult,
            )
        lap_inputs = LapInputs(
            slow_rating_points=float(slow_points),
            med_rating_points=float(med_points),
            high_rating_points=float(high_points),
            corner_rating_points=float(corner_points),
            straight_rating_points=float(effective_straight_points),
            team_pace_rating_points=float(team_pace_points),
            # Engine influence is handled via engine_power_rating in accel model.
            engine_pace_rating_points=0.0,
            braking_rating_points=float(braking_points),
            acceleration_rating_points=float(acceleration_points),
            engine_power_rating=float(max(0.0, min(100.0, engine_power_rating))),
            car_mass_kg=float(max(300.0, car_mass_kg)),
            engine_mass_kg=float(max(50.0, engine_mass_kg)),
            fuel_mass_kg=float(max(0.0, fuel_mass_kg)),
            mass_ref_kg=float(self.mass_ref_kg),
            driver_cornering_kmh_bonus=float(effective_corner_bonus),
            driver_braking_ms2_bonus=float(effective_brake_bonus),
            tyre_lat_mult=float(tyre_lat_mult),
            tyre_long_mult=float(tyre_long_mult),
            oval_driver_discipline_rating=(
                float(static_inputs.get("oval_driver_discipline_rating", 10.0))
            ),
            oval_driver_control_rating=(
                float(getattr(driver, "control", 10.0))
                if callable(getattr(driver, "active_discipline_rating", None))
                else 10.0
            ),
        )
        self._realistic_lap_inputs_cache[driver_name] = (cache_key, lap_inputs)
        if isinstance(step_cache, dict):
            step_cache[step_key] = lap_inputs
        return lap_inputs

    def _realistic_lap_inputs_for_driver(
        self,
        driver,
        apply_consistency: bool = True,
        apply_aero: bool = True,
        progress_override: Optional[float] = None,
        base_inputs: Optional[LapInputs] = None,
    ) -> LapInputs:
        if not isinstance(base_inputs, LapInputs):
            base_inputs = self._realistic_base_lap_inputs_for_driver(
                driver,
                apply_consistency=apply_consistency,
                apply_aero=apply_aero,
            )
        try:
            prog = (
                float(progress_override)
                if progress_override is not None
                else float(self.progress.get(getattr(driver, "name", ""), 0.0) or 0.0)
            )
        except Exception:
            prog = 0.0
        try:
            ers_bonus = float(
                self._ers_bonus_points_for_progress(
                    getattr(driver, "name", ""),
                    prog,
                )
            )
        except Exception:
            ers_bonus = 0.0
        if abs(float(ers_bonus)) <= 1e-9:
            return base_inputs
        return replace(
            base_inputs,
            acceleration_rating_points=float(base_inputs.acceleration_rating_points) + float(ers_bonus),
        )

    def _rebuild_realistic_base_laps(self) -> None:
        self.realistic_base_lap_by_driver = {}
        if not self.use_realistic_physics:
            return
        for driver in self.drivers:
            lap, _ = self.realistic_physics.compute_lap(
                self._realistic_lap_inputs_for_driver(
                    driver,
                    apply_consistency=False,
                    apply_aero=False,
                )
            )
            self.realistic_base_lap_by_driver[driver.name] = max(MIN_VALID_LAP_TIME, float(lap))

    @staticmethod
    def _normalize_sector_splits(raw_splits):
        # Target three sectors (S1/S2/S3): up to two custom split lines.
        return normalize_track_sector_splits(raw_splits, max_count=2)

    @staticmethod
    def _normalize_progress_marker(raw_value, default: float = 0.0) -> float:
        try:
            value = float(raw_value)
        except Exception:
            value = float(default)
        if math.isnan(value) or math.isinf(value):
            value = float(default)
        value = max(1e-6, min(0.999999, value))
        return float(value)

    @classmethod
    def _normalize_drs_zones(cls, raw_zones) -> list[dict]:
        out = []
        if not isinstance(raw_zones, list):
            return out
        for item in raw_zones:
            if not isinstance(item, dict):
                continue
            det = cls._normalize_progress_marker(
                item.get("detection_progress", item.get("detection", 0.0)),
                default=0.0,
            )
            start = cls._normalize_progress_marker(
                item.get("start_progress", item.get("start", det)),
                default=det,
            )
            end = cls._normalize_progress_marker(
                item.get("end_progress", item.get("end", start)),
                default=start,
            )
            out.append(
                {
                    "detection_progress": float(det),
                    "start_progress": float(start),
                    "end_progress": float(end),
                }
            )
        out.sort(key=lambda zone: float(zone.get("start_progress", 0.0)))
        return out

    @staticmethod
    def _progress_in_window(progress: float, start: float, end: float) -> bool:
        p = float(progress) % 1.0
        s = float(start) % 1.0
        e = float(end) % 1.0
        if abs(s - e) <= 1e-9:
            return False
        if s < e:
            return (p >= s) and (p < e)
        return (p >= s) or (p < e)

    @staticmethod
    def _segment_crosses_marker(start_prog: float, end_prog: float, marker: float) -> bool:
        try:
            s = float(start_prog)
            e = float(end_prog)
            m = float(marker)
        except Exception:
            return False
        return m > (s + 1e-9) and m <= (e + 1e-9)

    def _nearest_ahead_name_in_order(self, driver_name: str) -> Optional[str]:
        target = str(driver_name or "")
        if not target:
            return None
        prev_name = None
        for drv in self.order:
            name = str(getattr(drv, "name", "") or "")
            if not name:
                continue
            if name == target:
                return prev_name
            prev_name = name
        return None

    def _update_drs_detection_crossings(
        self,
        driver_name: str,
        start_prog: float,
        end_prog: float,
        *,
        segment_start_time: Optional[float] = None,
        progress_rate: Optional[float] = None,
    ) -> None:
        if not self._drs_enabled:
            return
        if self._drs_disabled_by_wet_track():
            return
        name = str(driver_name or "")
        if not name or name in self.finished or self.dnf.get(name, False):
            return
        try:
            lap_idx = int(self.laps.get(name, 0) or 0)
        except Exception:
            lap_idx = 0
        zone_laps = self._drs_zone_detection_lap.setdefault(name, {})
        zone_elig = self._drs_zone_eligible.setdefault(name, {})
        for zone_idx, zone in enumerate(self.drs_zones):
            try:
                det_prog = float(zone.get("detection_progress", 0.0))
            except Exception:
                det_prog = 0.0
            if not self._segment_crosses_marker(start_prog, end_prog, det_prog):
                continue
            if int(zone_laps.get(zone_idx, -1)) == lap_idx:
                continue
            zone_laps[zone_idx] = lap_idx
            # Clear any previous-lap result immediately. The chronologically
            # ordered crossing pass below will set this lap's eligibility once
            # every car in the simulation step has been observed.
            zone_elig[zone_idx] = False
            running_lap = lap_idx + 1
            try:
                rate = float(progress_rate) if progress_rate is not None else 0.0
            except (TypeError, ValueError):
                rate = 0.0
            try:
                crossing_time = float(
                    self.total_sim_time.get(name, 0.0)
                    if segment_start_time is None
                    else segment_start_time
                )
            except (TypeError, ValueError):
                crossing_time = 0.0
            if rate > 1e-12:
                crossing_time += max(0.0, det_prog - float(start_prog)) / rate
            pending = getattr(self, "_drs_pending_detection_crossings", None)
            if not isinstance(pending, list):
                pending = []
                self._drs_pending_detection_crossings = pending
            pending.append(
                (
                    float(crossing_time),
                    int(zone_idx),
                    str(name),
                    int(lap_idx),
                    int(running_lap),
                )
            )

    def _resolve_drs_detection_crossings(self) -> None:
        """Resolve DRS by physical detector order, independent of scored laps."""

        pending = getattr(self, "_drs_pending_detection_crossings", None)
        if not isinstance(pending, list) or not pending:
            return
        self._drs_pending_detection_crossings = []
        pending.sort(key=lambda item: (float(item[0]), int(item[1]), str(item[2])))
        last_by_zone = getattr(self, "_drs_last_detection_crossing", None)
        if not isinstance(last_by_zone, dict):
            last_by_zone = {}
            self._drs_last_detection_crossing = last_by_zone

        wet_disabled = bool(self._drs_disabled_by_wet_track())
        caution_active = bool(self.caution_active)
        threshold = max(0.0, float(self._drs_gap_threshold_s))
        for crossing_time, zone_idx, name, lap_idx, running_lap in pending:
            previous = last_by_zone.get(int(zone_idx))
            eligible = False
            if (
                self._drs_enabled
                and not wet_disabled
                and not caution_active
                and int(running_lap) >= int(self._drs_activation_lap)
                and isinstance(previous, tuple)
                and len(previous) >= 2
                and str(previous[1]) != str(name)
            ):
                try:
                    gap_s = float(crossing_time) - float(previous[0])
                except (TypeError, ValueError):
                    gap_s = float("inf")
                # The configured threshold remains authoritative. A tolerance
                # only absorbs floating-point interpolation at exactly 1.000s;
                # cars meaningfully beyond one second remain ineligible.
                if -1e-9 <= gap_s <= threshold + 1e-9:
                    eligible = True

            self._drs_zone_detection_lap.setdefault(str(name), {})[int(zone_idx)] = int(lap_idx)
            self._drs_zone_eligible.setdefault(str(name), {})[int(zone_idx)] = bool(eligible)
            if previous is None or float(crossing_time) >= float(previous[0]) - 1e-9:
                last_by_zone[int(zone_idx)] = (
                    float(crossing_time),
                    str(name),
                    int(lap_idx),
                )

    def _drs_boost_kmh_for_driver(self, driver_name: str) -> float:
        if not self._drs_enabled:
            return 0.0
        if self._drs_disabled_by_wet_track():
            return 0.0
        if self.caution_active:
            return 0.0
        name = str(driver_name or "")
        if not name:
            return 0.0
        try:
            lap_idx = int(self.laps.get(name, 0) or 0)
        except Exception:
            lap_idx = 0
        running_lap = lap_idx + 1
        if running_lap < int(self._drs_activation_lap):
            return 0.0
        progress = float(self.progress.get(name, 0.0) or 0.0)
        zone_laps = self._drs_zone_detection_lap.get(name, {})
        zone_elig = self._drs_zone_eligible.get(name, {})
        for zone_idx, zone in enumerate(self.drs_zones):
            det_lap = int(zone_laps.get(zone_idx, -1))
            if det_lap < 0:
                continue
            if not bool(zone_elig.get(zone_idx, False)):
                continue
            start = float(zone.get("start_progress", 0.0) or 0.0)
            end = float(zone.get("end_progress", 0.0) or 0.0)
            if not self._progress_in_window(progress, start, end):
                continue
            wraps_start_line = start > end
            if not wraps_start_line:
                if lap_idx != det_lap:
                    continue
                return float(self._drs_boost_kmh)
            # Wraparound zones are valid in two phases:
            # 1) [start..1.0) on the detection lap.
            # 2) [0.0..end) on the next lap.
            if progress >= start:
                if lap_idx != det_lap:
                    continue
                return float(self._drs_boost_kmh)
            if progress < end:
                if lap_idx != (det_lap + 1):
                    continue
                return float(self._drs_boost_kmh)
        return 0.0

    def _drs_accel_multiplier_from_kmh(self, drs_kmh: float) -> float:
        try:
            kmh = max(0.0, float(drs_kmh))
        except Exception:
            kmh = 0.0
        try:
            base = max(1.0, float(self._drs_accel_mult_base))
        except Exception:
            base = 1.0
        try:
            per_kmh = max(0.0, float(self._drs_accel_mult_per_kmh))
        except Exception:
            per_kmh = 0.0
        try:
            cap = max(1.0, float(self._drs_accel_mult_cap))
        except Exception:
            cap = 1.2
        mult = base + (kmh * per_kmh)
        if mult < 1.0:
            mult = 1.0
        if mult > cap:
            mult = cap
        return float(mult)

    def driver_drs_active(self, driver_name: str) -> bool:
        name = str(driver_name or "")
        if not name:
            return False
        if self._drs_disabled_by_wet_track():
            return False
        return bool(self._drs_active_now.get(name, False))

    def _drs_disabled_by_wet_track(self) -> bool:
        if not self._drs_enabled:
            return False
        try:
            threshold = float(self._drs_wet_disable_threshold_mm)
        except Exception:
            threshold = 1.0
        wetness = self.current_wetness()
        try:
            wet_value = float(wetness) if wetness is not None else 0.0
        except Exception:
            wet_value = 0.0
        return wet_value > threshold

    def _update_drs_wet_status_event(self) -> None:
        if not self._drs_enabled:
            return
        disabled = bool(self._drs_disabled_by_wet_track())
        previous = bool(getattr(self, "_drs_wet_disabled", False))
        if disabled == previous:
            return
        self._drs_wet_disabled = disabled
        if disabled:
            for name in list(self._drs_active_now.keys()):
                self._drs_active_now[name] = False
            self.events.append("RACE CONTROL: DRS disabled due to wet track")
        else:
            self.events.append("RACE CONTROL: DRS enabled")

    def _sector_index_for_progress(self, progress: float) -> int:
        try:
            p = float(progress)
        except Exception:
            p = 0.0
        p = max(0.0, min(0.999999, p))
        idx = 0
        for split in self.sector_splits:
            if p >= float(split):
                idx += 1
            else:
                break
        max_idx = max(0, len(self.sector_mixes) - 1)
        return max(0, min(idx, max_idx))

    def _record_sector_crossings_between(self, name, start_prog, end_prog, base_time, time_used, rate):
        if rate <= 0.0:
            return
        driver = self.driver_by_name.get(name)
        idx = int(self.current_sector_index.get(name, 0) or 0)
        if idx >= len(self.sector_splits):
            return
        cursor_prog = float(start_prog)
        cursor_offset = 0.0
        while idx < len(self.sector_splits):
            boundary = float(self.sector_splits[idx])
            if boundary <= cursor_prog + 1e-9:
                idx += 1
                self.current_sector_index[name] = idx
                continue
            if boundary > end_prog + 1e-9:
                break
            step = (boundary - cursor_prog) / rate
            if step < 0.0:
                break
            cursor_offset += step
            crossing_time = float(base_time) + float(time_used) + cursor_offset
            sector_time = max(0.0, crossing_time - float(self.current_sector_start_time.get(name, 0.0)))
            self.current_lap_sector_times.setdefault(name, []).append(sector_time)
            self.current_sector_start_time[name] = crossing_time
            if driver is not None:
                self._apply_sector_usage(driver, idx)
                self.driver_pace_modes.commit_pending(name)
                self.driver_engine_modes.commit_pending(name)
                self._evaluate_ai_race_pace_mode(name)
                self._evaluate_ai_race_engine_mode(name)
            idx += 1
            self.current_sector_index[name] = idx
            cursor_prog = boundary

    def _finalize_lap_sectors(self, name, crossing_time, lap_number):
        sectors = list(self.current_lap_sector_times.get(name, []))
        final_sector = max(0.0, float(crossing_time) - float(self.current_sector_start_time.get(name, 0.0)))
        sectors.append(final_sector)
        self.last_sector_times[name] = list(sectors)
        self.current_lap_sector_times[name] = []
        self.current_sector_index[name] = 0
        self.current_sector_start_time[name] = float(crossing_time)
        return sectors

    def set_starting_compound(self, driver_name: str, compound: str) -> bool:
        """Allow the human-controlled team to change the initial tyre choice."""

        if not compound or driver_name not in self.tyre_comp:
            return False
        if self.tick_count > 0 or any(self.laps.values()):
            return False
        if not self.player_team:
            return False
        if self.driver_team.get(driver_name) != self.player_team:
            return False

        available = list(getattr(self, "available_compounds", []))
        target = None
        if available:
            if compound in available:
                target = compound
            else:
                lookup = {str(opt).lower(): opt for opt in available}
                target = lookup.get(str(compound).lower())
        else:
            target = str(compound)

        if target is None:
            return False

        current = self.tyre_comp.get(driver_name)
        if current == target:
            return True

        replacement_set = None
        if (
            self._physics_series_mode != "oval"
            and self.weekend_tyre_manager is not None
            and self.weekend_tyre_manager.enabled()
            and self.weekend is not None
        ):
            replacement_set, _reason = self.weekend_tyre_manager.checkout(
                self.weekend, driver_name, target, "race"
            )
            if replacement_set is None:
                return False
            self.weekend_tyre_manager.release(
                self.weekend,
                driver_name,
                self.active_tyre_set_id.get(driver_name),
                self.tyre_wear.get(driver_name, 0.0),
                0,
            )
            self.active_tyre_set_id[driver_name] = replacement_set.get("set_id")
            self.active_tyre_set_start_lap[driver_name] = 0

        self.tyre_comp[driver_name] = target
        if driver_name in self.used_compounds:
            self.used_compounds[driver_name] = {target}
        else:
            self.used_compounds[driver_name] = {target}
        if driver_name in self.pending_compound:
            self.pending_compound[driver_name] = None
        if driver_name in self.manual_pit_requests:
            self.manual_pit_requests[driver_name] = None
        if driver_name in self.tyre_wear:
            self.tyre_wear[driver_name] = (
                float(replacement_set.get("wear", 0.0) or 0.0)
                if replacement_set is not None
                else 0.0
            )
        if self._physics_series_mode != "oval":
            self.tyre_stint_history[driver_name] = []
            self._start_tyre_stint_history(
                driver_name,
                start_lap=0.0,
                compound=target,
                start_wear=float(self.tyre_wear.get(driver_name, 0.0) or 0.0),
            )
        if self.oval_four_tyre_enabled:
            self.tyre_wear_by_corner[driver_name] = {
                key: 0.0 for key in OVAL_TYRE_KEYS
            }
        self.tyre_temp[driver_name] = float(
            self.tyre_model.initial_temperature_c(target, pit_out=False)
        )
        self._realistic_lap_inputs_cache.pop(driver_name, None)
        self._sync_lap_start_snapshot(driver_name)

        try:
            self.strategy.plan_initial_strategy(self)
        except Exception:
            pass

        return True

    def available_driver_pace_modes(self):
        return tuple(PACE_MODE_ORDER)

    def available_driver_control_modes(self):
        return ("manual", "automatic")

    def driver_control_mode(self, driver_name: str) -> str:
        try:
            automatic = bool(self.driver_pace_modes.automatic_control(driver_name))
        except Exception:
            automatic = False
        return "automatic" if automatic else "manual"

    def set_driver_control_mode(self, driver_name: str, mode: str) -> bool:
        name = str(driver_name or "")
        if not self.player_team or self.driver_team.get(name) != self.player_team:
            return False
        automatic = str(mode or "").strip().lower() == "automatic"
        ok = False
        try:
            ok = bool(self.driver_pace_modes.set_automatic_control(name, automatic))
        except Exception:
            ok = False
        try:
            self.driver_engine_modes.set_automatic_control(name, automatic)
        except Exception:
            pass
        try:
            self.driver_ers_modes.set_automatic_control(name, automatic)
        except Exception:
            pass
        if not ok:
            return False
        if automatic:
            self._evaluate_ai_race_pace_mode(name)
            self._evaluate_ai_race_engine_mode(name)
            self._evaluate_ai_race_ers_mode(name)
        else:
            try:
                self.driver_pace_modes.clear_pending(name)
            except Exception:
                pass
            try:
                self.driver_engine_modes.clear_pending(name)
            except Exception:
                pass
            try:
                self.driver_ers_modes.clear_pending(name)
            except Exception:
                pass
        return True

    def move_over_order_active(self, driver_name: str) -> bool:
        name = str(driver_name or "")
        order = getattr(self, "move_over_orders", {}).get(name) if name else None
        return bool(
            isinstance(order, dict)
            and str(order.get("phase") or "handover") != "settling"
        )

    def move_over_target_for(self, driver_name: str) -> Optional[str]:
        name = str(driver_name or "")
        order = getattr(self, "move_over_orders", {}).get(name)
        if isinstance(order, dict):
            target = order.get("target")
            return str(target) if target else None
        return None

    def _running_on_track(self, driver_name: str) -> bool:
        name = str(driver_name or "")
        if not name:
            return False
        if name in self.finished or self.dnf.get(name, False):
            return False
        if self.pit_remaining.get(name, 0.0) > 0.0:
            return False
        if self.freeze_remaining.get(name, 0.0) > 0.0:
            return False
        return True

    def _move_over_eligible_target(
        self,
        driver_name: str,
        max_gap_s: float = 1.0,
        *,
        require_player_team: bool = True,
        require_ai_team: bool = False,
    ) -> Optional[str]:
        name = str(driver_name or "")
        if not name:
            return None
        team = self.driver_team.get(name)
        if require_player_team and (not self.player_team or team != self.player_team):
            return None
        if require_ai_team and (not team or team == self.player_team):
            return None
        if self.caution_active or not self._running_on_track(name):
            return None
        try:
            idx = next(i for i, drv in enumerate(self.order) if getattr(drv, "name", None) == name)
        except StopIteration:
            return None
        if idx + 1 >= len(self.order):
            return None
        behind_name = str(getattr(self.order[idx + 1], "name", "") or "")
        if not behind_name or behind_name == name:
            return None
        if self.driver_team.get(behind_name) != self.driver_team.get(name):
            return None
        if not self._running_on_track(behind_name):
            return None
        try:
            lap_diff, gap_s = self.distance_reference_gap(behind_name, name)
        except Exception:
            return None
        if int(lap_diff or 0) != 0 or gap_s is None:
            return None
        try:
            gap_val = float(gap_s)
        except Exception:
            return None
        if 0.0 <= gap_val <= float(max_gap_s):
            return behind_name
        return None

    def can_request_move_over(self, driver_name: str) -> bool:
        return self._move_over_eligible_target(driver_name) is not None

    def _queue_move_over_order(self, driver_name: str, target_name: str, source: str = "player") -> bool:
        name = str(driver_name or "")
        target = str(target_name or "")
        if not name or not target:
            return False
        if name in self.move_over_orders:
            return False
        try:
            requested_sector = int(self._sector_index_for_progress(self.progress.get(name, 0.0)))
        except Exception:
            requested_sector = 0
        self.move_over_orders[name] = {
            "target": target,
            "requested_sector": requested_sector,
            "requested_lap": int(self.laps.get(name, 0) or 0),
            "armed": False,
            "phase": "handover",
            "source": str(source or "player"),
        }
        return True

    def toggle_move_over_order(self, driver_name: str) -> bool:
        name = str(driver_name or "")
        if not name:
            return False
        if name in self.move_over_orders:
            order = self.move_over_orders.get(name)
            if isinstance(order, dict) and str(order.get("phase") or "") == "settling":
                return False
            del self.move_over_orders[name]
            return True
        target = self._move_over_eligible_target(name)
        if not target:
            return False
        return self._queue_move_over_order(name, target, source="player")

    def _straightish_sector(self, sector_idx: int) -> bool:
        mixes = getattr(self, "sector_mixes", None)
        if not isinstance(mixes, list) or not mixes:
            return True
        try:
            idx = max(0, min(int(sector_idx), len(mixes) - 1))
            mix = mixes[idx] if isinstance(mixes[idx], dict) else {}
            straight = float(mix.get("straight", 0.0) or 0.0)
            slow = float(mix.get("slow", 0.0) or 0.0)
            med = float(mix.get("med", mix.get("medium", 0.0)) or 0.0)
            high = float(mix.get("high", 0.0) or 0.0)
        except Exception:
            return True
        return straight >= max(slow, med, high) or straight >= 0.35

    def move_over_attack_blocked(self, attacker_name: str, opponent_name: str) -> bool:
        """Prevent only the ordered driver's immediate pass-back attempt."""

        attacker = str(attacker_name or "")
        opponent = str(opponent_name or "")
        order = self.move_over_orders.get(attacker)
        return bool(
            attacker
            and opponent
            and isinstance(order, dict)
            and str(order.get("phase") or "handover") == "settling"
            and str(order.get("target") or "") == opponent
        )

    def _move_over_settle_release_reason(
        self,
        driver_name: str,
        target_name: str,
        positions: Optional[dict] = None,
    ) -> Optional[str]:
        """Return why a post-handover hold should end, if it should end now."""

        name = str(driver_name or "")
        target = str(target_name or "")
        order = self.move_over_orders.get(name)
        if not isinstance(order, dict) or str(order.get("phase") or "") != "settling":
            return "invalid"
        if (
            not target
            or not self._running_on_track(name)
            or not self._running_on_track(target)
            or self.driver_team.get(target) != self.driver_team.get(name)
        ):
            return "invalid"
        now = self._radio_sim_time()
        try:
            deadline = float(order.get("settle_deadline", now) or now)
        except Exception:
            deadline = now
        if now + 1e-9 >= deadline:
            return "timeout"

        if positions is None:
            positions = {
                getattr(drv, "name", None): idx for idx, drv in enumerate(self.order)
            }
        giver_pos = positions.get(name)
        target_pos = positions.get(target)
        if giver_pos is None or target_pos is None:
            return "invalid"

        # Only assess the intended clearance after the beneficiary is ahead.
        if target_pos < giver_pos:
            try:
                lap_diff, gap_s = self.distance_reference_gap(name, target)
                if (
                    int(lap_diff or 0) == 0
                    and gap_s is not None
                    and float(gap_s) + 1e-9 >= MOVE_OVER_SETTLE_TARGET_GAP_S
                ):
                    return "gap"
            except Exception:
                pass

        # Do not sacrifice another position to manufacture the teammate gap.
        # Classification order makes the next row the only immediate threat.
        if giver_pos + 1 < len(self.order):
            challenger = str(getattr(self.order[giver_pos + 1], "name", "") or "")
            if (
                challenger
                and challenger != target
                and self.driver_team.get(challenger) != self.driver_team.get(name)
                and self._running_on_track(challenger)
            ):
                try:
                    lap_diff, gap_s = self.distance_reference_gap(challenger, name)
                    if (
                        int(lap_diff or 0) == 0
                        and gap_s is not None
                        and 0.0 <= float(gap_s) <= MOVE_OVER_SETTLE_THREAT_GAP_S
                    ):
                        return "threat"
                except Exception:
                    pass
        return None

    def _move_over_progress_multiplier(self, driver_name: str, prev_progress: float) -> float:
        name = str(driver_name or "")
        order = self.move_over_orders.get(name)
        if not isinstance(order, dict) or self.caution_active:
            return 1.0
        target = str(order.get("target") or "")
        if not target or not self._running_on_track(name) or not self._running_on_track(target):
            return 1.0
        if self.driver_team.get(target) != self.driver_team.get(name):
            return 1.0
        if str(order.get("phase") or "handover") == "settling":
            if self._move_over_settle_release_reason(name, target) is not None:
                order["settle_release_requested"] = True
                return 1.0
            return MOVE_OVER_SETTLE_PROGRESS_MULTIPLIER
        current_target = self._move_over_eligible_target(
            name,
            max_gap_s=1.35,
            require_player_team=False,
        )
        if current_target != target:
            return 1.0
        try:
            current_sector = int(self._sector_index_for_progress(prev_progress))
        except Exception:
            current_sector = 0
        if not bool(order.get("armed", False)):
            try:
                lap_now = int(self.laps.get(name, 0) or 0)
                lap_requested = int(order.get("requested_lap", lap_now) or lap_now)
                sector_requested = int(order.get("requested_sector", current_sector) or current_sector)
                if lap_now > lap_requested or current_sector != sector_requested:
                    order["armed"] = True
                else:
                    return 1.0
            except Exception:
                order["armed"] = True
        return (
            MOVE_OVER_HANDOVER_PROGRESS_MULTIPLIER
            if self._straightish_sector(current_sector)
            else 1.0
        )

    def _move_over_should_slow(self, driver_name: str, prev_progress: float) -> bool:
        """Compatibility predicate retained for UI/tests using the old helper."""

        return self._move_over_progress_multiplier(driver_name, prev_progress) < 1.0 - 1e-9

    def _apply_move_over_morale(self, driver_name: str) -> None:
        driver = self.driver_by_name.get(str(driver_name or ""))
        if driver is None:
            return
        status = self._driver_contract_status(driver_name)
        if status == "first":
            amount, duration = -6.0, 6
        elif status == "equal":
            amount, duration = -3.0, 5
            try:
                state = getattr(getattr(self, "email_manager", None), "state", None)
                if state is not None and getattr(self, "player_team", None) and self.driver_team.get(driver_name) == getattr(self, "player_team", None):
                    flavor_state = getattr(state, "driver_flavor_event_state", None)
                    if not isinstance(flavor_state, dict):
                        flavor_state = {}
                    season = int(getattr(state, "season_no", 1) or 1)
                    if flavor_state.get("season") != season:
                        flavor_state = {"season": season, "sent": {}, "teammate_loss_streaks": {}}
                    flavor_state["equal_team_order_tension"] = True
                    state.driver_flavor_event_state = flavor_state
            except Exception:
                pass
        else:
            return
        try:
            personalities = set(canonical_driver_personalities(getattr(driver, "personalities", []), getattr(driver, "personality", "loyal"), 3))
            if "team_player" in personalities:
                amount *= 0.55
            if "prima_donna" in personalities:
                amount *= 1.45
            if "quiet_professional" in personalities:
                amount *= 0.50
        except Exception:
            pass
        try:
            state = getattr(getattr(self, "email_manager", None), "state", None)
            adjust = getattr(state, "oval_adjust_driver_mood_amount", None)
            if callable(adjust):
                amount = float(adjust(driver, amount))
        except Exception:
            pass
        effect = {
            "amount": amount,
            "initial_amount": amount,
            "weeks_remaining": duration,
            "duration": duration,
            "reason": "Asked to move over for teammate",
            "category": "team_order_move_over",
        }
        try:
            effects = getattr(driver, "morale_effects", None)
            if not isinstance(effects, list):
                effects = []
                setattr(driver, "morale_effects", effects)
            effects.append(effect)
            if len(effects) > 24:
                del effects[:-24]
        except Exception:
            pass

    def _driver_contract_status(self, driver_name: str) -> str:
        driver = self.driver_by_name.get(str(driver_name or ""))
        contract = getattr(driver, "contract", {}) if driver is not None else {}
        if not isinstance(contract, dict):
            contract = {}
        return canonical_driver_status(contract.get("status", contract.get("priority", "equal")))

    def _evaluate_ai_team_orders(self) -> None:
        if self.caution_active:
            self.ai_move_over_pair_counts.clear()
            return
        leader_name = self._current_on_track_leader_name(self.order)
        if not leader_name:
            self.ai_move_over_pair_counts.clear()
            return
        try:
            eval_key = (
                int(self.laps.get(leader_name, 0) or 0),
                int(self.current_sector_index.get(leader_name, 0) or 0),
            )
        except Exception:
            eval_key = (0, 0)
        if eval_key == self._ai_move_over_last_eval_key:
            return
        self._ai_move_over_last_eval_key = eval_key

        active_pairs = set()
        for idx in range(max(0, len(self.order) - 1)):
            front_name = str(getattr(self.order[idx], "name", "") or "")
            behind_name = str(getattr(self.order[idx + 1], "name", "") or "")
            if not front_name or not behind_name:
                continue
            team = self.driver_team.get(front_name)
            if not team or self.driver_team.get(behind_name) != team:
                continue
            if self.player_team and team == self.player_team:
                continue
            if front_name in self.move_over_orders:
                continue
            if not self._running_on_track(front_name) or not self._running_on_track(behind_name):
                continue
            if self._driver_contract_status(front_name) != "second":
                continue
            if self._driver_contract_status(behind_name) != "first":
                continue
            target = self._move_over_eligible_target(
                front_name,
                max_gap_s=1.0,
                require_player_team=False,
                require_ai_team=True,
            )
            if target != behind_name:
                continue
            pair_key = (front_name, behind_name)
            active_pairs.add(pair_key)
            streak = int(self.ai_move_over_pair_counts.get(pair_key, 0) or 0) + 1
            self.ai_move_over_pair_counts[pair_key] = streak
            if streak >= 3:
                if self._queue_move_over_order(front_name, behind_name, source="ai"):
                    self.ai_move_over_pair_counts.pop(pair_key, None)

        for pair_key in list(self.ai_move_over_pair_counts.keys()):
            if pair_key not in active_pairs:
                self.ai_move_over_pair_counts.pop(pair_key, None)

    def _finalize_move_over_orders(self) -> None:
        if not getattr(self, "move_over_orders", None):
            return
        positions = {getattr(drv, "name", None): idx for idx, drv in enumerate(self.order)}
        for name, order in list(self.move_over_orders.items()):
            target = str(order.get("target") or "") if isinstance(order, dict) else ""
            if not target:
                self.move_over_orders.pop(name, None)
                continue
            giver_pos = positions.get(name)
            target_pos = positions.get(target)
            if giver_pos is None or target_pos is None:
                self.move_over_orders.pop(name, None)
                continue
            phase = str(order.get("phase") or "handover")
            if phase == "settling":
                if bool(order.pop("settle_release_requested", False)) or (
                    self._move_over_settle_release_reason(name, target, positions) is not None
                ):
                    self.move_over_orders.pop(name, None)
                continue
            if target_pos < giver_pos:
                self._apply_move_over_morale(name)
                self.relationship_events.append(
                    {
                        "type": "team_order",
                        "ordered_driver": str(name),
                        "beneficiary": str(target),
                        "team": self.driver_team.get(name),
                    }
                )
                self._defer_event(f"TEAM ORDER: {name} let {target} through")
                now = self._radio_sim_time()
                order["phase"] = "settling"
                order["settle_started_at"] = now
                order["settle_deadline"] = now + MOVE_OVER_SETTLE_MAX_DURATION_S
                if self._move_over_settle_release_reason(name, target, positions) is not None:
                    self.move_over_orders.pop(name, None)

    def driver_pace_mode(self, driver_name: str) -> str:
        return self.driver_pace_modes.display_mode(driver_name)

    def set_driver_pace_mode(self, driver_name: str, mode: str) -> bool:
        if not self.player_team:
            return False
        if self.driver_team.get(driver_name) != self.player_team:
            return False
        if not self.driver_pace_modes.is_player_controlled(driver_name):
            return False
        self.driver_pace_modes.set_pending_mode(driver_name, mode)
        return True

    def available_driver_engine_modes(self):
        return tuple(ENGINE_MODE_ORDER)

    def driver_engine_mode(self, driver_name: str) -> str:
        return self.driver_engine_modes.display_mode(driver_name)

    def set_driver_engine_mode(self, driver_name: str, mode: str) -> bool:
        if not self.player_team:
            return False
        if self.driver_team.get(driver_name) != self.player_team:
            return False
        if not self.driver_engine_modes.is_player_controlled(driver_name):
            return False
        self.driver_engine_modes.set_pending_mode(driver_name, mode)
        return True

    def ers_enabled(self) -> bool:
        return bool(self._ers_regulation_enabled)

    def available_driver_ers_modes(self):
        return tuple(ERS_MODE_ORDER)

    def driver_ers_mode(self, driver_name: str) -> str:
        return self.driver_ers_modes.display_mode(driver_name)

    def set_driver_ers_mode(self, driver_name: str, mode: str) -> bool:
        if not self._ers_regulation_enabled:
            return False
        if self.driver_ers_broken(driver_name):
            return False
        if not self.player_team:
            return False
        if self.driver_team.get(driver_name) != self.player_team:
            return False
        if not self.driver_ers_modes.is_player_controlled(driver_name):
            return False
        self.driver_ers_modes.set_pending_mode(driver_name, mode)
        return True

    def driver_ers_charge_ratio(self, driver_name: str) -> float:
        if self.driver_ers_broken(driver_name):
            return 0.0
        try:
            cap = float(self._ers_capacity_for_driver(driver_name))
        except Exception:
            cap = 0.0
        if cap <= 1e-9:
            return 0.0
        try:
            charge = float(self.energy_store_by_driver.get(driver_name, 0.0) or 0.0)
        except Exception:
            charge = 0.0
        return max(0.0, min(1.0, charge / cap))

    def _ers_development_manager(self):
        return getattr(getattr(self, "state", None), "ers_development_manager", None)

    def driver_ers_broken(self, driver_name: str) -> bool:
        return bool((getattr(self, "ers_failed", {}) or {}).get(str(driver_name or ""), False))

    def _evaluate_ers_failure_for_lap(self, driver_name: str, lap_number: int) -> bool:
        name = str(driver_name or "")
        if not name or not self._ers_regulation_enabled or self.driver_ers_broken(name):
            return False
        try:
            lap_number = max(1, int(lap_number))
        except Exception:
            return False
        checked = int((getattr(self, "ers_failure_checked_lap", {}) or {}).get(name, 0) or 0)
        if lap_number <= checked:
            return False
        self.ers_failure_checked_lap[name] = lap_number
        probability = max(
            0.0,
            min(1.0, float((getattr(self, "ers_failure_probability_by_driver", {}) or {}).get(name, 0.0) or 0.0)),
        )
        if probability <= 0.0 or random.random() >= probability:
            return False
        self.ers_failed[name] = True
        self.energy_store_by_driver[name] = 0.0
        self.lap_deploy_used_by_driver[name] = 0.0
        self.lap_harvest_used_by_driver[name] = 0.0
        try:
            self.driver_ers_modes.set_active_mode(name, ERS_MODE_NO_DEPLOY)
        except Exception:
            pass
        cache = getattr(self, "_ers_mode_target_cache", None)
        if isinstance(cache, dict):
            for key in [key for key in cache if isinstance(key, tuple) and key and key[0] == name]:
                cache.pop(key, None)
        self._defer_event(f"{name}'s ERS has failed!")
        return True

    def _ers_capacity_for_driver(self, driver_name: str) -> float:
        if not self._ers_regulation_enabled:
            return 0.0
        if self.driver_ers_broken(driver_name):
            return 0.0
        cache = getattr(self, "_session_ers_effect_cache", None)
        key = ("capacity", str(driver_name))
        if isinstance(cache, dict) and key in cache:
            return float(cache[key])
        edm = self._ers_development_manager()
        if edm and hasattr(edm, "capacity_for_driver"):
            try:
                value = max(0.0, float(edm.capacity_for_driver(driver_name, self.driver_team.get(driver_name))))
                if isinstance(cache, dict):
                    cache[key] = value
                return value
            except Exception:
                pass
        value = float(self.ers_capacity_mj)
        if isinstance(cache, dict):
            cache[key] = value
        return value

    def _ers_deploy_cost_multiplier_for_driver(self, driver_name: str) -> float:
        cache = getattr(self, "_session_ers_effect_cache", None)
        key = ("deploy", str(driver_name))
        if isinstance(cache, dict) and key in cache:
            return float(cache[key])
        edm = self._ers_development_manager()
        if edm and hasattr(edm, "deploy_cost_multiplier_for_driver"):
            try:
                value = max(0.01, float(edm.deploy_cost_multiplier_for_driver(driver_name, self.driver_team.get(driver_name))))
                if isinstance(cache, dict):
                    cache[key] = value
                return value
            except Exception:
                pass
        if isinstance(cache, dict):
            cache[key] = 1.0
        return 1.0

    def _ers_harvest_multiplier_for_driver(self, driver_name: str) -> float:
        cache = getattr(self, "_session_ers_effect_cache", None)
        key = ("harvest", str(driver_name))
        if isinstance(cache, dict) and key in cache:
            return float(cache[key])
        edm = self._ers_development_manager()
        if edm and hasattr(edm, "harvest_multiplier_for_driver"):
            try:
                value = max(0.0, float(edm.harvest_multiplier_for_driver(driver_name, self.driver_team.get(driver_name))))
                if isinstance(cache, dict):
                    cache[key] = value
                return value
            except Exception:
                pass
        if isinstance(cache, dict):
            cache[key] = 1.0
        return 1.0

    def _ers_output_multiplier_for_driver(self, driver_name: str) -> float:
        cache = getattr(self, "_session_ers_effect_cache", None)
        key = ("output", str(driver_name))
        if isinstance(cache, dict) and key in cache:
            return float(cache[key])
        edm = self._ers_development_manager()
        if edm and hasattr(edm, "output_multiplier_for_driver"):
            try:
                value = max(0.0, float(edm.output_multiplier_for_driver(driver_name, self.driver_team.get(driver_name))))
                if isinstance(cache, dict):
                    cache[key] = value
                return value
            except Exception:
                pass
        if isinstance(cache, dict):
            cache[key] = 1.0
        return 1.0

    def _ers_uses_dynamic_energy_plan(self, driver_name: str) -> bool:
        cache = getattr(self, "_session_ers_effect_cache", None)
        key = ("dynamic_plan", str(driver_name))
        if isinstance(cache, dict) and key in cache:
            return bool(cache[key])
        edm = self._ers_development_manager()
        value = False
        if edm and hasattr(edm, "uses_dynamic_energy_plan_for_driver"):
            try:
                value = bool(
                    edm.uses_dynamic_energy_plan_for_driver(
                        driver_name,
                        self.driver_team.get(driver_name),
                    )
                )
            except Exception:
                value = False
        if isinstance(cache, dict):
            cache[key] = bool(value)
        return bool(value)

    def _ers_mode_targets_for_driver(self, driver_name: str) -> tuple[str, float, float, float]:
        if not self._ers_regulation_enabled:
            return ERS_MODE_BALANCED, 0.0, 0.0, 0.0
        name = str(driver_name or "")
        if self.driver_ers_broken(name):
            return ERS_MODE_NO_DEPLOY, 0.0, 0.0, 0.0
        try:
            controller = self.driver_ers_modes
            mode = (getattr(controller, "_active", {}) or {}).get(name)
            if mode not in ERS_MODE_SPECS:
                team_name = (getattr(controller, "driver_team", {}) or {}).get(name)
                mode = controller._default_mode_for_team(team_name)
            if mode not in ERS_MODE_SPECS:
                mode = controller.active_mode(name)
        except Exception:
            try:
                mode = self.driver_ers_modes.active_mode(name)
            except Exception:
                mode = ERS_MODE_BALANCED
        if mode not in ERS_MODE_SPECS:
            mode = ERS_MODE_BALANCED
        cache_key = (name, mode)
        cached = self._ers_mode_target_cache.get(cache_key)
        if cached is not None:
            return cached
        spec = ERS_MODE_SPECS.get(mode) or ERS_MODE_SPECS[ERS_MODE_BALANCED]
        deploy_cost = self._ers_deploy_cost_multiplier_for_driver(name)
        harvest_mult = self._ers_harvest_multiplier_for_driver(name)
        output_mult = self._ers_output_multiplier_for_driver(name)
        edm = self._ers_development_manager()
        energy_config = edm.mode_energy_balance_config() if edm and hasattr(edm, "mode_energy_balance_config") else None
        plan = build_ers_mode_energy_plan(
            mode,
            self._ers_capacity_for_driver(name),
            deploy_cost,
            harvest_mult,
            developed=self._ers_uses_dynamic_energy_plan(name),
            config=energy_config,
        )
        self._ers_mode_coverage_cache[cache_key] = float(plan.deployment_coverage_multiplier)
        out = (
            mode,
            float(plan.battery_deploy_target_mj),
            float(plan.harvest_target_mj),
            max(0.0, float(getattr(spec, "boost_points", 0.0) or 0.0) * float(output_mult)),
        )
        self._ers_mode_target_cache[cache_key] = out
        return out

    def _ers_deploy_windows_for_driver(self, driver_name: str):
        try:
            mode = self.driver_ers_modes.active_mode(driver_name)
        except Exception:
            mode = ERS_MODE_BALANCED
        cache_key = (str(driver_name or ""), str(mode or ""))
        cached = self._ers_deploy_window_cache.get(cache_key)
        if cached is not None:
            return cached
        self._ers_mode_targets_for_driver(driver_name)
        coverage = float(self._ers_mode_coverage_cache.get(cache_key, 1.0) or 0.0)
        windows = ers_deploy_windows_for_coverage(
            self._ers_deploy_windows,
            self._ers_extended_deploy_windows,
            coverage,
        )
        self._ers_deploy_window_cache[cache_key] = windows
        return windows

    def _ers_lap_targets(self, driver_name: str) -> tuple[float, float]:
        if not self._ers_regulation_enabled:
            return 0.0, 0.0
        _mode, deploy_target, harvest_target, _boost_points = self._ers_mode_targets_for_driver(driver_name)
        return deploy_target, harvest_target

    def _ers_reset_lap_state(self, driver_name: str) -> None:
        self.lap_deploy_used_by_driver[driver_name] = 0.0
        self.lap_harvest_used_by_driver[driver_name] = 0.0

    def _ers_bonus_points_for_progress(self, driver_name: str, progress: float) -> float:
        if not self._ers_regulation_enabled:
            return 0.0
        if self.driver_ers_broken(driver_name):
            return 0.0
        _mode, deploy_target, _harvest_target, boost_points = self._ers_mode_targets_for_driver(driver_name)
        if deploy_target <= 1e-9:
            return 0.0
        deploy_windows = self._ers_deploy_windows_for_driver(driver_name)
        try:
            used = float(self.lap_deploy_used_by_driver.get(driver_name, 0.0) or 0.0)
            charge = float(self.energy_store_by_driver.get(driver_name, 0.0) or 0.0)
            prog = max(0.0, min(1.0, float(progress)))
        except Exception:
            return 0.0
        if used >= deploy_target - 1e-9 or charge <= 1e-9:
            return 0.0
        if _REALISTIC_KERNELS is not None:
            try:
                return float(
                    _REALISTIC_KERNELS.ers_bonus_points_for_progress(
                        prog,
                        deploy_windows,
                        float(deploy_target),
                        float(used),
                        float(charge),
                        float(boost_points),
                    )
                )
            except Exception:
                pass
        for window in deploy_windows:
            try:
                start = float(window.get("start", 0.0) or 0.0)
                end = float(window.get("end", 0.0) or 0.0)
            except Exception:
                continue
            if start <= prog <= end:
                return boost_points
        return 0.0

    def _instantaneous_live_speed_kmh(
        self,
        driver,
        progress: float,
        base_inputs: Optional[LapInputs] = None,
        lap_inputs: Optional[LapInputs] = None,
        lap_inputs_ers_bonus: Optional[float] = None,
    ) -> float:
        if not self.use_realistic_physics:
            return 0.0
        try:
            prog = float(progress)
        except Exception:
            prog = 0.0
        if prog >= 1.0:
            prog = 0.999999
        prog = max(0.0, min(0.999999, prog))
        try:
            if not isinstance(base_inputs, LapInputs):
                base_inputs = self._realistic_base_lap_inputs_for_driver(driver)
            ers_bonus = float(
                self._ers_bonus_points_for_progress(
                    getattr(driver, "name", ""),
                    prog,
                )
            )
            if not (
                isinstance(lap_inputs, LapInputs)
                and lap_inputs_ers_bonus is not None
                and abs(float(lap_inputs_ers_bonus) - float(ers_bonus)) <= 1e-9
            ):
                lap_inputs = (
                    replace(
                        base_inputs,
                        acceleration_rating_points=float(base_inputs.acceleration_rating_points) + float(ers_bonus),
                    )
                    if abs(float(ers_bonus)) > 1e-9
                    else base_inputs
                )
            speed = self.realistic_physics.speed_kmh_at_progress(
                lap_inputs,
                prog,
                cache_key=(getattr(driver, "name", ""), "live_speed"),
            )
            racecraft = getattr(self, "oval_racecraft", None)
            if racecraft is not None:
                _rate_mult, speed_mult = racecraft.rate_and_speed_multipliers(
                    self,
                    getattr(driver, "name", ""),
                    prog,
                    base_speed_kmh=float(speed),
                )
                speed = float(speed) * float(speed_mult)
            return max(0.0, float(speed))
        except Exception:
            return 0.0

    def _ers_apply_progress_segment(self, driver_name: str, start_progress: float, end_progress: float) -> None:
        if not self._ers_regulation_enabled:
            return
        if self.driver_ers_broken(driver_name):
            self.energy_store_by_driver[str(driver_name)] = 0.0
            return
        deploy_target, harvest_target = self._ers_lap_targets(driver_name)
        if deploy_target <= 1e-9 and harvest_target <= 1e-9:
            return
        deploy_windows = self._ers_deploy_windows_for_driver(driver_name)
        harvest_windows = self._ers_harvest_windows
        capacity_mj = float(self._ers_capacity_for_driver(driver_name))
        try:
            charge = float(self.energy_store_by_driver.get(driver_name, capacity_mj) or 0.0)
        except Exception:
            charge = capacity_mj
        charge = max(0.0, min(capacity_mj, charge))
        used = float(self.lap_deploy_used_by_driver.get(driver_name, 0.0) or 0.0)
        harvested = float(self.lap_harvest_used_by_driver.get(driver_name, 0.0) or 0.0)
        charge, used, harvested = apply_ers_progress_segment(
            start_progress,
            end_progress,
            deploy_windows,
            harvest_windows,
            deploy_target,
            harvest_target,
            capacity_mj,
            used,
            harvested,
            charge,
        )

        self.energy_store_by_driver[driver_name] = max(0.0, min(capacity_mj, charge))
        self.lap_deploy_used_by_driver[driver_name] = max(0.0, used)
        self.lap_harvest_used_by_driver[driver_name] = max(0.0, harvested)

    def _ers_deploy_overlap_for_segment(
        self,
        driver_name: str,
        start_progress: float,
        end_progress: float,
    ) -> tuple[float, Optional[float]]:
        if not self._ers_regulation_enabled:
            return 0.0, None
        if self.driver_ers_broken(driver_name):
            return 0.0, None
        deploy_target, _ = self._ers_lap_targets(driver_name)
        if deploy_target <= 1e-9:
            return 0.0, None
        try:
            used = float(self.lap_deploy_used_by_driver.get(driver_name, 0.0) or 0.0)
            charge = float(self.energy_store_by_driver.get(driver_name, 0.0) or 0.0)
        except Exception:
            return 0.0, None
        if used >= deploy_target - 1e-9 or charge <= 1e-9:
            return 0.0, None
        deploy_windows = self._ers_deploy_windows_for_driver(driver_name)
        if _REALISTIC_KERNELS is not None:
            try:
                overlap_ratio, sample_progress = _REALISTIC_KERNELS.ers_deploy_overlap_for_segment(
                    float(start_progress),
                    float(end_progress),
                    deploy_windows,
                    float(deploy_target),
                    float(used),
                    float(charge),
                )
                return float(overlap_ratio), sample_progress
            except Exception:
                pass

        total_len = 0.0
        overlap_len = 0.0
        sample_progress = None
        for seg_start, seg_end in split_progress_segments(start_progress, end_progress):
            seg_len = max(0.0, float(seg_end) - float(seg_start))
            if seg_len <= 1e-12:
                continue
            total_len += seg_len
            for window in deploy_windows:
                try:
                    win_start = float(window.get("start", 0.0) or 0.0)
                    win_end = float(window.get("end", 0.0) or 0.0)
                except Exception:
                    continue
                ov_start = max(float(seg_start), win_start)
                ov_end = min(float(seg_end), win_end)
                if ov_end <= ov_start + 1e-12:
                    continue
                overlap_len += ov_end - ov_start
                if sample_progress is None:
                    sample_progress = 0.5 * (ov_start + ov_end)
        if total_len <= 1e-12 or overlap_len <= 1e-12:
            return 0.0, None
        return max(0.0, min(1.0, overlap_len / total_len)), sample_progress

    def _race_ai_gap_context(self, driver_name: str) -> dict:
        name = str(driver_name or "")
        if not name:
            return {
                "gap_ahead_s": 999.0,
                "gap_behind_s": 999.0,
                "ahead_same_lap": False,
                "behind_same_lap": False,
            }
        order_names = [str(getattr(d, "name", "") or "") for d in list(self.order or [])]
        try:
            idx = order_names.index(name)
        except ValueError:
            idx = -1
        ahead_name = order_names[idx - 1] if idx > 0 else None
        behind_name = order_names[idx + 1] if idx >= 0 and idx + 1 < len(order_names) else None
        out = {
            "gap_ahead_s": 999.0,
            "gap_behind_s": 999.0,
            "ahead_same_lap": False,
            "behind_same_lap": False,
        }
        if ahead_name:
            try:
                lap_diff, gap_s = self.distance_reference_gap(name, ahead_name)
                out["ahead_same_lap"] = int(lap_diff) == 0 and gap_s is not None
                if gap_s is not None:
                    out["gap_ahead_s"] = max(0.0, float(gap_s))
            except Exception:
                pass
        if behind_name:
            try:
                lap_diff, gap_s = self.distance_reference_gap(behind_name, name)
                out["behind_same_lap"] = int(lap_diff) == 0 and gap_s is not None
                if gap_s is not None:
                    out["gap_behind_s"] = max(0.0, float(gap_s))
            except Exception:
                pass
        return out

    def _race_ai_component_risk(self, driver_name: str) -> bool:
        risk_parts = 0
        try:
            for info in (self.part_cornering_profile.get(driver_name, {}) or {}).values():
                if isinstance(info, dict) and float(info.get("condition", 100.0) or 100.0) <= 50.0:
                    risk_parts += 1
        except Exception:
            pass
        try:
            for info in (self.supplier_part_specs.get(driver_name, {}) or {}).values():
                if isinstance(info, dict) and float(info.get("condition", 100.0) or 100.0) <= 50.0:
                    risk_parts += 1
        except Exception:
            pass
        return risk_parts >= 2

    def _race_ai_engine_risk(self, driver_name: str) -> bool:
        risk_parts = 0
        try:
            info = self.engine_unit_specs.get(driver_name)
            if isinstance(info, dict):
                start = float(info.get("condition", 100.0) or 100.0)
                delta = float(self.engine_wear_delta.get(driver_name, 0.0) or 0.0)
                if self._live_condition_pct(start, delta) <= 55.0:
                    risk_parts += 1
        except Exception:
            pass
        try:
            info = (self.supplier_part_specs.get(driver_name, {}) or {}).get("gearbox")
            if isinstance(info, dict):
                start = float(info.get("condition", 100.0) or 100.0)
                delta = float((self.supplier_part_wear_delta.get(driver_name, {}) or {}).get("gearbox", 0.0) or 0.0)
                if self._live_condition_pct(start, delta) <= 55.0:
                    risk_parts += 1
        except Exception:
            pass
        return risk_parts >= 1

    def _evaluate_ai_race_pace_mode(self, driver_name: str) -> None:
        name = str(driver_name or "")
        if (
            not name
            or not self.driver_pace_modes.is_ai_controlled(name)
            or name in self.finished
            or self.dnf.get(name, False)
            or self.pit_remaining.get(name, 0.0) > 0.0
        ):
            return
        token = int(self._ai_race_sector_tokens.get(name, 0) or 0) + 1
        self._ai_race_sector_tokens[name] = token
        gap_ctx = self._race_ai_gap_context(name)
        push_window = int(self._ai_race_push_window.get(name, 0) or 0)
        if push_window > 0:
            self._ai_race_push_window[name] = push_window - 1
        try:
            laps_done = int(self.laps.get(name, 0) or 0)
        except Exception:
            laps_done = 0
        try:
            laps_remaining = max(0, int(self.total_laps) - laps_done)
        except Exception:
            laps_remaining = 0
        isolated = (
            (not bool(gap_ctx.get("ahead_same_lap", False)) or float(gap_ctx.get("gap_ahead_s", 999.0)) >= 2.75)
            and (not bool(gap_ctx.get("behind_same_lap", False)) or float(gap_ctx.get("gap_behind_s", 999.0)) >= 2.75)
        )
        desired = self.driver_pace_modes.choose_race_mode(
            self,
            name,
            {
                "sector_token": token,
                "gap_ahead_s": gap_ctx.get("gap_ahead_s", 999.0),
                "gap_behind_s": gap_ctx.get("gap_behind_s", 999.0),
                "ahead_same_lap": gap_ctx.get("ahead_same_lap", False),
                "behind_same_lap": gap_ctx.get("behind_same_lap", False),
                "tyre_wear": float(self.tyre_wear.get(name, 0.0) or 0.0),
                "isolated": bool(isolated),
                "late_race": laps_remaining <= 10,
                "laps_remaining": laps_remaining,
                "component_risk": self._race_ai_component_risk(name),
                "start_push": token <= 3,
                "push_window": push_window > 0,
            },
        )
        if desired != self.driver_pace_modes.active_mode(name):
            self.driver_pace_modes.set_pending_mode(name, desired)

    def _evaluate_ai_race_engine_mode(self, driver_name: str) -> None:
        name = str(driver_name or "")
        if (
            not name
            or not self.driver_engine_modes.is_ai_controlled(name)
            or name in self.finished
            or self.dnf.get(name, False)
            or self.pit_remaining.get(name, 0.0) > 0.0
        ):
            return
        token = int(self._ai_race_sector_tokens.get(name, 0) or 0)
        gap_ctx = self._race_ai_gap_context(name)
        push_window = int(self._ai_race_push_window.get(name, 0) or 0)
        try:
            laps_done = int(self.laps.get(name, 0) or 0)
        except Exception:
            laps_done = 0
        try:
            laps_remaining = max(0, int(self.total_laps) - laps_done)
        except Exception:
            laps_remaining = 0
        isolated = (
            (not bool(gap_ctx.get("ahead_same_lap", False)) or float(gap_ctx.get("gap_ahead_s", 999.0)) >= 2.75)
            and (not bool(gap_ctx.get("behind_same_lap", False)) or float(gap_ctx.get("gap_behind_s", 999.0)) >= 2.75)
        )
        battle = (
            (bool(gap_ctx.get("ahead_same_lap", False)) and float(gap_ctx.get("gap_ahead_s", 999.0)) <= 1.0)
            or (bool(gap_ctx.get("behind_same_lap", False)) and float(gap_ctx.get("gap_behind_s", 999.0)) <= 1.0)
        )
        desired = self.driver_engine_modes.choose_race_mode(
            self,
            name,
            {
                "sector_token": token,
                "battle": battle,
                "isolated": bool(isolated),
                "laps_remaining": laps_remaining,
                "engine_risk": self._race_ai_engine_risk(name),
                "start_push": token <= 3,
                "push_window": push_window > 0,
            },
        )
        if desired != self.driver_engine_modes.active_mode(name):
            self.driver_engine_modes.set_pending_mode(name, desired)

    def _evaluate_ai_race_ers_mode(self, driver_name: str) -> None:
        name = str(driver_name or "")
        if (
            not name
            or not self._ers_regulation_enabled
            or not self.driver_ers_modes.is_ai_controlled(name)
            or name in self.finished
            or self.dnf.get(name, False)
        ):
            return
        gap_ctx = self._race_ai_gap_context(name)
        try:
            laps_done = int(self.laps.get(name, 0) or 0) + 1
        except Exception:
            laps_done = 1
        try:
            laps_remaining = max(0, int(self.total_laps) - laps_done)
        except Exception:
            laps_remaining = 0
        attack_window = int(self._ai_race_ers_attack_window.get(name, 0) or 0)
        if attack_window > 0:
            self._ai_race_ers_attack_window[name] = attack_window - 1
        desired = self.driver_ers_modes.choose_race_mode(
            self,
            name,
            {
                "lap_token": laps_done,
                "charge_ratio": self.driver_ers_charge_ratio(name),
                "gap_ahead_s": gap_ctx.get("gap_ahead_s", 999.0),
                "gap_behind_s": gap_ctx.get("gap_behind_s", 999.0),
                "ahead_same_lap": gap_ctx.get("ahead_same_lap", False),
                "behind_same_lap": gap_ctx.get("behind_same_lap", False),
                "pit_exit_push": attack_window > 0,
                "restart_push": attack_window > 0 and not self.caution_active,
                "final_laps_fight": laps_remaining <= 4
                and (
                    (bool(gap_ctx.get("ahead_same_lap", False)) and float(gap_ctx.get("gap_ahead_s", 999.0)) <= 0.9)
                    or (bool(gap_ctx.get("behind_same_lap", False)) and float(gap_ctx.get("gap_behind_s", 999.0)) <= 0.7)
                ),
            },
        )
        if desired != self.driver_ers_modes.active_mode(name):
            self.driver_ers_modes.set_pending_mode(name, desired)

    def _apply_crash_damage(self, driver_name: str):
        parts = self.part_wear_delta.setdefault(
            driver_name, {key: 0.0 for key in self.part_wear_per_lap.keys()}
        )
        if not parts:
            return
        try:
            choice = random.choice(list(self.part_wear_per_lap.keys()))
        except Exception:
            return
        parts[choice] = parts.get(choice, 0.0) + PART_CRASH_DAMAGE
        try:
            self._apply_runtime_part_wear_increment(
                driver_name, ((str(choice), float(PART_CRASH_DAMAGE)),)
            )
        except Exception:
            pass

    @staticmethod
    def _range_roll(raw, default=(1.0, 10.0)) -> float:
        try:
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                lo, hi = float(raw[0]), float(raw[1])
            else:
                lo, hi = float(default[0]), float(default[1])
        except Exception:
            lo, hi = float(default[0]), float(default[1])
        if hi < lo:
            lo, hi = hi, lo
        return random.uniform(lo, hi) if hi > lo else lo

    def _apply_collision_part_damage(self, driver_name: str, fraction: float) -> None:
        try:
            amount = max(0.0, float(PART_CRASH_DAMAGE) * max(0.0, float(fraction)))
        except Exception:
            amount = 0.0
        if amount <= 0.0:
            return
        parts = self.part_wear_delta.setdefault(
            driver_name, {key: 0.0 for key in self.part_wear_per_lap.keys()}
        )
        if not parts:
            return
        try:
            choice = random.choice(list(self.part_wear_per_lap.keys()))
        except Exception:
            return
        parts[choice] = parts.get(choice, 0.0) + amount
        try:
            self._apply_runtime_part_wear_increment(driver_name, ((str(choice), float(amount)),))
        except Exception:
            pass

    def _apply_collision_time_loss(self, driver_name: str, loss_s: float, *, flag_decay: int = 8) -> None:
        try:
            loss = max(0.0, float(loss_s))
        except Exception:
            loss = 0.0
        if loss <= 0.0:
            return
        self.freeze_remaining[driver_name] = float(self.freeze_remaining.get(driver_name, 0.0) or 0.0) + loss
        self.incident_flag_decay[driver_name] = max(
            int(self.incident_flag_decay.get(driver_name, 0) or 0),
            int(flag_decay),
        )

    @staticmethod
    def _choose_collision_outcome(incidents_cfg: dict) -> str:
        default_weights = {
            "both_retire": 0.20,
            "single_car_retire": 0.20,
            "minor_contact": 0.30,
            "spin_contact": 0.30,
        }
        raw = incidents_cfg.get("collision_outcomes", {}) if isinstance(incidents_cfg, dict) else {}
        weights = {}
        if isinstance(raw, dict):
            for key, default in default_weights.items():
                try:
                    value = float(raw.get(key, default))
                except Exception:
                    value = default
                weights[key] = max(0.0, value)
        else:
            weights = dict(default_weights)
        total = sum(weights.values())
        if total <= 0.0:
            weights = dict(default_weights)
            total = sum(weights.values())
        roll = random.random() * total
        acc = 0.0
        for key in ("both_retire", "single_car_retire", "minor_contact", "spin_contact"):
            acc += weights.get(key, 0.0)
            if roll <= acc:
                return key
        return "both_retire"

    @staticmethod
    def _relationship_pair_key(driver_a: str, driver_b: str) -> tuple:
        return tuple(sorted((str(driver_a or ""), str(driver_b or ""))))

    def _build_collision_relationship_values(self) -> dict:
        manager = getattr(self, "relationship_manager", None)
        values = {}
        if manager is None:
            return values
        names = [str(getattr(driver, "name", "") or "") for driver in self.drivers]
        names = [name for name in names if name]
        for idx, driver_a in enumerate(names):
            for driver_b in names[idx + 1:]:
                try:
                    value = int(manager.driver_relationship(driver_a, driver_b))
                except Exception:
                    value = 50
                values[self._relationship_pair_key(driver_a, driver_b)] = max(0, min(100, value))
        return values

    def _collision_relationship_multiplier(self, driver_a: str, driver_b: str, incidents_cfg: dict) -> float:
        cfg = incidents_cfg if isinstance(incidents_cfg, dict) else {}
        if not bool(cfg.get("collision_relationship_modifier_enabled", True)):
            return 1.0
        try:
            neutral = max(1.0, min(99.0, float(cfg.get("collision_relationship_neutral", 50.0))))
        except Exception:
            neutral = 50.0
        try:
            best_mult = max(0.0, float(cfg.get("collision_relationship_100_multiplier", 0.5)))
        except Exception:
            best_mult = 0.5
        try:
            worst_mult = max(0.0, float(cfg.get("collision_relationship_0_multiplier", 5.0)))
        except Exception:
            worst_mult = 5.0
        try:
            friendly_curve = max(0.1, float(cfg.get("collision_relationship_friendly_curve", 1.25)))
        except Exception:
            friendly_curve = 1.25
        try:
            rival_curve = max(0.1, float(cfg.get("collision_relationship_rival_curve", 1.75)))
        except Exception:
            rival_curve = 1.75
        key = self._relationship_pair_key(driver_a, driver_b)
        try:
            value = float((getattr(self, "collision_relationship_values", {}) or {}).get(key, neutral))
        except Exception:
            value = neutral
        value = max(0.0, min(100.0, value))
        if value >= neutral:
            span = max(1e-9, 100.0 - neutral)
            ratio = max(0.0, min(1.0, (value - neutral) / span))
            return max(0.0, 1.0 - ((1.0 - best_mult) * (ratio ** friendly_curve)))
        span = max(1e-9, neutral)
        ratio = max(0.0, min(1.0, (neutral - value) / span))
        return max(0.0, 1.0 + ((worst_mult - 1.0) * (ratio ** rival_curve)))

    def _record_collision_relationship_event(self, driver_a: str, driver_b: str, outcome: str, relationship_multiplier: float) -> None:
        self.relationship_events.append(
            {
                "type": "collision",
                "subtype": str(outcome),
                "drivers": [str(driver_a), str(driver_b)],
                "teammates": self.driver_team.get(driver_a) == self.driver_team.get(driver_b),
                "team": self.driver_team.get(driver_a)
                if self.driver_team.get(driver_a) == self.driver_team.get(driver_b)
                else None,
                "relationship_multiplier": float(relationship_multiplier),
            }
        )

    def _handle_collision_incident(self, driver_a: str, driver_b: str, incidents_cfg: dict) -> None:
        previous_context = getattr(self, "_active_race_incident_id", None)
        incident_id = self._next_race_incident_id()
        self._active_race_incident_id = incident_id
        try:
            self._handle_collision_incident_scoped(driver_a, driver_b, incidents_cfg)
        finally:
            self._active_race_incident_id = previous_context

    def _handle_collision_incident_scoped(self, driver_a: str, driver_b: str, incidents_cfg: dict) -> None:
        outcome = self._choose_collision_outcome(incidents_cfg if isinstance(incidents_cfg, dict) else {})
        major = outcome in {"single_car_retire", "both_retire"}
        self._emit_race_alert(
            "collision" if major else "contact",
            "major" if major else "minor",
            f"Collision between {driver_a} and {driver_b}",
            drivers=[driver_a, driver_b],
            outcomes=[str(outcome).replace("_", " ").title()],
        )
        loss_range = incidents_cfg.get("collision_time_loss_s", [1.0, 10.0]) if isinstance(incidents_cfg, dict) else [1.0, 10.0]
        minor_damage_fraction = (
            incidents_cfg.get("minor_contact_part_damage_fraction", 0.20)
            if isinstance(incidents_cfg, dict)
            else 0.20
        )
        single_damage_fraction = (
            incidents_cfg.get("single_car_contact_part_damage_fraction", 0.25)
            if isinstance(incidents_cfg, dict)
            else 0.25
        )
        spin_damage_fraction = (
            incidents_cfg.get("spin_contact_part_damage_fraction", 0.10)
            if isinstance(incidents_cfg, dict)
            else 0.10
        )

        if outcome == "single_car_retire":
            retired, continuing = random.choice(((driver_a, driver_b), (driver_b, driver_a)))
            loss = self._range_roll(loss_range)
            self._record_collision_relationship_event(driver_a, driver_b, outcome, 1.0)
            self.events.append(
                f"COLLISION: {retired} retired after contact with {continuing}; {continuing} lost {loss:.1f}s"
            )
            self._emit_radio_event("race_contact", retired, opponent=continuing)
            self._emit_radio_event("race_contact", continuing, opponent=retired)
            self._retire_driver(retired, reason=f"collision with {continuing}")
            if not self.dnf.get(continuing, False):
                self._apply_collision_time_loss(continuing, loss)
                self._apply_collision_part_damage(continuing, single_damage_fraction)
                self._roll_front_wing_damage(continuing, source=outcome)
            return

        if outcome == "minor_contact":
            loss_a = self._range_roll(loss_range)
            loss_b = self._range_roll(loss_range)
            self._record_collision_relationship_event(driver_a, driver_b, outcome, 0.5)
            self.events.append(
                f"CONTACT: {driver_a} and {driver_b} touched; {driver_a} lost {loss_a:.1f}s, {driver_b} lost {loss_b:.1f}s"
            )
            self._emit_radio_event("race_contact", driver_a, opponent=driver_b)
            self._emit_radio_event("race_contact", driver_b, opponent=driver_a)
            if not self.dnf.get(driver_a, False):
                self._apply_collision_time_loss(driver_a, loss_a, flag_decay=6)
                self._apply_collision_part_damage(driver_a, minor_damage_fraction)
                self._roll_front_wing_damage(driver_a, source=outcome)
            if not self.dnf.get(driver_b, False):
                self._apply_collision_time_loss(driver_b, loss_b, flag_decay=6)
                self._apply_collision_part_damage(driver_b, minor_damage_fraction)
                self._roll_front_wing_damage(driver_b, source=outcome)
            return

        if outcome == "spin_contact":
            spun, other = random.choice(((driver_a, driver_b), (driver_b, driver_a)))
            loss = self._range_roll(loss_range)
            self._record_collision_relationship_event(driver_a, driver_b, outcome, 0.5)
            self.events.append(f"CONTACT: {spun} spun after contact with {other} and lost {loss:.1f}s")
            self._emit_radio_event("race_contact", spun, opponent=other)
            self._emit_radio_event("race_contact", other, opponent=spun)
            if not self.dnf.get(spun, False):
                self._apply_collision_time_loss(spun, loss)
                self._apply_collision_part_damage(spun, spin_damage_fraction)
                self._roll_front_wing_damage(spun, source=outcome)
            return

        self._record_collision_relationship_event(driver_a, driver_b, "both_retire", 1.0)
        self.events.append(f"COLLISION: {driver_a} and {driver_b} crash out")
        self._emit_radio_event("race_contact", driver_a, opponent=driver_b)
        self._emit_radio_event("race_contact", driver_b, opponent=driver_a)
        self._retire_driver(driver_a, reason=f"collision with {driver_b}")
        self._retire_driver(driver_b, reason=f"collision with {driver_a}")

    def _player_manual_control(self, driver_name: str) -> bool:
        if not self.player_team or self.player_auto_pit:
            return False
        return self.driver_team.get(driver_name) == self.player_team

    def _retire_driver(self, name, reason="DNF"):
        previous_context = getattr(self, "_active_race_incident_id", None)
        owns_context = previous_context is None
        if owns_context:
            self._active_race_incident_id = self._next_race_incident_id()
        try:
            return self._retire_driver_scoped(name, reason=reason)
        finally:
            if owns_context:
                self._active_race_incident_id = previous_context

    def _retire_driver_scoped(self, name, reason="DNF"):
        if name in self.finished:
            return
        try:
            position = next(
                (idx + 1 for idx, driver in enumerate(self.order) if getattr(driver, "name", None) == name),
                0,
            )
        except Exception:
            position = 0
        self.relationship_events.append(
            {
                "type": "retirement",
                "driver": str(name),
                "team": self.driver_team.get(name),
                "reason": str(reason or ""),
                "position": int(position),
            }
        )
        self.dnf[name] = True
        self._pending_spin_marker.pop(name, None)
        self._pending_spin_loss_s.pop(name, None)
        self._pending_spin_lap.pop(name, None)
        # snapshot distance covered for proper DNF ordering
        self.dnf_laps[name] = self.laps.get(name, 0)
        self.dnf_prog[name] = self.progress.get(name, 0.0)
        tnow = self.total_sim_time.get(name, 0.0)
        self.finished.add(name)
        self.finish_time[name] = tnow
        self.events.append(f"DNF: {name} {reason}")
        self.dnf_reason[name] = reason
        reason_text = str(reason or "DNF").lower()
        is_player = bool(self._player_driver(name))
        is_crash = any(word in reason_text for word in ("crash", "collision", "accident"))
        is_mechanical = any(word in reason_text for word in ("mechanical", "engine", "failure"))
        alert_type = "crash" if is_crash else "mechanical_retirement" if is_mechanical else "retirement"
        self._emit_race_alert(
            alert_type,
            "major" if is_crash or is_player else "minor",
            f"{name} has retired from the race",
            drivers=[name],
            outcomes=[f"Retirement reason: {str(reason or 'DNF')}"],
            player_involved=is_player,
        )
        try:
            if isinstance(reason, str) and any(word in reason.lower() for word in ("crash", "collision")):
                self._apply_crash_damage(name)
        except Exception:
            pass
        inj_cfg = self.cfg.get("injuries", {})
        fatal = False
        fatality_chance = float(inj_cfg.get("fatality_chance", 0.0) or 0.0)
        reason_text = reason.lower() if isinstance(reason, str) else ""
        if (
            fatality_chance > 0.0
            and reason_text
            and any(word in reason_text for word in ("crash", "collision", "accident"))
            and random.random() < fatality_chance
        ):
            fatal = True
            event_label = self.event_name or "the event"
            for d in self.drivers:
                if d.name == name:
                    d.injury_weeks = 0
                    d.deceased = True
                    d.death_processed = False
                    d.death_event = event_label
                    d.retiring = False
                    d.retired = True
                    break
            self.events.append(f"FATAL ACCIDENT: {name}")
            if self.email_manager:
                try:
                    self.email_manager.send_driver_fatality_email(name, event_label)
                except Exception:
                    pass

        if not fatal and random.random() < float(inj_cfg.get("dnf_injury_chance", 0.0)):
            severe = random.random() < float(inj_cfg.get("severe_chance", 0.0))
            weeks = int(
                inj_cfg.get("severe_weeks" if severe else "minor_weeks", 0)
            )
            for d in self.drivers:
                if d.name == name:
                    d.injury_weeks = weeks
                    self.events.append(
                        f"INJURY: {name} ({'severe' if severe else 'minor'})"
                    )
                    recovery_note = ""
                    malus_cfg = inj_cfg.get("recovery_malus", {})
                    if isinstance(malus_cfg, dict) and malus_cfg.get("enabled") and weeks > 0:
                        try:
                            major_chance = float(malus_cfg.get("major_chance", 0.0) or 0.0)
                        except Exception:
                            major_chance = 0.0
                        try:
                            minor_chance = float(malus_cfg.get("minor_chance", 0.0) or 0.0)
                        except Exception:
                            minor_chance = 0.0
                        major_chance = max(0.0, min(1.0, major_chance))
                        minor_chance = max(0.0, min(1.0, minor_chance))
                        roll = random.random()
                        outcome = "none"
                        if roll < major_chance:
                            outcome = "major"
                        elif roll < major_chance + minor_chance:
                            outcome = "minor"
                        if outcome != "none":
                            try:
                                delta_key = "major_attr_delta" if outcome == "major" else "minor_attr_delta"
                                attr_delta = int(malus_cfg.get(delta_key, 0) or 0)
                            except Exception:
                                attr_delta = 0
                            try:
                                min_attr = float(malus_cfg.get("min_attribute", 0) or 0.0)
                            except Exception:
                                min_attr = 0.0
                            if attr_delta > 0:
                                for attr in ("cornering", "braking", "consistency", "smoothness", "control"):
                                    try:
                                        current = float(getattr(d, attr, 0.0) or 0.0)
                                    except Exception:
                                        current = 0.0
                                    setattr(d, attr, max(min_attr, current - float(attr_delta)))
                            if outcome == "minor":
                                recovery_note = (
                                    "The driver is expected to come back into the sport shakey upon their return."
                                )
                            else:
                                recovery_note = (
                                    "The driver is expected to come back into the sport having to re-learn how to drive."
                                )
                        else:
                            recovery_note = (
                                "The driver is expected to make a full recovery upon their return to racing."
                            )
                    team = self.driver_team.get(name) or getattr(d, "team", "Unknown")
                    if self.email_manager:
                        try:
                            team_label = (
                                self.state.team_display_name(team)
                                if hasattr(self, "state") and team in getattr(self.state, "teams", {})
                                else str(team)
                            )
                        except Exception:
                            team_label = str(team)
                        self.email_manager.send_driver_injury_email(name, team_label, weeks, recovery_note=recovery_note)
                    break
        # Preserve the existing DNF trigger and universal deployment chance;
        # the active regulation decides which neutralization may follow.
        self._maybe_deploy_incident_caution()

    def _finish_driver(self, name: str, *, took_flag: bool = False) -> None:
        if not name or name in self.finished or self.dnf.get(name, False):
            return
        tnow = float(self.total_sim_time.get(name, 0.0))
        self.finished.add(name)
        self.finish_time[name] = tnow
        if took_flag and self.checkered_flag_time is None:
            self.checkered_flag_time = tnow
            self.checkered_flag_winner = name
            self._defer_event(f"FINISH: {name} takes the flag!")
        else:
            laps_down = max(0, int(self.total_laps) - int(self.laps.get(name, 0)))
            if laps_down > 0:
                suffix = "lap" if laps_down == 1 else "laps"
                self._defer_event(f"FINISH: {name} finishes {laps_down} {suffix} down.")
            else:
                self._defer_event(f"FINISH: {name} finishes the race.")
        self.fuel_onboard[name] = 0.0
        self.live_speed_kmh[name] = 0.0

    def _spin_incident_marker(self, unit_value: float) -> float:
        markers = list(getattr(self, "lockup_markers", []) or [])
        if _REALISTIC_KERNELS is not None:
            try:
                return float(
                    _REALISTIC_KERNELS.incident_marker_from_unit(
                        markers,
                        float(unit_value),
                        0.15,
                        0.85,
                    )
                )
            except Exception:
                pass
        unit = max(0.0, min(0.999999999999, float(unit_value)))
        valid_markers = [
            float(marker)
            for marker in markers
            if 0.02 < float(marker) < 0.98
        ]
        if valid_markers:
            return valid_markers[min(len(valid_markers) - 1, int(unit * len(valid_markers)))]
        return 0.15 + (0.70 * unit)

    def _trigger_spin(self, name, loss_s):
        """Schedule a spontaneous spin at an on-track marker on the current lap."""
        driver_name = str(name or "")
        if not driver_name or driver_name in self.finished or self.dnf.get(driver_name, False):
            return False
        if driver_name in self._pending_spin_marker:
            return False
        marker = self._spin_incident_marker(random.random())
        self._pending_spin_marker[driver_name] = float(marker)
        self._pending_spin_loss_s[driver_name] = max(0.0, float(loss_s))
        self._pending_spin_lap[driver_name] = int(self.laps.get(driver_name, 0) or 0)
        return True

    def _maybe_trigger_spin_between(self, name: str, start_prog: float, end_prog: float) -> bool:
        driver_name = str(name or "")
        marker = self._pending_spin_marker.get(driver_name)
        if marker is None:
            return False
        if int(self._pending_spin_lap.get(driver_name, -1)) != int(self.laps.get(driver_name, 0) or 0):
            return False
        crossed = False
        if _REALISTIC_KERNELS is not None:
            try:
                crossed = bool(
                    _REALISTIC_KERNELS.incident_segment_crosses_marker(
                        float(start_prog),
                        float(end_prog),
                        float(marker),
                    )
                )
            except Exception:
                crossed = False
        if not crossed:
            crossed = self._segment_crosses_marker(start_prog, end_prog, marker)
        if not crossed:
            return False

        loss_s = max(0.0, float(self._pending_spin_loss_s.pop(driver_name, 0.0) or 0.0))
        self._pending_spin_marker.pop(driver_name, None)
        self._pending_spin_lap.pop(driver_name, None)
        self.freeze_remaining[driver_name] = max(
            float(self.freeze_remaining.get(driver_name, 0.0) or 0.0),
            loss_s,
        )
        self.incident_flag_decay[driver_name] = max(
            int(self.incident_flag_decay.get(driver_name, 0) or 0),
            8,
        )
        self.live_speed_kmh[driver_name] = 0.0
        self.events.append(f"SPIN: {driver_name} -{loss_s:.1f}s")
        self._emit_race_alert(
            "spin",
            "minor",
            f"{driver_name} has spun",
            drivers=[driver_name],
            outcomes=[f"Estimated time loss: {loss_s:.1f}s"],
        )
        self._emit_radio_event("race_spin", driver_name)
        return True

    def _load_lockup_config(self, incidents_cfg):
        lock_cfg = incidents_cfg.get("lock_up", incidents_cfg.get("lockup", {}))
        if not isinstance(lock_cfg, dict):
            lock_cfg = {}
        self.lockup_enabled = bool(lock_cfg.get("enabled", True))
        try:
            self.lockup_chance_per_zone = max(0.0, float(lock_cfg.get("chance_per_braking_zone", 0.001)))
        except Exception:
            self.lockup_chance_per_zone = 0.001
        time_loss = lock_cfg.get("time_loss_s", [0.15, 0.9])
        wear_add = lock_cfg.get("tyre_wear_add", [0.005, 0.02])
        try:
            time_lo = max(0.0, float(time_loss[0]))
            time_hi = max(0.0, float(time_loss[1]))
        except Exception:
            time_lo, time_hi = 0.15, 0.9
        try:
            wear_lo = max(0.0, float(wear_add[0]))
            wear_hi = max(0.0, float(wear_add[1]))
        except Exception:
            wear_lo, wear_hi = 0.005, 0.02
        if time_hi < time_lo:
            time_lo, time_hi = time_hi, time_lo
        if wear_hi < wear_lo:
            wear_lo, wear_hi = wear_hi, wear_lo
        self.lockup_time_loss_range = (time_lo, time_hi)
        self.lockup_tyre_wear_range = (wear_lo, wear_hi)
        self.lockup_markers = braking_zone_markers(
            getattr(self.realistic_physics, "micro_sectors", []) if self.realistic_physics else []
        )
        self._lockup_marker_lap = {d.name: {} for d in self.drivers}

    def _maybe_trigger_lockup_between(self, driver_name: str, start_prog: float, end_prog: float) -> None:
        if not self.lockup_enabled or self.lockup_chance_per_zone <= 0.0 or not self.lockup_markers:
            return
        name = str(driver_name or "")
        if not name or name in self.finished or self.dnf.get(name, False):
            return
        lap_idx = int(self.laps.get(name, 0) or 0)
        marker_laps = self._lockup_marker_lap.setdefault(name, {})
        for marker_idx, marker in enumerate(self.lockup_markers):
            if not self._segment_crosses_marker(start_prog, end_prog, marker):
                continue
            if int(marker_laps.get(marker_idx, -1)) == lap_idx:
                continue
            marker_laps[marker_idx] = lap_idx
            lockup_chance = self._caution_driver_error_probability(
                self.lockup_chance_per_zone
            )
            if random.random() >= lockup_chance:
                continue
            loss_lo, loss_hi = self.lockup_time_loss_range
            wear_lo, wear_hi = self.lockup_tyre_wear_range
            loss_s = random.uniform(loss_lo, loss_hi) if loss_hi > loss_lo else loss_lo
            wear_add = random.uniform(wear_lo, wear_hi) if wear_hi > wear_lo else wear_lo
            self.freeze_remaining[name] = float(self.freeze_remaining.get(name, 0.0) or 0.0) + float(loss_s)
            self.incident_flag_decay[name] = max(int(self.incident_flag_decay.get(name, 0) or 0), 6)
            if self.oval_four_tyre_enabled:
                per_corner = self.tyre_wear_by_corner.setdefault(
                    name, {key: 0.0 for key in OVAL_TYRE_KEYS}
                )
                # A lock-up is principally a front-axle event. Keep both
                # fronts affected because the current track data does not yet
                # author which wheel locked.
                for key in ("left_front", "right_front"):
                    per_corner[key] = min(
                        1.5,
                        max(
                            0.0,
                            float(per_corner.get(key, 0.0) or 0.0)
                            + float(wear_add),
                        ),
                    )
                self._sync_oval_aggregate_tyre_wear(name)
            else:
                self.tyre_wear[name] = min(
                    1.0,
                    max(
                        0.0,
                        float(self.tyre_wear.get(name, 0.0) or 0.0)
                        + float(wear_add),
                    ),
                )
            self.events.append(f"LOCK-UP: {name} locked up and lost {loss_s:.2f}s")
            self._emit_radio_event("race_lockup", name)
            break

    def _distance_lap_length_m(self) -> float:
        cached = getattr(self, "_distance_lap_length_cache_m", None)
        if cached is not None:
            return float(cached)
        try:
            length = float(getattr(self.realistic_physics, "track_length_m", 0.0))
        except Exception:
            length = 0.0
        if length <= 1e-6:
            try:
                length = float(getattr(self.track, "total_len", 0.0))
            except Exception:
                length = 0.0
        cached = max(1.0, length)
        self._distance_lap_length_cache_m = float(cached)
        return float(cached)

    def _driver_distance_along_track_m(self, name: str) -> float:
        lap = max(0.0, float(self.laps.get(name, 0)))
        try:
            prog = float(self.progress.get(name, 0.0))
        except Exception:
            prog = 0.0
        prog = max(0.0, min(1.0, prog))
        total_laps = lap + prog
        if bool((self.grid_start_pending or {}).get(name, False)):
            total_laps -= 1.0
        return total_laps * self._distance_lap_length_m()

    def _initialize_distance_gap_tracking(self) -> None:
        for d in self.drivers:
            name = d.name
            s = self._driver_distance_along_track_m(name)
            self.distance_along_track_m[name] = float(s)
            hist = self.distance_time_history.setdefault(
                name, deque(maxlen=GAP_HISTORY_MAX_SAMPLES)
            )
            hist.clear()
            hist.append((0.0, float(s)))

    def _record_distance_sample(self, name: str) -> None:
        now_t = float(self.total_sim_time.get(name, 0.0))
        dist = float(self._driver_distance_along_track_m(name))
        hist = self.distance_time_history.setdefault(
            name, deque(maxlen=GAP_HISTORY_MAX_SAMPLES)
        )
        if hist:
            last_t, last_s = hist[-1]
            # Keep distance monotonic for stable inverse interpolation.
            if dist < float(last_s):
                dist = float(last_s)
            if abs(now_t - float(last_t)) <= 1e-9:
                hist[-1] = (now_t, dist)
                self.distance_along_track_m[name] = dist
                return
            if abs(dist - float(last_s)) <= 1e-6 and (now_t - float(last_t)) <= 1e-3:
                self.distance_along_track_m[name] = dist
                return
        hist.append((now_t, dist))
        self.distance_along_track_m[name] = dist

    def _time_at_distance_from_history(self, name: str, target_s_m: float) -> Optional[float]:
        hist = self.distance_time_history.get(name)
        if not hist:
            return None
        pts = list(hist)
        if _REALISTIC_KERNELS is not None:
            try:
                t_fast = float(
                    _REALISTIC_KERNELS.time_at_distance_from_points(
                        pts,
                        float(target_s_m),
                    )
                )
                if not math.isnan(t_fast):
                    return t_fast
            except Exception:
                pass
        if len(pts) == 1:
            t0, s0 = pts[0]
            if target_s_m <= float(s0) + 1e-9:
                return float(t0)
            return None
        first_t, first_s = pts[0]
        if target_s_m <= float(first_s) + 1e-9:
            return float(first_t)
        last_t, last_s = pts[-1]
        if target_s_m > float(last_s) + 1e-9:
            return None
        lo = 0
        hi = len(pts) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if float(pts[mid][1]) < target_s_m:
                lo = mid
            else:
                hi = mid
        t0, s0 = pts[lo]
        t1, s1 = pts[hi]
        s0 = float(s0)
        s1 = float(s1)
        t0 = float(t0)
        t1 = float(t1)
        if abs(s1 - s0) <= 1e-9:
            return max(t0, t1)
        w = (target_s_m - s0) / (s1 - s0)
        return t0 + (t1 - t0) * max(0.0, min(1.0, w))

    def _predict_time_at_distance(self, name: str, target_s_m: float) -> Optional[float]:
        now_t = float(self.total_sim_time.get(name, 0.0))
        now_s = float(self.distance_along_track_m.get(name, self._driver_distance_along_track_m(name)))
        delta_s = float(target_s_m) - now_s
        if delta_s <= 1e-9:
            return now_t
        lap_len = self._distance_lap_length_m()
        if lap_len <= 1e-9:
            return None

        drv = self.driver_by_name.get(name)
        if drv is None:
            return None

        if self.use_realistic_physics:
            try:
                lap_inputs = self._realistic_lap_inputs_for_driver(
                    drv,
                    apply_consistency=True,
                    apply_aero=False,
                )
                delta_progress = max(0.0, delta_s / lap_len)
                dt = self.realistic_physics.elapsed_for_progress_delta(
                    lap_inputs,
                    self.progress.get(name, 0.0),
                    delta_progress,
                )
                return now_t + max(0.0, float(dt))
            except Exception:
                pass

        speed_ms = max(0.5, float(self.live_speed_kmh.get(name, 0.0)) / 3.6)
        return now_t + (delta_s / speed_ms)

    def distance_reference_gap(self, name: str, reference_name: str) -> tuple[int, Optional[float]]:
        if not name or not reference_name:
            return 0, None
        if name == reference_name:
            return 0, 0.0

        laps_name = int(self.laps.get(name, 0) or 0)
        laps_ref = int(self.laps.get(reference_name, 0) or 0)
        lap_diff = laps_ref - laps_name
        if lap_diff >= 1:
            return int(lap_diff), None

        if (
            name in self.finished
            and reference_name in self.finished
            and name in self.finish_time
            and reference_name in self.finish_time
        ):
            try:
                return 0, max(0.0, float(self.finish_time[name]) - float(self.finish_time[reference_name]))
            except Exception:
                return 0, None

        ref_s = float(
            self.distance_along_track_m.get(
                reference_name,
                self._driver_distance_along_track_m(reference_name),
            )
        )
        ref_now_t = float(self.total_sim_time.get(reference_name, 0.0))

        t_at_ref_s = self._time_at_distance_from_history(name, ref_s)
        if t_at_ref_s is None:
            t_at_ref_s = self._predict_time_at_distance(name, ref_s)
        if t_at_ref_s is None:
            return 0, None

        gap = float(t_at_ref_s) - ref_now_t
        return 0, max(0.0, gap)

    def gap_snapshot(self, mode: str = "leader") -> dict:
        """Return leader/ahead timing gaps cached for the current manager tick."""
        mode_key = "relative" if str(mode) == "relative" else "leader"
        if (
            self._gap_snapshot_tick == self.tick_count
            and self._gap_snapshot_mode == mode_key
            and isinstance(self._gap_snapshot, dict)
        ):
            return self._gap_snapshot
        try:
            order = list(self.order)
        except Exception:
            order = []
        running = [d for d in order if getattr(d, "name", None) not in self.finished]
        leader = running[0] if running else (order[0] if order else None)
        leader_name = getattr(leader, "name", None)
        ahead_by_name = {}
        leader_gaps = {}
        relative_gaps = {}
        prev = None
        for d in order:
            name = str(getattr(d, "name", "") or "")
            if not name:
                continue
            ahead_name = str(getattr(prev, "name", "") or "") if prev is not None else None
            ahead_by_name[name] = ahead_name
            if mode_key == "leader" and leader_name:
                try:
                    leader_gaps[name] = (
                        (0, 0.0)
                        if name == leader_name
                        else self.distance_reference_gap(name, leader_name)
                    )
                except Exception:
                    leader_gaps[name] = (0, None)
            if mode_key == "relative" and ahead_name:
                try:
                    relative_gaps[name] = self.distance_reference_gap(name, ahead_name)
                except Exception:
                    relative_gaps[name] = (0, None)
            elif mode_key == "relative":
                relative_gaps[name] = (0, None)
            prev = d
        snapshot = {
            "leader": leader_name,
            "ahead": ahead_by_name,
            "leader_gaps": leader_gaps,
            "relative_gaps": relative_gaps,
        }
        self._gap_snapshot_tick = self.tick_count
        self._gap_snapshot_mode = mode_key
        self._gap_snapshot = snapshot
        return snapshot

    def _snap_to_tail(self, name, prev_tail):
        self.laps[name] = self.laps.get(prev_tail, 0)
        self.progress[name] = self.progress.get(prev_tail, 0.0) - EPSILON
        if self.progress[name] < 0:
            self.progress[name] += 1.0
            self.laps[name] -= 1

    def _oval_formation_lane_ids(self) -> tuple[Optional[str], Optional[str]]:
        racecraft = getattr(self, "oval_racecraft", None)
        geometry = getattr(racecraft, "geometry", None)
        lines = list(getattr(geometry, "lines", []) or [])
        if not lines:
            return None, None
        line_ids = [str(row.get("id") or "") for row in lines if isinstance(row, dict)]
        line_ids = [line_id for line_id in line_ids if line_id]
        if not line_ids:
            return None, None
        preferred = str(getattr(geometry, "preferred_line", line_ids[0]) or line_ids[0])
        try:
            preferred_index = line_ids.index(preferred)
        except ValueError:
            preferred_index = 0
            preferred = line_ids[0]
        if len(line_ids) == 1:
            return preferred, preferred
        adjacent_index = preferred_index + 1 if preferred_index + 1 < len(line_ids) else preferred_index - 1
        return preferred, line_ids[adjacent_index]

    def _set_oval_double_file_lanes(self, ordered_names) -> None:
        racecraft = getattr(self, "oval_racecraft", None)
        if racecraft is None:
            return
        inside_line, outside_line = self._oval_formation_lane_ids()
        if not inside_line or not outside_line:
            return
        geometry = getattr(racecraft, "geometry", None)
        for idx, raw_name in enumerate(ordered_names or []):
            name = str(raw_name or "")
            state = (getattr(racecraft, "states", {}) or {}).get(name)
            if state is None:
                continue
            lane_id = inside_line if idx % 2 == 0 else outside_line
            try:
                lateral_m = float(geometry.line_offset_m(lane_id))
            except Exception:
                lateral_m = 0.0
            state.lane_current = lane_id
            state.lane_target = lane_id
            state.lateral_position_m = lateral_m
            state.lane_change_start_distance_m = 0.0
            state.lane_change_length_m = 0.0
            state.lane_change_start_lateral_m = lateral_m
            state.lane_change_target_lateral_m = lateral_m
            state.racecraft_intent = "pace_formation"
            state.battle_opponent = None
            state.battle_kind = "position"
            state.cooldown_until_distance_m = 0.0
            state.hold_line_until_distance_m = 0.0

    def _reset_oval_formation_grid(self, ordered_names) -> None:
        if not self.oval_pace_lap_start_enabled:
            return
        names = [str(name) for name in (ordered_names or []) if str(name) in self.driver_by_name]
        if not names:
            names = list(self.starting_grid)
        seen = set(names)
        for driver in self.drivers:
            if driver.name not in seen:
                names.append(driver.name)
                seen.add(driver.name)
        track_len_m = self._safety_car_track_length_m()
        formation_distances = {}
        for idx, name in enumerate(names):
            row_index = idx // 2
            distance_m = -float(self.oval_pace_car_leader_gap_m) - (
                float(row_index) * float(self.oval_pace_row_gap_m)
            )
            formation_distances[name] = distance_m
            self.progress[name] = (distance_m / track_len_m) % 1.0
            self.live_speed_kmh[name] = float(self.oval_pace_speed_kmh)
            self.distance_along_track_m[name] = distance_m
            self.laps[name] = 0
            self.grid_start_pending[name] = True
        self._oval_formation_distance_m_by_driver = formation_distances
        self.sc_distance_m = 0.0
        self.sc_progress = 0.0
        self.sc_live_speed_kmh = float(self.oval_pace_speed_kmh)
        self._set_oval_double_file_lanes(names)

    def _initialize_oval_formation_start(self, ordered_names=None) -> None:
        if not self.oval_pace_lap_start_enabled:
            return
        track_len_m = self._safety_car_track_length_m()
        self.oval_formation_active = True
        self.oval_formation_complete = False
        self._oval_formation_elapsed_s = 0.0
        self._oval_formation_target_sc_distance_m = track_len_m * float(self.oval_pace_laps)
        self.sc_active = True
        self.sc_phase = "formation_start"
        self.sc_laps_remaining = int(self.oval_pace_laps)
        self.sc_leader = self.starting_grid[0] if self.starting_grid else None
        self.sc_train = list(ordered_names or self.starting_grid)
        self._sc_clear_on_line = False
        self._reset_oval_formation_grid(ordered_names or self.starting_grid)
        self._strategy_last_sc_active = True
        self.events.append(
            f"PACE LAP: double-file formation at {self.oval_pace_speed_kmh / 1.609344:.0f} mph"
        )

    def _release_oval_formation_start(self) -> None:
        if not self.oval_formation_active:
            return
        track_len_m = self._safety_car_track_length_m()
        release_line_m = max(
            track_len_m,
            float(self._oval_formation_target_sc_distance_m or track_len_m),
        )
        self.oval_formation_active = False
        self.oval_formation_complete = True
        self.sc_active = False
        self.sc_phase = None
        self.sc_laps_remaining = 0
        self.sc_leader = None
        self.sc_train = []
        self.sc_progress = 0.0
        self.sc_distance_m = 0.0
        self.sc_live_speed_kmh = 0.0
        self._sc_clear_on_line = False
        for driver in self.drivers:
            name = driver.name
            absolute_distance = float(
                self._oval_formation_distance_m_by_driver.get(name, release_line_m)
            )
            race_distance = absolute_distance - release_line_m
            self.progress[name] = (race_distance / track_len_m) % 1.0
            self.laps[name] = 0
            self.grid_start_pending[name] = True
            self.total_sim_time[name] = 0.0
            self.current_lap_start[name] = 0.0
            self.current_sector_index[name] = 0
            self.current_sector_start_time[name] = 0.0
            self.current_lap_sector_times[name] = []
            self.current_lap_peak_speed_kmh[name] = float(self.oval_pace_speed_kmh)
            self.distance_along_track_m[name] = race_distance
            self.sc_lap_active[name] = False
            self._oval_release_elapsed_s[name] = 0.0
            lane_state = (getattr(getattr(self, "oval_racecraft", None), "states", {}) or {}).get(name)
            if lane_state is not None:
                lane_state.racecraft_intent = "follow"
        self._initialize_distance_gap_tracking()
        self._strategy_last_sc_active = False
        self.events.append("GREEN FLAG: pace car in, double-file field released")

    def _abandon_oval_formation_for_active_race(self) -> None:
        """Drop a stale formation state when restoring or externally seeding a live race."""

        if not self.oval_formation_active:
            return
        self.oval_formation_active = False
        self.oval_formation_complete = True
        self.sc_active = False
        self.sc_phase = None
        self.sc_laps_remaining = 0
        self.sc_leader = None
        self.sc_train = []
        self.sc_progress = 0.0
        self.sc_distance_m = 0.0
        self.sc_live_speed_kmh = 0.0
        self._sc_clear_on_line = False
        self._strategy_last_sc_active = False
        racecraft = getattr(self, "oval_racecraft", None)
        geometry = getattr(racecraft, "geometry", None)
        preferred_line = str(getattr(geometry, "preferred_line", "") or "")
        for driver in self.drivers:
            name = driver.name
            self._oval_release_elapsed_s[name] = float(self.oval_release_acceleration_s)
            self.sc_lap_active[name] = False
            lane_state = (getattr(racecraft, "states", {}) or {}).get(name)
            if lane_state is not None and preferred_line:
                lateral_m = float(geometry.line_offset_m(preferred_line))
                lane_state.lane_current = preferred_line
                lane_state.lane_target = preferred_line
                lane_state.lateral_position_m = lateral_m
                lane_state.lane_change_length_m = 0.0
                lane_state.lane_change_start_lateral_m = lateral_m
                lane_state.lane_change_target_lateral_m = lateral_m
                lane_state.racecraft_intent = "follow"

    def _advance_oval_formation_start(self, dt_sim: float) -> None:
        if not self.oval_formation_active or dt_sim <= 1e-9:
            return
        step_s = max(0.0, float(dt_sim))
        distance_step_m = (float(self.oval_pace_speed_kmh) / 3.6) * step_s
        self._oval_formation_elapsed_s += step_s
        self.sc_distance_m += distance_step_m
        track_len_m = self._safety_car_track_length_m()
        self.sc_progress = (self.sc_distance_m / track_len_m) % 1.0
        self.sc_live_speed_kmh = float(self.oval_pace_speed_kmh)
        for driver in self.drivers:
            name = driver.name
            distance_m = float(self._oval_formation_distance_m_by_driver.get(name, 0.0)) + distance_step_m
            self._oval_formation_distance_m_by_driver[name] = distance_m
            self.progress[name] = (distance_m / track_len_m) % 1.0
            self.distance_along_track_m[name] = distance_m
            self.live_speed_kmh[name] = float(self.oval_pace_speed_kmh)
        if self.sc_distance_m + 1e-9 >= float(self._oval_formation_target_sc_distance_m):
            self._release_oval_formation_start()

    def _safety_car_cfg_float(self, key: str, default: float) -> float:
        try:
            raw = self.cfg.get("safety_car", {}).get(key, default)
            return float(raw)
        except Exception:
            return float(default)

    @property
    def caution_active(self) -> bool:
        """Whether an in-race physical or virtual neutralization is active."""

        return bool(self.sc_active or self.vsc_active)

    def _caution_driver_error_probability(self, probability: float) -> float:
        """Temporarily suppress a driver-error roll during an SC or VSC.

        Callers pass their final calculated probability so driver, weather,
        track and setup modifiers retain their existing behavior. Green-flag
        probabilities are returned exactly as supplied.
        """

        value = float(probability)
        if self.caution_active:
            return value * CAUTION_DRIVER_ERROR_CHANCE_MULTIPLIER
        return value

    def _vsc_selection_probability(self) -> float:
        return max(
            0.0,
            min(
                1.0,
                self._safety_car_cfg_float("vsc_selection_probability", 0.5),
            ),
        )

    def _maybe_deploy_incident_caution(self) -> None:
        """Apply the active regulation after the universal DNF roll."""

        if self.caution_active or not self.order:
            return
        sc_cfg = self.cfg.get("safety_car", {}) if isinstance(self.cfg, dict) else {}
        try:
            deploy_chance = max(
                0.0,
                min(1.0, float(sc_cfg.get("dnf_deploy_chance", 0.2) or 0.0)),
            )
        except Exception:
            deploy_chance = 0.2
        # Consume this roll even when the regulation prohibits intervention so
        # unrelated race RNG stays aligned with the historical SC path.
        if random.random() >= deploy_chance:
            return
        if self._safety_car_policy == "none":
            return
        if self._safety_car_policy == "physical_only":
            self._deploy_safety_car()
            return
        if random.random() < self._vsc_selection_probability():
            self._deploy_virtual_safety_car()
        else:
            self._deploy_safety_car()

    def _safety_car_track_length_m(self) -> float:
        return max(1.0, float(self._distance_lap_length_m()))

    def _configured_safety_car_pickup_offset_m(
        self,
        track_len_m: Optional[float] = None,
    ) -> float:
        """Return the physical post-line pickup offset for this track.

        The configured 200 m default gives pit entry and classification time to
        settle on normal circuits.  The proportional cap keeps the same rule
        sensible on unusually short tracks and short ovals.
        """

        length = max(
            1.0,
            float(
                self._safety_car_track_length_m()
                if track_len_m is None
                else track_len_m
            ),
        )
        configured = max(
            0.0,
            self._safety_car_cfg_float("pickup_offset_m", 200.0),
        )
        return min(configured, length * 0.08)

    def _current_on_track_leader_name(self, order_list=None) -> Optional[str]:
        source = order_list if isinstance(order_list, list) else self.order
        for drv in source or []:
            name = str(getattr(drv, "name", "") or "")
            if not name or name in self.finished or self.dnf.get(name, False):
                continue
            if self.pit_remaining.get(name, 0.0) > 0.0:
                continue
            return name
        return None

    def _current_classified_leader_name(self, order_list=None) -> Optional[str]:
        """Return the running P1, including a leader currently in the pits."""

        source = order_list if isinstance(order_list, list) else self.order
        for drv in source or []:
            name = str(getattr(drv, "name", "") or "")
            if not name or name in self.finished or self.dnf.get(name, False):
                continue
            return name
        return None

    def _safety_car_immediate_ahead_map(self, order_list) -> dict[str, Optional[str]]:
        """Map each active car to its nearest classified caution predecessor.

        Pit stops, retirements and cars immobilized by an incident are deliberately
        transparent so the field may pass them. Ordinary pace differences are not.
        """

        out: dict[str, Optional[str]] = {}
        ahead_name = None
        for drv in order_list or []:
            candidate = str(getattr(drv, "name", "") or "")
            if not candidate or candidate in self.finished or self.dnf.get(candidate, False):
                continue
            if self.pit_remaining.get(candidate, 0.0) > 0.0:
                continue
            if self.freeze_remaining.get(candidate, 0.0) > 0.0:
                continue
            if (
                self.laps.get(candidate, 0) == 0
                and (
                    self.start_delay_remaining.get(candidate, 0.0) > 0.0
                    or self.start_reaction_delay_remaining.get(candidate, 0.0) > 0.0
                )
            ):
                continue
            out[candidate] = ahead_name
            ahead_name = candidate
        return out

    def _safety_car_target_speed_kmh(self, progress: Optional[float] = None) -> float:
        prog = float(self.sc_progress if progress is None else progress)
        target = 0.0
        if self.use_realistic_physics and getattr(self.realistic_physics, "enabled", False):
            try:
                target = 0.5 * float(
                    self.realistic_physics.speed_kmh_at_progress(
                        self._sc_profile_inputs,
                        prog,
                        cache_key="__safety_car__",
                    )
                    or 0.0
                )
            except Exception:
                target = 0.0
        if target <= 1e-6:
            leader_name = self._current_on_track_leader_name()
            if leader_name:
                try:
                    target = 0.5 * float(self.live_speed_kmh.get(leader_name, 0.0) or 0.0)
                except Exception:
                    target = 0.0
        return max(60.0, float(target or 0.0))

    def _vsc_target_speed_kmh(
        self,
        progress: float,
        fallback_green_speed_kmh: float = 0.0,
    ) -> float:
        """Return the common track-relative VSC delta speed."""

        factor = max(
            0.20,
            min(0.95, self._safety_car_cfg_float("vsc_pace_factor", 0.65)),
        )
        target = 0.0
        if self.use_realistic_physics and getattr(self.realistic_physics, "enabled", False):
            try:
                target = factor * float(
                    self.realistic_physics.speed_kmh_at_progress(
                        self._sc_profile_inputs,
                        float(progress),
                        cache_key="__virtual_safety_car__",
                    )
                    or 0.0
                )
            except Exception:
                target = 0.0
        if target <= 1e-6:
            target = factor * max(0.0, float(fallback_green_speed_kmh or 0.0))
        return max(20.0, float(target or 0.0))

    def _deploy_safety_car(self) -> None:
        if (
            self.caution_active
            or self._safety_car_policy not in {"physical_only", "all_types"}
            or not self.order
        ):
            return
        leader_name = self._current_on_track_leader_name() or getattr(self.order[0], "name", None)
        if not leader_name:
            return
        leader_dist = float(
            self.distance_along_track_m.get(
                leader_name,
                self._driver_distance_along_track_m(leader_name),
            )
        )
        track_len_m = self._safety_car_track_length_m()
        self._clear_safety_car_pit_order()
        self._safety_car_period_serial = (
            int(getattr(self, "_safety_car_period_serial", 0) or 0) + 1
        )
        self.sc_active = True
        self.sc_phase = "awaiting_pickup"
        self.sc_laps_remaining = 0
        self.sc_leader = None
        self.sc_train = []
        self._sc_pickup_candidate = None
        self._sc_collection_countdown_armed = False
        self.sc_collect_gap_s = max(
            0.2,
            self._safety_car_cfg_float("queue_gap_s", 0.3)
            * SAFETY_CAR_QUEUE_SPACING_MULTIPLIER,
        )
        self.sc_leader_gap_s = max(
            self.sc_collect_gap_s,
            self._safety_car_cfg_float("leader_gap_s", 1.0),
        )
        self._sc_clear_on_line = False
        # The SC waits shortly beyond the next start/finish crossing. Pit entry
        # therefore happens before pickup, allowing the on-track order to settle
        # before the current P1 reaches the stationary car.
        leader_lap_index = int(math.floor(max(0.0, leader_dist) / track_len_m))
        self._sc_pickup_offset_m = self._configured_safety_car_pickup_offset_m(
            track_len_m
        )
        self._sc_pickup_line_distance_m = float(
            ((leader_lap_index + 1) * track_len_m) + self._sc_pickup_offset_m
        )
        self.sc_distance_m = float(self._sc_pickup_line_distance_m)
        self.sc_progress = math.fmod(self.sc_distance_m / track_len_m, 1.0)
        if self.sc_progress < 0.0:
            self.sc_progress += 1.0
        self.sc_live_speed_kmh = 0.0
        self.events.append("SAFETY CAR DEPLOYED")
        self._emit_race_alert(
            "safety_car",
            "major",
            "The race has been neutralized",
        )
        self._emit_player_team_radio("race_safety_car")
        for name in self.sc_lap_active.keys():
            self.sc_lap_active[name] = False
        try:
            self.strategy.begin_safety_car_period(self)
        except Exception as exc:
            self._record_strategy_error("safety car deployment", exc)
        racecraft = getattr(self, "racecraft_model", None)
        begin_caution = getattr(racecraft, "begin_safety_car_period", None)
        if callable(begin_caution):
            try:
                begin_caution(self)
            except Exception:
                pass
        self._strategy_update_pending = True

    def _deploy_virtual_safety_car(self) -> None:
        """Neutralize racing at a common delta without collecting the field."""

        if (
            self.caution_active
            or self._safety_car_policy != "all_types"
            or not self.order
        ):
            return
        duration_min = max(
            1.0,
            self._safety_car_cfg_float("vsc_duration_s_min", 45.0),
        )
        duration_max = max(
            duration_min,
            self._safety_car_cfg_float("vsc_duration_s_max", 90.0),
        )
        self._clear_safety_car_pit_order()
        self._safety_car_period_serial = (
            int(getattr(self, "_safety_car_period_serial", 0) or 0) + 1
        )
        self.vsc_duration_s = float(random.uniform(duration_min, duration_max))
        self.vsc_remaining_s = float(self.vsc_duration_s)
        self.vsc_elapsed_s = 0.0
        self.vsc_active = True
        self.events.append("VIRTUAL SAFETY CAR DEPLOYED")
        self._emit_race_alert(
            "vsc",
            "major",
            "The race has been neutralized under Virtual Safety Car conditions",
        )
        self._emit_player_team_radio("race_vsc")
        try:
            self.strategy.begin_safety_car_period(self)
        except Exception as exc:
            self._record_strategy_error("virtual safety car deployment", exc)
        racecraft = getattr(self, "racecraft_model", None)
        begin_caution = getattr(racecraft, "begin_safety_car_period", None)
        if callable(begin_caution):
            try:
                begin_caution(self)
            except Exception:
                pass
        self._strategy_update_pending = True

    def _clear_virtual_safety_car(self) -> None:
        if not self.vsc_active:
            return
        self.vsc_active = False
        self.vsc_duration_s = 0.0
        self.vsc_remaining_s = 0.0
        self.vsc_elapsed_s = 0.0
        try:
            self.strategy.end_safety_car_period()
        except Exception as exc:
            self._record_strategy_error("virtual safety car clearance", exc)
        self._clear_safety_car_pit_order()
        racecraft = getattr(self, "racecraft_model", None)
        end_caution = getattr(racecraft, "end_safety_car_period", None)
        if callable(end_caution):
            try:
                end_caution(self)
            except Exception:
                pass
        # Append immediately so the telemetry recorder timestamps the exact
        # substep at which the VSC expires, including at accelerated speeds.
        self.events.append("VIRTUAL SAFETY CAR ENDED - GREEN FLAG")
        self._strategy_update_pending = True

    def _advance_virtual_safety_car(self, dt_sim: float) -> None:
        if not self.vsc_active or dt_sim <= 1e-9:
            return
        elapsed = max(0.0, float(dt_sim))
        self.vsc_elapsed_s = min(
            float(self.vsc_duration_s),
            float(self.vsc_elapsed_s) + elapsed,
        )
        self.vsc_remaining_s = max(0.0, float(self.vsc_remaining_s) - elapsed)
        if self.vsc_remaining_s <= 1e-9:
            self._clear_virtual_safety_car()

    def _safety_car_pickup_crossers(
        self,
        distance_before: dict,
        distance_after: dict,
    ) -> set[str]:
        """Return cars whose absolute-distance segment crossed the waiting SC."""

        if self.sc_phase != "awaiting_pickup":
            return set()
        pickup_distance = float(self._sc_pickup_line_distance_m or 0.0)
        if pickup_distance <= 0.0:
            return set()
        crossed = set()
        for raw_name, raw_after in (distance_after or {}).items():
            name = str(raw_name or "")
            if not name:
                continue
            try:
                before = float((distance_before or {}).get(name, raw_after))
                after = float(raw_after)
            except (TypeError, ValueError):
                continue
            if before < pickup_distance - 1e-9 and after >= pickup_distance - 1e-9:
                crossed.add(name)
        return crossed

    def _try_start_safety_car_pickup(
        self,
        order_list,
        pickup_crossers,
        distance_after,
    ) -> bool:
        """Pick up the classified, on-track P1 once it reaches the waiting SC."""

        if not self.sc_active or self.sc_phase != "awaiting_pickup":
            return False
        leader_name = self._current_classified_leader_name(order_list)
        self._sc_pickup_candidate = None
        if not leader_name:
            return False
        if self.pit_remaining.get(leader_name, 0.0) > 0.0:
            return False
        if self.freeze_remaining.get(leader_name, 0.0) > 0.0:
            return False
        try:
            if leader_name in (distance_after or {}):
                raw_distance = distance_after[leader_name]
            elif leader_name in self.distance_along_track_m:
                raw_distance = self.distance_along_track_m[leader_name]
            else:
                raw_distance = self._driver_distance_along_track_m(leader_name)
            leader_distance = float(raw_distance)
        except (TypeError, ValueError):
            return False
        reached_pickup = (
            leader_name in set(pickup_crossers or ())
            or leader_distance >= float(self._sc_pickup_line_distance_m) - 1e-9
        )
        if not reached_pickup:
            return False
        self._sc_pickup_candidate = leader_name
        self._start_safety_car_collecting(leader_name)
        return True

    def _start_safety_car_collecting(self, leader_name: str) -> None:
        if not leader_name:
            return
        self.sc_phase = "collecting"
        self.sc_laps_remaining = 3
        self.sc_leader = str(leader_name)
        self.sc_train = [self.sc_leader]
        self._sc_pickup_candidate = None
        self._sc_collection_countdown_armed = False
        track_len_m = self._safety_car_track_length_m()
        leader_distance = float(
            self.distance_along_track_m.get(
                self.sc_leader,
                self._driver_distance_along_track_m(self.sc_leader),
            )
        )
        target_speed = self._safety_car_target_speed_kmh(0.0)
        leader_gap_m = self._safety_car_gap_m(target_speed, self.sc_leader_gap_s)
        self.sc_distance_m = max(
            float(self._sc_pickup_line_distance_m),
            leader_distance + leader_gap_m,
        )
        self.sc_progress = math.fmod(self.sc_distance_m / track_len_m, 1.0)
        if self.sc_progress < 0.0:
            self.sc_progress += 1.0
        self.sc_live_speed_kmh = float(target_speed)
        self._sc_clear_on_line = False
        for name in self.sc_lap_active.keys():
            self.sc_lap_active[name] = True
        self._strategy_update_pending = True

    def _start_safety_car_return(self) -> None:
        if not self.sc_active:
            return
        self.sc_phase = "returning"
        self._sc_clear_on_line = True
        self._defer_event("SAFETY CAR IN THIS LAP")
        self._strategy_update_pending = True

    def _clear_safety_car(self) -> None:
        self.sc_active = False
        try:
            self.strategy.end_safety_car_period()
        except Exception as exc:
            self._record_strategy_error("safety car clearance", exc)
        self.sc_phase = None
        self.sc_laps_remaining = 0
        self.sc_leader = None
        self.sc_train = []
        self.sc_progress = 0.0
        self.sc_distance_m = 0.0
        self.sc_live_speed_kmh = 0.0
        self._sc_pickup_line_distance_m = 0.0
        self._sc_pickup_offset_m = 0.0
        self._sc_pickup_candidate = None
        self._sc_collection_countdown_armed = False
        self._sc_clear_on_line = False
        self._clear_safety_car_pit_order()
        racecraft = getattr(self, "racecraft_model", None)
        end_caution = getattr(racecraft, "end_safety_car_period", None)
        if callable(end_caution):
            try:
                end_caution(self)
            except Exception:
                pass
        for name in self.sc_lap_active.keys():
            self.sc_lap_active[name] = False
            if self.driver_pace_modes.is_ai_controlled(name):
                self._ai_race_push_window[name] = max(
                    int(self._ai_race_push_window.get(name, 0) or 0),
                    3,
                )
            if self.driver_ers_modes.is_ai_controlled(name):
                self._ai_race_ers_attack_window[name] = max(
                    int(self._ai_race_ers_attack_window.get(name, 0) or 0),
                    1,
                )
        self._defer_event("SAFETY CAR IN")
        self._strategy_update_pending = True

    def _advance_safety_car(self, dt_sim: float) -> bool:
        if not self.sc_active or dt_sim <= 1e-9:
            return False
        if self.sc_phase == "awaiting_pickup":
            track_len_m = self._safety_car_track_length_m()
            self.sc_progress = math.fmod(self.sc_distance_m / track_len_m, 1.0)
            if self.sc_progress < 0.0:
                self.sc_progress += 1.0
            self.sc_live_speed_kmh = 0.0
            return False
        prev_distance = float(self.sc_distance_m)
        prev_speed_ms = max(0.0, float(self.sc_live_speed_kmh) / 3.6)
        target_speed_ms = max(0.0, float(self._safety_car_target_speed_kmh()) / 3.6)
        accel_step = max(0.0, float(BASE_ACCEL_MS2) * 0.5 * dt_sim)
        brake_step = max(0.0, float(BASE_BRAKE_MS2) * 0.5 * dt_sim)
        if target_speed_ms >= prev_speed_ms:
            next_speed_ms = min(target_speed_ms, prev_speed_ms + accel_step)
        else:
            next_speed_ms = max(target_speed_ms, prev_speed_ms - brake_step)
        avg_speed_ms = max(0.0, (prev_speed_ms + next_speed_ms) * 0.5)
        self.sc_distance_m = float(prev_distance + (avg_speed_ms * dt_sim))
        self.sc_live_speed_kmh = float(next_speed_ms * 3.6)
        track_len_m = self._safety_car_track_length_m()
        self.sc_progress = math.fmod(self.sc_distance_m / track_len_m, 1.0)
        if self.sc_progress < 0.0:
            self.sc_progress += 1.0
        prev_lap = int(prev_distance // track_len_m)
        next_lap = int(self.sc_distance_m // track_len_m)
        return next_lap > prev_lap

    def _safety_car_gap_seconds(self, name: str) -> Optional[float]:
        if not self.sc_active or not name:
            return None
        try:
            car_dist = float(
                self.distance_along_track_m.get(
                    name,
                    self._driver_distance_along_track_m(name),
                )
            )
        except Exception:
            return None
        gap_m = float(self.sc_distance_m) - car_dist
        if gap_m < 0.0:
            return 0.0
        try:
            speed_ms = max(8.0, float(self.live_speed_kmh.get(name, 0.0) or 0.0) / 3.6)
        except Exception:
            speed_ms = 8.0
        return max(0.0, gap_m / speed_ms)

    def _safety_car_gap_m(self, ahead_speed_kmh: float, gap_s: float) -> float:
        try:
            speed_ms = max(12.0, float(ahead_speed_kmh) / 3.6)
        except Exception:
            speed_ms = 12.0
        return max(8.0, speed_ms * max(0.0, float(gap_s)))

    def _safety_car_queue_names(self, order_list) -> list[str]:
        if not self.sc_active or self.sc_phase not in ("collecting", "returning"):
            return []
        source = order_list if isinstance(order_list, list) else self.order
        if not source:
            return []
        leader_name = self.sc_leader or self._current_on_track_leader_name(source)
        if not leader_name:
            return []
        leader_lap = int(self.laps.get(leader_name, 0) or 0)
        eligible = []
        for drv in source:
            name = str(getattr(drv, "name", "") or "")
            if not name or name in self.finished or self.dnf.get(name, False):
                continue
            if self.pit_remaining.get(name, 0.0) > 0.0:
                continue
            if self.freeze_remaining.get(name, 0.0) > 0.0:
                continue
            if int(self.laps.get(name, 0) or 0) < leader_lap:
                continue
            eligible.append(name)
        if not eligible:
            return []
        eligible_set = set(eligible)
        out = []
        seen = set()
        if leader_name in eligible_set:
            out.append(leader_name)
            seen.add(leader_name)
        for name in list(self.sc_train or []):
            if name in eligible_set and name not in seen:
                out.append(name)
                seen.add(name)
        for name in eligible:
            if name not in seen:
                out.append(name)
                seen.add(name)
        return out

    def _sync_lap_start_snapshot(self, driver_name: str) -> None:
        self._lap_start_fuel[driver_name] = float(self.fuel_onboard.get(driver_name, 0.0))
        self._lap_start_wear[driver_name] = float(self.tyre_wear.get(driver_name, 0.0))

    def _start_tyre_stint_history(
        self,
        driver_name: str,
        *,
        start_lap: float,
        compound: Optional[str],
        start_wear: float,
    ) -> None:
        """Open a presentation-only tyre stint record."""

        if self._physics_series_mode == "oval":
            return
        try:
            lap_value = max(0.0, float(start_lap))
        except Exception:
            lap_value = 0.0
        try:
            wear_value = max(0.0, float(start_wear))
        except Exception:
            wear_value = 0.0
        records = self.tyre_stint_history.setdefault(str(driver_name), [])
        records.append(
            {
                "start_lap": lap_value,
                "compound": str(compound or "medium"),
                "start_wear": wear_value,
                "samples": [(lap_value, wear_value)],
                "end_lap": None,
                "end_wear": None,
            }
        )

    def _record_tyre_stint_sample(
        self,
        driver_name: str,
        *,
        lap_position: float,
        wear: Optional[float] = None,
    ) -> None:
        """Record one actual lap-boundary point for the strategy UI."""

        if self._physics_series_mode == "oval":
            return
        records = self.tyre_stint_history.get(str(driver_name))
        if not isinstance(records, list) or not records:
            self._start_tyre_stint_history(
                driver_name,
                start_lap=max(0.0, float(lap_position)),
                compound=self.tyre_comp.get(driver_name),
                start_wear=float(self.tyre_wear.get(driver_name, 0.0) or 0.0),
            )
            records = self.tyre_stint_history.get(str(driver_name))
        if not isinstance(records, list) or not records:
            return
        record = records[-1]
        if not isinstance(record, dict):
            return
        try:
            lap_value = max(float(record.get("start_lap", 0.0) or 0.0), float(lap_position))
        except Exception:
            return
        try:
            wear_value = max(
                0.0,
                float(self.tyre_wear.get(driver_name, 0.0) if wear is None else wear),
            )
        except Exception:
            return
        samples = record.setdefault("samples", [])
        if not isinstance(samples, list):
            samples = []
            record["samples"] = samples
        point = (lap_value, wear_value)
        if samples and abs(float(samples[-1][0]) - lap_value) <= 1e-9:
            samples[-1] = point
        else:
            samples.append(point)

    def _finish_tyre_stint_history(
        self,
        driver_name: str,
        *,
        end_lap: float,
        end_wear: float,
    ) -> None:
        """Close the active presentation stint before fitting another set."""

        if self._physics_series_mode == "oval":
            return
        self._record_tyre_stint_sample(
            driver_name,
            lap_position=end_lap,
            wear=end_wear,
        )
        records = self.tyre_stint_history.get(str(driver_name))
        if not isinstance(records, list) or not records or not isinstance(records[-1], dict):
            return
        records[-1]["end_lap"] = max(0.0, float(end_lap))
        records[-1]["end_wear"] = max(0.0, float(end_wear))

    def _record_formula_pit_service(self, driver_name, details):
        service = float(details.get("stationary_time_s", 0.0) or 0.0)
        if not math.isfinite(service) or service <= 0:
            return
        self.pit_service_last_time[driver_name] = service
        previous = self.pit_service_best_time.get(driver_name)
        self.pit_service_best_time[driver_name] = min(previous, service) if previous is not None else service
        position = float(self.laps.get(driver_name, 0)) + float(self.progress.get(driver_name, 0.0))
        self.pit_service_completed_lap[driver_name] = position
        self.pit_service_history[driver_name].append(dict(details, completed_lap=position))
        self.last_pit_service[driver_name] = dict(details)

    def _complete_pit_stop(self, driver_name: str) -> None:
        details = self._active_pit_stop_details.pop(driver_name, None)
        fuel_only = self.formula_refueling_allowed and isinstance(details, dict) and not details.get("change_tyres", True)
        if getattr(self, "formula_refueling_allowed", False) and isinstance(details, dict):
            self.fuel_onboard[driver_name] = min(self.formula_fuel_capacity_kg, finite_fuel(details.get("fuel_target_kg"), self.fuel_onboard.get(driver_name, 0)))
            self.formula_fuel_exhausted_early.discard(driver_name)
        tyre_wear_before_stop = max(
            0.0,
            float(self.tyre_wear.get(driver_name, 0.0) or 0.0),
        )
        if self._physics_series_mode == "oval":
            normalized = self._normalize_oval_pit_service_plan(
                driver_name,
                details
                if isinstance(details, dict)
                else self.pending_pit_service.get(driver_name),
            )
            if isinstance(details, dict):
                details.update(normalized)
            else:
                details = dict(normalized)
            if self.oval_refueling_allowed:
                fuel_before = max(
                    0.0, float(self.fuel_onboard.get(driver_name, 0.0) or 0.0)
                )
                fuel_target = max(
                    fuel_before,
                    min(
                        self.oval_fuel_capacity_kg,
                        float(normalized.get("fuel_target_kg", fuel_before) or fuel_before),
                    ),
                )
                self.fuel_onboard[driver_name] = fuel_target
                details["fuel_target_kg"] = fuel_target
                details["fuel_add_kg"] = max(0.0, fuel_target - fuel_before)
            else:
                details["fuel_target_kg"] = max(
                    0.0, float(self.fuel_onboard.get(driver_name, 0.0) or 0.0)
                )
                details["fuel_add_kg"] = 0.0
        if isinstance(details, dict):
            total = max(
                0.0,
                float(
                    details.get(
                        "time_s",
                        self.pit_stop_total_time.get(driver_name, 0.0),
                    )
                    or 0.0
                ),
            )
            self.pit_stop_elapsed[driver_name] = total
            self.pit_stop_last_time[driver_name] = total
            self.pit_stop_timer_serial[driver_name] = (
                int(self.pit_stop_timer_serial.get(driver_name, 0) or 0) + 1
            )
            self.pit_stop_completed_at_wall_time[driver_name] = time.monotonic()
            if self._physics_series_mode != "oval":
                self._record_formula_pit_service(driver_name, details)
            record_hook = getattr(self.state, "record_race_pit_stop", None)
            if callable(record_hook):
                try:
                    record_hook(driver_name, details)
                except Exception:
                    pass
        newc = self.pending_compound.get(driver_name)
        replacement_set = None
        replacement_wear = float(self.tyre_wear.get(driver_name, 0)) if fuel_only else 0.0
        formula_tyres_replaced = bool(newc and self._physics_series_mode != "oval")
        if (
            newc
            and self._physics_series_mode != "oval"
            and self.weekend_tyre_manager is not None
            and self.weekend_tyre_manager.enabled()
            and self.weekend is not None
        ):
            replacement_set, _reason = self.weekend_tyre_manager.checkout(
                self.weekend, driver_name, newc, "race"
            )
            if replacement_set is None:
                for fallback in self.formula_available_compounds(driver_name):
                    replacement_set, _reason = self.weekend_tyre_manager.checkout(
                        self.weekend, driver_name, fallback, "race"
                    )
                    if replacement_set is not None:
                        newc = fallback
                        break
            if replacement_set is not None:
                old_start = int(self.active_tyre_set_start_lap.get(driver_name, 0) or 0)
                current_lap = int(self.laps.get(driver_name, 0) or 0)
                self.weekend_tyre_manager.release(
                    self.weekend,
                    driver_name,
                    self.active_tyre_set_id.get(driver_name),
                    self.tyre_wear.get(driver_name, 0.0),
                    max(0, current_lap - old_start),
                )
                self.active_tyre_set_id[driver_name] = replacement_set.get("set_id")
                self.active_tyre_set_start_lap[driver_name] = current_lap
                replacement_wear = float(replacement_set.get("wear", 0.0) or 0.0)
            else:
                # No legal replacement exists: retain the fitted physical set
                # rather than manufacturing a fresh tyre from nothing.
                newc = self.tyre_comp.get(driver_name)
                replacement_wear = float(self.tyre_wear.get(driver_name, 0.0) or 0.0)
                formula_tyres_replaced = False
        if newc:
            self.tyre_comp[driver_name] = newc
            self.used_compounds[driver_name].add(newc)
        if self.oval_four_tyre_enabled:
            tyre_service = self._normalize_oval_tyre_service(
                details.get("tyres", "four") if isinstance(details, dict) else "four"
            )
            replace_keys = {
                "two_left": ("left_front", "left_rear"),
                "two_right": ("right_front", "right_rear"),
                "four": OVAL_TYRE_KEYS,
            }[tyre_service]
            per_corner = self.tyre_wear_by_corner.setdefault(
                driver_name, {key: 0.0 for key in OVAL_TYRE_KEYS}
            )
            for key in replace_keys:
                per_corner[key] = 0.0
            self._sync_oval_aggregate_tyre_wear(driver_name)
            self.last_pit_service[driver_name] = dict(details or {})
        else:
            if formula_tyres_replaced:
                self._finish_tyre_stint_history(
                    driver_name,
                    end_lap=float(self.laps.get(driver_name, 0) or 0),
                    end_wear=tyre_wear_before_stop,
                )
            self.tyre_wear[driver_name] = replacement_wear
        if not fuel_only:
            self.tyre_temp[driver_name] = float(
                self.tyre_model.initial_temperature_c(
                    self.tyre_comp.get(driver_name),
                    pit_out=True,
                )
            )
        self.pending_compound[driver_name] = None
        self.pending_pit_service[driver_name] = None
        self.pit_count[driver_name] = int(self.pit_count.get(driver_name, 0)) + 1
        if formula_tyres_replaced:
            self._start_tyre_stint_history(
                driver_name,
                start_lap=float(self.laps.get(driver_name, 0) or 0),
                compound=self.tyre_comp.get(driver_name),
                start_wear=float(self.tyre_wear.get(driver_name, 0.0) or 0.0),
            )
        self._replace_front_wing_at_pit_stop(driver_name)
        self._realistic_lap_inputs_cache.pop(driver_name, None)
        self._sync_lap_start_snapshot(driver_name)
        if getattr(self, "formula_refueling_allowed", False):
            self.formula_fuel_stints.setdefault(driver_name, []).append({"start_lap": int(self.laps.get(driver_name, 0)),
                "fuel_target_kg": self.fuel_onboard[driver_name], "compound": self.tyre_comp[driver_name],
                "start_wear": float(self.tyre_wear.get(driver_name, 0))})
        if self._physics_series_mode == "oval" and isinstance(details, dict):
            tyre_label = {
                "two_left": "2 left tyres",
                "two_right": "2 right tyres",
                "four": "4 tyres",
            }.get(str(details.get("tyres") or "four"), "4 tyres")
            fuel_added = max(0.0, float(details.get("fuel_add_kg", 0.0) or 0.0))
            self.events.append(
                f"PIT OUT: {driver_name} ({tyre_label}, +{fuel_added:.1f} kg fuel)"
            )
        else:
            self.events.append(f"PIT OUT: {driver_name} ({self.tyre_comp[driver_name]})")
        if self.driver_pace_modes.is_ai_controlled(driver_name):
            self._ai_race_push_window[driver_name] = max(
                int(self._ai_race_push_window.get(driver_name, 0) or 0),
                3,
            )
        if self.driver_ers_modes.is_ai_controlled(driver_name):
            self._ai_race_ers_attack_window[driver_name] = max(
                int(self._ai_race_ers_attack_window.get(driver_name, 0) or 0),
                2,
            )
        self._strategy_update_pending = True

    def _build_strategy_wear_rate_cache(self) -> None:
        self._strategy_wear_rate_by_compound = {}
        tyre_model = getattr(self, "tyre_model", None)
        if tyre_model is None:
            return
        compounds = list(getattr(self, "available_compounds", []) or [])
        if not compounds:
            compounds = list(tyre_model.compound_names() or [])
        if not compounds:
            compounds = [getattr(tyre_model, "default_compound", "medium")]
        lap_count = max(1, int(getattr(self, "total_laps", 0) or 0))
        for comp in compounds:
            try:
                comp_key = tyre_model.normalize_compound_name(comp)
            except Exception:
                comp_key = str(comp).lower().strip()
            rates = []
            for lap_idx in range(lap_count):
                wetness = self.lap_wetness(lap_idx)
                try:
                    rate = float(tyre_model.wear_rate(comp_key, wetness_mm=wetness))
                except Exception:
                    rate = 0.0
                rates.append(max(0.0, rate))
            self._strategy_wear_rate_by_compound[comp_key] = rates

    def strategy_wear_rate(self, compound: str, lap_index: int) -> float:
        tyre_model = getattr(self, "tyre_model", None)
        if tyre_model is None:
            return 0.0
        try:
            comp_key = tyre_model.normalize_compound_name(compound)
        except Exception:
            comp_key = str(compound).lower().strip()
        rates = getattr(self, "_strategy_wear_rate_by_compound", {}).get(comp_key)
        if rates:
            try:
                idx = int(lap_index)
            except Exception:
                idx = 0
            if idx < 0:
                idx = 0
            if idx >= len(rates):
                idx = len(rates) - 1
            if idx >= 0:
                try:
                    return max(0.0, float(rates[idx]))
                except Exception:
                    pass
        wetness = self.lap_wetness(lap_index)
        try:
            return max(0.0, float(tyre_model.wear_rate(comp_key, wetness_mm=wetness)))
        except Exception:
            return 0.0

    def _sector_fraction(self, sector_index: int) -> float:
        lengths = list(self.sector_lengths or [])
        if sector_index >= 0 and sector_index < len(lengths):
            return max(0.0, float(lengths[sector_index]))
        if lengths:
            return 1.0 / max(1, len(lengths))
        return 1.0

    @staticmethod
    def _load_gap_curve_kmh(raw_curve, *, value_keys=(), fallback=()):
        out = []
        if isinstance(raw_curve, list):
            for entry in raw_curve:
                if not isinstance(entry, dict):
                    continue
                try:
                    gap = max(0.0, float(entry.get("gap", 0.0)))
                except Exception:
                    continue
                value = None
                for key in value_keys:
                    if key in entry:
                        try:
                            value = float(entry.get(key, 0.0))
                        except Exception:
                            value = 0.0
                        break
                # Backwards compatibility with legacy dirty-air mult curves.
                if value is None and "mult" in entry:
                    try:
                        mult = float(entry.get("mult", 1.0))
                    except Exception:
                        mult = 1.0
                    value = max(0.0, (mult - 1.0) * 120.0)
                if value is None:
                    continue
                out.append((gap, max(0.0, float(value))))
        if not out:
            out = [(max(0.0, float(g)), max(0.0, float(v))) for g, v in (fallback or ())]
        out.sort(key=lambda item: item[0])
        return out

    @staticmethod
    def _curve_value_at_gap(curve, gap_s: float) -> float:
        if not curve:
            return 0.0
        try:
            gap = max(0.0, float(gap_s))
        except Exception:
            gap = 0.0
        if gap <= curve[0][0]:
            return float(curve[0][1])
        for idx in range(1, len(curve)):
            g0, v0 = curve[idx - 1]
            g1, v1 = curve[idx]
            if gap <= g1:
                if abs(g1 - g0) <= 1e-9:
                    return float(v1)
                t = (gap - g0) / (g1 - g0)
                return float(v0 + (v1 - v0) * t)
        return float(curve[-1][1])

    def _aero_driver_active(self, driver_name: str) -> bool:
        if not driver_name:
            return False
        if driver_name in self.finished or self.dnf.get(driver_name, False):
            return False
        if self.laps.get(driver_name, 0) == 0 and self.start_delay_remaining.get(driver_name, 0.0) > 0.0:
            return False
        if self.pit_remaining.get(driver_name, 0.0) > 0.0:
            return False
        if self.freeze_remaining.get(driver_name, 0.0) > 0.0:
            return False
        return True

    @staticmethod
    def _aero_wet_attenuation(wetness_mm: Optional[float]) -> float:
        try:
            wet = float(wetness_mm) if wetness_mm is not None else 0.0
        except Exception:
            wet = 0.0
        if wet >= 4.0:
            return 0.30
        if wet >= 3.0:
            return 0.50
        return 1.0

    def _slipstream_delta_seconds_from_kmh(self, kmh_gain: float) -> float:
        try:
            gain = max(0.0, float(kmh_gain))
        except Exception:
            gain = 0.0
        kmh_per_point = float(UNIVERSAL_KMH_PER_RATING.get("straight", 0.0))
        if _REALISTIC_KERNELS is not None:
            try:
                return float(
                    _REALISTIC_KERNELS.slipstream_delta_seconds_from_kmh(
                        float(gain),
                        float(kmh_per_point),
                        0.10,
                    )
                )
            except Exception:
                pass
        if kmh_per_point <= 1e-9 or gain <= 1e-9:
            return 0.0
        points = gain / kmh_per_point
        return -0.10 * float(points)

    def _quantize_slip_delta_seconds(self, delta_s: float) -> float:
        try:
            delta = float(delta_s)
        except Exception:
            delta = 0.0
        if abs(delta) <= 1e-12:
            return 0.0
        q = max(0.001, float(getattr(self, "_slip_delta_quantum_s", 0.01) or 0.01))
        return float(round(delta / q) * q)

    def _smoothed_slip_delta_seconds(self, driver_name: str, raw_delta_s: float) -> float:
        name = str(driver_name or "")
        if not name:
            return 0.0
        try:
            now_t = float(self.total_sim_time.get(name, 0.0) or 0.0)
        except Exception:
            now_t = 0.0
        try:
            prev = float(self._aero_slip_delta_cached_s.get(name, 0.0) or 0.0)
        except Exception:
            prev = 0.0
        try:
            next_t = float(self._aero_slip_delta_next_update_s.get(name, 0.0) or 0.0)
        except Exception:
            next_t = 0.0

        if _REALISTIC_KERNELS is not None:
            try:
                out_delta, out_next_t = _REALISTIC_KERNELS.smooth_slip_delta(
                    float(raw_delta_s),
                    float(prev),
                    float(now_t),
                    float(next_t),
                    float(getattr(self, "_slip_delta_quantum_s", 0.01) or 0.01),
                    float(getattr(self, "_slip_delta_deadband_s", 0.005) or 0.0),
                    float(getattr(self, "_slip_delta_update_interval_s", 0.10) or 0.0),
                )
                out_delta = float(out_delta)
                out_next_t = float(out_next_t)
                self._aero_slip_delta_cached_s[name] = out_delta
                self._aero_slip_delta_next_update_s[name] = out_next_t
                return out_delta
            except Exception:
                pass

        if now_t + 1e-9 < next_t:
            return prev

        quant = self._quantize_slip_delta_seconds(raw_delta_s)
        deadband = max(0.0, float(getattr(self, "_slip_delta_deadband_s", 0.005) or 0.0))
        if abs(quant - prev) <= deadband:
            quant = prev

        self._aero_slip_delta_cached_s[name] = float(quant)
        interval = max(0.0, float(getattr(self, "_slip_delta_update_interval_s", 0.10) or 0.0))
        self._aero_slip_delta_next_update_s[name] = float(now_t + interval)
        return float(quant)

    def _smoothed_dirty_brake_nerf_ms2(self, driver_name: str, raw_ms2: float) -> float:
        name = str(driver_name or "")
        if not name:
            return 0.0
        try:
            now_t = float(self.total_sim_time.get(name, 0.0) or 0.0)
        except Exception:
            now_t = 0.0
        try:
            prev = float(self._aero_dirty_brake_cached_ms2.get(name, 0.0) or 0.0)
        except Exception:
            prev = 0.0
        try:
            next_t = float(self._aero_dirty_brake_next_update_s.get(name, 0.0) or 0.0)
        except Exception:
            next_t = 0.0

        if _REALISTIC_KERNELS is not None:
            try:
                out_v, out_next_t = _REALISTIC_KERNELS.smooth_slip_delta(
                    float(raw_ms2),
                    float(prev),
                    float(now_t),
                    float(next_t),
                    float(getattr(self, "_dirty_brake_quantum_ms2", 0.01) or 0.01),
                    float(getattr(self, "_dirty_brake_deadband_ms2", 0.005) or 0.0),
                    float(getattr(self, "_dirty_brake_update_interval_s", 0.12) or 0.0),
                )
                out_v = max(0.0, float(out_v))
                out_next_t = float(out_next_t)
                self._aero_dirty_brake_cached_ms2[name] = out_v
                self._aero_dirty_brake_next_update_s[name] = out_next_t
                return out_v
            except Exception:
                pass

        if now_t + 1e-9 < next_t:
            return prev

        try:
            raw = max(0.0, float(raw_ms2))
        except Exception:
            raw = 0.0
        q = max(0.0005, float(getattr(self, "_dirty_brake_quantum_ms2", 0.01) or 0.01))
        quant = float(round(raw / q) * q) if raw > 1e-12 else 0.0
        deadband = max(0.0, float(getattr(self, "_dirty_brake_deadband_ms2", 0.005) or 0.0))
        if abs(quant - prev) <= deadband:
            quant = prev
        self._aero_dirty_brake_cached_ms2[name] = float(max(0.0, quant))
        interval = max(0.0, float(getattr(self, "_dirty_brake_update_interval_s", 0.12) or 0.0))
        self._aero_dirty_brake_next_update_s[name] = float(now_t + interval)
        return float(max(0.0, quant))

    def _compute_realistic_aero_state(self, local_rate_map=None) -> dict:
        if bool(getattr(self, "oval_formation_active", False)) and (
            any(int(value or 0) > 0 for value in self.laps.values())
            or any(not bool(value) for value in self.grid_start_pending.values())
        ):
            self._abandon_oval_formation_for_active_race()
        for name in list(self._drs_active_now.keys()):
            self._drs_active_now[name] = False
        if not self.use_realistic_physics:
            return {}
        if self.caution_active:
            return {}

        active_order = [
            d.name
            for d in self.order
            if self._aero_driver_active(getattr(d, "name", ""))
        ]
        if not active_order:
            return {}

        racecraft = getattr(self, "oval_racecraft", None)
        superspeedway_aero = getattr(
            racecraft,
            "compute_superspeedway_aero_state",
            None,
        )
        if callable(superspeedway_aero):
            try:
                superspeedway_out = superspeedway_aero(
                    self,
                    active_order,
                    local_rate_map,
                )
            except Exception:
                superspeedway_out = None
            if isinstance(superspeedway_out, dict):
                return superspeedway_out

        out = {}
        q = max(0.001, float(getattr(self, "_aero_quantum_kmh", 0.02) or 0.02))
        gap_margin_s = 0.30
        max_gap_s = max(0.0, float(getattr(self, "_aero_max_gap_s", 0.0) or 0.0))
        if max_gap_s <= 0.0:
            max_gap_s = 2.0
        max_gap_with_margin = max_gap_s + gap_margin_s
        rates = local_rate_map if isinstance(local_rate_map, dict) else {}
        ahead_gap_seconds = {}
        candidates = []
        gaps_arr = array("d")
        team_factor_arr = array("d")
        wet_atten_arr = array("d")
        dirty_trait_factor_arr = array("d")
        delta_prog_arr = array("d")
        local_rate_arr = array("d")
        lateral_dirty_air_factor_arr = array("d")
        lateral_slipstream_factor_arr = array("d")
        stack_layer2_gap_arr = array("d")
        stack_layer3_gap_arr = array("d")
        adjacent_gap_by_index = {}
        adjacent_delta_prog_by_index = {}
        adjacent_rate_by_index = {}
        aero_ahead_by_index = {}
        active_index_by_name = {name: idx for idx, name in enumerate(active_order)}
        for idx, name in enumerate(active_order):
            try:
                local_rate = float(rates.get(name, 0.0) or 0.0)
            except Exception:
                local_rate = 0.0
            if local_rate <= 1e-9:
                continue
            if racecraft is not None:
                try:
                    ahead_name = racecraft.aero_target(
                        self,
                        name,
                        max_gap_s=max_gap_s,
                        local_rate=local_rate,
                    )
                except Exception:
                    ahead_name = None
            else:
                ahead_name = active_order[idx - 1] if idx > 0 else None
            if not ahead_name:
                continue
            if racecraft is not None:
                try:
                    physical_gap_m = float(racecraft.forward_gap_m(self, name, ahead_name))
                    lap_length_m = max(1.0, float(self._distance_lap_length_m()))
                    delta_prog = physical_gap_m / lap_length_m
                except Exception:
                    continue
                if delta_prog <= 1e-9:
                    continue
                gap_s = max(0.0, float(delta_prog) / float(local_rate))
            else:
                lap_diff = int(self.laps.get(ahead_name, 0) or 0) - int(self.laps.get(name, 0) or 0)
                if lap_diff != 0:
                    continue
                ahead_total = float(self.laps.get(ahead_name, 0)) + float(self.progress.get(ahead_name, 0.0))
                self_total = float(self.laps.get(name, 0)) + float(self.progress.get(name, 0.0))
                delta_prog = ahead_total - self_total
                if delta_prog <= 1e-9:
                    continue
                gap_s = max(0.0, float(delta_prog) / float(local_rate))
                if self._aero_use_exact_gap:
                    lap_diff, gap = self.distance_reference_gap(name, ahead_name)
                    if int(lap_diff) != 0 or gap is None:
                        continue
                    try:
                        gap_s = max(0.0, float(gap))
                    except Exception:
                        continue
            adjacent_gap_by_index[idx] = float(gap_s)
            adjacent_delta_prog_by_index[idx] = float(delta_prog)
            adjacent_rate_by_index[idx] = float(local_rate)
            aero_ahead_by_index[idx] = str(ahead_name)
        for idx, name in enumerate(active_order):
            ahead_name = aero_ahead_by_index.get(idx)
            if not ahead_name:
                continue
            wet_atten = self._aero_wet_attenuation(self.wetness_for_driver(name))
            if wet_atten <= 1e-9:
                continue

            # Estimate same-lap gap in seconds from progress delta and last known
            # local progress rate. This avoids expensive reference-gap solves in
            # the hot aero loop while preserving the same gap-domain aero curves.
            if idx not in adjacent_gap_by_index:
                continue
            local_rate = float(adjacent_rate_by_index.get(idx, 0.0) or 0.0)
            delta_prog = float(adjacent_delta_prog_by_index.get(idx, 0.0) or 0.0)
            est_gap_s = float(adjacent_gap_by_index.get(idx, 0.0) or 0.0)
            if est_gap_s > max_gap_with_margin:
                continue
            gap_s = est_gap_s
            try:
                team_factor = float(
                    self.driver_dirty_air_sensitivity.get(
                        name,
                        self.team_meta.get(
                            self.driver_team.get(name), {}
                        ).get("dirty_air_sensitivity", 1.0),
                    )
                )
            except Exception:
                team_factor = 1.0
            team_factor = max(0.1, float(team_factor))
            dirty_trait_factor = max(
                0.0,
                float(self._driver_dirty_air_trait_factor.get(name, 1.0) or 1.0),
            )

            candidates.append(name)
            gaps_arr.append(float(gap_s))
            team_factor_arr.append(float(team_factor))
            wet_atten_arr.append(float(wet_atten))
            dirty_trait_factor_arr.append(float(dirty_trait_factor))
            delta_prog_arr.append(float(delta_prog))
            local_rate_arr.append(float(local_rate))
            slipstream_factor = 1.0
            dirty_air_factor = 1.0
            if racecraft is not None:
                pair_factor_fn = getattr(racecraft, "aero_pair_factors", None)
                if callable(pair_factor_fn):
                    try:
                        slipstream_factor, dirty_air_factor = pair_factor_fn(
                            self, name, ahead_name
                        )
                    except Exception:
                        slipstream_factor = 1.0
                        dirty_air_factor = 1.0
                else:
                    try:
                        slipstream_factor = racecraft.wake_factor(
                            self, name, ahead_name
                        )
                    except Exception:
                        slipstream_factor = 1.0
                    dirty_air_factor_fn = getattr(
                        racecraft, "dirty_air_wake_factor", None
                    )
                    if callable(dirty_air_factor_fn):
                        try:
                            dirty_air_factor = dirty_air_factor_fn(
                                self, name, ahead_name
                            )
                        except Exception:
                            dirty_air_factor = slipstream_factor
                    else:
                        dirty_air_factor = slipstream_factor
            lateral_dirty_air_factor_arr.append(
                max(0.0, min(1.0, float(dirty_air_factor)))
            )
            lateral_slipstream_factor_arr.append(
                max(0.0, min(1.0, float(slipstream_factor)))
            )
            layer2_gap = -1.0
            layer3_gap = -1.0
            if (
                self._dirty_air_stacking_enabled
                and self._dirty_air_enabled
                and gap_s <= float(self._dirty_air_stack_base_max_gap_s)
            ):
                upstream_idx = active_index_by_name.get(ahead_name, idx - 1)
                link2_gap = adjacent_gap_by_index.get(upstream_idx)
                if (
                    link2_gap is not None
                    and float(link2_gap) <= float(self._dirty_air_stack_layer2_max_gap_s)
                    and self._dirty_air_stack_layer2_mult > 0.0
                ):
                    layer2_gap = float(link2_gap)
                    upstream_name = aero_ahead_by_index.get(upstream_idx)
                    upstream2_idx = active_index_by_name.get(upstream_name, upstream_idx - 1)
                    link3_gap = adjacent_gap_by_index.get(upstream2_idx)
                    if (
                        link3_gap is not None
                        and float(link3_gap) <= float(self._dirty_air_stack_layer3_max_gap_s)
                        and self._dirty_air_stack_layer3_mult > 0.0
                    ):
                        layer3_gap = float(link3_gap)
            stack_layer2_gap_arr.append(float(layer2_gap))
            stack_layer3_gap_arr.append(float(layer3_gap))
            ahead_gap_seconds[name] = float(gap_s)

        base_effects = []
        trait_scaled_in_base_effects = False
        if candidates:
            base_effects = None
            if (not self._aero_use_exact_gap) and _REALISTIC_KERNELS is not None:
                try:
                    if self._aero_kernel_supports_trait_factor:
                        base_effects = _REALISTIC_KERNELS.compute_aero_effect_bases_from_progress(
                            delta_prog_arr,
                            local_rate_arr,
                            team_factor_arr,
                            wet_atten_arr,
                            dirty_trait_factor_arr,
                            self._dirty_air_curve_gaps_arr,
                            self._dirty_air_curve_vals_arr,
                            self._dirty_air_brake_curve_gaps_arr,
                            self._dirty_air_brake_curve_vals_arr,
                            self._slipstream_curve_gaps_arr,
                            self._slipstream_curve_vals_arr,
                            float(self.track_dirty_air_mult),
                            float(max_gap_with_margin),
                            bool(self._dirty_air_enabled),
                            bool(self._slipstream_enabled),
                        )
                        trait_scaled_in_base_effects = True
                    else:
                        base_effects = _REALISTIC_KERNELS.compute_aero_effect_bases_from_progress(
                            delta_prog_arr,
                            local_rate_arr,
                            team_factor_arr,
                            wet_atten_arr,
                            self._dirty_air_curve_gaps_arr,
                            self._dirty_air_curve_vals_arr,
                            self._dirty_air_brake_curve_gaps_arr,
                            self._dirty_air_brake_curve_vals_arr,
                            self._slipstream_curve_gaps_arr,
                            self._slipstream_curve_vals_arr,
                            float(self.track_dirty_air_mult),
                            float(max_gap_with_margin),
                            bool(self._dirty_air_enabled),
                            bool(self._slipstream_enabled),
                        )
                except Exception:
                    base_effects = None
            if _REALISTIC_KERNELS is not None:
                if not isinstance(base_effects, list) or len(base_effects) != len(candidates):
                    try:
                        if self._aero_kernel_supports_trait_factor:
                            base_effects = _REALISTIC_KERNELS.compute_aero_effect_bases(
                                gaps_arr,
                                team_factor_arr,
                                wet_atten_arr,
                                dirty_trait_factor_arr,
                                self._dirty_air_curve_gaps_arr,
                                self._dirty_air_curve_vals_arr,
                                self._dirty_air_brake_curve_gaps_arr,
                                self._dirty_air_brake_curve_vals_arr,
                                self._slipstream_curve_gaps_arr,
                                self._slipstream_curve_vals_arr,
                                float(self.track_dirty_air_mult),
                                bool(self._dirty_air_enabled),
                                bool(self._slipstream_enabled),
                            )
                            trait_scaled_in_base_effects = True
                        else:
                            base_effects = _REALISTIC_KERNELS.compute_aero_effect_bases(
                                gaps_arr,
                                team_factor_arr,
                                wet_atten_arr,
                                self._dirty_air_curve_gaps_arr,
                                self._dirty_air_curve_vals_arr,
                                self._dirty_air_brake_curve_gaps_arr,
                                self._dirty_air_brake_curve_vals_arr,
                                self._slipstream_curve_gaps_arr,
                                self._slipstream_curve_vals_arr,
                                float(self.track_dirty_air_mult),
                                bool(self._dirty_air_enabled),
                                bool(self._slipstream_enabled),
                            )
                    except Exception:
                        base_effects = None
            if not isinstance(base_effects, list) or len(base_effects) != len(candidates):
                base_effects = []
                trait_scaled_in_base_effects = True
                for i in range(len(candidates)):
                    gap_s = float(gaps_arr[i])
                    team_factor = float(team_factor_arr[i])
                    wet_atten = float(wet_atten_arr[i])
                    dirty_trait_factor = (
                        float(dirty_trait_factor_arr[i]) if i < len(dirty_trait_factor_arr) else 1.0
                    )
                    dirty_kmh = 0.0
                    slip_kmh = 0.0
                    dirty_brake_ms2 = 0.0
                    if self._dirty_air_enabled:
                        dirty_kmh = self._curve_value_at_gap(self._dirty_air_curve_kmh, gap_s)
                        dirty_kmh *= team_factor
                        dirty_kmh *= max(0.0, float(self.track_dirty_air_mult))
                        dirty_kmh *= wet_atten
                        dirty_kmh *= dirty_trait_factor
                        dirty_brake_ms2 = self._curve_value_at_gap(self._dirty_air_brake_curve_ms2, gap_s)
                        dirty_brake_ms2 *= team_factor
                        dirty_brake_ms2 *= max(0.0, float(self.track_dirty_air_mult))
                        dirty_brake_ms2 *= wet_atten
                        dirty_brake_ms2 *= dirty_trait_factor
                    if self._slipstream_enabled:
                        slip_kmh = self._curve_value_at_gap(self._slipstream_curve_kmh, gap_s)
                        slip_kmh *= wet_atten
                    base_effects.append(
                        (
                            float(max(0.0, dirty_kmh)),
                            float(max(0.0, slip_kmh)),
                            float(max(0.0, dirty_brake_ms2)),
                        )
                    )

        stack_effects = []
        if (
            candidates
            and self._dirty_air_stacking_enabled
            and self._dirty_air_enabled
        ):
            if _REALISTIC_KERNELS is not None and hasattr(_REALISTIC_KERNELS, "compute_dirty_air_stack_effects"):
                try:
                    stack_effects = _REALISTIC_KERNELS.compute_dirty_air_stack_effects(
                        stack_layer2_gap_arr,
                        stack_layer3_gap_arr,
                        team_factor_arr,
                        wet_atten_arr,
                        dirty_trait_factor_arr,
                        self._dirty_air_curve_gaps_arr,
                        self._dirty_air_curve_vals_arr,
                        self._dirty_air_brake_curve_gaps_arr,
                        self._dirty_air_brake_curve_vals_arr,
                        float(self.track_dirty_air_mult),
                        float(self._dirty_air_stack_layer2_mult),
                        float(self._dirty_air_stack_layer3_mult),
                        bool(self._dirty_air_enabled),
                    )
                except Exception:
                    stack_effects = []
            if not isinstance(stack_effects, list) or len(stack_effects) != len(candidates):
                stack_effects = []
                for i in range(len(candidates)):
                    team_factor = float(team_factor_arr[i])
                    wet_atten = float(wet_atten_arr[i])
                    dirty_trait_factor = (
                        float(dirty_trait_factor_arr[i]) if i < len(dirty_trait_factor_arr) else 1.0
                    )
                    dirty_kmh = 0.0
                    dirty_brake_ms2 = 0.0
                    if i < len(stack_layer2_gap_arr) and float(stack_layer2_gap_arr[i]) >= 0.0:
                        gap_s = float(stack_layer2_gap_arr[i])
                        dirty_kmh += self._curve_value_at_gap(self._dirty_air_curve_kmh, gap_s) * float(self._dirty_air_stack_layer2_mult)
                        dirty_brake_ms2 += self._curve_value_at_gap(self._dirty_air_brake_curve_ms2, gap_s) * float(self._dirty_air_stack_layer2_mult)
                    if i < len(stack_layer3_gap_arr) and float(stack_layer3_gap_arr[i]) >= 0.0:
                        gap_s = float(stack_layer3_gap_arr[i])
                        dirty_kmh += self._curve_value_at_gap(self._dirty_air_curve_kmh, gap_s) * float(self._dirty_air_stack_layer3_mult)
                        dirty_brake_ms2 += self._curve_value_at_gap(self._dirty_air_brake_curve_ms2, gap_s) * float(self._dirty_air_stack_layer3_mult)
                    dirty_kmh *= team_factor
                    dirty_kmh *= max(0.0, float(self.track_dirty_air_mult))
                    dirty_kmh *= wet_atten
                    dirty_kmh *= dirty_trait_factor
                    dirty_brake_ms2 *= team_factor
                    dirty_brake_ms2 *= max(0.0, float(self.track_dirty_air_mult))
                    dirty_brake_ms2 *= wet_atten
                    dirty_brake_ms2 *= dirty_trait_factor
                    stack_effects.append((float(max(0.0, dirty_kmh)), float(max(0.0, dirty_brake_ms2))))

        drs_applied = set()
        for idx, name in enumerate(candidates):
            try:
                dirty_kmh = max(0.0, float(base_effects[idx][0]))
            except Exception:
                dirty_kmh = 0.0
            try:
                slip_kmh = max(0.0, float(base_effects[idx][1]))
            except Exception:
                slip_kmh = 0.0
            try:
                dirty_brake_ms2 = max(0.0, float(base_effects[idx][2]))
            except Exception:
                gap_s = float(gaps_arr[idx]) if idx < len(gaps_arr) else 0.0
                team_factor = float(team_factor_arr[idx]) if idx < len(team_factor_arr) else 1.0
                wet_atten = float(wet_atten_arr[idx]) if idx < len(wet_atten_arr) else 1.0
                dirty_brake_ms2 = 0.0
                if self._dirty_air_enabled:
                    dirty_brake_ms2 = self._curve_value_at_gap(self._dirty_air_brake_curve_ms2, gap_s)
                    dirty_brake_ms2 *= team_factor
                    dirty_brake_ms2 *= max(0.0, float(self.track_dirty_air_mult))
                    dirty_brake_ms2 *= wet_atten
            if not trait_scaled_in_base_effects:
                dirty_trait_factor = (
                    float(dirty_trait_factor_arr[idx]) if idx < len(dirty_trait_factor_arr) else 1.0
                )
                dirty_kmh *= dirty_trait_factor
                dirty_brake_ms2 *= dirty_trait_factor
            if isinstance(stack_effects, list) and idx < len(stack_effects):
                try:
                    dirty_kmh += max(0.0, float(stack_effects[idx][0]))
                except Exception:
                    pass
                try:
                    dirty_brake_ms2 += max(0.0, float(stack_effects[idx][1]))
                except Exception:
                    pass
            dirty_air_factor = (
                float(lateral_dirty_air_factor_arr[idx])
                if idx < len(lateral_dirty_air_factor_arr)
                else 1.0
            )
            slipstream_factor = (
                float(lateral_slipstream_factor_arr[idx])
                if idx < len(lateral_slipstream_factor_arr)
                else 1.0
            )
            dirty_kmh *= dirty_air_factor
            dirty_brake_ms2 *= dirty_air_factor
            slip_kmh *= slipstream_factor
            drs_kmh = max(0.0, float(self._drs_boost_kmh_for_driver(name)))
            drs_accel_mult = self._drs_accel_multiplier_from_kmh(drs_kmh)
            drs_active = (drs_kmh > 1e-9) or (drs_accel_mult > 1.0 + 1e-9)
            dirty_q = max(0.0, round(dirty_kmh / q) * q)
            slip_q = max(0.0, round((slip_kmh + drs_kmh) / q) * q)
            dirty_brake_q = self._smoothed_dirty_brake_nerf_ms2(
                name,
                max(0.0, float(dirty_brake_ms2)),
            )
            if drs_active:
                self._drs_active_now[name] = True
            drs_applied.add(name)
            out[name] = {
                "dirty_corner_nerf_kmh": float(dirty_q),
                "dirty_brake_nerf_ms2": float(dirty_brake_q),
                "slip_kmh_gain": float(slip_q),
                "slip_straight_delta_s": float(self._slipstream_delta_seconds_from_kmh(slip_q)),
                "drs_kmh_gain": float(drs_kmh),
                "drs_accel_mult": float(drs_accel_mult),
                "drs_active": bool(drs_active),
            }

        # DRS must persist for eligible drivers across the full zone, including
        # cases where the driver becomes P1 in a local battle inside the zone.
        # That means DRS cannot depend on "has a car ahead in this sub-step".
        for name in active_order:
            if name in drs_applied:
                continue
            drs_kmh = max(0.0, float(self._drs_boost_kmh_for_driver(name)))
            drs_accel_mult = self._drs_accel_multiplier_from_kmh(drs_kmh)
            if drs_kmh <= 1e-9 and drs_accel_mult <= 1.0 + 1e-9:
                continue
            slip_q = max(0.0, round(drs_kmh / q) * q)
            self._drs_active_now[name] = True
            out[name] = {
                "dirty_corner_nerf_kmh": 0.0,
                "dirty_brake_nerf_ms2": 0.0,
                "slip_kmh_gain": float(slip_q),
                "slip_straight_delta_s": float(self._slipstream_delta_seconds_from_kmh(slip_q)),
                "drs_kmh_gain": float(drs_kmh),
                "drs_accel_mult": float(drs_accel_mult),
                "drs_active": True,
            }

        leader_name = active_order[0] if active_order else None
        wet_threshold = float(DRIVER_AID_WETNESS_THRESHOLD)
        for name in active_order:
            flags = self._driver_trait_flags.get(name)
            if not isinstance(flags, dict):
                continue
            corner_delta = 0.0
            brake_mult = 1.0

            if bool(flags.get("clean_air_merchant", False)):
                gap_ahead = float(ahead_gap_seconds.get(name, math.inf))
                if gap_ahead > 2.0:
                    corner_delta += 0.25

            if bool(flags.get("nervous", False)) and name == leader_name:
                corner_delta -= 0.25

            if bool(flags.get("rainmaster", False)):
                wet = self.wetness_for_driver(name)
                try:
                    wet_val = float(wet) if wet is not None else 0.0
                except Exception:
                    wet_val = 0.0
                if wet_val >= wet_threshold:
                    corner_delta += 1.0
                    brake_mult *= 1.01

            if abs(corner_delta) <= 1e-9 and abs(brake_mult - 1.0) <= 1e-9:
                continue

            entry = out.get(name)
            if not isinstance(entry, dict):
                entry = {
                    "dirty_corner_nerf_kmh": 0.0,
                    "dirty_brake_nerf_ms2": 0.0,
                    "slip_kmh_gain": 0.0,
                    "slip_straight_delta_s": 0.0,
                }
            entry["trait_corner_kmh_delta"] = float(corner_delta)
            entry["trait_brake_mult"] = float(brake_mult)
            out[name] = entry

        # Keep aero state compact: only store drivers with active effects.
        out = {
            name: info
            for name, info in out.items()
            if (
                abs(float(info.get("dirty_corner_nerf_kmh", 0.0) or 0.0)) > 1e-9
                or abs(float(info.get("dirty_brake_nerf_ms2", 0.0) or 0.0)) > 1e-9
                or abs(float(info.get("slip_straight_delta_s", 0.0) or 0.0)) > 1e-9
                or abs(float(info.get("drs_kmh_gain", 0.0) or 0.0)) > 1e-9
                or abs(float(info.get("drs_accel_mult", 1.0) or 1.0) - 1.0) > 1e-9
                or abs(float(info.get("trait_corner_kmh_delta", 0.0) or 0.0)) > 1e-9
                or abs(float(info.get("trait_brake_mult", 1.0) or 1.0) - 1.0) > 1e-9
            )
        }
        return out

    def _ambient_temperature_for_driver(self, driver_name: str) -> float:
        ambient = self.temperature_for_driver(driver_name)
        if ambient is None:
            ambient = self.average_temperature()
        if ambient is None:
            return 25.0
        try:
            return float(ambient)
        except Exception:
            return 25.0

    def tyre_temperature_target(self, driver_name: str, wetness_mm: Optional[float]) -> float:
        ambient_c = self._ambient_temperature_for_driver(driver_name)
        try:
            wet_mm = float(wetness_mm) if wetness_mm is not None else 0.0
        except Exception:
            wet_mm = 0.0
        offset_c = (
            TYRE_TEMP_TARGET_WET_OFFSET_C
            if wet_mm > TYRE_TEMP_WET_TARGET_THRESHOLD_MM
            else TYRE_TEMP_TARGET_DRY_OFFSET_C
        )
        pressure_delta = float(self.tyre_pressure_effects(driver_name).get("target_temp_delta_c", 0.0))
        return float(ambient_c + offset_c + pressure_delta)

    def driver_tyre_temp_window_shift_c(self, driver_or_name) -> float:
        name = driver_or_name
        if not isinstance(name, str):
            name = getattr(driver_or_name, "name", None)
        team = self.driver_team.get(name) if isinstance(name, str) else None
        if not team:
            return float(TyreModel.temperature_window_shift_for_skill(1.0))
        return float(
            TyreModel.temperature_window_shift_for_skill(
                self.team_tire_temp_skill.get(team, 1.0)
            )
        )

    def driver_tyre_temperature_window_c(
        self,
        driver_or_name,
        compound: Optional[str] = None,
    ) -> tuple[float, float]:
        name = driver_or_name
        if not isinstance(name, str):
            name = getattr(driver_or_name, "name", None)
        comp = compound
        if comp is None and isinstance(name, str):
            comp = self.tyre_comp.get(name, self.tyre_model.default_compound)
        return self.tyre_model.temperature_window_c(
            comp,
            window_shift_c=self.driver_tyre_temp_window_shift_c(name),
        )

    def _update_driver_tyre_temperature(
        self,
        driver_name: str,
        compound: str,
        sector_fraction: float,
        wetness_mm: Optional[float],
    ) -> float:
        try:
            current_temp = float(
                self.tyre_temp.get(
                    driver_name,
                    self.tyre_model.initial_temperature_c(compound, pit_out=False),
                )
            )
        except Exception:
            current_temp = float(self.tyre_model.initial_temperature_c(compound, pit_out=False))

        try:
            warmup_rate, cool_rate = self.tyre_model.thermal_rates(compound)
        except Exception:
            warmup_rate, cool_rate = 1.0, 1.0
        try:
            warmup_rate += float(self.tyre_pressure_effects(driver_name).get("warmup_rate_delta", 0.0))
        except Exception:
            pass
        warmup_rate = max(0.1, float(warmup_rate))

        target_temp = self.tyre_temperature_target(driver_name, wetness_mm)
        lap_step = TYRE_TEMP_STEP_C_PER_LAP * max(0.0, float(sector_fraction))
        if current_temp < target_temp:
            current_temp = min(target_temp, current_temp + (lap_step * float(warmup_rate)))
        elif current_temp > target_temp:
            current_temp = max(target_temp, current_temp - (lap_step * float(cool_rate)))

        current_temp = max(TYRE_TEMP_MIN_C, min(TYRE_TEMP_MAX_C, current_temp))
        self.tyre_temp[driver_name] = float(current_temp)
        return float(current_temp)

    def driver_tyre_management_factor(self, driver_or_name) -> float:
        name = None
        team = None
        if hasattr(driver_or_name, "name"):
            name = str(getattr(driver_or_name, "name", "") or "")
            team = getattr(driver_or_name, "team", None)
        elif driver_or_name is not None:
            name = str(driver_or_name)
        if name and team is None:
            team = self.driver_team.get(name)

        value = None
        if name:
            value = self.driver_tyre_management.get(name)
        if value is None:
            value = self.team_meta.get(team, {}).get("tyre_management", 1.0)
        try:
            value_f = float(value)
        except Exception:
            value_f = 1.0
        if value_f > 5.0:
            value_f = float(rating_to_tyre_wear(value_f))
        return max(TYRE_MANAGEMENT_MIN, min(TYRE_MANAGEMENT_MAX, value_f))

    @staticmethod
    def _driver_style_axis_offset(value, neutral, low, high, low_effect: float, high_effect: float) -> float:
        try:
            current = float(value)
            neutral_value = float(neutral)
            low_value = float(low)
            high_value = float(high)
        except Exception:
            return 0.0
        if current < neutral_value:
            span = max(1e-9, neutral_value - low_value)
            return max(0.0, min(1.0, (neutral_value - current) / span)) * float(low_effect)
        if current > neutral_value:
            span = max(1e-9, high_value - neutral_value)
            return max(0.0, min(1.0, (current - neutral_value) / span)) * float(high_effect)
        return 0.0

    def _driver_style_base_feel_for_driver(self, driver_name: str, driver=None) -> dict:
        try:
            static_inputs = (getattr(self, "_realistic_static_input_cache", {}) or {}).get(driver_name, {})
        except Exception:
            static_inputs = {}
        if isinstance(static_inputs, dict):
            feel = static_inputs.get("style_base_feel")
            if isinstance(feel, dict):
                return feel
        if driver is None:
            driver = self.driver_by_name.get(driver_name)
        team_name = getattr(driver, "team", None) if driver is not None else self.driver_team.get(driver_name)
        concept_key = fitted_chassis_concept_key(getattr(self, "state", None), team_name, driver_name)
        return car_concept_feel(getattr(self, "state", None), concept_key)

    def _driver_style_setup_feel_from_base(
        self,
        base_feel: dict,
        *,
        front_wing=None,
        rear_wing=None,
        front_pressure=None,
        rear_pressure=None,
        suspension_level=None,
    ) -> dict:
        feel = dict(base_feel or {})
        base_balance = clamp_style_value(feel.get("balance", 50.0))
        base_traction = clamp_style_value(feel.get("traction", 50.0))
        balance_delta = 0.0
        traction_delta = 0.0

        wing = self.wing_setup_cfg or {}
        wing_min = wing.get("min_level", 1)
        wing_max = wing.get("max_level", 20)
        wing_neutral = wing.get("neutral_level", 10)
        if front_wing is not None:
            balance_delta += self._driver_style_axis_offset(front_wing, wing_neutral, wing_min, wing_max, -30.0, 30.0)
        if rear_wing is not None:
            balance_delta += self._driver_style_axis_offset(rear_wing, wing_neutral, wing_min, wing_max, 20.0, -20.0)
            traction_delta += self._driver_style_axis_offset(rear_wing, wing_neutral, wing_min, wing_max, -20.0, 20.0)

        suspension = self.suspension_setup_cfg or {}
        susp_min = suspension.get("min_level", 1)
        susp_max = suspension.get("max_level", 11)
        susp_neutral = suspension.get("neutral_level", 6)
        if suspension_level is not None:
            balance_delta += self._driver_style_axis_offset(suspension_level, susp_neutral, susp_min, susp_max, -20.0, 20.0)
            traction_delta += self._driver_style_axis_offset(suspension_level, susp_neutral, susp_min, susp_max, 20.0, -20.0)

        pressures = self.tyre_pressure_cfg or {}
        front_cfg = pressures.get("front", {}) if isinstance(pressures, dict) else {}
        rear_cfg = pressures.get("rear", {}) if isinstance(pressures, dict) else {}
        if front_pressure is not None and isinstance(front_cfg, dict):
            traction_delta += self._driver_style_axis_offset(
                front_pressure,
                front_cfg.get("neutral_psi", 24.0),
                front_cfg.get("min_psi", 20.0),
                front_cfg.get("max_psi", 28.0),
                -10.0,
                10.0,
            )
        if rear_pressure is not None and isinstance(rear_cfg, dict):
            traction_delta += self._driver_style_axis_offset(
                rear_pressure,
                rear_cfg.get("neutral_psi", 22.5),
                rear_cfg.get("min_psi", 20.0),
                rear_cfg.get("max_psi", 25.0),
                10.0,
                -10.0,
            )

        feel["base_balance"] = base_balance
        feel["base_traction"] = base_traction
        feel["balance_delta"] = balance_delta
        feel["traction_delta"] = traction_delta
        feel["balance"] = clamp_style_value(base_balance + balance_delta)
        feel["traction"] = clamp_style_value(base_traction + traction_delta)
        return feel

    def _driver_style_mismatch_effects(
        self,
        driver_name: str,
        driver=None,
        *,
        front_wing=None,
        rear_wing=None,
        front_pressure=None,
        rear_pressure=None,
        suspension_level=None,
    ) -> dict:
        if str(getattr(self.state, "game_mode", "formula")) == "oval":
            return {}
        if driver is None:
            driver = self.driver_by_name.get(driver_name)
        if driver is None:
            return {}
        if front_wing is None or rear_wing is None:
            try:
                cur_front, cur_rear = self.get_wing_setup(driver_name)
                if front_wing is None:
                    front_wing = cur_front
                if rear_wing is None:
                    rear_wing = cur_rear
            except Exception:
                pass
        if front_pressure is None or rear_pressure is None:
            try:
                cur_front_pressure, cur_rear_pressure = self.get_tyre_pressures(driver_name)
                if front_pressure is None:
                    front_pressure = cur_front_pressure
                if rear_pressure is None:
                    rear_pressure = cur_rear_pressure
            except Exception:
                pass
        if suspension_level is None:
            try:
                suspension_level = self.get_suspension_setup(driver_name)
            except Exception:
                pass
        try:
            base_feel = self._driver_style_base_feel_for_driver(driver_name, driver)
            concept_key = str((base_feel or {}).get("concept") or "")
            balance_pref = clamp_style_value(getattr(driver, "balance_preference", 50.0))
            traction_pref = clamp_style_value(getattr(driver, "traction_preference", 50.0))
            cache_key = (
                driver_name,
                concept_key,
                round(balance_pref, 4),
                round(traction_pref, 4),
                None if front_wing is None else int(front_wing),
                None if rear_wing is None else int(rear_wing),
                None if front_pressure is None else round(float(front_pressure), 4),
                None if rear_pressure is None else round(float(rear_pressure), 4),
                None if suspension_level is None else int(suspension_level),
            )
            cache = getattr(self, "_driver_style_mismatch_cache", None)
            if not isinstance(cache, dict):
                cache = {}
                self._driver_style_mismatch_cache = cache
            cached = cache.get(cache_key)
            if isinstance(cached, dict):
                return cached

            feel = self._driver_style_setup_feel_from_base(
                base_feel,
                front_wing=front_wing,
                rear_wing=rear_wing,
                suspension_level=suspension_level,
                front_pressure=front_pressure,
                rear_pressure=rear_pressure,
            )
            balance_mismatch = abs(clamp_style_value(feel.get("balance", 50.0)) - balance_pref)
            traction_mismatch = abs(clamp_style_value(feel.get("traction", 50.0)) - traction_pref)
            out = {
                "balance_mismatch": float(balance_mismatch),
                "traction_mismatch": float(traction_mismatch),
                "corner_kmh_loss": float(balance_mismatch) * BALANCE_MISMATCH_CORNER_LOSS_KMH_PER_POINT,
                "spin_risk_mult": 1.0 + (float(traction_mismatch) * TRACTION_MISMATCH_SPIN_RISK_PER_POINT),
                "tyre_wear_mult": 1.0 + (float(traction_mismatch) * TRACTION_MISMATCH_TYRE_WEAR_PER_POINT),
            }
            cache[cache_key] = out
            return out
        except Exception:
            try:
                return driver_style_mismatch_effects(
                    getattr(self, "state", None),
                    getattr(driver, "team", None),
                    driver_name,
                    driver,
                    front_wing=front_wing,
                    rear_wing=rear_wing,
                    suspension_level=suspension_level,
                    front_pressure=front_pressure,
                    rear_pressure=rear_pressure,
                    wing_cfg=self.wing_setup_cfg,
                    suspension_cfg=self.suspension_setup_cfg,
                    pressure_cfg=self.tyre_pressure_cfg,
                )
            except Exception:
                return {}

    def _driver_style_tyre_wear_multiplier(self, driver_name: str, driver=None) -> float:
        try:
            return max(0.0, float(self._driver_style_mismatch_effects(driver_name, driver).get("tyre_wear_mult", 1.0)))
        except Exception:
            return 1.0

    def _driver_style_spin_risk_multiplier(self, driver_name: str, driver=None) -> float:
        try:
            return max(0.0, float(self._driver_style_mismatch_effects(driver_name, driver).get("spin_risk_mult", 1.0)))
        except Exception:
            return 1.0

    @staticmethod
    def _driver_aid_map_value(
        values: dict,
        driver_name: Optional[str],
        team_name: Optional[str],
        default: float,
    ) -> float:
        if isinstance(values, dict):
            if driver_name and driver_name in values:
                try:
                    return float(values.get(driver_name, default))
                except Exception:
                    return float(default)
            if team_name and team_name in values:
                try:
                    return float(values.get(team_name, default))
                except Exception:
                    return float(default)
        return float(default)

    def _apply_sector_usage(self, driver, sector_index: int) -> None:
        fraction = self._sector_fraction(sector_index)
        if fraction <= 0.0:
            return
        name = driver.name

        burn = float(self.fuel_burn_per_lap.get(name, 0.0))
        if burn > 0.0:
            burn_multiplier = (
                self._oval_fuel_burn_multiplier(name)
                if self.oval_refueling_allowed
                else 0.8 if getattr(self, "formula_refueling_allowed", False) and self.caution_active else 1.0
            )
            remaining = float(self.fuel_onboard.get(name, 0.0)) - (
                burn * burn_multiplier * fraction
            )
            if getattr(self, "formula_refueling_allowed", False) and remaining < -1e-8:
                self.formula_fuel_exhausted_early.add(name)
            if remaining < 0.0005:
                remaining = 0.0
            self.fuel_onboard[name] = max(0.0, remaining)

        try:
            comp = self.tyre_comp.get(
                name,
                self.tyre_model.default_compound,
            )
            wear_scale = (
                self.driver_tyre_management_factor(driver)
                * driver.tyre_wear_mult()
            )
            wear_scale *= self._driver_aid_map_value(
                self.driver_aid_tyre_wear,
                getattr(driver, "name", None),
                getattr(driver, "team", None),
                1.0,
            )
            wear_scale *= float(self.driver_pace_modes.tyre_wear_multiplier(name))
            wear_scale *= self._driver_style_tyre_wear_multiplier(name, driver)
            lap_index = max(0, int(self.laps.get(name, 0)))
            lap_wet = self.current_wetness()
            tyre_temp_c = self._update_driver_tyre_temperature(
                name,
                comp,
                fraction,
                lap_wet,
            )
            try:
                normalized_comp = self.tyre_model.normalize_compound_name(comp)
            except Exception:
                normalized_comp = str(comp).lower().strip()
            _, supplier_durability_rating = self._driver_supplier_ratings(name, normalized_comp)
            base_deg = float(
                self.tyre_model.wear_rate(
                    normalized_comp,
                    wetness_mm=lap_wet,
                    supplier_durability_rating=supplier_durability_rating,
                    tyre_temp_c=tyre_temp_c,
                    temp_window_shift_c=self.driver_tyre_temp_window_shift_c(name),
                )
            )
            base_deg += float(self.tyre_pressure_effects(name).get("wear_rate_add", 0.0) or 0.0)
            base_deg = max(0.0, float(base_deg))
            if self.vsc_active:
                sc_wear_mult = max(
                    0.0,
                    min(
                        1.0,
                        self._safety_car_cfg_float("vsc_tyre_wear_mult", 0.70),
                    ),
                )
            else:
                sc_wear_mult = 0.5 if self.sc_lap_active.get(name, False) else 1.0
            wear_increment = (
                base_deg
                * wear_scale
                * self.track_wear_mult
                * sc_wear_mult
                * fraction
            )
            if self.oval_four_tyre_enabled:
                per_corner = self.tyre_wear_by_corner.setdefault(
                    name, {key: 0.0 for key in OVAL_TYRE_KEYS}
                )
                for key in OVAL_TYRE_KEYS:
                    per_corner[key] = min(
                        1.5,
                        max(
                            0.0,
                            float(per_corner.get(key, 0.0) or 0.0)
                            + wear_increment
                            * float(self.oval_tyre_load_factors.get(key, 1.0)),
                        ),
                    )
                self._sync_oval_aggregate_tyre_wear(name)
            else:
                self.tyre_wear[name] += wear_increment
        except Exception:
            pass
        try:
            part_mult = float(self.driver_pace_modes.part_wear_multiplier(name))
            if part_mult > 0.0:
                part_pairs = tuple(
                    (part_key, float(rate) * float(part_mult) * float(fraction))
                    for part_key, rate in self.part_wear_per_lap.items()
                )
                self._apply_runtime_part_wear_increment(name, part_pairs)
                parts_map = self.part_wear_delta.setdefault(
                    name,
                    {key: 0.0 for key in self.part_wear_per_lap.keys()},
                )
                for part_key, rate in part_pairs:
                    if float(rate) <= 0.0:
                        continue
                    parts_map[str(part_key)] = float(parts_map.get(str(part_key), 0.0)) + float(rate)
        except Exception:
            pass
        try:
            sup_specs = (
                self.supplier_part_specs.get(name) or {}
                if isinstance(self.supplier_part_specs, dict)
                else {}
            )
            if sup_specs and not self.dnf.get(name, False):
                part_mult = float(self.driver_pace_modes.part_wear_multiplier(name))
                sup_delta = self.supplier_part_wear_delta.setdefault(name, {})
                for cat, spec in sup_specs.items():
                    if not isinstance(spec, dict):
                        continue
                    wear_per_lap = float(spec.get("wear_per_lap", 0.0) or 0.0)
                    if wear_per_lap > 0.0:
                        delta = wear_per_lap * part_mult * fraction
                        if str(cat) == "gearbox":
                            try:
                                delta *= float(self.driver_engine_modes.gearbox_wear_multiplier(name))
                            except Exception:
                                pass
                        sup_delta[cat] = float(sup_delta.get(cat, 0.0)) + float(delta)
        except Exception:
            pass

    def _precompute_realistic_step_data(self, dt_sim: Optional[float] = None):
        if not self.use_realistic_physics:
            self._realistic_aero_state = {}
            self._realistic_last_local_rate = {}
            return {}
        racecraft = getattr(self, "oval_racecraft", None)
        if racecraft is not None:
            try:
                racecraft.prepare_step(self, float(dt_sim or 0.0))
            except Exception:
                pass
        self._realistic_aero_state = self._compute_realistic_aero_state(
            self._realistic_last_local_rate
        )
        out = {}
        active_inputs = []
        prepared_start_inputs = {}
        prebuilt_profiles = {}
        if bool(
            getattr(
                getattr(self, "realistic_physics", None),
                "batch_profile_build_enabled",
                False,
            )
        ):
            batch_inputs = []
            for d in self.drivers:
                name = d.name
                if name in self.finished:
                    continue
                if self.laps.get(name, 0) == 0 and self.start_delay_remaining.get(name, 0.0) > 0.0:
                    continue
                if self.pit_remaining.get(name, 0.0) > 0.0:
                    continue
                if self.freeze_remaining.get(name, 0.0) > 0.0:
                    continue
                prog = float(self.progress.get(name, 0.0))
                try:
                    base_inputs = self._realistic_base_lap_inputs_for_driver(d)
                    start_ers_bonus = float(self._ers_bonus_points_for_progress(name, prog))
                    lap_inputs = (
                        replace(
                            base_inputs,
                            acceleration_rating_points=float(base_inputs.acceleration_rating_points) + start_ers_bonus,
                        )
                        if abs(start_ers_bonus) > 1e-9
                        else base_inputs
                    )
                except Exception:
                    continue
                prepared_start_inputs[name] = (
                    base_inputs,
                    float(start_ers_bonus),
                    lap_inputs,
                )
                batch_inputs.append(lap_inputs)
            try:
                prebuilt_profiles = self.realistic_physics.prebuild_profile_misses(
                    batch_inputs
                )
            except Exception:
                prebuilt_profiles = {}
        for d in self.drivers:
            name = d.name
            if name in self.finished:
                continue
            if self.laps.get(name, 0) == 0 and self.start_delay_remaining.get(name, 0.0) > 0.0:
                continue
            if self.pit_remaining.get(name, 0.0) > 0.0:
                continue
            if self.freeze_remaining.get(name, 0.0) > 0.0:
                continue
            prog = float(self.progress.get(name, 0.0))
            try:
                prepared = prepared_start_inputs.get(name)
                if isinstance(prepared, tuple) and len(prepared) == 3:
                    base_inputs, start_ers_bonus, lap_inputs = prepared
                else:
                    base_inputs = self._realistic_base_lap_inputs_for_driver(d)
                    start_ers_bonus = float(self._ers_bonus_points_for_progress(name, prog))
                    lap_inputs = (
                        replace(
                            base_inputs,
                            acceleration_rating_points=float(base_inputs.acceleration_rating_points) + start_ers_bonus,
                        )
                        if abs(start_ers_bonus) > 1e-9
                        else base_inputs
                    )
                local_rate, speed_kmh = self.realistic_physics.rate_and_speed_kmh_at_progress(
                    lap_inputs,
                    prog,
                    cache_key=name,
                    prebuilt_profiles=prebuilt_profiles,
                )
                if dt_sim is not None and float(local_rate or 0.0) > 1e-9:
                    est_end = prog + (float(local_rate) * max(0.0, float(dt_sim)))
                    overlap_frac, sample_progress = self._ers_deploy_overlap_for_segment(
                        name,
                        prog,
                        est_end,
                    )
                    if overlap_frac > 1e-6 and sample_progress is not None:
                        probe_ers_bonus = float(self._ers_bonus_points_for_progress(name, sample_progress))
                        probe_inputs = (
                            replace(
                                base_inputs,
                                acceleration_rating_points=(
                                    float(base_inputs.acceleration_rating_points) + probe_ers_bonus
                                ),
                            )
                            if abs(probe_ers_bonus) > 1e-9
                            else base_inputs
                        )
                        boosted_rate, _boosted_speed_kmh = self.realistic_physics.rate_and_speed_kmh_at_progress(
                            probe_inputs,
                            sample_progress,
                            cache_key=(name, "ers_probe"),
                        )
                        if float(boosted_rate or 0.0) > 1e-9:
                            base_rate = max(0.0, float(local_rate))
                            local_rate = (
                                (base_rate * max(0.0, 1.0 - float(overlap_frac)))
                                + (float(boosted_rate) * float(overlap_frac))
                            )
                if racecraft is not None:
                    rate_mult, speed_mult = racecraft.rate_and_speed_multipliers(
                        self,
                        name,
                        prog,
                        base_speed_kmh=float(speed_kmh),
                    )
                    local_rate = float(local_rate) * float(rate_mult)
                    speed_kmh = float(speed_kmh) * float(speed_mult)
            except Exception:
                base_inputs = None
                lap_inputs = None
                start_ers_bonus = 0.0
                local_rate = 0.0
                speed_kmh = 0.0
            base_lap = max(
                MIN_VALID_LAP_TIME,
                float(self.realistic_base_lap_by_driver.get(name, MIN_VALID_LAP_TIME)),
            )
            out[name] = (
                lap_inputs,
                max(0.0, float(local_rate)),
                max(0.0, float(speed_kmh)),
                float(base_lap),
                float(prog),
                base_inputs,
                float(start_ers_bonus),
            )
            if lap_inputs is not None:
                active_inputs.append((name, lap_inputs))

        if active_inputs:
            try:
                prewarm_count = max(
                    0,
                    min(
                        int(REALISTIC_PREWARM_DRIVERS_PER_STEP),
                        len(active_inputs),
                    ),
                )
            except Exception:
                prewarm_count = 0
            if prewarm_count > 0:
                cursor = int(self._realistic_prewarm_cursor) % len(active_inputs)
                for idx in range(prewarm_count):
                    name, lap_inputs = active_inputs[(cursor + idx) % len(active_inputs)]
                    try:
                        self.realistic_physics.prewarm_adjacent_mass_profiles(
                            lap_inputs,
                            levels=int(REALISTIC_PREWARM_LEVELS),
                        )
                    except Exception:
                        pass
                self._realistic_prewarm_cursor = (cursor + prewarm_count) % len(active_inputs)
        self._realistic_last_local_rate = {
            name: float(val[1] if isinstance(val, tuple) and len(val) >= 2 else 0.0)
            for name, val in out.items()
        }
        # Presentation-only handoff. RaceScreen reads this immutable snapshot
        # and never invokes the physics solver directly.
        self._realistic_last_step_data = dict(out)
        return out

    def get_realistic_physics_telemetry(self, driver_name: str):
        """Expose the profile already generated for a driver in the last step."""
        if not bool(getattr(self, "use_realistic_physics", False)):
            return None
        entry = (getattr(self, "_realistic_last_step_data", {}) or {}).get(driver_name)
        if not isinstance(entry, tuple) or len(entry) < 1:
            return None
        lap_inputs = entry[0]
        if not isinstance(lap_inputs, LapInputs):
            return None
        try:
            snapshot = self.realistic_physics.physics_telemetry_snapshot(lap_inputs, max_segments=18)
        except Exception:
            return None
        if not isinstance(snapshot, dict):
            return None
        out = dict(snapshot)
        out["driver_name"] = str(driver_name)
        try:
            out["progress"] = float(self.progress.get(driver_name, 0.0) or 0.0)
            out["lap"] = int(self.laps.get(driver_name, 0) or 0)
        except Exception:
            out["progress"] = 0.0
            out["lap"] = 0
        return out

    def _estimate_next_boundary_step(self, max_step: float, realistic_step_data=None) -> float:
        progress_arr = array("d")
        local_rate_arr = array("d")
        base_lap_arr = array("d")
        for d in self.drivers:
            name = d.name
            if name in self.finished:
                continue
            if self.laps.get(name, 0) == 0 and self.start_delay_remaining.get(name, 0.0) > 0.0:
                continue
            if self.pit_remaining.get(name, 0.0) > 0.0:
                continue
            if self.freeze_remaining.get(name, 0.0) > 0.0:
                continue

            prog = float(self.progress.get(name, 0.0))

            local_rate = 0.0
            base_lap = max(
                MIN_VALID_LAP_TIME,
                float(self.realistic_base_lap_by_driver.get(name, MIN_VALID_LAP_TIME)),
            )
            cached = (realistic_step_data or {}).get(name)
            if isinstance(cached, tuple) and len(cached) >= 2:
                local_rate = float(cached[1] or 0.0)
                if len(cached) >= 4:
                    try:
                        base_lap = max(MIN_VALID_LAP_TIME, float(cached[3]))
                    except Exception:
                        pass
                if len(cached) >= 5:
                    try:
                        prog = float(cached[4])
                    except Exception:
                        pass
            if local_rate <= 1e-9:
                try:
                    lap_inputs = self._realistic_lap_inputs_for_driver(d)
                    local_rate = float(
                        self.realistic_physics.progress_rate_per_second(
                            lap_inputs,
                            prog,
                            cache_key=name,
                        )
                        or 0.0
                    )
                except Exception:
                    local_rate = 0.0
            progress_arr.append(float(prog))
            local_rate_arr.append(float(max(0.0, local_rate)))
            base_lap_arr.append(float(base_lap))

        if not progress_arr:
            return float(max_step)
        if _REALISTIC_KERNELS is not None:
            try:
                return float(
                    _REALISTIC_KERNELS.estimate_next_boundary_step(
                        progress_arr,
                        local_rate_arr,
                        base_lap_arr,
                        self._sector_splits_arr,
                        float(max_step),
                        0.02,
                    )
                )
            except Exception:
                pass

        estimate = None
        for idx in range(len(progress_arr)):
            prog = float(progress_arr[idx])
            next_split = 1.0
            for split in self.sector_splits:
                s = float(split)
                if s > prog + 1e-9:
                    next_split = s
                    break
            gap = max(0.0, next_split - prog)
            if gap <= 1e-9:
                continue
            local_rate = float(local_rate_arr[idx])
            if local_rate > 1e-9:
                secs = gap / local_rate
            else:
                secs = gap * float(base_lap_arr[idx])
            if secs <= 1e-9:
                continue
            if estimate is None or secs < estimate:
                estimate = secs
        if estimate is None:
            return float(max_step)
        return max(0.02, min(float(max_step), float(estimate)))

    def _update_sim_step(self, dt_sim, realistic_step_data=None):
        # Track previous order for overtake detection
        prev_order = list(self.order)
        pit_names_this_step = {
            str(name)
            for name, remaining in (getattr(self, "pit_remaining", {}) or {}).items()
            if float(remaining or 0.0) > 0.0
        }
        caution_active_at_step_start = bool(self.caution_active)
        vsc_active_at_step_start = bool(self.vsc_active)
        lap_completed_in_step = False
        _gap_cache = {}

        def _distance_gap_cached(drv_name: str, ref_name: str):
            key = (str(drv_name), str(ref_name))
            if key in _gap_cache:
                return _gap_cache[key]
            val = self.distance_reference_gap(drv_name, ref_name)
            _gap_cache[key] = val
            return val

        step_order = list(prev_order)
        sc_caution_ahead_by_name = (
            self._safety_car_immediate_ahead_map(step_order)
            if self.caution_active
            else {}
        )
        lap_len_m = self._safety_car_track_length_m()
        step_distance_after = {
            str(getattr(drv, "name", "") or ""): float(
                self.distance_along_track_m.get(
                    getattr(drv, "name", ""),
                    self._driver_distance_along_track_m(getattr(drv, "name", "")),
                )
            )
            for drv in step_order
        }
        step_distance_before = dict(step_distance_after)
        sc_crossed_line = self._advance_safety_car(dt_sim) if self.sc_active else False
        sc_queue_names = self._safety_car_queue_names(step_order)
        if self.sc_phase in ("collecting", "returning") and sc_queue_names:
            self.sc_leader = sc_queue_names[0]

        base_ds_by_driver = {}
        step_base_inputs_by_driver = {}
        step_lap_inputs_by_driver = {}
        step_ers_bonus_by_driver = {}
        if dt_sim > 1e-9:
            pace_names = []
            pace_rates = array("d")
            pace_base_laps = array("d")
            for d in self.drivers:
                name = d.name
                if name in self.finished:
                    continue
                if self.laps.get(name, 0) == 0 and self.start_delay_remaining.get(name, 0.0) > 0.0:
                    continue
                if self.pit_remaining.get(name, 0.0) > 0.0:
                    continue
                if self.freeze_remaining.get(name, 0.0) > 0.0:
                    continue
                local_rate = 0.0
                cached = (realistic_step_data or {}).get(name)
                if isinstance(cached, tuple) and len(cached) >= 2:
                    local_rate = float(cached[1] or 0.0)
                    if len(cached) >= 1 and isinstance(cached[0], LapInputs):
                        step_lap_inputs_by_driver[name] = cached[0]
                    if len(cached) >= 6 and isinstance(cached[5], LapInputs):
                        step_base_inputs_by_driver[name] = cached[5]
                    if len(cached) >= 7:
                        try:
                            step_ers_bonus_by_driver[name] = float(cached[6] or 0.0)
                        except Exception:
                            step_ers_bonus_by_driver[name] = 0.0
                if local_rate <= 1e-9:
                    try:
                        lap_inputs = self._realistic_lap_inputs_for_driver(d)
                        local_rate = float(
                            self.realistic_physics.progress_rate_per_second(
                                lap_inputs,
                                self.progress.get(name, 0.0),
                                cache_key=name,
                            )
                            or 0.0
                        )
                    except Exception:
                        local_rate = 0.0
                base_lap = max(
                    MIN_VALID_LAP_TIME,
                    float(self.realistic_base_lap_by_driver.get(name, MIN_VALID_LAP_TIME)),
                )
                if isinstance(cached, tuple) and len(cached) >= 4:
                    try:
                        base_lap = max(MIN_VALID_LAP_TIME, float(cached[3]))
                    except Exception:
                        pass
                pace_names.append(name)
                pace_rates.append(float(max(0.0, local_rate)))
                pace_base_laps.append(float(base_lap))

            ds_values = None
            if _REALISTIC_KERNELS is not None and pace_names:
                try:
                    ds_values = _REALISTIC_KERNELS.compute_progress_deltas(
                        pace_rates,
                        pace_base_laps,
                        float(dt_sim),
                    )
                except Exception:
                    ds_values = None
            if isinstance(ds_values, list) and len(ds_values) == len(pace_names):
                for idx, name in enumerate(pace_names):
                    try:
                        base_ds_by_driver[name] = max(0.0, float(ds_values[idx]))
                    except Exception:
                        base_ds_by_driver[name] = 0.0
            else:
                for idx, name in enumerate(pace_names):
                    local_rate = float(pace_rates[idx])
                    if local_rate > 1e-9:
                        ds = local_rate * dt_sim
                    else:
                        ds = dt_sim / float(pace_base_laps[idx])
                    base_ds_by_driver[name] = max(0.0, float(ds))

        lap_crossers = set()

        for d in step_order:
            if d.name in self.finished:
                self.live_speed_kmh[d.name] = 0.0
                continue

            # Start stall delay (only affects lap 0)
            if self.laps.get(d.name, 0) == 0 and self.start_delay_remaining.get(d.name, 0.0) > 0.0:
                if d.name not in self.start_stall_announced:
                    self.events.append(f"{d.name} has stalled on the grid at the launch")
                    self.start_stall_announced.add(d.name)
                self.start_delay_remaining[d.name] = max(
                    0.0, self.start_delay_remaining.get(d.name, 0.0) - dt_sim
                )
                self.total_sim_time[d.name] += dt_sim
                self.live_speed_kmh[d.name] = 0.0
                continue

            # Race start reaction delay, independent of the existing stall system.
            if self.laps.get(d.name, 0) == 0 and self.start_reaction_delay_remaining.get(d.name, 0.0) > 0.0:
                self.start_reaction_delay_remaining[d.name] = max(
                    0.0,
                    self.start_reaction_delay_remaining.get(d.name, 0.0) - dt_sim,
                )
                self.total_sim_time[d.name] += dt_sim
                self.live_speed_kmh[d.name] = 0.0
                continue

            # Pit freeze
            if self.pit_remaining[d.name] > 0.0:
                pit_names_this_step.add(d.name)
                self.pit_remaining[d.name] -= dt_sim
                self.pit_stop_elapsed[d.name] = min(
                    float(self.pit_stop_total_time.get(d.name, 0.0) or 0.0),
                    max(
                        0.0,
                        float(self.pit_stop_total_time.get(d.name, 0.0) or 0.0)
                        - max(0.0, float(self.pit_remaining[d.name])),
                    ),
                )
                if self.pit_remaining[d.name] <= 0.0:
                    self._complete_pit_stop(d.name)
                self.total_sim_time[d.name] += dt_sim
                if self._pit_flag_decay[d.name] > 0:
                    self._pit_flag_decay[d.name] -= 1
                else:
                    self.pitted_last_lap[d.name] = False
                self.live_speed_kmh[d.name] = 0.0
                continue

            # Spin freeze
            if self.freeze_remaining[d.name] > 0.0:
                self.freeze_remaining[d.name] -= dt_sim
                self.total_sim_time[d.name] += dt_sim
                if self.incident_flag_decay[d.name] > 0:
                    self.incident_flag_decay[d.name] -= 1
                self.live_speed_kmh[d.name] = 0.0
                continue

            # decay flags
            if self._pit_flag_decay[d.name] > 0:
                self._pit_flag_decay[d.name] -= 1
            else:
                self.pitted_last_lap[d.name] = False
            if self.incident_flag_decay[d.name] > 0:
                self.incident_flag_decay[d.name] -= 1

            # Pace
            ds = float(base_ds_by_driver.get(d.name, 0.0))
            oval_rolling_start_limited = False
            if ds <= 1e-12:
                # Defensive fallback for any path that missed the precompute.
                local_rate = 0.0
                cached = (realistic_step_data or {}).get(d.name)
                if isinstance(cached, tuple) and len(cached) >= 2:
                    local_rate = float(cached[1] or 0.0)
                if local_rate <= 1e-9:
                    try:
                        lap_inputs = self._realistic_lap_inputs_for_driver(d)
                        local_rate = self.realistic_physics.progress_rate_per_second(
                            lap_inputs,
                            self.progress.get(d.name, 0.0),
                            cache_key=d.name,
                        )
                    except Exception:
                        local_rate = 0.0
                if local_rate > 1e-9:
                    ds = local_rate * dt_sim
                else:
                    base_lap = max(
                        MIN_VALID_LAP_TIME,
                        float(self.realistic_base_lap_by_driver.get(d.name, MIN_VALID_LAP_TIME)),
                    )
                    ds = dt_sim / base_lap
            if self.use_realistic_physics and self.laps.get(d.name, 0) == 0 and ds > 1e-12:
                base_speed_kmh = (
                    (ds * lap_len_m / dt_sim) * 3.6
                    if dt_sim > 1e-9 and lap_len_m > 1e-9
                    else None
                )
                launch_speed_factor = self._launch_speed_factor(
                    d.name,
                    dt_sim,
                    base_speed_kmh=base_speed_kmh,
                )
                oval_rolling_start_limited = bool(
                    self.oval_pace_lap_start_enabled
                    and launch_speed_factor < 1.0 - 1e-12
                )
                ds *= launch_speed_factor
                ds *= self._launch_accel_multiplier(d.name)
            if not self.caution_active and ds > 1e-12:
                aero_info = self._realistic_aero_state.get(d.name, {})
                try:
                    drs_accel_mult = float(aero_info.get("drs_accel_mult", 1.0) or 1.0)
                except Exception:
                    drs_accel_mult = 1.0
                if drs_accel_mult > 1.0 + 1e-9:
                    ds *= min(2.0, max(1.0, drs_accel_mult))
            prev_progress_for_move_over = float(self.progress.get(d.name, 0.0) or 0.0)
            if not self.caution_active and ds > 1e-12:
                ds *= self._move_over_progress_multiplier(
                    d.name,
                    prev_progress_for_move_over,
                )
            if not self.caution_active and ds > 1e-12:
                formula_racecraft = getattr(self, "racecraft_model", None)
                blue_flag_multiplier = getattr(
                    formula_racecraft,
                    "blue_flag_progress_multiplier",
                    None,
                )
                if callable(blue_flag_multiplier):
                    try:
                        ds *= max(
                            0.0,
                            min(
                                1.0,
                                float(
                                    blue_flag_multiplier(
                                        self,
                                        d.name,
                                        dt_sim,
                                    )
                                ),
                            ),
                        )
                    except Exception:
                        pass
            oval_fuel_limited = False
            if (
                self.oval_refueling_allowed
                and not self.caution_active
                and ds > 1e-12
                and float(self.fuel_onboard.get(d.name, 0.0) or 0.0) <= 1e-6
                and dt_sim > 1e-9
                and lap_len_m > 1e-9
            ):
                max_ds = (
                    (self.oval_empty_fuel_speed_cap_kmh / 3.6)
                    * dt_sim
                    / lap_len_m
                )
                if ds > max_ds:
                    ds = max(0.0, max_ds)
                    oval_fuel_limited = True
            current_distance = float(
                step_distance_after.get(
                    d.name,
                    self.distance_along_track_m.get(
                        d.name,
                        self._driver_distance_along_track_m(d.name),
                    ),
                )
            )
            if self.vsc_active and ds > 1e-12 and lap_len_m > 1e-9:
                green_speed_kmh = (
                    (ds * lap_len_m / dt_sim) * 3.6
                    if dt_sim > 1e-9
                    else 0.0
                )
                vsc_speed_kmh = self._vsc_target_speed_kmh(
                    self.progress.get(d.name, 0.0),
                    fallback_green_speed_kmh=green_speed_kmh,
                )
                vsc_max_ds = (vsc_speed_kmh / 3.6) * dt_sim / lap_len_m
                ds = min(ds, max(0.0, float(vsc_max_ds)))
            oval_traffic_limited = bool(oval_fuel_limited)
            if self.caution_active and ds > 1e-12 and lap_len_m > 1e-9:
                proposed_distance = current_distance + (ds * lap_len_m)
                min_follow_gap_m = SAFETY_CAR_MIN_FOLLOW_GAP_M

                if self.sc_active and self.sc_phase in ("collecting", "returning"):
                    leader_name = sc_queue_names[0] if sc_queue_names else self.sc_leader
                    leader_lap = int(self.laps.get(leader_name, 0) or 0) if leader_name else 0
                    is_lapped = leader_name is not None and int(self.laps.get(d.name, 0) or 0) < leader_lap
                    if not is_lapped:
                        queue_idx = sc_queue_names.index(d.name) if d.name in sc_queue_names else -1
                        if queue_idx == 0:
                            target_gap_m = self._safety_car_gap_m(
                                self.sc_live_speed_kmh,
                                self.sc_leader_gap_s,
                            )
                            proposed_distance = min(
                                proposed_distance,
                                max(current_distance, float(self.sc_distance_m) - target_gap_m),
                            )
                        elif queue_idx > 0:
                            ahead_name = sc_queue_names[queue_idx - 1]
                            ahead_distance = float(step_distance_after.get(ahead_name, current_distance))
                            ahead_speed = max(
                                float(self.live_speed_kmh.get(ahead_name, 0.0) or 0.0),
                                self.sc_live_speed_kmh,
                            )
                            target_gap_m = self._safety_car_gap_m(ahead_speed, self.sc_collect_gap_s)
                            proposed_distance = min(
                                proposed_distance,
                                max(current_distance, ahead_distance - target_gap_m),
                            )
                # Neutralize genuine on-track racing from the instant the SC is
                # deployed, including cars which have not yet caught the train
                # and lapped cars excluded from the lead-lap queue. The nearest
                # active classified predecessor is authoritative. Cars in the
                # pits, retired cars and cars immobilized by an incident remain
                # transparent so legitimate caution position changes continue.
                # A caution may be deployed by an earlier car during this
                # step's lap-boundary work. Build the map lazily so cars still
                # to be processed are neutralized immediately.
                if not sc_caution_ahead_by_name:
                    sc_caution_ahead_by_name = self._safety_car_immediate_ahead_map(
                        step_order
                    )
                caution_ahead = sc_caution_ahead_by_name.get(d.name)
                while caution_ahead and (
                    caution_ahead in self.finished
                    or self.dnf.get(caution_ahead, False)
                    or self.pit_remaining.get(caution_ahead, 0.0) > 0.0
                    or self.freeze_remaining.get(caution_ahead, 0.0) > 0.0
                    or (
                        self.laps.get(caution_ahead, 0) == 0
                        and (
                            self.start_delay_remaining.get(caution_ahead, 0.0) > 0.0
                            or self.start_reaction_delay_remaining.get(caution_ahead, 0.0) > 0.0
                        )
                    )
                ):
                    caution_ahead = sc_caution_ahead_by_name.get(caution_ahead)
                if caution_ahead:
                    ahead_distance = float(
                        step_distance_after.get(caution_ahead, current_distance)
                    )
                    proposed_distance = min(
                        proposed_distance,
                        max(current_distance, ahead_distance - min_follow_gap_m),
                    )

                proposed_distance = max(current_distance, proposed_distance)
                ds = max(0.0, float(proposed_distance - current_distance) / lap_len_m)
                step_distance_after[d.name] = proposed_distance
            else:
                racecraft = getattr(self, "oval_racecraft", None)
                if racecraft is not None and ds > 1e-12 and lap_len_m > 1e-9:
                    try:
                        unrestricted_ds = float(ds)
                        ds = racecraft.limit_progress_delta(
                            self,
                            d.name,
                            ds,
                            dt_sim,
                            current_distance,
                            step_distance_after,
                        )
                        oval_traffic_limited = bool(
                            oval_traffic_limited
                            or ds < unrestricted_ds - 1e-12
                        )
                    except Exception:
                        pass
                step_distance_after[d.name] = current_distance + (ds * lap_len_m)
            prev_prog = self.progress[d.name]
            new_prog = prev_prog + ds
            if self.use_realistic_physics and not self.caution_active:
                display_progress = new_prog
                if display_progress >= 1.0:
                    display_progress = 0.999999
                self.live_speed_kmh[d.name] = self._instantaneous_live_speed_kmh(
                    d,
                    display_progress,
                    base_inputs=step_base_inputs_by_driver.get(d.name),
                    lap_inputs=step_lap_inputs_by_driver.get(d.name),
                    lap_inputs_ers_bonus=step_ers_bonus_by_driver.get(d.name),
                )
                if (
                    oval_traffic_limited or oval_rolling_start_limited
                ) and dt_sim > 1e-9 and lap_len_m > 1e-9:
                    traffic_speed_kmh = (ds * lap_len_m / dt_sim) * 3.6
                    self.live_speed_kmh[d.name] = min(
                        float(self.live_speed_kmh[d.name]),
                        max(0.0, float(traffic_speed_kmh)),
                    )
            elif dt_sim > 1e-9 and self.realistic_physics.track_length_m > 0.0:
                meters_per_second = (ds * self.realistic_physics.track_length_m) / dt_sim
                self.live_speed_kmh[d.name] = max(0.0, float(meters_per_second) * 3.6)
            else:
                self.live_speed_kmh[d.name] = 0.0
            try:
                live_speed = max(0.0, float(self.live_speed_kmh.get(d.name, 0.0) or 0.0))
            except Exception:
                live_speed = 0.0
            prev_peak_speed = self.current_lap_peak_speed_kmh.get(d.name)
            if prev_peak_speed is None or live_speed > float(prev_peak_speed):
                self.current_lap_peak_speed_kmh[d.name] = live_speed

            time_used = 0.0
            rate = (ds / dt_sim) if dt_sim > 0 else 0.0

            if rate > 0.0:
                while (
                    new_prog >= 1.0 - 1e-9
                    and time_used < dt_sim - 1e-9
                    and d.name not in self.finished
                ):
                    needed = 1.0 - prev_prog
                    lap_dt = needed / rate if rate > 0.0 else dt_sim - time_used
                    lap_dt = max(0.0, min(lap_dt, dt_sim - time_used))
                    if lap_dt <= 0.0:
                        break
                    time_used += lap_dt
                    self.total_sim_time[d.name] += lap_dt
                    if bool((self.grid_start_pending or {}).get(d.name, False)):
                        self._arm_driver_race_start(d.name, self.total_sim_time[d.name])
                        self._evaluate_ers_failure_for_lap(d.name, 1)
                        prev_prog = 0.0
                        new_prog -= 1.0
                        continue

                    fuel_before = float(
                        self._lap_start_fuel.get(d.name, self.fuel_onboard.get(d.name, 0.0))
                    )
                    wear_before = float(
                        self._lap_start_wear.get(d.name, self.tyre_wear.get(d.name, 0.0))
                    )
                    self._ers_apply_progress_segment(d.name, prev_prog, 1.0)
                    self._record_sector_crossings_between(
                        d.name,
                        prev_prog,
                        1.0,
                        self.total_sim_time[d.name] - lap_dt,
                        time_used - lap_dt,
                        rate,
                    )
                    self._maybe_trigger_lockup_between(d.name, prev_prog, 1.0)
                    self._maybe_trigger_spin_between(d.name, prev_prog, 1.0)
                    self._update_drs_detection_crossings(
                        d.name,
                        prev_prog,
                        1.0,
                        segment_start_time=self.total_sim_time[d.name] - lap_dt,
                        progress_rate=rate,
                    )
                    self._apply_sector_usage(d, len(self.sector_splits))
                    self.driver_pace_modes.commit_pending(d.name)
                    self.driver_engine_modes.commit_pending(d.name)
                    self.driver_ers_modes.commit_pending(d.name)
                    self._ers_reset_lap_state(d.name)
                    self._evaluate_ai_race_pace_mode(d.name)
                    self._evaluate_ai_race_engine_mode(d.name)
                    self._evaluate_ai_race_ers_mode(d.name)
                    self.laps[d.name] += 1
                    if (
                        str(getattr(self.state, "game_mode", "formula") or "formula").lower()
                        == "oval"
                        and self.laps[d.name] > self._laps_led_scored_through
                    ):
                        # The first car to complete each scored lap is the
                        # leader at the line. The pace lap is excluded by the
                        # grid_start_pending branch immediately above.
                        self._laps_led_scored_through = int(self.laps[d.name])
                        self.laps_led[d.name] = int(self.laps_led.get(d.name, 0) or 0) + 1
                    if self.laps[d.name] < self.total_laps and self.checkered_flag_time is None:
                        self._evaluate_ers_failure_for_lap(d.name, self.laps[d.name] + 1)
                    race_finish_crossing = (
                        self.laps[d.name] >= self.total_laps
                        or self.checkered_flag_time is not None
                    )
                    if getattr(self, "formula_refueling_allowed", False) and (d.name in self.formula_fuel_exhausted_early or (not race_finish_crossing and self.fuel_onboard.get(d.name, 0) <= 1e-8)):
                        self._retire_driver(d.name, reason="fuel")
                        new_prog = 0.0
                        break
                    previous_rubber = float(getattr(self, "track_rubber", 0.0) or 0.0)
                    self.track_rubber = deposit_completed_car_lap(
                        previous_rubber,
                        len(self.drivers),
                        self.current_wetness(),
                        self.current_temperature(),
                        RUBBER_RACE_BUILDUP_MULTIPLIER,
                    )
                    if self.track_rubber > previous_rubber + 1e-12:
                        try:
                            self._realistic_lap_inputs_cache.clear()
                        except Exception:
                            pass
                    lap_completed_in_step = True
                    lap_crossers.add(d.name)

                    sector_times = self._finalize_lap_sectors(
                        d.name,
                        self.total_sim_time[d.name],
                        self.laps[d.name],
                    )
                    lap_time = float(sum(sector_times)) if sector_times else (
                        self.total_sim_time[d.name] - self.current_lap_start[d.name]
                    )
                    if lap_time >= MIN_VALID_LAP_TIME:
                        self.last_lap_time[d.name] = lap_time
                        if self.best_lap_time[d.name] is None or lap_time < self.best_lap_time[d.name]:
                            self.best_lap_time[d.name] = lap_time
                        if (
                            self.fastest_lap_time is None
                            or lap_time < self.fastest_lap_time
                        ):
                            self.fastest_lap_time = lap_time
                            self.fastest_lap_driver = d.name
                            self.fastest_lap_lap = int(self.laps[d.name])
                            m = int(lap_time // 60)
                            s = lap_time - 60 * m
                            self._defer_event(f"FASTEST LAP: {d.name} {m}:{s:06.3f}")
                    top_speed_kmh = None
                    try:
                        top_speed_kmh = max(
                            0.0,
                            float(self.current_lap_peak_speed_kmh.get(d.name, 0.0) or 0.0),
                        )
                    except Exception:
                        top_speed_kmh = None
                    if top_speed_kmh is not None:
                        prev_best = self.speed_trap_best_kmh.get(d.name)
                        if prev_best is None or top_speed_kmh > float(prev_best):
                            self.speed_trap_best_kmh[d.name] = top_speed_kmh
                    cornering_speed_kmh = None
                    try:
                        cornering_inputs = step_lap_inputs_by_driver.get(d.name)
                        if cornering_inputs is None:
                            cornering_inputs = step_base_inputs_by_driver.get(d.name)
                        cornering_speed_kmh = self.realistic_physics.corner_target_average_kmh(
                            cornering_inputs
                        )
                    except Exception:
                        cornering_speed_kmh = None
                    if cornering_speed_kmh is not None:
                        try:
                            cornering_speed_kmh = max(0.0, float(cornering_speed_kmh))
                            previous_cornering = self.cornering_speed_best_kmh.get(d.name)
                            if previous_cornering is None or cornering_speed_kmh > float(previous_cornering):
                                self.cornering_speed_best_kmh[d.name] = cornering_speed_kmh
                        except Exception:
                            cornering_speed_kmh = None
                    self.current_lap_start[d.name] = self.total_sim_time[d.name]
                    self.last_line_crossing_time[d.name] = self.total_sim_time[d.name]

                    try:
                        comp = self.tyre_comp.get(
                            d.name,
                            self.tyre_model.default_compound,
                        )
                        self._defer_lap_history_entry(
                            d.name,
                            int(self.laps[d.name]),
                            float(lap_time),
                            sector_times,
                            comp,
                            float(wear_before),
                            float(fuel_before),
                            top_speed_kmh=top_speed_kmh,
                            cornering_speed_kmh=cornering_speed_kmh,
                        )
                    except Exception:
                        pass
                    self.current_lap_peak_speed_kmh[d.name] = 0.0
                    self._record_tyre_stint_sample(
                        d.name,
                        lap_position=float(self.laps.get(d.name, 0) or 0),
                        wear=float(self.tyre_wear.get(d.name, 0.0) or 0.0),
                    )
                    self._sync_lap_start_snapshot(d.name)
                    self.sc_lap_active[d.name] = bool(self.sc_phase in ("collecting", "returning"))

                    # Supplier parts wear + independent failure model (does not
                    # feed into mechanical failure probability).
                    try:
                        sup_specs = (self.supplier_part_specs.get(d.name) or {}) if isinstance(self.supplier_part_specs, dict) else {}
                        if sup_specs and not self.dnf.get(d.name, False):
                            sup_delta = self.supplier_part_wear_delta.setdefault(d.name, {})
                            for cat, spec in sup_specs.items():
                                if not isinstance(spec, dict):
                                    continue
                                start_cond = float(spec.get("condition", 100.0) or 100.0)
                                current_cond = start_cond - float(sup_delta.get(cat, 0.0))
                                if current_cond < 50.0:
                                    fail_prob = float(spec.get("fail_prob_over_50", 0.0) or 0.0)
                                    if (
                                        not race_finish_crossing
                                        and fail_prob > 0.0
                                        and random.random() < fail_prob
                                    ):
                                        # External supplier parts (excluding fuel) should be destroyed on failure.
                                        # We apply this by forcing the wear delta to consume the entire remaining
                                        # condition so post-race application clamps the part to 0%.
                                        try:
                                            if str(cat).lower() != "fuel":
                                                sup_delta[cat] = max(float(sup_delta.get(cat, 0.0)), float(start_cond))
                                        except Exception:
                                            pass
                                        label = str(spec.get("label", cat) or cat)
                                        self._retire_driver(d.name, reason=f"{label} failure")
                                        break
                    except Exception:
                        pass

                    # Engine unit wear + independent failure model.
                    try:
                        eng_spec = (self.engine_unit_specs.get(d.name) or {}) if isinstance(self.engine_unit_specs, dict) else {}
                        if eng_spec and not self.dnf.get(d.name, False):
                            wear_per_lap = float(eng_spec.get("wear_per_lap", 0.0) or 0.0)
                            if wear_per_lap > 0.0:
                                try:
                                    wear_per_lap *= float(self.driver_engine_modes.engine_wear_multiplier(d.name))
                                except Exception:
                                    pass
                                self.engine_wear_delta[d.name] = float(self.engine_wear_delta.get(d.name, 0.0)) + wear_per_lap
                            start_cond = float(eng_spec.get("condition", 100.0) or 100.0)
                            current_cond = start_cond - float(self.engine_wear_delta.get(d.name, 0.0))
                            fail_prob_high = 0.0
                            try:
                                fail_prob_high = float(self.driver_engine_modes.race_fail_prob_per_lap(d.name))
                            except Exception:
                                fail_prob_high = 0.0
                            if (
                                not race_finish_crossing
                                and fail_prob_high > 0.0
                                and random.random() < fail_prob_high
                            ):
                                try:
                                    self.engine_wear_delta[d.name] = max(float(self.engine_wear_delta.get(d.name, 0.0)), float(start_cond))
                                except Exception:
                                    pass
                                self._retire_driver(d.name, reason="Engine failure")
                                continue
                            fail_prob_low_rel = float(eng_spec.get("low_reliability_fail_prob_per_lap", 0.0) or 0.0)
                            if (
                                not race_finish_crossing
                                and fail_prob_low_rel > 0.0
                                and random.random() < fail_prob_low_rel
                            ):
                                try:
                                    self.engine_wear_delta[d.name] = max(float(self.engine_wear_delta.get(d.name, 0.0)), float(start_cond))
                                except Exception:
                                    pass
                                self._retire_driver(d.name, reason="Engine failure")
                                continue
                            if current_cond < 50.0:
                                fail_prob = float(eng_spec.get("fail_prob_over_50", 0.0) or 0.0)
                                if (
                                    not race_finish_crossing
                                    and fail_prob > 0.0
                                    and random.random() < fail_prob
                                ):
                                    # Engine units are destroyed on failure: force wear delta to 0% condition.
                                    try:
                                        self.engine_wear_delta[d.name] = max(float(self.engine_wear_delta.get(d.name, 0.0)), float(start_cond))
                                    except Exception:
                                        pass
                                    self._retire_driver(d.name, reason="Engine failure")
                    except Exception:
                        pass

                    try:
                        pcfg = self.cfg.get("pitstops", {})
                        # Strategy pit decisions are evaluated only at lap boundaries,
                        # only for active (non-finished) laps, and only for AI-managed cars.
                        if (
                            (pcfg.get("enabled", False) or self.formula_refueling_allowed)
                            and self.laps[d.name] < self.total_laps
                            and (not self._player_manual_control(d.name))
                            and (not self.formula_pit_now_pending(d.name))
                        ):
                            wing_pit = self._should_ai_pit_for_front_wing(d.name)
                            should_pit, comp = self.strategy.choose_pit_action(self, d)
                            if should_pit:
                                if self.front_wing_damage_level(d.name) and self.driver_front_wing_change_available(d.name):
                                    self.pending_front_wing_change[d.name] = True
                                if (
                                    self._physics_series_mode == "oval"
                                    and isinstance(comp, dict)
                                ):
                                    plan = self._normalize_oval_pit_service_plan(
                                        d.name, comp
                                    )
                                    self.pending_pit_service[d.name] = dict(plan)
                                    self.pending_compound[d.name] = self.tyre_comp.get(
                                        d.name, self.tyre_model.default_compound
                                    )
                                elif getattr(self, "formula_refueling_allowed", False) and isinstance(comp, dict):
                                    self.pending_pit_service[d.name] = normalize_service(self, d.name, comp)
                                    self.pending_compound[d.name] = self.pending_pit_service[d.name]["compound"]
                                else:
                                    self.pending_compound[d.name] = comp
                                if (self._physics_series_mode != "oval" and self.formula_refueling_allowed
                                        and isinstance(comp, dict) and comp.get("ai_fuel_plan")
                                        and comp.get("change_tyres", True)):
                                    # No simulation or inventory update occurs
                                    # between this decision and entering service.
                                    self._trigger_pit(d.name, evaluated_service=comp)
                                else:
                                    self._trigger_pit(d.name)
                                self.strategy.consume_safety_car_pit_decision(d.name)
                            elif wing_pit:
                                self.pending_front_wing_change[d.name] = True
                                self.pending_compound[d.name] = self._ai_front_wing_pit_compound(d.name)
                                self._trigger_pit(d.name)
                    except Exception as exc:
                        self._record_strategy_error(d.name, exc)

                    try:
                        icfg = self.cfg.get("incidents", {})
                        spin_prob = float(icfg.get("spin_prob_per_lap", 0.0))
                        spin_prob = apply_spin_prob(d, spin_prob)
                        spin_prob *= d.spin_risk_mult()
                        spin_prob *= self._driver_style_spin_risk_multiplier(d.name, d)
                        spin_prob *= self.oval_tyre_imbalance_spin_multiplier(d.name)
                        spin_loss = icfg.get("spin_loss_s", [1.0, 10.0])
                        coll_prob = float(icfg.get("collision_prob_per_lap", 0.0))
                        coll_close = float(icfg.get("collision_close_s", 0.1))
                        mech_base = float(
                            icfg.get(
                                "mechanical_dnf_prob_per_lap",
                                icfg.get("mech_dnf_prob_per_lap", 0.0),
                            )
                        )
                        mech_prob = mech_base
                        mech_prob *= float(self.part_reliability_mult.get(d.name, 1.0))
                        mech_prob *= float(self.chassis_reliability_mult.get(d.name, 1.0))
                        crash_mult = d.crash_risk_mult()
                        pace_crash_mult = crash_mult * float(
                            self.driver_pace_modes.crash_risk_multiplier(d.name)
                        )
                        crash_prob = float(icfg.get("crash_dnf_prob_per_lap", 0.0)) * pace_crash_mult
                        crash_prob *= self._driver_aid_map_value(
                            self.driver_aid_crash_multiplier,
                            getattr(d, "name", None),
                            getattr(d, "team", None),
                            1.0,
                        )
                        coll_prob *= crash_mult

                        lap_wetness = self.current_wetness()
                        wet_val = float(lap_wetness) if lap_wetness is not None else 0.0
                        if wet_val >= 4.0:
                            spin_prob *= 1.75
                            coll_prob *= 1.4
                            crash_prob *= 1.45
                        elif wet_val >= 1.0:
                            spin_prob *= 1.35
                            coll_prob *= 1.2
                            crash_prob *= 1.2

                        dnf_mult = getattr(self, "track_dnf_mult", 1.0)
                        if dnf_mult is not None and abs(float(dnf_mult) - 1.0) > 1e-6:
                            mech_prob *= float(dnf_mult)
                            crash_prob *= float(dnf_mult)
                            coll_prob *= float(dnf_mult)

                        spin_prob = self._caution_driver_error_probability(spin_prob)
                        crash_prob = self._caution_driver_error_probability(crash_prob)
                        if self.caution_active:
                            coll_prob *= 0.1

                        if (
                            not race_finish_crossing
                            and not self.dnf.get(d.name, False)
                            and random.random() < mech_prob
                        ):
                            self._retire_driver(d.name, reason="mechanical failure")
                        elif (
                            not race_finish_crossing
                            and not self.dnf.get(d.name, False)
                            and crash_prob > 0.0
                            and random.random() < crash_prob
                        ):
                            self._retire_driver(d.name, reason="crash")
                        else:
                            if (
                                not race_finish_crossing
                                and not self.dnf.get(d.name, False)
                                and random.random() < spin_prob
                            ):
                                try:
                                    lo, hi = float(spin_loss[0]), float(spin_loss[1])
                                except Exception:
                                    lo, hi = 1.0, 10.0
                                self._trigger_spin(d.name, random.uniform(lo, hi))

                            if (
                                not race_finish_crossing
                                and not self.dnf.get(d.name, False)
                                and len(self.order) > 1
                                and coll_prob > 0.0
                                and getattr(self, "oval_racecraft", None) is None
                            ):
                                nearest = None
                                best_gap = 1e9
                                for other in self.order:
                                    if other.name == d.name or self.dnf.get(other.name, False):
                                        continue
                                    lap_diff, gap = _distance_gap_cached(d.name, other.name)
                                    if int(lap_diff) != 0 or gap is None:
                                        continue
                                    try:
                                        gap = float(gap)
                                    except Exception:
                                        continue
                                    if gap <= 1e-9:
                                        continue
                                    if gap < best_gap:
                                        best_gap = gap
                                        nearest = other
                                if (
                                    nearest
                                    and best_gap <= coll_close
                                    and random.random()
                                    < (
                                        coll_prob
                                        * self._collision_relationship_multiplier(
                                            str(d.name),
                                            str(nearest.name),
                                            icfg,
                                        )
                                    )
                                ):
                                    self._handle_collision_incident(str(d.name), str(nearest.name), icfg)
                    except Exception:
                        pass

                    took_flag = self.laps[d.name] >= self.total_laps
                    if took_flag or self.checkered_flag_time is not None:
                        self._finish_driver(d.name, took_flag=took_flag)
                        new_prog = 0.0
                        time_used = dt_sim
                        break

                    self._roll_realistic_consistency_execution(d)

                    manual_choice = self.manual_pit_requests.get(d.name)
                    urgent_formula_stop = self.formula_pit_now_pending(d.name)
                    if (
                        manual_choice is not None
                        and (
                            urgent_formula_stop
                            or (
                                self._player_manual_control(d.name)
                                and not self.player_auto_pit
                            )
                        )
                    ):
                        # honour the latest manual pit request once the driver reaches pit entry
                        if (
                            self._physics_series_mode == "oval"
                            and isinstance(manual_choice, dict)
                        ):
                            plan = self._normalize_oval_pit_service_plan(
                                d.name, manual_choice
                            )
                            self.pending_pit_service[d.name] = dict(plan)
                            self.pending_compound[d.name] = self.tyre_comp.get(
                                d.name, self.tyre_model.default_compound
                            )
                        elif getattr(self, "formula_refueling_allowed", False) and isinstance(manual_choice, dict):
                            self.pending_pit_service[d.name] = normalize_service(self, d.name, manual_choice)
                            self.pending_compound[d.name] = self.pending_pit_service[d.name]["compound"]
                        else:
                            self.pending_compound[d.name] = manual_choice
                        self._trigger_pit(d.name)
                        leftover = max(0.0, dt_sim - time_used)
                        if leftover > 0.0:
                            self.pit_remaining[d.name] = max(
                                0.0, self.pit_remaining[d.name] - leftover
                            )
                            self.pit_stop_elapsed[d.name] = min(
                                float(
                                    self.pit_stop_total_time.get(d.name, 0.0)
                                    or 0.0
                                ),
                                max(
                                    0.0,
                                    float(
                                        self.pit_stop_total_time.get(d.name, 0.0)
                                        or 0.0
                                    )
                                    - float(self.pit_remaining[d.name]),
                                ),
                            )
                            self.total_sim_time[d.name] += leftover
                            if self.pit_remaining[d.name] <= 0.0:
                                self._complete_pit_stop(d.name)
                        new_prog = 0.0
                        time_used = dt_sim
                        break

                    prev_prog = 0.0
                    new_prog -= 1.0

            if (
                rate > 0.0
                and d.name not in self.finished
                and new_prog > prev_prog + 1e-9
                and not bool((self.grid_start_pending or {}).get(d.name, False))
            ):
                self._ers_apply_progress_segment(d.name, prev_prog, new_prog)
                self._maybe_trigger_lockup_between(d.name, prev_prog, new_prog)
                self._maybe_trigger_spin_between(d.name, prev_prog, new_prog)
                self._update_drs_detection_crossings(
                    d.name,
                    prev_prog,
                    new_prog,
                    segment_start_time=self.total_sim_time[d.name],
                    progress_rate=rate,
                )
                self._record_sector_crossings_between(
                    d.name,
                    prev_prog,
                    new_prog,
                    self.total_sim_time[d.name],
                    time_used,
                    rate,
                )

            if d.name in self.finished:
                self.progress[d.name] = 0.0
            else:
                remaining = max(0.0, dt_sim - time_used)
                if remaining > 0.0:
                    self.total_sim_time[d.name] += remaining
                if new_prog >= 1.0:
                    new_prog = math.fmod(new_prog, 1.0)
                if new_prog < 0.0:
                    new_prog = 0.0
                self.progress[d.name] = new_prog
                if getattr(self, "formula_refueling_allowed", False) and not self.dnf.get(d.name, False) and self.fuel_onboard.get(d.name, 0) <= 1e-8:
                    self._retire_driver(d.name, reason="fuel")

        for d in self.drivers:
            try:
                self._record_distance_sample(d.name)
            except Exception:
                pass

        self._resolve_drs_detection_crossings()

        racecraft = getattr(self, "oval_racecraft", None)
        if racecraft is not None:
            try:
                racecraft.finalize_step(self, dt_sim)
                self._process_racecraft_contacts(
                    racecraft.consume_contact_pairs(),
                    dt_sim,
                )
            except Exception:
                pass

        self._update_weather_events()
        sc_queue_mode = bool(
            self.sc_active
            and self.sc_phase in ("collecting", "returning")
            and self.sc_train
        )
        sc_queue_index = {}
        if sc_queue_mode:
            sc_queue_index = {
                str(name): idx
                for idx, name in enumerate(self.sc_train)
                if name not in self.finished
                and not self.dnf.get(name, False)
                and self.pit_remaining.get(name, 0.0) <= 0.0
                and self.freeze_remaining.get(name, 0.0) <= 0.0
            }

        # Sort order: running → finishers → DNFs (DNFs by laps/progress/time)
        def sort_key(dd):
            n = dd.name
            if n not in self.finished:
                try:
                    dist = float(
                        self.distance_along_track_m.get(
                            n,
                            self._driver_distance_along_track_m(n),
                        )
                    )
                except Exception:
                    dist = 0.0
                if sc_queue_mode and n in sc_queue_index:
                    return (0, 0, int(sc_queue_index[n]), -dist)
                if sc_queue_mode:
                    pit_exit_key = self._safety_car_pit_exit_order_key(n)
                    if pit_exit_key is None:
                        # Preserve stable ordering for non-pitters at an exact
                        # spatial tie. A car that stayed out is ahead of a car
                        # represented at the same line while in the pit lane.
                        pit_tie = (0, 0.0, 0)
                    else:
                        pit_tie = (1, float(pit_exit_key[0]), int(pit_exit_key[1]))
                    return (
                        0,
                        1,
                        -dist,
                        -self.laps.get(n, 0),
                        -self.progress.get(n, 0.0),
                        *pit_tie,
                    )
                if self.caution_active:
                    pit_exit_key = self._safety_car_pit_exit_order_key(n)
                    if pit_exit_key is None:
                        pit_tie = (0, 0.0, 0)
                    else:
                        pit_tie = (1, float(pit_exit_key[0]), int(pit_exit_key[1]))
                    return (
                        0,
                        -dist,
                        -self.laps.get(n, 0),
                        -self.progress.get(n, 0.0),
                        *pit_tie,
                    )
                return (0, -dist, -self.laps.get(n, 0), -self.progress.get(n, 0.0))
            if self.dnf.get(n, False):
                return (2, -self.dnf_laps.get(n, 0), -self.dnf_prog.get(n, 0.0), self.finish_time.get(n, 0.0))
            laps_done = int(self.laps.get(n, 0) or 0)
            return (1, -laps_done, self.finish_time.get(n, 0.0))

        self.order.sort(key=sort_key)
        if self.sc_active:
            if self.sc_phase == "awaiting_pickup":
                # Pickup uses the SC's physical post-line position, after pit
                # entry and classification have both been processed. A leader
                # still in the pits remains authoritative; if that stop loses
                # P1, the inherited on-track leader is collected as it reaches
                # (or has just passed) the waiting SC rather than one lap later.
                pickup_crossers = self._safety_car_pickup_crossers(
                    step_distance_before,
                    step_distance_after,
                )
                self._try_start_safety_car_pickup(
                    self.order,
                    pickup_crossers,
                    step_distance_after,
                )
            if self.sc_phase in ("collecting", "returning"):
                self.sc_train = self._safety_car_queue_names(self.order)
                if self.sc_train:
                    self.sc_leader = self.sc_train[0]
                elif self.sc_phase == "collecting":
                    self.sc_leader = self._current_on_track_leader_name(self.order)
            if self.sc_phase == "collecting":
                if not self._sc_collection_countdown_armed:
                    # The pickup crossing establishes the train; it is not one
                    # of the three existing paced SC laps.
                    self._sc_collection_countdown_armed = True
                elif self.sc_leader in lap_crossers:
                    self.sc_laps_remaining -= 1
                    if self.sc_laps_remaining <= 0:
                        self._start_safety_car_return()
            if self.sc_phase == "returning" and sc_crossed_line:
                self._clear_safety_car()

        if not self.caution_active:
            self._evaluate_ai_team_orders()
            self._finalize_move_over_orders()

        # Detect overtakes only under green-flag conditions. Pit, retirement
        # and incident position changes remain reflected by classification.
        if not self.caution_active:
            racecraft = getattr(self, "oval_racecraft", None)
            if racecraft is not None:
                try:
                    completed_passes = racecraft.consume_completed_passes()
                except Exception:
                    completed_passes = []
                logged_pairs = set()
                for completed_pass in completed_passes:
                    try:
                        name, prior_name = completed_pass[0], completed_pass[1]
                        pass_kind = completed_pass[2] if len(completed_pass) > 2 else "position"
                    except Exception:
                        continue
                    if str(pass_kind) != "position":
                        continue
                    if not self._should_log_overtake(name, prior_name, pit_names_this_step):
                        continue
                    self._defer_event(f"OVERTAKE: {name} on {prior_name}")
                    self._emit_overtake_radio(name, prior_name)
                    logged_pairs.add((str(name), str(prior_name)))
                    if self.driver_pace_modes.is_ai_controlled(prior_name):
                        self._ai_race_push_window[prior_name] = max(
                            int(self._ai_race_push_window.get(prior_name, 0) or 0),
                            3,
                        )
                # Formula's lateral model normally reports a pass once the
                # attacker has cleared the car stored in ``battle_opponent``.
                # A car can, however, clear another member of a traffic group
                # while that state still points at the original opponent (or
                # while returning from an earlier move).  Classification then
                # changes correctly, but no completed-pass record is emitted.
                # Reconcile only those untracked physical order inversions.
                # Oval has its own event semantics, so it does not opt in.
                if bool(
                    getattr(
                        racecraft,
                        "reconcile_untracked_position_passes",
                        False,
                    )
                ):
                    prev_pos = {d.name: i for i, d in enumerate(prev_order)}
                    new_pos = {d.name: i for i, d in enumerate(self.order)}
                    states = getattr(racecraft, "states", {}) or {}
                    for name, old_index in prev_pos.items():
                        new_index = new_pos.get(name, old_index)
                        if new_index >= old_index:
                            continue
                        for prior in prev_order[:old_index]:
                            prior_name = str(getattr(prior, "name", "") or "")
                            pair = (str(name), prior_name)
                            if (
                                not prior_name
                                or pair in logged_pairs
                                or new_pos.get(prior_name, old_index) <= new_index
                            ):
                                continue
                            state = states.get(name)
                            if str(getattr(state, "battle_opponent", "") or "") == prior_name:
                                # This pair is still being tracked.  Logging it
                                # here would duplicate the model's event when
                                # full physical clearance is reached.
                                continue
                            try:
                                pass_kind = racecraft._pass_kind(
                                    self,
                                    str(name),
                                    prior_name,
                                )
                            except Exception:
                                pass_kind = "position"
                            if str(pass_kind) != "position":
                                continue
                            if not self._should_log_overtake(
                                name,
                                prior_name,
                                pit_names_this_step,
                            ):
                                continue
                            self._defer_event(f"OVERTAKE: {name} on {prior_name}")
                            self._emit_overtake_radio(name, prior_name)
                            logged_pairs.add(pair)
                            if self.driver_pace_modes.is_ai_controlled(prior_name):
                                self._ai_race_push_window[prior_name] = max(
                                    int(self._ai_race_push_window.get(prior_name, 0) or 0),
                                    3,
                                )
            else:
                prev_pos = {d.name: i for i, d in enumerate(prev_order)}
                new_pos = {d.name: i for i, d in enumerate(self.order)}
                for name, p_idx in prev_pos.items():
                    if name in self.finished or self.dnf.get(name, False):
                        continue
                    n_idx = new_pos.get(name, p_idx)
                    if n_idx < p_idx:
                        for prior in prev_order[:p_idx]:
                            if new_pos.get(prior.name, p_idx) > n_idx:
                                if not self._should_log_overtake(
                                    name,
                                    prior.name,
                                    pit_names_this_step,
                                ):
                                    # A retirement or pit-cycle position change is
                                    # not an on-track pass. Continue looking for a
                                    # legitimate driver crossed in this step.
                                    if prior.name in self.finished or self.dnf.get(prior.name, False):
                                        continue
                                    break
                                self._defer_event(f"OVERTAKE: {name} on {prior.name}")
                                self._emit_overtake_radio(name, prior.name)
                                if self.driver_pace_modes.is_ai_controlled(prior.name):
                                    self._ai_race_push_window[prior.name] = max(
                                        int(self._ai_race_push_window.get(prior.name, 0) or 0),
                                        3,
                                    )
                                break
        # The expiring physics step remains fully neutralized. Clearing here
        # resumes normal pace and passing on the next substep.
        if vsc_active_at_step_start and self.vsc_active:
            self._advance_virtual_safety_car(dt_sim)
        if lap_completed_in_step:
            self._strategy_update_pending = True
        if bool(self.caution_active) != caution_active_at_step_start:
            self._strategy_update_pending = True

    def _should_log_overtake(self, overtaker: str, passed_driver: str, pit_names=()) -> bool:
        """Return true only for a live, on-track position exchange."""
        overtaker = str(overtaker or "")
        passed_driver = str(passed_driver or "")
        if not overtaker or not passed_driver or overtaker == passed_driver:
            return False
        if (
            overtaker in self.finished
            or passed_driver in self.finished
            or self.dnf.get(overtaker, False)
            or self.dnf.get(passed_driver, False)
        ):
            return False
        pit_names = set(pit_names or ())
        if overtaker in pit_names or passed_driver in pit_names:
            return False
        if (
            self.pit_remaining.get(overtaker, 0.0) > 0.0
            or self.pit_remaining.get(passed_driver, 0.0) > 0.0
        ):
            return False
        return True

    def _process_racecraft_contacts(self, pairs, dt_sim: float) -> None:
        """Resolve physical footprint overlaps reported by a racecraft model."""

        if not pairs or self.caution_active:
            return
        icfg = self.cfg.get("incidents", {}) or {}
        try:
            base = max(0.0, float(icfg.get("collision_prob_per_lap", 0.0) or 0.0))
        except Exception:
            base = 0.0
        if base <= 0.0:
            return
        cooldowns = getattr(self, "_racecraft_contact_cooldowns", None)
        if not isinstance(cooldowns, dict):
            cooldowns = {}
            self._racecraft_contact_cooldowns = cooldowns
        now = max((float(value) for value in self.total_sim_time.values()), default=0.0)
        for first, second in pairs:
            pair = tuple(sorted((str(first), str(second))))
            if now < float(cooldowns.get(pair, 0.0) or 0.0):
                continue
            first_driver = self.driver_by_name.get(pair[0])
            second_driver = self.driver_by_name.get(pair[1])
            if first_driver is None or second_driver is None:
                continue
            if not self._running_on_track(pair[0]) or not self._running_on_track(pair[1]):
                continue
            risk_mult = 0.5 * (
                float(first_driver.crash_risk_mult()) + float(second_driver.crash_risk_mult())
            )
            relationship_mult = self._collision_relationship_multiplier(
                pair[0],
                pair[1],
                icfg,
            )
            chance = min(
                0.35,
                base * max(0.0, float(dt_sim)) * 0.35 * risk_mult * relationship_mult,
            )
            if random.random() < chance:
                cooldowns[pair] = now + 8.0
                self._handle_collision_incident(pair[0], pair[1], icfg)

    def update(self, dt_sim):
        try:
            total_step = float(dt_sim)
        except Exception:
            total_step = 0.0
        if total_step <= 0.0:
            return

        self.tick_count += 1
        self._update_compound_rule()

        previous_rubber = float(getattr(self, "track_rubber", 0.0) or 0.0)
        self.track_rubber = wash_rubber_for_weather(
            previous_rubber,
            total_step,
            self.current_wetness(),
            self.current_rain_intensity(),
        )
        if abs(self.track_rubber - previous_rubber) > 1e-12:
            try:
                self._realistic_lap_inputs_cache.clear()
            except Exception:
                pass

        if bool(getattr(self, "oval_formation_active", False)) and (
            any(int(value or 0) > 0 for value in self.laps.values())
            or any(not bool(value) for value in self.grid_start_pending.values())
        ):
            self._abandon_oval_formation_for_active_race()

        # The Oval formation lap is a pre-race presentation state. Advancing it
        # here avoids rebuilding race profiles and, more importantly, keeps its
        # elapsed time, fuel, tyre wear, incidents and lap crossings out of the
        # scored race simulation.
        if bool(getattr(self, "oval_formation_active", False)):
            self._advance_oval_formation_start(total_step)
            capture_session_telemetry(self, total_step)
            self._cap_event_log()
            synchronize_session_event_cursor(self)
            return

        # Simulate in smaller chunks so pace can react to sector-level fuel/wear changes.
        remaining = total_step
        max_step = float(getattr(self, "_race_update_max_step", 0.35) or 0.35)
        while remaining > 1e-9:
            preview_step = min(remaining, max_step)
            if self.vsc_active:
                preview_step = min(
                    preview_step,
                    max(1e-9, float(self.vsc_remaining_s)),
                )
            self._begin_step_lap_input_cache()
            try:
                realistic_step_data = (
                    self._precompute_realistic_step_data(preview_step)
                    if self.use_realistic_physics
                    else None
                )
                step = min(
                    preview_step,
                    self._estimate_next_boundary_step(preview_step, realistic_step_data),
                )
                self._update_sim_step(step, realistic_step_data)
                capture_session_telemetry(self, step)
            finally:
                self._clear_step_lap_input_cache()
            remaining -= step
        if bool(self.caution_active) != bool(
            getattr(self, "_strategy_last_sc_active", self.caution_active)
        ):
            self._strategy_update_pending = True
        if self._strategy_update_pending:
            try:
                self.strategy.update(self, total_step)
                self._strategy_update_pending = False
            except Exception:
                pass
        self._strategy_last_sc_active = bool(self.caution_active)
        force_flush = len(self.finished) >= len(self.drivers) if self.drivers else True
        self._flush_deferred_nonphysics(force=force_flush)
        capture_session_telemetry(self, 0.0)
        self._cap_event_log()
        synchronize_session_event_cursor(self)

    def is_race_over(self):
        complete = len(self.finished) == len(self.drivers)
        if complete:
            commit_session_surface(self.weekend, "race", self.track_rubber)
        return complete

    def release_runtime_memory(self) -> None:
        """Drop heavy runtime-only caches to free memory after leaving race view."""
        if self.drivers and len(self.finished) >= len(self.drivers):
            self._flush_deferred_nonphysics(force=True)
            finalize_session_telemetry(self, "race")
        try:
            self._realistic_lap_inputs_cache.clear()
        except Exception:
            pass
        try:
            self._realistic_step_base_lap_inputs_cache = None
        except Exception:
            pass
        try:
            self._realistic_aero_state = {}
        except Exception:
            pass
        try:
            self._aero_slip_delta_cached_s.clear()
            self._aero_slip_delta_next_update_s.clear()
        except Exception:
            pass
        try:
            self._aero_dirty_brake_cached_ms2.clear()
            self._aero_dirty_brake_next_update_s.clear()
        except Exception:
            pass
        try:
            self._drs_zone_detection_lap.clear()
            self._drs_zone_eligible.clear()
            self._drs_last_detection_crossing.clear()
            self._drs_pending_detection_crossings.clear()
            self._drs_active_now.clear()
        except Exception:
            pass
        try:
            self._realistic_last_local_rate = {}
        except Exception:
            pass
        try:
            self._gap_snapshot_tick = None
            self._gap_snapshot_mode = None
            self._gap_snapshot = None
        except Exception:
            pass
        try:
            self._deferred_events.clear()
            self._deferred_lap_history.clear()
            self._deferred_part_wear.clear()
        except Exception:
            pass
        try:
            self._clear_safety_car_pit_order()
        except Exception:
            pass
        try:
            if self.use_realistic_physics and hasattr(self.realistic_physics, "clear_runtime_cache"):
                self.realistic_physics.clear_runtime_cache()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

    def lap_temperature(self, lap_index):
        if not self.temperature_profile:
            return None
        try:
            idx = int(lap_index)
        except Exception:
            idx = 0
        if idx < 0:
            idx = 0
        if idx >= len(self.temperature_profile):
            idx = len(self.temperature_profile) - 1
        if idx < 0:
            return None
        return self.temperature_profile[idx]

    def temperature_for_driver(self, name):
        lap = self.laps.get(name, 0)
        return self.lap_temperature(lap)

    def current_temperature(self):
        if not self.temperature_profile:
            return None
        leader_lap = 0
        if self.laps:
            try:
                leader_lap = max(self.laps.values())
            except Exception:
                leader_lap = 0
        return self.lap_temperature(leader_lap)

    def average_temperature(self):
        if not self.temperature_profile:
            return None
        return sum(self.temperature_profile) / len(self.temperature_profile)

    def temperature_band(self, temp):
        if temp is None:
            return "neutral"
        try:
            value = float(temp)
        except Exception:
            return "neutral"
        if value < self._cold_threshold:
            return "cold"
        if value > self._hot_threshold:
            return "hot"
        return "neutral"

    def _temperature_pace_delta_base(self, compound, temp):
        band = self.temperature_band(temp)
        comp_key = compound.lower() if isinstance(compound, str) else compound
        if band not in self._temp_effects:
            return 0.0
        entry = self._temp_effects[band].get(comp_key)
        if not entry:
            return 0.0
        return float(entry.get("pace", 0.0))

    def temperature_pace_delta(self, compound, temp):
        if self.use_realistic_physics:
            return 0.0
        return self._temperature_pace_delta_base(compound, temp)

    def temperature_pace_delta_for_driver(self, driver_name, compound, temp):
        if self.use_realistic_physics:
            return 0.0
        return self._temperature_pace_delta_base(compound, temp)

    def _temperature_wear_multiplier_base(self, compound, temp):
        band = self.temperature_band(temp)
        comp_key = compound.lower() if isinstance(compound, str) else compound
        if band not in self._temp_effects:
            return 1.0
        entry = self._temp_effects[band].get(comp_key)
        if not entry:
            return 1.0
        return float(entry.get("wear", 1.0))

    def temperature_wear_multiplier(self, compound, temp):
        if self.use_realistic_physics:
            return 1.0
        return self._temperature_wear_multiplier_base(compound, temp)

    def temperature_wear_multiplier_for_driver(self, driver_name, compound, temp):
        if self.use_realistic_physics:
            return 1.0
        return self._temperature_wear_multiplier_base(compound, temp)

    def lap_wetness(self, lap_index):
        if not self.wetness_profile:
            return None
        try:
            idx = int(lap_index)
        except Exception:
            idx = 0
        if idx < 0:
            idx = 0
        if idx >= len(self.wetness_profile):
            idx = len(self.wetness_profile) - 1
        if idx < 0:
            return None
        return self.wetness_profile[idx]

    def weather_lap_index(self) -> int:
        """Return the single race-wide index into the lap-based weather profile."""
        if not self.wetness_profile:
            return 0
        leader_lap = 0
        if self.laps:
            try:
                leader_lap = int(max(self.laps.values()))
            except Exception:
                leader_lap = 0
        return max(0, min(len(self.wetness_profile) - 1, leader_lap))

    def wetness_for_driver(self, name):
        # Track water is global. A lapped car must not remain on an earlier
        # point of the weather timeline than the leader.
        return self.current_wetness()

    def current_wetness(self):
        if not self.wetness_profile:
            return None
        wetness = self.lap_wetness(self.weather_lap_index())
        self._observe_compound_rule_wetness(wetness)
        return wetness

    def current_rain_intensity(self):
        if not self.rain_intensity_profile:
            return 0.0
        idx = max(0, min(len(self.rain_intensity_profile) - 1, self.weather_lap_index()))
        return self.rain_intensity_profile[idx]

    def wetness_band(self, wetness):
        if wetness is None or not self._wet_bands:
            return self._wet_bands[0]["name"] if self._wet_bands else "dry"
        try:
            value = float(wetness)
        except Exception:
            return self._wet_bands[0]["name"] if self._wet_bands else "dry"
        band_name = self._wet_bands[0]["name"]
        for band in self._wet_bands:
            lo = band.get("min", self.wetness_range[0])
            hi = band.get("max", self.wetness_range[1])
            if value < lo:
                continue
            if value <= hi + 1e-6:
                band_name = band.get("name", band_name)
                break
        else:
            band_name = self._wet_bands[-1].get("name", band_name)
        return band_name

    def wetness_pace_delta(self, compound, wetness):
        band = self.wetness_band(wetness)
        comp_key = compound.lower() if isinstance(compound, str) else compound
        effects = self._wet_effects.get(band)
        if not effects:
            return 0.0
        entry = effects.get(comp_key)
        if not entry:
            return 0.0
        return float(entry.get("pace", 0.0))

    def wetness_wear_multiplier(self, compound, wetness):
        band = self.wetness_band(wetness)
        comp_key = compound.lower() if isinstance(compound, str) else compound
        effects = self._wet_effects.get(band)
        if not effects:
            return 1.0
        entry = effects.get(comp_key)
        if not entry:
            return 1.0
        return float(entry.get("wear", 1.0))

    def _update_weather_events(self):
        leader_lap = 0
        if self.laps:
            try:
                leader_lap = max(self.laps.values())
            except Exception:
                leader_lap = 0
        if leader_lap == self._last_weather_check_lap:
            return
        self._last_weather_check_lap = leader_lap
        self._update_drs_wet_status_event()
        self._log_wetness_if_needed()

    def _log_wetness_if_needed(self):
        current = self.current_wetness()
        band = self.wetness_band(current)
        if band == self._last_logged_wet_band:
            return
        if current is None:
            return
        if current >= 4.0:
            phrase = "Heavy rain now"
            radio_category = "race_rain_start"
        elif current >= 1.0:
            phrase = "Light rain on track"
            radio_category = "race_rain_start"
        else:
            phrase = "Track drying"
            radio_category = "race_track_drying"
        self.events.append(f"WEATHER: {phrase} ({current:.1f} mm)")
        self._emit_player_team_radio(radio_category)
        self._last_logged_wet_band = band

    def get_coords(self, name):
        racecraft = getattr(self, "oval_racecraft", None)
        if racecraft is not None:
            try:
                return racecraft.coords(name, self.progress[name])
            except Exception:
                pass
        return self.track.pos(self.progress[name])

    def get_safety_car_coords(self):
        return self.track.pos(self.sc_progress)

    def driver_live_speed_kmh(self, name: str) -> Optional[float]:
        try:
            value = float((self.live_speed_kmh or {}).get(name, 0.0))
        except Exception:
            return None
        return max(0.0, value)

    def _finisher_sort_key(self, name: str):
        return (
            -int(self.laps.get(name, 0) or 0),
            self._classification_crossing_time(name),
        )

    def _classification_crossing_time(self, name: str) -> float:
        if not self.dnf.get(name, False):
            return float(self.finish_time.get(name, 0.0) or 0.0)
        return float(
            self.last_line_crossing_time.get(
                name,
                self.current_lap_start.get(name, self.finish_time.get(name, 0.0)),
            )
            or 0.0
        )

    def classified_retirements(self):
        """Return retired cars that completed the official classification distance."""
        completed_laps = [
            int(self.laps.get(name, 0) or 0)
            for name in self.finish_time.keys()
            if not self.dnf.get(name, False)
        ]
        if not completed_laps:
            return set()
        winner_laps = max(completed_laps, default=0)
        if winner_laps <= 0:
            return set()
        minimum_laps = max(1, int(math.floor(float(winner_laps) * 0.90)))
        return {
            name
            for name in self.finish_time.keys()
            if self.dnf.get(name, False)
            and name not in getattr(self, "formula_tyre_disqualifications", set())
            and int(self.laps.get(name, 0) or 0) >= minimum_laps
        }

    def unclassified_retirements(self):
        return {
            name
            for name in self.finish_time.keys()
            if self.dnf.get(name, False)
        } - self.classified_retirements()

    def final_classification(self):
        # Classified retirements rank with finishers by completed laps and the
        # time they last crossed the line. Earlier retirements retain the
        # existing distance-covered ordering after all classified cars.
        if self.is_race_over():
            self._enforce_compound_rule()
            self._commit_formula_tyre_inventory()
        classified_retirements = self.classified_retirements()
        classified_names = [
            name
            for name in self.finish_time.keys()
            if not self.dnf.get(name, False) or name in classified_retirements
        ]
        classified_names.sort(key=self._finisher_sort_key)
        tyre_dsq = getattr(self, "formula_tyre_disqualifications", set())
        penalized = sorted(tyre_dsq, key=lambda n: (-int(self.laps.get(n, 0)), float(self.finish_time[n]), n))
        dnf_names = list(self.unclassified_retirements() - tyre_dsq)
        dnf_names.sort(key=lambda n: (-self.dnf_laps.get(n, 0), -self.dnf_prog.get(n, 0.0), self.finish_time.get(n, 0.0)))
        ordered = [
            (name, self._classification_crossing_time(name))
            for name in classified_names
        ] + [(name, self.finish_time[name]) for name in penalized + dnf_names]
        return [(i + 1, n, t) for i, (n, t) in enumerate(ordered)]

    def _commit_formula_tyre_inventory(self) -> None:
        if self._formula_tyre_inventory_committed:
            return
        if (
            self._physics_series_mode == "oval"
            or self.weekend_tyre_manager is None
            or not self.weekend_tyre_manager.enabled()
            or self.weekend is None
        ):
            self._formula_tyre_inventory_committed = True
            return
        for driver in self.drivers:
            name = driver.name
            start_lap = int(self.active_tyre_set_start_lap.get(name, 0) or 0)
            laps_added = max(0, int(self.laps.get(name, 0) or 0) - start_lap)
            self.weekend_tyre_manager.release(
                self.weekend,
                name,
                self.active_tyre_set_id.get(name),
                self.tyre_wear.get(name, 0.0),
                laps_added,
            )
        self.weekend_tyre_manager.commit_race_attempt(self.weekend)
        self._formula_tyre_inventory_committed = True
