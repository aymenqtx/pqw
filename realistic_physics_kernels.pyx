# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from array import array
from cython.parallel cimport prange
from libc.math cimport sqrt, NAN

# ----------------------------------------------------------------------------
# Reusable scratch buffers for build_lap_time_only_batch_numeric.
#
# The lap-time-only batch kernel used to allocate a dozen zero-filled arrays on
# every call.  At 7-16 batches per strategy decision that dominated the
# per-batch overhead.  The buffers below are grow-only and are fully written
# before they are read inside each call, so reuse is bit-identical.
#
# Only the single-threaded Python caller touches this pool; the prange body
# writes exclusively to its own candidate slot.
# ----------------------------------------------------------------------------
# Batches larger than this are one-offs and are not retained in the pool.
cdef Py_ssize_t _LTT_POOL_MAX_ELEMENTS = 1048576

cdef dict _LTT_SCRATCH_D = {}
cdef dict _LTT_SCRATCH_I = {}


cdef object _ltt_buffer_d(str name, Py_ssize_t size):
    cdef object buf
    if size < 1:
        size = 1
    if size > _LTT_POOL_MAX_ELEMENTS:
        return array("d", [0.0]) * size
    buf = _LTT_SCRATCH_D.get(name)
    if buf is None or len(buf) < size:
        buf = array("d", [0.0]) * size
        _LTT_SCRATCH_D[name] = buf
    return buf


cdef object _ltt_buffer_i(str name, Py_ssize_t size):
    cdef object buf
    if size < 1:
        size = 1
    if size > _LTT_POOL_MAX_ELEMENTS:
        return array("i", [0]) * size
    buf = _LTT_SCRATCH_I.get(name)
    if buf is None or len(buf) < size:
        buf = array("i", [0]) * size
        _LTT_SCRATCH_I[name] = buf
    return buf


def clear_lap_time_scratch_pool():
    """Release the pooled scratch buffers (diagnostics / teardown only)."""
    _LTT_SCRATCH_D.clear()
    _LTT_SCRATCH_I.clear()


cdef inline double _fp_clamp_double(double value, double lo, double hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


cdef inline double _fp_normalize_progress_double(double progress):
    if progress <= 0.0:
        return 0.0
    if progress >= 1.0:
        return 1.0
    return progress

cdef class FastProfile:
    cdef public double lap_time
    cdef public object macro_times
    cdef public object segment_end_progress
    cdef public object segments_obj
    cdef public int count
    cdef object start_progress_arr
    cdef object end_progress_arr
    cdef object progress_len_arr
    cdef object start_time_arr
    cdef object duration_arr
    cdef object distance_arr
    cdef object v_entry_arr
    cdef object v_exit_arr
    cdef object v_peak_arr
    cdef object dist_acc_arr
    cdef object dist_cruise_arr
    cdef object time_acc_arr
    cdef object time_cruise_arr
    cdef object accel_rate_arr
    cdef object accel_scaled_arr
    cdef object brake_rate_arr
    cdef object v_cap_arr
    cdef object drag_coeff_arr
    cdef object sample_dist_arrs
    cdef object sample_time_arrs
    cdef object sample_speed_arrs
    cdef double[:] start_progress
    cdef double[:] end_progress
    cdef double[:] progress_len
    cdef double[:] start_time
    cdef double[:] duration
    cdef double[:] distance
    cdef double[:] v_entry
    cdef double[:] v_exit
    cdef double[:] v_peak
    cdef double[:] dist_acc
    cdef double[:] dist_cruise
    cdef double[:] time_acc
    cdef double[:] time_cruise
    cdef double[:] accel_rate
    cdef double[:] accel_scaled
    cdef double[:] brake_rate
    cdef double[:] v_cap
    cdef double[:] drag_coeff
    cdef public double profile_solver_step_m
    cdef public bint completion_pending
    cdef object macro_bounds

    def __init__(self):
        self.lap_time = 0.001
        self.macro_times = []
        self.segment_end_progress = []
        self.segments_obj = None
        self.count = 0
        self.sample_dist_arrs = []
        self.sample_time_arrs = []
        self.sample_speed_arrs = []
        self.profile_solver_step_m = 1.0
        self.completion_pending = False
        self.macro_bounds = []

    def ensure_segment_samples(self, int idx):
        cdef object result
        if idx < 0 or idx >= self.count:
            return
        if self.sample_dist_arrs[idx] is not None:
            return
        result = segment_motion_profile_tuple(
            float(self.distance[idx]),
            float(self.v_entry[idx]),
            float(self.v_exit[idx]),
            float(self.v_cap[idx]),
            float(self.accel_scaled[idx]),
            float(self.brake_rate[idx]),
            float(self.drag_coeff[idx]),
            float(self.profile_solver_step_m),
        )
        if not isinstance(result, tuple) or len(result) < 9:
            self.sample_dist_arrs[idx] = array("d", [0.0])
            self.sample_time_arrs[idx] = array("d", [0.0])
            self.sample_speed_arrs[idx] = array("d", [float(self.v_exit[idx])])
            return
        self.v_peak[idx] = float(result[1])
        self.duration[idx] = float(result[0])
        self.dist_acc[idx] = float(result[2])
        self.dist_cruise[idx] = float(result[3])
        self.time_acc[idx] = float(result[4])
        self.time_cruise[idx] = float(result[5])
        self.sample_dist_arrs[idx] = result[6]
        self.sample_time_arrs[idx] = result[7]
        self.sample_speed_arrs[idx] = result[8]

    def ensure_complete(self):
        cdef int i, macro_idx, macro_count
        cdef double lap_time, duration, seg_start, seg_end, seg_progress
        cdef double b0, b1, overlap, macro_total, scale, even
        cdef list macro_times
        if not self.completion_pending:
            return
        macro_count = max(1, len(self.macro_bounds) - 1)
        macro_times = [0.0 for _ in range(macro_count)]
        lap_time = 0.0
        for i in range(self.count):
            self.ensure_segment_samples(i)
            duration = max(0.001, float(self.duration[i]))
            self.start_time[i] = lap_time
            seg_start = float(self.start_progress[i])
            seg_end = float(self.end_progress[i])
            seg_progress = max(1e-9, float(self.progress_len[i]))
            for macro_idx in range(macro_count):
                b0 = float(self.macro_bounds[macro_idx])
                b1 = float(self.macro_bounds[macro_idx + 1])
                overlap = min(seg_end, b1) - max(seg_start, b0)
                if overlap <= 1e-9:
                    continue
                macro_times[macro_idx] += duration * (overlap / seg_progress)
            lap_time += duration
        if lap_time <= 0.0:
            lap_time = 0.001
        macro_total = float(sum(macro_times))
        if macro_total > 0.0:
            scale = lap_time / macro_total
            macro_times = [max(0.001, float(v) * scale) for v in macro_times]
        else:
            even = lap_time / macro_count
            macro_times = [max(0.001, even) for _ in range(macro_count)]
        self.macro_times = list(macro_times)
        self.lap_time = float(sum(macro_times))
        self.completion_pending = False

    def get(self, object key, object default=None):
        if key == "lap_time":
            self.ensure_complete()
            return self.lap_time
        if key == "macro_times":
            self.ensure_complete()
            return self.macro_times
        if key == "segment_end_progress":
            return self.segment_end_progress
        if key == "segments":
            return self.segments()
        return default

    def segments(self):
        cdef int i
        cdef list out
        cdef object sample_dist, sample_time, sample_speed
        if self.segments_obj is not None:
            return self.segments_obj
        self.ensure_complete()
        out = []
        for i in range(self.count):
            self.ensure_segment_samples(i)
            sample_dist = self.sample_dist_arrs[i]
            sample_time = self.sample_time_arrs[i]
            sample_speed = self.sample_speed_arrs[i]
            if not isinstance(sample_dist, list):
                sample_dist = [float(v) for v in sample_dist]
            if not isinstance(sample_time, list):
                sample_time = [float(v) for v in sample_time]
            if not isinstance(sample_speed, list):
                sample_speed = [float(v) for v in sample_speed]
            out.append(
                {
                    "start_progress": float(self.start_progress[i]),
                    "end_progress": float(self.end_progress[i]),
                    "progress_len": float(self.progress_len[i]),
                    "start_time": float(self.start_time[i]),
                    "end_time": float(self.start_time[i] + self.duration[i]),
                    "duration": float(self.duration[i]),
                    "distance_m": float(self.distance[i]),
                    "v_entry_ms": float(self.v_entry[i]),
                    "v_exit_ms": float(self.v_exit[i]),
                    "v_peak_ms": float(self.v_peak[i]),
                    "dist_acc_m": float(self.dist_acc[i]),
                    "dist_cruise_m": float(self.dist_cruise[i]),
                    "time_acc_s": float(self.time_acc[i]),
                    "time_cruise_s": float(self.time_cruise[i]),
                    "sample_dist_m": sample_dist,
                    "sample_time_s": sample_time,
                    "sample_speed_ms": sample_speed,
                    "accel_rate_ms2": float(self.accel_rate[i]),
                    "brake_rate_ms2": float(self.brake_rate[i]),
                }
            )
        self.segments_obj = out
        return out


cdef inline object _as_double_array(object values):
    cdef object out
    cdef int i, n
    if isinstance(values, array):
        return values
    try:
        n = len(values)
    except Exception:
        return array("d", [0.0])
    out = array("d", [0.0]) * n
    for i in range(n):
        try:
            out[i] = float(values[i])
        except Exception:
            out[i] = 0.0
    return out


cdef inline double _sample_speed_from_arrays(
    object sample_dist_obj,
    object sample_speed_obj,
    double distance_into_seg_m,
    double distance_total,
    double v_entry,
    double v_exit,
    double v_peak,
    double accel,
    double brake,
    double dist_acc,
    double dist_cruise,
):
    cdef double[:] sample_dist
    cdef double[:] sample_speed
    cdef int n, lo, hi, mid, idx
    cdef double dist_total, x, x0, x1, v0, v1, t, remaining
    try:
        sample_dist = sample_dist_obj
        sample_speed = sample_speed_obj
        n = sample_dist.shape[0]
    except Exception:
        n = 0
    if n < 2 and isinstance(sample_dist_obj, list) and isinstance(sample_speed_obj, list):
        n = len(sample_dist_obj)
        if n >= 2 and len(sample_speed_obj) == n:
            dist_total = max(1e-9, float(sample_dist_obj[n - 1]))
            x = _fp_clamp_double(distance_into_seg_m, 0.0, dist_total)
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if float(sample_dist_obj[mid]) < x:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            if idx <= 0:
                return max(0.5, float(sample_speed_obj[0]))
            if idx >= n:
                return max(0.5, float(sample_speed_obj[n - 1]))
            x0 = float(sample_dist_obj[idx - 1])
            x1 = float(sample_dist_obj[idx])
            v0 = float(sample_speed_obj[idx - 1])
            v1 = float(sample_speed_obj[idx])
            if x1 - x0 <= 1e-9:
                return max(0.5, v1)
            t = (x - x0) / (x1 - x0)
            return max(0.5, v0 + (v1 - v0) * t)
    if n >= 2:
        dist_total = max(1e-9, sample_dist[n - 1])
        x = _fp_clamp_double(distance_into_seg_m, 0.0, dist_total)
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if sample_dist[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if idx <= 0:
            return max(0.5, sample_speed[0])
        if idx >= n:
            return max(0.5, sample_speed[n - 1])
        x0 = sample_dist[idx - 1]
        x1 = sample_dist[idx]
        v0 = sample_speed[idx - 1]
        v1 = sample_speed[idx]
        if x1 - x0 <= 1e-9:
            return max(0.5, v1)
        t = (x - x0) / (x1 - x0)
        return max(0.5, v0 + (v1 - v0) * t)

    x = _fp_clamp_double(distance_into_seg_m, 0.0, max(1e-9, distance_total))
    if x <= dist_acc + 1e-9:
        return sqrt(max((v_entry * v_entry) + (2.0 * accel * x), 0.0))
    if x <= dist_acc + dist_cruise + 1e-9:
        return v_peak
    remaining = max(0.0, distance_total - x)
    return sqrt(max((v_exit * v_exit) + (2.0 * brake * remaining), 0.0))


cdef inline double _sample_elapsed_from_arrays(
    object sample_dist_obj,
    object sample_time_obj,
    double distance_into_seg_m,
    double distance_total,
    double v_entry,
    double v_peak,
    double accel,
    double brake,
    double dist_acc,
    double dist_cruise,
    double time_acc,
    double time_cruise,
):
    cdef double[:] sample_dist
    cdef double[:] sample_time
    cdef int n, lo, hi, mid, idx
    cdef double dist_total, x, d0, d1, t0, t1, w, dist_into_dec, v2, v
    try:
        sample_dist = sample_dist_obj
        sample_time = sample_time_obj
        n = sample_dist.shape[0]
    except Exception:
        n = 0
    if n < 2 and isinstance(sample_dist_obj, list) and isinstance(sample_time_obj, list):
        n = len(sample_dist_obj)
        if n >= 2 and len(sample_time_obj) == n:
            dist_total = max(1e-9, float(sample_dist_obj[n - 1]))
            x = _fp_clamp_double(distance_into_seg_m, 0.0, dist_total)
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if float(sample_dist_obj[mid]) < x:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            if idx <= 0:
                return max(0.0, float(sample_time_obj[0]))
            if idx >= n:
                return max(0.0, float(sample_time_obj[n - 1]))
            d0 = float(sample_dist_obj[idx - 1])
            d1 = float(sample_dist_obj[idx])
            t0 = float(sample_time_obj[idx - 1])
            t1 = float(sample_time_obj[idx])
            if d1 - d0 <= 1e-9:
                return max(0.0, t1)
            w = (x - d0) / (d1 - d0)
            return max(0.0, t0 + (t1 - t0) * w)
    if n >= 2:
        dist_total = max(1e-9, sample_dist[n - 1])
        x = _fp_clamp_double(distance_into_seg_m, 0.0, dist_total)
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if sample_dist[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if idx <= 0:
            return max(0.0, sample_time[0])
        if idx >= n:
            return max(0.0, sample_time[n - 1])
        d0 = sample_dist[idx - 1]
        d1 = sample_dist[idx]
        t0 = sample_time[idx - 1]
        t1 = sample_time[idx]
        if d1 - d0 <= 1e-9:
            return max(0.0, t1)
        w = (x - d0) / (d1 - d0)
        return max(0.0, t0 + (t1 - t0) * w)

    x = _fp_clamp_double(distance_into_seg_m, 0.0, max(1e-9, distance_total))
    if x <= dist_acc + 1e-9:
        if accel <= 1e-9:
            return x / max(0.1, v_entry)
        v = sqrt(max((v_entry * v_entry) + (2.0 * accel * x), 0.0))
        return max(0.0, (v - v_entry) / accel)
    if x <= dist_acc + dist_cruise + 1e-9:
        return max(0.0, time_acc + (x - dist_acc) / max(0.1, v_peak))
    dist_into_dec = max(0.0, x - dist_acc - dist_cruise)
    if brake <= 1e-9:
        return max(0.0, time_acc + time_cruise + dist_into_dec / max(0.1, v_peak))
    v2 = max((v_peak * v_peak) - (2.0 * brake * dist_into_dec), 0.0)
    v = sqrt(v2)
    return max(0.0, time_acc + time_cruise + (v_peak - v) / brake)


cdef tuple _query_fast_profile_at_progress(
    FastProfile fp,
    double progress,
    object hint_obj=None,
):
    cdef int n, hint, idx, lo, hi, mid
    cdef double p, seg_start, seg_end, seg_len, ratio, dist_in_seg
    cdef double speed_ms, speed_kmh, elapsed_s
    fp.ensure_complete()
    n = fp.count
    if n <= 0:
        return (0.0, 0.0, -1)
    p = _fp_normalize_progress_double(progress)
    idx = -1
    try:
        hint = int(hint_obj)
    except Exception:
        hint = -1
    if p >= 1.0:
        idx = n - 1
    elif 0 <= hint < n:
        if p >= fp.start_progress[hint] - 1e-9 and p <= fp.end_progress[hint] + 1e-9:
            idx = hint
    if idx < 0:
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if fp.end_progress[mid] < p:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if idx >= n:
            idx = n - 1
    seg_start = fp.start_progress[idx]
    seg_len = max(1e-9, fp.progress_len[idx])
    ratio = _fp_clamp_double((p - seg_start) / seg_len, 0.0, 1.0)
    dist_in_seg = ratio * max(0.0, fp.distance[idx])
    fp.ensure_segment_samples(idx)
    speed_ms = _sample_speed_from_arrays(
        fp.sample_dist_arrs[idx],
        fp.sample_speed_arrs[idx],
        dist_in_seg,
        fp.distance[idx],
        fp.v_entry[idx],
        fp.v_exit[idx],
        fp.v_peak[idx],
        fp.accel_rate[idx],
        fp.brake_rate[idx],
        fp.dist_acc[idx],
        fp.dist_cruise[idx],
    )
    elapsed_s = fp.start_time[idx] + _sample_elapsed_from_arrays(
        fp.sample_dist_arrs[idx],
        fp.sample_time_arrs[idx],
        dist_in_seg,
        fp.distance[idx],
        fp.v_entry[idx],
        fp.v_peak[idx],
        fp.accel_rate[idx],
        fp.brake_rate[idx],
        fp.dist_acc[idx],
        fp.dist_cruise[idx],
        fp.time_acc[idx],
        fp.time_cruise[idx],
    )
    speed_kmh = max(0.0, speed_ms * 3.6)
    return (speed_kmh, max(0.0, elapsed_s), idx)


cdef tuple _query_fast_profile_speed_at_progress(
    FastProfile fp,
    double progress,
    object hint_obj=None,
):
    cdef int n, hint, idx, lo, hi, mid
    cdef double p, seg_start, seg_len, ratio, dist_in_seg, speed_ms
    n = fp.count
    if n <= 0:
        return (0.0, -1)
    p = _fp_normalize_progress_double(progress)
    idx = -1
    try:
        hint = int(hint_obj)
    except Exception:
        hint = -1
    if p >= 1.0:
        idx = n - 1
    elif 0 <= hint < n:
        if p >= fp.start_progress[hint] - 1e-9 and p <= fp.end_progress[hint] + 1e-9:
            idx = hint
    if idx < 0:
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if fp.end_progress[mid] < p:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if idx >= n:
            idx = n - 1
    seg_start = fp.start_progress[idx]
    seg_len = max(1e-9, fp.progress_len[idx])
    ratio = _fp_clamp_double((p - seg_start) / seg_len, 0.0, 1.0)
    dist_in_seg = ratio * max(0.0, fp.distance[idx])
    fp.ensure_segment_samples(idx)
    speed_ms = _sample_speed_from_arrays(
        fp.sample_dist_arrs[idx],
        fp.sample_speed_arrs[idx],
        dist_in_seg,
        fp.distance[idx],
        fp.v_entry[idx],
        fp.v_exit[idx],
        fp.v_peak[idx],
        fp.accel_rate[idx],
        fp.brake_rate[idx],
        fp.dist_acc[idx],
        fp.dist_cruise[idx],
    )
    return (max(0.0, speed_ms * 3.6), idx)


cdef double _fast_profile_progress_for_elapsed(
    FastProfile fp,
    double elapsed_s,
    double target_lap,
):
    cdef double lap_time, target, base_elapsed, cursor, next_cursor
    cdef int i
    cdef double elapsed_in_seg, dist_in_seg, frac, seg_dist, seg_start, seg_end
    fp.ensure_complete()
    if fp.count <= 0:
        return 0.0
    lap_time = max(1e-9, fp.lap_time)
    target = max(1e-9, target_lap)
    base_elapsed = _fp_clamp_double(elapsed_s, 0.0, target) * (lap_time / target)
    cursor = 0.0
    for i in range(fp.count):
        next_cursor = cursor + max(1e-9, fp.duration[i])
        if base_elapsed <= next_cursor + 1e-9:
            fp.ensure_segment_samples(i)
            elapsed_in_seg = max(0.0, base_elapsed - cursor)
            # Invert elapsed with a binary search over the stored exact sample
            # curve. This keeps the same 1 m solver samples used by the dict
            # fallback without walking Python dictionaries.
            dist_in_seg = _distance_from_elapsed_fast(
                fp.sample_time_arrs[i],
                fp.sample_dist_arrs[i],
                elapsed_in_seg,
                fp.distance[i],
                fp.v_entry[i],
                fp.v_peak[i],
                fp.accel_rate[i],
                fp.brake_rate[i],
                fp.dist_acc[i],
                fp.dist_cruise[i],
                fp.time_acc[i],
                fp.time_cruise[i],
            )
            seg_dist = max(1e-9, fp.distance[i])
            frac = _fp_clamp_double(dist_in_seg / seg_dist, 0.0, 1.0)
            seg_start = fp.start_progress[i]
            seg_end = fp.end_progress[i]
            return _fp_clamp_double(seg_start + (seg_end - seg_start) * frac, 0.0, 1.0)
        cursor = next_cursor
    return 1.0


cdef double _distance_from_elapsed_fast(
    object sample_time_obj,
    object sample_dist_obj,
    double elapsed_in_seg_s,
    double distance_total,
    double v_entry,
    double v_peak,
    double accel,
    double brake,
    double dist_acc,
    double dist_cruise,
    double time_acc,
    double time_cruise,
):
    cdef double[:] sample_time
    cdef double[:] sample_dist
    cdef int n, lo, hi, mid, idx
    cdef double total_t, t, t0, t1, d0, d1, w, t_after_acc, t_dec, dist, v2, v
    try:
        sample_time = sample_time_obj
        sample_dist = sample_dist_obj
        n = sample_time.shape[0]
    except Exception:
        n = 0
    if n < 2 and isinstance(sample_time_obj, list) and isinstance(sample_dist_obj, list):
        n = len(sample_time_obj)
        if n >= 2 and len(sample_dist_obj) == n:
            total_t = max(1e-9, float(sample_time_obj[n - 1]))
            t = _fp_clamp_double(elapsed_in_seg_s, 0.0, total_t)
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if float(sample_time_obj[mid]) < t:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            if idx <= 0:
                return max(0.0, float(sample_dist_obj[0]))
            if idx >= n:
                return max(0.0, float(sample_dist_obj[n - 1]))
            t0 = float(sample_time_obj[idx - 1])
            t1 = float(sample_time_obj[idx])
            d0 = float(sample_dist_obj[idx - 1])
            d1 = float(sample_dist_obj[idx])
            if t1 - t0 <= 1e-9:
                return _fp_clamp_double(d1, 0.0, max(0.0, distance_total))
            w = (t - t0) / (t1 - t0)
            return _fp_clamp_double(d0 + (d1 - d0) * w, 0.0, max(0.0, distance_total))
    if n >= 2:
        total_t = max(1e-9, sample_time[n - 1])
        t = _fp_clamp_double(elapsed_in_seg_s, 0.0, total_t)
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if sample_time[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if idx <= 0:
            return max(0.0, sample_dist[0])
        if idx >= n:
            return max(0.0, sample_dist[n - 1])
        t0 = sample_time[idx - 1]
        t1 = sample_time[idx]
        d0 = sample_dist[idx - 1]
        d1 = sample_dist[idx]
        if t1 - t0 <= 1e-9:
            return _fp_clamp_double(d1, 0.0, max(0.0, distance_total))
        w = (t - t0) / (t1 - t0)
        return _fp_clamp_double(d0 + (d1 - d0) * w, 0.0, max(0.0, distance_total))

    t = _fp_clamp_double(elapsed_in_seg_s, 0.0, max(1e-9, time_acc + time_cruise))
    if t <= time_acc + 1e-9:
        return _fp_clamp_double((v_entry * t) + (0.5 * accel * t * t), 0.0, distance_total)
    t_after_acc = t - time_acc
    if t_after_acc <= time_cruise + 1e-9:
        return _fp_clamp_double(dist_acc + (v_peak * t_after_acc), 0.0, distance_total)
    t_dec = t_after_acc - time_cruise
    dist = dist_acc + dist_cruise + (v_peak * t_dec) - (0.5 * brake * t_dec * t_dec)
    return _fp_clamp_double(dist, 0.0, distance_total)


cdef inline double _accel_curve_multiplier(double speed_kmh):
    cdef double x0, x1, y0, y1, t
    if speed_kmh <= 0.0:
        return 1.75
    elif speed_kmh <= 50.0:
        x0, x1, y0, y1 = 0.0, 50.0, 1.75, 1.60
    elif speed_kmh <= 100.0:
        x0, x1, y0, y1 = 50.0, 100.0, 1.60, 1.45
    elif speed_kmh <= 150.0:
        x0, x1, y0, y1 = 100.0, 150.0, 1.45, 1.25
    elif speed_kmh <= 200.0:
        x0, x1, y0, y1 = 150.0, 200.0, 1.25, 0.95
    elif speed_kmh <= 250.0:
        x0, x1, y0, y1 = 200.0, 250.0, 0.95, 0.75
    elif speed_kmh <= 300.0:
        x0, x1, y0, y1 = 250.0, 300.0, 0.75, 0.40
    elif speed_kmh <= 330.0:
        x0, x1, y0, y1 = 300.0, 330.0, 0.40, 0.20
    elif speed_kmh <= 350.0:
        x0, x1, y0, y1 = 330.0, 350.0, 0.20, 0.05
    else:
        return 0.05
    if x1 - x0 <= 1e-9:
        return y1
    t = (speed_kmh - x0) / (x1 - x0)
    return y0 + (y1 - y0) * t


cdef inline double _effective_accel_rate(
    double accel_base_scaled,
    double speed_ms,
    double drag_coeff,
):
    cdef double kmh, speed, base_rate, drag_loss, out
    kmh = speed_ms * 3.6
    if kmh < 0.0:
        kmh = 0.0
    speed = speed_ms
    if speed < 0.0:
        speed = 0.0
    base_rate = accel_base_scaled * _accel_curve_multiplier(kmh)
    drag_loss = drag_coeff * speed * speed
    out = base_rate - drag_loss
    if out < 0.05:
        out = 0.05
    return out


cdef inline double _curve_at_gap(
    double gap_s,
    double[:] curve_gaps,
    double[:] curve_vals,
):
    cdef int n = curve_gaps.shape[0]
    cdef int i
    cdef double g0, g1, v0, v1, t
    if n <= 0:
        return 0.0
    if gap_s <= curve_gaps[0]:
        return curve_vals[0]
    for i in range(1, n):
        g0 = curve_gaps[i - 1]
        g1 = curve_gaps[i]
        v0 = curve_vals[i - 1]
        v1 = curve_vals[i]
        if gap_s <= g1:
            if g1 - g0 <= 1e-9:
                return v1
            t = (gap_s - g0) / (g1 - g0)
            return v0 + (v1 - v0) * t
    return curve_vals[n - 1]


cdef inline double _clamp_double(double value, double lo, double hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


cdef inline int _zone_index(object zone_type):
    if zone_type == "low":
        return 0
    elif zone_type == "med":
        return 1
    elif zone_type == "high":
        return 2
    elif zone_type == "straight":
        return 3
    return -1


cdef inline double _normalize_progress_double(double progress):
    if progress <= 0.0:
        return 0.0
    if progress >= 1.0:
        return 1.0
    return progress


cdef double _segment_speed_ms_at_distance_kernel(object seg, double distance_into_seg_m) except? -1:
    cdef object sample_dist = seg.get("sample_dist_m")
    cdef object sample_speed = seg.get("sample_speed_ms")
    cdef int lo, hi, mid, n, idx
    cdef double dist_total, x, x0, x1, v0, v1, t
    cdef double v_entry, v_exit, v_peak, accel, brake, dist_acc, dist_cruise, remaining

    if isinstance(sample_dist, list) and isinstance(sample_speed, list):
        n = len(sample_dist)
        if n >= 2 and len(sample_speed) == n:
            dist_total = max(1e-9, float(sample_dist[n - 1]))
            x = _clamp_double(distance_into_seg_m, 0.0, dist_total)
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if float(sample_dist[mid]) < x:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            if idx <= 0:
                return max(0.5, float(sample_speed[0]))
            if idx >= n:
                return max(0.5, float(sample_speed[n - 1]))
            x0 = float(sample_dist[idx - 1])
            x1 = float(sample_dist[idx])
            v0 = float(sample_speed[idx - 1])
            v1 = float(sample_speed[idx])
            if x1 - x0 <= 1e-9:
                return max(0.5, v1)
            t = (x - x0) / (x1 - x0)
            return max(0.5, v0 + (v1 - v0) * t)

    dist_total = max(1e-9, float(seg.get("distance_m", 0.0)))
    x = _clamp_double(distance_into_seg_m, 0.0, dist_total)
    v_entry = max(0.5, float(seg.get("v_entry_ms", 0.5)))
    v_exit = max(0.5, float(seg.get("v_exit_ms", 0.5)))
    v_peak = max(v_entry, v_exit, float(seg.get("v_peak_ms", max(v_entry, v_exit))))
    accel = max(0.05, float(seg.get("accel_rate_ms2", 0.05)))
    brake = max(0.05, float(seg.get("brake_rate_ms2", 0.05)))
    dist_acc = max(0.0, float(seg.get("dist_acc_m", 0.0)))
    dist_cruise = max(0.0, float(seg.get("dist_cruise_m", 0.0)))

    if x <= dist_acc + 1e-9:
        return sqrt(max((v_entry * v_entry) + (2.0 * accel * x), 0.0))
    if x <= dist_acc + dist_cruise + 1e-9:
        return v_peak
    remaining = max(0.0, dist_total - x)
    return sqrt(max((v_exit * v_exit) + (2.0 * brake * remaining), 0.0))


cdef double _segment_elapsed_from_distance_kernel(object seg, double distance_into_seg_m) except? -1:
    cdef object sample_dist = seg.get("sample_dist_m")
    cdef object sample_time = seg.get("sample_time_s")
    cdef int lo, hi, mid, n, idx
    cdef double total_d, x, d0, d1, t0, t1, w
    cdef double dist_total, v0, vp, accel, brake, dist_acc, dist_cruise, t_acc, t_cruise
    cdef double dist_into_dec, v2, v

    if isinstance(sample_dist, list) and isinstance(sample_time, list):
        n = len(sample_dist)
        if n >= 2 and len(sample_time) == n:
            total_d = max(1e-9, float(sample_dist[n - 1]))
            x = _clamp_double(distance_into_seg_m, 0.0, total_d)
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) // 2
                if float(sample_dist[mid]) < x:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            if idx <= 0:
                return max(0.0, float(sample_time[0]))
            if idx >= n:
                return max(0.0, float(sample_time[n - 1]))
            d0 = float(sample_dist[idx - 1])
            d1 = float(sample_dist[idx])
            t0 = float(sample_time[idx - 1])
            t1 = float(sample_time[idx])
            if d1 - d0 <= 1e-9:
                return max(0.0, t1)
            w = (x - d0) / (d1 - d0)
            return max(0.0, t0 + (t1 - t0) * w)

    dist_total = max(1e-9, float(seg.get("distance_m", 0.0)))
    x = _clamp_double(distance_into_seg_m, 0.0, dist_total)
    v0 = max(0.5, float(seg.get("v_entry_ms", 0.5)))
    vp = max(v0, float(seg.get("v_peak_ms", v0)))
    accel = max(0.05, float(seg.get("accel_rate_ms2", 0.05)))
    brake = max(0.05, float(seg.get("brake_rate_ms2", 0.05)))
    dist_acc = max(0.0, float(seg.get("dist_acc_m", 0.0)))
    dist_cruise = max(0.0, float(seg.get("dist_cruise_m", 0.0)))
    t_acc = max(0.0, float(seg.get("time_acc_s", 0.0)))
    t_cruise = max(0.0, float(seg.get("time_cruise_s", 0.0)))

    if x <= dist_acc + 1e-9:
        if accel <= 1e-9:
            return x / max(0.1, v0)
        v = sqrt(max((v0 * v0) + (2.0 * accel * x), 0.0))
        return max(0.0, (v - v0) / accel)
    if x <= dist_acc + dist_cruise + 1e-9:
        return max(0.0, t_acc + (x - dist_acc) / max(0.1, vp))
    dist_into_dec = max(0.0, x - dist_acc - dist_cruise)
    if brake <= 1e-9:
        return max(0.0, t_acc + t_cruise + dist_into_dec / max(0.1, vp))
    v2 = max((vp * vp) - (2.0 * brake * dist_into_dec), 0.0)
    v = sqrt(v2)
    return max(0.0, t_acc + t_cruise + (vp - v) / brake)


cpdef tuple query_profile_at_progress(
    object profile_obj,
    double progress,
    object hint_obj=None,
):
    cdef object fast_profile
    cdef object segments
    cdef object ends
    cdef int n, hint, idx, lo, hi, mid
    cdef object seg
    cdef double p, seg_start, seg_len, seg_dist, ratio, dist_in_seg
    cdef double speed_kmh, elapsed_s, lap_time, seg_end
    if isinstance(profile_obj, FastProfile):
        return _query_fast_profile_at_progress(profile_obj, progress, hint_obj)
    try:
        fast_profile = profile_obj.get("_fast_profile")
        if isinstance(fast_profile, FastProfile):
            return _query_fast_profile_at_progress(fast_profile, progress, hint_obj)
    except Exception:
        pass
    segments = profile_obj.get("segments", [])
    ends = profile_obj.get("segment_end_progress", [])

    if not isinstance(segments, list):
        return (0.0, 0.0, -1)
    n = len(segments)
    if n <= 0:
        return (0.0, 0.0, -1)

    p = _normalize_progress_double(progress)
    idx = -1
    try:
        hint = int(hint_obj)
    except Exception:
        hint = -1

    if p >= 1.0:
        idx = n - 1
    elif 0 <= hint < n:
        seg = segments[hint]
        if isinstance(seg, dict):
            seg_start = float(seg.get("start_progress", 0.0))
            seg_end = float(seg.get("end_progress", seg_start))
            if p >= seg_start - 1e-9 and p <= seg_end + 1e-9:
                idx = hint

    if idx < 0 and isinstance(ends, list) and len(ends) == n:
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if float(ends[mid]) < p:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if idx >= n:
            idx = n - 1

    if idx < 0:
        for mid in range(n):
            seg = segments[mid]
            if not isinstance(seg, dict):
                continue
            seg_start = float(seg.get("start_progress", 0.0))
            seg_end = float(seg.get("end_progress", seg_start))
            if p <= seg_end + 1e-9 and p >= seg_start - 1e-9:
                idx = mid
                break

    if idx < 0:
        return (0.0, 0.0, -1)

    seg = segments[idx]
    if not isinstance(seg, dict):
        return (0.0, 0.0, idx)

    seg_start = float(seg.get("start_progress", 0.0))
    seg_len = max(1e-9, float(seg.get("progress_len", 0.0)))
    ratio = _clamp_double((p - seg_start) / seg_len, 0.0, 1.0)
    seg_dist = max(0.0, float(seg.get("distance_m", 0.0)))
    dist_in_seg = ratio * seg_dist
    speed_kmh = _segment_speed_ms_at_distance_kernel(seg, dist_in_seg) * 3.6
    elapsed_s = float(seg.get("start_time", 0.0)) + _segment_elapsed_from_distance_kernel(seg, dist_in_seg)

    if p <= 0.0:
        elapsed_s = 0.0
    elif p >= 1.0:
        lap_time = max(0.0, float(profile_obj.get("lap_time", 0.0)))
        if lap_time > elapsed_s:
            elapsed_s = lap_time

    if speed_kmh < 0.0:
        speed_kmh = 0.0
    if elapsed_s < 0.0:
        elapsed_s = 0.0
    return (float(speed_kmh), float(elapsed_s), idx)


cpdef tuple query_profile_speed_at_progress(
    object profile_obj,
    double progress,
    object hint_obj=None,
):
    """Return the exact local speed without materializing unrelated segments."""
    cdef object fast_profile
    cdef tuple result
    if isinstance(profile_obj, FastProfile):
        return _query_fast_profile_speed_at_progress(profile_obj, progress, hint_obj)
    try:
        fast_profile = profile_obj.get("_fast_profile")
        if isinstance(fast_profile, FastProfile):
            return _query_fast_profile_speed_at_progress(fast_profile, progress, hint_obj)
    except Exception:
        pass
    result = query_profile_at_progress(profile_obj, progress, hint_obj)
    return (float(result[0]), int(result[2]))


cpdef list compute_aero_effect_bases(
    object gaps_obj,
    object team_factors_obj,
    object wet_atten_obj,
    object dirty_trait_factors_obj,
    object dirty_curve_gaps_obj,
    object dirty_curve_vals_obj,
    object dirty_brake_curve_gaps_obj,
    object dirty_brake_curve_vals_obj,
    object slip_curve_gaps_obj,
    object slip_curve_vals_obj,
    double track_dirty_air_mult,
    bint dirty_enabled=True,
    bint slip_enabled=True,
):
    cdef double[:] gaps = gaps_obj
    cdef double[:] team_factors = team_factors_obj
    cdef double[:] wet = wet_atten_obj
    cdef double[:] dirty_trait_factors = dirty_trait_factors_obj
    cdef double[:] dirty_gaps = dirty_curve_gaps_obj
    cdef double[:] dirty_vals = dirty_curve_vals_obj
    cdef double[:] dirty_brake_gaps = dirty_brake_curve_gaps_obj
    cdef double[:] dirty_brake_vals = dirty_brake_curve_vals_obj
    cdef double[:] slip_gaps = slip_curve_gaps_obj
    cdef double[:] slip_vals = slip_curve_vals_obj
    cdef int n = gaps.shape[0]
    cdef int i
    cdef double g, dirty_kmh, slip_kmh, dirty_brake_ms2
    cdef list out = []

    if (
        team_factors.shape[0] != n
        or wet.shape[0] != n
        or dirty_trait_factors.shape[0] != n
    ):
        return out

    for i in range(n):
        g = gaps[i]
        if g < 0.0:
            g = 0.0
        dirty_kmh = 0.0
        slip_kmh = 0.0
        dirty_brake_ms2 = 0.0
        if dirty_enabled and dirty_gaps.shape[0] > 0 and dirty_vals.shape[0] == dirty_gaps.shape[0]:
            dirty_kmh = _curve_at_gap(g, dirty_gaps, dirty_vals)
            dirty_kmh *= team_factors[i]
            dirty_kmh *= track_dirty_air_mult
            dirty_kmh *= wet[i]
            dirty_kmh *= dirty_trait_factors[i]
            if dirty_kmh < 0.0:
                dirty_kmh = 0.0
        if dirty_enabled and dirty_brake_gaps.shape[0] > 0 and dirty_brake_vals.shape[0] == dirty_brake_gaps.shape[0]:
            dirty_brake_ms2 = _curve_at_gap(g, dirty_brake_gaps, dirty_brake_vals)
            dirty_brake_ms2 *= team_factors[i]
            dirty_brake_ms2 *= track_dirty_air_mult
            dirty_brake_ms2 *= wet[i]
            dirty_brake_ms2 *= dirty_trait_factors[i]
            if dirty_brake_ms2 < 0.0:
                dirty_brake_ms2 = 0.0
        if slip_enabled and slip_gaps.shape[0] > 0 and slip_vals.shape[0] == slip_gaps.shape[0]:
            slip_kmh = _curve_at_gap(g, slip_gaps, slip_vals)
            slip_kmh *= wet[i]
            if slip_kmh < 0.0:
                slip_kmh = 0.0
        out.append((float(dirty_kmh), float(slip_kmh), float(dirty_brake_ms2)))
    return out


cpdef list compute_dirty_air_stack_effects(
    object layer2_gaps_obj,
    object layer3_gaps_obj,
    object team_factors_obj,
    object wet_atten_obj,
    object dirty_trait_factors_obj,
    object dirty_curve_gaps_obj,
    object dirty_curve_vals_obj,
    object dirty_brake_curve_gaps_obj,
    object dirty_brake_curve_vals_obj,
    double track_dirty_air_mult,
    double layer2_mult,
    double layer3_mult,
    bint dirty_enabled=True,
):
    cdef double[:] layer2_gaps = layer2_gaps_obj
    cdef double[:] layer3_gaps = layer3_gaps_obj
    cdef double[:] team_factors = team_factors_obj
    cdef double[:] wet = wet_atten_obj
    cdef double[:] dirty_trait_factors = dirty_trait_factors_obj
    cdef double[:] dirty_gaps = dirty_curve_gaps_obj
    cdef double[:] dirty_vals = dirty_curve_vals_obj
    cdef double[:] dirty_brake_gaps = dirty_brake_curve_gaps_obj
    cdef double[:] dirty_brake_vals = dirty_brake_curve_vals_obj
    cdef int n = layer2_gaps.shape[0]
    cdef int i
    cdef double g, mult, dirty_kmh, dirty_brake_ms2
    cdef list out = []

    if (
        layer3_gaps.shape[0] != n
        or team_factors.shape[0] != n
        or wet.shape[0] != n
        or dirty_trait_factors.shape[0] != n
    ):
        return out

    for i in range(n):
        dirty_kmh = 0.0
        dirty_brake_ms2 = 0.0
        if dirty_enabled:
            g = layer2_gaps[i]
            mult = layer2_mult
            if g >= 0.0 and mult > 0.0:
                if dirty_gaps.shape[0] > 0 and dirty_vals.shape[0] == dirty_gaps.shape[0]:
                    dirty_kmh += _curve_at_gap(g, dirty_gaps, dirty_vals) * mult
                if dirty_brake_gaps.shape[0] > 0 and dirty_brake_vals.shape[0] == dirty_brake_gaps.shape[0]:
                    dirty_brake_ms2 += _curve_at_gap(g, dirty_brake_gaps, dirty_brake_vals) * mult

            g = layer3_gaps[i]
            mult = layer3_mult
            if g >= 0.0 and mult > 0.0:
                if dirty_gaps.shape[0] > 0 and dirty_vals.shape[0] == dirty_gaps.shape[0]:
                    dirty_kmh += _curve_at_gap(g, dirty_gaps, dirty_vals) * mult
                if dirty_brake_gaps.shape[0] > 0 and dirty_brake_vals.shape[0] == dirty_brake_gaps.shape[0]:
                    dirty_brake_ms2 += _curve_at_gap(g, dirty_brake_gaps, dirty_brake_vals) * mult

            dirty_kmh *= team_factors[i]
            dirty_kmh *= track_dirty_air_mult
            dirty_kmh *= wet[i]
            dirty_kmh *= dirty_trait_factors[i]
            dirty_brake_ms2 *= team_factors[i]
            dirty_brake_ms2 *= track_dirty_air_mult
            dirty_brake_ms2 *= wet[i]
            dirty_brake_ms2 *= dirty_trait_factors[i]
            if dirty_kmh < 0.0:
                dirty_kmh = 0.0
            if dirty_brake_ms2 < 0.0:
                dirty_brake_ms2 = 0.0
        out.append((float(dirty_kmh), float(dirty_brake_ms2)))
    return out


cpdef list compute_aero_effect_bases_from_progress(
    object delta_progress_obj,
    object local_rates_obj,
    object team_factors_obj,
    object wet_atten_obj,
    object dirty_trait_factors_obj,
    object dirty_curve_gaps_obj,
    object dirty_curve_vals_obj,
    object dirty_brake_curve_gaps_obj,
    object dirty_brake_curve_vals_obj,
    object slip_curve_gaps_obj,
    object slip_curve_vals_obj,
    double track_dirty_air_mult,
    double max_gap_s_with_margin,
    bint dirty_enabled=True,
    bint slip_enabled=True,
):
    cdef double[:] delta_prog = delta_progress_obj
    cdef double[:] rates = local_rates_obj
    cdef double[:] team_factors = team_factors_obj
    cdef double[:] wet = wet_atten_obj
    cdef double[:] dirty_trait_factors = dirty_trait_factors_obj
    cdef double[:] dirty_gaps = dirty_curve_gaps_obj
    cdef double[:] dirty_vals = dirty_curve_vals_obj
    cdef double[:] dirty_brake_gaps = dirty_brake_curve_gaps_obj
    cdef double[:] dirty_brake_vals = dirty_brake_curve_vals_obj
    cdef double[:] slip_gaps = slip_curve_gaps_obj
    cdef double[:] slip_vals = slip_curve_vals_obj
    cdef int n = delta_prog.shape[0]
    cdef int i
    cdef double dprog, rate, gap_s, dirty_kmh, slip_kmh, dirty_brake_ms2
    cdef list out = []

    if (
        rates.shape[0] != n
        or team_factors.shape[0] != n
        or wet.shape[0] != n
        or dirty_trait_factors.shape[0] != n
    ):
        return out

    for i in range(n):
        dprog = delta_prog[i]
        rate = rates[i]
        if dprog <= 1e-9 or rate <= 1e-9:
            out.append((0.0, 0.0))
            continue
        gap_s = dprog / rate
        if gap_s < 0.0:
            gap_s = 0.0
        if gap_s > max_gap_s_with_margin:
            out.append((0.0, 0.0))
            continue

        dirty_kmh = 0.0
        slip_kmh = 0.0
        dirty_brake_ms2 = 0.0
        if dirty_enabled and dirty_gaps.shape[0] > 0 and dirty_vals.shape[0] == dirty_gaps.shape[0]:
            dirty_kmh = _curve_at_gap(gap_s, dirty_gaps, dirty_vals)
            dirty_kmh *= team_factors[i]
            dirty_kmh *= track_dirty_air_mult
            dirty_kmh *= wet[i]
            dirty_kmh *= dirty_trait_factors[i]
            if dirty_kmh < 0.0:
                dirty_kmh = 0.0
        if dirty_enabled and dirty_brake_gaps.shape[0] > 0 and dirty_brake_vals.shape[0] == dirty_brake_gaps.shape[0]:
            dirty_brake_ms2 = _curve_at_gap(gap_s, dirty_brake_gaps, dirty_brake_vals)
            dirty_brake_ms2 *= team_factors[i]
            dirty_brake_ms2 *= track_dirty_air_mult
            dirty_brake_ms2 *= wet[i]
            dirty_brake_ms2 *= dirty_trait_factors[i]
            if dirty_brake_ms2 < 0.0:
                dirty_brake_ms2 = 0.0
        if slip_enabled and slip_gaps.shape[0] > 0 and slip_vals.shape[0] == slip_gaps.shape[0]:
            slip_kmh = _curve_at_gap(gap_s, slip_gaps, slip_vals)
            slip_kmh *= wet[i]
            if slip_kmh < 0.0:
                slip_kmh = 0.0
        out.append((float(dirty_kmh), float(slip_kmh), float(dirty_brake_ms2)))
    return out


cpdef double estimate_next_boundary_step(
    object progress_obj,
    object local_rates_obj,
    object base_laps_obj,
    object sector_splits_obj,
    double max_step,
    double min_step=0.02,
):
    cdef double[:] progress = progress_obj
    cdef double[:] rates = local_rates_obj
    cdef double[:] base_laps = base_laps_obj
    cdef double[:] splits = sector_splits_obj
    cdef int n = progress.shape[0]
    cdef int split_n = splits.shape[0]
    cdef int i, j
    cdef double prog, next_split, gap, rate, secs
    cdef double estimate = -1.0

    if rates.shape[0] != n or base_laps.shape[0] != n:
        return max_step
    if n <= 0:
        return max_step

    for i in range(n):
        prog = progress[i]
        next_split = 1.0
        for j in range(split_n):
            if splits[j] > prog + 1e-9:
                next_split = splits[j]
                break
        gap = next_split - prog
        if gap <= 1e-9:
            continue
        rate = rates[i]
        if rate > 1e-9:
            secs = gap / rate
        else:
            secs = gap * base_laps[i]
        if secs <= 1e-9:
            continue
        if estimate < 0.0 or secs < estimate:
            estimate = secs

    if estimate < 0.0:
        return max_step
    if estimate > max_step:
        estimate = max_step
    if estimate < min_step:
        estimate = min_step
    return estimate


cpdef double time_at_distance_from_points(
    object points_obj,
    double target_s_m,
):
    cdef Py_ssize_t n, lo, hi, mid
    cdef tuple pt0, pt1, pth
    cdef double t0, s0, t1, s1, w

    try:
        n = len(points_obj)
    except Exception:
        return NAN
    if n <= 0:
        return NAN

    try:
        pt0 = points_obj[0]
        t0 = float(pt0[0])
        s0 = float(pt0[1])
    except Exception:
        return NAN

    if n == 1:
        if target_s_m <= s0 + 1e-9:
            return t0
        return NAN
    if target_s_m <= s0 + 1e-9:
        return t0

    try:
        pt1 = points_obj[n - 1]
        t1 = float(pt1[0])
        s1 = float(pt1[1])
    except Exception:
        return NAN
    if target_s_m > s1 + 1e-9:
        return NAN

    lo = 0
    hi = n - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        try:
            pth = points_obj[mid]
            if float(pth[1]) < target_s_m:
                lo = mid
            else:
                hi = mid
        except Exception:
            return NAN

    try:
        pt0 = points_obj[lo]
        pt1 = points_obj[hi]
        t0 = float(pt0[0])
        s0 = float(pt0[1])
        t1 = float(pt1[0])
        s1 = float(pt1[1])
    except Exception:
        return NAN
    if s1 - s0 <= 1e-9:
        if t0 > t1:
            return t0
        return t1
    w = (target_s_m - s0) / (s1 - s0)
    if w < 0.0:
        w = 0.0
    elif w > 1.0:
        w = 1.0
    return t0 + (t1 - t0) * w


cpdef list compute_progress_deltas(
    object local_rates_obj,
    object base_laps_obj,
    double dt_sim,
):
    cdef double[:] local_rates = local_rates_obj
    cdef double[:] base_laps = base_laps_obj
    cdef int n = local_rates.shape[0]
    cdef int i
    cdef double lr, ds
    cdef list out = []

    if base_laps.shape[0] != n:
        return out
    if n <= 0 or dt_sim <= 0.0:
        return out

    for i in range(n):
        lr = local_rates[i]
        if lr > 1e-9:
            ds = lr * dt_sim
        else:
            ds = dt_sim / base_laps[i]
        if ds < 0.0:
            ds = 0.0
        out.append(float(ds))
    return out


cpdef bint incident_segment_crosses_marker(
    double start_progress,
    double end_progress,
    double marker_progress,
) noexcept:
    """Fast pending-incident marker crossing check for a forward lap segment."""
    return (
        marker_progress > start_progress + 1e-9
        and marker_progress <= end_progress + 1e-9
    )


cpdef double incident_marker_from_unit(
    object markers_obj,
    double unit_value,
    double fallback_start=0.15,
    double fallback_end=0.85,
):
    """Select a safe on-track incident point from authored markers or a fallback range."""
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t index = 0
    cdef double unit = unit_value
    cdef double marker
    cdef double start = fallback_start
    cdef double end = fallback_end

    if unit < 0.0:
        unit = 0.0
    elif unit >= 1.0:
        unit = 0.999999999999
    try:
        count = len(markers_obj)
    except Exception:
        count = 0
    if count > 0:
        index = <Py_ssize_t>(unit * count)
        if index >= count:
            index = count - 1
        try:
            marker = float(markers_obj[index])
        except Exception:
            marker = -1.0
        if marker > 0.02 and marker < 0.98:
            return marker

    if start < 0.02:
        start = 0.02
    if end > 0.98:
        end = 0.98
    if end < start:
        marker = start
        start = end
        end = marker
    return start + ((end - start) * unit)


cpdef double slipstream_delta_seconds_from_kmh(
    double kmh_gain,
    double kmh_per_point,
    double delta_seconds_per_rating,
):
    cdef double gain = kmh_gain
    cdef double points
    if gain < 0.0:
        gain = 0.0
    if kmh_per_point <= 1e-9 or gain <= 1e-9:
        return 0.0
    points = gain / kmh_per_point
    return -delta_seconds_per_rating * points


cpdef tuple smooth_slip_delta(
    double raw_delta_s,
    double prev_delta_s,
    double now_t,
    double next_update_t,
    double quantum_s,
    double deadband_s,
    double interval_s,
):
    cdef double out_delta
    cdef double q
    cdef double db
    cdef double next_t

    # Cadence hold.
    if now_t + 1e-9 < next_update_t:
        return (float(prev_delta_s), float(next_update_t))

    q = quantum_s
    if q < 0.001:
        q = 0.001
    out_delta = raw_delta_s
    if out_delta > -1e-12 and out_delta < 1e-12:
        out_delta = 0.0
    else:
        out_delta = round(out_delta / q) * q

    db = deadband_s
    if db < 0.0:
        db = 0.0
    if out_delta - prev_delta_s <= db and out_delta - prev_delta_s >= -db:
        out_delta = prev_delta_s

    next_t = now_t + (interval_s if interval_s > 0.0 else 0.0)
    return (float(out_delta), float(next_t))


cpdef tuple build_lap_inputs_cache_key(
    object team_name,
    double slow_delta,
    double med_delta,
    double high_delta,
    double effective_straight_delta,
    double team_pace_delta,
    double braking,
    double acceleration,
    double engine_power_rating,
    double car_mass_kg,
    double engine_mass_kg,
    double fuel_mass_kg,
    double driver_cornering,
    double driver_braking,
    object normalized_comp,
    double wear,
    double wetness,
    double tyre_temp,
    double mass_ref_kg,
    double driver_cornering_kmh_per_point,
    double driver_braking_ms2_per_point,
    bint apply_consistency,
    double corner_nerf,
    double brake_nerf,
    double part_corner_nerf,
    double aero_corner_nerf,
    double aero_brake_nerf,
    double aero_corner_trait_delta,
    double aero_brake_trait_mult,
    double aero_straight_delta,
    double supplier_pace_rating,
    double contract_grip_bonus_mult,
    double mass_bin,
    double wear_quantum,
):
    cdef double mb = mass_bin
    cdef double wq = wear_quantum
    cdef double fuel_bucket
    cdef double wear_bucket
    if mb < 0.5:
        mb = 0.5
    if wq < 0.001:
        wq = 0.001
    fuel_bucket = round((fuel_mass_kg if fuel_mass_kg > 0.0 else 0.0) / mb) * mb
    wear_bucket = round((wear if wear > 0.0 else 0.0) / wq) * wq
    return (
        team_name,
        round(slow_delta, 3),
        round(med_delta, 3),
        round(high_delta, 3),
        round(effective_straight_delta, 3),
        round(team_pace_delta, 3),
        round(braking, 3),
        round(acceleration, 3),
        round(engine_power_rating, 3),
        round((car_mass_kg if car_mass_kg > 300.0 else 300.0), 3),
        round((engine_mass_kg if engine_mass_kg > 50.0 else 50.0), 3),
        round(fuel_bucket, 3),
        round(driver_cornering, 3),
        round(driver_braking, 3),
        normalized_comp,
        round(wear_bucket, 3),
        round(wetness, 3),
        round(tyre_temp, 3),
        round(mass_ref_kg, 3),
        round(driver_cornering_kmh_per_point, 3),
        round(driver_braking_ms2_per_point, 3),
        bool(apply_consistency),
        round(corner_nerf, 3),
        round(brake_nerf, 3),
        round(part_corner_nerf, 3),
        round(aero_corner_nerf, 2),
        round(aero_brake_nerf, 3),
        round(aero_corner_trait_delta, 3),
        round(aero_brake_trait_mult, 3),
        round(aero_straight_delta, 3),
        round(supplier_pace_rating, 3),
        round(contract_grip_bonus_mult, 6),
    )


cpdef tuple assemble_lap_inputs_numeric(
    double slow_points,
    double med_points,
    double high_points,
    double straight_points,
    double team_pace_points,
    double braking_points,
    double acceleration_points,
    double setup_slow_kmh,
    double setup_med_kmh,
    double setup_high_kmh,
    double setup_straight_kmh,
    double pressure_slow_kmh,
    double pressure_med_kmh,
    double pressure_high_kmh,
    double pressure_straight_kmh,
    double pressure_accel_ms2,
    double suspension_slow_kmh,
    double suspension_med_kmh,
    double suspension_high_kmh,
    double suspension_straight_kmh,
    double low_kmh_per_point,
    double med_kmh_per_point,
    double high_kmh_per_point,
    double straight_kmh_per_point,
    double driver_cornering_kmh_per_point,
    double driver_cornering,
    double weekend_cornering_variability_kmh,
    double driver_braking_ms2_per_point,
    double driver_braking,
    double corner_nerf,
    double brake_nerf,
    double part_corner_nerf,
    double aero_corner_nerf,
    double aero_brake_nerf,
    double aero_corner_trait_delta,
    double aero_brake_trait_mult,
    double aero_straight_delta_s,
    double delta_seconds_per_rating,
    double accel_ms2_per_rating,
):
    cdef double effective_corner_bonus
    cdef double effective_brake_bonus
    cdef double effective_straight_points
    cdef double aero_straight_points
    cdef double slow_delta_equiv
    cdef double med_delta_equiv
    cdef double high_delta_equiv
    cdef double straight_delta_equiv
    cdef double team_pace_delta_equiv
    cdef double braking_delta_equiv
    cdef double acceleration_delta_equiv

    if low_kmh_per_point > 1e-12:
        slow_points += setup_slow_kmh / low_kmh_per_point
    if med_kmh_per_point > 1e-12:
        med_points += setup_med_kmh / med_kmh_per_point
    if high_kmh_per_point > 1e-12:
        high_points += setup_high_kmh / high_kmh_per_point
    if straight_kmh_per_point > 1e-12:
        straight_points += setup_straight_kmh / straight_kmh_per_point

    if low_kmh_per_point > 1e-12:
        slow_points += pressure_slow_kmh / low_kmh_per_point
    if med_kmh_per_point > 1e-12:
        med_points += pressure_med_kmh / med_kmh_per_point
    if high_kmh_per_point > 1e-12:
        high_points += pressure_high_kmh / high_kmh_per_point
    if straight_kmh_per_point > 1e-12:
        straight_points += pressure_straight_kmh / straight_kmh_per_point

    if low_kmh_per_point > 1e-12:
        slow_points += suspension_slow_kmh / low_kmh_per_point
    if med_kmh_per_point > 1e-12:
        med_points += suspension_med_kmh / med_kmh_per_point
    if high_kmh_per_point > 1e-12:
        high_points += suspension_high_kmh / high_kmh_per_point
    if straight_kmh_per_point > 1e-12:
        straight_points += suspension_straight_kmh / straight_kmh_per_point

    effective_corner_bonus = (
        (driver_cornering_kmh_per_point * driver_cornering)
        + weekend_cornering_variability_kmh
        - corner_nerf
        - part_corner_nerf
        - aero_corner_nerf
        + aero_corner_trait_delta
    )
    effective_brake_bonus = (
        (driver_braking_ms2_per_point * driver_braking * aero_brake_trait_mult)
        - brake_nerf
        - aero_brake_nerf
    )
    aero_straight_points = -aero_straight_delta_s / delta_seconds_per_rating
    effective_straight_points = straight_points + aero_straight_points
    acceleration_points += pressure_accel_ms2 / accel_ms2_per_rating

    slow_delta_equiv = -slow_points * delta_seconds_per_rating
    med_delta_equiv = -med_points * delta_seconds_per_rating
    high_delta_equiv = -high_points * delta_seconds_per_rating
    straight_delta_equiv = -effective_straight_points * delta_seconds_per_rating
    team_pace_delta_equiv = -team_pace_points * delta_seconds_per_rating
    braking_delta_equiv = -braking_points * delta_seconds_per_rating
    acceleration_delta_equiv = -acceleration_points * delta_seconds_per_rating

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


cpdef list build_zone_targets_kmh(
    object micro_sectors_obj,
    double slow_rating_points,
    double med_rating_points,
    double high_rating_points,
    double team_pace_rating_points,
    double driver_cornering_kmh_bonus,
    double tyre_lat_mult,
    double car_mass_kg,
    double engine_mass_kg,
    double fuel_mass_kg,
    double mass_ref_kg,
):
    cdef Py_ssize_t n, i, j
    cdef list out = []
    cdef list steady = []
    cdef object sector, zone_type
    cdef int zone_idx
    cdef double base_target, target
    cdef double attr_points, low_bound, high_bound
    cdef double total_mass, ref_mass, raw_mass, corner_mass_mult
    cdef double lat_mult
    cdef double prev_target, next_target

    cdef double attr_kmh_per_rating[4]
    cdef double universal_kmh_per_rating[4]
    cdef double default_target_kmh[4]
    cdef double speed_low[4]
    cdef double speed_high[4]

    attr_kmh_per_rating[0] = 0.084375
    attr_kmh_per_rating[1] = 0.09375
    attr_kmh_per_rating[2] = 0.15
    attr_kmh_per_rating[3] = 0.0
    universal_kmh_per_rating[0] = 0.045
    universal_kmh_per_rating[1] = 0.055
    universal_kmh_per_rating[2] = 0.07
    universal_kmh_per_rating[3] = 0.18
    default_target_kmh[0] = 115.0
    default_target_kmh[1] = 175.0
    default_target_kmh[2] = 225.0
    default_target_kmh[3] = 300.0
    speed_low[0] = 40.0
    speed_low[1] = 70.0
    speed_low[2] = 95.0
    speed_low[3] = 120.0
    speed_high[0] = 220.0
    speed_high[1] = 280.0
    speed_high[2] = 340.0
    speed_high[3] = 420.0

    try:
        n = len(micro_sectors_obj)
    except Exception:
        return out
    if n <= 0:
        return out

    lat_mult = _clamp_double(tyre_lat_mult, 0.30, 1.40)
    ref_mass = mass_ref_kg
    if ref_mass < 300.0:
        ref_mass = 300.0
    total_mass = car_mass_kg + engine_mass_kg + (fuel_mass_kg if fuel_mass_kg > 0.0 else 0.0)
    if total_mass < 300.0:
        total_mass = 300.0
    raw_mass = sqrt(ref_mass / total_mass)
    corner_mass_mult = _clamp_double(1.0 + (raw_mass - 1.0) * 0.50, 0.84, 1.08)

    for i in range(n):
        try:
            sector = micro_sectors_obj[i]
            zone_type = sector.get("type", "low")
        except Exception:
            steady.append(None)
            continue
        zone_idx = _zone_index(zone_type)
        if zone_idx < 0 or zone_idx > 3:
            steady.append(None)
            continue
        if zone_idx == 0:
            attr_points = slow_rating_points
        elif zone_idx == 1:
            attr_points = med_rating_points
        elif zone_idx == 2:
            attr_points = high_rating_points
        else:
            attr_points = 0.0
        try:
            base_target = float(sector.get("target_speed_kmh", default_target_kmh[zone_idx]))
        except Exception:
            base_target = default_target_kmh[zone_idx]
        target = base_target
        target += attr_kmh_per_rating[zone_idx] * attr_points
        if zone_idx != 3:
            target += universal_kmh_per_rating[zone_idx] * team_pace_rating_points
        if zone_idx <= 2:
            target += driver_cornering_kmh_bonus
            target *= corner_mass_mult
            target *= lat_mult
        low_bound = speed_low[zone_idx]
        high_bound = speed_high[zone_idx]
        target = _clamp_double(target, low_bound, high_bound)
        if zone_idx <= 3:
            steady.append(float(target))
        else:
            steady.append(None)

    for i in range(n):
        try:
            sector = micro_sectors_obj[i]
            zone_type = sector.get("type", "low")
        except Exception:
            out.append(200.0)
            continue
        zone_idx = _zone_index(zone_type)
        target = -1.0
        if zone_idx >= 0 and zone_idx <= 3:
            try:
                target = float(steady[i])
            except Exception:
                target = -1.0
        if target < 0.0:
            prev_target = -1.0
            next_target = -1.0
            for j in range(i + 1, n):
                if steady[j] is not None:
                    next_target = steady[j]
                    break
            for j in range(i - 1, -1, -1):
                if steady[j] is not None:
                    prev_target = steady[j]
                    break
            if next_target >= 0.0 and prev_target >= 0.0:
                target = prev_target if prev_target > next_target else next_target
            elif next_target >= 0.0:
                target = next_target
            elif prev_target >= 0.0:
                target = prev_target
            elif zone_idx >= 0 and zone_idx <= 3:
                target = default_target_kmh[zone_idx]
            else:
                target = 200.0
        if target < 5.0:
            target = 5.0
        out.append(float(target))
    return out


cpdef tuple prepare_lap_time_inputs(object model, list inputs_batch, object constants):
    """Same Formula input calculations, one compiled pass over the batch.

    Fixed acceleration/braking/power/drag data is prepared once per distinct
    car in this batch. Targets use the existing compiled target builder.
    """
    cdef dict fixed = {}
    cdef object inputs, key, car
    cdef list targets = [], accelerations = [], brakes = [], scaled = [], drags = []
    cdef double accel_base = constants[0], accel_step = constants[1]
    cdef double accel_min = constants[2], accel_max = constants[3]
    cdef double brake_base = constants[4], brake_step = constants[5]
    cdef double brake_min = constants[6], brake_max = constants[7]
    cdef double mass_sensitivity = constants[8]
    cdef double longitudinal, acceleration, braking, power, drag, ref, total
    cdef double raw, mass_mult, acceleration_scaled, fuel
    for inputs in inputs_batch:
        key = (inputs.acceleration_rating_points, inputs.braking_rating_points,
               inputs.driver_braking_ms2_bonus, inputs.engine_power_rating,
               inputs.straight_rating_points, inputs.car_mass_kg,
               inputs.engine_mass_kg, inputs.mass_ref_kg)
        car = fixed.get(key)
        if car is None:
            acceleration = accel_base + float(key[0])*accel_step
            braking = brake_base + float(key[1])*brake_step
            braking = braking + float(key[2])
            car = (acceleration, braking, model._engine_power_multiplier(key[3]),
                   model._drag_coeff(key[4]), float(key[5])+float(key[6]),
                   model._mass_reference_kg(inputs))
            fixed[key] = car
        longitudinal = _clamp_double(float(inputs.tyre_long_mult),.25,1.50)
        acceleration = _clamp_double(float(car[0])*longitudinal,accel_min,accel_max)
        braking = _clamp_double(float(car[1])*longitudinal,brake_min,brake_max)
        fuel = max(0.,float(inputs.fuel_mass_kg))
        total = max(300.,float(car[4])+fuel)
        raw = float(car[5])/total
        mass_mult = _clamp_double(1.+(raw-1.)*mass_sensitivity,.70,1.25)
        power = car[2]
        acceleration_scaled = acceleration*power
        acceleration_scaled = acceleration_scaled*mass_mult
        targets.append(build_zone_targets_kmh(model.micro_sectors,
            float(inputs.slow_rating_points),float(inputs.med_rating_points),
            float(inputs.high_rating_points),float(inputs.team_pace_rating_points),
            float(inputs.driver_cornering_kmh_bonus),float(inputs.tyre_lat_mult),
            float(inputs.car_mass_kg),float(inputs.engine_mass_kg),
            float(inputs.fuel_mass_kg),float(inputs.mass_ref_kg)))
        accelerations.append(acceleration)
        brakes.append(braking)
        scaled.append(acceleration_scaled)
        drags.append(car[3])
    return targets,accelerations,brakes,scaled,drags


cdef inline double _segment_motion_duration_only(
    double distance_m,
    double v_entry_ms,
    double v_exit_ms,
    double v_cap_ms,
    double accel_ms2,
    double brake_ms2,
    double drag_coeff,
    double profile_solver_step_m,
):
    """Return the exact segment duration without allocating profile samples."""
    cdef double distance, brake, v, v_exit, v_cap
    cdef double x, t, step, remaining, brake_needed, accel
    cdef double v2, v_next, avg_v, dt

    distance = distance_m
    if distance < 0.0:
        distance = 0.0
    brake = brake_ms2
    if brake < 0.05:
        brake = 0.05
    v = v_entry_ms
    if v < 0.5:
        v = 0.5
    v_exit = v_exit_ms
    if v_exit < 0.5:
        v_exit = 0.5
    v_cap = v_cap_ms
    if v_cap < 0.5:
        v_cap = 0.5

    if distance <= 1e-9:
        return 0.001
    if profile_solver_step_m <= 1e-9:
        profile_solver_step_m = 1.0

    x = 0.0
    t = 0.0
    while x < distance - 1e-9:
        step = profile_solver_step_m
        remaining = distance - x
        if step > remaining:
            step = remaining
        brake_needed = (v * v - v_exit * v_exit) / (2.0 * brake)
        if brake_needed < 0.0:
            brake_needed = 0.0

        if brake_needed >= max(0.0, remaining - step * 0.5):
            accel = -brake
        elif v >= v_cap - 1e-9:
            accel = 0.0
        else:
            accel = _effective_accel_rate(accel_ms2, v, drag_coeff)

        v2 = v * v + 2.0 * accel * step
        if v2 < 0.0:
            v2 = 0.0
        v_next = sqrt(v2)
        if accel >= 0.0 and v_next > v_cap:
            v_next = v_cap
        avg_v = 0.5 * (v + v_next)
        if avg_v < 0.1:
            avg_v = 0.1
        dt = step / avg_v

        x += step
        t += dt
        v = v_next
        if v < 0.5:
            v = 0.5

    if t < 0.001:
        t = 0.001
    return t


cpdef dict build_profile_pipeline(
    object micro_sectors_obj,
    object zone_targets_obj,
    double track_length_m,
    object sector_splits_obj,
    double accel_rate,
    double braking_rate,
    double accel_base_scaled,
    double drag_coeff,
    double forward_solver_step_m=1.0,
    int solve_passes=12,
    double profile_solver_step_m=1.0,
):
    cdef list macro_bounds = [0.0] + list(sector_splits_obj) + [1.0]
    cdef list segment_defs = []
    cdef list zone_targets = list(zone_targets_obj)
    cdef int n_targets = len(zone_targets)
    cdef int idx, n, i, rot_i, orig_i, macro_idx
    cdef object sector, zone_type
    cdef double length_pct, length_frac, distance_m
    cdef double segment_start, segment_end, segment_progress
    cdef double progress_cursor, cap, seg_start, seg_end, seg_progress
    cdef double target_speed_kmh
    cdef double lap_time, segment_time, seg_time_start, seg_time_end
    cdef double b0, b1, overlap, macro_total, scale, even
    cdef list caps_ms, distances_m, order, caps_rot, dist_rot
    cdef list node_speeds, entry_ms, exit_ms, macro_times, segments
    cdef object segdef, seg_motion
    cdef double v_in, v_out_max, v_out, v_in_max

    progress_cursor = 0.0
    try:
        n = len(micro_sectors_obj)
    except Exception:
        n = 0
    for idx in range(n):
        try:
            sector = micro_sectors_obj[idx]
        except Exception:
            continue
        try:
            length_pct = float(sector.get("length_pct", 0.0))
        except Exception:
            continue
        if length_pct < 0.0:
            length_pct = 0.0
        if length_pct <= 1e-9:
            continue
        length_frac = length_pct / 100.0
        distance_m = track_length_m * length_frac
        zone_type = str(sector.get("type", "low"))
        segment_start = progress_cursor
        segment_end = segment_start + length_frac
        if segment_end > 1.0:
            segment_end = 1.0
        segment_progress = segment_end - segment_start
        if segment_progress < 1e-9:
            segment_progress = 1e-9
        if idx < n_targets:
            try:
                target_speed_kmh = float(zone_targets[idx])
            except Exception:
                target_speed_kmh = 200.0
        else:
            target_speed_kmh = 200.0
        segment_defs.append(
            {
                "start_progress": float(segment_start),
                "end_progress": float(segment_end),
                "progress_len": float(segment_progress),
                "distance_m": float(distance_m),
                "target_speed_kmh": float(target_speed_kmh),
                "zone_type": str(zone_type),
            }
        )
        progress_cursor = segment_end

    if not segment_defs:
        return {
            "lap_time": 0.001,
            "macro_times": [0.001 for _ in range(max(1, len(macro_bounds) - 1))],
            "segments": [],
        }

    n = len(segment_defs)
    caps_ms = []
    distances_m = []
    for segdef in segment_defs:
        try:
            cap = float(segdef.get("target_speed_kmh", 200.0)) / 3.6
        except Exception:
            cap = 200.0 / 3.6
        if cap < 0.5:
            cap = 0.5
        caps_ms.append(cap)
        try:
            distance_m = float(segdef.get("distance_m", 0.0))
        except Exception:
            distance_m = 0.0
        if distance_m < 0.0:
            distance_m = 0.0
        distances_m.append(distance_m)

    i = 0
    cap = float(caps_ms[0])
    for idx in range(1, n):
        if float(caps_ms[idx]) < cap:
            cap = float(caps_ms[idx])
            i = idx
    order = list(range(i, n)) + list(range(0, i))
    caps_rot = [float(caps_ms[k]) for k in order]
    dist_rot = [float(distances_m[k]) for k in order]

    node_speeds = solve_node_speeds(
        caps_rot,
        dist_rot,
        float(accel_base_scaled),
        float(braking_rate),
        float(drag_coeff),
        float(forward_solver_step_m),
        int(solve_passes),
    )
    if not isinstance(node_speeds, list) or len(node_speeds) != (n + 1):
        if solve_passes <= 0:
            solve_passes = 1
        node_speeds = [float(caps_rot[0]) for _ in range(n + 1)]
        for _ in range(solve_passes):
            if float(node_speeds[0]) > float(caps_rot[0]):
                node_speeds[0] = float(caps_rot[0])
            for i in range(n):
                v_in = float(node_speeds[i])
                if v_in > float(caps_rot[i]):
                    v_in = float(caps_rot[i])
                v_out_max = forward_exit_speed(
                    float(v_in),
                    float(dist_rot[i]),
                    float(caps_rot[i]),
                    float(accel_base_scaled),
                    float(drag_coeff),
                    float(forward_solver_step_m),
                )
                node_speeds[i + 1] = min(float(caps_rot[i]), float(v_out_max))
            node_speeds[n] = min(float(node_speeds[n]), float(node_speeds[0]), float(caps_rot[0]))
            for i in range(n - 1, -1, -1):
                v_out = min(float(node_speeds[i + 1]), float(caps_rot[i]))
                v_in_max = sqrt(max((v_out * v_out) + (2.0 * float(braking_rate) * float(dist_rot[i])), 0.0))
                node_speeds[i] = min(float(node_speeds[i]), float(caps_rot[i]), float(v_in_max))
            node_speeds[n] = min(float(node_speeds[n]), float(node_speeds[0]))

    entry_ms = [0.0 for _ in range(n)]
    exit_ms = [0.0 for _ in range(n)]
    for rot_i, orig_i in enumerate(order):
        entry_ms[orig_i] = float(node_speeds[rot_i])
        exit_ms[orig_i] = float(node_speeds[rot_i + 1])

    lap_time = 0.0
    macro_times = [0.0 for _ in range(max(1, len(macro_bounds) - 1))]
    segments = []
    for idx, segdef in enumerate(segment_defs):
        seg_motion = segment_motion_profile(
            float(segdef.get("distance_m", 0.0)),
            float(entry_ms[idx]),
            float(exit_ms[idx]),
            float(caps_ms[idx]),
            float(accel_base_scaled),
            float(braking_rate),
            float(drag_coeff),
            float(profile_solver_step_m),
        )
        if not isinstance(seg_motion, dict):
            continue
        segment_time = float(seg_motion.get("duration", 0.001))
        seg_start = float(segdef.get("start_progress", 0.0))
        seg_end = float(segdef.get("end_progress", seg_start))
        seg_progress = float(segdef.get("progress_len", seg_end - seg_start))
        if seg_progress < 1e-9:
            seg_progress = 1e-9
        seg_time_start = lap_time
        seg_time_end = seg_time_start + segment_time

        for macro_idx in range(len(macro_times)):
            b0 = float(macro_bounds[macro_idx])
            b1 = float(macro_bounds[macro_idx + 1])
            overlap = min(seg_end, b1) - max(seg_start, b0)
            if overlap <= 1e-9:
                continue
            macro_times[macro_idx] += segment_time * (overlap / seg_progress)

        segments.append(
            {
                "start_progress": seg_start,
                "end_progress": seg_end,
                "progress_len": seg_progress,
                "start_time": float(seg_time_start),
                "end_time": float(seg_time_end),
                "duration": float(segment_time),
                "distance_m": float(segdef.get("distance_m", 0.0)),
                "start_speed_kmh": float(entry_ms[idx] * 3.6),
                "end_speed_kmh": float(exit_ms[idx] * 3.6),
                "target_speed_kmh": float(segdef.get("target_speed_kmh", 200.0)),
                "zone_type": str(segdef.get("zone_type", "steady")),
                "v_entry_ms": float(entry_ms[idx]),
                "v_exit_ms": float(exit_ms[idx]),
                "v_peak_ms": float(seg_motion.get("v_peak_ms", max(entry_ms[idx], exit_ms[idx]))),
                "dist_acc_m": float(seg_motion.get("dist_acc_m", 0.0)),
                "dist_cruise_m": float(seg_motion.get("dist_cruise_m", 0.0)),
                "dist_dec_m": float(seg_motion.get("dist_dec_m", 0.0)),
                "time_acc_s": float(seg_motion.get("time_acc_s", 0.0)),
                "time_cruise_s": float(seg_motion.get("time_cruise_s", 0.0)),
                "time_dec_s": float(seg_motion.get("time_dec_s", 0.0)),
                "sample_dist_m": seg_motion.get("sample_dist_m", []) or [],
                "sample_time_s": seg_motion.get("sample_time_s", []) or [],
                "sample_speed_ms": seg_motion.get("sample_speed_ms", []) or [],
                "accel_rate_ms2": float(accel_rate),
                "brake_rate_ms2": float(braking_rate),
            }
        )
        lap_time = seg_time_end

    if lap_time <= 0.0:
        lap_time = 0.001
    macro_total = sum(macro_times)
    if macro_total > 0.0:
        scale = lap_time / macro_total
        macro_times = [max(0.001, float(t) * scale) for t in macro_times]
    elif macro_times:
        even = lap_time / float(len(macro_times))
        macro_times = [max(0.001, even) for _ in macro_times]

    return {
        "lap_time": float(sum(macro_times)),
        "macro_times": list(macro_times),
        "segments": segments,
        "segment_end_progress": [float(seg.get("end_progress", 1.0)) for seg in segments],
    }


cpdef object build_fast_profile_pipeline(
    object micro_sectors_obj,
    object zone_targets_obj,
    double track_length_m,
    object sector_splits_obj,
    double accel_rate,
    double braking_rate,
    double accel_base_scaled,
    double drag_coeff,
    double forward_solver_step_m=1.0,
    int solve_passes=12,
    double profile_solver_step_m=1.0,
):
    cdef list macro_bounds = [0.0] + list(sector_splits_obj) + [1.0]
    cdef list zone_targets = list(zone_targets_obj)
    cdef int n_targets = len(zone_targets)
    cdef int idx, n, i, rot_i, orig_i, macro_idx
    cdef object sector, zone_type, seg_motion
    cdef double length_pct, length_frac, distance_m
    cdef double segment_start, segment_end, segment_progress
    cdef double progress_cursor, cap, seg_start, seg_end, seg_progress
    cdef double target_speed_kmh
    cdef double lap_time, segment_time, seg_time_start, seg_time_end
    cdef double b0, b1, overlap, macro_total, scale, even
    cdef double v_in, v_out_max, v_out, v_in_max
    cdef list starts, ends, prog_lens, distances, targets
    cdef list caps_ms, distances_m, order, caps_rot, dist_rot
    cdef list node_speeds, entry_ms, exit_ms, macro_times
    cdef list start_times, durations, v_entries, v_exits, v_peaks
    cdef list dist_accs, dist_cruises, time_accs, time_cruises
    cdef list accel_rates, brake_rates, sample_dist_arrs, sample_time_arrs, sample_speed_arrs
    cdef FastProfile fp
    cdef object arr_obj

    starts = []
    ends = []
    prog_lens = []
    distances = []
    targets = []
    progress_cursor = 0.0
    try:
        n = len(micro_sectors_obj)
    except Exception:
        n = 0
    for idx in range(n):
        try:
            sector = micro_sectors_obj[idx]
        except Exception:
            continue
        try:
            length_pct = float(sector.get("length_pct", 0.0))
        except Exception:
            continue
        if length_pct < 0.0:
            length_pct = 0.0
        if length_pct <= 1e-9:
            continue
        length_frac = length_pct / 100.0
        distance_m = track_length_m * length_frac
        segment_start = progress_cursor
        segment_end = segment_start + length_frac
        if segment_end > 1.0:
            segment_end = 1.0
        segment_progress = segment_end - segment_start
        if segment_progress < 1e-9:
            segment_progress = 1e-9
        if idx < n_targets:
            try:
                target_speed_kmh = float(zone_targets[idx])
            except Exception:
                target_speed_kmh = 200.0
        else:
            target_speed_kmh = 200.0
        starts.append(float(segment_start))
        ends.append(float(segment_end))
        prog_lens.append(float(segment_progress))
        distances.append(float(distance_m))
        targets.append(float(target_speed_kmh))
        progress_cursor = segment_end

    if not starts:
        fp = FastProfile()
        fp.lap_time = 0.001
        fp.macro_times = [0.001 for _ in range(max(1, len(macro_bounds) - 1))]
        fp.segment_end_progress = []
        fp.count = 0
        return fp

    n = len(starts)
    caps_ms = []
    distances_m = []
    for idx in range(n):
        cap = float(targets[idx]) / 3.6
        if cap < 0.5:
            cap = 0.5
        caps_ms.append(cap)
        distance_m = float(distances[idx])
        if distance_m < 0.0:
            distance_m = 0.0
        distances_m.append(distance_m)

    i = 0
    cap = float(caps_ms[0])
    for idx in range(1, n):
        if float(caps_ms[idx]) < cap:
            cap = float(caps_ms[idx])
            i = idx
    order = list(range(i, n)) + list(range(0, i))
    caps_rot = [float(caps_ms[k]) for k in order]
    dist_rot = [float(distances_m[k]) for k in order]

    node_speeds = solve_node_speeds(
        caps_rot,
        dist_rot,
        float(accel_base_scaled),
        float(braking_rate),
        float(drag_coeff),
        float(forward_solver_step_m),
        int(solve_passes),
    )
    if not isinstance(node_speeds, list) or len(node_speeds) != (n + 1):
        if solve_passes <= 0:
            solve_passes = 1
        node_speeds = [float(caps_rot[0]) for _ in range(n + 1)]
        for _ in range(solve_passes):
            if float(node_speeds[0]) > float(caps_rot[0]):
                node_speeds[0] = float(caps_rot[0])
            for i in range(n):
                v_in = float(node_speeds[i])
                if v_in > float(caps_rot[i]):
                    v_in = float(caps_rot[i])
                v_out_max = forward_exit_speed(
                    float(v_in),
                    float(dist_rot[i]),
                    float(caps_rot[i]),
                    float(accel_base_scaled),
                    float(drag_coeff),
                    float(forward_solver_step_m),
                )
                node_speeds[i + 1] = min(float(caps_rot[i]), float(v_out_max))
            node_speeds[n] = min(float(node_speeds[n]), float(node_speeds[0]), float(caps_rot[0]))
            for i in range(n - 1, -1, -1):
                v_out = min(float(node_speeds[i + 1]), float(caps_rot[i]))
                v_in_max = sqrt(max((v_out * v_out) + (2.0 * float(braking_rate) * float(dist_rot[i])), 0.0))
                node_speeds[i] = min(float(node_speeds[i]), float(caps_rot[i]), float(v_in_max))
            node_speeds[n] = min(float(node_speeds[n]), float(node_speeds[0]))

    entry_ms = [0.0 for _ in range(n)]
    exit_ms = [0.0 for _ in range(n)]
    for rot_i, orig_i in enumerate(order):
        entry_ms[orig_i] = float(node_speeds[rot_i])
        exit_ms[orig_i] = float(node_speeds[rot_i + 1])

    lap_time = 0.0
    macro_times = [0.0 for _ in range(max(1, len(macro_bounds) - 1))]
    start_times = []
    durations = []
    v_entries = []
    v_exits = []
    v_peaks = []
    dist_accs = []
    dist_cruises = []
    time_accs = []
    time_cruises = []
    accel_rates = []
    brake_rates = []
    sample_dist_arrs = []
    sample_time_arrs = []
    sample_speed_arrs = []

    for idx in range(n):
        seg_motion = segment_motion_profile_tuple(
            float(distances[idx]),
            float(entry_ms[idx]),
            float(exit_ms[idx]),
            float(caps_ms[idx]),
            float(accel_base_scaled),
            float(braking_rate),
            float(drag_coeff),
            float(profile_solver_step_m),
        )
        if not isinstance(seg_motion, tuple) or len(seg_motion) < 9:
            continue
        segment_time = float(seg_motion[0])
        seg_start = float(starts[idx])
        seg_end = float(ends[idx])
        seg_progress = float(prog_lens[idx])
        if seg_progress < 1e-9:
            seg_progress = 1e-9
        seg_time_start = lap_time
        seg_time_end = seg_time_start + segment_time

        for macro_idx in range(len(macro_times)):
            b0 = float(macro_bounds[macro_idx])
            b1 = float(macro_bounds[macro_idx + 1])
            overlap = min(seg_end, b1) - max(seg_start, b0)
            if overlap <= 1e-9:
                continue
            macro_times[macro_idx] += segment_time * (overlap / seg_progress)

        start_times.append(float(seg_time_start))
        durations.append(float(segment_time))
        v_entries.append(float(entry_ms[idx]))
        v_exits.append(float(exit_ms[idx]))
        v_peaks.append(float(seg_motion[1]))
        dist_accs.append(float(seg_motion[2]))
        dist_cruises.append(float(seg_motion[3]))
        time_accs.append(float(seg_motion[4]))
        time_cruises.append(float(seg_motion[5]))
        accel_rates.append(float(accel_rate))
        brake_rates.append(float(braking_rate))
        sample_dist_arrs.append(seg_motion[6])
        sample_time_arrs.append(seg_motion[7])
        sample_speed_arrs.append(seg_motion[8])
        lap_time = seg_time_end

    if lap_time <= 0.0:
        lap_time = 0.001
    macro_total = sum(macro_times)
    if macro_total > 0.0:
        scale = lap_time / macro_total
        macro_times = [max(0.001, float(t) * scale) for t in macro_times]
    elif macro_times:
        even = lap_time / float(len(macro_times))
        macro_times = [max(0.001, even) for _ in macro_times]

    fp = FastProfile()
    fp.lap_time = float(sum(macro_times))
    fp.macro_times = list(macro_times)
    fp.segment_end_progress = [float(v) for v in ends]
    fp.count = len(start_times)
    fp.start_progress_arr = _as_double_array(starts)
    fp.end_progress_arr = _as_double_array(ends)
    fp.progress_len_arr = _as_double_array(prog_lens)
    fp.start_time_arr = _as_double_array(start_times)
    fp.duration_arr = _as_double_array(durations)
    fp.distance_arr = _as_double_array(distances[: fp.count])
    fp.v_entry_arr = _as_double_array(v_entries)
    fp.v_exit_arr = _as_double_array(v_exits)
    fp.v_peak_arr = _as_double_array(v_peaks)
    fp.dist_acc_arr = _as_double_array(dist_accs)
    fp.dist_cruise_arr = _as_double_array(dist_cruises)
    fp.time_acc_arr = _as_double_array(time_accs)
    fp.time_cruise_arr = _as_double_array(time_cruises)
    fp.accel_rate_arr = _as_double_array(accel_rates)
    fp.brake_rate_arr = _as_double_array(brake_rates)
    fp.sample_dist_arrs = sample_dist_arrs
    fp.sample_time_arrs = sample_time_arrs
    fp.sample_speed_arrs = sample_speed_arrs
    fp.start_progress = fp.start_progress_arr
    fp.end_progress = fp.end_progress_arr
    fp.progress_len = fp.progress_len_arr
    fp.start_time = fp.start_time_arr
    fp.duration = fp.duration_arr
    fp.distance = fp.distance_arr
    fp.v_entry = fp.v_entry_arr
    fp.v_exit = fp.v_exit_arr
    fp.v_peak = fp.v_peak_arr
    fp.dist_acc = fp.dist_acc_arr
    fp.dist_cruise = fp.dist_cruise_arr
    fp.time_acc = fp.time_acc_arr
    fp.time_cruise = fp.time_cruise_arr
    fp.accel_rate = fp.accel_rate_arr
    fp.brake_rate = fp.brake_rate_arr
    return fp


cpdef double build_lap_time_only_pipeline(
    object micro_sectors_obj,
    object zone_targets_obj,
    double track_length_m,
    object sector_splits_obj,
    double accel_rate,
    double braking_rate,
    double accel_base_scaled,
    double drag_coeff,
    double forward_solver_step_m=1.0,
    int solve_passes=12,
    double profile_solver_step_m=1.0,
):
    """Run the fast profile solver but retain only its final lap time."""
    cdef list macro_bounds = [0.0] + list(sector_splits_obj) + [1.0]
    cdef list zone_targets = list(zone_targets_obj)
    cdef int n_targets = len(zone_targets)
    cdef int idx, n, i, rot_i, orig_i, macro_idx
    cdef object sector
    cdef double length_pct, length_frac, distance_m
    cdef double segment_start, segment_end, segment_progress
    cdef double progress_cursor, cap, seg_start, seg_end, seg_progress
    cdef double target_speed_kmh
    cdef double lap_time, segment_time
    cdef double b0, b1, overlap, macro_total, scale, even
    cdef double v_in, v_out_max, v_out, v_in_max
    cdef list starts, ends, prog_lens, distances, targets
    cdef list caps_ms, distances_m, order, caps_rot, dist_rot
    cdef list node_speeds, entry_ms, exit_ms, macro_times

    starts = []
    ends = []
    prog_lens = []
    distances = []
    targets = []
    progress_cursor = 0.0
    try:
        n = len(micro_sectors_obj)
    except Exception:
        n = 0
    for idx in range(n):
        try:
            sector = micro_sectors_obj[idx]
        except Exception:
            continue
        try:
            length_pct = float(sector.get("length_pct", 0.0))
        except Exception:
            continue
        if length_pct < 0.0:
            length_pct = 0.0
        if length_pct <= 1e-9:
            continue
        length_frac = length_pct / 100.0
        distance_m = track_length_m * length_frac
        segment_start = progress_cursor
        segment_end = segment_start + length_frac
        if segment_end > 1.0:
            segment_end = 1.0
        segment_progress = segment_end - segment_start
        if segment_progress < 1e-9:
            segment_progress = 1e-9
        if idx < n_targets:
            try:
                target_speed_kmh = float(zone_targets[idx])
            except Exception:
                target_speed_kmh = 200.0
        else:
            target_speed_kmh = 200.0
        starts.append(float(segment_start))
        ends.append(float(segment_end))
        prog_lens.append(float(segment_progress))
        distances.append(float(distance_m))
        targets.append(float(target_speed_kmh))
        progress_cursor = segment_end

    if not starts:
        return float(
            sum(
                [0.001 for _ in range(max(1, len(macro_bounds) - 1))]
            )
        )

    n = len(starts)
    caps_ms = []
    distances_m = []
    for idx in range(n):
        cap = float(targets[idx]) / 3.6
        if cap < 0.5:
            cap = 0.5
        caps_ms.append(cap)
        distance_m = float(distances[idx])
        if distance_m < 0.0:
            distance_m = 0.0
        distances_m.append(distance_m)

    i = 0
    cap = float(caps_ms[0])
    for idx in range(1, n):
        if float(caps_ms[idx]) < cap:
            cap = float(caps_ms[idx])
            i = idx
    order = list(range(i, n)) + list(range(0, i))
    caps_rot = [float(caps_ms[k]) for k in order]
    dist_rot = [float(distances_m[k]) for k in order]

    node_speeds = solve_node_speeds(
        caps_rot,
        dist_rot,
        float(accel_base_scaled),
        float(braking_rate),
        float(drag_coeff),
        float(forward_solver_step_m),
        int(solve_passes),
    )
    if not isinstance(node_speeds, list) or len(node_speeds) != (n + 1):
        if solve_passes <= 0:
            solve_passes = 1
        node_speeds = [float(caps_rot[0]) for _ in range(n + 1)]
        for _ in range(solve_passes):
            if float(node_speeds[0]) > float(caps_rot[0]):
                node_speeds[0] = float(caps_rot[0])
            for i in range(n):
                v_in = float(node_speeds[i])
                if v_in > float(caps_rot[i]):
                    v_in = float(caps_rot[i])
                v_out_max = forward_exit_speed(
                    float(v_in),
                    float(dist_rot[i]),
                    float(caps_rot[i]),
                    float(accel_base_scaled),
                    float(drag_coeff),
                    float(forward_solver_step_m),
                )
                node_speeds[i + 1] = min(float(caps_rot[i]), float(v_out_max))
            node_speeds[n] = min(float(node_speeds[n]), float(node_speeds[0]), float(caps_rot[0]))
            for i in range(n - 1, -1, -1):
                v_out = min(float(node_speeds[i + 1]), float(caps_rot[i]))
                v_in_max = sqrt(max((v_out * v_out) + (2.0 * float(braking_rate) * float(dist_rot[i])), 0.0))
                node_speeds[i] = min(float(node_speeds[i]), float(caps_rot[i]), float(v_in_max))
            node_speeds[n] = min(float(node_speeds[n]), float(node_speeds[0]))

    entry_ms = [0.0 for _ in range(n)]
    exit_ms = [0.0 for _ in range(n)]
    for rot_i, orig_i in enumerate(order):
        entry_ms[orig_i] = float(node_speeds[rot_i])
        exit_ms[orig_i] = float(node_speeds[rot_i + 1])

    lap_time = 0.0
    macro_times = [0.0 for _ in range(max(1, len(macro_bounds) - 1))]
    for idx in range(n):
        segment_time = _segment_motion_duration_only(
            float(distances[idx]),
            float(entry_ms[idx]),
            float(exit_ms[idx]),
            float(caps_ms[idx]),
            float(accel_base_scaled),
            float(braking_rate),
            float(drag_coeff),
            float(profile_solver_step_m),
        )
        seg_start = float(starts[idx])
        seg_end = float(ends[idx])
        seg_progress = float(prog_lens[idx])
        if seg_progress < 1e-9:
            seg_progress = 1e-9
        lap_time += segment_time

        for macro_idx in range(len(macro_times)):
            b0 = float(macro_bounds[macro_idx])
            b1 = float(macro_bounds[macro_idx + 1])
            overlap = min(seg_end, b1) - max(seg_start, b0)
            if overlap <= 1e-9:
                continue
            macro_times[macro_idx] += segment_time * (overlap / seg_progress)

    if lap_time <= 0.0:
        lap_time = 0.001
    macro_total = sum(macro_times)
    if macro_total > 0.0:
        scale = lap_time / macro_total
        macro_times = [max(0.001, float(t) * scale) for t in macro_times]
    elif macro_times:
        even = lap_time / float(len(macro_times))
        macro_times = [max(0.001, even) for _ in macro_times]
    return float(sum(macro_times))


cpdef list build_lap_time_only_batch_pipeline(
    object micro_sectors_obj,
    object zone_targets_batch_obj,
    double track_length_m,
    object sector_splits_obj,
    object accel_rates_obj,
    object braking_rates_obj,
    object accel_base_scaled_obj,
    object drag_coeffs_obj,
    double forward_solver_step_m=1.0,
    int solve_passes=12,
    double profile_solver_step_m=1.0,
):
    """Evaluate ordered lap-time candidates while parsing track geometry once.

    Every candidate follows the same operation and summation order as
    ``build_lap_time_only_pipeline``.  Only immutable circuit preparation is
    shared between candidates; candidate physics remains fully independent.
    """
    cdef list macro_bounds = [0.0] + list(sector_splits_obj) + [1.0]
    cdef list starts = []
    cdef list ends = []
    cdef list prog_lens = []
    cdef list distances = []
    cdef list zone_targets, targets
    cdef list caps_ms, distances_m, order, caps_rot, dist_rot
    cdef list node_speeds, entry_ms, exit_ms, macro_times
    cdef list results = []
    cdef int candidate_count, candidate_idx
    cdef int idx, n, n_targets, i, rot_i, orig_i, macro_idx
    cdef object sector
    cdef double length_pct, length_frac, distance_m
    cdef double segment_start, segment_end, segment_progress
    cdef double progress_cursor, cap, seg_start, seg_end, seg_progress
    cdef double target_speed_kmh
    cdef double lap_time, segment_time
    cdef double b0, b1, overlap, macro_total, scale, even
    cdef double braking_rate, accel_base_scaled, drag_coeff

    progress_cursor = 0.0
    try:
        n = len(micro_sectors_obj)
    except Exception:
        n = 0
    for idx in range(n):
        try:
            sector = micro_sectors_obj[idx]
        except Exception:
            continue
        try:
            length_pct = float(sector.get("length_pct", 0.0))
        except Exception:
            continue
        if length_pct < 0.0:
            length_pct = 0.0
        if length_pct <= 1e-9:
            continue
        length_frac = length_pct / 100.0
        distance_m = track_length_m * length_frac
        segment_start = progress_cursor
        segment_end = segment_start + length_frac
        if segment_end > 1.0:
            segment_end = 1.0
        segment_progress = segment_end - segment_start
        if segment_progress < 1e-9:
            segment_progress = 1e-9
        starts.append(float(segment_start))
        ends.append(float(segment_end))
        prog_lens.append(float(segment_progress))
        distances.append(float(distance_m))
        progress_cursor = segment_end

    try:
        candidate_count = len(zone_targets_batch_obj)
    except Exception:
        candidate_count = 0
    if not starts:
        even = float(
            sum([0.001 for _ in range(max(1, len(macro_bounds) - 1))])
        )
        return [even for _ in range(candidate_count)]

    n = len(starts)
    distances_m = []
    for idx in range(n):
        distance_m = float(distances[idx])
        if distance_m < 0.0:
            distance_m = 0.0
        distances_m.append(distance_m)

    for candidate_idx in range(candidate_count):
        try:
            zone_targets = list(zone_targets_batch_obj[candidate_idx])
            braking_rate = float(braking_rates_obj[candidate_idx])
            accel_base_scaled = float(accel_base_scaled_obj[candidate_idx])
            drag_coeff = float(drag_coeffs_obj[candidate_idx])
            # Retain parity with the scalar signature.  The duration-only
            # solver consumes the already-scaled acceleration value.
            float(accel_rates_obj[candidate_idx])
        except Exception:
            results.append(0.0)
            continue
        n_targets = len(zone_targets)
        targets = []
        for idx in range(n):
            if idx < n_targets:
                try:
                    target_speed_kmh = float(zone_targets[idx])
                except Exception:
                    target_speed_kmh = 200.0
            else:
                target_speed_kmh = 200.0
            targets.append(float(target_speed_kmh))

        caps_ms = []
        for idx in range(n):
            cap = float(targets[idx]) / 3.6
            if cap < 0.5:
                cap = 0.5
            caps_ms.append(cap)

        i = 0
        cap = float(caps_ms[0])
        for idx in range(1, n):
            if float(caps_ms[idx]) < cap:
                cap = float(caps_ms[idx])
                i = idx
        order = list(range(i, n)) + list(range(0, i))
        caps_rot = [float(caps_ms[k]) for k in order]
        dist_rot = [float(distances_m[k]) for k in order]

        node_speeds = solve_node_speeds(
            caps_rot,
            dist_rot,
            float(accel_base_scaled),
            float(braking_rate),
            float(drag_coeff),
            float(forward_solver_step_m),
            int(solve_passes),
        )
        if not isinstance(node_speeds, list) or len(node_speeds) != (n + 1):
            results.append(0.0)
            continue

        entry_ms = [0.0 for _ in range(n)]
        exit_ms = [0.0 for _ in range(n)]
        for rot_i, orig_i in enumerate(order):
            entry_ms[orig_i] = float(node_speeds[rot_i])
            exit_ms[orig_i] = float(node_speeds[rot_i + 1])

        lap_time = 0.0
        macro_times = [0.0 for _ in range(max(1, len(macro_bounds) - 1))]
        for idx in range(n):
            segment_time = _segment_motion_duration_only(
                float(distances[idx]),
                float(entry_ms[idx]),
                float(exit_ms[idx]),
                float(caps_ms[idx]),
                float(accel_base_scaled),
                float(braking_rate),
                float(drag_coeff),
                float(profile_solver_step_m),
            )
            seg_start = float(starts[idx])
            seg_end = float(ends[idx])
            seg_progress = float(prog_lens[idx])
            if seg_progress < 1e-9:
                seg_progress = 1e-9
            lap_time += segment_time

            for macro_idx in range(len(macro_times)):
                b0 = float(macro_bounds[macro_idx])
                b1 = float(macro_bounds[macro_idx + 1])
                overlap = min(seg_end, b1) - max(seg_start, b0)
                if overlap <= 1e-9:
                    continue
                macro_times[macro_idx] += segment_time * (overlap / seg_progress)

        if lap_time <= 0.0:
            lap_time = 0.001
        macro_total = sum(macro_times)
        if macro_total > 0.0:
            scale = lap_time / macro_total
            macro_times = [max(0.001, float(t) * scale) for t in macro_times]
        elif macro_times:
            even = lap_time / float(len(macro_times))
            macro_times = [max(0.001, even) for _ in macro_times]
        results.append(float(sum(macro_times)))

    return results


cdef inline double _batch_accel_curve_multiplier(double speed_kmh) noexcept nogil:
    cdef double x0, x1, y0, y1, t
    if speed_kmh <= 0.0:
        return 1.75
    elif speed_kmh <= 50.0:
        x0, x1, y0, y1 = 0.0, 50.0, 1.75, 1.60
    elif speed_kmh <= 100.0:
        x0, x1, y0, y1 = 50.0, 100.0, 1.60, 1.45
    elif speed_kmh <= 150.0:
        x0, x1, y0, y1 = 100.0, 150.0, 1.45, 1.25
    elif speed_kmh <= 200.0:
        x0, x1, y0, y1 = 150.0, 200.0, 1.25, 0.95
    elif speed_kmh <= 250.0:
        x0, x1, y0, y1 = 200.0, 250.0, 0.95, 0.75
    elif speed_kmh <= 300.0:
        x0, x1, y0, y1 = 250.0, 300.0, 0.75, 0.40
    elif speed_kmh <= 330.0:
        x0, x1, y0, y1 = 300.0, 330.0, 0.40, 0.20
    elif speed_kmh <= 350.0:
        x0, x1, y0, y1 = 330.0, 350.0, 0.20, 0.05
    else:
        return 0.05
    if x1 - x0 <= 1e-9:
        return y1
    t = (speed_kmh - x0) / (x1 - x0)
    return y0 + (y1 - y0) * t


cdef inline double _batch_effective_accel_rate(
    double accel_base_scaled,
    double speed_ms,
    double drag_coeff,
) noexcept nogil:
    cdef double kmh = speed_ms * 3.6
    cdef double speed = speed_ms
    cdef double base_rate, drag_loss, out
    if kmh < 0.0:
        kmh = 0.0
    if speed < 0.0:
        speed = 0.0
    # Keep the scalar kernel's intermediate operation boundaries.  These
    # assignments are deliberate: combining the expression can move a result
    # by one ULP under an optimizing C compiler.
    base_rate = accel_base_scaled * _batch_accel_curve_multiplier(kmh)
    drag_loss = drag_coeff * speed * speed
    out = base_rate - drag_loss
    if out < 0.05:
        out = 0.05
    return out


cdef inline double _batch_forward_exit_speed(
    double v_entry_ms,
    double distance_m,
    double v_cap_ms,
    double accel_base_scaled,
    double drag_coeff,
    double forward_solver_step_m,
) noexcept nogil:
    cdef double distance = distance_m
    cdef double v = v_entry_ms
    cdef double v_cap = v_cap_ms
    cdef double x = 0.0
    cdef double step, accel, v2, v_next
    if distance < 0.0:
        distance = 0.0
    if v < 0.5:
        v = 0.5
    if v_cap < 0.5:
        v_cap = 0.5
    if distance <= 1e-9:
        return v if v < v_cap else v_cap
    if forward_solver_step_m <= 1e-9:
        forward_solver_step_m = 1.0
    while x < distance - 1e-9:
        if v >= v_cap - 1e-9:
            return v_cap
        step = forward_solver_step_m
        if step > distance - x:
            step = distance - x
        accel = _batch_effective_accel_rate(accel_base_scaled, v, drag_coeff)
        v2 = v * v + 2.0 * accel * step
        if v2 < 0.0:
            v2 = 0.0
        v_next = sqrt(v2)
        if v_next > v_cap:
            v_next = v_cap
        v = v_next
        x += step
    if v < 0.5:
        v = 0.5
    if v > v_cap:
        v = v_cap
    return v


cdef inline double _batch_segment_motion_duration_only(
    double distance_m,
    double v_entry_ms,
    double v_exit_ms,
    double v_cap_ms,
    double accel_ms2,
    double brake_ms2,
    double drag_coeff,
    double profile_solver_step_m,
) noexcept nogil:
    cdef double distance = distance_m
    cdef double brake = brake_ms2
    cdef double v = v_entry_ms
    cdef double v_exit = v_exit_ms
    cdef double v_cap = v_cap_ms
    cdef double x = 0.0
    cdef double t = 0.0
    cdef double step, remaining, brake_needed, accel
    cdef double v2, v_next, avg_v, dt, threshold
    if distance < 0.0:
        distance = 0.0
    if brake < 0.05:
        brake = 0.05
    if v < 0.5:
        v = 0.5
    if v_exit < 0.5:
        v_exit = 0.5
    if v_cap < 0.5:
        v_cap = 0.5
    if distance <= 1e-9:
        return 0.001
    if profile_solver_step_m <= 1e-9:
        profile_solver_step_m = 1.0
    while x < distance - 1e-9:
        step = profile_solver_step_m
        remaining = distance - x
        if step > remaining:
            step = remaining
        brake_needed = (v * v - v_exit * v_exit) / (2.0 * brake)
        if brake_needed < 0.0:
            brake_needed = 0.0
        threshold = remaining - step * 0.5
        if threshold < 0.0:
            threshold = 0.0
        if brake_needed >= threshold:
            accel = -brake
        elif v >= v_cap - 1e-9:
            accel = 0.0
        else:
            accel = _batch_effective_accel_rate(accel_ms2, v, drag_coeff)
        v2 = v * v + 2.0 * accel * step
        if v2 < 0.0:
            v2 = 0.0
        v_next = sqrt(v2)
        if accel >= 0.0 and v_next > v_cap:
            v_next = v_cap
        avg_v = 0.5 * (v + v_next)
        if avg_v < 0.1:
            avg_v = 0.1
        dt = step / avg_v
        x += step
        t += dt
        v = v_next
        if v < 0.5:
            v = 0.5
    if t < 0.001:
        t = 0.001
    return t

# function to precompute immutable geometry used by every exact lap time batch
# in old code, every invocation of numeric batch solve rebuilt those same exact arrays
# now, they can be built once and reused
cpdef tuple prepare_lap_time_track_numeric(
    object micro_sectors_obj,
    double track_length_m,
    object sector_splits_obj,
):
    cdef int raw_n, macro_count, raw_idx, n, idx
    cdef object sector, starts_arr, ends_arr, prog_arr, distances_arr, bounds_arr
    cdef double[:] starts, ends, prog_lens, distances, bounds
    cdef double length_pct, length_frac, distance_m, progress_cursor, segment_start, segment_end, segment_progress

    try:
        raw_n = len(micro_sectors_obj)
    except Exception:
        raw_n = 0
    macro_count = max(1, len(sector_splits_obj) + 1)
    starts_arr = array("d", [0.0]) * max(1, raw_n)
    ends_arr = array("d", [0.0]) * max(1, raw_n)
    prog_arr = array("d", [0.0]) * max(1, raw_n)
    distances_arr = array("d", [0.0]) * max(1, raw_n)
    starts = starts_arr
    ends = ends_arr
    prog_lens = prog_arr
    distances = distances_arr
    progress_cursor = 0.0
    n = 0
    for raw_idx in range(raw_n):
        try:
            sector = micro_sectors_obj[raw_idx]
            length_pct = float(sector.get("length_pct", 0.0))
        except Exception:
            continue
        if length_pct < 0.0:
            length_pct = 0.0
        if length_pct <= 1e-9:
            continue
        length_frac = length_pct / 100.0
        distance_m = track_length_m * length_frac
        segment_start = progress_cursor
        segment_end = segment_start + length_frac
        if segment_end > 1.0:
            segment_end = 1.0
        segment_progress = segment_end - segment_start
        if segment_progress < 1e-9:
            segment_progress = 1e-9
        starts[n] = segment_start
        ends[n] = segment_end
        prog_lens[n] = segment_progress
        distances[n] = distance_m
        progress_cursor = segment_end
        n += 1
    bounds_arr = array("d", [0.0]) * (macro_count + 1)
    bounds = bounds_arr
    bounds[0] = 0.0
    for idx in range(len(sector_splits_obj)):
        bounds[idx + 1] = float(sector_splits_obj[idx])
    bounds[macro_count] = 1.0
    return ("LTT_CONTEXT_V1", starts_arr, ends_arr, prog_arr, distances_arr, bounds_arr, n, macro_count)

# pure numeric ordered batch of the scalar lap only solver
cpdef list build_lap_time_only_batch_numeric(
    object micro_sectors_obj,
    object zone_targets_batch_obj,
    double track_length_m,
    object sector_splits_obj,
    object accel_rates_obj,
    object braking_rates_obj,
    object accel_base_scaled_obj,
    object drag_coeffs_obj,
    double forward_solver_step_m=1.0,
    int solve_passes=12,
    double profile_solver_step_m=1.0,
):
    cdef bint converged
    cdef int raw_n, n, candidate_count, macro_count
    cdef int raw_idx, idx, candidate_idx, base, node_base, macro_base
    cdef int i, p, orig_i, rot_i, macro_idx, min_idx
    cdef object sector, zone_targets
    cdef list macro_values
    cdef double length_pct, length_frac, distance_m
    cdef double progress_cursor, segment_start, segment_end, segment_progress
    cdef double target, cap, v_in, v_out_max, candidate, v_out, v_in_max
    cdef double braking_rate, accel_base_scaled, drag_coeff
    cdef double segment_time, lap_time, seg_start, seg_end, seg_progress
    cdef double b0, b1, overlap, macro_total, scale, even, total

    try:
        raw_n = len(micro_sectors_obj)
    except Exception:
        raw_n = 0
    try:
        candidate_count = len(zone_targets_batch_obj)
    except Exception:
        candidate_count = 0
    macro_count = max(1, len(sector_splits_obj) + 1)
    if candidate_count <= 0:
        return []

    cdef object starts_arr
    cdef object ends_arr
    cdef object prog_arr
    cdef object distances_arr
    cdef object bounds_arr
    cdef double[:] starts
    cdef double[:] ends
    cdef double[:] prog_lens
    cdef double[:] distances
    cdef double[:] bounds
    cdef bint prepared_context = (isinstance(micro_sectors_obj, tuple)
                                  and len(micro_sectors_obj) >= 8
                                  and micro_sectors_obj[0] == "LTT_CONTEXT_V1")

    if prepared_context:
        starts_arr = micro_sectors_obj[1]
        ends_arr = micro_sectors_obj[2]
        prog_arr = micro_sectors_obj[3]
        distances_arr = micro_sectors_obj[4]
        bounds_arr = micro_sectors_obj[5]
        n = int(micro_sectors_obj[6])
        macro_count = int(micro_sectors_obj[7])
    else:
        starts_arr = array("d", [0.0]) * max(1, raw_n)
        ends_arr = array("d", [0.0]) * max(1, raw_n)
        prog_arr = array("d", [0.0]) * max(1, raw_n)
        distances_arr = array("d", [0.0]) * max(1, raw_n)
        progress_cursor = 0.0
        n = 0
        for raw_idx in range(raw_n):
            try:
                sector = micro_sectors_obj[raw_idx]
                length_pct = float(sector.get("length_pct", 0.0))
            except Exception:
                continue
            if length_pct < 0.0:
                length_pct = 0.0
            if length_pct <= 1e-9:
                continue
            length_frac = length_pct / 100.0
            distance_m = track_length_m * length_frac
            segment_start = progress_cursor
            segment_end = segment_start + length_frac
            if segment_end > 1.0:
                segment_end = 1.0
            segment_progress = segment_end - segment_start
            if segment_progress < 1e-9:
                segment_progress = 1e-9
            starts_arr[n] = segment_start
            ends_arr[n] = segment_end
            prog_arr[n] = segment_progress
            distances_arr[n] = distance_m
            progress_cursor = segment_end
            n += 1
        bounds_arr = array("d", [0.0]) * (macro_count + 1)
        bounds_arr[0] = 0.0
        for idx in range(len(sector_splits_obj)):
            bounds_arr[idx + 1] = float(sector_splits_obj[idx])
        bounds_arr[macro_count] = 1.0

    starts = starts_arr
    ends = ends_arr
    prog_lens = prog_arr
    distances = distances_arr
    bounds = bounds_arr

    if n <= 0:
        even = float(sum([0.001 for _ in range(macro_count)]))
        return [even for _ in range(candidate_count)]

    cdef int task_size = candidate_count * n
    cdef object targets_arr = _ltt_buffer_d("targets", task_size)
    cdef object braking_arr = _ltt_buffer_d("braking", candidate_count)
    cdef object accel_scaled_arr = _ltt_buffer_d("accel_scaled", candidate_count)
    cdef object drag_arr = _ltt_buffer_d("drag", candidate_count)
    cdef object valid_arr = _ltt_buffer_i("valid", candidate_count)
    cdef double[:] targets = targets_arr
    cdef double[:] braking = braking_arr
    cdef double[:] accel_scaled = accel_scaled_arr
    cdef double[:] drag = drag_arr
    cdef int[:] valid = valid_arr
    # Pooled buffers are not zero-filled on reuse, so restore the default
    # the old per-call allocation provided.
    for candidate_idx in range(candidate_count):
        valid[candidate_idx] = 1
    for candidate_idx in range(candidate_count):
        base = candidate_idx * n
        try:
            zone_targets = zone_targets_batch_obj[candidate_idx]
            braking[candidate_idx] = float(braking_rates_obj[candidate_idx])
            accel_scaled[candidate_idx] = float(accel_base_scaled_obj[candidate_idx])
            drag[candidate_idx] = float(drag_coeffs_obj[candidate_idx])
            float(accel_rates_obj[candidate_idx])
        except Exception:
            valid[candidate_idx] = 0
            continue
        for idx in range(n):
            try:
                target = float(zone_targets[idx])
            except Exception:
                target = 200.0
            targets[base + idx] = target

    cdef object caps_arr = _ltt_buffer_d("caps", task_size)
    cdef object caps_rot_arr = _ltt_buffer_d("caps_rot", task_size)
    cdef object dist_rot_arr = _ltt_buffer_d("dist_rot", task_size)
    cdef object order_arr = _ltt_buffer_i("order", task_size)
    cdef object nodes_arr = _ltt_buffer_d("nodes", candidate_count * (n + 1))
    cdef object previous_nodes_arr = _ltt_buffer_d("previous_nodes", candidate_count * (n + 1))
    cdef object entry_arr = _ltt_buffer_d("entry", task_size)
    cdef object exit_arr = _ltt_buffer_d("exits", task_size)
    cdef object macro_arr = _ltt_buffer_d("macro", candidate_count * macro_count)
    cdef object lap_arr = _ltt_buffer_d("lap", candidate_count)
    cdef object result_arr = _ltt_buffer_d("result", candidate_count)
    cdef double[:] caps = caps_arr
    cdef double[:] caps_rot = caps_rot_arr
    cdef double[:] dist_rot = dist_rot_arr
    cdef int[:] order = order_arr
    cdef double[:] nodes = nodes_arr
    cdef double[:] previous_nodes = previous_nodes_arr
    cdef double[:] entry = entry_arr
    cdef double[:] exits = exit_arr
    cdef double[:] macro_times = macro_arr
    cdef double[:] lap_times = lap_arr
    cdef double[:] results = result_arr

    if solve_passes <= 0:
        solve_passes = 1
    # Candidate rows are independent: each retains the scalar operation order,
    # writes only to its original array slot, and contains no RNG.  The Python
    # caller still performs winner selection sequentially in candidate order.
    # Independent candidate rows can have different convergence costs; guided scheduling reduces tail idle time.
    for candidate_idx in prange(candidate_count, nogil=True, schedule='guided'):
            if valid[candidate_idx] == 0:
                results[candidate_idx] = 0.0
                continue
            base = candidate_idx * n
            node_base = candidate_idx * (n + 1)
            macro_base = candidate_idx * macro_count
            braking_rate = braking[candidate_idx]
            accel_base_scaled = accel_scaled[candidate_idx]
            drag_coeff = drag[candidate_idx]

            min_idx = 0
            cap = targets[base] / 3.6
            if cap < 0.5:
                cap = 0.5
            caps[base] = cap
            for idx in range(1, n):
                target = targets[base + idx] / 3.6
                if target < 0.5:
                    target = 0.5
                caps[base + idx] = target
                if target < cap:
                    cap = target
                    min_idx = idx

            for i in range(n):
                orig_i = min_idx + i
                if orig_i >= n:
                    orig_i -= n
                order[base + i] = orig_i
                caps_rot[base + i] = caps[base + orig_i]
                distance_m = distances[orig_i]
                if distance_m < 0.0:
                    distance_m = 0.0
                dist_rot[base + i] = distance_m
                nodes[node_base + i] = caps_rot[base]
            nodes[node_base + n] = caps_rot[base]

            for p in range(solve_passes):
                for i in range(n + 1):
                    previous_nodes[node_base + i] = nodes[node_base + i]
                if nodes[node_base] > caps_rot[base]:
                    nodes[node_base] = caps_rot[base]
                for i in range(n):
                    v_in = nodes[node_base + i]
                    if caps_rot[base + i] < v_in:
                        v_in = caps_rot[base + i]
                    v_out_max = _batch_forward_exit_speed(
                        v_in,
                        dist_rot[base + i],
                        caps_rot[base + i],
                        accel_base_scaled,
                        drag_coeff,
                        forward_solver_step_m,
                    )
                    nodes[node_base + i + 1] = (
                        caps_rot[base + i]
                        if caps_rot[base + i] < v_out_max
                        else v_out_max
                    )
                candidate = nodes[node_base + n]
                if nodes[node_base] < candidate:
                    candidate = nodes[node_base]
                if caps_rot[base] < candidate:
                    candidate = caps_rot[base]
                nodes[node_base + n] = candidate
                for i in range(n - 1, -1, -1):
                    v_out = nodes[node_base + i + 1]
                    if caps_rot[base + i] < v_out:
                        v_out = caps_rot[base + i]
                    candidate = v_out * v_out + 2.0 * braking_rate * dist_rot[base + i]
                    if candidate < 0.0:
                        candidate = 0.0
                    v_in_max = sqrt(candidate)
                    candidate = nodes[node_base + i]
                    if caps_rot[base + i] < candidate:
                        candidate = caps_rot[base + i]
                    if v_in_max < candidate:
                        candidate = v_in_max
                    nodes[node_base + i] = candidate
                if nodes[node_base + n] > nodes[node_base]:
                    nodes[node_base + n] = nodes[node_base]

                # An unchanged complete pass is an exact fixed point. Repeating
                # it cannot change the profile; retain all passes otherwise.
                # No tolerance or reduced solver resolution is used here.
                converged = True
                for i in range(n + 1):
                    if nodes[node_base + i] != previous_nodes[node_base + i]:
                        converged = False
                        break
                if converged:
                    break

            for rot_i in range(n):
                orig_i = order[base + rot_i]
                entry[base + orig_i] = nodes[node_base + rot_i]
                exits[base + orig_i] = nodes[node_base + rot_i + 1]
            for macro_idx in range(macro_count):
                macro_times[macro_base + macro_idx] = 0.0

            lap_time = 0.0
            for idx in range(n):
                segment_time = _batch_segment_motion_duration_only(
                    distances[idx],
                    entry[base + idx],
                    exits[base + idx],
                    caps[base + idx],
                    accel_base_scaled,
                    braking_rate,
                    drag_coeff,
                    profile_solver_step_m,
                )
                seg_start = starts[idx]
                seg_end = ends[idx]
                seg_progress = prog_lens[idx]
                if seg_progress < 1e-9:
                    seg_progress = 1e-9
                lap_time = lap_time + segment_time
                for macro_idx in range(macro_count):
                    b0 = bounds[macro_idx]
                    b1 = bounds[macro_idx + 1]
                    overlap = (seg_end if seg_end < b1 else b1) - (
                        seg_start if seg_start > b0 else b0
                    )
                    if overlap <= 1e-9:
                        continue
                    macro_times[macro_base + macro_idx] += (
                        segment_time * (overlap / seg_progress)
                    )

            if lap_time <= 0.0:
                lap_time = 0.001
            lap_times[candidate_idx] = lap_time

    # Preserve the scalar path's Python 3.12+ compensated float summation for
    # macro-sector normalization.  This inexpensive final pass remains ordered
    # by candidate index after the numeric work completes.
    for candidate_idx in range(candidate_count):
            if valid[candidate_idx] == 0:
                results[candidate_idx] = 0.0
                continue
            macro_base = candidate_idx * macro_count
            lap_time = lap_times[candidate_idx]
            # Python 3.12+ uses a compensated float summation in ``sum``.
            # The established scalar path calls that builtin for macro-sector
            # totals, so retain it here to guarantee bit-identical projections.
            macro_values = [
                float(macro_times[macro_base + macro_idx])
                for macro_idx in range(macro_count)
            ]
            macro_total = float(sum(macro_values))
            if macro_total > 0.0:
                scale = lap_time / macro_total
                macro_values = [
                    max(0.001, float(total) * scale)
                    for total in macro_values
                ]
            else:
                even = lap_time / macro_count
                macro_values = [
                    max(0.001, even) for _ in range(macro_count)
                ]
            results[candidate_idx] = float(sum(macro_values))

    return [float(results[idx]) for idx in range(candidate_count)]


cdef void _fill_fast_profile_candidate(
    int candidate_idx,
    int n,
    int macro_count,
    int sample_total,
    int solve_passes,
    double forward_solver_step_m,
    double profile_solver_step_m,
    int[:] valid,
    int[:] sample_offsets,
    double[:] starts,
    double[:] ends,
    double[:] prog_lens,
    double[:] distances,
    double[:] bounds,
    double[:] targets,
    double[:] braking,
    double[:] accel_rates,
    double[:] accel_scaled,
    double[:] drag,
    double[:] caps,
    double[:] caps_rot,
    double[:] dist_rot,
    int[:] order,
    double[:] nodes,
    double[:] entry,
    double[:] exits,
    double[:] start_times,
    double[:] durations,
    double[:] peaks,
    double[:] dist_accs,
    double[:] dist_cruises,
    double[:] time_accs,
    double[:] time_cruises,
    double[:] macro_times,
    double[:] lap_times,
    double[:] sample_dist,
    double[:] sample_time,
    double[:] sample_speed,
) noexcept nogil:
    cdef bint converged
    cdef double previous_end
    cdef int base, node_base, macro_base, idx, i, p, orig_i, rot_i
    cdef int macro_idx, min_idx, phase, sample_base, sample_idx
    cdef double braking_rate, accel_base_scaled, drag_coeff
    cdef double cap, target, distance_m, v_in, v_out_max, candidate
    cdef double v_out, v_in_max, lap_time, seg_start, seg_end, seg_progress
    cdef double b0, b1, overlap, x, t, step, remaining, brake_needed
    cdef double accel, v, v_exit, v_cap, v2, v_next, avg_v, dt, threshold
    cdef double dist_acc, dist_cruise, time_acc, time_cruise, v_peak

    if valid[candidate_idx] == 0:
        return
    base = candidate_idx * n
    node_base = candidate_idx * (n + 1)
    macro_base = candidate_idx * macro_count
    braking_rate = braking[candidate_idx]
    accel_base_scaled = accel_scaled[candidate_idx]
    drag_coeff = drag[candidate_idx]

    min_idx = 0
    cap = targets[base] / 3.6
    if cap < 0.5:
        cap = 0.5
    caps[base] = cap
    for idx in range(1, n):
        target = targets[base + idx] / 3.6
        if target < 0.5:
            target = 0.5
        caps[base + idx] = target
        if target < cap:
            cap = target
            min_idx = idx

    for i in range(n):
        orig_i = min_idx + i
        if orig_i >= n:
            orig_i -= n
        order[base + i] = orig_i
        caps_rot[base + i] = caps[base + orig_i]
        distance_m = distances[orig_i]
        if distance_m < 0.0:
            distance_m = 0.0
        dist_rot[base + i] = distance_m
        nodes[node_base + i] = caps_rot[base]
    nodes[node_base + n] = caps_rot[base]

    for p in range(solve_passes):
        # Reuse entry scratch until the final node profile is ready.
        for i in range(n):
            entry[base+i] = nodes[node_base+i]
        previous_end = nodes[node_base+n]
        if nodes[node_base] > caps_rot[base]:
            nodes[node_base] = caps_rot[base]
        for i in range(n):
            v_in = nodes[node_base + i]
            if caps_rot[base + i] < v_in:
                v_in = caps_rot[base + i]
            v_out_max = _batch_forward_exit_speed(
                v_in,
                dist_rot[base + i],
                caps_rot[base + i],
                accel_base_scaled,
                drag_coeff,
                forward_solver_step_m,
            )
            nodes[node_base + i + 1] = (
                caps_rot[base + i]
                if caps_rot[base + i] < v_out_max
                else v_out_max
            )
        candidate = nodes[node_base + n]
        if nodes[node_base] < candidate:
            candidate = nodes[node_base]
        if caps_rot[base] < candidate:
            candidate = caps_rot[base]
        nodes[node_base + n] = candidate
        for i in range(n - 1, -1, -1):
            v_out = nodes[node_base + i + 1]
            if caps_rot[base + i] < v_out:
                v_out = caps_rot[base + i]
            candidate = v_out * v_out + 2.0 * braking_rate * dist_rot[base + i]
            if candidate < 0.0:
                candidate = 0.0
            v_in_max = sqrt(candidate)
            candidate = nodes[node_base + i]
            if caps_rot[base + i] < candidate:
                candidate = caps_rot[base + i]
            if v_in_max < candidate:
                candidate = v_in_max
            nodes[node_base + i] = candidate
        if nodes[node_base + n] > nodes[node_base]:
            nodes[node_base + n] = nodes[node_base]

        converged = nodes[node_base+n] == previous_end
        for i in range(n):
            if entry[base+i] != nodes[node_base+i]:
                converged = False
                break
        if converged:
            break

    for rot_i in range(n):
        orig_i = order[base + rot_i]
        entry[base + orig_i] = nodes[node_base + rot_i]
        exits[base + orig_i] = nodes[node_base + rot_i + 1]
    for macro_idx in range(macro_count):
        macro_times[macro_base + macro_idx] = 0.0

    lap_time = 0.0
    for idx in range(n):
        distance_m = distances[idx]
        if distance_m < 0.0:
            distance_m = 0.0
        braking_rate = braking[candidate_idx]
        if braking_rate < 0.05:
            braking_rate = 0.05
        v = entry[base + idx]
        if v < 0.5:
            v = 0.5
        v_exit = exits[base + idx]
        if v_exit < 0.5:
            v_exit = 0.5
        v_cap = caps[base + idx]
        if v_cap < 0.5:
            v_cap = 0.5

        sample_base = candidate_idx * sample_total + sample_offsets[idx]
        sample_idx = 0
        x = 0.0
        t = 0.0
        sample_dist[sample_base] = 0.0
        sample_time[sample_base] = 0.0
        sample_speed[sample_base] = v
        dist_acc = 0.0
        dist_cruise = 0.0
        time_acc = 0.0
        time_cruise = 0.0
        v_peak = v

        if distance_m <= 1e-9:
            t = 0.001
            if v_exit > v_peak:
                v_peak = v_exit
            sample_speed[sample_base] = v_exit
        else:
            while x < distance_m - 1e-9:
                step = profile_solver_step_m
                remaining = distance_m - x
                if step > remaining:
                    step = remaining
                brake_needed = (v * v - v_exit * v_exit) / (2.0 * braking_rate)
                if brake_needed < 0.0:
                    brake_needed = 0.0
                threshold = remaining - step * 0.5
                if threshold < 0.0:
                    threshold = 0.0
                if brake_needed >= threshold:
                    accel = -braking_rate
                    phase = 2
                elif v >= v_cap - 1e-9:
                    accel = 0.0
                    phase = 1
                else:
                    accel = _batch_effective_accel_rate(
                        accel_base_scaled,
                        v,
                        drag_coeff,
                    )
                    phase = 0
                v2 = v * v + 2.0 * accel * step
                if v2 < 0.0:
                    v2 = 0.0
                v_next = sqrt(v2)
                if accel >= 0.0 and v_next > v_cap:
                    v_next = v_cap
                avg_v = 0.5 * (v + v_next)
                if avg_v < 0.1:
                    avg_v = 0.1
                dt = step / avg_v
                x += step
                t += dt
                v = v_next
                if v < 0.5:
                    v = 0.5
                if v > v_peak:
                    v_peak = v
                sample_idx += 1
                sample_dist[sample_base + sample_idx] = x
                sample_time[sample_base + sample_idx] = t
                sample_speed[sample_base + sample_idx] = v
                if phase == 0:
                    dist_acc += step
                    time_acc += dt
                elif phase == 1:
                    dist_cruise += step
                    time_cruise += dt
            sample_speed[sample_base + sample_idx] = v_exit
            if t < 0.001:
                t = 0.001

        start_times[base + idx] = lap_time
        durations[base + idx] = t
        peaks[base + idx] = v_peak
        dist_accs[base + idx] = dist_acc
        dist_cruises[base + idx] = dist_cruise
        time_accs[base + idx] = time_acc
        time_cruises[base + idx] = time_cruise
        seg_start = starts[idx]
        seg_end = ends[idx]
        seg_progress = prog_lens[idx]
        if seg_progress < 1e-9:
            seg_progress = 1e-9
        for macro_idx in range(macro_count):
            b0 = bounds[macro_idx]
            b1 = bounds[macro_idx + 1]
            overlap = (seg_end if seg_end < b1 else b1) - (
                seg_start if seg_start > b0 else b0
            )
            if overlap <= 1e-9:
                continue
            macro_times[macro_base + macro_idx] += t * (overlap / seg_progress)
        lap_time += t
    if lap_time <= 0.0:
        lap_time = 0.001
    lap_times[candidate_idx] = lap_time


cpdef list build_fast_profile_batch_numeric(
    object micro_sectors_obj,
    object zone_targets_batch_obj,
    double track_length_m,
    object sector_splits_obj,
    object accel_rates_obj,
    object braking_rates_obj,
    object accel_base_scaled_obj,
    object drag_coeffs_obj,
    double forward_solver_step_m=1.0,
    int solve_passes=12,
    double profile_solver_step_m=1.0,
    bint lazy_samples=False,
):
    """Build ordered full profiles while sharing immutable circuit parsing.

    Candidate rows are independent and therefore run in parallel.  Each row
    intentionally retains the scalar fast-profile operation order and double
    precision.  Python installs the returned profiles into its LRU cache later,
    in the original request order, so this function has no cache or RNG side
    effects.
    """
    cdef bint converged
    cdef double previous_end
    cdef int raw_n, n, candidate_count, macro_count
    cdef int raw_idx, idx, candidate_idx, base, node_base, macro_base
    cdef int i, p, orig_i, rot_i, macro_idx, min_idx, phase
    cdef int sample_total, sample_base, sample_idx, sample_count
    cdef object sector, zone_targets
    cdef list macro_values, profiles, sample_dist_list, sample_time_list, sample_speed_list
    cdef double length_pct, length_frac, distance_m
    cdef double progress_cursor, segment_start, segment_end, segment_progress
    cdef double target, cap, v_in, v_out_max, candidate, v_out, v_in_max
    cdef double braking_rate, accel_rate, accel_base_scaled, drag_coeff
    cdef double segment_time, lap_time, seg_start, seg_end, seg_progress
    cdef double b0, b1, overlap, macro_total, scale, even
    cdef double x, t, step, remaining, brake_needed, accel
    cdef double v, v_exit, v_cap, v2, v_next, avg_v, dt, threshold
    cdef double dist_acc, dist_cruise, time_acc, time_cruise, v_peak
    cdef FastProfile fp

    try:
        raw_n = len(micro_sectors_obj)
    except Exception:
        raw_n = 0
    try:
        candidate_count = len(zone_targets_batch_obj)
    except Exception:
        candidate_count = 0
    macro_count = max(1, len(sector_splits_obj) + 1)
    if candidate_count <= 0:
        return []

    cdef object starts_arr = array("d", [0.0]) * max(1, raw_n)
    cdef object ends_arr = array("d", [0.0]) * max(1, raw_n)
    cdef object prog_arr = array("d", [0.0]) * max(1, raw_n)
    cdef object distances_arr = array("d", [0.0]) * max(1, raw_n)
    cdef double[:] starts = starts_arr
    cdef double[:] ends = ends_arr
    cdef double[:] prog_lens = prog_arr
    cdef double[:] distances = distances_arr
    progress_cursor = 0.0
    n = 0
    for raw_idx in range(raw_n):
        try:
            sector = micro_sectors_obj[raw_idx]
            length_pct = float(sector.get("length_pct", 0.0))
        except Exception:
            continue
        if length_pct < 0.0:
            length_pct = 0.0
        if length_pct <= 1e-9:
            continue
        length_frac = length_pct / 100.0
        distance_m = track_length_m * length_frac
        segment_start = progress_cursor
        segment_end = segment_start + length_frac
        if segment_end > 1.0:
            segment_end = 1.0
        segment_progress = segment_end - segment_start
        if segment_progress < 1e-9:
            segment_progress = 1e-9
        starts[n] = segment_start
        ends[n] = segment_end
        prog_lens[n] = segment_progress
        distances[n] = distance_m
        progress_cursor = segment_end
        n += 1

    if n <= 0:
        profiles = []
        for candidate_idx in range(candidate_count):
            fp = FastProfile()
            fp.lap_time = 0.001
            fp.macro_times = [0.001 for _ in range(macro_count)]
            profiles.append(fp)
        return profiles

    cdef object bounds_arr = array("d", [0.0]) * (macro_count + 1)
    cdef double[:] bounds = bounds_arr
    bounds[0] = 0.0
    for idx in range(len(sector_splits_obj)):
        bounds[idx + 1] = float(sector_splits_obj[idx])
    bounds[macro_count] = 1.0

    # Sample counts depend only on immutable segment length and solver step.
    cdef object sample_offsets_arr = array("i", [0]) * (n + 1)
    cdef int[:] sample_offsets = sample_offsets_arr
    if profile_solver_step_m <= 1e-9:
        profile_solver_step_m = 1.0
    sample_total = 0
    for idx in range(n):
        sample_offsets[idx] = sample_total
        x = 0.0
        sample_count = 1
        distance_m = distances[idx]
        if distance_m < 0.0:
            distance_m = 0.0
        while x < distance_m - 1e-9:
            step = profile_solver_step_m
            if step > distance_m - x:
                step = distance_m - x
            x += step
            sample_count += 1
        sample_total += sample_count
    sample_offsets[n] = sample_total

    cdef int task_size = candidate_count * n
    cdef int sample_buffer_size
    cdef object targets_arr = array("d", [0.0]) * task_size
    cdef object braking_arr = array("d", [0.0]) * candidate_count
    cdef object accel_arr = array("d", [0.0]) * candidate_count
    cdef object accel_scaled_arr = array("d", [0.0]) * candidate_count
    cdef object drag_arr = array("d", [0.0]) * candidate_count
    cdef object valid_arr = array("i", [1]) * candidate_count
    cdef double[:] targets = targets_arr
    cdef double[:] braking = braking_arr
    cdef double[:] accel_rates = accel_arr
    cdef double[:] accel_scaled = accel_scaled_arr
    cdef double[:] drag = drag_arr
    cdef int[:] valid = valid_arr
    for candidate_idx in range(candidate_count):
        base = candidate_idx * n
        try:
            zone_targets = zone_targets_batch_obj[candidate_idx]
            braking[candidate_idx] = float(braking_rates_obj[candidate_idx])
            accel_rates[candidate_idx] = float(accel_rates_obj[candidate_idx])
            accel_scaled[candidate_idx] = float(accel_base_scaled_obj[candidate_idx])
            drag[candidate_idx] = float(drag_coeffs_obj[candidate_idx])
        except Exception:
            valid[candidate_idx] = 0
            continue
        for idx in range(n):
            try:
                target = float(zone_targets[idx])
            except Exception:
                target = 200.0
            targets[base + idx] = target

    cdef object caps_arr = array("d", [0.0]) * task_size
    cdef object caps_rot_arr = array("d", [0.0]) * task_size
    cdef object dist_rot_arr = array("d", [0.0]) * task_size
    cdef object order_arr = array("i", [0]) * task_size
    cdef object nodes_arr = array("d", [0.0]) * (candidate_count * (n + 1))
    cdef object entry_arr = array("d", [0.0]) * task_size
    cdef object exit_arr = array("d", [0.0]) * task_size
    cdef object start_time_arr = array("d", [0.0]) * task_size
    cdef object duration_arr = array("d", [0.0]) * task_size
    cdef object peak_arr = array("d", [0.0]) * task_size
    cdef object dist_acc_arr = array("d", [0.0]) * task_size
    cdef object dist_cruise_arr = array("d", [0.0]) * task_size
    cdef object time_acc_arr = array("d", [0.0]) * task_size
    cdef object time_cruise_arr = array("d", [0.0]) * task_size
    cdef object macro_arr = array("d", [0.0]) * (candidate_count * macro_count)
    cdef object lap_arr = array("d", [0.0]) * candidate_count
    sample_buffer_size = 1 if lazy_samples else (candidate_count * sample_total)
    cdef object sample_dist_arr = array("d", [0.0]) * sample_buffer_size
    cdef object sample_time_arr = array("d", [0.0]) * sample_buffer_size
    cdef object sample_speed_arr = array("d", [0.0]) * sample_buffer_size
    cdef double[:] caps = caps_arr
    cdef double[:] caps_rot = caps_rot_arr
    cdef double[:] dist_rot = dist_rot_arr
    cdef int[:] order = order_arr
    cdef double[:] nodes = nodes_arr
    cdef double[:] entry = entry_arr
    cdef double[:] exits = exit_arr
    cdef double[:] start_times = start_time_arr
    cdef double[:] durations = duration_arr
    cdef double[:] peaks = peak_arr
    cdef double[:] dist_accs = dist_acc_arr
    cdef double[:] dist_cruises = dist_cruise_arr
    cdef double[:] time_accs = time_acc_arr
    cdef double[:] time_cruises = time_cruise_arr
    cdef double[:] macro_times = macro_arr
    cdef double[:] lap_times = lap_arr
    cdef double[:] sample_dist = sample_dist_arr
    cdef double[:] sample_time = sample_time_arr
    cdef double[:] sample_speed = sample_speed_arr

    if solve_passes <= 0:
        solve_passes = 1
    for candidate_idx in prange(candidate_count, nogil=True, schedule='static'):
        if valid[candidate_idx] == 0:
            continue
        base = candidate_idx * n
        node_base = candidate_idx * (n + 1)
        macro_base = candidate_idx * macro_count
        braking_rate = braking[candidate_idx]
        accel_rate = accel_rates[candidate_idx]
        accel_base_scaled = accel_scaled[candidate_idx]
        drag_coeff = drag[candidate_idx]

        min_idx = 0
        cap = targets[base] / 3.6
        if cap < 0.5:
            cap = 0.5
        caps[base] = cap
        for idx in range(1, n):
            target = targets[base + idx] / 3.6
            if target < 0.5:
                target = 0.5
            caps[base + idx] = target
            if target < cap:
                cap = target
                min_idx = idx

        for i in range(n):
            orig_i = min_idx + i
            if orig_i >= n:
                orig_i -= n
            order[base + i] = orig_i
            caps_rot[base + i] = caps[base + orig_i]
            distance_m = distances[orig_i]
            if distance_m < 0.0:
                distance_m = 0.0
            dist_rot[base + i] = distance_m
            nodes[node_base + i] = caps_rot[base]
        nodes[node_base + n] = caps_rot[base]

        for p in range(solve_passes):
            # Reuse entry scratch until the final node profile is ready.
            for i in range(n):
                entry[base+i] = nodes[node_base+i]
            previous_end = nodes[node_base+n]
            if nodes[node_base] > caps_rot[base]:
                nodes[node_base] = caps_rot[base]
            for i in range(n):
                v_in = nodes[node_base + i]
                if caps_rot[base + i] < v_in:
                    v_in = caps_rot[base + i]
                v_out_max = _batch_forward_exit_speed(
                    v_in,
                    dist_rot[base + i],
                    caps_rot[base + i],
                    accel_base_scaled,
                    drag_coeff,
                    forward_solver_step_m,
                )
                nodes[node_base + i + 1] = (
                    caps_rot[base + i]
                    if caps_rot[base + i] < v_out_max
                    else v_out_max
                )
            candidate = nodes[node_base + n]
            if nodes[node_base] < candidate:
                candidate = nodes[node_base]
            if caps_rot[base] < candidate:
                candidate = caps_rot[base]
            nodes[node_base + n] = candidate
            for i in range(n - 1, -1, -1):
                v_out = nodes[node_base + i + 1]
                if caps_rot[base + i] < v_out:
                    v_out = caps_rot[base + i]
                candidate = v_out * v_out + 2.0 * braking_rate * dist_rot[base + i]
                if candidate < 0.0:
                    candidate = 0.0
                v_in_max = sqrt(candidate)
                candidate = nodes[node_base + i]
                if caps_rot[base + i] < candidate:
                    candidate = caps_rot[base + i]
                if v_in_max < candidate:
                    candidate = v_in_max
                nodes[node_base + i] = candidate
            if nodes[node_base + n] > nodes[node_base]:
                nodes[node_base + n] = nodes[node_base]

            converged = nodes[node_base+n] == previous_end
            for i in range(n):
                if entry[base+i] != nodes[node_base+i]:
                    converged = False
                    break
            if converged:
                break

        for rot_i in range(n):
            orig_i = order[base + rot_i]
            entry[base + orig_i] = nodes[node_base + rot_i]
            exits[base + orig_i] = nodes[node_base + rot_i + 1]
        for macro_idx in range(macro_count):
            macro_times[macro_base + macro_idx] = 0.0

        if lazy_samples:
            lap_times[candidate_idx] = 0.001
            continue

        lap_time = 0.0
        for idx in range(n):
            distance_m = distances[idx]
            if distance_m < 0.0:
                distance_m = 0.0
            braking_rate = braking[candidate_idx]
            if braking_rate < 0.05:
                braking_rate = 0.05
            v = entry[base + idx]
            if v < 0.5:
                v = 0.5
            v_exit = exits[base + idx]
            if v_exit < 0.5:
                v_exit = 0.5
            v_cap = caps[base + idx]
            if v_cap < 0.5:
                v_cap = 0.5

            if lazy_samples:
                t = _batch_segment_motion_duration_only(
                    distance_m,
                    v,
                    v_exit,
                    v_cap,
                    accel_base_scaled,
                    braking_rate,
                    drag_coeff,
                    profile_solver_step_m,
                )
                start_times[base + idx] = lap_time
                durations[base + idx] = t
                peaks[base + idx] = v if v > v_exit else v_exit
                dist_accs[base + idx] = 0.0
                dist_cruises[base + idx] = 0.0
                time_accs[base + idx] = 0.0
                time_cruises[base + idx] = 0.0
                seg_start = starts[idx]
                seg_end = ends[idx]
                seg_progress = prog_lens[idx]
                if seg_progress < 1e-9:
                    seg_progress = 1e-9
                for macro_idx in range(macro_count):
                    b0 = bounds[macro_idx]
                    b1 = bounds[macro_idx + 1]
                    overlap = (seg_end if seg_end < b1 else b1) - (
                        seg_start if seg_start > b0 else b0
                    )
                    if overlap <= 1e-9:
                        continue
                    macro_times[macro_base + macro_idx] += t * (overlap / seg_progress)
                lap_time = lap_time + t
                continue

            sample_base = candidate_idx * sample_total + sample_offsets[idx]
            sample_idx = 0
            x = 0.0
            t = 0.0
            sample_dist[sample_base] = 0.0
            sample_time[sample_base] = 0.0
            sample_speed[sample_base] = v
            dist_acc = 0.0
            dist_cruise = 0.0
            time_acc = 0.0
            time_cruise = 0.0
            v_peak = v

            if distance_m <= 1e-9:
                t = 0.001
                if v_exit > v_peak:
                    v_peak = v_exit
                sample_speed[sample_base] = v_exit
            else:
                while x < distance_m - 1e-9:
                    step = profile_solver_step_m
                    remaining = distance_m - x
                    if step > remaining:
                        step = remaining
                    brake_needed = (v * v - v_exit * v_exit) / (2.0 * braking_rate)
                    if brake_needed < 0.0:
                        brake_needed = 0.0
                    threshold = remaining - step * 0.5
                    if threshold < 0.0:
                        threshold = 0.0
                    if brake_needed >= threshold:
                        accel = -braking_rate
                        phase = 2
                    elif v >= v_cap - 1e-9:
                        accel = 0.0
                        phase = 1
                    else:
                        accel = _batch_effective_accel_rate(
                            accel_base_scaled,
                            v,
                            drag_coeff,
                        )
                        phase = 0
                    v2 = v * v + 2.0 * accel * step
                    if v2 < 0.0:
                        v2 = 0.0
                    v_next = sqrt(v2)
                    if accel >= 0.0 and v_next > v_cap:
                        v_next = v_cap
                    avg_v = 0.5 * (v + v_next)
                    if avg_v < 0.1:
                        avg_v = 0.1
                    dt = step / avg_v
                    x = x + step
                    t = t + dt
                    v = v_next
                    if v < 0.5:
                        v = 0.5
                    if v > v_peak:
                        v_peak = v
                    sample_idx = sample_idx + 1
                    sample_dist[sample_base + sample_idx] = x
                    sample_time[sample_base + sample_idx] = t
                    sample_speed[sample_base + sample_idx] = v
                    if phase == 0:
                        dist_acc = dist_acc + step
                        time_acc = time_acc + dt
                    elif phase == 1:
                        dist_cruise = dist_cruise + step
                        time_cruise = time_cruise + dt
                sample_speed[sample_base + sample_idx] = v_exit
                if t < 0.001:
                    t = 0.001

            start_times[base + idx] = lap_time
            durations[base + idx] = t
            peaks[base + idx] = v_peak
            dist_accs[base + idx] = dist_acc
            dist_cruises[base + idx] = dist_cruise
            time_accs[base + idx] = time_acc
            time_cruises[base + idx] = time_cruise
            seg_start = starts[idx]
            seg_end = ends[idx]
            seg_progress = prog_lens[idx]
            if seg_progress < 1e-9:
                seg_progress = 1e-9
            for macro_idx in range(macro_count):
                b0 = bounds[macro_idx]
                b1 = bounds[macro_idx + 1]
                overlap = (seg_end if seg_end < b1 else b1) - (
                    seg_start if seg_start > b0 else b0
                )
                if overlap <= 1e-9:
                    continue
                macro_times[macro_base + macro_idx] += t * (overlap / seg_progress)
            lap_time = lap_time + t
        if lap_time <= 0.0:
            lap_time = 0.001
        lap_times[candidate_idx] = lap_time

    # Construct Python-owned FastProfile objects after the nogil work.  This
    # ordered pass is also where scalar-compatible Python float summation is
    # retained for macro-sector normalization.
    profiles = []
    for candidate_idx in range(candidate_count):
        if valid[candidate_idx] == 0:
            profiles.append(None)
            continue
        base = candidate_idx * n
        macro_base = candidate_idx * macro_count
        lap_time = lap_times[candidate_idx]
        if lazy_samples:
            macro_values = [0.001 for _ in range(macro_count)]
        else:
            macro_values = [
                float(macro_times[macro_base + macro_idx])
                for macro_idx in range(macro_count)
            ]
            macro_total = float(sum(macro_values))
            if macro_total > 0.0:
                scale = lap_time / macro_total
                macro_values = [max(0.001, float(target) * scale) for target in macro_values]
            else:
                even = lap_time / macro_count
                macro_values = [max(0.001, even) for _ in range(macro_count)]

        fp = FastProfile()
        fp.lap_time = float(sum(macro_values))
        fp.macro_times = list(macro_values)
        fp.segment_end_progress = [float(ends[idx]) for idx in range(n)]
        fp.count = n
        # Memoryview slices retain their shared batch buffers without copying
        # thousands of one-metre samples back through Python per profile.
        fp.start_progress_arr = starts[:n]
        fp.end_progress_arr = ends[:n]
        fp.progress_len_arr = prog_lens[:n]
        fp.start_time_arr = start_times[base:base + n]
        fp.duration_arr = durations[base:base + n]
        fp.distance_arr = distances[:n]
        fp.v_entry_arr = entry[base:base + n]
        fp.v_exit_arr = exits[base:base + n]
        fp.v_peak_arr = peaks[base:base + n]
        fp.dist_acc_arr = dist_accs[base:base + n]
        fp.dist_cruise_arr = dist_cruises[base:base + n]
        fp.time_acc_arr = time_accs[base:base + n]
        fp.time_cruise_arr = time_cruises[base:base + n]
        fp.accel_rate_arr = array("d", [float(accel_rates[candidate_idx])]) * n
        fp.accel_scaled_arr = array("d", [float(accel_scaled[candidate_idx])]) * n
        fp.brake_rate_arr = array("d", [float(braking[candidate_idx])]) * n
        fp.v_cap_arr = caps[base:base + n]
        fp.drag_coeff_arr = array("d", [float(drag[candidate_idx])]) * n
        fp.profile_solver_step_m = float(profile_solver_step_m)
        fp.completion_pending = bool(lazy_samples)
        fp.macro_bounds = [float(bounds[idx]) for idx in range(macro_count + 1)]
        sample_dist_list = []
        sample_time_list = []
        sample_speed_list = []
        for idx in range(n):
            if lazy_samples:
                sample_dist_list.append(None)
                sample_time_list.append(None)
                sample_speed_list.append(None)
                continue
            sample_base = candidate_idx * sample_total + sample_offsets[idx]
            sample_count = sample_offsets[idx + 1] - sample_offsets[idx]
            sample_dist_list.append(
                sample_dist[sample_base:sample_base + sample_count]
            )
            sample_time_list.append(
                sample_time[sample_base:sample_base + sample_count]
            )
            sample_speed_list.append(
                sample_speed[sample_base:sample_base + sample_count]
            )
        fp.sample_dist_arrs = sample_dist_list
        fp.sample_time_arrs = sample_time_list
        fp.sample_speed_arrs = sample_speed_list
        fp.start_progress = fp.start_progress_arr
        fp.end_progress = fp.end_progress_arr
        fp.progress_len = fp.progress_len_arr
        fp.start_time = fp.start_time_arr
        fp.duration = fp.duration_arr
        fp.distance = fp.distance_arr
        fp.v_entry = fp.v_entry_arr
        fp.v_exit = fp.v_exit_arr
        fp.v_peak = fp.v_peak_arr
        fp.dist_acc = fp.dist_acc_arr
        fp.dist_cruise = fp.dist_cruise_arr
        fp.time_acc = fp.time_acc_arr
        fp.time_cruise = fp.time_cruise_arr
        fp.accel_rate = fp.accel_rate_arr
        fp.accel_scaled = fp.accel_scaled_arr
        fp.brake_rate = fp.brake_rate_arr
        fp.v_cap = fp.v_cap_arr
        fp.drag_coeff = fp.drag_coeff_arr
        profiles.append(fp)
    return profiles


cpdef double progress_for_elapsed_from_profile(
    object profile_obj,
    double elapsed_s,
    double target_lap,
):
    if isinstance(profile_obj, FastProfile):
        return _fast_profile_progress_for_elapsed(profile_obj, elapsed_s, target_lap)
    try:
        fast = profile_obj.get("_fast_profile")
        if isinstance(fast, FastProfile):
            return _fast_profile_progress_for_elapsed(fast, elapsed_s, target_lap)
    except Exception:
        pass
    return NAN


cpdef double forward_exit_speed(
    double v_entry_ms,
    double distance_m,
    double v_cap_ms,
    double accel_base_scaled,
    double drag_coeff,
    double forward_solver_step_m=1.0,
):
    cdef double distance, v, v_cap, x, step, accel, v2, v_next
    distance = distance_m
    if distance < 0.0:
        distance = 0.0
    if distance <= 1e-9:
        if v_entry_ms < 0.5:
            v_entry_ms = 0.5
        if v_cap_ms < 0.5:
            v_cap_ms = 0.5
        return v_entry_ms if v_entry_ms < v_cap_ms else v_cap_ms

    v = v_entry_ms
    if v < 0.5:
        v = 0.5
    v_cap = v_cap_ms
    if v_cap < 0.5:
        v_cap = 0.5
    x = 0.0
    if forward_solver_step_m <= 1e-9:
        forward_solver_step_m = 1.0

    while x < distance - 1e-9:
        if v >= v_cap - 1e-9:
            return v_cap
        step = forward_solver_step_m
        if step > distance - x:
            step = distance - x
        accel = _effective_accel_rate(accel_base_scaled, v, drag_coeff)
        v2 = v * v + 2.0 * accel * step
        if v2 < 0.0:
            v2 = 0.0
        v_next = sqrt(v2)
        if v_next > v_cap:
            v_next = v_cap
        v = v_next
        x += step
    if v < 0.5:
        v = 0.5
    if v > v_cap:
        v = v_cap
    return v


cpdef tuple segment_motion_profile_tuple(
    double distance_m,
    double v_entry_ms,
    double v_exit_ms,
    double v_cap_ms,
    double accel_ms2,
    double brake_ms2,
    double drag_coeff,
    double profile_solver_step_m=1.0,
):
    cdef double distance, brake, v, v_exit, v_cap
    cdef double x, t, dist_acc, dist_dec, dist_cruise
    cdef double time_acc, time_dec, time_cruise, v_peak
    cdef double step, remaining, brake_needed, accel
    cdef double v2, v_next, avg_v, dt
    cdef int phase
    cdef object sample_dist, sample_time, sample_speed

    distance = distance_m
    if distance < 0.0:
        distance = 0.0
    brake = brake_ms2
    if brake < 0.05:
        brake = 0.05
    v = v_entry_ms
    if v < 0.5:
        v = 0.5
    v_exit = v_exit_ms
    if v_exit < 0.5:
        v_exit = 0.5
    v_cap = v_cap_ms
    if v_cap < 0.5:
        v_cap = 0.5

    if distance <= 1e-9:
        return (
            0.001,
            v if v > v_exit else v_exit,
            0.0,
            0.0,
            0.0,
            0.0,
            array("d", [0.0]),
            array("d", [0.0]),
            array("d", [v_exit]),
        )

    if profile_solver_step_m <= 1e-9:
        profile_solver_step_m = 1.0

    x = 0.0
    t = 0.0
    # These samples are retained by each FastProfile cache entry.  Keep the
    # same IEEE-754 doubles and append order without allocating a Python float
    # object plus list pointer for every metre of every cached profile.
    sample_dist = array("d", [0.0])
    sample_time = array("d", [0.0])
    sample_speed = array("d", [v])
    dist_acc = 0.0
    dist_dec = 0.0
    dist_cruise = 0.0
    time_acc = 0.0
    time_dec = 0.0
    time_cruise = 0.0
    v_peak = v

    while x < distance - 1e-9:
        step = profile_solver_step_m
        remaining = distance - x
        if step > remaining:
            step = remaining
        brake_needed = (v * v - v_exit * v_exit) / (2.0 * brake)
        if brake_needed < 0.0:
            brake_needed = 0.0

        if brake_needed >= max(0.0, remaining - step * 0.5):
            accel = -brake
            phase = 2
        elif v >= v_cap - 1e-9:
            accel = 0.0
            phase = 1
        else:
            accel = _effective_accel_rate(accel_ms2, v, drag_coeff)
            phase = 0

        v2 = v * v + 2.0 * accel * step
        if v2 < 0.0:
            v2 = 0.0
        v_next = sqrt(v2)
        if accel >= 0.0 and v_next > v_cap:
            v_next = v_cap
        avg_v = 0.5 * (v + v_next)
        if avg_v < 0.1:
            avg_v = 0.1
        dt = step / avg_v

        x += step
        t += dt
        v = v_next
        if v < 0.5:
            v = 0.5
        if v > v_peak:
            v_peak = v
        sample_dist.append(float(x))
        sample_time.append(float(t))
        sample_speed.append(float(v))

        if phase == 0:
            dist_acc += step
            time_acc += dt
        elif phase == 2:
            dist_dec += step
            time_dec += dt
        else:
            dist_cruise += step
            time_cruise += dt

    if sample_speed:
        sample_speed[len(sample_speed) - 1] = float(v_exit)

    if t < 0.001:
        t = 0.001
    return (
        float(t),
        float(v_peak),
        float(dist_acc),
        float(dist_cruise),
        float(time_acc),
        float(time_cruise),
        sample_dist,
        sample_time,
        sample_speed,
    )


cpdef dict segment_motion_profile(
    double distance_m,
    double v_entry_ms,
    double v_exit_ms,
    double v_cap_ms,
    double accel_ms2,
    double brake_ms2,
    double drag_coeff,
    double profile_solver_step_m=1.0,
):
    cdef double distance, brake, v, v_exit, v_cap
    cdef double x, t, dist_acc, dist_dec, dist_cruise
    cdef double time_acc, time_dec, time_cruise, v_peak
    cdef double step, remaining, brake_needed, accel
    cdef double v2, v_next, avg_v, dt
    cdef int phase
    cdef list sample_dist, sample_time, sample_speed

    distance = distance_m
    if distance < 0.0:
        distance = 0.0
    brake = brake_ms2
    if brake < 0.05:
        brake = 0.05
    v = v_entry_ms
    if v < 0.5:
        v = 0.5
    v_exit = v_exit_ms
    if v_exit < 0.5:
        v_exit = 0.5
    v_cap = v_cap_ms
    if v_cap < 0.5:
        v_cap = 0.5

    if distance <= 1e-9:
        return {
            "duration": 0.001,
            "v_peak_ms": v if v > v_exit else v_exit,
            "dist_acc_m": 0.0,
            "dist_cruise_m": 0.0,
            "dist_dec_m": 0.0,
            "time_acc_s": 0.0,
            "time_cruise_s": 0.0,
            "time_dec_s": 0.0,
            "sample_dist_m": [0.0],
            "sample_time_s": [0.0],
            "sample_speed_ms": [v_exit],
        }

    if profile_solver_step_m <= 1e-9:
        profile_solver_step_m = 1.0

    x = 0.0
    t = 0.0
    sample_dist = [0.0]
    sample_time = [0.0]
    sample_speed = [v]
    dist_acc = 0.0
    dist_dec = 0.0
    dist_cruise = 0.0
    time_acc = 0.0
    time_dec = 0.0
    time_cruise = 0.0
    v_peak = v

    while x < distance - 1e-9:
        step = profile_solver_step_m
        remaining = distance - x
        if step > remaining:
            step = remaining
        brake_needed = (v * v - v_exit * v_exit) / (2.0 * brake)
        if brake_needed < 0.0:
            brake_needed = 0.0

        if brake_needed >= max(0.0, remaining - step * 0.5):
            accel = -brake
            phase = 2
        elif v >= v_cap - 1e-9:
            accel = 0.0
            phase = 1
        else:
            accel = _effective_accel_rate(accel_ms2, v, drag_coeff)
            phase = 0

        v2 = v * v + 2.0 * accel * step
        if v2 < 0.0:
            v2 = 0.0
        v_next = sqrt(v2)
        if accel >= 0.0 and v_next > v_cap:
            v_next = v_cap
        avg_v = 0.5 * (v + v_next)
        if avg_v < 0.1:
            avg_v = 0.1
        dt = step / avg_v

        x += step
        t += dt
        v = v_next
        if v < 0.5:
            v = 0.5
        if v > v_peak:
            v_peak = v
        sample_dist.append(float(x))
        sample_time.append(float(t))
        sample_speed.append(float(v))

        if phase == 0:
            dist_acc += step
            time_acc += dt
        elif phase == 2:
            dist_dec += step
            time_dec += dt
        else:
            dist_cruise += step
            time_cruise += dt

    if sample_speed:
        sample_speed[len(sample_speed) - 1] = float(v_exit)

    if t < 0.001:
        t = 0.001
    return {
        "duration": float(t),
        "v_peak_ms": float(v_peak),
        "dist_acc_m": float(dist_acc),
        "dist_cruise_m": float(dist_cruise),
        "dist_dec_m": float(dist_dec),
        "time_acc_s": float(time_acc),
        "time_cruise_s": float(time_cruise),
        "time_dec_s": float(time_dec),
        "sample_dist_m": sample_dist,
        "sample_time_s": sample_time,
        "sample_speed_ms": sample_speed,
    }


cpdef list solve_node_speeds(
    object caps_rot_obj,
    object dist_rot_obj,
    double accel_base_scaled,
    double braking_rate,
    double drag_coeff,
    double forward_solver_step_m=1.0,
    int passes=12,
):
    cdef bint converged
    cdef double[:] previous
    cdef object previous_arr
    cdef int n, i, p
    cdef double[:] caps
    cdef double[:] dist
    cdef double[:] node
    cdef double v_in, v_out_max, v_out, v_in_max, candidate
    cdef object caps_arr, dist_arr, node_arr

    n = len(caps_rot_obj)
    if n <= 0:
        return []

    caps_arr = array("d", [0.0]) * n
    dist_arr = array("d", [0.0]) * n
    for i in range(n):
        caps_arr[i] = max(0.5, float(caps_rot_obj[i]))
        dist_arr[i] = max(0.0, float(dist_rot_obj[i]))

    node_arr = array("d", [caps_arr[0]]) * (n + 1)
    caps = caps_arr
    dist = dist_arr
    node = node_arr
    previous_arr = array("d", [0.0]) * (n+1)
    previous = previous_arr

    if passes <= 0:
        passes = 1

    for p in range(passes):
        for i in range(n+1):
            previous[i] = node[i]
        if node[0] > caps[0]:
            node[0] = caps[0]
        for i in range(n):
            v_in = node[i] if node[i] < caps[i] else caps[i]
            v_out_max = forward_exit_speed(
                v_in,
                dist[i],
                caps[i],
                accel_base_scaled,
                drag_coeff,
                forward_solver_step_m,
            )
            node[i + 1] = caps[i] if caps[i] < v_out_max else v_out_max
        candidate = node[n]
        if node[0] < candidate:
            candidate = node[0]
        if caps[0] < candidate:
            candidate = caps[0]
        node[n] = candidate
        for i in range(n - 1, -1, -1):
            v_out = node[i + 1] if node[i + 1] < caps[i] else caps[i]
            candidate = (v_out * v_out) + (2.0 * braking_rate * dist[i])
            if candidate < 0.0:
                candidate = 0.0
            v_in_max = sqrt(candidate)
            candidate = node[i]
            if caps[i] < candidate:
                candidate = caps[i]
            if v_in_max < candidate:
                candidate = v_in_max
            node[i] = candidate
        if node[n] > node[0]:
            node[n] = node[0]

        converged = True
        for i in range(n+1):
            if previous[i] != node[i]:
                converged = False
                break
        if converged:
            break

    return [float(v) for v in node_arr]


cpdef double overlap_fraction(
    double start,
    double end,
    double win_start,
    double win_end,
):
    cdef double lo = start if start > win_start else win_start
    cdef double hi = end if end < win_end else win_end
    cdef double width = win_end - win_start
    if width < 1e-9:
        width = 1e-9
    if hi <= lo + 1e-12:
        return 0.0
    return _clamp_double((hi - lo) / width, 0.0, 1.0)


cdef tuple _ers_apply_progress_segment_inner(
    double seg_start,
    double seg_end,
    object deploy_windows_obj,
    object harvest_windows_obj,
    double deploy_target,
    double harvest_target,
    double capacity,
    double used,
    double harvested,
    double charge,
):
    cdef object window
    cdef double frac, requested, remaining_target, room, actual, weight
    cdef double win_start, win_end

    if seg_end <= seg_start + 1e-12:
        return (charge, used, harvested)

    if deploy_target > 1e-9:
        for window in deploy_windows_obj:
            try:
                win_start = float(window.get("start", 0.0) or 0.0)
                win_end = float(window.get("end", 0.0) or 0.0)
                weight = float(window.get("weight", 0.0) or 0.0)
            except Exception:
                continue
            frac = overlap_fraction(seg_start, seg_end, win_start, win_end)
            if frac <= 1e-12:
                continue
            requested = deploy_target * weight * frac
            if requested <= 1e-12:
                continue
            remaining_target = deploy_target - used
            if remaining_target < 0.0:
                remaining_target = 0.0
            actual = requested
            if remaining_target < actual:
                actual = remaining_target
            if charge < actual:
                actual = charge
            if actual <= 1e-12:
                continue
            used += actual
            charge -= actual

    if harvest_target > 1e-9:
        for window in harvest_windows_obj:
            try:
                win_start = float(window.get("start", 0.0) or 0.0)
                win_end = float(window.get("end", 0.0) or 0.0)
                weight = float(window.get("weight", 0.0) or 0.0)
            except Exception:
                continue
            frac = overlap_fraction(seg_start, seg_end, win_start, win_end)
            if frac <= 1e-12:
                continue
            requested = harvest_target * weight * frac
            if requested <= 1e-12:
                continue
            remaining_target = harvest_target - harvested
            if remaining_target < 0.0:
                remaining_target = 0.0
            room = capacity - charge
            if room < 0.0:
                room = 0.0
            actual = requested
            if remaining_target < actual:
                actual = remaining_target
            if room < actual:
                actual = room
            if actual <= 1e-12:
                continue
            harvested += actual
            charge += actual

    return (charge, used, harvested)


cpdef tuple ers_apply_progress_segment(
    double start_progress,
    double end_progress,
    object deploy_windows_obj,
    object harvest_windows_obj,
    double deploy_target,
    double harvest_target,
    double capacity,
    double used,
    double harvested,
    double charge,
):
    if start_progress <= end_progress:
        return _ers_apply_progress_segment_inner(
            start_progress,
            end_progress,
            deploy_windows_obj,
            harvest_windows_obj,
            deploy_target,
            harvest_target,
            capacity,
            used,
            harvested,
            charge,
        )

    charge, used, harvested = _ers_apply_progress_segment_inner(
        start_progress,
        1.0,
        deploy_windows_obj,
        harvest_windows_obj,
        deploy_target,
        harvest_target,
        capacity,
        used,
        harvested,
        charge,
    )
    return _ers_apply_progress_segment_inner(
        0.0,
        end_progress,
        deploy_windows_obj,
        harvest_windows_obj,
        deploy_target,
        harvest_target,
        capacity,
        used,
        harvested,
        charge,
    )


cdef tuple _ers_deploy_overlap_inner(
    double seg_start,
    double seg_end,
    object deploy_windows_obj,
    double overlap_len,
    double total_len,
    object sample_progress,
):
    cdef object window
    cdef double seg_len, win_start, win_end, ov_start, ov_end

    seg_len = seg_end - seg_start
    if seg_len <= 1e-12:
        return (overlap_len, total_len, sample_progress)
    total_len += seg_len
    for window in deploy_windows_obj:
        try:
            win_start = float(window.get("start", 0.0) or 0.0)
            win_end = float(window.get("end", 0.0) or 0.0)
        except Exception:
            continue
        ov_start = seg_start if seg_start > win_start else win_start
        ov_end = seg_end if seg_end < win_end else win_end
        if ov_end <= ov_start + 1e-12:
            continue
        overlap_len += ov_end - ov_start
        if sample_progress is None:
            sample_progress = 0.5 * (ov_start + ov_end)
    return (overlap_len, total_len, sample_progress)


cpdef tuple ers_deploy_overlap_for_segment(
    double start_progress,
    double end_progress,
    object deploy_windows_obj,
    double deploy_target,
    double used,
    double charge,
):
    cdef double total_len = 0.0
    cdef double overlap_len = 0.0
    cdef object sample_progress = None
    if deploy_target <= 1e-9 or used >= deploy_target - 1e-9 or charge <= 1e-9:
        return (0.0, None)
    if start_progress <= end_progress:
        overlap_len, total_len, sample_progress = _ers_deploy_overlap_inner(
            start_progress,
            end_progress,
            deploy_windows_obj,
            overlap_len,
            total_len,
            sample_progress,
        )
    else:
        overlap_len, total_len, sample_progress = _ers_deploy_overlap_inner(
            start_progress,
            1.0,
            deploy_windows_obj,
            overlap_len,
            total_len,
            sample_progress,
        )
        overlap_len, total_len, sample_progress = _ers_deploy_overlap_inner(
            0.0,
            end_progress,
            deploy_windows_obj,
            overlap_len,
            total_len,
            sample_progress,
        )
    if total_len <= 1e-12 or overlap_len <= 1e-12:
        return (0.0, None)
    return (_clamp_double(overlap_len / total_len, 0.0, 1.0), sample_progress)


cpdef double ers_bonus_points_for_progress(
    double progress,
    object deploy_windows_obj,
    double deploy_target,
    double used,
    double charge,
    double boost_points,
):
    cdef object window
    cdef double prog, win_start, win_end
    if deploy_target <= 1e-9 or used >= deploy_target - 1e-9 or charge <= 1e-9:
        return 0.0
    prog = _clamp_double(progress, 0.0, 1.0)
    for window in deploy_windows_obj:
        try:
            win_start = float(window.get("start", 0.0) or 0.0)
            win_end = float(window.get("end", 0.0) or 0.0)
        except Exception:
            continue
        if prog >= win_start and prog <= win_end:
            return 0.0 if boost_points <= 0.0 else boost_points
    return 0.0
