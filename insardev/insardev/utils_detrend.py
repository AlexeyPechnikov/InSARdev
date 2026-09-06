# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------
"""
Static utility functions for detrending operations.

These functions contain the core algorithms for 1D and 2D polynomial
trend fitting.
"""
import numpy as np
import numba as nb


def _warmup_numba_cache():
    """Compile numba kernels once in the main process so dask workers load from cache."""
    _c = np.zeros((3, 1), dtype=np.complex64)
    _w = np.ones((1, 1), dtype=np.float32)
    _d = np.array([-1.0, 0.0, 1.0])
    _trend1d_numba_kernel(_c, _w, _d, True, True, True, False, 128)
    _wf = np.ones((3, 1), dtype=np.float32)
    _threshold_pairs_numba_kernel(_c, _wf, 1, 3, np.pi * 0.5)
    _trend1d_pairs_numba_kernel(
        _c, _wf, 1, 2, 3,
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1.0, -1.0, 0.5]),
        np.array([1.0, -1.0, 1.0]),
        np.array([0, 2, 3], dtype=np.int64),
        np.array([0, 0, 1], dtype=np.int64),
        np.array([1, 1, 0], dtype=np.int64),
        np.array([1.0, 1.0, 2.0]),  # pair_dt
        np.array([0.0, 0.5, 1.0]),  # date_days_norm (real times)
        0, True,
    )
    # the gridded transform's spreaders, one call per rank: a worker that has
    # to compile them itself does it while every other worker compiles the
    # same thing into the same cache
    for _k in (1, 2, 3):
        trend2d_spread(np.ones((1, 1), np.complex128),
                       np.zeros((1, _k)), 8)


@nb.njit(cache=True)
def _threshold_pairs_numba_kernel(
    data_flat,         # (n_pairs, n_pixels) complex64/128
    weight_flat,       # (n_pairs, n_pixels) float32
    n_pixels,
    n_pairs,
    threshold,         # cstd threshold in radians
):
    """Per-pixel weighted cstd check. Returns mask: True = keep, False = reject."""
    mask = np.zeros(n_pixels, dtype=nb.boolean)
    for px in range(n_pixels):
        wcos = 0.0; wsin = 0.0; wsum = 0.0
        for p in range(n_pairs):
            c = data_flat[p, px]
            re = np.float64(c.real)
            im = np.float64(c.imag)
            ang = np.arctan2(im, re)
            if (re == 0.0 and im == 0.0) or not np.isfinite(ang):
                continue
            pw = np.float64(weight_flat[p, px])
            if pw <= 0.0:
                continue
            wcos += pw * np.cos(ang)
            wsin += pw * np.sin(ang)
            wsum += pw
        if wsum < 1e-10:
            continue
        R = np.sqrt((wcos / wsum)**2 + (wsin / wsum)**2)
        if R < 1e-10:
            continue
        R = min(R, 1 - 1e-10)
        if np.sqrt(-2.0 * np.log(R)) < threshold:
            mask[px] = True
    return mask


def threshold_pairs_array(data_chunk, weight_chunk, threshold=np.pi/2):
    """Apply cstd threshold to complex pair data. Returns filtered copy.

    Pixels with weighted cstd >= threshold have all pairs set to 0+0j.
    """
    import numpy as np

    if isinstance(data_chunk, list):
        data_np = np.asarray(data_chunk[0]) if len(data_chunk) == 1 else np.concatenate([np.asarray(c) for c in data_chunk], axis=0)
    else:
        data_np = np.asarray(data_chunk)

    if data_np.ndim == 2:
        n_pairs, nx = data_np.shape
        ny = 1
        data_np = data_np.reshape(n_pairs, ny, nx)
    else:
        n_pairs, ny, nx = data_np.shape
    n_pixels = ny * nx

    data_flat = np.ascontiguousarray(data_np.reshape(n_pairs, n_pixels))

    if isinstance(weight_chunk, list):
        weight_np = np.asarray(weight_chunk[0]) if len(weight_chunk) == 1 else np.concatenate([np.asarray(c) for c in weight_chunk], axis=0)
    elif weight_chunk is not None:
        weight_np = np.asarray(weight_chunk)
    else:
        weight_np = None

    if weight_np is not None:
        weight_flat = np.ascontiguousarray(weight_np.reshape(n_pairs, n_pixels).astype(np.float32))
        weight_flat[~np.isfinite(weight_flat)] = 0.0
        weight_flat[weight_flat < 0] = 0.0
    else:
        weight_flat = np.ones((n_pairs, n_pixels), dtype=np.float32)
    del weight_np

    mask = _threshold_pairs_numba_kernel(data_flat, weight_flat, n_pixels, n_pairs, threshold)

    # NaN rejected pixels
    result = data_np.copy()
    mask_2d = mask.reshape(ny, nx)
    nan_val = np.complex64(np.nan + 0j)
    for iy in range(ny):
        for ix in range(nx):
            if not mask_2d[iy, ix]:
                result[:, iy, ix] = nan_val
    return result


@nb.njit(cache=True)
def _trend1d_pairs_numba_kernel(
    data_flat,         # (n_pairs, n_pixels) complex128 or float64
    weight_flat,       # (n_pairs, n_pixels) float32 — correlation weights
    n_pixels,
    n_dates,
    n_pairs,
    date_pair_flat,    # flattened pair indices
    date_time_flat,    # flattened time values (normalized)
    date_sign_flat,    # flattened signs
    date_offsets,      # (n_dates+1,) start offsets into flat arrays
    pair_ref_didx,     # (n_pairs,) ref date index
    pair_rep_didx,     # (n_pairs,) rep date index
    pair_dt,           # (n_pairs,) temporal baseline in intervals (unnormalized)
    date_days_norm,    # (n_dates,) REAL acquisition times normalized to [0, 1]
    max_refine,
    is_complex=True,   # True for wrapped (complex), False for unwrapped (real)
):
    """Per-pixel atmospheric phase estimation using global velocity derotation
    + weighted circular mean.

    1. Global velocity: 16-bin periodogram on all pairs vs temporal baseline.
    2. Derotate pair phases by global velocity.
    3. Per-date weighted circular mean of derotated signed phases (iterative).
    4. Output trend = atmospheric model only (velocity preserved in detrended data).

    Returns
    -------
    trend : (n_pairs, n_pixels) complex64
        Per-pair atmospheric trend. NaN where input is invalid.
    """
    model_angles = np.zeros((n_dates, n_pixels), dtype=np.float64)
    trend = np.full((n_pairs, n_pixels), np.nan + 0j, dtype=np.complex64)

    pixel_angles = np.empty(n_pairs, dtype=np.float64)
    pixel_weights = np.empty(n_pairs, dtype=np.float64)

    for px in range(n_pixels):
        # Extract angles and correlation weights in float64
        if is_complex:
            for p in range(n_pairs):
                c = data_flat[p, px]
                re = np.float64(c.real)
                im = np.float64(c.imag)
                ang = np.arctan2(im, re)
                if (re == 0.0 and im == 0.0) or not np.isfinite(ang):
                    pixel_angles[p] = np.nan
                    pixel_weights[p] = 0.0
                else:
                    pixel_angles[p] = ang
                    pw = np.float64(weight_flat[p, px])
                    pixel_weights[p] = pw if pw > 0.0 else 0.0
        else:
            for p in range(n_pairs):
                pixel_angles[p] = data_flat[p, px].real
                pw = np.float64(weight_flat[p, px])
                pixel_weights[p] = pw if pw > 0.0 else 0.0

        corrected = np.empty(n_pairs, dtype=np.float64)
        local_models = np.zeros(n_dates, dtype=np.float64)

        # Global velocity estimation: periodogram on all pairs vs temporal
        # baseline. Removes the dominant velocity trend so per-date periodogram
        # only needs to find residual (seasonal/nonlinear) slope + atmospheric.
        global_v = 0.0
        if is_complex:
            n_valid_gv = 0
            for p in range(n_pairs):
                if np.isfinite(pixel_angles[p]) and pixel_weights[p] > 0.0:
                    n_valid_gv += 1
            if n_valid_gv >= 4:
                # Multi-level periodogram (16 coarse + 16 fine = level 1).
                # Range: π/2 per shortest baseline (unambiguous for noisy phase).
                gv_dt_min = 1e30
                for p in range(n_pairs):
                    if np.isfinite(pixel_angles[p]) and pixel_weights[p] > 0.0:
                        adt = abs(pair_dt[p])
                        if adt > 1e-10 and adt < gv_dt_min:
                            gv_dt_min = adt
                if gv_dt_min > 1e20:
                    gv_dt_min = 1.0
                gv_range = (np.pi * 0.5) / gv_dt_min
                # symmetric grid: both endpoints scanned (see velocity kernel)
                gv_step = 2.0 * gv_range / 15
                best_gS = -1.0
                best_gv = 0.0
                scan_lo = -gv_range
                for level in range(1 + max_refine):
                    for bi in range(16):
                        v_try = scan_lo + gv_step * bi
                        sr = 0.0; si = 0.0
                        for p in range(n_pairs):
                            if not (np.isfinite(pixel_angles[p]) and pixel_weights[p] > 0.0):
                                continue
                            ang = pixel_angles[p] - v_try * pair_dt[p]
                            ang = ang - 2.0 * np.pi * np.floor((ang + np.pi) / (2.0 * np.pi))
                            sr += pixel_weights[p] * np.cos(ang)
                            si += pixel_weights[p] * np.sin(ang)
                        S = sr * sr + si * si
                        if S > best_gS:
                            best_gS = S; best_gv = v_try
                    scan_lo = best_gv - gv_step
                    gv_step = 2.0 * gv_step / 15
                global_v = best_gv

        # Single-pass atmospheric fit: derotate by velocity, then per-date circular mean.
        for p in range(n_pairs):
            corrected[p] = pixel_angles[p] - global_v * pair_dt[p]

        # Fit each date
        for d in range(n_dates):
                d_start = date_offsets[d]
                d_end = date_offsets[d + 1]
                n_d = d_end - d_start
                if n_d < 4:
                    continue

                # Count valid pairs
                n_valid = 0
                for k in range(n_d):
                    pidx = date_pair_flat[d_start + k]
                    val = corrected[pidx] * date_sign_flat[d_start + k]
                    if np.isfinite(val) and pixel_weights[pidx] > 0.0:
                        n_valid += 1
                if n_valid < 4:
                    continue

                # Prepare per-pair arrays for this date
                phases = np.empty(n_d, dtype=np.float64)
                t_vals = np.empty(n_d, dtype=np.float64)
                valid = np.empty(n_d, dtype=nb.boolean)
                w_irls = np.empty(n_d, dtype=np.float64)

                for k in range(n_d):
                    pidx = date_pair_flat[d_start + k]
                    val = corrected[pidx] * date_sign_flat[d_start + k]
                    phases[k] = val
                    t_vals[k] = date_time_flat[d_start + k]
                    pw = pixel_weights[pidx]
                    valid[k] = np.isfinite(val) and pw > 0.0
                    if valid[k]:
                        w_irls[k] = pw  # correlation as initial IRLS weight
                    else:
                        phases[k] = 0.0
                        w_irls[k] = 0.0

                # Per-date periodogram search (with global velocity already removed).
                # Finds residual slope (seasonal/nonlinear) + atmospheric intercept.
                if is_complex:
                    # Search range π/4: after global velocity removal, residual
                    # slope is from seasonal variations only. π/4 = π/2 (noisy
                    # limit) / 2 (two dates per pair) — the maximum stable slope.
                    b_range = np.pi * 0.25
                    n_scan = 32
                    scan_step = 2.0 * b_range / n_scan
                    # Precompute cos/sin of phases
                    cos_ph = np.empty(n_d, dtype=np.float64)
                    sin_ph = np.empty(n_d, dtype=np.float64)
                    for k in range(n_d):
                        if valid[k]:
                            cos_ph[k] = np.cos(phases[k])
                            sin_ph[k] = np.sin(phases[k])
                        else:
                            cos_ph[k] = 0.0; sin_ph[k] = 0.0
                    # Coarse scan with trig recurrence
                    cos_step = np.empty(n_d, dtype=np.float64)
                    sin_step = np.empty(n_d, dtype=np.float64)
                    cos_cur = np.empty(n_d, dtype=np.float64)
                    sin_cur = np.empty(n_d, dtype=np.float64)
                    b0 = -b_range
                    for k in range(n_d):
                        if valid[k]:
                            st = scan_step * t_vals[k]
                            cos_step[k] = np.cos(st); sin_step[k] = np.sin(st)
                            bt = b0 * t_vals[k]
                            cos_cur[k] = np.cos(bt); sin_cur[k] = np.sin(bt)
                        else:
                            cos_step[k] = 1.0; sin_step[k] = 0.0
                            cos_cur[k] = 1.0; sin_cur[k] = 0.0
                    best_S = -1.0; best_b = 0.0; best_a = 0.0
                    for bi in range(n_scan):
                        sr = 0.0; si = 0.0
                        for k in range(n_d):
                            if not valid[k]: continue
                            sr += w_irls[k] * (cos_ph[k]*cos_cur[k] + sin_ph[k]*sin_cur[k])
                            si += w_irls[k] * (sin_ph[k]*cos_cur[k] - cos_ph[k]*sin_cur[k])
                        S = sr*sr + si*si
                        if S > best_S:
                            best_S = S; best_b = b0 + scan_step*bi
                            best_a = np.arctan2(si, sr)
                        for k in range(n_d):
                            if valid[k]:
                                c = cos_cur[k]*cos_step[k] - sin_cur[k]*sin_step[k]
                                s = sin_cur[k]*cos_step[k] + cos_cur[k]*sin_step[k]
                                cos_cur[k] = c; sin_cur[k] = s
                    # Fine refinement
                    fine_step = 2.0 * scan_step / n_scan
                    fine_lo = best_b - scan_step
                    for k in range(n_d):
                        if valid[k]:
                            st = fine_step * t_vals[k]
                            cos_step[k] = np.cos(st); sin_step[k] = np.sin(st)
                            bt = fine_lo * t_vals[k]
                            cos_cur[k] = np.cos(bt); sin_cur[k] = np.sin(bt)
                    for bi in range(n_scan):
                        sr = 0.0; si = 0.0
                        for k in range(n_d):
                            if not valid[k]: continue
                            sr += w_irls[k] * (cos_ph[k]*cos_cur[k] + sin_ph[k]*sin_cur[k])
                            si += w_irls[k] * (sin_ph[k]*cos_cur[k] - cos_ph[k]*sin_cur[k])
                        S = sr*sr + si*si
                        if S > best_S:
                            best_S = S; best_b = fine_lo + fine_step*bi
                            best_a = np.arctan2(si, sr)
                        for k in range(n_d):
                            if valid[k]:
                                c = cos_cur[k]*cos_step[k] - sin_cur[k]*sin_step[k]
                                s = sin_cur[k]*cos_step[k] + cos_cur[k]*sin_step[k]
                                cos_cur[k] = c; sin_cur[k] = s
                    c0 = best_a - 2.0 * np.pi * np.floor((best_a + np.pi) / (2.0 * np.pi))
                else:
                    wsum = 0.0; wval = 0.0
                    for k in range(n_d):
                        if valid[k]:
                            wsum += w_irls[k]; wval += w_irls[k] * phases[k]
                    c0 = wval / (wsum + 1e-30)

                local_models[d] = c0

        # Remove linear trend from per-date models — prevents atmospheric
        # model from absorbing net deformation after global velocity removal.
        # Uses periodogram on models vs REAL acquisition time (NOT date index:
        # acquisition intervals vary — e.g. a 36-day gap in a 12-day sequence —
        # and a physically linear-in-time ramp is kinked in index space, so an
        # index-based fit mis-removes it). Handles wrapping (models near ±π).
        if is_complex and n_dates > 2:
            d_arr = date_days_norm
            # Periodogram: find slope of models vs date index
            # Search b ∈ [-π/4, π/4] (same limit as per-date slopes)
            mt_range = np.pi * 0.25
            mt_scan = 16
            # symmetric grid: both endpoints scanned (see velocity kernel)
            mt_step = 2.0 * mt_range / (mt_scan - 1)
            mt_best_S = -1.0; mt_best_b = 0.0; mt_best_a = 0.0
            for bi in range(mt_scan):
                b_try = -mt_range + mt_step * bi
                sr = 0.0; si = 0.0
                for d in range(n_dates):
                    ang = local_models[d] - b_try * d_arr[d]
                    ang = ang - 2.0 * np.pi * np.floor((ang + np.pi) / (2.0 * np.pi))
                    sr += np.cos(ang); si += np.sin(ang)
                S = sr * sr + si * si
                if S > mt_best_S:
                    mt_best_S = S; mt_best_b = b_try; mt_best_a = np.arctan2(si, sr)
            # Fine
            mt_fine_lo = mt_best_b - mt_step
            mt_fine_step = 2.0 * mt_step / mt_scan
            for bi in range(mt_scan):
                b_try = mt_fine_lo + mt_fine_step * bi
                sr = 0.0; si = 0.0
                for d in range(n_dates):
                    ang = local_models[d] - b_try * d_arr[d]
                    ang = ang - 2.0 * np.pi * np.floor((ang + np.pi) / (2.0 * np.pi))
                    sr += np.cos(ang); si += np.sin(ang)
                S = sr * sr + si * si
                if S > mt_best_S:
                    mt_best_S = S; mt_best_b = b_try; mt_best_a = np.arctan2(si, sr)
            # Subtract trend: model[d] -= (a + b*d), wrapped
            for d in range(n_dates):
                correction = mt_best_a + mt_best_b * d_arr[d]
                local_models[d] = local_models[d] - correction
                local_models[d] = local_models[d] - 2.0 * np.pi * np.floor(
                    (local_models[d] + np.pi) / (2.0 * np.pi))

        for d in range(n_dates):
            model_angles[d, px] = local_models[d]

        # Reconstruct per-pair trend (atmospheric only)
        for p in range(n_pairs):
            if np.isfinite(pixel_angles[p]):
                diff = local_models[pair_ref_didx[p]] - local_models[pair_rep_didx[p]]
                if is_complex:
                    trend[p, px] = np.complex64(np.exp(1j * diff))
                else:
                    trend[p, px] = np.complex64(diff)

    return trend, model_angles


@nb.njit(cache=True)
def _trend1d_numba_kernel(
    data_flat,      # (n_samples, n_pixels) complex128 or float64
    w_flat,         # (n_samples, n_pixels) float32 or None-like
    dim_norm,       # (n_samples,) float64  — normalized dim values
    intercept,      # bool — include intercept in output
    slope,          # bool — include slope in output
    is_complex,      # bool — True for wrapped (complex) phase, False for unwrapped (real)
    has_weight,     # bool — True if w_flat contains real weights, False if unit weights
    bins,           # int — periodogram bins (0=skip periodogram, use circular mean init)
):
    """Per-pixel IRLS linear fitting for detrend1d.

    For wrapped (complex) phase: periodogram init finds the slope globally
    (handles multi-cycle wrapping), then IRLS refines from that init.
    bins controls the periodogram search: range = bins/2, step = 1 rad.
    bins=256 covers DEM errors up to ~500m for C-band Sentinel-1.

    Analytical 2x2 weighted least squares solve per pixel:
    y = a + b*t, 5 accumulators (sw, swt, swt2, swy, swty), Cramer's rule.

    Returns
    -------
    result : (n_samples, n_pixels) complex64 if is_complex, else float32.
        Complex: unit-magnitude trend exp(1j*fit). Real: fitted values.
    slopes : (n_pixels,) float64
        Fitted slope c1 per pixel in normalized dim units. NaN where invalid.
    """
    n_samples, n_pixels = data_flat.shape
    result = np.full((n_samples, n_pixels), np.nan + 0j, dtype=np.complex64)
    slopes = np.full(n_pixels, np.nan, dtype=np.float64)

    # Per-pixel working arrays (reused across pixels)
    angles = np.empty(n_samples, dtype=np.float64)
    w_irls = np.empty(n_samples, dtype=np.float64)
    valid = np.empty(n_samples, dtype=nb.boolean)

    for px in range(n_pixels):
        # Extract angles per-pixel from complex input, or use values directly
        n_valid = 0
        if is_complex:
            for s in range(n_samples):
                c = data_flat[s, px]
                re = np.float64(c.real)
                im = np.float64(c.imag)
                if re == 0.0 and im == 0.0:
                    angles[s] = np.nan
                    valid[s] = False
                else:
                    a = np.arctan2(im, re)
                    if np.isfinite(a):
                        angles[s] = a
                        valid[s] = True
                        n_valid += 1
                    else:
                        angles[s] = np.nan
                        valid[s] = False
        else:
            for s in range(n_samples):
                val = data_flat[s, px].real  # real input stored as complex with imag=0
                if np.isfinite(val):
                    angles[s] = val
                    valid[s] = True
                    n_valid += 1
                else:
                    angles[s] = np.nan
                    valid[s] = False

        if n_valid < 3:
            continue

        # Initialize IRLS weights
        for s in range(n_samples):
            if valid[s]:
                w_irls[s] = np.sqrt(w_flat[s, px]) if has_weight else 1.0
            else:
                w_irls[s] = 0.0

        # Init: periodogram (bins>0) or circular mean (bins=0)
        if is_complex and bins > 0:
            # Periodogram init — single-level scan with trig recurrence.
            # range = bins/2, step = 1 rad. Finds slope globally, IRLS refines.
            cos_ph = np.empty(n_samples, dtype=np.float64)
            sin_ph = np.empty(n_samples, dtype=np.float64)
            for s in range(n_samples):
                if valid[s]:
                    cos_ph[s] = np.cos(angles[s])
                    sin_ph[s] = np.sin(angles[s])
                else:
                    cos_ph[s] = 0.0; sin_ph[s] = 0.0

            p_range = 0.5 * bins
            p_step = 2.0 * p_range / bins  # = 1.0
            scan_lo = -p_range
            best_S = -1.0; best_b = 0.0; best_a = 0.0

            # Precompute step and initial rotations per sample
            p_cos_step = np.empty(n_samples, dtype=np.float64)
            p_sin_step = np.empty(n_samples, dtype=np.float64)
            p_cos_cur = np.empty(n_samples, dtype=np.float64)
            p_sin_cur = np.empty(n_samples, dtype=np.float64)
            for s in range(n_samples):
                if valid[s]:
                    st = p_step * dim_norm[s]
                    p_cos_step[s] = np.cos(st); p_sin_step[s] = np.sin(st)
                    bt = scan_lo * dim_norm[s]
                    p_cos_cur[s] = np.cos(bt); p_sin_cur[s] = np.sin(bt)
                else:
                    p_cos_step[s] = 1.0; p_sin_step[s] = 0.0
                    p_cos_cur[s] = 1.0; p_sin_cur[s] = 0.0

            for bi in range(bins):
                sr = 0.0; si = 0.0
                for s in range(n_samples):
                    if not valid[s]: continue
                    sr += w_irls[s] * (cos_ph[s]*p_cos_cur[s] + sin_ph[s]*p_sin_cur[s])
                    si += w_irls[s] * (sin_ph[s]*p_cos_cur[s] - cos_ph[s]*p_sin_cur[s])
                S = sr*sr + si*si
                if S > best_S:
                    best_S = S; best_b = scan_lo + p_step*bi
                    best_a = np.arctan2(si, sr)
                # Trig recurrence: rotate by step
                for s in range(n_samples):
                    if valid[s]:
                        c = p_cos_cur[s]*p_cos_step[s] - p_sin_cur[s]*p_sin_step[s]
                        sn = p_sin_cur[s]*p_cos_step[s] + p_cos_cur[s]*p_sin_step[s]
                        p_cos_cur[s] = c; p_sin_cur[s] = sn

            c0 = best_a - 2.0 * np.pi * np.floor((best_a + np.pi) / (2.0 * np.pi))
            c1 = best_b
        elif is_complex:
            # Circular mean init (bins=0)
            re_sum = 0.0; im_sum = 0.0
            for s in range(n_samples):
                if valid[s]:
                    re_sum += np.float64(data_flat[s, px].real)
                    im_sum += np.float64(data_flat[s, px].imag)
            c0 = np.arctan2(im_sum, re_sum)
            c1 = 0.0
        else:
            c0 = 0.0
            c1 = 0.0

        epsilon = 0.1
        for irls_iter in range(10):
            sw = 0.0; swt = 0.0; swt2 = 0.0
            swy = 0.0; swty = 0.0
            max_dw = 0.0
            for s in range(n_samples):
                if not valid[s]:
                    continue
                t = dim_norm[s]
                fit_val = c0 + c1 * t
                if is_complex:
                    # Wrap residual, then "unwrap" around current model
                    res = angles[s] - fit_val
                    res = res - 2.0 * np.pi * np.floor((res + np.pi) / (2.0 * np.pi))
                    y = fit_val + res
                else:
                    y = angles[s]
                    res = y - fit_val
                w = w_irls[s]
                sw += w; swt += w * t; swt2 += w * t * t
                swy += w * y; swty += w * t * y

                base_w = np.sqrt(w_flat[s, px]) if has_weight else 1.0
                new_w = base_w / (abs(res) + epsilon)
                if new_w > 10.0 * base_w:
                    new_w = 10.0 * base_w
                dw = abs(new_w - w_irls[s])
                if dw > max_dw:
                    max_dw = dw
                w_irls[s] = new_w

            det = sw * swt2 - swt * swt + 1e-30
            c0 = (swt2 * swy - swt * swty) / det
            c1 = (sw * swty - swt * swy) / det

            if max_dw < 1e-3:
                break

        # Store slope (normalized units — caller denormalizes)
        slopes[px] = c1

        # Write final values directly as output type
        for s in range(n_samples):
            t = dim_norm[s]
            if not intercept and not slope:
                fit_val = 0.0
            elif not intercept:
                fit_val = c1 * t
            elif not slope:
                fit_val = c0
            else:
                fit_val = c0 + c1 * t
            if is_complex:
                result[s, px] = np.complex64(np.exp(1j * fit_val))
            else:
                result[s, px] = np.complex64(fit_val)

    return result, slopes


def trend1d_array(data, dim_values, weight, intercept=True, slope=True, is_complex=True, bins=128):
    """
    Fit linear trend along first dimension at each (y, x) pixel.

    Passes data directly to numba kernel — no intermediate float64 arrays.
    Complex: kernel extracts angles per-pixel. Real: passes values through.

    Parameters
    ----------
    data : np.ndarray or list
        3D array (n_samples, y, x) — complex or real. Or list of chunk arrays.
    dim_values : np.ndarray
        1D array of x-values for fitting (length n_samples).
    weight : np.ndarray or list or None
        Weight array (real), same shape as data, or list of chunk arrays.
    intercept : bool
        If True, include intercept (constant term) in output. If False, zero it out.
    slope : bool
        If True, include slope in output. If False, zero it out.
    is_complex : bool
        If True (default), treat as complex wrapped phase. If False, treat as real unwrapped phase.

    Returns
    -------
    np.ndarray
        Fitted values, shape (n_samples, y, x).
        Complex: complex64 unit-magnitude trend. Real: float32 trend.
    """
    if isinstance(data, list):
        data = np.asarray(data[0]) if len(data) == 1 else np.concatenate([np.asarray(c) for c in data], axis=0)
    n_samples, ny, nx = data.shape
    n_pixels = ny * nx

    # Pass data directly to kernel — no intermediate float64 arrays
    if is_complex:
        data[data == 0] = np.nan + 0j
    data_flat = np.ascontiguousarray(data.reshape(n_samples, n_pixels))

    # Weights: pass raw float32 to kernel, sqrt done per-pixel inside
    if isinstance(weight, list):
        weight = np.asarray(weight[0]) if len(weight) == 1 else np.concatenate([np.asarray(c) for c in weight], axis=0)
    has_weight = weight is not None
    if has_weight:
        w_flat = weight.reshape(n_samples, n_pixels).astype(np.float32)
    else:
        w_flat = np.empty((1, 1), dtype=np.float32)  # dummy, not accessed

    # Normalize dim values
    dim_absmax = np.max(np.abs(dim_values))
    if dim_absmax > 0:
        dim_norm = (dim_values / dim_absmax).astype(np.float64)
    else:
        dim_norm = np.zeros(n_samples, dtype=np.float64)

    result, slopes_norm = _trend1d_numba_kernel(data_flat, w_flat, dim_norm,
                                    intercept, slope, is_complex, has_weight, bins)
    # Denormalize slope: kernel fits in normalized dim, convert to original units
    slopes_2d = (slopes_norm / dim_absmax).astype(np.float32).reshape(ny, nx) if dim_absmax > 0 \
        else np.full((ny, nx), np.nan, dtype=np.float32)

    if is_complex:
        return result.reshape(n_samples, ny, nx), slopes_2d
    else:
        return result.real.astype(np.float32).reshape(n_samples, ny, nx), slopes_2d


# Backward compatibility alias
regression1d_array = trend1d_array


def trend1d_pairs_array(data_chunk, weight_chunk, ref_values, rep_values,
                         max_refine=3, is_complex=True, return_models=False):
    """
    Estimate per-date atmospheric phase from interferometric network.

    For each unique date, gathers all pairs sharing that date, fits a
    linear model (intercept + slope) to phase vs temporal baseline using
    all pairs, and stores the model at zero temporal baseline (intercept).
    Pair trends are reconstructed as model[ref] - model[rep] (real) or
    model[ref] * conj(model[rep]) (complex).

    Iterative refinement (max_refine > 0): after the initial per-date fit,
    pair-wise corrections from accumulated models are subtracted from the
    original data, and the per-date fit is repeated on the corrected data.

    Uses Numba-compiled per-pixel parallel loop.

    Parameters
    ----------
    data_chunk : np.ndarray or list
        3D array (n_pairs, chunk_y, chunk_x) — complex or real.
    weight_chunk : np.ndarray or None
        Weight array (real), same shape as data_chunk.
    ref_values : np.ndarray
        1D array of ref dates as int64 (nanoseconds since epoch).
    rep_values : np.ndarray
        1D array of rep dates as int64 (nanoseconds since epoch).
    max_refine : int
        Maximum refinement iterations (0 = single-pass). Default 3.
    is_complex : bool
        If True (default), treat as complex wrapped phase. If False, treat as real unwrapped phase.

    Returns
    -------
    np.ndarray
        Trend array (n_pairs, chunk_y, chunk_x), complex64 or float32.
    """
    # Materialize data from chunk list (avoid copy for single chunk)
    if isinstance(data_chunk, list):
        data_np = np.asarray(data_chunk[0]) if len(data_chunk) == 1 else np.concatenate([np.asarray(c) for c in data_chunk], axis=0)
    else:
        data_np = np.asarray(data_chunk)

    n_pairs, ny, nx = data_np.shape

    if is_complex:
        # Convert 0+0j to NaN in-place (skipped dask blocks)
        data_np[data_np == 0] = np.nan + 0j

    out_dtype = np.complex64 if is_complex else np.float32
    n_pixels = ny * nx

    # Convert int64 nanoseconds to days
    ns_per_day = 86400 * 1e9
    ref_days = ref_values / ns_per_day
    rep_days = rep_values / ns_per_day
    unique_days = np.unique(np.concatenate([ref_days, rep_days]))
    n_dates = len(unique_days)

    # Build per-date info flattened for numba (ragged arrays → flat + offsets)
    day_to_idx = {d: i for i, d in enumerate(unique_days)}
    all_pairs, all_times, all_signs = [], [], []
    offsets = [0]
    for date_day in unique_days:
        is_ref = np.isclose(ref_days, date_day)
        is_rep = np.isclose(rep_days, date_day)
        mask = is_ref | is_rep
        pidx = np.where(mask)[0]
        for idx in pidx:
            all_pairs.append(idx)
            if is_rep[idx]:
                all_times.append(ref_days[idx] - date_day)
                all_signs.append(-1.0)
            else:
                all_times.append(rep_days[idx] - date_day)
                all_signs.append(1.0)
        offsets.append(len(all_pairs))

    # Normalize time per date
    all_times_np = np.array(all_times, dtype=np.float64)
    offsets_np = np.array(offsets, dtype=np.int64)
    for d in range(n_dates):
        s, e = offsets_np[d], offsets_np[d + 1]
        if e > s:
            t_absmax = np.max(np.abs(all_times_np[s:e]))
            if t_absmax > 0:
                all_times_np[s:e] /= t_absmax

    # Pair → date index mapping
    pair_ref_didx = np.array([day_to_idx[d] for d in
                              unique_days[np.searchsorted(unique_days, ref_days)]])
    pair_rep_didx = np.array([day_to_idx[d] for d in
                              unique_days[np.searchsorted(unique_days, rep_days)]])

    # Pass data directly to kernel — no intermediate float64 arrays
    if is_complex:
        data_np[data_np == 0] = np.nan + 0j
    data_flat = np.ascontiguousarray(data_np.reshape(n_pairs, n_pixels))
    del data_np

    # Prepare weight array (correlation)
    if isinstance(weight_chunk, list):
        weight_np = np.asarray(weight_chunk[0]) if len(weight_chunk) == 1 else np.concatenate([np.asarray(c) for c in weight_chunk], axis=0)
    elif weight_chunk is not None:
        weight_np = np.asarray(weight_chunk)
    else:
        weight_np = None

    if weight_np is not None:
        weight_flat = np.ascontiguousarray(weight_np.reshape(n_pairs, n_pixels).astype(np.float32))
        weight_flat[~np.isfinite(weight_flat)] = 0.0
        weight_flat[weight_flat < 0] = 0.0
    else:
        weight_flat = np.ones((n_pairs, n_pixels), dtype=np.float32)
    del weight_np

    # Run numba kernel
    # REAL per-date times, normalized to [0, 1] (intervals vary; index != time)
    _span = float(unique_days[-1] - unique_days[0])
    date_days_norm = ((unique_days - unique_days[0]) /
                      (_span if _span > 0 else 1.0)).astype(np.float64)

    trend_data, model_data = _trend1d_pairs_numba_kernel(
        data_flat, weight_flat, n_pixels, n_dates, n_pairs,
        np.array(all_pairs, dtype=np.int64),
        all_times_np,
        np.array(all_signs, dtype=np.float64),
        offsets_np,
        pair_ref_didx, pair_rep_didx,
        (rep_days - ref_days).astype(np.float64),  # pair_dt in days
        date_days_norm,
        max_refine,
        is_complex,
    )
    del data_flat, weight_flat

    if return_models:
        # per-date models; the caller can interpolate THESE and difference them,
        # which keeps the correction per-date and closure exact. Interpolating
        # the per-pair trend instead is nonlinear in the phase and destroys
        # triplet closure.
        return model_data.reshape(n_dates, ny, nx)
    return trend_data.reshape(n_pairs, ny, nx)







# ============================================================================
# Complex-phase 2-D trend: coherent sum over a gradient lattice
# ============================================================================
# NOTHING IS UNWRAPPED: the trend is the peak of |sum z exp(-i g.v)| over a
# lattice, bounded in TURNS across the extent the samples span. Outliers enter
# as unit vectors and cancel as sqrt(N), and the sum has no adjacency term, so
# a sparse raster solves as well as a full one.


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_walk1(A, ax0, ur, ui, out_r, out_i):
    n = A.shape[0]; n0 = ax0.size
    d0 = ax0[1] - ax0[0] if n0 > 1 else 0.0
    nd = ur.shape[0]
    for p in range(n):
        a0 = A[p, 0]
        s0r = np.cos(a0 * d0); s0i = -np.sin(a0 * d0)
        ph = a0 * ax0[0]
        cr = np.cos(ph); ci = -np.sin(ph)
        for i in range(n0):
            for d in range(nd):
                zr = ur[d, p]; zi = ui[d, p]
                out_r[d, i] += zr * cr - zi * ci
                out_i[d, i] += zr * ci + zi * cr
            t = cr * s0r - ci * s0i
            ci = cr * s0i + ci * s0r
            cr = t


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_walk2(A, ax0, ax1, ur, ui, out_r, out_i):
    n = A.shape[0]; n0 = ax0.size; n1 = ax1.size
    d0 = ax0[1] - ax0[0] if n0 > 1 else 0.0
    d1 = ax1[1] - ax1[0] if n1 > 1 else 0.0
    nd = ur.shape[0]
    for p in range(n):
        a0 = A[p, 0]; a1 = A[p, 1]
        s0r = np.cos(a0 * d0); s0i = -np.sin(a0 * d0)
        s1r = np.cos(a1 * d1); s1i = -np.sin(a1 * d1)
        ph = a0 * ax0[0] + a1 * ax1[0]
        br = np.cos(ph); bi = -np.sin(ph)
        for i in range(n0):
            cr = br; ci = bi
            for j in range(n1):
                idx = i * n1 + j
                for d in range(nd):
                    zr = ur[d, p]; zi = ui[d, p]
                    out_r[d, idx] += zr * cr - zi * ci
                    out_i[d, idx] += zr * ci + zi * cr
                t = cr * s1r - ci * s1i
                ci = cr * s1i + ci * s1r
                cr = t
            t = br * s0r - bi * s0i
            bi = br * s0i + bi * s0r
            br = t


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_walk3(A, ax0, ax1, ax2, ur, ui, out_r, out_i):
    n = A.shape[0]; n0 = ax0.size; n1 = ax1.size; n2 = ax2.size
    d0 = ax0[1] - ax0[0] if n0 > 1 else 0.0
    d1 = ax1[1] - ax1[0] if n1 > 1 else 0.0
    d2 = ax2[1] - ax2[0] if n2 > 1 else 0.0
    nd = ur.shape[0]
    for p in range(n):
        a0 = A[p, 0]; a1 = A[p, 1]; a2 = A[p, 2]
        s0r = np.cos(a0 * d0); s0i = -np.sin(a0 * d0)
        s1r = np.cos(a1 * d1); s1i = -np.sin(a1 * d1)
        s2r = np.cos(a2 * d2); s2i = -np.sin(a2 * d2)
        ph = a0 * ax0[0] + a1 * ax1[0] + a2 * ax2[0]
        br = np.cos(ph); bi = -np.sin(ph)
        for i in range(n0):
            mr = br; mi = bi
            for j in range(n1):
                cr = mr; ci = mi
                for l in range(n2):
                    idx = (i * n1 + j) * n2 + l
                    for d in range(nd):
                        zr = ur[d, p]; zi = ui[d, p]
                        out_r[d, idx] += zr * cr - zi * ci
                        out_i[d, idx] += zr * ci + zi * cr
                    t = cr * s2r - ci * s2i
                    ci = cr * s2i + ci * s2r
                    cr = t
                t = mr * s1r - mi * s1i
                mi = mr * s1i + mi * s1r
                mr = t
            t = br * s0r - bi * s0i
            bi = br * s0i + bi * s0r
            br = t


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_coherent_kernel(A, G, ur, ui, out_r, out_i):
    """Coherent sum per date per candidate, accumulated in place.

    Serial: dask parallelises across blocks, and parallel=True inside a
    multi-threaded worker trips numba's workqueue layer and kills the worker.

    No table is built -- the sine and cosine go straight into the accumulator,
    computed once per (candidate, sample) and reused across dates.
    """
    nd, n = ur.shape
    K, k = G.shape
    for j in range(K):
        for p in range(n):
            ph = 0.0
            for i in range(k):
                ph += A[p, i] * G[j, i]
            c = np.cos(ph)
            s = np.sin(ph)
            for d in range(nd):
                out_r[d, j] += ur[d, p] * c + ui[d, p] * s
                out_i[d, j] += ui[d, p] * c - ur[d, p] * s


def trend2d_coherent_partial(z, A, axes, G=None):
    """Coherent sums for one spatial chunk: (nd, K) complex64.

    z : (nd, n) complex64 -- flattened valid samples, NaN already zeroed
    A : (n, k) float32    -- centred variables
    G : (K, k) float32    -- gradient lattice

    A reduction with no reuse, so it stays on the CPU. The candidates are
    WALKED: along a linspaced axis the phase advances by a constant step, so
    the next one is a complex multiply instead of a sine and cosine.
    """
    import numpy as _np

    def _np_(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else _np.asarray(x)

    z = _np_(z)
    A = _np.ascontiguousarray(_np_(A), dtype=_np.float32)
    ax = [_np.ascontiguousarray(_np_(a), dtype=_np.float32) for a in axes]
    K = int(_np.prod([a.size for a in ax]))
    if G is None and len(ax) > 3:
        G = _np.stack(_np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, len(ax))
    if G is not None:
        G = _np.ascontiguousarray(_np_(G), dtype=_np.float32)
    if z.ndim == 1:
        z = z[None]
    ur = _np.ascontiguousarray(z.real, dtype=_np.float32)
    ui = _np.ascontiguousarray(z.imag, dtype=_np.float32)
    out_r = _np.zeros((z.shape[0], K), _np.float64)
    out_i = _np.zeros((z.shape[0], K), _np.float64)
    if z.shape[1]:
        if len(ax) == 1:
            _trend2d_walk1(A, ax[0], ur, ui, out_r, out_i)
        elif len(ax) == 2:
            _trend2d_walk2(A, ax[0], ax[1], ur, ui, out_r, out_i)
        elif len(ax) == 3:
            _trend2d_walk3(A, ax[0], ax[1], ax[2], ur, ui, out_r, out_i)
        else:
            _trend2d_coherent_kernel(A, G, ur, ui, out_r, out_i)
    return (out_r + 1j * out_i).astype(_np.complex64)


# ---------------------------------------------------------------------------
# The coherent sum by gridding: S(g) = sum u exp(-i g.v) on a lattice of
# candidates IS a type-1 non-uniform DFT of the phasors at the positions v.
# Spreading each sample onto a grid in VARIABLE space with a smooth kernel and
# transforming once evaluates every candidate at the cost of the kernel's
# footprint -- w**k per sample instead of one pass per candidate. The grid is
# linear in the samples, so blocks and bursts add exactly as before.
# ---------------------------------------------------------------------------

TREND2D_W = 7                 # kernel half-support in cells, each side
TREND2D_BETA = 2.30           # exponential-of-semicircle shape, per unit width
TREND2D_FFT_BUDGET = 64 << 20    # of the transform to hold at once; the
                                 # transform and its correction need a few
                                 # copies of a batch live at the same time


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_spread1(A, ur, ui, cells, w, beta, M, st, gr, gi):
    for p in range(A.shape[0]):
        t = (A[p, 0] + 0.5) * cells[0] + w
        i0 = int(np.ceil(t - 0.5 * w))
        for d in range(w):
            i = i0 + d
            z = 2.0 * (i - t) / w
            if z <= -1.0 or z >= 1.0:
                continue
            kw = np.exp(beta * (np.sqrt(1.0 - z * z) - 1.0))
            for q in range(ur.shape[0]):
                gr[q, i] += ur[q, p] * kw
                gi[q, i] += ui[q, p] * kw


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_spread2(A, ur, ui, cells, w, beta, M, st, gr, gi):
    kv = np.empty((2, w))
    base = np.empty(2, np.int64)
    for p in range(A.shape[0]):
        for a in range(2):
            t = (A[p, a] + 0.5) * cells[a] + w
            i0 = int(np.ceil(t - 0.5 * w))
            base[a] = i0
            for d in range(w):
                z = 2.0 * (i0 + d - t) / w
                kv[a, d] = (np.exp(beta * (np.sqrt(1.0 - z * z) - 1.0))
                            if -1.0 < z < 1.0 else 0.0)
        for d0 in range(w):
            k0 = kv[0, d0]
            if k0 == 0.0:
                continue
            r0 = (base[0] + d0) * st[0]
            for d1 in range(w):
                kk = k0 * kv[1, d1]
                if kk == 0.0:
                    continue
                idx = r0 + (base[1] + d1) * st[1]
                for q in range(ur.shape[0]):
                    gr[q, idx] += ur[q, p] * kk
                    gi[q, idx] += ui[q, p] * kk


@nb.njit(parallel=False, cache=True, fastmath=True)
def _trend2d_spread3(A, ur, ui, cells, w, beta, M, st, gr, gi):
    kv = np.empty((3, w))
    base = np.empty(3, np.int64)
    for p in range(A.shape[0]):
        for a in range(3):
            t = (A[p, a] + 0.5) * cells[a] + w
            i0 = int(np.ceil(t - 0.5 * w))
            base[a] = i0
            for d in range(w):
                z = 2.0 * (i0 + d - t) / w
                kv[a, d] = (np.exp(beta * (np.sqrt(1.0 - z * z) - 1.0))
                            if -1.0 < z < 1.0 else 0.0)
        for d0 in range(w):
            k0 = kv[0, d0]
            if k0 == 0.0:
                continue
            r0 = (base[0] + d0) * st[0]
            for d1 in range(w):
                k1 = k0 * kv[1, d1]
                if k1 == 0.0:
                    continue
                r1 = r0 + (base[1] + d1) * st[1]
                for d2 in range(w):
                    kk = k1 * kv[2, d2]
                    if kk == 0.0:
                        continue
                    idx = r1 + (base[2] + d2) * st[2]
                    for q in range(ur.shape[0]):
                        gr[q, idx] += ur[q, p] * kk
                        gi[q, idx] += ui[q, p] * kk


def trend2d_grid_shape(cells):
    """The cells the samples can reach: the extent plus the kernel's footprint."""
    import numpy as _np
    return _np.atleast_1d(_np.asarray(cells, _np.int64)) + 2 * TREND2D_W


def trend2d_spread(z, A, cells):
    """Samples -> the spread grid, (nd, prod(cells + 2w)) real and imaginary.

    A is in [-0.5, 0.5] per axis, the variable centred on its midpoint and
    divided by its extent. Only the cells the samples can reach are held; the
    zero padding that sets the candidate spacing is added once, in the finalize.
    """
    import numpy as _np
    w = TREND2D_W
    beta = TREND2D_BETA * w
    k = A.shape[1]
    cells = _np.broadcast_to(_np.asarray(cells, _np.float64).ravel(),
                             (k,)).copy()
    M = trend2d_grid_shape(cells)
    st = _np.ones(k, _np.int64)
    for a in range(k - 2, -1, -1):
        st[a] = st[a + 1] * M[a + 1]
    z = _np.asarray(z)
    if z.ndim == 1:
        z = z[None]
    A = _np.ascontiguousarray(A, dtype=_np.float64)
    ur = _np.ascontiguousarray(z.real, dtype=_np.float64)
    ui = _np.ascontiguousarray(z.imag, dtype=_np.float64)
    size = int(_np.prod(M))
    gr = _np.zeros((z.shape[0], size), _np.float64)
    gi = _np.zeros((z.shape[0], size), _np.float64)
    if z.shape[1]:
        fn = {1: _trend2d_spread1, 2: _trend2d_spread2,
              3: _trend2d_spread3}.get(k)
        if fn is None:
            raise ValueError(f"trend2d(): {k} variables, the gridded transform "
                             f"is written for one, two or three.")
        fn(A, ur, ui, cells, w, beta, M, st, gr, gi)
    return gr, gi


def trend2d_deapodise(cells, bins, k):
    """The correction, taken from the SPREADER ITSELF.

    One sample at the centre must transform to a flat 1, so whatever spreading
    and the transform do to it is exactly what divides out -- including the
    discretisation, which the kernel's analytic transform would miss.
    """
    import numpy as _np
    gr, gi = trend2d_spread(_np.ones((1, 1), _np.complex128),
                            _np.zeros((1, 1)), cells)
    return _np.fft.fft(_trend2d_embed(gr[0] + 1j * gi[0], cells, bins, 1))


def _trend2d_embed(grid, cells, bins, k):
    """Place the occupied cells in the padded box the transform runs over."""
    import numpy as _np
    w = TREND2D_W
    M = int(cells) + 2 * w
    P = int(cells) * int(bins)
    off = (P - int(cells)) // 2 - w
    out = _np.zeros((P,) * k, grid.dtype)
    sl = tuple(slice(off, off + M) for _ in range(k))
    out[sl] = grid.reshape((M,) * k)
    return out


def trend2d_reach(cells):
    """Turns the transform can report, in cycles across the extent.

    HALF the computed band, not all of it: past that the kernel's transform has
    decayed and dividing by it amplifies noise into a peak at the edge.
    """
    return cells / 4.0


def trend2d_peak(total, cells, bins, k, guard=0.2):
    """The summed spread grids -> one plane per date.

    The transform gives the objective at every candidate at once; the peak
    locates the gradient, a parabola through its neighbours puts it between
    nodes, and the peak's ARGUMENT is the constant, since there the residual
    phasors align.

    NaN, never the best of a bad grid, for a peak in the outer `guard` of the
    band -- the box is periodic, so a trend past the reach folds back to the
    far edge and would otherwise be believed -- or below sqrt(n ln K), which is
    what n random phasors produce on their own over K candidates.

    Returns (gradients in cycles across the extent, constants, coherence,
    resolved, why), why being 0 resolved, 1 no pixels, 2 on the rim, 3 in the
    mud.
    """
    import numpy as np
    cells = int(cells)
    bins = int(bins)
    M = cells + 2 * TREND2D_W
    nd = total.shape[0]
    K = M ** k
    S = total[:, :K] + 1j * total[:, K:2 * K]
    n = total[:, -1]
    P = cells * bins
    j = np.fft.fftfreq(P, d=1.0 / P)
    kh = trend2d_deapodise(cells, bins, k)
    corr = np.where(np.abs(kh) > 1e-12, 1.0 / kh, 0.0)
    # ASCENDING, not transform order: the peak's neighbours have to be its
    # neighbours in frequency, or the bracket test skips the node at zero and
    # the interpolation silently returns the node itself
    cyc = np.fft.fftshift(j) / bins
    reach = trend2d_reach(cells)
    band = np.abs(cyc) <= reach
    edge = np.abs(cyc) > (1.0 - guard) * reach
    bc = cyc[band]
    be = edge[band]
    g = np.full((nd, k), np.nan)
    c = np.full(nd, np.nan)
    coh = np.zeros(nd)
    why = np.zeros(nd, np.int64)
    # AS MANY DATES PER TRANSFORM AS THE BUDGET HOLDS. One at a time wastes the
    # planning and the memory traffic; all of them at once is nd padded grids
    # at once, which is where a long stack runs out of room.
    _bytes = (P ** k) * 16
    _batch = max(1, int(TREND2D_FFT_BUDGET / max(_bytes, 1)))
    _F = {}
    for _b0 in range(0, nd, _batch):
        _sel = [d for d in range(_b0, min(_b0 + _batch, nd)) if n[d] > 0]
        if not _sel:
            continue
        _blk = np.stack([_trend2d_embed(S[d], cells, bins, k) for d in _sel])
        for a in range(1, k + 1):          # axis at a time, in place
            _blk = np.fft.fft(_blk, axis=a)
        for a in range(k):
            sh = [1] * (k + 1)
            sh[a + 1] = P
            _blk *= corr.reshape(sh)
        _blk = np.fft.fftshift(_blk, axes=tuple(np.arange(1, k + 1)))
        for _i, d in enumerate(_sel):      # cropped, so only the band is kept
            _F[d] = _blk[_i][np.ix_(*([band] * k))].astype(np.complex64)
        del _blk
    for d in range(nd):
        if n[d] <= 0:
            why[d] = 1
            continue
        F = _F[d]
        mag2 = np.abs(F) ** 2
        q = np.unravel_index(mag2.argmax(), mag2.shape)
        peak = F[q]
        coh[d] = float(abs(peak) / max(n[d], 1.0))
        if any(be[q[a]] for a in range(k)):
            why[d] = 2
            continue
        for a in range(k):
            g[d, a] = bc[q[a]]
            if 0 < q[a] < mag2.shape[a] - 1:
                sl = list(q)
                sl[a] = q[a] - 1
                ym = mag2[tuple(sl)]
                sl[a] = q[a] + 1
                yp = mag2[tuple(sl)]
                den = ym - 2 * mag2[q] + yp
                if den != 0:
                    g[d, a] += (np.clip(0.5 * (ym - yp) / den, -1, 1)
                                / bins)
        c[d] = float(np.angle(peak))
    floor = np.sqrt(np.log(max(int(band.sum()) ** k, 2)) / np.maximum(n, 1.0))
    mud = (why == 0) & (coh <= floor)
    why[mud] = 3
    resolved = why == 0
    g[~resolved] = np.nan
    c[~resolved] = np.nan
    return g, c, coh, resolved, why


def trend2d_axes(span, rng, half):
    """Candidate positions per variable: the SEARCH IS IN TURNS, not in rates.

    So the rate depends on how far the samples reach, known only after reading.
    One node past `rng`, so a trend of exactly `rng` is bracketed, not on the
    rim where it would be called unresolved.
    """
    import numpy as np
    out = []
    for i in range(len(half)):
        h = int(half[i])
        g = (float(rng[i]) * h / max(h - 1, 1)
             / max(float(span[i]), 1e-30))
        out.append(np.linspace(-g, g, 2 * h + 1, dtype=np.float32))
    return out


def trend2d_accumulate(data_blk, transform_blk, stats, cells, dims=None):
    """One spatial block -> the spread grid the fit reads, per date.

    Everything the estimator reads is a sum over pixels, so a block contributes
    its share and the caller adds them: the grid, real and imaginary, plus the
    sample count the coherence floor needs.
    """
    import numpy as np
    k = len(transform_blk)
    nb = data_blk.shape[0]
    stats = np.asarray(stats, np.float64).ravel()
    mu = stats[:k]
    span = np.maximum(stats[k:2 * k], 1e-30)
    M = int(cells) + 2 * TREND2D_W
    K = M ** k
    out = np.zeros((nb, 1, 1, 2 * K + 1), np.float64)

    # the positions this block can contribute: geometry, and a date with phase
    ny, nx = data_blk.shape[-2:]
    if dims is None:
        dims = ['yx'] * len(transform_blk)
    V = [np.asarray(b, np.float32) for b in transform_blk]
    keep = np.ones(ny * nx, bool)
    for v, d in zip(V, dims):
        if d == 'yx':
            keep &= np.isfinite(v.reshape(-1))
    have = np.zeros(ny * nx, bool)
    for t in range(nb):
        a = np.abs(data_blk[t]).reshape(-1)
        have |= np.isfinite(a) & (a > 0)
    keep &= have
    # A VARIABLE ALONG ONE AXIS RULES OUT WHOLE ROWS OR COLUMNS, and normally
    # none, so the raster-sized mask is only built if it has to be
    for v, d in zip(V, dims):
        if d != 'yx' and not np.isfinite(v).all():
            bad = ~np.isfinite(v)
            keep &= ~(np.repeat(bad, nx) if d == 'y' else np.tile(bad, ny))
    idx = np.flatnonzero(keep)
    npts = idx.size
    if npts == 0:
        return out

    # INDEXED, NEVER BROADCAST, and scaled to the box the grid spans: the
    # midpoint centring is what puts every sample inside [-1/2, 1/2]
    rows = idx // nx
    A = np.empty((npts, k), np.float64)
    for i, (v, d) in enumerate(zip(V, dims)):
        if d == 'yx':
            A[:, i] = v.reshape(-1)[idx]
        elif d == 'y':
            A[:, i] = v[rows]
        else:
            A[:, i] = v[idx - rows * nx]
        A[:, i] = (A[:, i] - mu[i]) / span[i]
    u = np.zeros((nb, npts), np.complex128)
    for t in range(nb):
        z = data_blk[t].reshape(-1)[idx]
        a = np.abs(z)
        np.divide(z, a, out=u[t], where=np.isfinite(a) & (a > 0))
    out[:, 0, 0, -1] = (np.abs(u) > 0).sum(1)

    gr, gi = trend2d_spread(u, A, cells)
    out[:, 0, 0, :K] = gr
    out[:, 0, 0, K:2 * K] = gi
    return out


def trend2d_finalize(total, axes):
    """The combined accumulators -> one plane per date.

    The accumulator IS the objective, sampled: the peak locates the gradient, a
    parabola through its neighbours puts it between nodes, and the peak's
    ARGUMENT is the constant, since there the residual phasors align.

    NaN, never the best of a bad lattice, for a peak on the rim -- not a
    maximum but where the search stopped -- or below sqrt(n ln K), which is
    what n random phasors produce on their own over K candidates.

    Returns (gradients, constants, coherence, resolved, why), where why is
    0 resolved, 1 no pixels, 2 on the rim, 3 in the mud.
    """
    import numpy as np
    shape = tuple(int(a.size) for a in axes)
    K = int(np.prod(shape))
    nd, k = total.shape[0], len(axes)
    S = total[:, :K] + 1j * total[:, K:2 * K]
    n = total[:, -1]
    g = np.zeros((nd, k), np.float64)
    c = np.zeros(nd, np.float64)
    coh = np.zeros(nd, np.float64)
    rim = np.zeros(nd, bool)
    mag2 = (np.abs(S) ** 2).reshape((nd,) + shape)
    for d in range(nd):
        j = np.unravel_index(mag2[d].argmax(), shape)
        for i in range(k):
            step = float(axes[i][1] - axes[i][0]) if shape[i] > 1 else 0.0
            g[d, i] = float(axes[i][j[i]])
            if 0 < j[i] < shape[i] - 1:
                sl = list(j)
                sl[i] = j[i] - 1; ym = mag2[d][tuple(sl)]
                sl[i] = j[i] + 1; yp = mag2[d][tuple(sl)]
                den = ym - 2 * mag2[d][j] + yp
                if den != 0:
                    g[d, i] += np.clip(0.5 * (ym - yp) / den, -1, 1) * step
            elif shape[i] > 1:
                rim[d] = True
        peak = S[d].reshape(shape)[j]
        c[d] = float(np.angle(peak))
        coh[d] = float(abs(peak) / max(n[d], 1.0))
    floor = np.sqrt(np.log(max(K, 2)) / np.maximum(n, 1.0))
    resolved = (n > 0) & ~rim & (coh > floor)
    why = np.where(n <= 0, 1, np.where(rim, 2, np.where(coh <= floor, 3, 0)))
    g[~resolved] = np.nan
    c[~resolved] = np.nan
    return g, c, coh, resolved, why


# Populate numba file cache on first import so dask workers skip compilation.
# LAST, so every kernel above is defined by the time it runs.
_warmup_numba_cache()
