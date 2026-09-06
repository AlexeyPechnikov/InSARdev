"""Arc counting and the atmospheric solve on a persistent-scatterer network.

An arc is the double difference between two pixels on the same dates, so the
atmospheric screen cancels in it and the arc measures pixel quality with the
atmosphere already removed. Two pixels closer than the INDEPENDENCE CELL in
both axes are one sample of the ground, not two, so a coherence between them
measures the sensor's impulse response rather than the terrain -- every arc
here clears the cell.

  `arcs()`     -> sparse selection of independent, connected pixels
  `fit3d()`      -> best arc coherence selects the nodes; a joint (height, velocity)
                  model is fitted on the arcs, integrated over the network, and
                  the temporally-rough remainder is the atmospheric phase.
"""
import numba as _numba
import numpy as np
import threading as _threading
import collections.abc as _abc
import time
import warnings


_RAYLEIGH_MEAN = np.sqrt(np.pi) / 2.0
# The scan runs this much wider than the caller's max_dh/max_dv so that their
# range is entirely interior: a solution at 0.99 * max is found on its merits,
# not pinned against a boundary. Solutions landing in the guard band are
# rejected -- their truth is beyond the range and cannot be recovered.
_GUARD = 1.1
# cells either way across the annual amplitude range: the lattice picks the
# BASIN and the refinement does the rest, so this is resolution, not accuracy
_SEASONAL_STEPS = 0   # 0 = spacing `sat` (one seed for small ranges)

# How many times the (dh, v) and seasonal lattices are alternated. One pass
# leaves the height anchored to a fit that never saw the annual term, which is
# harmless at small amplitudes and fatal at large ones.
_SEASONAL_ROUNDS = 3

# How many partners each pixel offers the spanning forest, ranked by raw
# coherence. The forest takes only what connectivity needs -- about one arc per
# node -- so this only has to be deep enough that a pixel's usable partner is
# among them.
_ARC_CANDIDATES = 6

# The most components one solve may return. It is the int8 label's capacity,
# not a preference: past 127 a label folds onto another and two unrelated
# datums read as one. It also bounds the work -- every component costs its own
# kriging pass, so a shattered network would otherwise turn into thousands of
# solves over a single raster.
_MAX_COMPONENTS = 127

# Reweighting passes used to find arcs the network contradicts. The weights
# converge quickly because each pass only has to separate a heavy tail from a
# tight core, not to fit anything.
# A per-node sigma can collapse to zero when a node's few measurements happen
# to land identically, and a zero scale rejects everything else. The floor is
# this fraction of the scene's own scatter -- a scale cannot be finer than the
# measurement is. The SAME fraction on both halves of the solve: the two
# residuals are defined differently, because an arc is one equation in a joint
# system while a partner is a direct estimate, but nothing justifies judging
# them at different resolutions.
# THE FLOOR IS HUBER'S TUNING ON A CALIBRATED SCALE. A robust scale needs a
# lower bound only so a degenerate set cannot drive it to zero, and the MAD is
# already that scale: sigma = 1.4826 * MAD. Huber's constant is 1.345 sigma,
# which is 2 MAD, so the bound is stated in MADs and the constant is the one
# the estimator was designed around rather than a number chosen here.
#
# Set as a small FRACTION of the MAD it did real damage: the IRLS weight is
# `gamma / max(|r|, floor)`, so at a twentieth of a MAD a measurement sitting
# on the running centre outweighed one a MAD away by twenty to one and the
# estimate collapsed onto whichever happened to be nearest. On arcs it rejected
# twice as many as it should, and what it rejected was not outliers.
class _ThreadStats(_abc.MutableMapping):
    """`.stats` as PER-THREAD state, so any dask shape is a valid one.

    A worker with more than one thread runs several blocks in one PROCESS, and
    a module-level dict is then shared by all of them: whatever the last block
    wrote is what the next one reads. That is a wrong answer whenever the two
    happen to be the same length and a crash when they are not, and neither is
    something a caller can be asked to avoid by choosing a different cluster --
    `threads_per_worker=2` exists so a worker can read while it computes.

    Every mapping operation resolves against the CALLING thread's dict, so the
    existing `_3d_fit_ps_array.stats[...]` call sites need no change; only the
    whole-dict assignments become `reset()`, since rebinding the attribute
    would swap the proxy out from under the other threads.
    """

    __slots__ = ('_tl',)

    def __init__(self):
        self._tl = _threading.local()

    @property
    def _d(self):
        d = getattr(self._tl, 'd', None)
        if d is None:
            d = {}
            self._tl.d = d
        return d

    def reset(self, mapping=None, **kw):
        d = dict(mapping) if mapping else {}
        d.update(kw)
        self._tl.d = d
        return self

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        self._d[k] = v

    def __delitem__(self, k):
        del self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __repr__(self):
        return f'_ThreadStats({self._d!r})'


_SIGMA_FLOOR = 2.0

# Arcs each node should end up with. One (a bare spanning forest) connects the
# network but leaves every node on a single measurement, which shows in the
# height. Two restores the averaging at about twice the fitting, still ~1 fit
# per pixel.
_ARC_DEGREE = 2


# WHY max_seasonal DEFAULTS TO 10 mm RATHER THAN 0.
# The annual term is not optional. Leaving it out biases the fit: the height
# degrades from a few mm of annual upward and the rate can land on a whole
# sideband. The main branch always carried it for exactly this reason -- its
# velocity() fitted {1, t, sin, cos} to "separate velocity from annual seasonal,
# unbiased for any time span", and its periodogram removed the annual per
# velocity candidate to "prevent seasonal signal from biasing the velocity".
#
# What was wrong there was the term being an UNBOUNDED linear projection: two
# free parameters with no amplitude limit, fitted to arcs that carry only a
# small differential seasonal, which is overfitting and costs held-out
# coherence. Bounding the amplitude keeps the necessity and drops the
# overfitting.
#
# 10 mm because a seasonal signal is long-wavelength: an arc sees only the few
# mm that does not cancel between two pixels tens of metres apart. Set larger,
# the search is thousands of points wide for an amplitude an arc cannot carry;
# set smaller, marginal arcs attach but fit poorly, and the screen injects a
# long-wavelength rate that costs the references their velocity.

# SEASONAL. An annual term of amplitude A radians leaves coherence |J0(A)| at
# the true rate and |J1(A)| one cycle/yr away; they cross at A = 1.435 rad,
# above which the sideband is genuinely the higher maximum and NO
# coherence-maximising estimator returns the truth from a {dh, v} model. The
# annual term therefore has to be IN the model, and it has to be SEEDED: adding
# its columns to the refinement alone still fails, because a linearised step
# saturates once the annual phase exceeds about pi/2.
#
# It is seeded with a lattice rather than a swarm of refinements, because the
# complex amplitude enters the phase linearly exactly as (dh, v) do -- same or
# better accuracy than a multi-seed refinement search, at a fraction of the
# cost.

# The STAGES the network is built in, as fractions of the selection ranked by
# arc coherence. Each stage solves the atmosphere on the points it admits,
# removes it, and hands the corrected scenes to the next one, so a later stage
# tests its arcs against scenes the earlier stages have already cleaned.
#
# Fractions of the ranking, not quality levels: a level would have to be
# guessed per scene, since the coherence a scatterer reaches depends on the
# stack length and the terrain, while a fraction adapts to both.
_CORE_STAGES = (1.0,)
# TRIED AND WORSE, kept as one tuple so it can be retried against a change that
# addresses the reason. Staging the network -- solve the strongest points,
# remove their screen, admit the next tier against the cleaned scenes -- is
# sound in principle and does fix the network: the core passes a far higher
# fraction of its arcs than everything at once. What it does NOT survive is the
# accumulation of screens. A sparse stage's screen is mostly its own nodes'
# NOISE, because few nodes average little of it away, and multiplying such
# screens together adds that noise once per stage while the atmosphere is only
# captured once. The accumulated screen then injects a long-wavelength rate
# where a single pass injects none.


def _3d_consensus(consensus):
    """How many agreeing measurements a value must rest on. An int.

    One argument for both halves of the solve, because it is one question asked
    twice: an arc must agree with the network, a partner must agree with the
    other partners.

    IT USED TO CARRY THE OUTLIER BOUND AND THE PASS COUNT AS WELL, as
    `(n, k, i)`. Neither belonged with it. The bound is now stated in physical
    units by `err_dh`/`err_dv`, where a caller can reason about it -- a robust
    sigma is a property of whatever scatter happens to be present, so a pixel
    whose partners were uniformly wrong widened the bar that judged them and
    passed. The pass count is `iterations`, which already governs the other
    iterative refinement in the same solve.

    A COUNT IS REQUIRED. `None` used to mean "ask for none of it", but the
    network stage has no path without something to judge against: it leaves its
    residuals unset and fails hundreds of lines later, inside a dask block.
    Parsed in ONE place and called from the public method as well, so a bad
    value raises where it was written rather than inside a dask block minutes
    later.
    """
    if consensus is None or isinstance(consensus, bool):
        raise ValueError(
            'consensus must be an integer count of agreeing measurements; '
            f'got {consensus!r}')
    if isinstance(consensus, (tuple, list)):
        raise ValueError(
            'consensus is now a plain count; the outlier bound moved to '
            f'err_dh/err_dv and the pass count to iterations. Got {consensus!r}')
    try:
        ma = int(consensus)
    except (TypeError, ValueError):
        raise ValueError(
            f'consensus must be an integer count; got {consensus!r}') from None
    if ma != consensus or ma < 1:
        raise ValueError(
            f'consensus must be a whole number >= 1; got {consensus!r}')
    return ma
def _3d_arc_offsets(window_y, window_x, cell_y=2, cell_x=8):
    """Neighbour offsets inside the window that are not the same sample.

    A pair is INADMISSIBLE when it lies inside `cell` in BOTH axes:
    |dy| < cell_y and |dx| < cell_x. The default (2, 8) is 16 x 16 m on a
    standalone Sentinel-1 IW grid (8 x 2 m, no upscaling) -- isotropic in
    ground units, which a rectangular pixel grid is not.

    This replaces a measured independence cell, which did not measure what its
    name said. That estimate came from how far coherence stays high with
    offset, and coherence stays high wherever the SCENE stays similar -- land
    cover, terrain -- not merely across the impulse response. It therefore
    returned cells many times the ground-range resolution, varying block to
    block, and excluding out that far discarded real scatterers a few metres
    apart: far more candidates survive once the exclusion is cut to the
    resolution scale.

    The window is the BOX, so the offsets run to half of it either way:
    (16, 64) on an 8 x 2 m grid is 128 x 128 m of ground and the offsets reach
    +-64 m. A scatterer whose only partners lie beyond that needs a bigger
    window -- widening the reach inside a given window would just relabel the
    argument.

    A pair is INADMISSIBLE only when the two pixels touch. Such a pair is a
    single sample measured twice, so its coherence reflects the impulse
    response, not the ground. Oversampled copies still enter the network --
    they are independent of any ORIGINATOR further away, and are found there.
    """
    return [(dy, dx)
            for dy in range(-(window_y // 2), window_y // 2 + 1)
            for dx in range(-(window_x // 2), window_x // 2 + 1)
            if (dy or dx) and not (abs(dy) < cell_y and abs(dx) < cell_x)]

@_numba.njit(nogil=True, cache=True)
def _3d_ds_partners(Uv, Ub, cy, cx, fg, hy, hx, hy2, hx2,
                    cell_y, cell_x, kk, thr, out_v, out_j, early, lo, hi):
    """Each candidate's best `kk` FIXED partners, searched OUTWARD.

    The DS window is the scale over which the atmosphere is taken to be
    common, so it is where a partner is worth having; the doubled window is
    a fallback for candidates the window itself could not serve, and it is
    entered ONLY by those. Widening for everyone costs 16x the box for
    partners that are worse by construction.

    Nothing is materialised per pair: a candidate holds `kk` slots and a
    partner either displaces the weakest or is forgotten, so the working set
    is `kk` per candidate rather than one entry per pair in reach.

    `fg` carries the fixed node's index at its pixel and -1 elsewhere, which
    is what makes the search bipartite -- a candidate never partners another
    candidate, whose value does not exist yet.
    """
    n = Uv.shape[0]
    ny, nx = fg.shape
    # SERIAL ON PURPOSE, threaded by the caller in candidate bands. A
    # parallel=True kernel called from a multi-threaded dask worker trips
    # numba's workqueue layer -- "not threadsafe ... concurrent access has
    # been detected" -- and takes the whole worker down, surfacing only as a
    # crash. Every other kernel here is serial for the same reason, and a
    # candidate writes only its own row, so bands need no coordination.
    for c in range(lo, hi):
        for s in range(kk):
            out_v[c, s] = -1.0
            out_j[c, s] = -1
        yc = cy[c]
        xc = cx[c]
        for _pass in range(2):
            ry = hy if _pass == 0 else hy2
            rx = hx if _pass == 0 else hx2
            ya = yc - ry
            if ya < 0:
                ya = 0
            yb = yc + ry + 1
            if yb > ny:
                yb = ny
            xa = xc - rx
            if xa < 0:
                xa = 0
            xb = xc + rx + 1
            if xb > nx:
                xb = nx
            for py in range(ya, yb):
                dy = py - yc
                ady = dy if dy >= 0 else -dy
                for px in range(xa, xb):
                    j = fg[py, px]
                    if j < 0:
                        continue
                    dx = px - xc
                    adx = dx if dx >= 0 else -dx
                    # ONE SAMPLE OF THE GROUND is not an arc: inside the
                    # independence cell the two pixels share an impulse
                    # response and their coherence reports it, not the terrain.
                    if ady < cell_y and adx < cell_x:
                        continue
                    # the second pass re-walks the first pass's box; skip what
                    # it already weighed rather than paying for it twice
                    if _pass == 1 and ady <= hy and adx <= hx:
                        continue
                    sr = 0.0
                    si = 0.0
                    for d in range(n):
                        ar = Uv[d, c].real
                        ai = Uv[d, c].imag
                        br = Ub[d, j].real
                        bi = Ub[d, j].imag
                        sr += ar * br + ai * bi
                        si += ai * br - ar * bi
                    v = np.sqrt(sr * sr + si * si) / n
                    mi = 0
                    mv = out_v[c, 0]
                    for s in range(1, kk):
                        if out_v[c, s] < mv:
                            mv = out_v[c, s]
                            mi = s
                    if v > mv:
                        out_v[c, mi] = v
                        out_j[c, mi] = j
            got = 0
            for s in range(kk):
                if out_v[c, s] >= thr:
                    got += 1
            if got >= kk:
                # WHETHER THE WIDER PASS IS EVER SKIPPED. With a shortlist
                # near the node count of the inner box this can never fire,
                # and the two passes are one search written twice; the
                # counter is what says which regime a run is in.
                early[c] = 1
                break


@_numba.njit(nogil=True, cache=True)
def _3d_topk_tile(tv, mask, want_own, want_par, tk_v, tk_y, tk_x,
                  y, x0, hx, gx0, xa, xb):
    """One tile's top-k bookkeeping, GIL-free (numba nogil).

    The numpy twin held the GIL in tens of thousands of small partial sorts
    and scatters, which is what stopped the banded threads from scaling; a
    replace-the-minimum insertion per wanted pixel does the same selection in
    one compiled pass.
    """
    w, ndy, span = tv.shape
    kk = tk_v.shape[2]
    for i in range(w):                       # own pixels of the tile row
        if not want_own[i]:
            continue
        xi = x0 + i
        for dy in range(ndy):
            base = dy * span
            for c in range(span):
                if mask[i, base + c] == 0.0:
                    continue
                v = tv[i, dy, c]
                mi = 0
                mv = tk_v[y, xi, 0]
                for s in range(1, kk):
                    if tk_v[y, xi, s] < mv:
                        mv = tk_v[y, xi, s]; mi = s
                if v > mv:
                    tk_v[y, xi, mi] = v
                    tk_y[y, xi, mi] = dy
                    tk_x[y, xi, mi] = c - hx - i
    for dy in range(ndy):                    # the partner ends, offsets reversed
        for col in range(xa, xb):
            if not want_par[dy, col - xa]:
                continue
            yy = y + dy
            for i in range(w):
                if mask[i, dy * span + (col - gx0)] == 0.0:
                    continue
                v = tv[i, dy, col - gx0]
                mi = 0
                mv = tk_v[yy, col, 0]
                for s in range(1, kk):
                    if tk_v[yy, col, s] < mv:
                        mv = tk_v[yy, col, s]; mi = s
                if v > mv:
                    tk_v[yy, col, mi] = v
                    tk_y[yy, col, mi] = -dy
                    tk_x[yy, col, mi] = x0 + i - col


def _3d_topk_kernel(block, window_y, window_x, cell, budget, kk, want,
                    threads=1):
    """Best `kk` partners of the WANTED pixels only -- the topk twin of
    `_3d_arcs_kernel`, named apart because it answers a narrower question.

    The selection is the raster: unchosen pixels neither collect a partner
    nor are one, exactly as the masked path of the full kernel enforces by
    zeroing their phasors. What the full kernel spends besides -- the best-arc
    raster nobody reads at these call sites, and GIL-holding numpy
    bookkeeping -- is dropped, so row bands scale across threads.

    Returns (coherence, dy, dx) shaped (ny, nx, kk); unwanted pixels hold
    coherence -1.
    """
    S = np.asarray(block)
    n, ny, nx = S.shape
    cy, cx = (int(cell[0]), int(cell[1])) if cell is not None else (2, 8)
    wy, wx = int(window_y), int(window_x)
    hy, hx = wy // 2, wx // 2
    kk = int(kk)
    _th = max(1, int(threads))
    if _th > 1 and ny >= 2 * (hy + 1):
        from concurrent.futures import ThreadPoolExecutor
        _mb = _3d_budget_mb(budget) / _th
        H = max(hy + 1, -(-ny // _th))
        bands = [(a, min(a + H, ny)) for a in range(0, ny, H)]
        tv_o = np.empty((ny, nx, kk), np.float32)
        ty_o = np.empty((ny, nx, kk), np.int16)
        tx_o = np.empty((ny, nx, kk), np.int16)

        def _band(band):
            ya, yb = band
            a0 = max(0, ya - hy); b0 = min(ny, yb + hy)
            v, yy, xx = _3d_topk_kernel(S[:, a0:b0], wy, wx, (cy, cx), _mb,
                                        kk, np.asarray(want)[a0:b0])
            sl = slice(ya - a0, yb - a0)
            tv_o[ya:yb] = v[sl]; ty_o[ya:yb] = yy[sl]; tx_o[ya:yb] = xx[sl]
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_band, bands))
        return tv_o, ty_o, tx_o

    want = np.asarray(want, bool)
    K = 2 * n
    Xp = np.zeros((ny, nx + 2 * hx, K), dtype=np.float32)
    ok = np.zeros((ny, nx), bool)
    slab = max(1, min(ny, int(64 * 1024 * 1024 // max(n * nx * 8, 1))))
    for y0 in range(0, ny, slab):
        y1 = min(y0 + slab, ny)
        blk = S[:, y0:y1, :]
        a = np.abs(blk)
        f = np.isfinite(a) & (a > 0)
        o = f.all(axis=0) & want[y0:y1]
        ok[y0:y1] = o
        with np.errstate(invalid='ignore', divide='ignore'):
            u = np.where(f, blk / np.where(f, a, 1), 0)
        u *= o[None, :, :]
        Xp[y0:y1, hx:hx + nx, :n] = np.moveaxis(u.real, 0, -1)
        Xp[y0:y1, hx:hx + nx, n:] = np.moveaxis(u.imag, 0, -1)
        del blk, a, f, o, u
    tile_cap = _3d_budget_mb(budget) * 1024 * 1024
    Bx = max(1, hx)
    while Bx > 8:
        span_ = Bx + 2 * hx
        if (hy + 1) * span_ * 4 * (K + 2 * Bx) <= tile_cap:
            break
        Bx //= 2
    tk_v = np.full((ny, nx, kk), -1.0, np.float32)
    tk_y = np.zeros((ny, nx, kk), np.int16)
    tk_x = np.zeros((ny, nx, kk), np.int16)
    masks = {}
    for y in range(ny):
        ndy = min(hy + 1, ny - y)
        for x0 in range(0, nx, Bx):
            w = min(Bx, nx - x0)
            span = w + 2 * hx
            gx0 = x0 - hx
            xa, xb = max(0, gx0), min(nx, gx0 + span)
            own = want[y, x0:x0 + w]
            par = want[y:y + ndy, xa:xb]
            if not (own.any() or par.any()):
                continue
            A1 = Xp[y, hx + x0:hx + x0 + w, :]
            A2 = np.empty((w, K), np.float32)
            A2[:, :n] = A1[:, n:]; A2[:, n:] = -A1[:, :n]
            Bk = np.ascontiguousarray(
                Xp[y:y + ndy, x0:x0 + span, :].transpose(2, 0, 1)
            ).reshape(K, ndy * span)
            t = A1 @ Bk
            Ci = A2 @ Bk
            np.multiply(t, t, out=t); np.multiply(Ci, Ci, out=Ci)
            t += Ci
            del Ci, Bk, A2
            key = (w, ndy)
            if key not in masks:
                dxm = (np.arange(span)[None, None, :] - hx
                       - np.arange(w)[:, None, None])
                dyv = np.arange(ndy)[None, :, None]
                mm = ((np.abs(dxm) <= hx)
                      & ~((dyv < cy) & (np.abs(dxm) < cx))
                      & ~((dyv == 0) & (dxm <= 0)))
                masks[key] = mm.reshape(w, ndy * span).astype(np.float32)
            m = masks[key]
            t *= m
            _3d_topk_tile(t.reshape(w, ndy, span), m,
                          np.ascontiguousarray(own), np.ascontiguousarray(par),
                          tk_v, tk_y, tk_x, y, x0, hx, gx0, xa, xb)
    good = tk_v > 0
    _sq = np.full(tk_v.shape, -1.0, np.float32)
    np.sqrt(tk_v, out=_sq, where=good)
    tk_v = np.where(good, _sq / n, -1.0).astype(np.float32)
    tk_v[~ok] = -1.0
    return tk_v, tk_y, tk_x


def _3d_arcs_kernel(block, window_y, window_x, cell=(2, 8), budget=None,
                    topk=None, topk_mask=None, threads=1):
    """The BEST arc coherence each pixel reaches -- its PS quality.

    For every admissible separation (dy, dx) the whole raster is correlated
    against itself shifted by it, in one pass:

        gamma(dy, dx) = |sum_d u[d, y, x] * conj(u[d, y+dy, x+dx])| / n_valid

    and each pixel keeps the MAXIMUM over all separations. Not a count of how
    many partners cleared some level, and not a mean: a count reports how
    crowded a neighbourhood is, and a mean is dragged to the noise floor by the
    many partners any pixel has that are simply unrelated to it. The maximum
    answers the only question the selection asks -- does this pixel have a
    partner it agrees with.

    No tiles. Tiling made a pixel's answer depend on where it fell relative to
    the tile edges; sliding the whole array by each offset evaluates every pair
    exactly once, at full resolution, and credits it to BOTH endpoints, so half
    the offsets suffice.

    A pair is skipped when it lies inside the independence cell in BOTH axes:
    those two pixels are one sample of the ground, so their coherence reports
    the impulse response and not the terrain.

    The neighbourhood is a BOX CENTRED on the pixel: separations run to
    +-window//2, so `window` is the full extent, not the reach in one
    direction.

    budget sizes the transient working set and MUST be resolved by the
    caller, in the main process: dask workers are separate processes and do
    not inherit dask.config, so reading `array.chunk-size` in here would
    return the 128 MB default whatever the notebook set (see Stack.py:1435
    for the same trap). None falls back to that read, which is right only
    when the kernel is called directly.

    block : (n_dates, ny, nx) complex
    cell  : (dy, dx) independence cell in pixels. If its exclusion covers the
            whole +-window//2 box no pair is admissible and the result is all
            NaN -- an unmeasurable setting reports nothing rather than a
            number that looks like an answer.

    Returns (ny, nx) float32, the best arc coherence per pixel; NaN where no
    arc was observable -- a pixel that cannot be assessed is not a pixel that
    failed. Threshold it yourself, e.g. `>= 0.6`.

    topk : int or None
        With a count, additionally return the BEST `topk` partners per pixel
        rather than only the best one, as (coherence, dy, dx) arrays shaped
        (ny, nx, topk). The correlation block this reads from is the one the
        maximum is already taken over, so the extra cost is a partial sort of
        values that were computed anyway -- nothing like searching pair by
        pair. A pixel with fewer admissible partners than `topk` has its
        remaining slots at coherence -1 and offset 0.

        This is what a network over these pixels needs and the maximum cannot
        give: which partners, and where.

    topk_mask : (ny, nx) bool or None
        The pixels this call is about. They are the only ones that collect
        partners AND the only ones that can be partners, so a mask makes the
        kernel work over that selection alone -- the caller does not zero a
        copy of the scene to express it.

        The correlation stays dense, which is what makes it fast; what the
        mask removes is the top-`topk` bookkeeping at pixels nobody asked
        about. A network is built over a few per cent of a raster, and
        keeping a sorted list for the other 97% is the whole cost of the
        option: the correlation is a BLAS product, the bookkeeping is a
        partial sort and a scatter per tile.
    """
    S = np.asarray(block)
    n, ny, nx = S.shape
    wy, wx = int(window_y), int(window_x)
    cy, cx = (int(cell[0]), int(cell[1])) if cell is not None else (2, 8)
    hy, hx = wy // 2, wx // 2
    if n < 2 or ny == 0 or nx == 0:
        return np.full((ny, nx), np.nan, dtype=np.float32)

    # ROW BANDS WITH AN hy HALO. A pixel's partners live within +-hy rows, so
    # a band's own rows carry the full-raster answer; the halo pairs are
    # recomputed, which is the whole price. Only a caller that OWNS the host
    # may raise `threads` -- the fit3d gate does, a per-chunk task must not.
    _th = max(1, int(threads))
    if _th > 1 and ny >= 2 * (hy + 1):
        from concurrent.futures import ThreadPoolExecutor
        _mb = _3d_budget_mb(budget) / _th
        H = max(hy + 1, -(-ny // _th))
        bands = [(a, min(a + H, ny)) for a in range(0, ny, H)]
        kk0 = int(topk) if topk else 0
        best_o = np.empty((ny, nx), np.float32)
        if kk0:
            tv_o = np.empty((ny, nx, kk0), np.float32)
            ty_o = np.empty((ny, nx, kk0), np.int16)
            tx_o = np.empty((ny, nx, kk0), np.int16)

        def _band(band):
            ya, yb = band
            a0 = max(0, ya - hy); b0 = min(ny, yb + hy)
            m = None if topk_mask is None else np.asarray(topk_mask)[a0:b0]
            r = _3d_arcs_kernel(S[:, a0:b0], wy, wx, (cy, cx), _mb,
                                topk=topk, topk_mask=m)
            sl = slice(ya - a0, yb - a0)
            if kk0:
                best_o[ya:yb] = r[0][sl]
                tv_o[ya:yb] = r[1][sl]
                ty_o[ya:yb] = r[2][sl]
                tx_o[ya:yb] = r[3][sl]
            else:
                best_o[ya:yb] = r[sl]
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_band, bands))
        return (best_o, tv_o, ty_o, tx_o) if kk0 else best_o

    # ---- BUILD THE OPERAND IN SLABS -------------------------------------
    # The whole-pixel rule first: a pixel is inside the radar extent or it is
    # not, and out there the samples are noise. A pixel valid on all but a few
    # dates is not a scatterer, and keeping it would cost a per-PAIR valid
    # count -- a second reduction as large as the phasor one, half the
    # arithmetic -- to serve a negligible share of pixels. Dropped, n_valid is
    # the constant n for every surviving pair and stops being computed at all.
    #
    # Built a slab of rows at a time, because materialising |S|, the unit
    # phasors and their real and imaginary planes whole costs several times the
    # output array in temporaries. A slab bounds that, and the peak becomes the
    # operand itself.
    K = 2 * n
    Xp = np.zeros((ny, nx + 2 * hx, K), dtype=np.float32)
    ok = np.zeros((ny, nx), dtype=bool)
    slab = max(1, min(ny, int(64 * 1024 * 1024 // max(n * nx * 8, 1))))
    for y0 in range(0, ny, slab):
        y1 = min(y0 + slab, ny)
        blk = S[:, y0:y1, :]
        a = np.abs(blk)
        f = np.isfinite(a) & (a > 0)
        o = f.all(axis=0)
        if topk_mask is not None:
            # THE SELECTION IS THE RASTER, as far as this call is concerned.
            # A network is built over chosen pixels, so the unchosen may
            # neither collect a partner nor BE one -- and applying that here,
            # where observability is already decided, spares the caller
            # zeroing a copy of the scene to say the same thing.
            o = o & np.asarray(topk_mask, bool)[y0:y1]
        ok[y0:y1] = o
        with np.errstate(invalid='ignore', divide='ignore'):
            u = np.where(f, blk / np.where(f, a, 1), 0)
        u *= o[None, :, :]
        Xp[y0:y1, hx:hx + nx, :n] = np.moveaxis(u.real, 0, -1)
        Xp[y0:y1, hx:hx + nx, n:] = np.moveaxis(u.imag, 0, -1)
        del blk, a, f, o, u

    # ---- 2-D TILES ------------------------------------------------------
    # gamma(p, d) = |sum_dates u[p] conj(u[p+d])| / n is an inner product, so
    # the offsets at a fixed row separation are a BAND of a matrix product.
    # Walking one offset at a time re-reads the whole array once per offset --
    # far more memory traffic than arithmetic, so it runs nowhere near compute
    # bound.
    #
    # The shape of the product decides everything: a matmul small in BOTH
    # dimensions leaves most of the machine idle, while a wide right-hand
    # operand reaches full rate. So one tile gathers ALL the row separations
    # into N at once: a (Bx x K) @ (K x (hy+1)*span) per tile, trading a gather
    # for a much faster matmul.
    #
    # The dy halo is ONE-SIDED (0..hy, and dx > 0 at dy == 0) so each pair is
    # still evaluated once and credited to both of its ends; a symmetric halo
    # would double the arithmetic for nothing.
    #
    # The complex product needs only TWO real matrix products: with
    # A1 = [Re, Im] and A2 = [Im, -Re] against the same operand, A1 @ B is the
    # real part and A2 @ B the imaginary, contracting over 2n.
    # SEEDED BELOW ZERO, not at it. A pixel whose window holds no admissible
    # partner -- every neighbour inside the independence cell, or the window
    # clipped at a block corner -- would otherwise keep its seed and be
    # returned as coherence 0.0, which reads as a measured failure rather than
    # as nothing measured. The accumulator only ever takes maxima of squared
    # magnitudes, so a negative seed cannot be reached by a real arc and marks
    # exactly the pixels no pair ever touched.
    best = np.full((ny, nx), -1.0, dtype=np.float32)
    kk = int(topk) if topk else 0
    if kk:
        want = (np.ones((ny, nx), bool) if topk_mask is None
                else np.asarray(topk_mask, bool))
        tk_v = np.full((ny, nx, kk), -1.0, dtype=np.float32)
        tk_y = np.zeros((ny, nx, kk), dtype=np.int16)
        tk_x = np.zeros((ny, nx, kk), dtype=np.int16)

        def _merge(sy, sx, v, oy, ox):
            """Keep the best `kk` of what is held and what just arrived."""
            av = np.concatenate([tk_v[sy, sx], v], axis=1)
            ay = np.concatenate([tk_y[sy, sx], oy], axis=1)
            ax = np.concatenate([tk_x[sy, sx], ox], axis=1)
            j = np.argpartition(av, -kk, axis=1)[:, -kk:]
            tk_v[sy, sx] = np.take_along_axis(av, j, axis=1)
            tk_y[sy, sx] = np.take_along_axis(ay, j, axis=1)
            tk_x[sy, sx] = np.take_along_axis(ax, j, axis=1)
    masks = {}
    # Bx FOLLOWS THE HALO, BUT IS CAPPED. The block computes Bx x (Bx + 2 hx)
    # pairs of which Bx x (2 hx + 1) are wanted, so a small block wastes little
    # but gives BLAS a thin matrix and a large one recomputes the halo.
    # Runtime is flat between hx and 2 hx, so Bx follows hx.
    #
    # The cap exists because the tile's working set grows as hx^2 hy: the
    # score block is Bx x (hy+1) x (Bx + 2 hx) floats and appears twice, so a
    # large window reaches hundreds of MB per tile -- and dask runs one per
    # thread.
    # Shrinking Bx bounds it without touching the result, and costs nothing at
    # the window sizes where Bx is already below the cap.
    # The ceiling is the DASK CHUNK BUDGET, as core sizes every working set
    # (utils_dask.rechunk3d, BatchCore.velocity) -- not a constant of its own.
    # One setting the caller has already tuned for its machine governs this
    # too, so a large window cannot silently allocate hundreds of MB per dask
    # thread while still honouring a raised budget when there is room.
    tile_cap = _3d_budget_mb(budget) * 1024 * 1024
    Bx = max(1, hx)
    while Bx > 8:
        span_ = Bx + 2 * hx
        if (hy + 1) * span_ * 4 * (K + 2 * Bx) <= tile_cap:
            break
        Bx //= 2
    for y in range(ny):
        ndy = min(hy + 1, ny - y)
        for x0 in range(0, nx, Bx):
            w = min(Bx, nx - x0)
            span = w + 2 * hx
            # the tile's own pixels, and the same vectors rotated for Im
            A1 = Xp[y, hx + x0:hx + x0 + w, :]
            A2 = np.empty((w, K), dtype=np.float32)
            A2[:, :n] = A1[:, n:]
            A2[:, n:] = -A1[:, :n]
            # every partner the tile can reach, as one (K, ndy*span) operand
            Bk = np.ascontiguousarray(
                Xp[y:y + ndy, x0:x0 + span, :].transpose(2, 0, 1)
            ).reshape(K, ndy * span)
            t = A1 @ Bk
            Ci = A2 @ Bk
            np.multiply(t, t, out=t)
            np.multiply(Ci, Ci, out=Ci)
            t += Ci
            del Ci, Bk, A2
            key = (w, ndy)
            if key not in masks:
                dxm = (np.arange(span)[None, None, :] - hx
                       - np.arange(w)[:, None, None])
                dyv = np.arange(ndy)[None, :, None]
                mm = ((np.abs(dxm) <= hx)
                      & ~((dyv < cy) & (np.abs(dxm) < cx))
                      & ~((dyv == 0) & (dxm <= 0)))
                masks[key] = mm.reshape(w, ndy * span).astype(np.float32)
            t *= masks[key]
            tv = t.reshape(w, ndy, span)
            # the pixel's own best, and the same values at the partner ends
            np.maximum(best[y, x0:x0 + w], tv.max(axis=(1, 2)),
                       out=best[y, x0:x0 + w])
            tg = tv.max(axis=0)                     # (ndy, span)
            gx0 = x0 - hx
            xa, xb = max(0, gx0), min(nx, gx0 + span)
            if xb > xa:
                np.maximum(best[y:y + ndy, xa:xb], tg[:, xa - gx0:xb - gx0],
                           out=best[y:y + ndy, xa:xb])
            if kk:
                # THE SAME BLOCK, PARTIALLY SORTED. `tv` holds every partner
                # this tile can reach; the maximum above is one reduction of
                # it and the best `kk` is another.
                wsel = np.flatnonzero(want[y, x0:x0 + w])
                if len(wsel):
                    fl = tv.reshape(w, -1)[wsel]
                    m_ = min(kk, fl.shape[1])
                    j = np.argpartition(fl, -m_, axis=1)[:, -m_:]
                    vv = np.take_along_axis(fl, j, axis=1)
                    oy = (j // span).astype(np.int16)
                    ox = ((j % span) - hx
                          - wsel[:, None]).astype(np.int16)
                    _merge(y, x0 + wsel, vv, oy, ox)
                # and credited to the partner ends, where the offset reverses
                if xb > xa:
                    nxs = xb - xa
                    wm = want[y:y + ndy, xa:xb].ravel()
                    if wm.any():
                        tpm = tv[:, :, xa - gx0:xb - gx0].transpose(
                            1, 2, 0).reshape(ndy * nxs, w)[wm]
                        m2 = min(kk, w)
                        j2 = np.argpartition(tpm, -m2, axis=1)[:, -m2:]
                        v2 = np.take_along_axis(tpm, j2, axis=1)
                        gy = np.repeat(np.arange(ndy), nxs)[wm][:, None]
                        gx = np.tile(np.arange(xa, xb), ndy)[wm][:, None]
                        o2y = np.broadcast_to((-gy).astype(np.int16), v2.shape)
                        o2x = ((x0 + j2) - gx).astype(np.int16)
                        ry = (y + np.repeat(np.arange(ndy), nxs)[wm])
                        rx = np.tile(np.arange(xa, xb), ndy)[wm]
                        _merge(ry, rx, v2, o2y, o2x)
                        del tpm, j2, v2
            del t, tv, tg

    seen = best >= 0
    out = np.sqrt(best, out=best, where=seen) / n
    res = np.where(ok & seen, out, np.nan).astype(np.float32)
    if kk:
        good = tk_v > 0
        _sq = np.full(tk_v.shape, -1.0, dtype=np.float32)
        np.sqrt(tk_v, out=_sq, where=good)
        tk_v = np.where(good, _sq / n, -1.0).astype(np.float32)
        tk_v[~(ok & seen)] = -1.0
        return res, tk_v, tk_y, tk_x
    return res


def _3d_depth(chunks, window):
    """Halo depth per axis, and the check that the given chunks can carry it.

    `chunks` is (chunks_y, chunks_x) as dask reports them. Returns
    (depth_y, depth_x).

    An arc reaches at most half the PS extent from the pixel, so half is what a
    block must see beyond its own edge. An axis held in ONE chunk already has
    the whole raster and needs no halo -- and asking for one raises, because
    dask refuses a depth wider than the array.

    THE CHUNKING IS THE CALLER'S AND IS NEVER CHANGED HERE, only checked --
    and checked before `da.overlap` is reached, because that calls
    `ensure_minimum_chunksize()`, which silently re-splits any chunk shorter
    than the depth into lengths of its own choosing. Refusing first is what
    keeps the layout the caller asked for.

    Blocks are solved independently, so a different chunking gives a different
    set of scatterers near the seams. That is the design, not an error: a
    pixel at a block edge sees the neighbourhood its block affords.
    """
    # THE DS BOX IS THE REACH. What a block computes beyond its own edge is
    # the neighbourhood the DS window looks over; the PS extent bounds nothing
    # a halo can carry, since a node reaches every other node of its block.
    wy, wx, _, _ = _3d_windows(window)
    depth = []
    for _cs, _w in zip((tuple(chunks[0]), tuple(chunks[1])), (wy, wx)):
        if len(_cs) == 1:
            depth.append(0)
            continue
        if min(_cs) < _w:
            raise ValueError(f'chunk size {min(_cs)} less than processing '
                             f'window size {_w}, enlarge chunks or decrease '
                             f'window')
        depth.append(_w // 2)
    return tuple(depth)


def _3d_windows(window):
    """(wy, wx) or (wy, wx, py, px) -> the DS box and the PS extent, validated.

    Two numbers give the DS box and the PS extent follows as THREE TIMES it,
    which is the layout where the nine patches are all equal: the centre box is
    the DS window and the eight around it are the same size again. Four numbers
    set the two independently, because the range over which the atmosphere is
    common is a property of the site and nothing in the data states it. Where
    coherence is poor a caller wants a SMALLER DS box -- fewer, better
    neighbours -- and a much larger PS extent, so that pixels which the short
    test cannot certify are still reachable at range: (24, 96, 256, 1024) is
    a 192 x 192 m DS box inside a 2048 x 2048 m PS search.

    Both are FULL extents of a box centred on the pixel, like `window`
    everywhere else here, so the PS extent must exceed the DS box on both axes
    -- otherwise the ring between them is empty and there is nowhere to look.
    """
    w = tuple(int(v) for v in window)
    if len(w) == 2:
        w = w + (3 * w[0], 3 * w[1])
    if len(w) != 4:
        raise ValueError(
            f'window takes 2 values (wy, wx) -- the PS extent is then 3x it -- '
            f'or 4 (wy, wx, ps_y, ps_x); got {len(w)}: {window}')
    wy, wx, py, px = w
    if wy < 2 or wx < 2:
        raise ValueError(f'the DS window must be at least 2 pixels per axis, '
                         f'got ({wy}, {wx})')
    if py < wy + 2 or px < wx + 2:
        raise ValueError(
            f'the PS extent ({py}, {px}) must exceed the DS window ({wy}, {wx}) '
            f'by at least 2 pixels on each axis: the PS partners come from the '
            f'ring between them, and this one is empty')
    return wy, wx, py, px


def _3d_ps_kernel(block, window, quality, ele2phase, t, meter2rad,
                  threshold=0.5, budget=None, iterations=8):
    """Fitted arc coherence to ONE partner per bearing -- the PS test.

    A persistent scatterer carries no dominant noise, so an arc to a distant
    partner is limited only by the atmospheric difference between the two ends,
    which is small over kilometres. A distributed scatterer is coherent with a
    near neighbour and loses it as soon as the common atmosphere has cancelled.
    So the window is the boundary: coherent within (wy, wx) of the pixel is DS
    evidence, coherent beyond it is PS evidence, and the caller sets the window
    for their area.

    THE LONG ARC MUST BE FITTED, not correlated. `_3d_arcs_kernel` measures a raw
    inner product, which is right inside the window because two near pixels
    differ little in height. Over a kilometre they do not: at ele2phase*meter2rad ~ 0.13
    rad/m a 20 m difference is 2.6 rad of baseline-dependent rotation, and raw
    coherence collapses to the noise floor however good both scatterers are.
    Raw long-arc coherence selects essentially nothing at a useful gate, while
    the fitted test still finds km-scale pairs above it. The fit
    solves ONE DIFFERENTIAL (dh, dv) for the pair, so neither pixel's absolute
    parameters are needed.

    The test is mutual by construction: a long arc scores well only when BOTH
    ends are good, since one noisy end sinks it. It must also run BEFORE any
    screen is removed -- afterwards a corrected DS pair is coherent at range too
    and the two classes merge.

A NINE-PATCH RING AROUND EVERY CANDIDATE. The centre patch is the pixel's
    OWN window -- the same +-(wy//2, wx//2) the short raster measured, since a
    window is the full extent of a box centred on each pixel -- and the eight
    others tile the ring between it and the PS extent, cut by extending the
    centre patch's own edges. A partner therefore begins exactly where DS
    evidence ends and reaches the PS extent at the corners, so the boundary
    between the two classes is the window the caller set and not some multiple
    of it. When the PS extent is three times the DS window all nine patches are
    the same size and the ring is the plain 3x3; widening it stretches the eight
    outward without moving the boundary or changing the number of arcs, since
    each patch still contributes exactly one.

    THE GRID SELECTS, THE RING MEASURES. The DS grid decides WHICH pixel
    stands for a window; the ring is then centred on that pixel, not on a tile.
    So the boundary between DS and PS evidence is the pixel's own window
    wherever it sits, and no exclusion rule is needed to undo a tile edge -- a
    grid-aligned ring would leave a pixel near a boundary with partners a few
    pixels away in the "neighbouring" window, and pushing them clear would move
    everyone else's twice as far as the definition asks.

    ONE ARC PER PATCH. Every patch is the SAME AREA, which is the only sense
    in which arcs at different bearings are comparable -- their lengths are not
    equal and cannot be made so. Three things set an arc's coherence: its
    length, its direction, and the partner's own quality. Length comes out FLAT
    across a patch, direction is what the eight patches sweep, and
    the partner's quality is already the best the patch holds -- so a second arc
    into the same patch would vary only the least informative of the three.

    The best candidate in every patch is found by a running maximum over the
    whole raster, not a search per pixel: a patch centred at a fixed offset from
    one pixel is the patch centred on another, so a filtered raster read at
    eight fixed offsets gives all eight partners. The ring has three patch
    shapes -- corner, side, top -- so three filters serve all eight, and one
    when the PS extent is 3x the DS window. Quality and index travel
    together in one integer -- quality in the high bits -- so the maximum
    carries its own argmax and no second pass is needed. Nothing here refers to
    a window GRID, so the answer does not move when a chunk boundary does.

    window : (wy, wx), or (wy, wx, ps_y, ps_x) to set the PS extent apart from
    the DS box. Two values make the PS extent three times the DS window. See
    `_3d_windows`.

    ONE THRESHOLD, NOT TWO, AND IT BARELY MATTERS. `threshold` is the same
    level the caller applies to both rasters afterwards -- PS and DS are not two
    populations to be gated separately, they are one set of coherent pixels
    sorted into two classes by the RANGE at which they hold up. Because only a
    window's BEST candidate is carried forward, and that one sits well clear of
    any plausible level, the result is nearly insensitive to it. What it really
    decides is how many windows hold a candidate at all, so it costs empty
    windows, not scatterers.

    There is no `reach`: the PS extent states how far to look, in the same
    units and the same style as every other window here.

    THE THIRD LEVEL OF THREE. The independence cell says which two pixels are
    one sample of the ground. The DS window collects cells and gives every pixel
    a coherence -- that raster, complete, is all the DS stage produces. This
    stage reads it, takes the best pixel in each DS window, and pairs those.
    Picking the best per window is a plain argmax over the raster and needs
    nothing prepared upstream.

    The search therefore shrinks at each level rather than growing: a PS window
    of (128, 512) spans 4 x 4 DS windows, so it holds one central pixel and
    about fifteen to pair with, however many raw pixels lie beneath it.

    Returns (ny, nx) float32: the best fitted long-arc coherence at the pixels
    the DS grids selected, NaN elsewhere -- about two per DS window once the
    four offset grids are unioned. Complete at WINDOW resolution, as the DS
    raster is complete at pixel resolution, which is what the atmospheric
    screen wants: good nodes spread over the scene, not every pixel that could
    have been one. Against making every candidate a source these score
    marginally lower at the same pixels, since a partner is the best pixel of
    some window and not the best of the whole patch -- for a small fraction of
    the time, and flat in the PS extent.
    """
    S = np.asarray(block)
    n, ny, nx = S.shape
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    if n < 2 or ny == 0 or nx == 0:
        return out
    wy, wx, py, px = _3d_windows(window)
    q = np.asarray(quality)
    cand = np.isfinite(q) & (q >= float(threshold))
    iy, ix = np.where(cand)
    if len(iy) < 2:
        return out
    # VALIDITY AT THE CANDIDATES, IN DATE BATCHES. Testing it on the whole block
    # -- np.abs(S) and a unit-phasor copy of every pixel -- costs several times
    # the block in temporaries and is governed by nothing, only to end up using
    # a few hundred columns. Only candidates can become nodes, so only candidates are
    # tested, and the batch is sized by the dask budget like every other
    # transient here.
    cap = max(1, int(_3d_budget_mb(budget) * 1024 * 1024
                     // max(len(iy) * 16, 1)))
    ok = np.ones(len(iy), dtype=bool)
    for d0 in range(0, n, cap):
        sd = S[d0:d0 + cap][:, iy, ix]
        ad = np.abs(sd)
        ok &= (np.isfinite(ad) & (ad > 0)).all(axis=0)
    iy, ix = iy[ok], ix[ok]
    if len(iy) < 2:
        return out
    from scipy.ndimage import maximum_filter
    # quality in the high bits, the candidate's own index in the low ones, so
    # the running maximum returns WHICH pixel won and not merely how good it
    # was. Zero means no candidate, and every candidate code exceeds it.
    # ONE CANDIDATE PER DS WINDOW, FROM FOUR HALF-OFFSET GRIDS. The window is
    # chosen as the range over which the atmosphere is common, so the candidates
    # inside one share it and are not independent evidence -- typically the same
    # few scatterers sampled repeatedly. The best of them stands for the window,
    # as SOURCE and as PARTNER both, which is what makes the search small: a PS
    # window of (128, 512) spans 4 x 4 DS windows, so it holds one central pixel
    # and about fifteen others to pair with, however many raw pixels lie under
    # it. That brings the candidate count down by orders of magnitude, and the
    # fitted search with it.
    #
    # ONE tiling would keep exactly one per window, so two strong scatterers
    # sharing a tile lose one of them -- not for any physical reason, but
    # because of where the grid's origin fell. Asking from four origins, half a
    # window apart in each axis, and keeping the UNION recovers them at
    # negligible cost. The rescued ones are not marginal and they pass at the
    # same rate, which is what says the single grid was suppressing them rather
    # than filtering them.
    # The same remedy _3d_arcs_select uses one level down.
    qi = np.rint(np.clip(np.nan_to_num(q[iy, ix], nan=0.0), 0, 1)
                 * 1000000).astype(np.int64)
    # Each grid starts at the block's own first row and column, which is right
    # BECAUSE THE CALLER ALIGNS THE BLOCKS: a halo of whole DS windows over
    # chunks of whole DS windows means every block begins on a window boundary,
    # so all of them tile the same ground the same way with nothing passed in.
    # Get that wrong and the answer moves with the chunking -- on unaligned
    # blocks, two chunks and one disagree on a large share of pixels -- which
    # is why `arcs()` rounds both to the window.
    stride = nx // wx + 3
    keep = []
    for oy_, ox_ in ((0, 0), (wy // 2, 0), (0, wx // 2), (wy // 2, wx // 2)):
        wid = ((iy + oy_) // wy).astype(np.int64) * stride + ((ix + ox_) // wx)
        o = np.lexsort((-qi, wid))
        ws = wid[o]
        keep.append(o[np.r_[True, ws[1:] != ws[:-1]]])
    lead = np.unique(np.concatenate(keep))
    if len(lead) < 2:
        return out
    ly, lx = iy[lead], ix[lead]
    # the low bits index the LEADERS, not the candidates: every arc has a leader
    # at both ends -- the code raster is written nowhere else -- so nothing but
    # the leaders' phasors is ever needed, and there are a few hundred of them
    # against a few hundred thousand candidates.
    code = np.zeros((ny, nx), dtype=np.int64)
    code[ly, lx] = (qi[lead] << 32) | (np.arange(len(lead), dtype=np.int64) + 1)
    sl = S[:, ly, lx]
    al = np.abs(sl)
    with np.errstate(invalid='ignore', divide='ignore'):
        Un = np.ascontiguousarray(
            np.where(al > 0, sl / np.where(al > 0, al, 1), 0).astype(np.complex64))
    del sl, al
    # the ring's three patch shapes, and where their centres sit. Each spans
    # from the edge of the DS box to the edge of the PS extent on whichever
    # axes its bearing moves, and the DS box's own extent on the axes it does
    # not -- so the eight together tile the ring exactly and none enters the
    # centre. At ps == 3 * ds these collapse to one shape at one offset.
    # DERIVED FROM THE BOUNDARIES, not from (py - wy) // 2 and (py + wy) // 4:
    # those two truncations only agree when the sizes are even, and when they
    # disagree the ring OVERLAPS the DS window, leaving some cells doubly
    # covered and others uncovered. That admits short arcs as PS evidence,
    # silently, which is the one thing the ring exists to prevent. Stating the
    # first row outside the centre and the last row inside the extent, and
    # fitting the patch to them, is exact for every parity and reproduces the
    # old numbers wherever the old ones were right.
    #
    # The centre reaches +-(wy//2) INCLUSIVE, because that is what the short
    # test measured: _3d_arc_offsets ranges over -(wy//2) .. wy//2, so a
    # (32, 128) window is 33 rows and not 32. The ring therefore starts one row
    # further out than the filter box would suggest; starting at wy//2 would
    # hand the ring a separation the DS test had already claimed.
    lo_y, hi_y = wy // 2 + 1, py - py // 2 - 1
    lo_x, hi_x = wx // 2 + 1, px - px // 2 - 1
    bh, bw = hi_y - lo_y + 1, hi_x - lo_x + 1
    oy, ox = lo_y + bh // 2, lo_x + bw // 2
    # PADDED BY THE PATCH OFFSET, so that a patch whose CENTRE falls off the
    # raster is still evaluated: it can be mostly on the raster and full of
    # candidates, and the filter's own zero boundary already handles the part
    # that is not. Testing the centre for liveness instead threw those bearings
    # away entirely, and the wider the PS search the more of the scene it hit --
    # at ps_x = 1024 the centre is 280 px out, so 22% of columns lost a bearing
    # they had partners in. Clamping the centre would be worse than dropping it:
    # it reads the patch centred somewhere else.
    pad = np.pad(code, ((oy, oy), (ox, ox)), constant_values=0)
    corner = maximum_filter(pad, size=(bh, bw), mode='constant', cval=0)
    vert = (corner if wx == bw else            # (+-1, 0): above and below
            maximum_filter(pad, size=(bh, wx), mode='constant', cval=0))
    horiz = (corner if wy == bh else           # (0, +-1): left and right
             maximum_filter(pad, size=(wy, bw), mode='constant', cval=0))
    src, tgt = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue                          # the pixel's own window
            best_in_box = corner if (dr and dc) else (vert if dr else horiz)
            # every centre is inside the padded raster by construction, so
            # there is nothing to test but whether a candidate was found
            c = best_in_box[ly + dr * oy + oy, lx + dc * ox + ox]
            hit = np.where(c > 0)[0]
            src.append(hit)
            tgt.append((c[hit] & 0xFFFFFFFF) - 1)
    src = np.concatenate(src) if src else np.empty(0, np.int64)
    if not len(src):
        return out
    tgt = np.concatenate(tgt)
    # k picking j and j picking k is ONE arc; fitting it twice costs twice and
    # scores the same two pixels, since the update below is mutual either way
    lo, hi = np.minimum(src, tgt), np.maximum(src, tgt)
    _, u = np.unique(lo * np.int64(len(lead)) + hi, return_index=True)
    src, tgt = lo[u], hi[u]
    _3d_ps_kernel.stats = dict(candidates=int(len(iy)),
                               sources=int(len(lead)), arcs=int(len(src)))
    # seeded BELOW any coherence, not with NaN: np.maximum(nan, x) is nan, so a
    # NaN seed makes every update a no-op and nothing is ever scored
    best = np.full(len(lead), -1.0, dtype=np.float32)
    step = max(1, int(_3d_budget_mb(budget) * 1024 * 1024 // max(n * 16, 1)))
    for b0 in range(0, len(src), step):
        a_, b_ = src[b0:b0 + step], tgt[b0:b0 + step]
        arc = (Un[:, a_] * np.conj(Un[:, b_])).astype(np.complex64)
        # the budget is PASSED ON. _3d_budget_mb(None) re-reads dask.config,
        # and a dask worker is a separate process that never inherited it, so
        # dropping it here would silently size the fit's largest operand by the
        # 128 MB default whatever the caller configured.
        g, _, _, _ = _3d_arc_fit(np.ascontiguousarray(arc), ele2phase, t, meter2rad,
                              budget=budget,
                              iterations=iterations)
        for e in (a_, b_):
            np.maximum.at(best, e, np.where(np.isfinite(g), g, -1.0))
    out[ly, lx] = np.where(best >= 0, best, np.nan)
    return out


def _3d_arcs_select(U, quality, window, threshold, cell=(2, 8)):
    """Sparse, independent, MUTUALLY CONNECTED pixels -- what the network uses.

    Counting arcs per pixel says how well connected a pixel is, but not to
    WHOM, and that distinction decides the network. Ranking candidates
    independently and suppressing a cell box around each winner keeps only a
    fraction of a winner's verified partners, because coherent partnership is a
    specific pairing and independent thinning does not respect it. The
    triangulation that follows then joins pixels never tested against each
    other, so a node's Delaunay neighbours score far below the best partner
    actually available to it and most of those edges fail the arc test.

    Selecting here keeps the pairing, because here the partners are known. Take
    the best-connected unblocked candidate, suppress its cell neighbourhood,
    then test it against the candidates in its window and promote the ones that
    PASS -- they are cell-independent by the same rule and joined to it by an
    arc already known to work -- and carry on from them. Note the partners are
    never inside the seed's own cell: the kernel excluded intra-cell pairs
    before counting, so every partner is an independent sample by construction.

    Same threshold and the kernel's own validity rule: no new parameter.

    Returns (selection, edges): the best arc coherence at the selected pixels with
    NaN everywhere else, and the VERIFIED pairs as (y0, x0, y1, x1) rows. The edges
    are not a by-product to be discarded -- they are the arcs this stage proved
    work, and re-deriving the network by triangulation instead throws them
    away: a pixel selected here on three good close partners can be
    triangulated to whichever nodes happen to be nearest, fail all of them, and
    drop out with degree zero.
    """
    from collections import deque
    n, ny, nx = U.shape
    wy, wx = int(window[0]), int(window[1])
    cy, cx = (int(cell[0]), int(cell[1])) if cell is not None else (2, 8)
    cy_, cx_ = cy - 1, cx - 1
    # THE SAME WHOLE-PIXEL RULE THE KERNEL USES. A pixel is inside the radar
    # extent or it is not; out there the samples are noise. Carrying a
    # per-PAIR valid count here would be a second, different notion of
    # validity in the same file -- and it would let a pixel be selected on an
    # arc to noise, which is a false positive by construction, not a weak
    # scatterer. Two stages must not disagree about what a valid pixel is.
    pix_ok = (np.abs(U) > 0).all(axis=0)
    # A pixel is a candidate only if its OWN best arc reached the threshold.
    # The growth below promotes a partner on a measured arc >= threshold, but a
    # SEED is taken untested at _take() -- so with `quality > 0` every pixel
    # that is not blocked becomes a point, whatever its arcs did. Ungated
    # against planted truth, most selected pixels are decorrelated ground and
    # few of the triangulated arcs survive the fit; gated, essentially all of
    # them are real. The seed's best arc is itself a measured arc, so this is
    # the same rule the growth applies, not a second parameter.
    cand = np.isfinite(quality) & (quality >= threshold) & pix_ok
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    edges = []
    if not cand.any():
        return out, np.zeros((0, 4), dtype=np.int64)

    blocked = np.zeros((ny, nx), dtype=bool)
    taken = np.zeros((ny, nx), dtype=bool)
    owner = np.full((ny, nx), -1, dtype=np.int64)
    ys, xs = np.where(cand)
    # rank by the BEST ARC COHERENCE, not the arc count: of a touching
    # group -- one sample seen several times -- the pixel whose best arc works
    # best is the one to keep, whereas the count mostly reports how crowded
    # that pixel's neighbourhood is
    order = np.argsort(-quality[ys, xs], kind='stable')

    def _take(y_, x_):
        taken[y_, x_] = True
        out[y_, x_] = quality[y_, x_]
        sl = (slice(max(0, y_ - cy_), y_ + cy_ + 1),
              slice(max(0, x_ - cx_), x_ + cx_ + 1))
        # remember WHICH node covers each blocked pixel: a blocked partner is
        # within one cell of that node, i.e. the same ground sample, so an arc
        # verified against the partner is an arc against the node
        owner[sl] = np.where(blocked[sl], owner[sl], y_ * nx + x_)
        blocked[sl] = True

    for p in order:
        y0, x0 = int(ys[p]), int(xs[p])
        if blocked[y0, x0]:
            continue
        _take(y0, x0)
        q = deque([(y0, x0)])
        while q:
            yi, xi = q.popleft()
            ya, yb = max(0, yi - wy // 2), min(ny, yi + wy // 2 + 1)
            xa, xb = max(0, xi - wx // 2), min(nx, xi + wx // 2 + 1)
            sub = cand[ya:yb, xa:xb] & ~blocked[ya:yb, xa:xb]
            if not sub.any():
                continue
            ly, lx = np.where(sub)
            gy, gx = ly + ya, lx + xa
            keep = ~((np.abs(gy - yi) < cy) & (np.abs(gx - xi) < cx))
            if not keep.any():
                continue
            gy, gx = gy[keep], gx[keep]
            ok_ = pix_ok[gy, gx]
            if not ok_.any():
                continue
            gy, gx = gy[ok_], gx[ok_]
            g = np.abs(U[:, yi, xi].conj() @ U[:, gy, gx]) / n
            hit = np.where(g >= threshold)[0]
            # Only touching pixels are one sample, and the offsets excluded
            # those already -- everything reaching here is a real partner.
            for k in hit[np.argsort(-g[hit])]:
                yk, xk = int(gy[k]), int(gx[k])
                if not taken[yk, xk]:
                    if blocked[yk, xk]:
                        # the partner is one cell from an existing node, so it
                        # IS that node's sample: keep the arc, against the node
                        o = int(owner[yk, xk])
                        if o >= 0:
                            oy, ox = divmod(o, nx)
                            if not ((abs(oy - yi) < cy) and (abs(ox - xi) < cx)):
                                edges.append((yi, xi, oy, ox))
                        continue
                    _take(yk, xk)
                    q.append((yk, xk))
                edges.append((yi, xi, yk, xk))
    E = (np.array(edges, dtype=np.int64) if edges
         else np.zeros((0, 4), dtype=np.int64))
    return out, E


def _3d_arc_fit(arc, ele2phase, t, meter2rad, max_dh=100.0, max_dv=25.0,
                step_dh=4.0, step_dv=2.0, budget=None, max_seasonal=5.0,
                iterations=8, seed_th=None):
    """Joint (height, velocity) fit on many arcs at once, WITHOUT priors.

    arc     : (n_dates, n_arcs) COMPLEX arc, u_i * conj(u_j). Phase never
              leaves the complex plane here: taking np.angle only to feed
              np.exp(1j.) straight back is a round trip that buys nothing and
              invites the wrapping bugs it looks like it is avoiding.
    ele2phase    : (n_dates,)  height-to-phase factor, B_perp / (R sin theta)
    t       : (n_dates,)  time in years from the reference epoch
    meter2rad    : float       4 pi / wavelength
    max_dh  : metres,  half-width of the height search. These are DIFFERENTIAL
              heights between neighbours a few tens of metres apart, so 200 m
              is already generous.
    max_dv  : mm/yr,   half-width of the rate search, likewise differential.
    step_dh : metres,  lattice step. It sets which BASIN is found, not the
              accuracy: the refinement below is continuous and absorbs the
              quantisation over a wide range of steps.
    step_dv : mm/yr,   lattice step in rate.

    Two stages, because neither alone is both correct and affordable.

    LATTICE. gamma = |sum_d z_d exp(-i theta.u_d)| / n_valid over a grid, as
    one (arcs x candidates) product. The grid is built as arange(-k, k+1)*step
    so it always CONTAINS ZERO: a grid that misses the origin biases every
    solution by half a cell, and at +-200 m with a 16 m step -- 200/16 = 12.5
    -- that took the largest connected component from 862 nodes to 63.

    REFINEMENT. Majorise-minimise on the phasors, seeded at the lattice
    argmax. h(psi) = 1 - cos(psi) has h'' <= 1 for ALL psi, so the unit-
    curvature quadratic is a GLOBAL majoriser and the step

        delta = pinv(U) @ Im(R conj(mu/|mu|)),   U = [hh - mean, tt - mean]

    is non-descending from any start, with no line search, damping or trust
    region. The constant phase is profiled out exactly by the rotation, never
    estimated and never gauged to one epoch -- gauging injects that epoch's
    noise into every other. Against an exhaustive scan of the same space it is
    never below the scan on a single arc.

    Omitting the mu rotation is not a small error: the same refinement without
    it loses coherence outright and fails silently on some arcs.

    WHY NOT A LADDER. Rungs on doubling baselines are discontinuous by
    construction -- each searches a window around wherever the last one landed
    -- so one bad step is never undone, and against an exhaustive scan it
    returns badly wrong heights at a fraction of the achievable coherence.

    WHY NOT A GRID ALONE. A grid argmax is discontinuous in the data: perturb
    every phase slightly and a small fraction of arcs jump a FULL cell while
    the median does not move at all -- the instability is rare, large, and
    invisible in any summary statistic. The refined answer barely moves.

    UNRESOLVABLE ARCS RETURN NaN. Two conditions, both of which detect truth
    planted OUTSIDE the search range without rejecting in-range arcs.

      edge     the solution sits outside max_dh/max_dv, i.e. in the guard band
               the scan adds beyond them, so the truth is past the range the
               caller asked for and the peak is a boundary, not a maximum
      runaway  the refinement travelled more than two lattice cells, so the
               lattice argmax was not the basin the data actually prefer

    Height and rate are always solved TOGETHER: the perpendicular baseline is
    not a smooth function of time, so they separate only jointly -- chained, a
    planted (+20 m, -12.7 mm/yr) comes back as (+370 m, -0.65).

    Invalid samples are zeros and stay zeros, so they drop out of every sum and
    the normalisation counts only what is actually there. Going through angles
    lost this: np.angle(0) is 0 and np.exp(1j.0) is 1, which silently turned
    every masked date into a perfectly coherent observation.

    Returns (gamma, height_rad, velocity_rad_yr, seasonal_rad), each (n_arcs,).
    Height is radians per unit ele2phase and rate is radians per year: the
    library works in phase throughout and only displacement_los()
    converts to a length. `max_dh`/`max_dv` remain physical, since they
    state what the caller wants bounded.

    `seasonal_rad` is COMPLEX and in radians: the annual term contributes
    seasonal.real * cos(2 pi t) + seasonal.imag * sin(2 pi t) to the model
    phase, so abs() is its amplitude and angle() its position.

    THAT POSITION IS MEASURED FROM t = 0, and t = 0 belongs at the MASTER --
    the epoch where B_perp = 0, which is the one moment a single-master stack
    defines: the scene differenced with itself, phase zero by construction, and
    the height term vanishing with the baseline that carries it. Callers here
    build t that way (see _3d_fit_ps_array). Anchoring it elsewhere leaves
    the model referenced to two epochs at once, rotating the annual phase by
    the offset between them for nothing. Rate and height are indifferent
    -- a shift in t adds a constant and constants are profiled out exactly -- so
    only the annual's phase is at stake, and only its phase.

    It is still NOT the day of the year: to land in the calendar add the
    master's day of year, (angle / 2 pi) * 365.25 + doy(t0) modulo the year.
    Reconstructing the fitted model needs no conversion at all, only the same
    `t` the fit was given, so the value is returned unconverted. It
    is returned because a caller cannot otherwise rebuild the model that was
    actually fitted, and whatever is not subtracted stays in the RESIDUAL --
    which is the atmospheric screen. Dropping it leaves the fitted annual in
    that residual, which shows up as temporal correlation in a screen required
    to be white in time; returning and removing it clears that. Zero when
    max_seasonal is 0,
    because the model then holds no annual term -- that is a value and not an
    absence, so it is not NaN. NaN only where the arc is unresolved, alongside
    height and rate.
    """
    arc = np.asarray(arc)
    if not np.iscomplexobj(arc):
        raise TypeError('arc must be the COMPLEX arc u_i * conj(u_j), '
                        f'got {arc.dtype}; passing np.angle() of it is the '
                        'round trip this signature exists to prevent')
    if not (step_dh > 0 and step_dv > 0 and max_dh > 0 and max_dv > 0):
        raise ValueError('max_dh, max_dv, step_dh, step_dv must all be > 0, '
                         f'got {max_dh}, {max_dv}, {step_dh}, {step_dv}')
    n, m = arc.shape

    # SCAN 10% WIDER than the caller asked for, and reject what lands outside
    # THEIR range. max_dv=100 then means exactly what it says: 99 mm/yr is
    # detected, because it sits inside the scan with headroom either side and
    # is nowhere near the boundary where a peak is the edge rather than a
    # maximum. The guard band is what makes the promise exact, and it stays an
    # implementation detail -- the caller never reasons about it.
    #
    # arange(-k, k+1) * step is symmetric about zero BY CONSTRUCTION, so the
    # no-model solution is always a candidate and no solution is biased by a
    # half cell. See the docstring: getting this wrong is catastrophic and
    # entirely silent.
    # the epsilon is not cosmetic: 1.1 * 200 is 220.00000000000003 in binary
    # floating point, so a bare ceil() adds a whole cell at each end whenever
    # the guard lands exactly on the lattice
    # ele2phase=None means the series carries no usable baseline, so the height
    # term is NOT estimated: its grid collapses to {0} and dh comes back NaN.
    # Passing zeros instead would leave dh unconstrained and the runaway gate
    # would then reject every series.
    no_h = ele2phase is None
    # PHASE THROUGHOUT, CONVERTED ONCE AT THE DOOR. `max_dh` and `step_dh` are
    # stated in metres and `max_dv`/`step_dv` in mm/yr because that is what a
    # caller can reason about, but everything inside is radians, as everywhere
    # else in the library. Scaling a bound is exact, so `|dh| > max_dh` and
    # `|dh_rad| > max_dh * meter2rad` are the same gate -- and the constants
    # come out cleaner in phase: the sideband comb below is exactly 2 pi and
    # the saturation limit exactly pi/2.
    #
    # The alternative, converting at the RETURN, put a unit boundary in the
    # middle of the function: `dh` meant metres above it and radians below,
    # and a seeded caller handing back a value it had just been given was
    # wrong by meter2rad with nothing to catch it.
    _m2h = float(meter2rad)                     # rad per metre of dh
    _m2v = float(meter2rad) * 1e-3              # rad/yr per mm/yr of dv
    max_dh_r, step_dh_r = float(max_dh) * _m2h, float(step_dh) * _m2h
    max_dv_r, step_dv_r = float(max_dv) * _m2v, float(step_dv) * _m2v
    kh = 0 if no_h else int(np.ceil(_GUARD * max_dh_r / step_dh_r - 1e-9))
    kv = int(np.ceil(_GUARD * max_dv_r / step_dv_r - 1e-9))
    gh = np.arange(-kh, kh + 1) * step_dh_r
    gv = np.arange(-kv, kv + 1) * step_dv_r
    # The origin is the no-model solution. It being a candidate is what makes
    # "the fit is never worse than not fitting" true, and the refinement is
    # monotone FROM the lattice argmax, so losing it costs that guarantee as
    # well as the half cell. When a grid misses zero the network collapses to a
    # fraction of its nodes with no error raised anywhere, which is why this is
    # asserted rather than trusted.
    assert (gh == 0).any() and (gv == 0).any(), (
        f'search grid lost the origin: dh {gh[0]}..{gh[-1]} step {step_dh}, '
        f'dv {gv[0]}..{gv[-1]} step {step_dv}')
    ncand = gh.size * gv.size

    # The (arcs x candidates) product is the peak by a wide margin; the
    # candidate bank and the (dates x arcs) working copies are small beside
    # it. Arcs are independent, so splitting them changes nothing but memory.
    step = max(1, int(_3d_budget_mb(budget) * 1e6 / max(ncand * 8, 1)))
    if m > step:
        gs = np.empty(m, np.float32)
        hs = np.empty(m)
        vs = np.empty(m)
        ss = np.empty(m, np.complex128)
        for s0 in range(0, m, step):
            sl = slice(s0, min(s0 + step, m))
            gs[sl], hs[sl], vs[sl], ss[sl] = _3d_arc_fit(
                arc[:, sl], ele2phase, t, meter2rad, max_dh, max_dv,
                step_dh, step_dv, budget, max_seasonal,
                iterations,
                seed_th=None if seed_th is None else seed_th[sl])
        return gs, hs, vs, ss

    A = np.abs(arc)
    Z = np.where(A > 0, arc / np.where(A > 0, A, 1.0), 0).astype(np.complex64)
    nv = (A > 0).sum(axis=0)
    del A
    # the model phase is dh_rad * ele2phase_t + dv_rad * t_t, so with the
    # parameters in phase the design columns are the geometry itself
    tt = np.asarray(t, dtype=np.float64)
    hh = (np.zeros_like(tt) if no_h
          else np.asarray(ele2phase, dtype=np.float64))

    # ---- stage 1: lattice, one product ---------------------------------
    P = np.stack(np.meshgrid(gh, gv, indexing='ij'), -1).reshape(-1, 2)
    C = np.exp(-1j * (np.outer(hh, P[:, 0])
                      + np.outer(tt, P[:, 1]))).astype(np.complex64)
    # no division by nv here: it is constant per arc, so it cannot move the
    # argmax, and gamma is computed once at the end from the refined model
    # THE PRODUCT IS BATCHED BY ITS OWN OUTPUT, not by the arcs going in. The
    # callers size their batches so the ARC array fits `budget`, but this
    # allocates (arcs x lattice), which is thousands of times larger -- so the
    # peak here is nothing like what `budget` accounted for, and it is batched
    # down to what `budget` allows.
    # A SEED SKIPS THE LATTICE ENTIRELY. The caller already knows where this
    # arc's optimum is -- from the network, which has solved both ends onto one
    # datum -- so the search that finds the basin has nothing left to find. The
    # refinement below still runs, because the seed is a prediction and the arc
    # is entitled to move within its own cell. What is dropped is the (arcs x
    # candidates) product, which is the whole cost.
    if seed_th is not None:
        # SEEDED: the caller already knows where this arc's optimum is, because
        # the network has solved both ends onto one datum. The search that
        # finds the basin has nothing left to find, so the (arcs x candidates)
        # product -- the whole cost -- is skipped. The refinement below still
        # runs: a seed is a prediction, and the arc is entitled to move within
        # its own cell.
        TH0 = np.ascontiguousarray(
            np.asarray(seed_th, dtype=np.float64).reshape(m, -1))
        if no_h and TH0.shape[1] > 1:
            TH0 = TH0[:, 1:]
        if not (max_seasonal and max_seasonal > 0):
            del C
    else:
        _L = C.shape[1]
        _mb = _3d_budget_mb(budget)
        _blk = max(1, int(_mb * 1024 * 1024 // max(_L * 8, 1)))
        k = np.empty(Z.shape[1], dtype=np.int64)
        for _b0 in range(0, Z.shape[1], _blk):
            _sl = slice(_b0, min(_b0 + _blk, Z.shape[1]))
            k[_sl] = np.argmax(np.abs(Z[:, _sl].T @ C), axis=1)
        if not (max_seasonal and max_seasonal > 0):
            del C                  # kept below: the seasonal stage re-solves on it
        TH0 = (P[k][:, 1:] if no_h else P[k]).astype(np.float64)

    # ---- stage 2: majorise-minimise refinement -------------------------
    U = (np.stack([tt - tt.mean()], 1) if no_h
         else np.stack([hh - hh.mean(), tt - tt.mean()], 1))
    # COLUMN-NORMALISE BEFORE THE PSEUDO-INVERSE. The two columns carry
    # different physical units -- ele2phase is O(1e-4) per metre while dt is
    # O(1) in years -- so U as built is ill-conditioned by four orders of
    # magnitude for no reason other than the choice of units. Some LAPACK
    # builds fail to converge on it (`LinAlgError: SVD did not converge`),
    # and whether they do depends on the thread that gets there, so it
    # surfaces as an intermittent failure deep inside a dask block.
    #
    # Scaling is EXACT, not a tolerance: U = Us diag(sc), so
    # pinv(U) = diag(1/sc) pinv(Us) whenever the columns are non-zero. The
    # refinement below is unchanged; only the conditioning of the solve is.
    _sc = np.linalg.norm(U, axis=0)
    _sc = np.where(_sc > 0, _sc, 1.0)
    PINV = np.linalg.pinv(U / _sc) / _sc[:, None]
    if iterations <= 0:
        # NO REFINEMENT: the lattice argmax IS the answer. The refinement
        # starts at that argmax and stays
        # inside that cell, so the lattice value already orders candidates the
        # way the refined one does -- close enough to choose WHICH partners
        # are worth refining. It is the same code and the same model, just
        # stopped one step early; the chosen few are then refined normally.
        TH = TH0
    else:
        TH = TH0.copy()
        for _ in range(iterations):
            R = _3d_rotate(Z, U @ TH.T)
            mu = R.sum(axis=0)
            R *= np.conj(mu / np.where(np.abs(mu) > 0, np.abs(mu), 1.0))[None, :]
            TH = TH + (PINV @ R.imag).T

    R = _3d_rotate(Z, U @ TH.T)
    gam = (np.abs(R.sum(axis=0)) / np.maximum(nv, 1)).astype(np.float32)

    if max_seasonal and max_seasonal > 0:
        # The annual term as ONE COMPLEX AMPLITUDE C against the carrier
        # exp(2 pi i t) built from the DATES: the model phase gains
        # Re(C exp(2 pi i t)). No angle of the data is taken and nothing wraps.
        #
        # C enters the phase LINEARLY, exactly like (dh, v), so it gets a
        # LATTICE of its own rather than a swarm of refinements: for each
        # candidate sideband the (dh, v) model is divided out and the residual
        # correlated against a grid of complex amplitudes as one product. At
        # max_seasonal = 60 mm the annual phase is 13.6 rad, which Jacobi-Anger
        # spreads over ~14 sidebands either side, so a seeded-MM search would
        # need ~1800 refinements; this needs one GEMM per sideband and a single
        # refinement at the end.
        k_mm = meter2rad * 1e-3                  # radians per mm of LOS
        car = np.exp(2j * np.pi * np.asarray(t, dtype=np.float64))
        # IN PHASE THESE ARE CONSTANTS, not stack properties: one sideband is
        # exactly a cycle of rate, and the linearised amplitude step saturates
        # at a quarter cycle. Carrying them in mm/yr obscured that.
        comb = 2.0 * np.pi                       # sideband spacing, rad/yr
        sat = np.pi / 2.0                        # linearised step saturation, rad
        # sidebands to cover: |k| <~ A in radians, plus margin
        nt = int(np.ceil(k_mm * _GUARD * float(max_seasonal))) + 2
        # amplitude lattice, spaced by the saturation limit so the final
        # refinement never has to travel further than it can
        # SCANNED 10% WIDE AND REJECTED OUTSIDE, exactly as dh and dv are. The
        # bound was previously a hard wall on the lattice, so a fit that wanted
        # more annual than the caller allowed came back PRESSED AGAINST it
        # rather than refused -- and that is the signature of an annual being
        # spent to buy a sideband instead of to describe a real signal: aliased
        # attachments press against the bound while clean ones sit well inside
        # it.
        #
        # The guard band makes `max_seasonal` mean what the other two bounds
        # mean: a solution inside it is admissible and one outside returns NaN,
        # rather than being quietly clipped to the boundary and reported as a
        # maximum. A peak AT the wall is an edge, not an optimum.
        c_guard_r = _GUARD * float(max_seasonal) * k_mm
        c_guard = c_guard_r
        # ONE SEED IS ENOUGH HERE, AND THAT IS MEASURED, NOT ASSUMED. Spaced
        # at `sat` the bank collapses to a SINGLE POINT {0} for every
        # max_seasonal <= 6.3 mm, which looks alarming next to the full lattice
        # (dh, v) get. It is not: the annual is still fitted, because the MM
        # refinement below solves every parameter jointly and continuously --
        # the two seasonal columns included -- starting from that seed, and the
        # objective is unimodal in C over this range (the basin half-width is
        # pi/k_mm = 13.9 mm, wider than any admissible amplitude).
        #
        # Planted annuals on real dates and baselines, 3000 arcs at 0.55 rad of
        # noise, one seed against a 197-point lattice:
        #
        #   planted   1 seed   197 seeds
        #     1.0 mm    1.06      1.05
        #     2.0 mm    2.05      2.04
        #     3.0 mm    3.04      3.03
        #
        # identical to two decimals, |C| error 0.25 mm either way, and the same
        # rate error (0.22 mm/yr). The dense lattice buys nothing and costs a
        # GEMM per bank point per sideband -- at 197 points it exhausted numpy's
        # array limit on a real block. `_SEASONAL_STEPS` can force the dense
        # bank for anyone who wants to re-measure this; 0 keeps the seed.
        c_step = (min(sat, c_guard / _SEASONAL_STEPS)
                  if (_SEASONAL_STEPS and c_guard > 0) else sat)
        ka = int(np.ceil(c_guard / c_step - 1e-9))
        ax = np.arange(-ka, ka + 1) * c_step
        CR, CI = np.meshgrid(ax, ax, indexing='ij')
        keepc = (CR ** 2 + CI ** 2) <= c_guard ** 2 + 1e-9
        CG = np.stack([CR[keepc], CI[keepc]], 1)          # (nC, 2)
        Cbank = np.exp(-1j * (np.outer(car.real, CG[:, 0])
                              + np.outer(car.imag, CG[:, 1]))
                       ).astype(np.complex64)             # (n, nC)
        rate_col = -1 if no_h else 1

        # TWO ANCHORS, because neither is reliable alone.
        #
        # The (dh, v) lattice above never saw the annual term, and at large
        # amplitude its height is meaningless, leaving the whole search far
        # from a truth that would score well.
        #
        # The second anchor exploits the term being exactly ONE YEAR long: two
        # acquisitions a year apart carry the SAME annual phase, so it cancels in
        # their difference while dh and v survive. The residual leakage is a
        # small fraction of the amplitude, which brings a large annual back
        # below the sideband crossover and the height back with it.
        # But every such pair has the same one-year separation, so a rate folds
        # at 2 pi / k = one sideband: that anchor fixes dh and leaves v ambiguous
        # by exactly the spacing the sideband loop already scans.
        #
        # Under noise the year-pair anchor is the WORSE of the two (it spends 115
        # pairs where the full series has 88 dates, each pair carrying two dates'
        # noise): at 0 mm seasonal and sigma 0.8 it aliases 25.5% against 0.0%.
        # So both run and the higher gamma wins -- legitimate here only because
        # the annual is IN the model, which is what makes gamma rank correctly.
        # EFFORT FOLLOWS THE AMPLITUDE. Both the second anchor and the extra
        # alternation rounds exist for the LARGE-amplitude case, where the
        # seasonal-blind (dh, v) lattice is useless: its median height error is
        # 3.5 m at 12 mm of annual, 8.6 m at 30 mm, and 101.9 m at 50 mm. Below
        # that they buy nothing and cost 6x -- each extra round re-runs the full
        # (dh, v) lattice, and each anchor doubles the whole thing.
        ann_rad = k_mm * float(max_seasonal)      # annual phase in radians
        n_rounds = 1 if ann_rad <= 1.435 else (2 if ann_rad <= 4.0
                                              else _SEASONAL_ROUNDS)
        two_anchors = ann_rad > 4.0
        anchors = [TH.copy()]
        dsec = ((np.asarray(t, dtype=np.float64) - t[0]) * 365.25)
        iu_, ju_ = np.triu_indices(n, 1)
        dtp = dsec[ju_] - dsec[iu_]
        ytol = float(np.median(np.diff(np.sort(dsec)))) if n > 1 else 0.0
        ysel = np.abs(dtp - 365.25) <= max(ytol, 1.0)
        if two_anchors and ysel.sum() >= max(8, n // 8):
            ia_, ib_ = iu_[ysel], ju_[ysel]
            Bp = np.stack([hh[ib_] - hh[ia_], tt[ib_] - tt[ia_]], 1)
            bank_p = np.exp(-1j * (np.outer(Bp[:, 0], P[:, 0])
                                   + np.outer(Bp[:, 1], P[:, 1]))
                            ).astype(np.complex64)
            Zp = (Z[ib_] * np.conj(Z[ia_])).astype(np.complex64)
            kp_ = np.argmax(np.abs(Zp.T @ bank_p), axis=1)
            anchors.append((P[kp_][:, 1:] if no_h else P[kp_]).astype(np.float64))
            del bank_p, Zp

        Us = np.concatenate(
            [U, np.stack([car.real, car.imag], 1)], axis=1)
        Us = Us - Us.mean(axis=0, keepdims=True)
        PINVs = np.linalg.pinv(Us)
        # THE SEASONAL-FREE SOLUTION COMPETES. Seeding gbest at -1 discarded it
        # unconditionally, so the annual model won even when it was WORSE, and
        # that is how a whole sideband gets returned: the (dh, v) lattice is
        # scanned across sidebands below and the argmax is taken, so with two
        # extra free parameters a wrong basin can outscore the right one, and
        # attached pixels then come back a whole sideband from their
        # neighbourhood. The same arcs peak next to the neighbourhood when the
        # annual is not free to move them.
        #
        # Seeded with the plain answer the annual has to EARN the basin it
        # moves to, which is what the crossover argument above assumes: below
        # 6.3 mm of real annual {height, rate} is already the higher maximum and
        # must be allowed to stay.
        gbest = gam.astype(np.float64).copy()
        thbest = np.concatenate([TH, np.zeros((m, 2))], axis=1)
        seedbest = TH.copy()

        for TH_a in anchors:
            TH_r = TH_a.copy()
            # ALTERNATE the two lattices: solve (dh, v), solve C, re-solve
            # (dh, v) with C divided out. Each block is solved exactly by its own
            # lattice and each pass can only raise gamma, so this is monotone.
            for _rnd in range(n_rounds):
                bg = np.full(m, -1.0)
                bth = np.zeros((m, U.shape[1] + 2))
                bseed = np.zeros((m, U.shape[1]))
                for kt in range(-nt, nt + 1):
                    TH_k = TH_r.copy()
                    TH_k[:, rate_col] = TH_k[:, rate_col] + kt * comb
                    Rk = _3d_rotate(Z, U @ TH_k.T)
                    G = np.abs(Rk.T @ Cbank)              # (m, nC)
                    kc = np.argmax(G, axis=1)
                    g_ = G[np.arange(m), kc] / np.maximum(nv, 1)
                    up = g_ > bg
                    bg = np.where(up, g_, bg)
                    bth = np.where(up[:, None],
                                   np.concatenate([TH_k, CG[kc]], axis=1), bth)
                    bseed = np.where(up[:, None], TH_k, bseed)
                    del Rk, G
                if _rnd + 1 < n_rounds:
                    Cm = np.exp(1j * (np.outer(car.real, np.ones(m)) * bth[:, -2]
                                      + np.outer(car.imag, np.ones(m)) * bth[:, -1])
                                ).astype(np.complex64)
                    k2 = np.argmax(np.abs((Z * np.conj(Cm)).T @ C), axis=1)
                    TH_r = (P[k2][:, 1:] if no_h else P[k2]).astype(np.float64)
                    del Cm
            THs = bth
            for _ in range(iterations):
                Rs = _3d_rotate(Z, Us @ THs.T)
                mus = Rs.sum(axis=0)
                Rs *= np.conj(mus / np.where(np.abs(mus) > 0,
                                             np.abs(mus), 1.0))[None, :]
                THs = THs + (PINVs @ Rs.imag).T
            Rs = _3d_rotate(Z, Us @ THs.T)
            g_a = np.abs(Rs.sum(axis=0)) / np.maximum(nv, 1)
            up = g_a > gbest
            gbest = np.where(up, g_a, gbest)
            thbest = np.where(up[:, None], THs, thbest)
            seedbest = np.where(up[:, None], bseed, seedbest)
            del Rs, THs, bth, bseed

        gam = gbest.astype(np.float32)
        TH = thbest[:, :U.shape[1]]
        # THE ANNUAL IS KEPT, NOT DROPPED. It was fitted and refined alongside
        # (dh, dv) -- these are the last two columns of the same solution -- and
        # slicing it away here used to end its life. That was not free: the term
        # then stays in the residual, and the residual IS the atmospheric
        # screen. The discarded annual leaves the node screen correlated in
        # time, which is the one property the screen is required not to have.
        # Returning it clears that and removes the energy it carried, since
        # its phase resultant across nodes is 0.064, so it is not one season
        # over the scene and therefore not a stratified delay.
        seas = thbest[:, U.shape[1]] + 1j * thbest[:, U.shape[1] + 1]
        TH0 = seedbest
        del Cbank, C, Us, PINVs, thbest, seedbest

    else:
        # not fitted means the model contains no annual, which is a value and
        # not an absence: zero, where NaN would claim it could not be assessed
        seas = np.zeros(m, dtype=np.complex128)
    if no_h:
        dh, dv = np.full(m, np.nan), TH[:, 0]
    else:
        dh, dv = TH[:, 0], TH[:, 1]

    # An arc we cannot resolve returns NaN, never a plausible number: a wrong
    # value that clears the threshold is invisible to everything downstream,
    # while a NaN is simply not an arc.
    # How far the refinement may travel from its seed before the answer is
    # a different solution rather than a refined one. Plain path: the seeds are
    # lattice cells, and travel stays well inside one, so two cells is
    # generous. Seasonal path: the rate seeds are spaced by a whole sideband,
    # so the basin is half a sideband -- using two cells there rejects many
    # correctly recovered pixels as runaways.
    seasonal = bool(max_seasonal and max_seasonal > 0)
    # THE GATE MUST MATCH THE KIND OF SEED IT IS JUDGING.
    #
    # From a LATTICE seed, two cells is exact reasoning: the grid search found
    # the global maximum, so the optimum is within half a cell and anything
    # travelling further has left its basin. That rule is scale-free because
    # the seed error IS the cell.
    #
    # From a PREDICTED seed it is wrong. The seed is the network's estimate of
    # this arc, and its error is set by the measurements, not by a grid the
    # prediction never touched -- measured at p50 0.48 and p90 1.70 of a
    # default cell. Expressed in cells that gate tightens as the caller
    # refines the grid: at `step_dh=2` the same physical travel becomes 3.4
    # cells and is rejected, which cost 26% of the pixels for no reason in the
    # data. So a predicted seed is judged against a fixed physical distance,
    # stated as the default cell, and the caller's step no longer decides who
    # survives.
    _PRED_H, _PRED_V = 4.0 * _m2h, 2.0 * _m2v      # the documented defaults
    if seed_th is None:
        rate_tol = np.pi if seasonal else 2.0 * step_dv_r
        dh_tol = 2.0 * step_dh_r
    else:
        rate_tol = np.pi if seasonal else 2.0 * _PRED_V
        dh_tol = 2.0 * _PRED_H
    if seasonal and not no_h:
        # Height and rate are CORRELATED in this design, so when the refinement
        # moves the rate to another sideband the height must follow, dragged
        # by their correlation. Gating dh at two lattice cells then rejects
        # pixels whose rate is correct. The tolerance follows the coupling
        # instead of a fixed cell count.
        cor = abs(float(np.corrcoef(hh, tt)[0, 1]))
        dh_tol = max(dh_tol, cor * (np.std(tt) / max(np.std(hh), 1e-30))
                     * rate_tol)
    edge = np.abs(dv) > max_dv_r
    if max_seasonal and max_seasonal > 0:
        # the annual is bounded like the other two: outside the range the
        # caller stated, the answer is NaN and not a clipped one
        edge = edge | (np.abs(seas) > c_guard_r / _GUARD)
    runaway = np.abs(dv - TH0[:, -1]) > rate_tol
    if not no_h:
        edge = edge | (np.abs(dh) > max_dh_r)
        runaway = runaway | (np.abs(dh - TH0[:, 0]) > dh_tol)
    bad = (nv < 1) | edge | runaway
    gam = np.where(bad, np.nan, gam).astype(np.float32)
    # RADIANS OUT. The gates above run in the units the ARGUMENTS are stated in
    # -- max_dh in metres, max_dv in mm/yr -- because that is what the caller
    # asked to bound. Everything downstream works in phase, so the conversion
    # happens once, here, and nothing converts again: velocity() returns this
    # value untouched and displacement_los() is the single place a length is
    # produced. The round trip it replaces (mm -> rad in velocity(), rad -> m in
    # displacement_los) applied a sign convention at each end, which is where
    # the sign of the reported rate became hard to trace.
    return (gam,
            np.where(bad, np.nan, dh),                 # rad per unit ele2phase
            np.where(bad, np.nan, dv),                 # rad/yr
            np.where(bad, np.nan, seas))               # rad, complex


def _3d_arc_fit_brute(arc, ele2phase, t, meter2rad, h_range=150.0, v_range=60.0,
                      h_step=0.5, v_step=0.25, budget=None):
    """Exhaustive (height, rate) scan -- the REFERENCE the ladder is checked against.

    _3d_arc_fit walks a coarse-to-fine ladder and can land in the wrong basin;
    that is not hypothetical, it happened: it returned a low coherence at a
    wildly wrong height where a much better solution existed nearby. Nothing in
    the ladder detects that, because a search cannot
    report a maximum it never visited. An exhaustive scan can.

    Every candidate is scored for every arc as ONE matrix product,

        gamma(a, c) = |sum_d Z[d, a] conj(E[d, c])| / n_valid

    with E the (dates x candidates) model bank, so it is a GEMM like the arc
    kernel's, and a full grid over thousands of arcs stays cheap.

    UNLIKE the ladder this takes a RANGE, which is a prior. That is why it is a
    reference and not the estimator: the ladder's search window is set by the
    baselines alone. Use it to verify, to debug a suspect pixel, or in a test
    that asserts the ladder finds what is there.

    Returns (gamma, height_rad, velocity_rad_yr), each (n_arcs,).
    Height is radians per unit ele2phase and rate is radians per year: the
    library works in phase throughout and only displacement_los()
    converts to a length. `max_dh`/`max_dv` remain physical, since they
    state what the caller wants bounded.
    """
    arc = np.asarray(arc)
    if not np.iscomplexobj(arc):
        raise TypeError(f'arc must be COMPLEX phasors, got {arc.dtype}')
    A = np.abs(arc)
    Z = np.where(A > 0, arc / np.where(A > 0, A, 1.0), 0).astype(np.complex64)
    n, m = Z.shape
    nv = np.maximum((A > 0).sum(axis=0), 1)
    # PHASE THROUGHOUT, as `_3d_arc_fit` and `_3d_pair_fit` do -- the ranges
    # and steps are the caller's, stated in metres and mm/yr, and converted
    # once here so the scan, the gates and the return all speak one unit.
    hh = np.asarray(ele2phase, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    _m2h, _m2v = float(meter2rad), float(meter2rad) * 1e-3
    # (-k, k+1) * step rather than arange(-range, range, step): the latter
    # MISSES THE ORIGIN whenever the step does not divide the range -- 200 m
    # in 3 m steps runs .. -2, 1, 4 .. -- which biases every solution by up to
    # half a cell and silently removes the no-model candidate.
    kh = int(np.ceil(float(h_range) / float(h_step) - 1e-9))
    kv = int(np.ceil(float(v_range) / float(v_step) - 1e-9))
    gh = np.arange(-kh, kh + 1) * float(h_step) * _m2h
    gv = np.arange(-kv, kv + 1) * float(v_step) * _m2v
    assert (gh == 0).any() and (gv == 0).any(), (
        f'scan grid lost the origin: dh {gh[0]}..{gh[-1]} step {h_step}, '
        f'dv {gv[0]}..{gv[-1]} step {v_step}')

    best_g = np.zeros(m, np.float32)
    best_h = np.zeros(m)
    best_v = np.zeros(m)
    # one velocity at a time keeps the model bank to (n x n_h): the full
    # product would be (n x n_h n_v) and that is where the memory goes
    cap = _3d_budget_mb(budget) * 1024 * 1024
    hchunk = max(1, min(len(gh), int(cap // max(n * 8 * 4, 1))))
    for v0 in gv:
        Zv = Z * np.exp(-1j * (tt * v0))[:, None]
        for a in range(0, len(gh), hchunk):
            gsub = gh[a:a + hchunk]
            E = np.exp(-1j * np.outer(hh, gsub)).astype(np.complex64)
            sc = np.abs(Zv.T @ E) / nv[:, None]        # (m, n_h)
            k = np.argmax(sc, axis=1)
            g = sc[np.arange(m), k]
            up = g > best_g
            best_g = np.where(up, g, best_g)
            best_h = np.where(up, gsub[k], best_h)
            best_v = np.where(up, v0, best_v)
    return best_g.astype(np.float32), best_h, best_v





def _3d_budget_mb(budget):
    """Working-set budget in MB; None reads the dask chunk size.

    The project sizes every transient against `array.chunk-size` so one
    setting governs the whole pipeline on a given machine. A hardcoded default
    -- this returned 1024 MB regardless -- silently ignored that, and on a
    machine configured for small chunks it would allocate eight times what the
    caller asked for.
    """
    if budget is not None:
        return float(budget)
    from .utils_dask import get_dask_chunk_size_mb
    return float(get_dask_chunk_size_mb())





def _3d_arc_batch(Us, Ut, src, tgt, ele2phase, t, meter2rad, max_dh, max_dv,
                  step_dh, step_dv, budget, iterations, seed_th=None,
                  threads=1):
    """Fit every (src, tgt) arc between two phasor sets, in budgeted batches.

    NO DIFFERENTIAL ANNUAL. Both callers attach a pixel to a neighbour tens of
    metres away, and there is no seasonal GRADIENT at that scale: stratified
    and thermal delay vary with elevation and over kilometres, not across a
    courtyard. Fitting one is fitting noise, and two free parameters on a
    marginal arc buy enough coherence to carry the rate a whole sideband away.

    threads : arcs are independent, so slices of them fit concurrently; the
    GEMMs and ufuncs release the GIL. Only a caller that OWNS the host may
    raise it -- the fit3d gate does, a per-chunk dask task must not.
    """
    n = Us.shape[0]
    ga = np.empty(len(src), np.float32)
    dha = np.empty(len(src)); dva = np.empty(len(src))
    dsa = np.empty(len(src), np.complex128)
    _th = max(1, int(threads))
    if _th > 1 and len(src) > _th:
        from concurrent.futures import ThreadPoolExecutor
        _mb = _3d_budget_mb(budget) / _th
        bnd = np.linspace(0, len(src), _th * 4 + 1).astype(np.int64)

        def _slice(i):
            sl = slice(bnd[i], bnd[i + 1])
            ga[sl], dha[sl], dva[sl], dsa[sl] = _3d_arc_batch(
                Us, Ut, src[sl], tgt[sl], ele2phase, t, meter2rad, max_dh,
                max_dv, step_dh, step_dv, _mb, iterations,
                seed_th=None if seed_th is None else seed_th[sl])
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_slice, range(_th * 4)))
        return ga, dha, dva, dsa
    step = max(1, int(_3d_budget_mb(budget) * 1024 * 1024 // max(n * 16, 1)))
    for b0 in range(0, len(src), step):
        sl = slice(b0, min(b0 + step, len(src)))
        arc = np.ascontiguousarray(
            (Us[:, src[sl]] * np.conj(Ut[:, tgt[sl]])).astype(np.complex64))
        ga[sl], dha[sl], dva[sl], dsa[sl] = _3d_arc_fit(
            arc, ele2phase, t, meter2rad, max_dh, max_dv, step_dh, step_dv,
            budget, 0.0, iterations=iterations,
            seed_th=None if seed_th is None else seed_th[sl])
    return ga, dha, dva, dsa


def _3d_ds_solve(n_ds, ei, ep, e_dv, e_dh, e_g, ni, nj, p_dv, p_dh, p_g,
                 ps_vel, ps_hgt, err_v, err_h, passes):
    """Rate and height for every DS at once, from all of its equations.

        DS_i - PS_p = dv_ip     PS FIXED, so this pins the DS to the datum
        DS_i - DS_j = dv_ij     what relates one DS to the next

    Returns (vel, hgt, n_surviving, n_anchor) -- the values and, per DS, how
    many of its own equations came through the gate, and how many of THOSE
    tie it to the fixed layer rather than to a peer at its own level.

    The PS are held rather than solved with the DS. They are the certified
    layer and they carry the datum; a hundred thousand weak DS solved jointly
    with a few hundred nodes would outvote the network that anchors them.
    Holding them also leaves this system with no free constant, so nothing
    here can drift as a body.

    REJECT, RE-SOLVE, RE-CHECK, WITH THE SCALE HELD -- as the node network
    does. Reweighting alone leaves an outlier pulling and its error spread
    over the neighbours it contradicts; only removal stops that, and the scale
    must not be re-estimated from the survivors or the gate feeds on itself
    and erodes the network instead of settling.

    RATE AND HEIGHT ARE JUDGED TOGETHER, also as the node network does. Both
    are solved every pass and an arc answers for the worse of its two
    residuals, each against its own scale. Judged on the rate alone, an arc
    that is metres out in height -- a facade, a roof edge, two scatterers at
    different elevations inside one window -- passes the gate and its height
    enters the answer unchallenged, so the heights could not be trusted even
    where the rates were sound.
    """
    # a list per CALL, published per THREAD: the two lsqr threads inside
    # _both append through the closure, while concurrent attach blocks in
    # other worker threads each publish their own list -- a bare function
    # attribute was one shared list, and one block's reset wiped another's
    conv = []
    _3d_ds_solve._tl.conv = conv
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import lsqr
    m1, m2 = len(ei), len(ni)
    if m1 == 0:
        return (np.full(n_ds, np.nan), np.full(n_ds, np.nan),
                np.zeros(n_ds, np.int64))
    rows = np.r_[np.arange(m1), np.repeat(np.arange(m1, m1 + m2), 2)]
    cols = np.r_[ei, np.c_[ni, nj].ravel()]
    vals = np.r_[np.ones(m1), np.tile([1.0, -1.0], m2)]
    G = coo_matrix((vals, (rows, cols)), shape=(m1 + m2, n_ds)).tocsr()
    w0 = np.r_[np.asarray(e_g, float), np.asarray(p_g, float)]
    rhs_v = np.r_[ps_vel[ep] + e_dv, p_dv]
    rhs_h = np.r_[ps_hgt[ep] + e_dh, p_dh]

    def _solve(Gm, r, w):
        # LSQR STARTS AT ZERO, so stopping early is shrinkage toward zero --
        # a Landweber-type regularisation nobody asked for. `istop` and `itn`
        # say whether that happened; discarding them hides it.
        sw = np.sqrt(np.maximum(w, 0.0))
        _o = lsqr(diags(sw) @ Gm, sw * r, atol=1e-10, btol=1e-10,
                  iter_lim=500)
        conv.append((int(_o[1]), int(_o[2]), int(Gm.shape[1])))
        return _o[0]

    def _both(Gm, w_, rv, rh):
        """Rate and height together: two solves, one matrix, two threads.

        The two right-hand sides share the matrix and depend on nothing of
        each other's, and `lsqr` spends its time in kernels that release the
        GIL, so the pair costs little more than one of them. That matters
        because the gate now judges both residuals, so BOTH are solved on
        every pass rather than height once at the end.
        """
        import threading
        out = [None, None]

        def _run(i, r):
            out[i] = _solve(Gm, r, w_)

        th = (threading.Thread(target=_run, args=(0, rv)),
              threading.Thread(target=_run, args=(1, rh)))
        for t in th:
            t.start()
        for t in th:
            t.join()
        return out[0], out[1]

    def _mad(r):
        return max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-12)

    # BOTH RESIDUALS DECIDE, as the node network's gate does. An arc carries a
    # differential rate AND a differential height, and the two fail
    # independently: a facade or a roof edge can be metres out in height while
    # its rate looks ordinary, and judged on the rate alone that arc is kept
    # and its height goes into the answer unchallenged. Each residual is
    # scored against its own scale, since a metre and a millimetre per year
    # are not comparable numbers, and the worse of the two is what the arc is
    # judged by.
    def _z(Gm, xv, xh, rv, rh, s_v=None, s_h=None):
        r_v = Gm @ xv - rv
        r_h = Gm @ xh - rh
        # THE SCALE IS THE BOUND THE CALLER STATED, not one read off the
        # residuals: a set that is uniformly wrong produces a wide robust
        # sigma and passes itself. Same rule as the network and the vote.
        s_v = err_v if s_v is None else s_v
        s_h = err_h if s_h is None else s_h
        return np.maximum(np.abs(r_v) / s_v, np.abs(r_h) / s_h), s_v, s_h

    w = w0.copy()
    xv, xh = _both(G, w, rhs_v, rhs_h)
    for _ in range(max(1, int(passes))):
        z, _, _ = _z(G, xv, xh, rhs_v, rhs_h)
        w = w0 / np.maximum(z, 1.0)
        xv, xh = _both(G, w, rhs_v, rhs_h)
    live = np.ones(m1 + m2, dtype=bool)
    if err_v is not None:
        # the scales are taken once and HELD, so the gate cannot feed on its
        # own survivors -- the same rule the reweighting above answers to
        _, s_v, s_h = _z(G, xv, xh, rhs_v, rhs_h)
        for _ in range(max(1, int(passes))):
            idx = np.flatnonzero(live)
            xs_v, xs_h = _both(G[idx], w[idx], rhs_v[idx], rhs_h[idx])
            zz, _, _ = _z(G[idx], xs_v, xs_h, rhs_v[idx], rhs_h[idx],
                          s_v, s_h)
            keep = zz <= 1.0
            xv, xh = xs_v, xs_h
            if keep.all():
                break
            nl = np.zeros(m1 + m2, dtype=bool)
            nl[idx[keep]] = True
            if nl.sum() < 2:
                break
            live = nl
    idx = np.flatnonzero(live)
    vel, hgt = _both(G[idx], w[idx], rhs_v[idx], rhs_h[idx])
    # TWO KINDS OF EQUATION, COUNTED APART. `DS - PS` ties a pixel to the
    # FIXED layer and so to the datum; `DS - DS` is purely relative and ties
    # it only to its neighbours. Summed together a cluster can clear any
    # threshold on its own internal edges alone -- and with no surviving
    # anchor its block of the system is rank-deficient in the datum
    # direction, so lsqr returns the minimum-norm answer and the whole
    # cluster is pulled toward zero.
    nsurv = np.zeros(n_ds, dtype=np.int64)
    nanch = np.zeros(n_ds, dtype=np.int64)
    np.add.at(nanch, ei[live[:m1]], 1)
    nsurv += nanch
    if m2:
        l2 = live[m1:]
        np.add.at(nsurv, ni[l2], 1)
        np.add.at(nsurv, nj[l2], 1)
    return vel, hgt, nsurv, nanch


_3d_ds_solve._tl = _threading.local()


def _3d_pair_fit(Us, Ut, src, tgt, ele2phase, t, err_v, err_h, passes,
                 budget, threads=1):
    """Differential height and rate for SHORT arcs, without a lattice.

    Both ends are already aligned by their own attachment, so there is nothing
    to align; and over metres the differential cannot wrap, which is what a
    lattice search exists to resolve. Searching +-50 mm/yr in 2 mm/yr steps
    would quantise a sub-millimetre difference and its argmax could land in a
    neighbouring cell, inventing a differential where the truth is near zero.

    So the estimate is a linear fit of the differential PHASE on (elevation,
    time). The design has two columns and is the same for every arc, so the
    normal equations are 2x2 and solve in closed form, vectorised over all
    arcs at once -- no per-arc loop and no grid.

    IRLS WITH EXCLUSION over the dates, because a pair has as many epochs as
    the stack is deep and a few bad ones would otherwise set the answer. The
    phasor fits elsewhere are robust by construction -- an outlier date is a
    bounded rotation on a unit circle -- but a least squares on the angle has
    no such protection.

    Returns (gamma, dh, dv) with gamma computed from the residuals AFTER
    exclusion, so an arc that is sound apart from two epochs ranks on what it
    actually is.
    """
    a1 = np.asarray(ele2phase, dtype=np.float64)
    a2 = np.asarray(t, dtype=np.float64)
    m = len(src)
    g = np.empty(m, np.float32)
    dh = np.empty(m); dv = np.empty(m)
    _th = max(1, int(threads))
    if _th > 1 and m > _th:
        # arcs are independent; slices of them fit concurrently
        from concurrent.futures import ThreadPoolExecutor
        _mb = _3d_budget_mb(budget) / _th
        bnd = np.linspace(0, m, _th * 4 + 1).astype(np.int64)

        def _slice(i):
            sl = slice(bnd[i], bnd[i + 1])
            g[sl], dh[sl], dv[sl] = _3d_pair_fit(
                Us, Ut, src[sl], tgt[sl], ele2phase, t, err_v, err_h,
                passes, _mb)
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_slice, range(_th * 4)))
        return g, dh, dv
    step = max(1, int(_3d_budget_mb(budget) * 1024 * 1024
                      // max(len(a1) * 64, 1)))
    for b0 in range(0, m, step):
        sl = slice(b0, min(b0 + step, m))
        ph = np.angle(Us[:, src[sl]] * np.conj(Ut[:, tgt[sl]])).astype(np.float64)
        w = np.ones_like(ph)
        th_h = th_v = res = None
        for _ in range(max(1, int(passes))):
            Sxx = (w * a1[:, None] ** 2).sum(0)
            Sxy = (w * (a1 * a2)[:, None]).sum(0)
            Syy = (w * a2[:, None] ** 2).sum(0)
            bx = (w * a1[:, None] * ph).sum(0)
            by = (w * a2[:, None] * ph).sum(0)
            det = Sxx * Syy - Sxy * Sxy
            det = np.where(np.abs(det) > 1e-30, det, 1e-30)
            th_h = (Syy * bx - Sxy * by) / det
            th_v = (Sxx * by - Sxy * bx) / det
            res = ph - (a1[:, None] * th_h + a2[:, None] * th_v)
            sc = np.maximum(1.4826 * np.median(
                np.abs(res - np.median(res, 0)), 0), 1e-9)
            z = np.abs(res) / sc
            w = (np.where(z <= 1.0, 1.0 / np.maximum(z, 1.0), 0.0)
                 if err_v is not None else 1.0 / np.maximum(z, 1.0))
        g[sl] = np.abs(np.exp(1j * res).mean(axis=0))
        dh[sl] = th_h
        dv[sl] = th_v
        del ph, w, res
    return g, dh, dv



def _3d_lap(t0):
    """Seconds since `t0`, and the mark for the next stage.

    Stage timings ride the debug stream because the work runs inside dask
    tasks on other processes, where a profiler attached to the caller sees
    nothing. Returned as a pair so a caller can print one stage and start the
    next from the same instant, leaving no gap between them.
    """
    _t = time.monotonic()
    return _t - t0, _t

def _3d_model_removed(U, ele2phase, t, h, v):
    """Divide each column's OWN fitted model out of its phasors.

    An arc's model phase is `dh * e2p_t + dv * t_t` with dh = h_i - h_p, so the
    exponential splits and each end can be corrected on its own. With both ends
    corrected the arc coherence collapses to a plain inner product,

        gamma_ip = |sum_t u~_i,t conj(u~_p,t)| / n

    and the two-parameter search that used to find it has nothing left to find.

    HEIGHT AND RATE ONLY -- the annual stays. It is long-wavelength, so it is
    shared across an arc and cancels there; dividing it out of one end alone
    would leave the other end's annual exposed as a residual the model has no
    term for, and the arc pays its full amplitude in lost coherence.
    """
    return (np.asarray(U) * np.exp(-1j * (np.outer(ele2phase, h)
                                          + np.outer(t, v)))).astype(np.complex64)


def _3d_predict_gamma(Us_c, Ut_c, src, tgt, budget, threads=1):
    """Arc coherence at the PREDICTED model, for corrected phasors.

    Batched, because the gather is what costs memory here: the product itself
    is one column per arc, but `Us_c[:, src]` materialises (dates x arcs).
    """
    n = Us_c.shape[0]
    out = np.empty(len(src), dtype=np.float32)
    _th = max(1, int(threads))
    if _th > 1 and len(src) > _th:
        # batches are independent; einsum releases the GIL
        from concurrent.futures import ThreadPoolExecutor
        bnd = np.linspace(0, len(src), _th * 4 + 1).astype(np.int64)

        def _slice(i):
            sl = slice(bnd[i], bnd[i + 1])
            out[sl] = _3d_predict_gamma(Us_c, Ut_c, src[sl], tgt[sl],
                                        _3d_budget_mb(budget) / _th)
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_slice, range(_th * 4)))
        return out
    step = max(1, int(_3d_budget_mb(budget) * 1024 * 1024 // max(n * 32, 1)))
    for b0 in range(0, len(src), step):
        sl = slice(b0, min(b0 + step, len(src)))
        out[sl] = (np.abs(np.einsum('tj,tj->j', Us_c[:, src[sl]],
                                    np.conj(Ut_c[:, tgt[sl]]))) / n)
    return out


def _3d_rotate(Z, X):
    """`Z * exp(-i X)` in complex64, without the complex128 detour.

    `np.exp(-1j * X)` evaluates the transcendental in float64 and builds a
    complex128 array at sixteen bytes an element, which is then copied down.
    cos and sin written straight into the halves of a complex64 result skip
    both, and the reduction below is what makes the narrower evaluation safe.
    """
    # REDUCED BEFORE IT IS NARROWED. cos and sin are accurate in float32 only
    # for a small argument, and these phases run to hundreds of radians, where
    # a float32 argument costs three decimal digits. Folding into [-pi, pi] in
    # float64 first keeps the error at the float32 floor whatever the phase.
    x = np.asarray(X, np.float64)
    x = (x - (2 * np.pi) * np.round(x / (2 * np.pi))).astype(np.float32)
    e = np.empty(x.shape, np.complex64)
    np.cos(x, out=e.real)
    np.sin(x, out=e.imag)
    np.negative(e.imag, out=e.imag)
    e *= Z
    return e


@_numba.njit(nogil=True, cache=True)
def _3d_topk_stream(score, src, nsrc, k):
    """One pass of replace-the-minimum per source -- the lexsort's answer
    at O(arcs x k) instead of a sort of the whole graph. Strict `>` keeps
    the EARLIEST arc on ties, exactly as the stable sort did."""
    vals = np.full((nsrc, k), -np.inf, np.float64)
    idxs = np.full((nsrc, k), -1, np.int64)
    for a in range(len(score)):
        v = score[a]
        if not np.isfinite(v):
            continue
        s = src[a]
        mi = 0
        mv = vals[s, 0]
        for j in range(1, k):
            if vals[s, j] < mv or (vals[s, j] == mv
                                   and idxs[s, j] > idxs[s, mi]):
                mv = vals[s, j]; mi = j
        if v > mv:
            # evict the LATEST arc among the minimum slots, so ties keep
            # the earliest -- the stable sort's choice
            vals[s, mi] = v
            idxs[s, mi] = a
    return idxs


def _3d_topk_per_src(score, src, nsrc, k):
    """Indices of the best `k` arcs of each source, NaN ranked last."""
    idxs = _3d_topk_stream(np.asarray(score, np.float64),
                           np.asarray(src, np.int64), int(nsrc), int(k))
    out = idxs.ravel()
    return out[out >= 0]


def _3d_reach_tiles(pos):
    """DS candidates by tile, with the nodes that tile can reach.

    `pos` is `(src_y, src_x, tgt_y, tgt_x, ry, rx, hy, hx)` in pixels: the
    candidate and node positions, the HALF PS extent -- the same radius the
    node-to-node arcs get from their Chebyshev query, so a DS reaches exactly
    as far as a PS does -- and the DS window, which is the tile.

    A TILE OF CANDIDATES IS ANSWERED BY ONE DENSE PRODUCT. The operand is the
    nodes within the PS extent of the tile's own bounds, and since every pixel
    in the tile lies inside those bounds, its whole window is inside the
    operand: no pair in the product needs testing, and none that belongs is
    missing. The reach the tile adds is its own half-size, and a tile is
    counted in DS windows -- the scale the DS window is DEFINED by, the area
    over which the atmospheric phase does not change -- so the extent and the
    extent grown by a tile stand in the same atmosphere and the arcs mean the
    same thing.

    THE TILE IS THE DS WINDOW, and nothing else has to be decided. Larger
    tiles reach further and so rank each candidate against more nodes, which
    costs more than the products it saves; smaller ones cut the product into
    pieces too thin to be worth a call. Neither margin is close, so there is
    no size to tune and no memory to budget: a window's worth of candidates
    against the nodes it reaches is a few megabytes whatever the block is,
    and it falls out of the two windows the caller already declared.
    """
    sy, sx, ty, tx, ry, rx, wy, wx = pos
    wy, wx = max(int(wy), 1), max(int(wx), 1)
    kx = (sx // wx).astype(np.int64)
    key = (sy // wy).astype(np.int64) * (int(kx.max()) + 1) + kx
    order = np.argsort(key, kind='stable')
    ks = key[order]
    cut = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for a, b in zip(cut[:-1], cut[1:]):
        idx = order[a:b]
        _y, _x = sy[idx], sx[idx]
        tsel = np.nonzero((ty >= _y.min() - ry) & (ty <= _y.max() + ry)
                          & (tx >= _x.min() - rx) & (tx <= _x.max() + rx))[0]
        yield idx, tsel


def _3d_shortlist_ds_ps(Us, Ut, base_lab, nsrc, ele2phase, t, meter2rad,
                        max_dh, max_dv, step_dh, step_dv, budget,
                        iterations, min_agreeing, threshold,
                        stats=None, prefix='ds_', debug=False,
                        fix_h=None, fix_v=None, threads=1, pos=None):
    """DS to PS over the 3x3 DS-WINDOW neighbourhood, every candidate fitted.

    THE WINDOW LATTICE IS THE NEIGHBOURHOOD. A candidate takes the nodes whose
    DS window is its own or one of the eight around it. That is an integer
    index test, not a distance, so there is no reach to tune and no geometry to
    compute; and since the winner grid holds one node per half-window cell, a
    window offers at most four and the candidate set is bounded at thirty-six
    by construction.

    THE BOUND IS WHY NOTHING IS RANKED. A ranking exists to avoid fitting, and
    with a bounded set the fit costs less than the machinery that would choose
    within it -- so there is no raw score, no seed, and no provisional model.
    Removing them also removes an error. A raw score is a coherence, so it
    decays with the arc's own height at `ele2phase * meter2rad` radians per
    metre: a tall scatterer's true partner scores like noise and ranks last,
    which is exactly where the height is the thing being measured. A ranking
    built on it must also estimate the candidate's model before it can rank,
    and estimating that model from whatever the reach admits makes the ruler
    move when the reach does. Fitting every candidate cannot make either
    mistake.

    Returns (ksrc, ktgt, ga, dha, dva, dsa, good), aligned.
    """
    if pos is None:
        raise ValueError('_3d_shortlist_ds_ps needs `pos` to name the windows')
    dy_, dx_, ny_, nx_, _ry, _rx, wy, wx = pos
    dy_, dx_ = np.asarray(dy_), np.asarray(dx_)
    ny_, nx_ = np.asarray(ny_), np.asarray(nx_)
    wy, wx = max(int(wy), 1), max(int(wx), 1)
    # THE NODES ARE BUCKETED BY WINDOW ONCE, then each of the nine offsets is a
    # pair of `searchsorted` bounds into that order -- the candidates never
    # meet the nodes as a product, so nothing of graph length is materialised.
    kdy, kdx = dy_ // wy, dx_ // wx
    kpy, kpx = ny_ // wy, nx_ // wx
    _stride = int(max(int(kdx.max(initial=0)), int(kpx.max(initial=0)))) + 3
    _order = np.argsort((kpy + 1) * _stride + (kpx + 1), kind='stable')
    _sorted = ((kpy + 1) * _stride + (kpx + 1))[_order]
    _ks, _kt = [], []
    for _oy in (-1, 0, 1):
        for _ox in (-1, 0, 1):
            _k = (kdy + 1 + _oy) * _stride + (kdx + 1 + _ox)
            _lo = np.searchsorted(_sorted, _k, 'left')
            _hi = np.searchsorted(_sorted, _k, 'right')
            _cnt = _hi - _lo
            _has = np.flatnonzero(_cnt > 0)
            if not len(_has):
                continue
            _c = _cnt[_has]
            _within = (np.arange(int(_c.sum()))
                       - np.repeat(np.r_[0, np.cumsum(_c)[:-1]], _c))
            _ks.append(np.repeat(_has.astype(np.int64), _c))
            _kt.append(_order[np.repeat(_lo[_has], _c) + _within]
                       .astype(np.int64))
    ksrc = np.concatenate(_ks) if _ks else np.empty(0, np.int64)
    ktgt = np.concatenate(_kt) if _kt else np.empty(0, np.int64)
    if len(ksrc):
        # the offsets arrive one ring at a time; the caller indexes by
        # candidate, so restore that order
        _o = np.lexsort((ktgt, ksrc))
        ksrc, ktgt = ksrc[_o], ktgt[_o]

    ga = np.full(len(ksrc), np.nan, np.float32)
    dha = np.full(len(ksrc), np.nan)
    dva = np.full(len(ksrc), np.nan)
    dsa = np.zeros(len(ksrc), dtype=np.complex128)
    if len(ksrc):
        ga, dha, dva, dsa = _3d_arc_batch(
            Us, Ut, ksrc, ktgt, ele2phase, t, meter2rad, max_dh, max_dv,
            step_dh, step_dv, budget, iterations, threads=threads)
        ga = np.asarray(ga, np.float32)
    good = np.isfinite(ga) & (ga >= float(threshold)) & np.isfinite(dha)

    # ONE COMPONENT PER CANDIDATE. Components carry their own free datum, so a
    # candidate holding partners from two of them measures the offset between
    # the datums rather than its own value. The component is named by the
    # candidate's best partner, as it was when a score chose it -- but the
    # score is now the fitted coherence, which is what the choice meant.
    _lb = np.asarray(base_lab, np.int64)
    if len(ksrc) and len(np.unique(_lb)) > 1:
        _rank = np.where(good, ga, -np.inf)
        _o = np.lexsort((-_rank, ksrc))
        _first = _o[np.r_[True, np.diff(ksrc[_o]) > 0]]
        _first = _first[good[_first]]
        _own = np.full(int(nsrc), -1, np.int64)
        _own[ksrc[_first]] = _lb[ktgt[_first]]
        good &= (_own[ksrc] >= 0) & (_lb[ktgt] == _own[ksrc])

    if stats is not None:
        stats[prefix + 'ranked_arcs'] = 0
        stats[prefix + 'searched_arcs'] = int(len(ksrc))
        stats[prefix + 'provisional'] = int(len(np.unique(ksrc))) if len(ksrc) else 0
    return ksrc, ktgt, ga, dha, dva, dsa, good


def _3d_seed_ds_ds(Us, Ut, src, tgt, nsrc, k, budget, threads=1):
    """DS to DS: the best `k` partners of every candidate, over the LISTED arcs.

    The vouching window is a small box on purpose -- a candidate is attached by
    neighbours it can be compared against, and short arcs are the better ones.
    The graph is therefore sparse, and a dense product would score pairs the
    window deliberately excluded.
    """
    g0 = _3d_predict_gamma(np.asarray(Us, np.complex64), Ut, src, tgt,
                           budget, threads=threads)
    sd = _3d_topk_per_src(g0, src, nsrc, k)
    return src[sd], tgt[sd]


def _3d_partner_shortlist(Us, Ut, src, tgt, base_lab, nsrc, ele2phase, t,
                          meter2rad, max_dh, max_dv, step_dh, step_dv, budget,
                          iterations, min_agreeing, threshold,
                          stats=None, prefix='ds_', debug=False,
                          fix_h=None, fix_v=None, seed=None, threads=1):
    """Rank every arc, name ONE component, refine the best `min_agreeing`.

    RANK FIRST, REFINE THE SHORTLIST. Only `min_agreeing` partners are used, so
    refining all of them spends the larger half of the fit on candidates that
    are discarded. Ranking needs the model only well enough to order it, and
    the refinement cannot leave its own lattice cell, so it reorders
    neighbours at most.

    ONE PASS TO RANK, AND THAT IS A CORRECTNESS FLOOR RATHER THAN A SETTING.
    The lattice scores every candidate at a QUANTISED model, so neighbours can
    share a grid point and tie exactly. One pass takes each onto its own
    optimum, which is what makes the comparison mean anything.

    Returns (ga, dha, dva, dsa, good).
    """
    if fix_h is not None:
        # ---- SEED, PREDICT, SCORE --------------------------------------
        # The partners are already solved onto one datum, so their model
        # divides out and an arc's coherence becomes a plain inner product:
        # gamma = |sum_t u~_i conj(u~_p)| / n, n multiply-adds and no lattice.
        # What the pixel still needs is its OWN (h, v), and that is the only
        # thing a search is spent on.
        Ut_c = _3d_model_removed(Ut, ele2phase, t, fix_h, fix_v)

        # (a) a cheap prior, to choose what to search. Scoring against the
        # corrected partner while assuming the pixel carries no offset is
        # already far better than raw coherence, and it only has to be good
        # enough to put `consensus` usable arcs in front.
        _nk = int(min_agreeing) if min_agreeing is not None else 1
        _ssel, _tsel = seed(Us, Ut_c, src, tgt, nsrc, _nk,
                            budget, threads=threads)

        # (b) the seeds get the FULL search, and ALL of them are used. Each
        # returns a complete answer for the pixel -- h_p + dh -- so they are
        # `consensus` measurements of one quantity, and the provisional model
        # is their robust centre. Taking the best single one would rest the
        # pixel on one arc, which is the star attachment this replaces.
        gs_, hs_, vs_, _ = _3d_arc_batch(
            Us, Ut, _ssel, _tsel, ele2phase, t, meter2rad, max_dh,
            max_dv, step_dh, step_dv, budget, iterations, threads=threads)
        _ph = np.full((nsrc, _nk), np.nan)
        _pv = np.full((nsrc, _nk), np.nan)
        _o = np.lexsort((-np.where(np.isfinite(gs_), gs_, -np.inf), _ssel))
        _cn = np.bincount(_ssel[_o], minlength=nsrc)
        _cl = np.arange(len(_o)) - np.repeat(np.r_[0, np.cumsum(_cn)[:-1]], _cn)
        _ph[_ssel[_o], _cl] = (fix_h[_tsel] + hs_)[_o]
        _pv[_ssel[_o], _cl] = (fix_v[_tsel] + vs_)[_o]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN rows
            prov_h = np.nanmedian(_ph, axis=1)
            prov_v = np.nanmedian(_pv, axis=1)
        _ok = np.isfinite(prov_h) & np.isfinite(prov_v)

        # (c) with the pixel's own model divided out too, EVERY candidate is
        # scored at n multiply-adds -- including ones the prior ranked poorly,
        # which is the point: the prior chose what to search, the model
        # chooses what to keep.
        Us_c = np.zeros_like(np.asarray(Us, np.complex64))
        Us_c[:, _ok] = _3d_model_removed(
            np.asarray(Us, np.complex64)[:, _ok], ele2phase, t,
            prov_h[_ok], prov_v[_ok])
        ga = _3d_predict_gamma(Us_c, Ut_c, src, tgt, budget, threads=threads)
        ga = np.where(_ok[src], ga, np.nan).astype(np.float32)
        del Us_c
        if min_agreeing is not None and len(np.unique(base_lab)) > 1:
            # the best partner names the component, as on the lattice path
            _l0 = base_lab[tgt].astype(np.int64)
            _t1 = np.full(nsrc, -np.inf)
            np.maximum.at(_t1, src, np.where(np.isfinite(ga), ga, -np.inf))
            _w = np.full(nsrc, -1, dtype=np.int64)
            _is1 = (ga >= _t1[src]) & np.isfinite(ga)
            _w[src[_is1]] = _l0[_is1]
            ga = np.where(_l0 != _w[src], np.nan, ga).astype(np.float32)

        # (d) what is kept becomes MEASUREMENTS -- the consensus test and the
        # network solve consume dh, dv, so they are refined for real. Seeded
        # from the prediction, so the refinement runs without a lattice.
        _short = _nk
        _keep = _3d_topk_per_src(ga, src, nsrc, _short)
        dha = np.full(len(src), np.nan)
        dva = np.full(len(src), np.nan)
        dsa = np.zeros(len(src), dtype=np.complex128)
        if len(_keep):
            _seed = np.stack([prov_h[src[_keep]] - fix_h[tgt[_keep]],
                              prov_v[src[_keep]] - fix_v[tgt[_keep]]], axis=1)
            g2, h2, v2, s2 = _3d_arc_batch(
                Us, Ut, src[_keep], tgt[_keep], ele2phase, t, meter2rad,
                max_dh, max_dv, step_dh, step_dv, budget, iterations,
                seed_th=_seed, threads=threads)
            ga[_keep], dha[_keep], dva[_keep], dsa[_keep] = g2, h2, v2, s2
        _unref = np.ones(len(src), dtype=bool)
        _unref[_keep] = False
        ga[_unref] = np.nan
        if stats is not None:
            stats[prefix + 'ranked_arcs'] = int(len(src))
            stats[prefix + 'searched_arcs'] = int(len(_ssel))
            stats[prefix + 'provisional'] = int(_ok.sum())
        good = np.isfinite(ga) & (ga >= float(threshold)) & np.isfinite(dha)
        return ga, dha, dva, dsa, good

    ga, dha, dva, dsa = _3d_arc_batch(
        Us, Ut, src, tgt, ele2phase, t, meter2rad, max_dh, max_dv,
        step_dh, step_dv, budget, 1, threads=threads)
    _lab = base_lab[tgt].astype(np.int64)
    if min_agreeing is not None and len(np.unique(base_lab)) > 1:
        # THE BEST PARTNER NAMES THE COMPONENT, AND THEN ONLY ITS NODES ARE
        # USED. Components carry unrelated datums, so partners drawn from two
        # of them disagree by that offset however good every arc is. The
        # single most coherent arc says which network this pixel belongs to;
        # if that component then cannot field `min_agreeing` partners, or they
        # disagree, the pixel is unmeasured -- no other component is tried.
        # Arc coherence is settled before any velocity is read, so the best
        # arc cannot be picked to produce a result.
        _gf = np.where(np.isfinite(ga), ga, -np.inf)
        _top1 = np.full(nsrc, -np.inf)
        np.maximum.at(_top1, src, _gf)
        _is1 = (_gf >= _top1[src]) & (_gf > -np.inf)
        _win = np.full(nsrc, -1, dtype=np.int64)
        _win[src[_is1]] = _lab[_is1]
        if debug and stats is not None:
            # what the restriction forbids, counted before it acts: shortlists
            # that would have spanned two components. Not "saw more than one",
            # which nearly every pixel does once a second one exists in reach.
            _so = np.lexsort((-_gf, src))
            _sn = np.bincount(src[_so], minlength=nsrc)
            _sf = np.r_[0, np.cumsum(_sn)[:-1]]
            _tp = _so[(np.arange(len(_so)) - np.repeat(_sf, _sn)) < min_agreeing]
            _lo = np.full(nsrc, np.iinfo(np.int64).max, np.int64)
            _hi = np.full(nsrc, -1, dtype=np.int64)
            np.minimum.at(_lo, src[_tp], _lab[_tp])
            np.maximum.at(_hi, src[_tp], _lab[_tp])
            stats[prefix + 'shortlist_straddled'] = int(
                np.count_nonzero((_hi >= 0) & (_hi != _lo)))
            stats[prefix + 'multi_component'] = int(np.count_nonzero(
                np.bincount(src[_lab != _win[src]], minlength=nsrc) > 0))
        ga[_lab != _win[src]] = np.nan
    _short = int(min_agreeing) if min_agreeing is not None else len(src)
    _ord = np.lexsort((-np.where(np.isfinite(ga), ga, -np.inf), src))
    _cnt0 = np.bincount(src[_ord], minlength=nsrc)
    _off0 = np.r_[0, np.cumsum(_cnt0)[:-1]]
    _col0 = np.arange(len(_ord)) - np.repeat(_off0, _cnt0)
    # ADMISSIBLE ONLY. Ranking puts NaN last, but taking the first `_short`
    # POSITIONS still reaches them when a pixel has fewer admissible
    # candidates -- and refitting one hands it a fresh finite coherence, so an
    # arc ruled out before the ranking would come back holding a vote.
    _keep = _ord[(_col0 < _short) & np.isfinite(ga[_ord])]
    if len(_keep):
        g2, h2, v2, s2 = _3d_arc_batch(
            Us, Ut, src[_keep], tgt[_keep], ele2phase, t, meter2rad,
            max_dh, max_dv, step_dh, step_dv, budget, iterations,
            threads=threads)
        ga[_keep], dha[_keep], dva[_keep], dsa[_keep] = g2, h2, v2, s2
    # anything not refined cannot be used, whatever its lattice value
    _unref = np.ones(len(src), dtype=bool)
    _unref[_keep] = False
    ga[_unref] = np.nan
    good = np.isfinite(ga) & (ga >= float(threshold)) & np.isfinite(dha)
    return ga, dha, dva, dsa, good


def _3d_partner_consensus(src, ga, good, v_abs, nsrc, min_agreeing,
                          err_v, passes, labels=None, stats=None,
                          prefix='ds_', h_abs=None, err_h=None):
    """Do a pixel's best `min_agreeing` partners agree? Returns (first, votes, ok).

    ONLY THE BEST `min_agreeing` PARTNERS ENTER, AND NOTHING ELSE DOES. A pixel
    sees tens of candidates spanning every quality from just above `threshold`
    upwards, and a median across that mixture estimates nothing: it summarises
    several populations, so it describes neither the good arcs nor the bad.
    Robustness is not what makes it meaningless -- the mixture is.

    So the partners are CHOSEN first, by arc coherence, which is settled before
    any value is read and so cannot be picked to suit the answer. Everything
    downstream -- the centre, the scale, the rejection -- then sees one
    homogeneous set of comparable measurements, which is the only situation
    where a robust scale means anything.
    """
    _ma, _ii = min_agreeing, passes
    o2 = np.lexsort((-np.where(good, ga, -np.inf), src))
    o2 = o2[good[o2]]
    cnt = np.bincount(src[o2], minlength=nsrc)
    if not len(cnt) or not cnt.max():
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32),
                np.zeros(nsrc, dtype=bool))
    off = np.r_[0, np.cumsum(cnt)[:-1]]
    col = np.arange(len(o2)) - np.repeat(off, cnt)
    # columns are gamma-descending, so the first `_ma` ARE the best `_ma`; a
    # row with fewer admissible arcs cannot fill them and is rejected for
    # having too few
    if _ma is not None:
        take = col < _ma
        o2, col = o2[take], col[take]
        cnt = np.minimum(cnt, _ma)
    kmax = int(cnt.max())
    row = src[o2]
    V = np.full((nsrc, kmax), np.nan)
    G = np.zeros((nsrc, kmax))
    IDX = np.full((nsrc, kmax), -1, dtype=np.int64)
    V[row, col] = v_abs[o2]
    G[row, col] = ga[o2]
    IDX[row, col] = o2
    fin = np.isfinite(V)
    # rows with no admissible arc are all NaN and are rejected below; taking a
    # median of one would only warn
    anyf = fin.any(axis=1)
    e = np.zeros(nsrc)
    if anyf.any():
        e[anyf] = np.nanmedian(np.where(fin, V, np.nan)[anyf], axis=1)
    d0 = np.abs(V - e[:, None])
    floor = max(_SIGMA_FLOOR * float(np.nanmedian(d0[fin])), 1e-9) \
        if fin.any() else 1e-9
    for _ in range(_ii):
        w = np.where(fin, G / np.maximum(np.abs(V - e[:, None]), floor), 0.0)
        sw = w.sum(axis=1)
        e = np.where(sw > 0, (w * np.where(fin, V, 0.0)).sum(axis=1)
                     / np.maximum(sw, 1e-30), e)
    r = np.abs(V - e[:, None])
    # PER PIXEL, from its own partners: they are what say how much this
    # pixel's measurements scatter. The floor keeps a row whose partners
    # happen to land identically from rejecting everything else on a zero
    # scale.
    sig = np.full(nsrc, floor)
    if fin.any():
        rr = np.where(fin, r, np.nan)
        enough = fin.sum(axis=1) >= _ma
        if enough.any():
            sig[enough] = np.maximum(
                1.4826 * np.nanmedian(rr[enough], axis=1), floor)
    # THE BOUND IS ABSOLUTE, AND IT TESTS BOTH QUANTITIES. A robust sigma is a
    # property of whatever scatter is present, so a pixel whose partners were
    # uniformly wrong grew a scale to match and certified itself unanimously.
    # A stated bound cannot be widened by what it is judging. Height is tested
    # too: the vote used to read velocity alone, and a partner could agree on
    # the rate while placing the pixel metres away.
    _HVg = None
    if h_abs is not None:
        _HVg = np.full((nsrc, kmax), np.nan)
        _HVg[row, col] = np.asarray(h_abs)[o2]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            _eh = np.nanmedian(np.where(fin, _HVg, np.nan), axis=1)
        _eh = np.where(np.isfinite(_eh), _eh, 0.0)
    keep = fin & (r <= float(err_v))
    if _HVg is not None and err_h is not None:
        keep = keep & (np.abs(_HVg - _eh[:, None]) <= float(err_h))
    nkeep = keep.sum(axis=1)
    if _ma is not None:
        # ALL `min_agreeing` OF THEM, UNANIMOUSLY. Naming WHICH partners have
        # to agree is what makes it a test rather than "some few of many",
        # which any unimodal scatter passes on its shape alone. Allowing one
        # dissenter is not the mild relaxation it reads as: there are only
        # `_ma` columns, so a pixel holding `_ma - 1` admissible partners
        # fills every column it has and passes, spending the tolerance meant
        # for one partner DISAGREEING on one being ABSENT.
        ok = keep[:, :_ma].sum(axis=1) == _ma
    else:
        ok = nkeep >= 1
    # rows are gamma-descending, so the first survivor is the best arc of the
    # consistent set
    sel_col = np.argmax(keep, axis=1)
    first = IDX[np.flatnonzero(ok), sel_col[ok]]
    votes = nkeep[ok].astype(np.int32)
    # THE CENTRE IS THE ANSWER, NOT ONLY THE TEST. Every partner is a complete
    # measurement of this pixel, so a pixel holding `nkeep` of them holds that
    # many measurements of one quantity. Naming the best and discarding the
    # rest throws away the averaging they were gathered for; the robust centre
    # of the voting set is what they collectively say.
    def _centre(X):
        _xk = np.where(keep, X, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            c = np.nanmedian(_xk, axis=1)
            _fl = max(_SIGMA_FLOOR * float(np.nanmedian(
                np.abs(_xk - c[:, None]))), 1e-9)
        c = np.where(np.isfinite(c), c, 0.0)
        for _ in range(max(int(passes) if passes else 1, 1)):
            w = np.where(keep, G / np.maximum(np.abs(X - c[:, None]), _fl), 0.0)
            sw = w.sum(axis=1)
            c = np.where(sw > 0,
                         (w * np.where(keep, X, 0.0)).sum(axis=1)
                         / np.maximum(sw, 1e-30), c)
        return np.where(nkeep > 0, c, np.nan)
    _cen_v = _centre(V)
    _cen_h = None if _HVg is None else _centre(_HVg)
    if stats is not None:
        stats[prefix + 'centre_v'] = _cen_v
        stats[prefix + 'centre_h'] = _cen_h
        # THE ARCS THAT VOTED. A pixel's consistency has to be read over the
        # partners its consensus actually rested on, the way the network's
        # closure is read over the arcs its solve rested on. Every admissible
        # partner includes the ones this test threw out, which is a different
        # question and a harsher one.
        _vm = keep & ok[:, None]
        _vi = IDX[_vm]
        stats[prefix + 'vote_arcs'] = _vi[_vi >= 0]
        reach = int(np.count_nonzero(cnt))
        stats[prefix + 'admissible'] = reach
        stats[prefix + 'no_consensus'] = reach - len(first)
        stats[prefix + 'too_few'] = int(
            reach - np.count_nonzero(fin[:, :_ma].sum(axis=1) >= _ma)
            if _ma is not None else 0)
        if labels is not None:
            # THE INVARIANT, CHECKED RATHER THAN ARGUED. The shortlist behind
            # an attached pixel must lie in ONE component: components carry
            # unrelated datums, so a decision taken across two is a numerical
            # error, not a noisier answer. Must read 0 on any scene with any
            # parameters. On `fin`, NOT on `keep` -- a cross-component partner
            # differs by the datum offset, which is what makes it an outlier,
            # so the rejection discards it before `keep` exists.
            _rw = np.flatnonzero(ok)
            _ix2 = IDX[_rw]
            _lb2 = np.where(fin[_rw] & (_ix2 >= 0),
                            labels[np.clip(_ix2, 0, None)].astype(np.int64), -1)
            _hi2 = _lb2.max(axis=1)
            _lo2 = np.where(_lb2 >= 0, _lb2, np.iinfo(np.int64).max).min(axis=1)
            stats[prefix + 'cross_component_votes'] = int(
                np.count_nonzero((_hi2 >= 0) & (_hi2 != _lo2)))
    return first, votes, ok


def _3d_fit_ps_array(scenes, date_values, *, spacing, bperp=None,
                       window=(32, 128), threshold=0.5, cell=(2, 8),
                       geometry, budget=None, level=1,
                       max_dh=100.0, max_dv=25.0, step_dh=4.0, step_dv=2.0,
                       max_seasonal=5.0,
                       consensus, iterations=8, threads=None, debug=False):
    """Ground phase and velocity at PERSISTENT scatterers, per connected component.

    The measurement is per ARC, never per pixel: a single scatterer carries an
    unknown constant, an unknown height error and an unknown rate, and nothing
    in its own time series separates them. A double difference removes the
    constant and lets (height, rate, annual) be fitted, so the network is what
    turns arc differences into per-node values.

    ONLY PS MAY SOURCE IT. Distributed scatterers are coherent with a near
    neighbour and lose it as soon as the common atmosphere has cancelled, so
    they cannot hold a long arc and cannot tie a component together.

    WHAT IS RETURNED IS DISPLACEMENT, not a residual. Only the height term is
    removed from each node's phase; rate, seasonal and any other real motion
    stay in. Every pixel that is not a node is NaN, and nothing is
    interpolated into it.

    THERE IS NO ATMOSPHERIC SCREEN HERE ANY MORE, and that is a tested
    decision rather than a simplification. Building one per node and kriging it
    to the ground cost coherence at every separation, because a node's residual
    is dominated by its own noise rather than by correlated signal, so what the
    interpolation spreads is mostly that error. Many variants were tried --
    kriging predictors with the nugget in or out of the system, first-order
    drift, per-epoch variograms, fitted and fixed temporal filters,
    scale-mixture kernels, per-epoch ramps, and a per-epoch stratified term
    proportional to elevation. Only the last is worth anything, and only
    marginally, concentrated at long range. None of it belongs in the product
    that feeds displacement.

    The datum is PER COMPONENT and free by one constant each. Two components
    are two networks with nothing measured between them, so their heights,
    rates and seasonals are not comparable across the boundary; `labels` names
    which is which so a caller can see it rather than assume it.

    Returns `(labels, velocity, height, seasonal, coherence)` -- NO phase: fit3d() returns the model only and predict() rebuilds phase from it -- rasters,
    all NaN where nothing was solved. RADIANS THROUGHOUT: velocity rad/yr,
    height rad per unit ele2phase, seasonal complex rad. `displacement_los()` is the
    only place a length is produced.

    `.stats` additionally carries the network solution per node -- `iy`, `ix`,
    `label`, `degree`, `height_rad`, `velocity_rad_yr`, `seasonal_rad` -- but it
    is a FUNCTION ATTRIBUTE written by whichever block ran last, so under dask
    it describes one chunk. Use the rasters; stats is for single-block work.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import lsqr
    from scipy.spatial import cKDTree

    # ONE FIT AT A TIME, cluster-wide. The solve saturates the host by itself,
    # so a second concurrent block buys contention rather than throughput; a
    # cluster semaphore serialises the blocks while the graph stays lazy and
    # any single block stays independently computable.
    from distributed import Semaphore
    _gate = Semaphore(max_leases=1, name='insardev-fit3d')
    _gate.acquire()
    try:
        return __3d_fit_ps_array_gated(
            scenes, date_values, spacing=spacing, bperp=bperp, window=window,
            threshold=threshold, cell=cell, geometry=geometry, budget=budget,
            level=level, max_dh=max_dh, max_dv=max_dv, step_dh=step_dh,
            step_dv=step_dv, max_seasonal=max_seasonal, consensus=consensus,
            iterations=iterations, threads=threads, debug=debug)
    finally:
        _gate.release()



def _3d_fit_frame(date_values, bperp, geometry, n):
    """The time base and the height-to-phase factors the three stages share.

    Pulled out because the stages run in separate tasks under `union=True` and
    each has to rebuild the same frame; deriving it twice from the same inputs
    is exact, while passing it between tasks would let the two drift.
    """
    t = np.asarray(date_values)
    if t.dtype.kind == 'M':
        t = t.astype('datetime64[D]').astype(np.float64)
    B = np.zeros(n) if bperp is None else np.asarray(bperp, dtype=np.float64)
    # zero at the master, where the phase is zero by construction and the
    # height term vanishes with the baseline that carries it
    t = (t - t[int(np.argmin(np.abs(B)))]) / 365.25
    wavelength, r_sin = geometry
    meter2rad = 4.0 * np.pi / wavelength
    ele2phase = B / r_sin
    car = np.exp(2j * np.pi * t)
    return t, ele2phase, meter2rad, car


def _3d_ps_nodes(scenes, date_values, *, spacing, bperp=None,
                 window=(32, 128), threshold=0.5, cell=(2, 8), geometry,
                 budget=None, iterations=8, threads=None, debug=False):
    """PASS 1, per block: the rank raster and this block's PS nodes.

    The block is read once, scanned, and left behind: what leaves is the rank
    raster and a table of nodes -- their positions and their unit phasors --
    which is sparse where the block is dense. That is what lets one network be
    solved over bursts that were never merged into a single raster.
    """
    import os as _os
    _nth = max(1, int(threads) if threads else (_os.process_cpu_count() or 1))
    S = np.ascontiguousarray(scenes, dtype=np.complex64)
    n, ny, nx = S.shape
    wy, wx, pey, pex = _3d_windows(window)
    if n < 2 or ny == 0 or nx == 0:
        return None
    sy, sx = float(spacing[0]), float(spacing[1])
    if not (sy > 0 and sx > 0):
        raise ValueError(f'spacing must be positive, got {spacing}')
    t, ele2phase, meter2rad, car = _3d_fit_frame(date_values, bperp,
                                                 geometry, n)
    # THE BOUNDS ARE STATED IN METRES AND MM/YR AND USED IN RADIANS. A metre of
    # height and a mm/yr of rate carry different amounts of phase, so the two
    # bounds are not interchangeable numbers -- they are the same statement
    # ---- the nodes: PS, not DS -----------------------------------------
    q = _3d_arcs_kernel(S, wy, wx, tuple(cell), budget, threads=_nth)
    ps = _3d_ps_kernel(S, (wy, wx, pey, pex), q, ele2phase, t, meter2rad,
                       threshold=float(threshold), budget=budget,
                       iterations=iterations)
    iy, ix = np.where(np.isfinite(ps) & (ps >= float(threshold)))
    if debug:
        _cand = int(np.count_nonzero(np.isfinite(q) & (q >= float(threshold))))
        print(f'DEBUG: PS test  {len(iy)} nodes at >= {float(threshold)}'
              f'  ({_cand} DS candidates in the same raster)', flush=True)
    if len(iy) < 2:
        if debug:
            print('DEBUG: fewer than 2 nodes -- nothing to solve', flush=True)
        # EMPTY, not "one node": the positions and the phasor columns are
        # one table and must stay the same length. A lone position with no
        # column silently misaligns every node concatenated after it when the
        # union stage gathers the blocks.
        return dict(q=q, iy=iy[:0], ix=ix[:0],
                    U=np.zeros((n, 0), np.complex64))
    a = np.abs(S[:, iy, ix])
    with np.errstate(invalid='ignore', divide='ignore'):
        Un = np.ascontiguousarray(
            np.where(a > 0, S[:, iy, ix] / np.where(a > 0, a, 1), 0
                     ).astype(np.complex64))
    del a

    return dict(q=q, iy=iy, ix=ix, U=Un)


# HOW MANY ARCS A NODE MAY BRING TO THE NETWORK. Not a quality threshold --
# `threshold` is that -- but a bound on redundancy: beyond this many arcs a
# node is drawing repeatedly from the same neighbourhood, so the rows stop
# carrying independent information while the solve keeps paying for them.
_ARC_CAP = 100


def _3d_ps_network(U, iy, ix, date_values, *, bperp=None, window=(32, 128),
                   threshold=0.5, geometry, budget=None, consensus,
                   iterations=8, max_dh=100.0, max_dv=25.0, step_dh=4.0,
                   step_dv=2.0, max_seasonal=5.0, err_dh=5.0, err_dv=1.0,
                   threads=None, debug=False):
    """The network over the nodes ALONE -- no raster, so no scene in memory.

    Every argument is a node quantity, which is why this stage can be run once
    over nodes gathered from several blocks or several bursts: the arcs are
    fitted between phasor columns and the datum is per component, neither of
    which asks where the pixels were stored.

    Returns the solved node table, or None when no network could be formed.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import lsqr
    from scipy.spatial import cKDTree
    import os as _os
    _nth = max(1, int(threads) if threads else (_os.process_cpu_count() or 1))
    _ma = _3d_consensus(consensus)
    _ii = max(1, int(iterations))
    Un = np.ascontiguousarray(U, dtype=np.complex64)
    iy = np.asarray(iy)
    ix = np.asarray(ix)
    n = Un.shape[0]
    wy, wx, pey, pex = _3d_windows(window)
    _3d_fit_ps_array.stats.reset(nodes=0, arcs=0, dropped=0, components=[],
                                 fill_order=[])
    _mark = time.monotonic()
    if n < 2 or len(iy) < 2:
        return None
    t, ele2phase, meter2rad, car = _3d_fit_frame(date_values, bperp,
                                                 geometry, n)
    # THE BOUNDS ARE STATED IN METRES AND MM/YR AND USED IN RADIANS. A metre of
    # height and a mm/yr of rate carry different amounts of phase, so the two
    # are not interchangeable numbers -- they are one statement about how far a
    # measurement may sit from the solve, each in its own unit.
    _err_h = float(err_dh) * meter2rad
    _err_v = float(err_dv) * meter2rad / 1e3
    # ---- the arcs: every pair inside the PS window ----------------------
    # scaled so the window becomes the unit box, then a Chebyshev query is
    # exactly "inside the window" and costs O(N k) rather than O(N^2)
    hy, hx = max(pey // 2, 1), max(pex // 2, 1)
    tree = cKDTree(np.c_[iy / hy, ix / hx])
    pairs = tree.query_pairs(1.0, p=np.inf, output_type='ndarray')
    if len(pairs) < 3:
        return None
    ai, aj = pairs[:, 0], pairs[:, 1]
    g = np.empty(len(ai), np.float32)
    dh = np.empty(len(ai)); dv = np.empty(len(ai))
    ds_ = np.empty(len(ai), np.complex128)

    def _fit_pairs(sel, budget_):
        step = max(1, int(_3d_budget_mb(budget_) * 1024 * 1024
                          // max(n * 16, 1)))
        for b0 in range(0, len(sel), step):
            s_ = sel[b0:min(b0 + step, len(sel))]
            arc = np.ascontiguousarray(
                (Un[:, ai[s_]] * np.conj(Un[:, aj[s_]])).astype(np.complex64))
            g[s_], dh[s_], dv[s_], ds_[s_] = _3d_arc_fit(
                arc, ele2phase, t, meter2rad, max_dh, max_dv, step_dh,
                step_dv, budget_, max_seasonal,
                iterations=iterations)
    if _nth > 1 and len(ai) > _nth:
        # arcs are independent; slices of them fit concurrently
        from concurrent.futures import ThreadPoolExecutor
        _bnd = np.linspace(0, len(ai), _nth * 4 + 1).astype(np.int64)
        _idx = np.arange(len(ai))
        with ThreadPoolExecutor(_nth) as _ex:
            list(_ex.map(lambda i: _fit_pairs(
                _idx[_bnd[i]:_bnd[i + 1]],
                _3d_budget_mb(budget) / _nth), range(_nth * 4)))
    else:
        _fit_pairs(np.arange(len(ai)), budget)
    keep = np.isfinite(g) & (g >= float(threshold))
    _lap, _mark = _3d_lap(_mark)
    if debug:
        _gk = g[keep]
        print(f'DEBUG: arcs     {len(ai):,} pairs fitted, {int(keep.sum()):,} '
              f'>= {float(threshold)}  ({100 * keep.mean():.1f}%)'
              f'   {_lap:.1f}s', flush=True)
        if keep.any():
            print(f'DEBUG:          arc gamma p50 {np.median(_gk):.3f}  '
                  f'p90 {np.percentile(_gk, 90):.3f}  max {_gk.max():.3f}',
                  flush=True)
    if keep.sum() < 3:
        if debug:
            print('DEBUG: fewer than 3 arcs cleared the threshold', flush=True)
        return None
    ai, aj, dh, dv, ds_ = ai[keep], aj[keep], dh[keep], dv[keep], ds_[keep]
    gk = g[keep]

    # ---- A NODE'S BEST `_ARC_CAP` ARCS, AND NO MORE ---------------------
    # A node's degree is its PS density times the window's area, so it grows
    # without bound as a scene gets denser while the window stays fixed -- and
    # the least-squares system grows with it. Past some number of arcs a node
    # is not being measured any better: the extra ones are drawn from the same
    # neighbourhood as the ones already kept, so they add rows without adding
    # independent constraint, and the robust pass has to judge every one of
    # them. On a sparse scene the cap is inert; on a dense one it is what keeps
    # the solve proportional to the ground rather than to the density.
    #
    # AN ARC SURVIVES IF EITHER END STILL WANTS IT, which is what makes this
    # safe. Capping each node's own list independently would drop the long
    # arcs that tie distant groups together -- they rank low for both ends
    # because coherence falls with distance -- and the network would fall into
    # pieces, each with its own free datum, which is far worse than any noise
    # the cap removes. Keeping an arc that either end still ranks means a node
    # cannot be isolated by another node's budget, and a long arc survives as
    # long as one of its ends has room for it.
    #
    # Redundancy is still what the screen is made of; only the excess beyond
    # what a node can use is rationed, and the robust pass below decides which
    # of the rest the network believes.
    if _ARC_CAP and len(ai) > 1:
        _cap_keep = np.zeros(len(ai), dtype=bool)
        for _ends in (ai, aj):
            _o = np.lexsort((-gk, _ends))
            _e = _ends[_o]
            _starts = np.r_[0, np.flatnonzero(np.diff(_e)) + 1]
            _group = np.cumsum(np.r_[True, np.diff(_e) != 0]) - 1
            _cap_keep[_o[np.arange(len(_e)) - _starts[_group] < _ARC_CAP]] = True
        if not _cap_keep.all():
            _lap, _mark = _3d_lap(_mark)
            if debug:
                print(f'DEBUG: cap      {int((~_cap_keep).sum()):,} of '
                      f'{len(ai):,} arcs beyond {_ARC_CAP} per node dropped'
                      f'   {_lap:.1f}s', flush=True)
            ai, aj = ai[_cap_keep], aj[_cap_keep]
            dh, dv, ds_ = dh[_cap_keep], dv[_cap_keep], ds_[_cap_keep]
            gk = gk[_cap_keep]
    N = len(iy)
    gtake = gk

    def _incidence(a_, b_, m):
        return coo_matrix((np.tile([1.0, -1.0], m),
                           (np.repeat(np.arange(m), 2), np.c_[a_, b_].ravel())),
                          shape=(m, N)).tocsr()

    def _wsolve(Gm, rhs, w):
        sw = np.sqrt(np.maximum(w, 0.0))
        return lsqr(diags(sw) @ Gm, sw * rhs,
                    atol=1e-12, btol=1e-12, iter_lim=2000)[0]

    def _wsolve2(Gm, rhs_a, rhs_b, w):
        """The height and the rate solved at once: one matrix, two threads.

        Every pass of the reweighting and every pass of the gate solves the
        network TWICE, once for each quantity, and the two share the matrix and
        depend on nothing of each other's. `lsqr` spends its time in kernels
        that release the GIL, so the pair costs little more than one of them --
        and these solves are the great majority of the stage.
        """
        import threading
        out = [None, None]

        def _run(i, r):
            out[i] = _wsolve(Gm, r, w)

        th = (threading.Thread(target=_run, args=(0, rhs_a)),
              threading.Thread(target=_run, args=(1, rhs_b)))
        for t in th:
            t.start()
        for t in th:
            t.join()
        return out[0], out[1]

    def _mad(r):
        return 1.4826 * float(np.median(np.abs(r - np.median(r))))

    def _node_sigma(res, a_, b_, nnodes, floor, min_n, cap=None):
        """Robust scale of the arc residuals AT EACH NODE.

        A node's own arcs say how much its measurements scatter, and that is
        the scale its outliers have to stand out against. One scale for the
        whole scene would judge a quiet node by the noise of a loud one, and it
        is what asking for `min_n` measurements per node is FOR. That count is
        the caller's `consensus`, not a constant: it has to be big enough to
        carry a scale, which is a stricter requirement than redundancy.
        """
        who = np.r_[a_, b_]
        val = np.r_[np.abs(res), np.abs(res)]
        o = np.argsort(who, kind='stable')
        who, val = who[o], val[o]
        cut = np.r_[0, np.flatnonzero(np.diff(who)) + 1, len(who)]
        sig = np.full(nnodes, floor)
        for i0, i1 in zip(cut[:-1], cut[1:]):
            if i1 - i0 >= min_n:
                v = val[i0:i1]
                sig[who[i0]] = max(1.4826 * float(np.median(
                    np.abs(v - np.median(v)))), floor)
        # A CEILING AS WELL AS A FLOOR. The floor stops a node whose arcs
        # happen to land together from rejecting everything else on a scale of
        # nearly zero. Without a ceiling the converse is unguarded: a node
        # whose arcs mostly DISAGREE gets a wide scale, and the gate that is
        # supposed to judge it is switched off by the very contamination it
        # exists to catch. A node may be stricter than the network as a whole,
        # never far looser.
        if cap is not None:
            sig = np.minimum(sig, cap)
        return sig

    # ---- REJECT ARCS THE NETWORK CONTRADICTS ---------------------------
    # An arc's coherence says how well it fits its OWN phase; it does not say
    # whether it agrees with the rest of the network, and the two are close to
    # independent. Plain least squares cannot tell the difference: it has no
    # way to reject one equation, so a contradicted arc is absorbed by
    # spreading its error over every arc that shares a node with it. On a
    # thinly connected node that is the whole solution.
    #
    # IRLS finds them, then they are DROPPED rather than down-weighted, and
    # the survivors are solved exactly. A down-weighted arc still perturbs the
    # answer and still holds its nodes in the component; a dropped one leaves
    # a node unsupported, which is the honest outcome -- that node had no
    # consistent measurement.
    G = _incidence(ai, aj, len(ai))
    w_ = gtake.astype(np.float64)
    r_h = r_v = None
    for _ in range(_ii):
        _xh, _xv = _wsolve2(G, dh, dv, w_)
        r_h = G @ _xh - dh
        r_v = G @ _xv - dv
        # per NODE, not pooled: judged against its own arcs' scatter
        f_h, f_v = _SIGMA_FLOOR * _mad(r_h), _SIGMA_FLOOR * _mad(r_v)
        s_h = _node_sigma(r_h, ai, aj, N, max(f_h, 1e-12), _ma,
                          cap=max(_mad(r_h), 1e-12))
        s_v = _node_sigma(r_v, ai, aj, N, max(f_v, 1e-12), _ma,
                          cap=max(_mad(r_v), 1e-12))
        # an arc must hold up as seen from BOTH of its ends
        z = np.maximum(np.abs(r_h) / np.minimum(s_h[ai], s_h[aj]),
                       np.abs(r_v) / np.minimum(s_v[ai], s_v[aj]))
        w_ = gtake / np.maximum(z, 1.0)
    # scale from the ROBUST fit, so the outliers do not set their own bar
    # ---- REJECT, RE-SOLVE, RE-CHECK, UNTIL IT SETTLES -------------------
    # `reject_sigma` promises that no surviving arc sits further than that
    # many node scales from the solution. Rejecting once cannot deliver it:
    # removing arcs MOVES the solution, and an arc validated against the old
    # one may sit well outside the gate of the new one -- measured, a fifth of
    # the arcs were rejected, and 2.4% of the survivors then had a residual
    # more than three times what the gate had seen.
    #
    # So the gate is re-applied to the solution it produced, until the
    # surviving set stops changing. Then the promise is true of the answer
    # that ships rather than of an intermediate nobody receives. It converges
    # in a few passes because each one removes less than the last; the cap is
    # only to bound the work if it ever oscillates between two sets.
    # THE SCALE IS ESTIMATED ONCE AND HELD. It describes the measurement
    # noise, which does not change because arcs were deleted -- only the
    # SOLUTION does. Re-estimating it every pass makes the gate feed on
    # itself: a cleaner set gives a tighter scale, which rejects more, which
    # tightens it again, and the loop erodes the network instead of settling.
    # Measured that way it removed a further 2 000 arcs and 35 nodes and still
    # left arcs at z = 101, because the bar moved under them.
    _fix_sh = _node_sigma(r_h, ai, aj, N, max(_SIGMA_FLOOR * _mad(r_h), 1e-12),
                          _ma, cap=max(_mad(r_h), 1e-12))
    _fix_sv = _node_sigma(r_v, ai, aj, N, max(_SIGMA_FLOOR * _mad(r_v), 1e-12),
                          _ma, cap=max(_mad(r_v), 1e-12))

    def _gate(idx):
        """Residual of the solve on `idx`, scored against the HELD scale."""
        a_, b_ = ai[idx], aj[idx]
        G_ = _incidence(a_, b_, len(idx))
        _xh_, _xv_ = _wsolve2(G_, dh[idx], dv[idx], w_[idx])
        rh_ = G_ @ _xh_ - dh[idx]
        rv_ = G_ @ _xv_ - dv[idx]
        # ABSOLUTE, NOT RELATIVE. Scored against a robust sigma an arc could
        # widen the bar that judged it: a node whose arcs were uniformly wrong
        # got a scale to match and kept them all. The bound the caller stated
        # cannot be widened by what it is judging.
        z_ = np.maximum(np.abs(rh_) / _err_h, np.abs(rv_) / _err_v)
        return z_, _fix_sv, rv_

    if True:
        keep_arc = np.ones(len(ai), dtype=bool)
        _passes = 0
        # HOW MANY TIMES TO REJECT AND RE-SOLVE. The same count that governs
        # the IRLS reweighting: both are iterative refinements of one solve
        # and there is no reason for a caller to reason about them separately.
        # The loop stops as soon as the surviving set repeats, so this only
        # bounds an oscillation -- how many are actually used depends on
        # `reject_sigma`, since a tighter gate removes more per pass and takes
        # longer to settle.
        for _passes in range(1, _ii + 1):
            _idx = np.flatnonzero(keep_arc)
            _z, _sv, _rv2 = _gate(_idx)
            _ok = _z <= 1.0
            if _ok.all():
                break                      # the gate holds on its own solution
            _new = np.zeros(len(ai), dtype=bool)
            _new[_idx[_ok]] = True
            if _new.sum() < 3:
                break
            keep_arc = _new
        if debug:
            # evaluated ONCE on the set that actually survived, so the numbers
            # describe the arcs the answer is built from
            _idx = np.flatnonzero(keep_arc)
            _dbg_z, _dbg_sv, _dbg_rv = _gate(_idx)
            _gate_passes = int(_passes)
    if keep_arc.sum() < 3:
        return None
    rejected = int((~keep_arc).sum())
    _lap, _mark = _3d_lap(_mark)
    if debug:
        print(f'DEBUG: IRLS     {rejected:,} of {len(ai):,} arcs rejected '
              f'beyond {err_dh:g} m / {err_dv:g} mm/yr  '
              f'({100 * rejected / max(len(ai), 1):.1f}%)'
              f'   {_lap:.1f}s', flush=True)
    # how much of each node's own support the rejection took away
    drej = np.bincount(np.r_[ai[~keep_arc], aj[~keep_arc]], minlength=N)

    # ---- THE FLOOR IS A FIXED POINT, NOT ONE PASS ----------------------
    # Dropping a node whose support is too thin takes its arcs with it, and
    # that lowers its PARTNERS' counts -- which can put them under the floor
    # in turn. Testing once leaves nodes standing on support that has since
    # been removed.
    #
    # The worse half is what it does to the COMPONENTS. A candidate below the
    # floor is never reported, but its arcs stay in the graph, so it still
    # BRIDGES: two groups that share no tested path are welded into one
    # component and handed a common datum on the strength of a node the same
    # rule just refused to report. The output then shows plainly disconnected
    # ground carrying the main component's label, which is not a thin answer
    # but a wrong one -- values presented as comparable that rest on no
    # measured connection.
    #
    # So the test is applied until it stops removing anything. That is the
    # k-core of the surviving-arc graph at k = `consensus`, and it is what
    # the single pass was always reaching for.
    _flr = _ma
    # counted BEFORE the floor runs: nodes the robust pass left with no arc at
    # all. Afterwards every removed node has none, so the two stop being
    # different questions and the distinction has to be taken here.
    _iso0 = int((np.bincount(np.r_[ai[keep_arc], aj[keep_arc]],
                             minlength=N) == 0).sum())
    _kalive = np.ones(N, dtype=bool)
    _kpass = 0
    while True:
        _kd = np.bincount(np.r_[ai[keep_arc], aj[keep_arc]], minlength=N)
        _kdrop = _kalive & (_kd < _flr)
        if not _kdrop.any():
            break
        _kalive &= ~_kdrop
        keep_arc = keep_arc & _kalive[ai] & _kalive[aj]
        _kpass += 1
        if keep_arc.sum() < 3:
            break
    if debug and _kpass > 1:
        print(f'DEBUG: floor    {_kpass} passes to reach the fixed point; '
              f'{int((~_kalive).sum())} of {N} candidates under {_flr} arc(s)',
              flush=True)
    if keep_arc.sum() < 3:
        if debug:
            print('DEBUG: fewer than 3 arcs survive the consensus floor',
                  flush=True)
        return None

    if debug:
        # RE-EVALUATED ON THE ARCS THAT SHIP. The gate above ran before the
        # floor removed anything, so its residuals described a larger set than
        # the answer is built from -- and `arc_z` would no longer line up with
        # `arc_dh`, which is the one thing a caller reading them together
        # needs.
        _idx = np.flatnonzero(keep_arc)
        _dbg_z, _dbg_sv, _dbg_rv = _gate(_idx)

    # THE CLOSURE BELOW IS CIRCULAR IF IT SEES SURVIVORS ONLY. IRLS rejects
    # the arcs that disagree with the solve, so a residual measured over what
    # is left is a property of the selection as much as of the solution. Keep
    # the arcs as they were, and report both.
    _pre = ((ai.copy(), aj.copy(), dh.copy(), dv.copy(), keep_arc.copy())
            if debug else None)
    ai, aj, dh, dv, ds_, gtake = (ai[keep_arc], aj[keep_arc], dh[keep_arc],
                                  dv[keep_arc], ds_[keep_arc], gtake[keep_arc])

    dcount = np.bincount(np.r_[ai, aj], minlength=N)
    # each node's coherence is the mean over the arcs that actually hold it
    _gsum = np.bincount(np.r_[ai, aj], weights=np.r_[gtake, gtake], minlength=N)
    gnode = np.where(dcount > 0, _gsum / np.maximum(dcount, 1), np.nan)
    ncomp, lab = connected_components(
        coo_matrix((np.ones(len(ai)), (ai, aj)), shape=(N, N)), directed=False)
    # CONNECTED IS THE ONLY REQUIREMENT. A node with no surviving arc has no
    # datum -- nothing places it against anything else -- so it cannot be
    # reported. One surviving arc is enough, because of WHAT survival means
    # here: the robust pass has already rejected every arc the network
    # contradicts, so a node holding one arc holds one the network AGREES
    # with. Counting arcs a second time would re-ask a question the rejection
    # already answered.
    #
    # `consensus` SURVIVING ARCS, THE SAME NUMBER THE ATTACHMENT ASKS OF A DS.
    # One question asked twice: an arc must agree with the network, a partner
    # must agree with the other partners. A node reported on fewer than that
    # rests on measurements too few to have been checked against each other.
    #
    # These are SURVIVORS, so the count is what remains after the robust pass
    # rejected every arc the network contradicted -- the agreement is tested
    # there, and this requires enough of it to have been tested at all.
    #
    # It was briefly relaxed to 1, while `degree` capped arcs per node: the cap
    # starved nodes below the count and the ARC BUDGET decided how many PS
    # existed, which is not a property of the data. `degree` is gone and every
    # arc clearing `threshold` now enters, so a node short of survivors is
    # genuinely short of support.
    live = dcount >= _ma
    _lap, _mark = _3d_lap(_mark)
    if debug:
        # NODES LOST BEFORE ANY COMPONENT EXISTS. A node whose every arc was
        # rejected holds no datum and cannot be reported. Counted separately
        # from the component floor below: the two are different losses at
        # different stages, and a single "kept" total hides which is which.
        print(f'DEBUG: solve     {int((~live).sum())} of {len(live)} nodes '
              f'left under {_flr} surviving arc(s)'
              f'  ({_iso0} held none even before the floor)'
              f'   {_lap:.1f}s', flush=True)
    if not live.any():
        return None

    # ---- integrate, one free datum per component -----------------------
    # THE SOLUTION THE GATE VALIDATED IS THE SOLUTION THAT SHIPS. The robust
    # pass rejects arcs whose residual exceeds `reject_sigma` node scales --
    # but of ITS OWN solution. Re-solving the survivors under different
    # weights answers a different question, and the guarantee then attaches to
    # a solution nobody receives: measured, 4% of surviving arcs had a
    # residual more than three times larger in the coherence-weighted resolve
    # than in the one the gate approved.
    #
    # So the robust weights are carried through rather than discarded. They
    # already contain the coherence -- `w_ = gtake / max(z, 1)` -- so an arc
    # is weighted by its own quality AND by how far it sits from the
    # consensus, which is strictly more information than coherence alone.
    m_ = len(ai)
    G = _incidence(ai, aj, m_)
    w_fin = w_[keep_arc]
    hgt, vel = _wsolve2(G, dh, dv, w_fin)
    anr = _wsolve(G, ds_.real, w_fin)
    ani = _wsolve(G, ds_.imag, w_fin)
    for c in np.unique(lab[live]):
        k = live & (lab == c)
        hgt[k] -= np.median(hgt[k])
        vel[k] -= np.median(vel[k])
        # the annual datum is a complex MEAN: a componentwise median does not
        # transform as a complex number, so the gauge it fixes would depend on
        # which epoch t was measured from
        anr[k] -= np.mean(anr[k])
        ani[k] -= np.mean(ani[k])

    sel = np.where(live)[0]
    # ---- model parameters at every node ---------------------------------
    # NO PHASE IS RETURNED. fit3d() emits the model only; phase is
    # reconstructed on demand by predict(model), which lets the caller
    # choose what to remove. Assembling n_dates complex planes here and
    # throwing them away cost one plane per date per chunk for nothing.
    # TOPOGRAPHY IS THE ONLY THING REMOVED. The height term is the one part
    # of the fitted model that is not ground motion, so taking it out leaves
    # a DISPLACEMENT series: rate, seasonal and whatever else the scatterer
    # actually did all stay. Removing rate and seasonal as well would leave a
    # residual, which is a different product and the one the atmospheric
    # screen used to be built from.
    #
    # Nothing is interpolated. A node has a measurement and its neighbours do
    # not, and spreading one node's value over ground that was never measured
    # is what the kriged screen did -- with per-node noise dominating the
    # correlated signal, it cost coherence at every separation.
    # Pixels without a node stay NaN, which is what they are.

    sy_, sx_ = iy[sel], ix[sel]
    comps = []
    # A COMPONENT MUST BE ABLE TO MEET THE CONSENSUS IT IS JUDGED BY. Each
    # component carries its own free datum, so a small one is not a sparser
    # answer -- it is a separately-datumed one, resting on however few arcs
    # its handful of nodes could form. `consensus` already states how many
    # agreeing measurements a value must rest on before it is reported; a
    # component with fewer nodes than that cannot supply them even in
    # principle. It is the caller's own requirement applied to the network
    # itself, not a second threshold: ask for less and smaller ones qualify,
    # and with `consensus=None` the check is off along with all the others.
    _cmin = int(_ma) if _ma is not None else 1
    for c in np.unique(lab[sel]):
        k = np.where(lab[sel] == c)[0]
        if len(k) < _cmin:
            continue
        comps.append((len(k), float(np.median(dcount[sel][k])), k))
    if not comps:
        return None
    # AT MOST _MAX_COMPONENTS, LARGEST FIRST -- past it an int8 label would
    # fold component 128 onto -128 and two unrelated datums would read as one.
    # What is dropped is the smallest, which is also the least trustworthy.
    comps.sort(key=lambda c: -c[0])
    if debug:
        _seen = sorted((int((lab[sel] == c).sum())
                        for c in np.unique(lab[sel])), reverse=True)
        _drop = [z for z in _seen if z < _cmin]
        print(f'DEBUG: network  {len(_seen)} connected component(s); '
              f'{len(_drop)} below the {_cmin}-node consensus floor',
              flush=True)
        for _n_, _d_, _k_ in comps:
            print(f'DEBUG:          size {_n_:>6,}   arcs/node {_d_:5.1f}',
                  flush=True)
        if _drop:
            # SIZES, not just a count. "3 dropped" hides whether that is six
            # nodes or twenty; the floor is per COMPONENT, so a total far
            # above it can still be made of pieces every one of which is
            # below it.
            print(f'DEBUG:          dropped sizes {_drop}  '
                  f'= {sum(_drop)} nodes, each below {_cmin}', flush=True)
    dropped = max(0, len(comps) - _MAX_COMPONENTS)
    comps = comps[:_MAX_COMPONENTS]
    order_size = sorted(range(len(comps)), key=lambda z: -comps[z][0])
    label_of = {z: r for r, z in enumerate(order_size)}
    order_prio = sorted(range(len(comps)), key=lambda z: (-comps[z][1],
                                                          -comps[z][0]))
    # Each node belongs to exactly one component, so writing them cannot
    # contest a pixel -- the arbitration the kriged version needed went with
    # the interpolation.
    keep = []
    for z in order_prio:
        k = comps[z][2]
        keep.append(k)
    kk = np.concatenate(keep)
    lab_all = np.array([label_of[z] for z in order_prio
                        for _ in range(len(comps[z][2]))], dtype=np.int8)
    k_mm = meter2rad * 1e-3

    _3d_fit_ps_array.stats.reset(
        nodes=int(len(kk)), arcs=int(len(ai)), dropped=int(dropped),
        arcs_rejected=int(rejected),
        degree_rejected=drej[sel][kk].astype(np.int32),
        components=[dict(label=label_of[z], size=comps[z][0],
                         degree=comps[z][1],
                         iy=sy_[comps[z][2]].copy(), ix=sx_[comps[z][2]].copy())
                    for z in order_prio],
        fill_order=[label_of[z] for z in order_prio],
        # the network solution itself, per node, in physical units -- the
        # datum is per component and already applied, so these are relative
        # to their own component and free by one constant each
        iy=sy_[kk].copy(), ix=sx_[kk].copy(),
        label=lab_all,
        degree=dcount[sel][kk].astype(np.int32),
        # RADIANS, matching `_3d_arc_fit`'s (gamma, height_rad,
        # velocity_rad_yr, seasonal_rad). Converting to mm or metres in here
        # would put a second length convention inside the library, when
        # `displacement_los()` is meant to be the only place one appears.
        height_rad=hgt[sel][kk].astype(np.float32),
        velocity_rad_yr=vel[sel][kk].astype(np.float32),
        seasonal_rad=(anr[sel][kk] + 1j * ani[sel][kk]).astype(np.complex64))

    if debug:
        _dn = dcount[sel][kk]
        _gn = gnode[sel][kk]
        _gn = _gn[np.isfinite(_gn)]
        print(f'DEBUG: solved   {len(kk)} of {len(live)} nodes kept, '
              f'{int(dropped)} component(s) dropped past the int8 label limit'
              f'   {_3d_lap(_mark)[0]:.1f}s', flush=True)
        print(f'DEBUG:          arcs/node p50 {np.median(_dn):.0f}  '
              f'min {_dn.min()}  max {_dn.max()}', flush=True)
        if len(_gn):
            print(f'DEBUG:          node gamma p50 {np.median(_gn):.3f}  '
                  f'p10 {np.percentile(_gn, 10):.3f}', flush=True)
        # DOES THE SOLUTION SATISFY ITS OWN ARCS? The network solve defines the
        # datum every attached pixel inherits, so its self-consistency is the
        # property to report -- not how coherent the arcs were, which is a
        # different question already answered above. Differences, so the free
        # datum cannot enter.
        _rv = np.abs((vel[ai] - vel[aj]) - dv) / meter2rad * 1e3      # mm/yr
        _rh = np.abs((hgt[ai] - hgt[aj]) - dh) / meter2rad            # m

        # PER COMPONENT, NOT POOLED. Each carries its own datum, so its
        # closure is its own property: a small component whose arcs agree is
        # usable on that datum, and pooling would hide it behind a larger one
        # that does not. Arcs belong to a component when BOTH ends do.
        if _pre is not None:
            _pa, _pj, _pdh, _pdv, _pk = _pre
            _prv = np.abs((vel[_pa] - vel[_pj]) - _pdv) / meter2rad * 1e3
            _prh = np.abs((hgt[_pa] - hgt[_pj]) - _pdh) / meter2rad
            _f = np.isfinite(_prv) & np.isfinite(_prh)
            if _f.any():
                print(f'DEBUG:          PS closure over ALL {int(_f.sum()):,} '
                      f'fitted arcs, rejected included -- the unselected view:',
                      flush=True)
                print(f'DEBUG:            per arc   rate p50 '
                      f'{np.median(_prv[_f]):.3f} p90 '
                      f'{np.percentile(_prv[_f], 90):.3f} mm/yr   height p50 '
                      f'{np.median(_prh[_f]):.2f} p90 '
                      f'{np.percentile(_prh[_f], 90):.2f} m', flush=True)
        _gt = np.asarray(gtake, float)
        _gt = _gt[np.isfinite(_gt)]
        if len(_gt):
            # WHAT THE SOLVE ACTUALLY RESTED ON. The cap keeps a node's best
            # arcs, so the solved set sits far above the threshold that let
            # them in, and judging the residual against `threshold` judges a
            # population that was never used.
            print(f'DEBUG:          solved arcs gamma p50 {np.median(_gt):.3f}'
                  f'  p10 {np.percentile(_gt, 10):.3f}'
                  f'  p90 {np.percentile(_gt, 90):.3f}'
                  f'   (threshold was {float(threshold):.2f})', flush=True)
        print(f'DEBUG:          PS closure by component, SURVIVING arcs only '
              f'(selected for agreeing -- optimistic):', flush=True)
        for _z in order_prio:
            _kk2 = comps[_z][2]
            _nodes = sel[_kk2]
            _mask = np.zeros(N, dtype=bool); _mask[_nodes] = True
            _sel_a = _mask[ai] & _mask[aj]
            if not _sel_a.any():
                continue
            _cv, _ch = _rv[_sel_a], _rh[_sel_a]
            # THE TAIL, NOT JUST THE EXTREME. The rejection cuts at a multiple
            # of a robust scale, so the largest survivor is near that bar by
            # construction and says nothing on its own; how many arcs sit out
            # there is the question the bar cannot answer.
            # WHERE THE BAD ARCS ARE. Most arcs close far better than their
            # own coherence implies, so the ones that do not are a separate
            # population rather than the tail of one. If they concentrate on a
            # few nodes the fault is those nodes; if they are spread evenly it
            # is the arc fit.
            _bd = np.isfinite(_ch) & (_ch > 1.0)
            if _bd.any():
                _ea = ai[_sel_a]; _eb = aj[_sel_a]
                _bc = np.bincount(np.r_[_ea[_bd], _eb[_bd]], minlength=N)
                _tc = np.bincount(np.r_[_ea, _eb], minlength=N)
                _bn = _bc[_nodes]; _tn = _tc[_nodes]
                _sv = np.sort(_bn)[::-1]
                _tp = max(1, int(round(0.05 * len(_bn))))
                _fr = np.where(_tn > 0, _bn / np.maximum(_tn, 1), np.nan)
                print(f'DEBUG:             bad arcs (>1 m): {int(_bd.sum()):,}'
                      f' on {int((_bn > 0).sum()):,} of {len(_bn):,} nodes;'
                      f' worst 5% of nodes carry '
                      f'{_sv[:_tp].sum() / max(_sv.sum(), 1):.0%};'
                      f' per-node share p50 {np.nanmedian(_fr):.1%}'
                      f' p90 {np.nanpercentile(_fr, 90):.1%}'
                      f' max {np.nanmax(_fr):.0%}', flush=True)
            _cf = np.isfinite(_ch)
            if _cf.any():
                _hh = _ch[_cf]
                print(f'DEBUG:             height tail  p90 '
                      f'{np.percentile(_hh, 90):.2f}  p99 '
                      f'{np.percentile(_hh, 99):.2f} m   '
                      f'over 1 m: {int((_hh > 1).sum()):,} '
                      f'({(_hh > 1).mean():.1%})   over 2 m: '
                      f'{int((_hh > 2).sum()):,} ({(_hh > 2).mean():.2%})',
                      flush=True)
            # PER NODE, OVER THIS COMPONENT'S ARCS ONLY -- the same arcs the
            # per-arc figures above describe. Accumulating over every arc
            # would pull in ones reaching outside the component, including
            # nodes below the survival floor that never received a datum, and
            # a node's mean could then exceed the worst arc it averages.
            _acc = np.zeros(N); _cnt2 = np.zeros(N); _acch = np.zeros(N)
            np.add.at(_acc, ai[_sel_a], _cv); np.add.at(_acc, aj[_sel_a], _cv)
            np.add.at(_acch, ai[_sel_a], _ch); np.add.at(_acch, aj[_sel_a], _ch)
            np.add.at(_cnt2, ai[_sel_a], 1.0); np.add.at(_cnt2, aj[_sel_a], 1.0)
            _cn = np.where(_cnt2[_nodes] > 0,
                           _acc[_nodes] / np.maximum(_cnt2[_nodes], 1.0), np.nan)
            _cnh = np.where(_cnt2[_nodes] > 0,
                            _acch[_nodes] / np.maximum(_cnt2[_nodes], 1.0),
                            np.nan)
            _deg2 = _cnt2[_nodes]
            _cnh = _cnh[np.isfinite(_cnh)]
            _cn = _cn[np.isfinite(_cn)]
            if not len(_cn):
                continue
            # PER ARC and PER NODE are different scales and are labelled as
            # such: a node averages its own arcs, so its worst is always
            # milder than the worst single arc. Printing both unlabelled on
            # one line reads as a contradiction.
            print(f'DEBUG:           label {label_of[_z]}  size {comps[_z][0]:>6,}'
                  f'  arcs {int(_sel_a.sum()):>7,}', flush=True)
            print(f'DEBUG:             per arc   rate p50 {np.median(_cv):.3f}'
                  f' max {_cv.max():.3f} mm/yr'
                  f'   height p50 {np.median(_ch):.2f} max {_ch.max():.2f} m',
                  flush=True)
            print(f'DEBUG:             per node  rate p50 {np.median(_cn):.3f}'
                  f' max {_cn.max():.3f} mm/yr'
                  f'   over 1 mm/yr: {int((_cn > 1).sum())} of {len(_cn)}',
                  flush=True)
            # THE OUTPUT, NOT THE INPUT. What a caller receives is the node
            # value, and its precision is the arc scatter divided by the
            # support that averaged it -- the arcs are the measurements, the
            # nodes are the answer, and only the second is delivered.
            if len(_cnh) and _deg2.max() > 0:
                _sd = np.sqrt(np.maximum(_deg2, 1.0))
                print(f'DEBUG:             per node  height p50 '
                      f'{np.median(_cnh):.2f} max {_cnh.max():.2f} m'
                      f'   over 1 m: {int((_cnh > 1).sum()):,} of {len(_cnh):,}',
                      flush=True)
                print(f'DEBUG:             node precision (arc scatter / '
                      f'sqrt support, support p50 {np.median(_deg2):.0f}): '
                      f'rate {np.median(_cn / _sd[:len(_cn)]):.4f} mm/yr   '
                      f'height {np.median(_cnh / _sd[:len(_cnh)]):.3f} m',
                      flush=True)

    if debug:
        # THE ARCS THEMSELVES, so the network's self-consistency can be
        # measured on the solve that actually ran rather than on a copy of it.
        # Set AFTER the stats dict is rebuilt above, or they would be wiped.
        # THE ARCS TO A FILE WHEN ASKED. `.stats` is per-thread, so a caller in
        # the main thread cannot read what a dask worker thread wrote; a dump
        # is the only way to get the solved network out for offline analysis.
        _dump = __import__('os').environ.get('INSARDEV_DUMP_ARCS')
        if _dump:
            np.savez(_dump, ai=ai, aj=aj, dh=dh, dv=dv,
                     g=np.asarray(gtake, dtype=np.float32),
                     z=_dbg_z, resid=_dbg_rv,
                     node_iy=iy, node_ix=ix, node_index=sel[kk],
                     vel=vel[sel][kk], hgt=hgt[sel][kk])
            print(f'DEBUG: dumped {len(ai):,} network arcs to {_dump}', flush=True)
        _3d_fit_ps_array.stats.update(
            arc_i=ai.copy(), arc_j=aj.copy(),
            arc_dh=dh.copy(), arc_dv=dv.copy(),
            arc_gamma=np.asarray(gtake, dtype=np.float32).copy(),
            # arcs index the FULL node space; these map it to the reported
            # arrays, and to the raster, so an arc can be located
            arc_gate_passes=_gate_passes,
            arc_z=_dbg_z.copy(),
            arc_sigma_v=np.minimum(_dbg_sv[ai], _dbg_sv[aj]).copy(),
            arc_resid_irls=_dbg_rv.copy(),
            node_index=sel[kk].copy(),
            node_iy=iy.copy(), node_ix=ix.copy(),
            node_vel=vel.copy(), node_hgt=hgt.copy())

    # THE NETWORK SOLUTION, and nothing raster-shaped: the caller writes these
    # into whichever block holds each node. Full precision -- the attachment
    # reads a node's model to give a DS its datum, so rounding here would
    # round every DS that leans on it.
    return dict(iy=sy_[kk], ix=sx_[kk], label=lab_all,
                U=np.ascontiguousarray(Un[:, sel][:, kk]),
                vel=vel[sel][kk], hgt=hgt[sel][kk], coh=gnode[sel][kk],
                sea=(anr[sel][kk] + 1j * ani[sel][kk]),
                # WHAT THE SOLVE COUNTED, carried with the table. The stats are
                # a function attribute, so under `union=True` -- where the
                # network runs in one process and the attachments in others --
                # they cannot be read where they were written.
                stats=dict(_3d_fit_ps_array.stats))


def _3d_ds_attach(S, cand_ds, ds_nodes, _oy, _ox, lab_out, vel_out,
                  hgt_out, sea_out, coh_out, lvl_out, level_id,
                  ele2phase, t, meter2rad, *,
                  ny, nx, wy, wx, cell, budget, threshold, level,
                  max_dh, max_dv, step_dh, step_dv, iterations,
                  _ma, _ii, _err_h, _err_v, _nth, _st, debug=False):
    """LEVEL 2: DS hung off the DS of level 1, which are FIXED INPUT.

    Level 1 attaches DS to the PS network; this attaches what level 1 could not
    reach to what it did. The distinction that matters is that the level-1
    values arrive as an argument and are never recomputed here -- a chunk's
    partners include nodes another chunk owns and solved, so recomputing them
    would answer with a different network's numbers.

    The output planes are updated IN PLACE; nothing is returned.
    """
    _fy = np.asarray(ds_nodes['iy']).copy()
    _fx = np.asarray(ds_nodes['ix']).copy()
    _fv = ds_nodes['vel'].astype(float)
    _fh = ds_nodes['hgt'].astype(float)
    _fs = np.asarray(ds_nodes['sea']).copy()
    _fl = np.asarray(ds_nodes['label']).copy()
    _done = np.zeros((ny, nx), dtype=bool)
    _done[_oy, _ox] = True
    _done[_fy, _fx] = True
    _st['vouch_rounds'] = []
    for _rnd in range(int(level) - 1):
        vy, vx = np.where(cand_ds & ~_done)
        _st['vouch_candidates'] = int(len(vy))
        if not len(vy):
            break
        _before = int(_st.get('vouch_attached', 0))
        _st['vouch_attached'] = 0
        if True:
            _av = np.abs(S[:, vy, vx])
            Uv = np.ascontiguousarray(
                np.where(_av > 0, S[:, vy, vx] / np.where(_av > 0, _av, 1),
                         0).astype(np.complex64))
            del _av
            # REACH: the DS WINDOW, widened to two only where one did not
            # serve. The window is the scale over which the atmosphere is
            # taken to be common, so it is where a partner is worth having,
            # and the read's halo is exactly one window -- reaching further
            # asks for pixels the block does not hold and is silently
            # truncated at its edges.
            #
            # THE SHORTLIST IS BOUNDED, `consensus` partners per candidate.
            # Pairing every fixed node with every candidate in reach is the
            # same selection arrived at by materialising it first: the count
            # is n_fixed x the density of its box, it GROWS with the network
            # every level, and the arcs beyond the best few are fitted only
            # to be discarded by the consensus. Selecting first costs `kk`
            # slots per candidate instead.
            _hy1, _hx1 = max(wy // 2, 1), max(wx // 2, 1)
            _hy2, _hx2 = max(wy, 1), max(wx, 1)
            _ab = np.abs(S[:, _fy, _fx])
            Ub = np.ascontiguousarray(
                np.where(_ab > 0, S[:, _fy, _fx] / np.where(_ab > 0, _ab, 1),
                         0).astype(np.complex64))
            del _ab
            _fg = np.full((ny, nx), -1, dtype=np.int64)
            _fg[_fy, _fx] = np.arange(len(_fy))
            # THE SAME CAP THE NETWORK USES, and for the same reason. The
            # shortlist is chosen on raw coherence while the consensus votes
            # on FITTED values, so every partner the gates reject is one the
            # vote never sees and `consensus` partners offered is `consensus`
            # only if none is rejected. What the cap must prevent is the
            # opposite case -- a candidate with thousands of nodes in reach --
            # and the network already answers how many arcs are worth keeping
            # per node: beyond `_ARC_CAP` they were measured to add nothing.
            # Level 1's thirty-six is NOT the precedent here; that bound is
            # structural, four PS per window times the nine-window
            # neighbourhood, and the level-2 fixed layer is dense DS with no
            # such lattice.
            _kk = int(max(_ARC_CAP, _ma))
            _ov = np.empty((len(vy), _kk), dtype=np.float32)
            _oj = np.empty((len(vy), _kk), dtype=np.int64)
            _ea = np.zeros(len(vy), dtype=np.int8)
            _vy64, _vx64 = vy.astype(np.int64), vx.astype(np.int64)
            _args = (Uv, Ub, _vy64, _vx64, _fg, _hy1, _hx1, _hy2, _hx2,
                     int(cell[0]), int(cell[1]), int(_kk), float(threshold),
                     _ov, _oj, _ea)
            if _nth > 1 and len(vy) > _nth:
                from concurrent.futures import ThreadPoolExecutor
                _step = -(-len(vy) // _nth)
                _bnd = [(a, min(a + _step, len(vy)))
                        for a in range(0, len(vy), _step)]
                with ThreadPoolExecutor(_nth) as _ex:
                    list(_ex.map(lambda b: _3d_ds_partners(*_args, b[0], b[1]),
                                 _bnd))
            else:
                _3d_ds_partners(*_args, 0, len(vy))
            del _fg
            if debug:
                # RECORDED, NOT PRINTED. One chunk's numbers describe one
                # chunk; the caller has every chunk of the level and reduces
                # them to the one line a reader can actually use.
                _np_ = (_oj >= 0).sum(1)
                _st['lvl_kk'] = int(_kk)
                _st['lvl_early'] = int(_ea.sum())
                _st['lvl_partners'] = _np_.astype(np.int32)
            _kp = _oj >= 0
            ds_s = np.repeat(np.arange(len(vy)), _kk).reshape(-1, _kk)[_kp]
            ds_t = _oj[_kp]
            del _ov, _oj, _kp
            _st['vouch_arcs'] = int(len(ds_s))
            if debug:
                # THE SIZE OF THE PROBLEM, recorded where it is decided rather
                # than after the fit has already allocated over it.
                _st['lvl_cands'] = int(len(vy))
                _st['lvl_fixed'] = int(len(_fy))
                _st['lvl_arcs'] = int(len(ds_s))
            if len(ds_s):
                gd, hd, vd, sd, goodd = _3d_partner_shortlist(
                    Uv, Ub, ds_s, ds_t, _fl, len(vy), ele2phase, t,
                    meter2rad, max_dh, max_dv, step_dh, step_dv, budget,
                    iterations, _ma, threshold,
                    stats=_st, prefix='vouch_', debug=debug,
                    fix_h=_fh, fix_v=_fv, seed=_3d_seed_ds_ds,
                    threads=_nth)
                # THE REPRESENTATIVE IS THE HIGHEST-GAMMA SURVIVOR.
                # Two error terms are in play -- the new arc's, and the
                # base's own inherited one -- and only the first is
                # selectable here. Choosing instead on the base's
                # attachment coherence was measured and is WORSE: base
                # gamma says how coherently that DS attached, not how
                # correct its value is, so optimising for it selects
                # confidence rather than accuracy. The inherited error is
                # bounded by the gate, which has already discarded every
                # partner whose value disagreed.
                _vst = {}
                # RATE AND HEIGHT TOGETHER, as level 1 does and as the
                # solve does. `err_h` was being passed without the heights it
                # judges, so the gate could never run: level 2 was accepting
                # partners on the rate alone, and an arc metres out in height
                # -- a facade, a roof edge, two scatterers at different
                # elevations in one window -- entered the answer unchallenged.
                v2first, v2votes, _okv = _3d_partner_consensus(
                    ds_s, gd, goodd, _fv[ds_t] + vd, len(vy),
                    _ma, _err_v, _ii, h_abs=_fh[ds_t] + hd, err_h=_err_h,
                    labels=(_fl[ds_t] if debug else None),
                    stats=_vst, prefix='vouch_')
                _st.update(_vst)
                _st['vouch_attached'] = int(len(v2first))
                if len(v2first):
                    _si, _bj = ds_s[v2first], ds_t[v2first]
                    vy2, vx2 = vy[_si], vx[_si]

                    # ---- SOLVE THE LEVEL-2 DS AS A NETWORK ---------
                    # Every extension solves its new layer against the
                    # layers already solved, held FIXED: the PS anchor the
                    # level-1 DS, and the PS plus those DS anchor these.
                    # Without it the layer is a star -- each pixel takes
                    # its value from one partner and neighbouring pixels
                    # were never compared -- which is exactly the state
                    # level 1 was in before its own solve existed.
                    _a2 = np.full(len(vy), -1, dtype=np.int64)
                    _a2[_si] = np.arange(len(_si))
                    _e2 = np.flatnonzero(goodd & (_a2[ds_s] >= 0))
                    _e2i, _e2p = _a2[ds_s[_e2]], ds_t[_e2]
                    _m3 = np.zeros((ny, nx), dtype=bool)
                    _m3[vy2, vx2] = True
                    _tv3, _ty3, _tx3 = _3d_topk_kernel(
                        S, wy, wx, tuple(cell), budget, _ma, _m3,
                        threads=_nth)
                    _ix3 = np.full((ny, nx), -1, dtype=np.int64)
                    _ix3[vy2, vx2] = np.arange(len(_si))
                    _hv3 = _tv3[vy2, vx2] >= float(threshold)
                    _s3 = np.repeat(np.arange(len(_si)), _ma
                                    ).reshape(-1, _ma)[_hv3]
                    _py3 = np.clip(vy2[:, None] + _ty3[vy2, vx2],
                                   0, ny - 1)[_hv3]
                    _px3 = np.clip(vx2[:, None] + _tx3[vy2, vx2],
                                   0, nx - 1)[_hv3]
                    _t3 = _ix3[_py3, _px3]
                    _ok3 = _t3 >= 0
                    _n3i, _n3j = _s3[_ok3], _t3[_ok3]
                    _u3 = _n3i < _n3j
                    _n3i, _n3j = _n3i[_u3], _n3j[_u3]
                    _st['vouch_pairs'] = int(len(_n3i))
                    if len(_n3i):
                        # FITTED, not predicted: these are the measurements
                        # the solve consumes, and predicting them from the
                        # values being solved for would be circular.
                        _av3 = np.abs(S[:, vy2, vx2])
                        _Uv3 = np.ascontiguousarray(
                            np.where(_av3 > 0, S[:, vy2, vx2]
                                     / np.where(_av3 > 0, _av3, 1), 0
                                     ).astype(np.complex64))
                        del _av3
                        _pg3, _pdh3, _pdv3 = _3d_pair_fit(
                            _Uv3, _Uv3, _n3i, _n3j, ele2phase, t,
                            _err_v, _err_h, _ii, budget,
                            threads=_nth)
                        _k3 = np.isfinite(_pg3) & (_pg3 >= float(threshold))
                        # THE SAME BOUND ON BOTH KINDS OF EQUATION. The DS->
                        # fixed arcs are already held to `err_dv`/`err_dh`
                        # against the pixel's consensus centre; these were
                        # held to coherence alone, so the one unvetted input
                        # to the solve was the half that ties DS to DS. An
                        # edge whose delta contradicts what BOTH endpoints
                        # independently concluded is not a measurement of the
                        # ground between them, and letting it into the first
                        # least-squares is what lets it choose which arcs the
                        # reject pass then throws out.
                        # THE SAME BOUND, NOT A LOOSER ONE. A difference of
                        # two err-bounded estimates could in principle sit
                        # 2*err apart, and the measured median residual lands
                        # exactly on 1*err, which looks like a threshold cutting
                        # through good edges. Measured, it is not: doubling the
                        # bound keeps twice the edges, attaches exactly the same
                        # pixels, and puts the outliers back (per-DS rate max
                        # 0.84 -> 1.02, height 3.02 -> 3.34). The edges past the
                        # bound are inconsistent, not merely differenced.
                        _cc_v = _vst.get('vouch_centre_v')
                        _cc_h = _vst.get('vouch_centre_h')
                        if _cc_v is not None and len(_n3i):
                            _cv_ = np.asarray(_cc_v, float)[_si]
                            _rv3 = np.abs((_cv_[_n3i] - _cv_[_n3j]) - _pdv3)
                            _k3 &= ~np.isfinite(_rv3) | (_rv3 <= _err_v)
                            if _cc_h is not None:
                                _ch_ = np.asarray(_cc_h, float)[_si]
                                _rh3 = np.abs(
                                    (_ch_[_n3i] - _ch_[_n3j]) - _pdh3)
                                _k3 &= ~np.isfinite(_rh3) | (_rh3 <= _err_h)
                        if debug:
                            # WHICH bound rejects, and by how much. A
                            # difference of two independently estimated
                            # centres carries both their errors, so a
                            # tolerance sized for ONE residual may be
                            # rejecting edges that are consistent.
                            _st['lvl_dsds_in'] = int(len(_n3i))
                            _st['lvl_dsds_kept'] = int(_k3.sum())
                            _cg = np.isfinite(_pg3) & (
                                _pg3 >= float(threshold))
                            if _cc_v is not None and len(_n3i):
                                _fv_ = _cg & np.isfinite(_rv3) & (
                                    _rv3 > _err_v)
                                _st['lvl_dsds_failv'] = int(_fv_.sum())
                                _st['lvl_dsds_rv'] = (
                                    _rv3[_cg & np.isfinite(_rv3)]
                                    / _err_v).astype(np.float32)
                                if _cc_h is not None:
                                    _fh_ = _cg & np.isfinite(_rh3) & (
                                        _rh3 > _err_h)
                                    _st['lvl_dsds_failh'] = int(_fh_.sum())
                                    _st['lvl_dsds_both'] = int(
                                        (_fv_ & _fh_).sum())
                                    _st['lvl_dsds_rh'] = (
                                        _rh3[_cg & np.isfinite(_rh3)]
                                        / _err_h).astype(np.float32)
                        _n3i, _n3j = _n3i[_k3], _n3j[_k3]
                        _pdh3, _pdv3, _pg3 = _pdh3[_k3], _pdv3[_k3], _pg3[_k3]
                    else:
                        _pdh3 = _pdv3 = np.zeros(0)
                        _pg3 = np.zeros(0, np.float32)
                    _v3, _h3, _ns3, _anc3 = _3d_ds_solve(
                        len(_si), _e2i, _e2p, vd[_e2], hd[_e2], gd[_e2],
                        _n3i, _n3j, _pdv3, _pdh3, _pg3,
                        _fv, _fh, _err_v, _err_h, _ii)
                    if debug:
                        # THE FUNNEL: what the consensus vetted, what the
                        # solve was handed, and what it kept. The gap between
                        # the first two is the arcs no gate in physical units
                        # ever saw.
                        _vv = _st.get('vouch_vote_arcs')
                        _st['lvl_vetted'] = int(len(_vv)) if _vv is not None \
                            else 0
                        _st['lvl_solve_in'] = int(len(_e2) + len(_n3i))
                        _st['lvl_solve_kept'] = int(_ns3.sum())
                        # how far the UNVETTED arcs sit from the pixel's own
                        # consensus centre, in the units the bound is stated in
                        _cvv = _st.get('vouch_centre_v')
                        if _cvv is not None and len(_e2):
                            _cc = np.asarray(_cvv)[ds_s[_e2]]
                            _pp = _fv[ds_t[_e2]] + vd[_e2]
                            _dd = np.abs(_pp - _cc) / meter2rad * 1e3
                            _dd = _dd[np.isfinite(_dd)]
                            if len(_dd):
                                _st['lvl_offcentre'] = _dd.astype(np.float32)
                    # CONSENSUS TIES TO THE NETWORK, NOT TO PEERS. A
                    # `DS - DS` equation relates a pixel to another pixel of
                    # its OWN level and carries no datum; counted toward the
                    # same floor it lets a cluster certify itself on its
                    # internal edges. The consensus demanded five partners in
                    # the FIXED layer before the solve, so the solve must
                    # leave five standing.
                    _lv3 = _anc3 >= _ma
                    if debug:
                        _cv2 = getattr(_3d_ds_solve._tl, 'conv', None) or []
                        if _cv2:
                            _it = np.array([c[1] for c in _cv2])
                            _sp = np.array([c[0] for c in _cv2])
                            _nu = np.array([c[2] for c in _cv2])
                            _st['lvl_lsqr_hit'] = int((_sp == 7).sum())
                            _st['lvl_lsqr_n'] = int(len(_sp))
                            _st['lvl_lsqr_itn'] = int(_it.max())
                            _st['lvl_lsqr_unk'] = int(_nu.max())
                        _anc = _anc3
                        if _anc is not None:
                            _lm = _ns3 >= _ma
                            _st['lvl_noanchor'] = int((_lm & (_anc == 0)).sum())
                            _st['lvl_fewanchor'] = int(
                                (_lm & (_anc < _ma)).sum())
                            _st['lvl_passed'] = int(_lm.sum())
                    _st['vouch_unconfirmed'] = int((~_lv3).sum())
                    _v3 = np.where(_lv3, _v3, np.nan)
                    _h3 = np.where(_lv3, _h3, np.nan)
                    # ---- WHAT THIS LEVEL IS WORTH -----------------------
                    # Yield alone cannot say whether a level added good
                    # pixels or merely more of them. This is the level-1 DS
                    # closure applied to this layer: each voting partner
                    # predicts the candidate from its own FIXED value plus
                    # the arc, and the disagreement with what the solve
                    # returned is the error, in mm/yr and m. Like level 1's,
                    # this set is not selected for agreeing, so it needs no
                    # second view.
                    _va3 = _vst.get('vouch_vote_arcs')
                    if debug and _va3 is not None and len(_va3):
                        _vk = np.asarray(_va3, np.int64)
                        _sl = _a2[ds_s[_vk]]
                        _mv = _sl >= 0
                        _sl, _vk = _sl[_mv], _vk[_mv]
                        if len(_sl):
                            _fw = np.isfinite(_v3[_sl]) & np.isfinite(_h3[_sl])
                            _sl, _vk = _sl[_fw], _vk[_fw]
                    else:
                        _sl = np.zeros(0, np.int64)
                    if debug and len(_sl):
                        _cv3 = np.abs((_fv[ds_t[_vk]] + vd[_vk])
                                      - _v3[_sl]) / meter2rad * 1e3
                        _ch3 = np.abs((_fh[ds_t[_vk]] + hd[_vk])
                                      - _h3[_sl]) / meter2rad
                        # PER PIXEL, as the network reports per node: a DS
                        # averages its partners, so its worst is milder than
                        # the worst single partner and the two scales must
                        # be labelled apart.
                        _qa = np.zeros(len(_si)); _qb = np.zeros(len(_si))
                        _qc = np.zeros(len(_si))
                        np.add.at(_qa, _sl, _cv3)
                        np.add.at(_qb, _sl, _ch3)
                        np.add.at(_qc, _sl, 1.0)
                        _qh = _qc > 0
                        _pv3 = _qa[_qh] / _qc[_qh]
                        _ph3 = _qb[_qh] / _qc[_qh]
                        # THE SAMPLES, NOT A SUMMARY OF THEM. Percentiles do
                        # not average across chunks -- a median of medians is
                        # not the median -- so the level's reducer needs the
                        # values themselves to answer for the whole level.
                        _st['lvl_clo_gamma'] = float(np.median(gd[v2first]))
                        _st['lvl_clo_arc_v'] = _cv3.astype(np.float32)
                        _st['lvl_clo_arc_h'] = _ch3.astype(np.float32)
                        _st['lvl_clo_ds_v'] = _pv3.astype(np.float32)
                        _st['lvl_clo_ds_h'] = _ph3.astype(np.float32)
                    # gated on the solve, and written AFTER it --
                    # the label used to be set before the solve ran
                    lab_out[vy2, vx2] = np.where(_lv3, _fl[_bj], -1)
                    vel_out[vy2, vx2] = _v3.astype(np.float32)
                    # WHICH LEVEL FOUND THIS PIXEL, so a caller can see the
                    # ladder rather than only its total.
                    lvl_out[vy2, vx2] = np.where(
                        _lv3, np.int8(level_id + _rnd), np.int8(-1))
                    hgt_out[vy2, vx2] = _h3.astype(np.float32)
                    coh_out[vy2, vx2] = gd[v2first].astype(np.float32)
                    sea_out[vy2, vx2] = (
                        (_fs[_bj].real + sd[v2first].real)
                        + 1j * (_fs[_bj].imag + sd[v2first].imag)
                        ).astype(np.complex64)
                    _st.update(
                        vouch_iy=vy2.copy(), vouch_ix=vx2.copy(),
                        vouch_label=_fl[_bj],
                        vouch_gamma=gd[v2first].copy(),
                        vouch_votes=np.asarray(v2votes, dtype=np.int32),
                        vouch_velocity_rad_yr=_v3.astype(np.float32),
                        vouch_height_rad=_h3.astype(np.float32))
        _got = int(_st.get('vouch_attached', 0))
        _st['vouch_rounds'].append(_got)
        if debug:
            _st['lvl_left'] = int(len(vy))
            _st['lvl_attached'] = int(_got)
            _st['lvl_network'] = int(len(_fy) + _got)
        if not _got:
            break                  # nothing added: further rounds cannot
        _ny_ = _st['vouch_iy']; _nx_ = _st['vouch_ix']
        _keepn = np.isfinite(_st['vouch_velocity_rad_yr'])
        _fy = np.r_[_fy, _ny_[_keepn]]
        _fx = np.r_[_fx, _nx_[_keepn]]
        _fv = np.r_[_fv, _st['vouch_velocity_rad_yr'][_keepn].astype(float)]
        _fh = np.r_[_fh, _st['vouch_height_rad'][_keepn].astype(float)]
        _fs = np.r_[_fs, sea_out[_ny_[_keepn], _nx_[_keepn]]]
        _fl = np.r_[_fl, _st['vouch_label'][_keepn]]
        _done[_ny_, _nx_] = True


def _3d_ps_attach(scenes, q, nodes, date_values, *, spacing, bperp=None,
                  window=(32, 128), threshold=0.5, cell=(2, 8), geometry,
                  budget=None, level=1, max_dh=100.0, max_dv=25.0,
                  step_dh=4.0, step_dv=2.0, consensus, iterations=8,
                  err_dh=5.0, err_dv=1.0, threads=None, debug=False,
                  out_stats=None):
    """PASS 2, per block: the network written out, and the DS hung off it.

    `nodes` is the solved table from `_3d_ps_network`, with positions in THIS
    block's index space. Under `union=True` those nodes were solved together
    with other blocks' -- so the datum crossing the seam is the point -- but
    the pixels never left their own block.
    """
    import os as _os
    _nth = max(1, int(threads) if threads else (_os.process_cpu_count() or 1))
    _ma = _3d_consensus(consensus)
    _ii = max(1, int(iterations))
    S = np.ascontiguousarray(scenes, dtype=np.complex64)
    n, ny, nx = S.shape
    wy, wx, pey, pex = _3d_windows(window)
    lab_out = np.full((ny, nx), -1, dtype=np.int8)
    vel_out = np.full((ny, nx), np.nan, dtype=np.float32)
    hgt_out = np.full((ny, nx), np.nan, dtype=np.float32)
    coh_out = np.full((ny, nx), np.nan, dtype=np.float32)
    sea_out = np.full((ny, nx), np.nan + 1j * np.nan, dtype=np.complex64)
    # THE LEVEL THAT PRODUCED EACH PIXEL: 0 for a PS node, 1 for a DS attached
    # to the network, n for one attached to the level n-1 DS. -1 where nothing
    # was measured. Carried so the ladder can be looked at, not just counted.
    lvl_out = np.full((ny, nx), -1, dtype=np.int8)
    # CLEARED BEFORE THE EARLY RETURN, so a block that solves nothing
    # cannot leave the previous block's DS table visible on this thread.
    _3d_fit_ps_array.stats.reset(nodes=0, arcs=0, dropped=0,
                                 components=[], fill_order=[])
    if nodes is None or len(np.asarray(nodes['iy'])) == 0:
        return lab_out, vel_out, hgt_out, sea_out, coh_out, lvl_out
    t, ele2phase, meter2rad, car = _3d_fit_frame(date_values, bperp,
                                                 geometry, n)
    # stated in metres and mm/yr, used in radians -- see the network
    _err_h = float(err_dh) * meter2rad
    _err_v = float(err_dv) * meter2rad / 1e3
    iy = np.asarray(nodes['iy'])
    ix = np.asarray(nodes['ix'])
    Un = np.ascontiguousarray(nodes['U'], dtype=np.complex64)
    lab_all = np.asarray(nodes['label'], dtype=np.int8)
    vel = np.asarray(nodes['vel'])
    hgt = np.asarray(nodes['hgt'])
    gnode = np.asarray(nodes['coh'])
    anr = np.asarray(np.real(nodes['sea']))
    ani = np.asarray(np.imag(nodes['sea']))
    # the table arrives already reduced to the kept nodes, so the selections
    # the attachment applies are the identity -- kept so the code below reads
    # the same whether it ran here or in a network solved somewhere else
    sel = np.arange(len(iy))
    kk = np.arange(len(iy))
    # Each node belongs to exactly one component, so writing them cannot
    # contest a pixel.
    # A NODE MAY STAND OUTSIDE THIS BLOCK. Under `union=True` the table is the
    # whole scene's, and a node beyond these bounds is still a partner a DS
    # here can reach -- the PS extent is far wider than a chunk -- but its
    # model belongs to the block that holds its pixel, and only that block
    # writes it. This is the reach a per-block network cannot offer: without
    # it a DS near a chunk edge is judged on the handful of nodes its own
    # chunk happens to contain.
    _own = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    _oy, _ox = iy[_own], ix[_own]
    lab_out[_oy, _ox] = lab_all[_own]
    vel_out[_oy, _ox] = vel[_own].astype(np.float32)           # rad/yr
    lvl_out[_oy, _ox] = np.where(np.isfinite(vel[_own]), 0, -1)  # PS nodes
    hgt_out[_oy, _ox] = hgt[_own].astype(np.float32)           # rad
    coh_out[_oy, _ox] = gnode[_own].astype(np.float32)
    sea_out[_oy, _ox] = (anr[_own] + 1j * ani[_own]).astype(np.complex64)
    if q is None:
        q = np.full((ny, nx), np.nan, dtype=np.float32)
    # THE STATS BELONG TO WHOEVER SOLVED, and are REBUILT here for every
    # block. They are a function attribute, so under `union=True` -- the
    # network in one process, the attachments in others -- they cannot be read
    # where they were written, and a worker that fitted a previous block still
    # holds that block's dict. Seeded from the table this block was handed, so
    # what is reported is what this block actually wrote.
    _3d_fit_ps_array.stats.reset(nodes=0, arcs=0, dropped=0, components=[],
                                 fill_order=[])
    _3d_fit_ps_array.stats.update(nodes.get('stats') or {})
    _3d_fit_ps_array.stats['iy'] = iy
    _3d_fit_ps_array.stats['ix'] = ix
    # ---- attach the DS to the network ----------------------------------
    # THE PS EXTENT IS THE REACH, NOT THE DS WINDOW. A PS is defined by holding
    # a coherent arc out to the PS extent -- that is what separates it from a DS
    # -- so limiting the attachment to the DS window contradicts the test that
    # selected the partner in the first place.
    #
    # The DS window was justified by a reach table showing DS->PS coherence
    # dying past ~200 m, but that decays that fast only for a RAW inner product.
    # These arcs are FITTED (`_3d_arc_fit` solves the differential height and
    # rate), and a fitted arc carries much further -- the same distinction
    # arcs() documents for the long PS test, where raw coherence selected
    # nothing at km range and the fitted test still found pairs.
    #
    # It matters because the DS window holds too few PS to vote with: the
    # consensus rule needs `min_agreeing` partners and the window rarely supplies
    # them, so DS were being rejected for the reach of the search rather than
    # for the quality of their arcs.
    #
    # BEST, not nearest. Where several nodes are in reach the arc coherence says
    # which one actually carries the datum, and distance does not: gamma over a
    # DS window varies far more between partners than with the few tens of
    # metres separating them.
    #
    # A DS that clears the gate inherits its partner's height, rate, seasonal
    # AND component label, so it lands on the same datum as the node -- that is
    # the whole point of attaching rather than solving it alone. One that does
    # not clear it stays NaN: a DS with no coherent path to the network has no
    # datum, and a value written without one would be a different network's.
    if level >= 1:
        cand_ds = np.isfinite(q) & (q >= float(threshold))
        cand_ds[_oy, _ox] = False                        # nodes are not DS here
        dy_, dx_ = np.where(cand_ds)
        if len(dy_):
            ny_ps, nx_ps = iy[sel][kk], ix[sel][kk]
            # THE PARTNERS ARE THE 3x3 DS-WINDOW NEIGHBOURHOOD. The DS window
            # is the scale over which the atmosphere is taken to be common, so
            # it is also the scale over which an arc means anything; a
            # candidate takes the nodes of its own window and the eight around
            # it and fits every one. The set is bounded by the winner grid, so
            # the cost is a property of the windows rather than of the scene.
            _3d_fit_ps_array.stats['ds_candidates'] = int(len(dy_))
            _3d_fit_ps_array.stats['ds_reached'] = int(len(dy_))
            # unit phasors once for the candidates and once for the nodes,
            # rather than re-slicing the scene inside every batch
            _ad = np.abs(S[:, dy_, dx_])
            Ud_all = np.ascontiguousarray(
                np.where(_ad > 0, S[:, dy_, dx_] / np.where(_ad > 0, _ad, 1),
                         0).astype(np.complex64))
            del _ad
            Ups = np.ascontiguousarray(Un[:, sel][:, kk])
            if len(ny_ps):
                # the DS window names the neighbourhood; the two extent
                # slots are carried for shape and are not a reach any more
                _pos = (dy_, dx_, ny_ps, nx_ps, 0, 0, wy, wx)
                _t_ds = time.monotonic()
                ksrc, ktgt, ga, dha, dva, dsa, good = _3d_shortlist_ds_ps(
                    Ud_all, Ups, lab_all, len(dy_),
                    ele2phase, t, meter2rad, max_dh, max_dv, step_dh, step_dv,
                    budget, iterations, _ma, threshold,
                    stats=_3d_fit_ps_array.stats, prefix='ds_', debug=debug,
                    # THE PS ARE THE FIXED LAYER: solved onto one datum, so
                    # a partner's value is its own plus what the arc measures.
                    fix_h=hgt[sel][kk], fix_v=vel[sel][kk], threads=_nth,
                    pos=_pos)
                _3d_fit_ps_array.stats['ds_arcs'] = int(len(ksrc))
                _lap, _t_ds = _3d_lap(_t_ds)
                _3d_fit_ps_array.stats['ds_fit_s'] = _lap
                # THE PARTNERS ARE MEASUREMENTS, SO THEY ARE SOLVED THE WAY
                # THE ARCS ARE. Each partner gives the DS a complete answer, so
                # several partners are repeated measurements of one quantity --
                # the same situation as a node's arcs, and it gets the same
                # treatment: IRLS to find the consistent set, rejection beyond
                # `reject_sigma` robust sigma, and `min_agreeing` survivors before
                # the value counts as measured at all. With one an error is
                # invisible and with two it cannot be localised, whether the two
                # are arcs or partners.
                #
                # Where the pixel is a clean scatterer the partners agree and
                # this changes nothing. Where it is a MIXTURE its phase belongs
                # to no single target, the arcs fit different parameters, and
                # the strongest one is as likely to hold the wrong component as
                # the right one.
                #
                # The representative is the best surviving arc, not the fitted
                # centre: height, seasonal and component label still come from
                # one real partner exactly as before -- only WHICH partner
                # changes. A fitted centre would sit between partners and be
                # backed by none of them.
                if debug:
                    # THE WHOLE SHORTLIST, not just the partner that won. A
                    # joint DS solve needs every DS->PS equation, and the star
                    # attachment keeps only one per pixel.
                    _gsl = np.flatnonzero(good)
                    _3d_fit_ps_array.stats.update(
                        dsarc_src=ksrc[_gsl].copy(),        # index into dy_/dx_
                        dsarc_ds_iy=dy_.copy(), dsarc_ds_ix=dx_.copy(),
                        dsarc_tgt=ktgt[_gsl].copy(),        # index into the nodes
                        dsarc_dv=dva[_gsl].copy(),
                        dsarc_dh=dha[_gsl].copy(),
                        dsarc_gamma=ga[_gsl].copy())
                v_abs = vel[sel][kk][ktgt] + dva
                h_abs = hgt[sel][kk][ktgt] + dha
                # A LOCAL DICT, NOT THE MODULE'S. `stats` is a function
                # attribute shared by every task in the worker PROCESS, and
                # the centres come back out of it to become the DS values --
                # so with more than one thread per worker a block reads the
                # centres of whichever block wrote last. That is a wrong
                # answer when the lengths happen to match and an IndexError
                # when they do not. Results must not travel through global
                # state; only the diagnostics may, and they are merged after.
                _cst = {}
                first, votes, _okds = _3d_partner_consensus(
                    ksrc, ga, good, v_abs, len(dy_), _ma, _err_v, _ii,
                    h_abs=h_abs, err_h=_err_h,
                    labels=(lab_all[ktgt] if debug else None),
                    stats=_cst, prefix='ds_')
                _3d_fit_ps_array.stats.update(_cst)
                _lap, _t_ds = _3d_lap(_t_ds)
                _3d_fit_ps_array.stats['ds_consensus_s'] = _lap
                if len(first):
                    ds_i, ps_i = ksrc[first], ktgt[first]
                    yy2, xx2 = dy_[ds_i], dx_[ds_i]
                    # THE VALUE IS THE CONSENSUS, NOT ITS BEST WITNESS. The
                    # partners that voted are repeated measurements of this
                    # pixel, and the vote already found which of them agree;
                    # taking the highest-gamma one and dropping the rest
                    # spends the redundancy on the test and keeps none of it
                    # for the answer. The component still comes from the best
                    # partner: a label is named, not averaged.
                    if debug:
                        # ARE THE PARTNERS INDEPENDENT OF EACH OTHER? The
                        # independence cell is enforced between the CANDIDATE
                        # and each partner, never between the partners: five
                        # of them inside one cell are one sample of the ground
                        # counted five times, and they agree because they are
                        # the same measurement, not because the value is right.
                        _va0 = _cst.get('ds_vote_arcs')
                        if _va0 is not None and len(_va0):
                            _vk = np.asarray(_va0, np.int64)
                            _cy0, _cx0 = int(cell[0]), int(cell[1])
                            _pc = (ny_ps[ktgt[_vk]] // max(_cy0, 1)
                                   ).astype(np.int64) * (1 << 20) + \
                                  (nx_ps[ktgt[_vk]] // max(_cx0, 1))
                            _o0 = np.lexsort((_pc, ksrc[_vk]))
                            _ss, _pp = ksrc[_vk][_o0], _pc[_o0]
                            _new = np.r_[True, (_ss[1:] != _ss[:-1]) |
                                         (_pp[1:] != _pp[:-1])]
                            _ncell = np.bincount(_ss[_new], minlength=len(dy_))
                            _nvote = np.bincount(ksrc[_vk], minlength=len(dy_))
                            _hv0 = _nvote > 0
                            _3d_fit_ps_array.stats['lvl_pcells'] = \
                                _ncell[_hv0].astype(np.int32)
                            _3d_fit_ps_array.stats['lvl_pvotes'] = \
                                _nvote[_hv0].astype(np.int32)
                            # HOW FAR the agreeing partners actually are. The
                            # PS extent is the reach, and an isolated DS takes
                            # whatever it can reach -- five properly separated
                            # partners, good coherence, and every arc far past
                            # the separation where a differential model still
                            # describes one piece of ground.
                            _dyp = (ny_ps[ktgt[_vk]] - dy_[ksrc[_vk]]) \
                                * float(spacing[0])
                            _dxp = (nx_ps[ktgt[_vk]] - dx_[ksrc[_vk]]) \
                                * float(spacing[1])
                            _rr = np.sqrt(_dyp**2 + _dxp**2)
                            _sd = np.zeros(len(dy_)); _sn = np.zeros(len(dy_))
                            np.add.at(_sd, ksrc[_vk], _rr)
                            np.add.at(_sn, ksrc[_vk], 1.0)
                            _3d_fit_ps_array.stats['lvl_parcm'] = (
                                _sd[_hv0] / np.maximum(_sn[_hv0], 1)
                                ).astype(np.float32)
                    if debug:
                        # DO CELL-MATES SHARE PARTNERS? Two pixels inside one
                        # independence cell are one sample of the ground, so
                        # they should attach to the same nodes and agree. Their
                        # raw coherences differ slightly, though, and the
                        # shortlist is RANKED -- a tiny difference reorders the
                        # top `consensus` and hands them different partner sets,
                        # whose errors are then inherited separately.
                        _vs = _cst.get('ds_vote_arcs')
                        if _vs is not None and len(_vs):
                            _vk2 = np.asarray(_vs, np.int64)
                            _cid = (dy_ // max(int(cell[0]), 1)).astype(
                                np.int64) * (1 << 20) + \
                                (dx_ // max(int(cell[1]), 1))
                            # partner-set signature per candidate
                            _ord2 = np.lexsort((ktgt[_vk2], ksrc[_vk2]))
                            _ss2, _tt2 = ksrc[_vk2][_ord2], ktgt[_vk2][_ord2]
                            _bnd = np.r_[0, np.flatnonzero(
                                _ss2[1:] != _ss2[:-1]) + 1, len(_ss2)]
                            _who = _ss2[_bnd[:-1]]
                            _sets = [frozenset(_tt2[a:b].tolist())
                                     for a, b in zip(_bnd[:-1], _bnd[1:])]
                            _bycell = {}
                            for _w, _st5 in zip(_who, _sets):
                                _bycell.setdefault(int(_cid[_w]),
                                                   []).append((_w, _st5))
                            _jac, _dv = [], []
                            for _grp in _bycell.values():
                                if len(_grp) < 2:
                                    continue
                                for _i2 in range(min(len(_grp), 6)):
                                    for _j2 in range(_i2 + 1,
                                                     min(len(_grp), 6)):
                                        _a5, _b5 = _grp[_i2][1], _grp[_j2][1]
                                        _u = len(_a5 | _b5)
                                        _jac.append(len(_a5 & _b5) / max(_u, 1))
                            if _jac:
                                _3d_fit_ps_array.stats['lvl_mate_jac'] = \
                                    np.asarray(_jac, np.float32)
                    _cv = _cst.get('ds_centre_v')
                    _ch = _cst.get('ds_centre_h')
                    _si = ksrc[first]
                    h_ds = (hgt[sel][kk][ps_i] + dha[first] if _ch is None
                            else np.asarray(_ch)[_si])
                    lab_ds = lab_all[ps_i]
                    v_ds = (vel[sel][kk][ps_i] + dva[first] if _cv is None
                            else np.asarray(_cv)[_si])
                    if debug:
                        # ---- DS CLOSURE ---------------------------------
                        # Every accepted partner against the value the pixel
                        # adopted. The PS are the fixed layer, so a DS holds
                        # no free datum and each partner is a complete answer
                        # for it -- their spread IS the attachment's
                        # consistency. Unlike the network's, this set is not
                        # selected for agreeing, so it needs no second view.
                        _vad = np.full(len(dy_), np.nan)
                        _had = np.full(len(dy_), np.nan)
                        _vad[ds_i] = v_ds
                        _had[ds_i] = h_ds
                        _va = _cst.get('ds_vote_arcs')
                        _m2 = np.zeros(len(ksrc), dtype=bool)
                        if _va is not None and len(_va):
                            _m2[np.asarray(_va, np.int64)] = True
                        _m2 &= np.isfinite(_vad[ksrc])
                        if _m2.any():
                            _crv = np.abs(
                                (vel[sel][kk][ktgt[_m2]] + dva[_m2])
                                - _vad[ksrc[_m2]]) / meter2rad * 1e3
                            _crh = np.abs(
                                (hgt[sel][kk][ktgt[_m2]] + dha[_m2])
                                - _had[ksrc[_m2]]) / meter2rad
                            # PER PIXEL, the same way the network reports per
                            # node: a DS averages its own partners, so its
                            # worst is milder than the worst single partner and
                            # the two scales must be labelled apart.
                            _na = np.zeros(len(dy_)); _nc = np.zeros(len(dy_))
                            _nb = np.zeros(len(dy_))
                            np.add.at(_na, ksrc[_m2], _crv)
                            np.add.at(_nb, ksrc[_m2], _crh)
                            np.add.at(_nc, ksrc[_m2], 1.0)
                            _hv = _nc > 0
                            _pv2 = _na[_hv] / _nc[_hv]
                            _ph2 = _nb[_hv] / _nc[_hv]
                            # THE SAMPLES, for the level's reducer. The
                            # summaries below describe THIS block; percentiles
                            # do not average, so a level spanning many blocks
                            # has to pool the values themselves.
                            _3d_fit_ps_array.stats.update(
                                lvl_clo_arc_v=_crv.astype(np.float32),
                                lvl_clo_arc_h=_crh.astype(np.float32),
                                lvl_clo_ds_v=_pv2.astype(np.float32),
                                lvl_clo_ds_h=_ph2.astype(np.float32))
                            _3d_fit_ps_array.stats.update(
                                ds_clo_n=int(_m2.sum()),
                                ds_clo_rv=float(np.nanmedian(_crv)),
                                ds_clo_rv90=float(np.nanpercentile(_crv, 90)),
                                ds_clo_rh=float(np.nanmedian(_crh)),
                                ds_clo_rh90=float(np.nanpercentile(_crh, 90)),
                                ds_clo_pix=int(_hv.sum()),
                                ds_clo_nv=float(np.median(_pv2)),
                                ds_clo_nvmax=float(_pv2.max()),
                                ds_clo_nh=float(np.median(_ph2)),
                                ds_clo_nhmax=float(_ph2.max()),
                                ds_clo_over1=int((_pv2 > 1.0).sum()))

                    # ---- SOLVE THE DS AS A NETWORK ----------------------
                    # Each DS has just been validated against `consensus` PS
                    # partners -- but only against ITS OWN. Two DS a few
                    # metres apart were never compared, so nothing related
                    # them and nothing could notice when they disagreed.
                    # Measured before this existed: a fifth of neighbouring
                    # pairs differed by more than 1 mm/yr and the worst by
                    # tens, every one of them individually unanimous.
                    #
                    # So the attachment's equations are KEPT rather than
                    # reduced to one partner each, the neighbour equations
                    # are added, and the lot is solved together:
                    #
                    #     DS_i - PS_p = dv_ip     the PS are FIXED
                    #     DS_i - DS_j = dv_ij     what was missing
                    #
                    # The PS stay fixed because they are the certified layer
                    # and define the datum: a hundred thousand weak DS solved
                    # WITH a few hundred nodes would outvote the network that
                    # anchors them. Fixing them also leaves the DS system
                    # with no free constant of its own.
                    _att = np.full(len(dy_), -1, dtype=np.int64)
                    _att[ds_i] = np.arange(len(ds_i))
                    _e = np.flatnonzero(good & (_att[ksrc] >= 0))
                    _ei, _ep = _att[ksrc[_e]], ktgt[_e]
                    # neighbours, best `consensus` by COHERENCE inside the DS
                    # window -- the search the arc kernel already does, run
                    # only at the attached pixels
                    _m2 = np.zeros((ny, nx), dtype=bool)
                    _m2[yy2, xx2] = True
                    _tv, _ty, _tx = _3d_topk_kernel(
                        S, wy, wx, tuple(cell), budget, _ma, _m2,
                        threads=_nth)
                    _idx = np.full((ny, nx), -1, dtype=np.int64)
                    _idx[yy2, xx2] = np.arange(len(ds_i))
                    _have = _tv[yy2, xx2] >= float(threshold)
                    _si = np.repeat(np.arange(len(ds_i)), _ma
                                    ).reshape(-1, _ma)[_have]
                    _py = np.clip(yy2[:, None] + _ty[yy2, xx2], 0, ny - 1)[_have]
                    _px = np.clip(xx2[:, None] + _tx[yy2, xx2], 0, nx - 1)[_have]
                    _ti = _idx[_py, _px]
                    _ok = _ti >= 0
                    _ni, _nj = _si[_ok], _ti[_ok]
                    _u = _ni < _nj                      # each pair once
                    _ni, _nj = _ni[_u], _nj[_u]
                    _st2 = _3d_fit_ps_array.stats
                    _st2['ds_pairs'] = int(len(_ni))
                    if len(_ni):
                        _adn = np.abs(S[:, yy2, xx2])
                        _Ud = np.ascontiguousarray(
                            np.where(_adn > 0, S[:, yy2, xx2]
                                     / np.where(_adn > 0, _adn, 1), 0
                                     ).astype(np.complex64))
                        del _adn
                        _pg, _pdh, _pdv = _3d_pair_fit(
                            _Ud, _Ud, _ni, _nj, ele2phase, t,
                            _err_v, _err_h, _ii,
                            budget, threads=_nth)
                        _keep2 = np.isfinite(_pg) & (_pg >= float(threshold))
                        # THE SAME BOUND THE DS->PS ARCS ANSWER TO. Those are
                        # held to `err_dv`/`err_dh` against each pixel's
                        # consensus centre; these were held to coherence
                        # alone, so the half of the solve that ties DS to DS
                        # was the one input no gate in physical units ever
                        # saw. An edge whose delta contradicts what BOTH
                        # endpoints independently concluded is not a
                        # measurement of the ground between them.
                        if _cv is not None and len(_ni):
                            _cvd = np.asarray(_cv, float)[ds_i]
                            _rv2 = np.abs((_cvd[_ni] - _cvd[_nj]) - _pdv)
                            _keep2 &= ~np.isfinite(_rv2) | (_rv2 <= _err_v)
                        if _ch is not None and len(_ni):
                            _chd = np.asarray(_ch, float)[ds_i]
                            _rh2 = np.abs((_chd[_ni] - _chd[_nj]) - _pdh)
                            _keep2 &= ~np.isfinite(_rh2) | (_rh2 <= _err_h)
                        if debug:
                            _st2['lvl_dsds_in'] = int(len(_ni))
                            _st2['lvl_dsds_kept'] = int(_keep2.sum())
                        _ni, _nj = _ni[_keep2], _nj[_keep2]
                        _pdh, _pdv = _pdh[_keep2], _pdv[_keep2]
                        _pg = _pg[_keep2]
                    else:
                        _pdh = _pdv = np.zeros(0)
                        _pg = np.zeros(0, np.float32)
                    v_ds, h_ds, _nsurv, _anc1 = _3d_ds_solve(
                        len(ds_i), _ei, _ep, dva[_e], dha[_e], ga[_e],
                        _ni, _nj, _pdv, _pdh, _pg,
                        vel[sel][kk], hgt[sel][kk], _err_v, _err_h, _ii)
                    # THE SAME RULE THE NODES GET: enough of a pixel's own
                    # equations must survive the gate. One that keeps too few
                    # was not confirmed by the network, and a value carried by
                    # what is left would be a compromise nothing backs.
                    # the same rule at level 1: five surviving DS->PS
                    # equations, not five equations of any kind
                    _live2 = _anc1 >= _ma
                    if debug:
                        _cv2 = getattr(_3d_ds_solve._tl, 'conv', None) or []
                        if _cv2:
                            _it = np.array([c[1] for c in _cv2])
                            _sp = np.array([c[0] for c in _cv2])
                            _nu = np.array([c[2] for c in _cv2])
                            _st2['lvl_lsqr_hit'] = int((_sp == 7).sum())
                            _st2['lvl_lsqr_n'] = int(len(_sp))
                            _st2['lvl_lsqr_itn'] = int(_it.max())
                            _st2['lvl_lsqr_unk'] = int(_nu.max())
                        _anc = _anc1
                        if _anc is not None:
                            _lm = _nsurv >= _ma
                            _st2['lvl_noanchor'] = int((_lm & (_anc == 0)).sum())
                            _st2['lvl_fewanchor'] = int(
                                (_lm & (_anc < _ma)).sum())
                            _st2['lvl_passed'] = int(_lm.sum())
                    _st2['ds_unconfirmed'] = int((~_live2).sum())
                    v_ds = np.where(_live2, v_ds, np.nan)
                    h_ds = np.where(_live2, h_ds, np.nan)

                    # THE LABEL IS PART OF THE ANSWER, SO IT IS GATED WITH
                    # IT. `conncomp` says which datum a value belongs to, and a
                    # pixel the joint solve declined to place has no value and
                    # therefore no datum -- writing the partner's label anyway
                    # reports membership of a component the solve refused to
                    # grant, and a caller masking on `conncomp` rather than on
                    # the value would take it.
                    lab_out[yy2, xx2] = np.where(_live2, lab_ds, -1)
                    vel_out[yy2, xx2] = v_ds.astype(np.float32)
                    lvl_out[yy2, xx2] = np.where(_live2, 1, -1)  # DS on the PS
                    hgt_out[yy2, xx2] = h_ds.astype(np.float32)
                    coh_out[yy2, xx2] = ga[first].astype(np.float32)
                    sea_out[yy2, xx2] = ((anr[sel][kk][ps_i] + dsa[first].real)
                                         + 1j * (ani[sel][kk][ps_i]
                                                 + dsa[first].imag)
                                         ).astype(np.complex64)
                    # THE NODE TABLE IS A RESULT, so it also goes somewhere
                    # this CALL owns. Level 2 stands on it, and read back from
                    # a process-global dict a block can inherit the nodes of
                    # whichever block shared its worker.
                    _dsn = dict(
                        ds_attached=int(len(first)),
                        ds_partners=np.bincount(
                            ksrc[good], minlength=len(dy_))[ksrc[first]],
                        ds_gamma=ga[first].copy(),
                        ds_votes=np.asarray(votes, dtype=np.int32),
                        ds_iy=yy2.copy(), ds_ix=xx2.copy(), ds_label=lab_ds,
                        ds_height_rad=h_ds.astype(np.float32),
                        ds_velocity_rad_yr=v_ds.astype(np.float32),
                        # the annual travels with the DS as well: it is fitted
                        # on the attaching arc and made absolute by its
                        # partner's own value, exactly as height and rate are
                        ds_seasonal_rad=((anr[sel][kk][ps_i] + dsa[first].real)
                                         + 1j * (ani[sel][kk][ps_i]
                                                 + dsa[first].imag)
                                         ).astype(np.complex64))
                    _dsn['lvl_cands'] = int(len(dy_))
                    _dsn['lvl_fixed'] = int(len(np.unique(ktgt)))
                    _dsn['lvl_arcs'] = int(len(ksrc))
                    _dsn['lvl_left'] = int(_3d_fit_ps_array.stats.get(
                        'ds_admissible', 0))
                    _dsn['lvl_attached'] = int(len(first))
                    for _k in ('lvl_clo_arc_v', 'lvl_clo_arc_h',
                               'lvl_clo_ds_v', 'lvl_clo_ds_h',
                               'lvl_dsds_in', 'lvl_dsds_kept',
                               'lvl_dsds_failv', 'lvl_dsds_failh',
                               'lvl_dsds_both', 'lvl_noanchor',
                               'lvl_fewanchor', 'lvl_passed',
                               'lvl_pcells', 'lvl_pvotes',
                               'lvl_parcm', 'lvl_mate_jac',
                               'lvl_lsqr_hit', 'lvl_lsqr_n',
                               'lvl_lsqr_itn', 'lvl_lsqr_unk'):
                        _v = _3d_fit_ps_array.stats.get(_k)
                        if _v is not None:
                            _dsn[_k] = _v
                    _3d_fit_ps_array.stats.update(_dsn)
                    if out_stats is not None:
                        out_stats.update(_dsn)
        # ---- ROUND 2: DS TO DS, THE SAME WAY DS ATTACHED TO PS -----------
        # A pixel can be plainly connectable and still fail round 1 -- not
        # because anything contradicts it, but because the PS are too sparse
        # there to field `min_agreeing` of them. That is a property of the
        # ground, not of the pixel.
        #
        # So round 2 offers the ATTACHED DS as partners, under exactly the
        # rules round 1 used: the best `min_agreeing` by arc coherence, one
        # component, IRLS to find the consistent set, rejection beyond
        # `reject_sigma`, unanimity among all of them, and then the value from
        # the best SURVIVING arc rather than from the fitted centre.
        #
        # The partners are worth trusting because of what they already
        # survived: every one of them was itself carried by `min_agreeing`
        # agreeing PS. And they are near -- inside the DS window arcs are
        # short and coherent where the reach to a PS is not.
        #
        # REACH DIFFERS BY WHAT EACH SIDE PROVED. A PS is the pixel certified
        # to hold an arc out to the PS extent, which is why round 1 may reach
        # that far. A DS carries no such certificate, so it may only vouch
        # inside the window it was itself measured in.
        _st = _3d_fit_ps_array.stats
        # ---- DENSIFY UNTIL IT STOPS ADDING ----------------------------
        # Round 2 attaches what is left to the DS that round 1 placed. Once it
        # has, the network is LARGER, and pixels that had too few partners
        # before may now have enough -- so the same round run again reaches
        # further, with no new rule and no new code. `level` counts how many
        # times it may run: 2 is one round, 3 is two, and so on.
        #
        # Each round holds every earlier layer FIXED, exactly as round 2 holds
        # the PS and the level-1 DS. That is what keeps one datum across all
        # of them, and it is why the rounds compose at all.
        if level >= 2 and _ma is not None and int(_st.get('ds_attached', 0)):
            _3d_ds_attach(
                S, cand_ds, dict(iy=_st['ds_iy'], ix=_st['ds_ix'],
                                 vel=_st['ds_velocity_rad_yr'],
                                 hgt=_st['ds_height_rad'],
                                 sea=_st['ds_seasonal_rad'],
                                 label=_st['ds_label'],
                                 gamma=_st.get('ds_gamma')),
                _oy, _ox, lab_out, vel_out, hgt_out, sea_out, coh_out,
                lvl_out, 2, ele2phase, t, meter2rad, ny=ny, nx=nx, wy=wy,
                wx=wx,
                cell=cell, budget=budget, threshold=threshold, level=level,
                max_dh=max_dh, max_dv=max_dv, step_dh=step_dh,
                step_dv=step_dv, iterations=iterations, _ma=_ma, _ii=_ii,
                _err_h=_err_h, _err_v=_err_v, _nth=_nth, _st=_st,
                debug=debug)
    if debug:
        _s = _3d_fit_ps_array.stats
        _att = int(_s.get('ds_attached', 0))
        _cnd = int(_s.get('ds_candidates', 0))
        if level < 1:
            print('DEBUG: DS       level=0 -- the PS network only', flush=True)
        elif _cnd:
            print(f'DEBUG: DS       {_cnd:,} candidates, '
                  f'{int(_s.get("ds_reached", 0)):,} reached a node over '
                  f'{int(_s.get("ds_arcs", 0)):,} arcs'
                  f'   fit {_s.get("ds_fit_s", 0.0):.1f}s'
                  f' + consensus {_s.get("ds_consensus_s", 0.0):.1f}s',
                  flush=True)
            # THE BLOCK'S NUMBERS ARE NOT THE LEVEL'S. Level 1 runs once per
            # block, so printing here repeats every line as many times as
            # there are chunks -- unreadable at a few dozen, and still not the
            # answer, since the level's yield is the sum and its error
            # distribution the pooled samples. Recorded for the level's
            # reducer, which prints once when every block has finished.
            _s['lvl_no_consensus'] = int(_s.get('ds_no_consensus', 0))
            _s['lvl_too_few'] = int(_s.get('ds_too_few', 0))
            _s['lvl_multi_comp'] = int(_s.get('ds_multi_component', 0))
            _s['lvl_straddled'] = int(_s.get('ds_shortlist_straddled', 0))
            _s['lvl_cross_votes'] = int(_s.get('ds_cross_component_votes') or 0)
            _g = _s.get('ds_gamma')
            if _g is not None and len(_g):
                _s['lvl_gamma'] = np.asarray(_g, np.float32)
        else:
            print('DEBUG: DS       no candidates cleared the threshold',
                  flush=True)
        _vc = int(_s.get('vouch_candidates', 0))
        _va = int(_s.get('vouch_attached', 0))
        if _vc:
            _vno = int(_s.get('vouch_no_consensus', 0))
            _vfew = int(_s.get('vouch_too_few', 0))
            _vg = _s.get('vouch_gamma')
            print(f'DEBUG: DS->DS   {_vc:,} still unresolved over '
                  f'{int(_s.get("vouch_arcs", 0)):,} arcs to attached DS',
                  flush=True)
            print(f'DEBUG:          {int(_s.get("vouch_admissible", 0)):,} held '
                  f'an admissible arc: {_va:,} attached, {_vno:,} did not'
                  + (f'   gamma p50 {np.median(_vg):.3f}'
                     if _vg is not None and len(_vg) else ''), flush=True)
            print(f'DEBUG:          of those {_vno:,}: {_vfew:,} had too few '
                  f'partners, {_vno - _vfew:,} had enough and disagreed',
                  flush=True)
            _vx2 = _s.get('vouch_cross_component_votes')
            if _vx2 is not None:
                print(f'DEBUG:          cross-component votes among attached: '
                      f'{int(_vx2):,}' + ('' if _vx2 == 0 else '   <-- BUG'),
                      flush=True)
        # the closure of this level is reported once, by the level's reducer,
        # over the pooled samples of every block -- see `lvl_clo_*` above
        # the total is the LEVEL's, and one block does not know it
        _s['lvl_ps'] = int(len(_s.get('iy', ())))
    return lab_out, vel_out, hgt_out, sea_out, coh_out, lvl_out


def __3d_fit_ps_array_gated(scenes, date_values, *, spacing, bperp=None,
                            window=(32, 128), threshold=0.5, cell=(2, 8),
                            geometry, budget=None, level=1,
                            max_dh=100.0, max_dv=25.0, step_dh=4.0,
                            step_dv=2.0, max_seasonal=5.0,
                            consensus, iterations=8, threads=None,
                            debug=False):
    # THE THREE STAGES, COMPOSED. One block in, one model out -- the split
    # exists so `union=True` can run the middle stage ONCE over nodes gathered
    # from every block, without ever merging the blocks themselves into a
    # raster. Run in sequence here, it is the fit as it always was.
    _3d_fit_ps_array.stats.reset(nodes=0, arcs=0, dropped=0, components=[],
                                 fill_order=[])
    _kw = dict(spacing=spacing, bperp=bperp, window=window,
               threshold=threshold, cell=cell, geometry=geometry,
               budget=budget, iterations=iterations, threads=threads,
               debug=debug)
    nodes0 = _3d_ps_nodes(scenes, date_values, **_kw)
    if nodes0 is None:
        S = np.asarray(scenes)
        ny, nx = (S.shape[1], S.shape[2]) if S.ndim == 3 else (0, 0)
        return (np.full((ny, nx), -1, dtype=np.int8),
                np.full((ny, nx), np.nan, dtype=np.float32),
                np.full((ny, nx), np.nan, dtype=np.float32),
                np.full((ny, nx), np.nan + 1j * np.nan, dtype=np.complex64),
                np.full((ny, nx), np.nan, dtype=np.float32),
                np.full((ny, nx), -1, dtype=np.int8))
    net = _3d_ps_network(nodes0['U'], nodes0['iy'], nodes0['ix'], date_values,
                         bperp=bperp, window=window, threshold=threshold,
                         geometry=geometry, budget=budget,
                         consensus=consensus, iterations=iterations,
                         max_dh=max_dh, max_dv=max_dv, step_dh=step_dh,
                         step_dv=step_dv, max_seasonal=max_seasonal,
                         threads=threads, debug=debug)
    return _3d_ps_attach(scenes, nodes0['q'], net, date_values,
                         spacing=spacing, bperp=bperp, window=window,
                         threshold=threshold, cell=cell, geometry=geometry,
                         budget=budget, level=level, max_dh=max_dh,
                         max_dv=max_dv, step_dh=step_dh, step_dv=step_dv,
                         consensus=consensus, iterations=iterations,
                         threads=threads, debug=debug)



# PER-THREAD FROM THE START, so no call site ever meets a plain dict here.
_3d_fit_ps_array.stats = _ThreadStats()


@_numba.njit(nogil=True, cache=True)
def _cascade_tile(t2, mask, thr2, cnt, mx, y, x0, hx, gx0, ny_cnt, nx_cnt):
    """One tile's fused count+max with both ends credited, GIL-free.

    `t2` holds squared coherent-sum magnitudes for the tile's own pixels
    against their one-sided neighbourhood; every admissible score is counted
    against the threshold and folded into the running maximum AT BOTH ENDS.
    Credits land wherever they fall -- the caller clips to the owned region
    by slicing the accumulators it hands in.
    """
    w, ndy, span = t2.shape
    for i in range(w):
        xi = x0 + i
        for dy in range(ndy):
            base = dy * span
            for c in range(span):
                if mask[i, base + c] == 0.0:
                    continue
                v = t2[i, dy, c]
                col = gx0 + c
                if v >= thr2:
                    cnt[y, xi] += 1
                    if 0 <= col < nx_cnt and y + dy < ny_cnt:
                        cnt[y + dy, col] += 1
                if v > mx[y, xi]:
                    mx[y, xi] = v
                if 0 <= col < nx_cnt and y + dy < ny_cnt:
                    if v > mx[y + dy, col]:
                        mx[y + dy, col] = v


def _cascade_count_max(S, wy, wx, cell, thr):
    """Per-pixel coherent-arc COUNT and best-arc coherence, in one sweep.

    The same one-sided rectangular products as `_3d_arcs_kernel`, reduced to
    the two numbers the cascade needs: how many admissible arcs clear `thr`,
    and the best coherence reached. Returns (count int32, best float32) for
    the WHOLE array handed in -- the caller slices owned pixels out.
    """
    n, ny, nx = S.shape
    cy, cx = int(cell[0]), int(cell[1])
    hy, hx = wy // 2, wx // 2
    K = 2 * n
    Xp = np.zeros((ny, nx + 2 * hx, K), dtype=np.float32)
    slab = max(1, min(ny, int(64 * 1024 * 1024 // max(n * nx * 8, 1))))
    for y0 in range(0, ny, slab):
        y1 = min(y0 + slab, ny)
        blk = S[:, y0:y1, :]
        a = np.abs(blk)
        f = np.isfinite(a) & (a > 0)
        o = f.all(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            u = np.where(f, blk / np.where(f, a, 1), 0)
        u *= o[None, :, :]
        Xp[y0:y1, hx:hx + nx, :n] = np.moveaxis(u.real, 0, -1)
        Xp[y0:y1, hx:hx + nx, n:] = np.moveaxis(u.imag, 0, -1)
        del blk, a, f, o, u
    cnt = np.zeros((ny, nx), np.int32)
    mx = np.zeros((ny, nx), np.float32)
    thr2 = (float(thr) * n) ** 2
    Bx = max(8, hx)
    masks = {}
    for y in range(ny):
        ndy = min(hy + 1, ny - y)
        for x0 in range(0, nx, Bx):
            w = min(Bx, nx - x0)
            span = w + 2 * hx
            A1 = Xp[y, hx + x0:hx + x0 + w, :]
            A2 = np.empty((w, K), dtype=np.float32)
            A2[:, :n] = A1[:, n:]
            A2[:, n:] = -A1[:, :n]
            Bk = np.ascontiguousarray(
                Xp[y:y + ndy, x0:x0 + span, :].transpose(2, 0, 1)
            ).reshape(K, ndy * span)
            t = A1 @ Bk
            Ci = A2 @ Bk
            np.multiply(t, t, out=t)
            np.multiply(Ci, Ci, out=Ci)
            t += Ci
            key = (w, ndy)
            if key not in masks:
                dxm = (np.arange(span)[None, None, :] - hx
                       - np.arange(w)[:, None, None])
                dyv = np.arange(ndy)[None, :, None]
                mm = ((np.abs(dxm) <= hx)
                      & ~((dyv < cy) & (np.abs(dxm) < cx))
                      & ~((dyv == 0) & (dxm <= 0)))
                masks[key] = mm.reshape(w, ndy * span).astype(np.float32)
            m = masks[key]
            t *= m
            _cascade_tile(t.reshape(w, ndy, span), m, thr2, cnt, mx,
                          y, x0, hx, x0 - hx, ny, nx)
    best = np.sqrt(mx, out=mx) / n
    return cnt, best


def _cascade_pass1(block, owned, origin, wy, wx, cell, thr, consensus,
                   threads=1, token=None):
    """One chunk of the cascade's dense scan: the DS-candidate rank raster and
    the PS-candidate winners, from a single pass over the chunk's data.

    `block` is the chunk WITH its full-window halo; `owned` = (y0, y1, x0, x1)
    names the region this task answers for, in block-local indices, and
    `origin` = (gy, gx) is the block's [0,0] in FULL-raster coordinates -- the
    winner cells live on the GLOBAL (wy//2, wx//2) lattice, so a chunk that
    does not start on a lattice line must still cut cells where the raster
    does. The halo is a full DS window per
    side because a winner cell owned by its origin reaches half a window past
    the owned edge, and gating those pixels needs THEIR windows complete.

    Returns (rank, Wser, wiy, wix):
      rank : (owned) float32 -- best arc coherence where the pixel holds at
             least `consensus` admissible arcs over `thr`, NaN otherwise.
      Wser : (dates, cells_y, cells_x) complex64 -- the winner pixel's raw
             series per (wy//2, wx//2) cell whose origin is owned; NaN series
             where the cell holds no gated pixel.
      wiy, wix : (cells_y, cells_x) int32 -- the winner's pixel position in
             the FULL raster, -1 where none.
    """
    S = np.asarray(block, dtype=np.complex64)
    n, ny, nx = S.shape
    y0, y1, x0, x1 = owned
    hy, hx = wy // 2, wx // 2
    _th = max(1, int(threads))
    if _th > 1 and ny >= 2 * (hy + 1):
        from concurrent.futures import ThreadPoolExecutor
        H = max(hy + 1, -(-ny // _th))
        bands = [(a, min(a + H, ny)) for a in range(0, ny, H)]
        cnt = np.empty((ny, nx), np.int32)
        best = np.empty((ny, nx), np.float32)

        def _band(b):
            ya, yb = b
            a0 = max(0, ya - hy)
            b0 = min(ny, yb + hy)
            c_, m_ = _cascade_count_max(S[:, a0:b0], wy, wx, cell, thr)
            cnt[ya:yb] = c_[ya - a0:yb - a0]
            best[ya:yb] = m_[ya - a0:yb - a0]
        with ThreadPoolExecutor(_th) as ex:
            list(ex.map(_band, bands))
    else:
        cnt, best = _cascade_count_max(S, wy, wx, cell, thr)
    rank_full = np.where(cnt >= int(consensus), best, np.nan).astype(np.float32)
    rank = rank_full[y0:y1, x0:x1]

    # winners: one per (hy, hx) cell of the GLOBAL lattice whose origin is
    # owned, chosen as the argmax of the gated rank over the cell -- pixels
    # up to half a window into the halo, whose gating the full-window halo
    # made exact
    gy, gx = origin
    gy0, gx0 = gy + y0, gx + x0
    cy0 = (-(-gy0 // hy) * hy) - gy
    cx0 = (-(-gx0 // hx) * hx) - gx
    oy = np.arange(cy0, y1, hy)
    ox = np.arange(cx0, x1, hx)
    Wser = np.full((n, len(oy), len(ox)), np.nan, dtype=np.complex64)
    wiy = np.full((len(oy), len(ox)), -1, dtype=np.int32)
    wix = np.full((len(oy), len(ox)), -1, dtype=np.int32)
    for a, ya in enumerate(oy):
        for b, xa in enumerate(ox):
            cell_r = rank_full[ya:min(ya + hy, ny), xa:min(xa + hx, nx)]
            if not np.isfinite(cell_r).any():
                continue
            k = np.nanargmax(cell_r)
            dy_, dx_ = divmod(int(k), cell_r.shape[1])
            Wser[:, a, b] = S[:, ya + dy_, xa + dx_]
            wiy[a, b] = gy + ya + dy_
            wix[a, b] = gx + xa + dx_
    return rank, Wser, wiy, wix


def _cascade_ps(Wser, wiy, wix, ele2phase, t, meter2rad, wy, wx, py, px,
                thr, consensus, max_dh=100.0, max_dv=25.0, step_dh=4.0,
                step_dv=2.0, iterations=8, threads=1):
    """PS candidates from the merged winner grid: fitted long arcs only.

    The winner grid is the raster one pyramid level up, so reach is counted in
    (wy//2, wx//2) CELLS: partners closer than the DS window in both axes are
    the short-arc regime and excluded; partners beyond the PS window carry no
    common atmosphere and are not attempted. Partners are ranked raw and the
    best few FITTED -- a long arc only counts fitted.

    Returns per-cell (gamma, dh, dv, arcs): the best fitted long-arc
    coherence, its differentials, and how many fitted long arcs cleared
    `thr`; NaN/0 where the cell has no winner or no coherent long arc.
    """
    n, NY, NX = Wser.shape
    flat = Wser.reshape(n, -1)
    a = np.abs(flat)
    ok = np.isfinite(a).all(axis=0) & (a > 0).all(axis=0)
    idx = np.flatnonzero(ok)
    gamma = np.full((NY, NX), np.nan, np.float32)
    dh = np.full((NY, NX), np.nan, np.float32)
    dv = np.full((NY, NX), np.nan, np.float32)
    arcs = np.zeros((NY, NX), np.int32)
    if len(idx) < 2:
        return gamma, dh, dv, arcs
    with np.errstate(invalid='ignore', divide='ignore'):
        U = (flat[:, idx] / a[:, idx]).astype(np.complex64)
    cy_ = np.asarray(idx // NX, np.int64)
    cx_ = np.asarray(idx % NX, np.int64)
    # REACH IN CELLS: beyond the DS window, inside the PS window. The extent is
    # a FULL box centred on the cell, so the reach is HALF of it -- the same
    # rule the node network gets from its Chebyshev query and the attachment
    # from its tiles. Counted from the extent itself, this test reached twice
    # as far as the window it documents, and twice as far as fit3d for the
    # same parameter.
    ry = max(1, (py // 2) // (wy // 2))
    rx = max(1, (px // 2) // (wx // 2))
    ey, ex = wy // (wy // 2), wx // (wx // 2)
    m = len(idx)
    kk = min(int(consensus), m - 1)
    # THE REACH IS APPLIED TO THE OPERAND, NOT THE PRODUCT. Scoring every
    # candidate against every other and masking afterwards computes the arcs
    # the window has already refused -- most of them, since the extent covers a
    # fraction of the lattice -- and then sorts all of it. Tiling the
    # candidates and taking the cells a tile can reach multiplies only what can
    # be kept, and turns the sort into a partition over that much smaller set.
    # The winners sit on a regular lattice in row-major order, so a tile's
    # reachable rows are a contiguous slice and only the columns need a test.
    src_l, tgt_l = [], []
    ty, tx = max(1, ry // 2), max(1, rx // 2)
    for r0 in range(0, NY, ty):
        r1 = min(r0 + ty, NY)
        s0, s1 = np.searchsorted(cy_, r0), np.searchsorted(cy_, r1)
        if s1 <= s0:
            continue
        t0 = np.searchsorted(cy_, max(0, r0 - ry))
        t1 = np.searchsorted(cy_, r1 - 1 + ry, side='right')
        for c0 in range(0, NX, tx):
            c1 = min(c0 + tx, NX)
            si = s0 + np.flatnonzero((cx_[s0:s1] >= c0) & (cx_[s0:s1] < c1))
            if not len(si):
                continue
            ti = t0 + np.flatnonzero((cx_[t0:t1] >= c0 - rx)
                                     & (cx_[t0:t1] <= c1 - 1 + rx))
            if not len(ti):
                continue
            G = np.abs(U[:, si].conj().T @ U[:, ti]) / n
            ddy = np.abs(cy_[si][:, None] - cy_[ti][None, :])
            ddx = np.abs(cx_[si][:, None] - cx_[ti][None, :])
            G[(ddy < ey) & (ddx < ex)] = -1.0      # short-arc regime
            G[(ddy > ry) | (ddx > rx)] = -1.0      # the tile over-reaches a
            #                                        little; the pair test does
            #                                        not
            k_ = min(kk, G.shape[1])
            if k_ < 1:
                continue
            top = (np.argpartition(-G, k_ - 1, axis=1)[:, :k_]
                   if k_ < G.shape[1] else np.argsort(-G, axis=1)[:, :k_])
            keep = np.take_along_axis(G, top, 1) > 0
            src_l.append(np.repeat(si, k_)[keep.ravel()])
            tgt_l.append(ti[top.ravel()][keep.ravel()])
    src = np.concatenate(src_l) if src_l else np.empty(0, np.int64)
    tgt = np.concatenate(tgt_l) if tgt_l else np.empty(0, np.int64)
    if len(src):
        ga, dha, dva, _ = _3d_arc_batch(
            U, U, src, tgt, ele2phase, t, meter2rad, max_dh, max_dv,
            step_dh, step_dv, 1024.0, iterations, threads=threads)
        good = np.isfinite(ga) & (ga >= float(thr))
        np.add.at(arcs.ravel(), idx[src[good]], 1)
        order = np.lexsort((-np.where(good, ga, -np.inf), src))
        first = order[np.r_[True, np.diff(src[order]) > 0]]
        first = first[good[first]]
        gamma.ravel()[idx[src[first]]] = ga[first]
        dh.ravel()[idx[src[first]]] = dha[first]
        dv.ravel()[idx[src[first]]] = dva[first]
    return gamma, dh, dv, arcs


def _warmup_numba_cache():
    """Compile numba kernels once in the main process so dask workers load from cache."""
    _tv = np.zeros((1, 1, 3), np.float32)
    _m = np.ones((1, 3), np.float32)
    _own = np.ones(1, bool)
    _par = np.ones((1, 1), bool)
    _v = np.full((1, 1, 2), -1.0, np.float32)
    _y = np.zeros((1, 1, 2), np.int16)
    _x = np.zeros((1, 1, 2), np.int16)
    _3d_topk_tile(_tv, _m, _own, _par, _v, _y, _x, 0, 0, 1, -1, 0, 1)
    _3d_topk_stream(np.array([0.5, np.nan, 0.7]),
                    np.array([0, 0, 0], np.int64), 1, 2)
    _cascade_tile(np.ones((1, 1, 3), np.float32), np.ones((1, 3), np.float32),
                  0.5, np.zeros((1, 1), np.int32), np.zeros((1, 1), np.float32),
                  0, 0, 1, -1, 1, 1)


_warmup_numba_cache()
