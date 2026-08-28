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
import numpy as np


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
_SIGMA_FLOOR = 0.05

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
    """How much agreement is required, as (min_agreeing, reject_sigma, passes).

    One argument for both halves of the solve, because it is one question asked
    twice: an arc must agree with the network, a partner must agree with the
    other partners.

        None        ask for none of it -- plain least squares on the network and
                    the best arc for a DS. That is what a caller wants when THEY
                    are choosing `threshold` and do not want a second rule
                    moving the answer underneath them.
        n           n survivors required, the rest left at their defaults
        (n, k)      and outliers rejected beyond k robust sigma
        (n, k, i)   and i reweighting passes

    `min_agreeing` and `reject_sigma` come back None when the checks are off.
    Parsed in ONE place and called from the public method as well, so a bad
    value raises where it was written rather than inside a dask block minutes
    later.
    """
    # THE DEFAULTS LIVE HERE AND ONLY HERE. `consensus` exists so these are
    # the caller's to set, so keeping module constants for them would put the
    # real values somewhere the caller cannot see or reach.
    #
    # reject_sigma: an outlier bound in the usual statistical sense, not a
    # quality level -- arc coherence already decides what is admitted at all.
    # The sigma comes from the robust solution, so an outlier cannot widen the
    # bar that judges it.
    REJECT_SIGMA, IRLS_PASSES = 5.0, 5
    # The binding constraint is ESTIMATING the scale, not redundancy. With one
    # measurement an error is invisible and with two it cannot be localised, but
    # three does not fix it: the robust scale of three residuals is the smaller
    # of the two gaps, and of four it is the mean of the middle two, so in both
    # cases whichever pair happens to land closest sets the bar. Two partners
    # close together collapse the scale and the rest are rejected on the
    # accident of spacing rather than on the pixel.
    #
    # Below the floor the gate rejects on that accident rather than on quality,
    # so it costs coverage without testing anything. Asking for less is asking
    # the library to report points it cannot defend.
    # `consensus=None` is the honest way to say the caller is tuning `threshold`
    # instead, and it stays available -- so this refuses an undefendable middle
    # ground, not a capability.
    FLOOR = 5
    if consensus is None:
        return None, None, IRLS_PASSES
    c = ((consensus,) if isinstance(consensus, (int, float, np.integer,
                                                np.floating))
         else tuple(consensus))
    if len(c) == 1:
        ma, ar, ii = int(c[0]), REJECT_SIGMA, IRLS_PASSES
    elif len(c) == 2:
        ma, ar, ii = int(c[0]), float(c[1]), IRLS_PASSES
    elif len(c) == 3:
        ma, ar, ii = int(c[0]), float(c[1]), int(c[2])
    else:
        raise ValueError(
            'consensus takes None, min_agreeing, (min_agreeing, reject_sigma) '
            f'or (min_agreeing, reject_sigma, irls_passes); got {consensus!r}')
    if ar <= 0 or ii < 1:
        raise ValueError(f'consensus values must be positive; got {consensus!r}')
    if ma < FLOOR:
        raise ValueError(
            f'consensus needs at least {FLOOR} agreeing; got {consensus!r}. '
            'Use consensus=None to switch the checks off entirely.')
    return ma, ar, ii


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

def _3d_arcs_kernel(block, window_y, window_x, cell=(2, 8), budget=None):
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
    """
    S = np.asarray(block)
    n, ny, nx = S.shape
    wy, wx = int(window_y), int(window_x)
    cy, cx = (int(cell[0]), int(cell[1])) if cell is not None else (2, 8)
    hy, hx = wy // 2, wx // 2
    if n < 2 or ny == 0 or nx == 0:
        return np.full((ny, nx), np.nan, dtype=np.float32)

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
            del t, tv, tg

    seen = best >= 0
    out = np.sqrt(best, out=best, where=seen) / n
    return np.where(ok & seen, out, np.nan).astype(np.float32)


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
                  threshold=0.5, budget=None, device='cpu', iterations=8):
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
                              budget=budget, device=device,
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


def _3d_arc_fit(arc, ele2phase, t, meter2rad, max_dh=100.0, max_dv=50.0,
                step_dh=4.0, step_dv=2.0, budget=None, max_seasonal=5.0,
                device='cpu', iterations=8):
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
    kh = 0 if no_h else int(np.ceil(_GUARD * float(max_dh) / float(step_dh) - 1e-9))
    kv = int(np.ceil(_GUARD * float(max_dv) / float(step_dv) - 1e-9))
    gh = np.arange(-kh, kh + 1) * float(step_dh)
    gv = np.arange(-kv, kv + 1) * float(step_dv)
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
                step_dh, step_dv, budget, max_seasonal, device,
                iterations)
        return gs, hs, vs, ss

    A = np.abs(arc)
    Z = np.where(A > 0, arc / np.where(A > 0, A, 1.0), 0).astype(np.complex64)
    nv = (A > 0).sum(axis=0)
    del A
    tt = np.asarray(t, dtype=np.float64) * meter2rad * 1e-3
    hh = (np.zeros_like(tt) if no_h
          else np.asarray(ele2phase, dtype=np.float64) * meter2rad)

    # BOTH STAGES RUN ON THE GPU WHEN ONE IS ASKED FOR. They are large regular
    # kernels -- one product and a fixed loop -- so they map straight across
    # and the arithmetic is identical rather than approximated. Whether that
    # pays at a given batch size is HARDWARE, not something to hardcode: the
    # batch follows the dask chunk budget, and a device with cheap launches
    # wins at sizes where one with expensive launches does not. `device` is
    # honoured as asked; `budget` is the dial that sizes the work.
    _dev = None
    if device not in (None, 'cpu'):
        from .utils_torch import get_torch_device
        _d = str(getattr(get_torch_device(device), 'type', 'cpu'))
        if _d != 'cpu':
            _dev = _d

    # ---- stage 1: lattice, one product ---------------------------------
    P = np.stack(np.meshgrid(gh, gv, indexing='ij'), -1).reshape(-1, 2)
    C = np.exp(-1j * (np.outer(hh, P[:, 0])
                      + np.outer(tt, P[:, 1]))).astype(np.complex64)
    # no division by nv here: it is constant per arc, so it cannot move the
    # argmax, and gamma is computed once at the end from the refined model
    if _dev is not None:
        import torch
        _Zg = torch.from_numpy(Z).to(_dev)
        k = torch.argmax(torch.abs(_Zg.T @ torch.from_numpy(C).to(_dev)),
                         dim=1).cpu().numpy()
    else:
        k = np.argmax(np.abs(Z.T @ C), axis=1)
    if not (max_seasonal and max_seasonal > 0):
        del C                      # kept below: the seasonal stage re-solves on it
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
    elif _dev is not None:
        import torch
        _U = torch.from_numpy(U.astype(np.float32)).to(_dev)
        _P = torch.from_numpy(PINV.astype(np.float32)).to(_dev)
        _TH = torch.from_numpy(TH0.astype(np.float32)).to(_dev)
        for _ in range(iterations):
            _ph = _U @ _TH.T
            _R = _Zg * torch.exp(torch.complex(torch.zeros_like(_ph), -_ph))
            _mu = _R.sum(dim=0)
            _am = torch.abs(_mu)
            _R = _R * torch.conj(
                _mu / torch.where(_am > 0, _am, torch.ones_like(_am)))[None, :]
            _TH = _TH + (_P @ _R.imag).T
        TH = _TH.cpu().numpy().astype(np.float64)
        del _Zg, _U, _P, _TH, _ph, _R, _mu, _am
        if _dev == 'mps':
            torch.mps.empty_cache()
    else:
        TH = TH0.copy()
        for _ in range(iterations):
            R = Z * np.exp(-1j * (U @ TH.T)).astype(np.complex64)
            mu = R.sum(axis=0)
            R *= np.conj(mu / np.where(np.abs(mu) > 0, np.abs(mu), 1.0))[None, :]
            TH = TH + (PINV @ R.imag).T

    R = Z * np.exp(-1j * (U @ TH.T)).astype(np.complex64)
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
        k_mm = meter2rad * 1e-3                       # radians per mm of LOS
        car = np.exp(2j * np.pi * np.asarray(t, dtype=np.float64))
        comb = 2.0 * np.pi / k_mm                # sideband spacing, mm/yr
        sat = np.pi / (2.0 * k_mm)               # where a linearised step saturates
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
        c_guard = _GUARD * float(max_seasonal)
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
        Cbank = np.exp(-1j * k_mm * (np.outer(car.real, CG[:, 0])
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
            [U, np.stack([k_mm * car.real, k_mm * car.imag], 1)], axis=1)
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
                    Rk = (Z * np.exp(-1j * (U @ TH_k.T))).astype(np.complex64)
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
                    Cm = np.exp(1j * k_mm * (np.outer(car.real, np.ones(m)) * bth[:, -2]
                                             + np.outer(car.imag, np.ones(m)) * bth[:, -1])
                                ).astype(np.complex64)
                    k2 = np.argmax(np.abs((Z * np.conj(Cm)).T @ C), axis=1)
                    TH_r = (P[k2][:, 1:] if no_h else P[k2]).astype(np.float64)
                    del Cm
            THs = bth
            for _ in range(iterations):
                Rs = Z * np.exp(-1j * (Us @ THs.T)).astype(np.complex64)
                mus = Rs.sum(axis=0)
                Rs *= np.conj(mus / np.where(np.abs(mus) > 0,
                                             np.abs(mus), 1.0))[None, :]
                THs = THs + (PINVs @ Rs.imag).T
            Rs = Z * np.exp(-1j * (Us @ THs.T)).astype(np.complex64)
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
        seas = (thbest[:, U.shape[1]] + 1j * thbest[:, U.shape[1] + 1]) * k_mm
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
    rate_tol = (np.pi / (meter2rad * 1e-3)) if seasonal else 2.0 * float(step_dv)
    dh_tol = 2.0 * float(step_dh)
    if seasonal and not no_h:
        # Height and rate are CORRELATED in this design, so when the refinement
        # moves the rate to another sideband the height must follow, dragged
        # by their correlation. Gating dh at two lattice cells then rejects
        # pixels whose rate is correct. The tolerance follows the coupling
        # instead of a fixed cell count.
        cor = abs(float(np.corrcoef(hh, tt)[0, 1]))
        dh_tol = max(dh_tol, cor * (np.std(tt) / max(np.std(hh), 1e-30))
                     * rate_tol)
    edge = np.abs(dv) > float(max_dv)
    if max_seasonal and max_seasonal > 0:
        # the annual is bounded like the other two: outside the range the
        # caller stated, the answer is NaN and not a clipped one
        edge = edge | (np.abs(seas) / k_mm > float(max_seasonal))
    runaway = np.abs(dv - TH0[:, -1]) > rate_tol
    if not no_h:
        edge = edge | (np.abs(dh) > float(max_dh))
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
            np.where(bad, np.nan, dh * meter2rad),          # rad per unit ele2phase
            np.where(bad, np.nan, dv * meter2rad * 1e-3),   # rad/yr
            np.where(bad, np.nan, seas))               # rad, complex


def _3d_arc_fit_brute(arc, ele2phase, t, meter2rad, h_range=150.0, v_range=60.0,
                      h_step=0.5, v_step=0.25, device=None, budget=None):
    """Exhaustive (height, rate) scan -- the REFERENCE the ladder is checked against.

    _3d_arc_fit walks a coarse-to-fine ladder and can land in the wrong basin;
    that is not hypothetical, it happened: it returned a low coherence at a
    wildly wrong height where a much better solution existed nearby. Nothing in
    the ladder detects that, because a search cannot
    report a maximum it never visited. An exhaustive scan can.

    Every candidate is scored for every arc as ONE matrix product,

        gamma(a, c) = |sum_d Z[d, a] conj(E[d, c])| / n_valid

    with E the (dates x candidates) model bank, so it is a GEMM and batches on
    a GPU exactly like the arc kernel. On this machine a 101 x 81 grid over
    10000 arcs is seconds.

    UNLIKE the ladder this takes a RANGE, which is a prior. That is why it is a
    reference and not the estimator: the ladder's search window is set by the
    baselines alone. Use it to verify, to debug a suspect pixel, or in a test
    that asserts the ladder finds what is there.

    device: 'cuda' / 'mps' / 'cpu' / None to pick the best available.
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
    hh = np.asarray(ele2phase, dtype=np.float64) * meter2rad
    tt = np.asarray(t, dtype=np.float64) * meter2rad * 1e-3
    # (-k, k+1) * step rather than arange(-range, range, step): the latter
    # MISSES THE ORIGIN whenever the step does not divide the range -- 200 m
    # in 3 m steps runs .. -2, 1, 4 .. -- which biases every solution by up to
    # half a cell and silently removes the no-model candidate.
    kh = int(np.ceil(float(h_range) / float(h_step) - 1e-9))
    kv = int(np.ceil(float(v_range) / float(v_step) - 1e-9))
    gh = np.arange(-kh, kh + 1) * float(h_step)
    gv = np.arange(-kv, kv + 1) * float(v_step)
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


def _3d_deterministic_detrend(X, t, bperp=None, iters=60):
    """Remove the DETERMINISTIC part of each series, keep atmosphere and noise.

    The screen handed to the interpolation must keep atmosphere AND noise --
    they are both white in time and cannot be separated per pixel, only in
    space -- and must lose whatever is deterministic, because anything smooth
    left in the residual is deformation that a caller would subtract from the
    data.

    A smoothness prior cannot decide this here. Gaussian-process regression
    with a fitted noise ratio fails in exactly the regime this data occupies:
    on a noise-dominated series its own marginal likelihood concludes there is
    no smooth part, keeps only the constant, and passes the WHOLE deterministic
    signal into the screen. Node residuals ARE noise-dominated, so the screen
    it produces is the whole residual, deformation included.

    A fixed basis cannot be fooled that way: it removes its own degrees of
    freedom and no more, whatever the noise level or the cadence. Against
    planted signals what survives is the estimator's own coefficient noise
    rather than the signal, and nearly all of the white part is left intact.

    The basis is {1, t}: offset and rate, matching the (dv, dh) model the arcs
    are fitted with. The annual term is deliberately NOT here. Tropospheric
    delay was once assumed to be seasonal itself, which would have made a
    cos/sin term strip real atmosphere. Whether it actually is seasonal has to
    be tested on the stack in hand -- against a permuted-date null, and by
    looking for a spectral peak at one cycle per year -- not assumed. This
    function assumes the screen is white in time, and the cos/sin term is not
    removed here, so a seasonal component would survive to be seen.

    Including cos/sin costs held-out coherence: with only a small differential
    seasonal in the arcs, two extra free parameters are fitting noise, and a
    held-out test punishes exactly that. So the term stays out of THIS function
    -- but see max_seasonal on the arc fit, where a small allowance helps.

    What this does not remove is a quadratic or a transient: those are
    genuinely non-parametric, and catching them was the GP's justification,
    which it does not deliver.

    Fitted on the PHASORS by Gauss-Newton on Im(Z conj(model)), the component
    of the unit radius vector perpendicular to the model. It equals the phase
    error to first order and saturates instead of jumping a turn at +-pi, so
    nothing is ever wrapped or unwrapped. Because it saturates, it also
    converges SLOWLY when the series carries a lot of phase, and these do, so
    the step is scaled to a Newton one (see below). A screen with a residual
    rate biases every velocity it is ever removed from.

    THE BASELINE IS DETERMINISTIC TOO. Atmospheric delay is not correlated
    with the perpendicular baseline -- that is what distinguishes it from
    topography -- so any part of a residual proportional to ele2phase(d) is by
    definition NOT atmosphere and must not be carried into the screen. The
    per-node height came out with the network model, but a network solved on
    few arcs leaves a common height error behind, and a sparse stage's screen
    then spreads it over everything the interpolation reaches. Removing the
    baseline direction here costs nothing when there is none to remove.

    X     : (n_dates, n_series) complex residual phasors.
    bperp : (n_dates,) height-to-phase factor, or None to leave the baseline
            direction in (it is then only removed if it happens to be
            collinear with time).
    Returns the remainder, also as unit phasors.
    """
    X = np.asarray(X)
    if not np.iscomplexobj(X):
        raise TypeError(f'X must be complex phasors, got {X.dtype}')
    n = X.shape[0]
    if n < 6 or X.shape[1] == 0:
        return (X / np.maximum(np.abs(X), 1e-30)).astype(np.complex64)
    ts = np.asarray(t, dtype=np.float64)
    cols = [np.ones(n), ts - ts.mean()]
    if bperp is not None:
        bb = np.asarray(bperp, dtype=np.float64)
        bb = bb - bb.mean()
        if np.any(np.abs(bb) > 0):
            cols.append(bb / np.max(np.abs(bb)))
    B = np.stack(cols, axis=1)
    pinv = np.linalg.pinv(B)
    A = np.abs(X)
    Z = np.where(A > 0, X / np.where(A > 0, A, 1.0), 0).astype(np.complex128)
    ok = A > 0
    c = np.zeros((B.shape[1], Z.shape[1]))
    for _ in range(int(iters)):
        r = Z * np.exp(-1j * (B @ c))
        # A PLAIN step, deliberately. The update rides on sin(error), whose
        # slope is <cos(error)> ~ exp(-sigma^2/2), so it travels only part of
        # the way and needs many iterations -- and scaling by that slope to
        # make it a Newton step, which converges in a tenth of them on planted
        # series, is UNSTABLE here: the factor is 1/coherence, and a weak node
        # gets its step amplified enormously and overshoots. Iterating plainly
        # costs a negligible fraction of the solve.
        c = c + pinv @ np.where(ok, r.imag, 0.0)
    # The rate and the baseline term come out; the CONSTANT stays. The constant
    # is the node's base, and the network has already set it relative to every
    # other node -- removing it here would give each node an independent base
    # again and the screens would no longer average to a field.
    c[0] = 0.0
    out = Z * np.exp(-1j * (B @ c))
    out = np.where(ok, out, 0.0)
    return (out / np.maximum(np.abs(out), 1e-30)).astype(np.complex64)



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


def _3d_krige(vals, py, px, ny, nx, spacing, window, nlags=24):
    """The screen at every pixel: all nodes in the WINDOW, covariance-weighted.

    `vals` is (n_dates, n_nodes) COMPLEX unit phasors. Kriging the wrapped
    ANGLE instead is wrong and measurably so: two nodes whose phases differ by
    more than pi average to a value near neither, so the interpolant is worse
    than copying the nearest node. Phase is circular, so the components are
    interpolated and recombined; the variogram is built on the phase DIFFERENCE
    between nodes, which is well defined however the two are wrapped.

    THE NEIGHBOURHOOD IS THE WINDOW, not a fixed number of nodes. A count is a
    hidden distance: a fixed count of nearest nodes reaches only a few metres
    where nodes are dense, averaging many samples of ONE resolution element and
    cancelling no noise at all. `window` is already the spatial scale the caller
    asked for, and the atmosphere is correlated across that range: a single
    node's screen transfers very little on its own, flat with separation, so the
    screen exists but is buried in per-node noise and only appears once many
    nodes are averaged.

    That average is the whole point of the interpolation, so it must be over
    every node in the window rather than a truncated list. Taking all of them
    turns the estimator into a convolution, which is also what makes it
    affordable: an FFT per date instead of a linear solve per pixel.
    """
    from scipy.signal import fftconvolve
    n = vals.shape[0]
    sy, sx = float(spacing[0]), float(spacing[1])
    wy, wx = int(window[0]), int(window[1])
    py = np.asarray(py, dtype=np.int64)
    px = np.asarray(px, dtype=np.int64)
    P = np.c_[py.astype(float) * sy, px.astype(float) * sx]

    rng = np.random.default_rng(0)
    npair = int(min(200000, max(1000, len(P) * 40)))
    i = rng.integers(0, len(P), npair)
    j = rng.integers(0, len(P), npair)
    ok = i != j
    i, j = i[ok], j[ok]
    d = np.hypot(P[i, 0] - P[j, 0], P[i, 1] - P[j, 1])
    # THE VARIOGRAM WITHOUT TAKING AN ANGLE. For unit phasors the coherence of
    # a pair over the dates, R = |mean_d z_i conj(z_j)|, is exp(-sigma^2/2) for
    # the variance of their phase difference, so gamma = -ln R exactly. Only
    # the MAGNITUDE of a complex mean is used and nothing is ever wrapped.
    #
    # The squared-angle form this replaces was the last wrapping operation in
    # the whole path, and it saturates: an angle cannot exceed pi, so once the
    # screen varies by more than that the estimate stops growing and reports
    # structure that is not there: it inflates the nugget towards the sill,
    # and the fitted range then swings wildly on a modest change in the node
    # set -- which sets the kernel size, and so the coverage.
    R_ = np.abs(np.mean(vals[:, i] * np.conj(vals[:, j]), axis=0))
    semi = -np.log(np.clip(R_, 1e-6, 1.0))
    hi = np.percentile(d, 95)
    edges = np.linspace(0.0, max(hi, 1.0), nlags + 1)
    which = np.clip(np.digitize(d, edges) - 1, 0, nlags - 1)
    lagd = np.full(nlags, np.nan)
    lagg = np.full(nlags, np.nan)
    for k in range(nlags):
        m = which == k
        if m.any():
            lagd[k] = d[m].mean()
            lagg[k] = semi[m].mean()
    m = np.isfinite(lagd) & np.isfinite(lagg)
    if m.sum() < 3:
        sill = float(max(np.nanmean(semi), 1e-6))
        nug = 0.1 * sill
        rng_ = float(max(np.median(d), 1.0))
    else:
        sill = float(np.nanmax(lagg[m]))
        nug = float(np.clip(lagg[m][0], 0.0, 0.9 * sill))
        half = np.where(lagg[m] >= 0.5 * (sill + nug))[0]
        rng_ = float(lagd[m][half[0]]) if len(half) else float(np.median(d))
        rng_ = max(rng_, 1.0)

    # the kernel is the covariance over the window extent; the nugget stays out
    # of it, since it is the part of a node's value that does NOT transfer
    # THE WINDOW IS THE BOX, so the reach is HALF of it either way -- (32, 128)
    # px on an 8 x 2 m grid is 256 x 256 m of ground and the kernel runs to
    # +-128 m, the same convention _3d_arc_offsets uses. Spanning a full window
    # each way doubles the reach and pulls in nodes the variogram says carry
    # nothing: with a nugget-dominated variogram the far half of that kernel
    # adds weight without adding information.
    # THE REACH STOPS WHERE THE COVARIANCE DOES, not where the window does.
    # The kernel is exp(-3h/range), so at one range it is exp(-3) = 0.05 and
    # past that there is nothing left to weight with: several ranges out the
    # weight is many orders of magnitude below that. Interpolating from that is
    # not interpolation, it is round-off with a direction, and it shows: pixels
    # held by only a few nodes carry large steps between NEIGHBOURING pixels --
    # near reversals -- where densely held ones are smooth. The window stays
    # the outer bound, because it is what the
    # caller set; the range is what the data can support inside it.
    ry = min(wy // 2, max(1, int(np.ceil(rng_ / sy))))
    rx = min(wx // 2, max(1, int(np.ceil(rng_ / sx))))
    ky = np.arange(-ry, ry + 1)
    kx = np.arange(-rx, rx + 1)
    hh = np.hypot(np.outer(ky * sy, np.ones_like(kx)),
                  np.outer(np.ones_like(ky), kx * sx))
    # TAPERED TO ZERO AT ITS OWN EDGE. A kernel that stops abruptly makes the
    # screen jump: crossing the boundary adds or drops a node whose weight was
    # still a noticeable fraction of the peak, and where few nodes carry the
    # estimate that one is most of it. Without the taper, pixels with only a
    # few nodes in reach show large steps between NEIGHBOURING pixels.
    #
    # Subtracting the edge value makes the weight reach zero continuously, so a
    # node entering the neighbourhood arrives with no weight at all and nothing
    # can step. The support is unchanged in the interior, where the estimate was
    # never in doubt.
    R = min(rng_, float(np.hypot(ry * sy, rx * sx)))
    K = ((sill - nug) * np.maximum(np.exp(-3.0 * hh / rng_)
                                   - np.exp(-3.0 * R / rng_), 0.0)
         ).astype(np.float32)

    field = np.zeros((ny, nx), dtype=np.complex64)
    mask = np.zeros((ny, nx), dtype=np.float32)
    mask[py, px] = 1.0
    # COVERAGE IS A BOX TEST, NOT A THRESHOLD ON THE TRANSFORM. A pixel is
    # reached exactly when a node lies inside the window centred on it, and
    # maximum_filter answers that with no arithmetic to go wrong.
    #
    # It used to ask `fftconvolve(mask, K) > 1e-12`, which is a test orders of
    # magnitude below the noise of the thing being tested: the convolution
    # carries float32 round-off far outside the kernel's support, including
    # NEGATIVE values, which a sum of positive weights cannot produce. Every
    # one of them passed. The screen was therefore extrapolated across the entire block
    # however small the window: from a cluster spanning 8 rows it covered all
    # 400, a reach of 198 rows where 128 was the most the window allows.
    from scipy.ndimage import maximum_filter
    good = maximum_filter(mask, size=K.shape, mode='constant', cval=0.0) > 0
    den = fftconvolve(mask, K, mode='same')
    # the taper zeroes the kernel outside its disc, so the box above is now
    # generous: a pixel in the corner of the box but beyond the disc collects no
    # weight at all. Its den is pure transform round-off, so anything a
    # millionth of ONE node's peak weight is real support and anything below it
    # is noise wearing a direction.
    good &= den > 1e-6 * float(K.max())
    # ACCURACY IS A PROPERTY OF EACH PIXEL, NOT OF A CROP. A convex hull was
    # tried and is wrong: it is a polygon, so it discards a pixel lying metres
    # from a node merely because that node happens to sit on the boundary,
    # while admitting anything inside however thin the support there is.
    #
    # The estimate's own variance says it per pixel and costs one convolution.
    # With weights w = K/den, simple kriging leaves
    #     sigma^2(x) = C(0) - sum_i w_i C(x - x_i) = K(0) - den2/den
    # which falls to zero where the pixel sits on a node and rises to K(0)
    # where the weights are spread over nodes too far to say anything. It is
    # geometry alone -- no value enters it -- so it is the same number whatever
    # the atmosphere did that day.
    den2 = fftconvolve(mask, (K ** 2).astype(np.float32), mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        var = float(K.max()) - den2 / np.maximum(den, 1e-30)
    _3d_krige.var = np.where(good, var / max(float(K.max()), 1e-30), np.nan)
    # AND IT MUST BE AN INTERPOLATION. Kish's effective sample size,
    # (sum K)^2 / sum K^2, is 1 when one node carries the estimate however many
    # are nominally in reach, and rises only as the weight genuinely spreads.
    # Below 2 the screen is one node copied outward, which is a nearest-node
    # assignment wearing the name of an interpolation -- and it is where the
    # field tears: even with the taper in place, pixels at low n_eff carry
    # large steps between neighbours where high-n_eff ones are smooth.
    #
    # 2 is structural rather than chosen: it is the point at which a second
    # node contributes at all. Stricter cuts keep buying smoothness, trading
    # coverage away as they go -- but those are preferences about how much
    # ground to trade for how much
    # quiet, and the caller controls that already through `window`.
    den2 = fftconvolve(mask, (K ** 2).astype(np.float32), mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        neff = np.where(den2 > 0, den ** 2 / np.maximum(den2, 1e-30), 0.0)
    good &= neff >= 2.0
    out = np.empty((n, ny, nx), dtype=np.complex64)
    for d_ in range(n):
        field[:] = 0
        field[py, px] = vals[d_]
        num = (fftconvolve(field.real, K, mode='same')
               + 1j * fftconvolve(field.imag, K, mode='same'))
        with np.errstate(invalid='ignore', divide='ignore'):
            z = np.where(good, num, np.nan + 1j * np.nan)
            # A ZERO NUMERATOR IS NOT A SCREEN. den > 0 only says some node lies
            # under the kernel; the numerator is a signed sum and can cancel, or
            # underflow in float32, at the fringe where the weights are tiny.
            # Dividing it by its own magnitude then yields 0 + 0j, which is not
            # a unit phasor and would ZERO the data it multiplies rather than
            # rotate it. Before the guard this hit a scattering of samples,
            # always a few dates at a pixel and never all of them, so nothing
            # downstream would have flagged it.
            mag = np.abs(z)
            out[d_] = np.where(mag > 0, z / np.maximum(mag, 1e-30),
                               np.nan + 1j * np.nan).astype(np.complex64)
    return out



def _3d_fit_ps_array(scenes, date_values, *, spacing, bperp=None,
                       window=(32, 128), threshold=0.5, cell=(2, 8),
                       geometry, budget=None, densify=True,
                       max_dh=100.0, max_dv=50.0, step_dh=4.0, step_dv=2.0,
                       max_seasonal=5.0,
                       consensus, device='cpu', iterations=8, debug=False):
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

    # HOW MUCH AGREEMENT IS REQUIRED, in one argument for both halves of the
    # solve, because it is one question asked twice: an arc must agree with the
    # network and a partner must agree with the other partners.
    #
    _ma, _ar, _ii = _3d_consensus(consensus)

    S = np.ascontiguousarray(scenes, dtype=np.complex64)
    n, ny, nx = S.shape
    wy, wx, pey, pex = _3d_windows(window)
    # int8 states the intent: a scene has a handful of components, not
    # thousands, and a label that needs more than a byte means the network has
    # shattered rather than resolved. -1 is nodata, so 0..127 are available.
    lab_out = np.full((ny, nx), -1, dtype=np.int8)
    # VELOCITY AND HEIGHT AS RASTERS, not only in `.stats`. The stats dict is a
    # function attribute set by whichever block ran last in the worker, so under
    # dask it describes one chunk and silently misdescribes the rest -- it can
    # report 200 nodes for a raster carrying 29118. A caller cannot reproduce
    # the product from it, which means the product is not really returned.
    vel_out = np.full((ny, nx), np.nan, dtype=np.float32)
    hgt_out = np.full((ny, nx), np.nan, dtype=np.float32)
    # COHERENCE of the arc that ties the pixel to the network: for a node the
    # mean over its own kept arcs, for an attached DS the arc it came in on.
    # rmse = sqrt(-2 ln gamma) is derived from it upstairs rather than carried
    # as a second plane, since the two are exact inverses.
    coh_out = np.full((ny, nx), np.nan, dtype=np.float32)
    # the fitted annual, complex, in RADIANS -- part of the model and otherwise
    # unreachable from the public API
    sea_out = np.full((ny, nx), np.nan + 1j * np.nan, dtype=np.complex64)
    # Cleared FIRST, because several guards below return early and a stats dict
    # left over from the previous call describes a different solve: without
    # this, a run that produced nothing still reported components from the
    # previous one, which is worse than reporting none.
    _3d_fit_ps_array.stats = dict(nodes=0, arcs=0, dropped=0, components=[],
                                    fill_order=[])
    if n < 2 or ny == 0 or nx == 0:
        return lab_out, vel_out, hgt_out, sea_out, coh_out
    sy, sx = float(spacing[0]), float(spacing[1])
    if not (sy > 0 and sx > 0):
        raise ValueError(f'spacing must be positive, got {spacing}')
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

    # ---- the nodes: PS, not DS -----------------------------------------
    q = _3d_arcs_kernel(S, wy, wx, tuple(cell), budget)
    ps = _3d_ps_kernel(S, (wy, wx, pey, pex), q, ele2phase, t, meter2rad,
                       threshold=float(threshold), budget=budget,
                       device=device, iterations=iterations)
    iy, ix = np.where(np.isfinite(ps) & (ps >= float(threshold)))
    if debug:
        _cand = int(np.count_nonzero(np.isfinite(q) & (q >= float(threshold))))
        print(f'DEBUG: PS test  {len(iy)} nodes at >= {float(threshold)}'
              f'  ({_cand} DS candidates in the same raster)', flush=True)
    if len(iy) < 2:
        if debug:
            print('DEBUG: fewer than 2 nodes -- nothing to solve', flush=True)
        return lab_out, vel_out, hgt_out, sea_out, coh_out
    a = np.abs(S[:, iy, ix])
    with np.errstate(invalid='ignore', divide='ignore'):
        Un = np.ascontiguousarray(
            np.where(a > 0, S[:, iy, ix] / np.where(a > 0, a, 1), 0
                     ).astype(np.complex64))
    del a

    # ---- the arcs: every pair inside the PS window ----------------------
    # scaled so the window becomes the unit box, then a Chebyshev query is
    # exactly "inside the window" and costs O(N k) rather than O(N^2)
    hy, hx = max(pey // 2, 1), max(pex // 2, 1)
    tree = cKDTree(np.c_[iy / hy, ix / hx])
    pairs = tree.query_pairs(1.0, p=np.inf, output_type='ndarray')
    if len(pairs) < 3:
        return lab_out, vel_out, hgt_out, sea_out, coh_out
    ai, aj = pairs[:, 0], pairs[:, 1]
    step = max(1, int(_3d_budget_mb(budget) * 1024 * 1024 // max(n * 16, 1)))
    g = np.empty(len(ai), np.float32)
    dh = np.empty(len(ai)); dv = np.empty(len(ai))
    ds_ = np.empty(len(ai), np.complex128)
    for b0 in range(0, len(ai), step):
        s_ = slice(b0, min(b0 + step, len(ai)))
        arc = np.ascontiguousarray(
            (Un[:, ai[s_]] * np.conj(Un[:, aj[s_]])).astype(np.complex64))
        g[s_], dh[s_], dv[s_], ds_[s_] = _3d_arc_fit(
            arc, ele2phase, t, meter2rad, max_dh, max_dv, step_dh, step_dv,
            budget, max_seasonal, device, iterations=iterations)
    keep = np.isfinite(g) & (g >= float(threshold))
    if debug:
        _gk = g[keep]
        print(f'DEBUG: arcs     {len(ai):,} pairs fitted, {int(keep.sum()):,} '
              f'>= {float(threshold)}  ({100 * keep.mean():.1f}%)', flush=True)
        if keep.any():
            print(f'DEBUG:          arc gamma p50 {np.median(_gk):.3f}  '
                  f'p90 {np.percentile(_gk, 90):.3f}  max {_gk.max():.3f}',
                  flush=True)
    if keep.sum() < 3:
        if debug:
            print('DEBUG: fewer than 3 arcs cleared the threshold', flush=True)
        return lab_out, vel_out, hgt_out, sea_out, coh_out
    ai, aj, dh, dv, ds_ = ai[keep], aj[keep], dh[keep], dv[keep], ds_[keep]
    gk = g[keep]

    # ---- EVERY ARC THE FIT CERTIFIED ENTERS THE NETWORK -----------------
    # No cap. An arc that cleared `threshold` is a measurement, and the only
    # reason to refuse one would be cost -- which there is none of: the pairs
    # are ALREADY FITTED above, before anything could select among them, so
    # discarding some afterwards saves no fitting at all. It only shrinks the
    # least-squares system, and that system is thousands of rows against the
    # millions of arcs the attachment fits.
    #
    # What a cap did cost was connectivity. Taking a fixed number of arcs per
    # node best-first spends a node's budget on its closest, most coherent
    # neighbours -- which are the arcs most likely to be redundant with each
    # other -- and the long arcs that tie distant groups together are exactly
    # the ones it drops. The network then falls into pieces, each piece gets
    # its own free datum, and pixels near the seam are lost twice over: their
    # velocities are no longer comparable, and a DS drawing partners from both
    # sides sees them disagree by the datum offset and is rejected.
    #
    # Redundancy is still what the screen is made of; it is simply taken
    # rather than rationed, and the robust pass below decides which arcs the
    # network believes.
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

    def _mad(r):
        return 1.4826 * float(np.median(np.abs(r - np.median(r))))

    def _node_sigma(res, a_, b_, nnodes, floor, min_n):
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
    for _ in range(_ii if _ar is not None else 0):
        r_h = G @ _wsolve(G, dh, w_) - dh
        r_v = G @ _wsolve(G, dv, w_) - dv
        # per NODE, not pooled: judged against its own arcs' scatter
        f_h, f_v = _SIGMA_FLOOR * _mad(r_h), _SIGMA_FLOOR * _mad(r_v)
        s_h = _node_sigma(r_h, ai, aj, N, max(f_h, 1e-12), _ma or 1)
        s_v = _node_sigma(r_v, ai, aj, N, max(f_v, 1e-12), _ma or 1)
        # an arc must hold up as seen from BOTH of its ends
        z = np.maximum(np.abs(r_h) / np.minimum(s_h[ai], s_h[aj]),
                       np.abs(r_v) / np.minimum(s_v[ai], s_v[aj]))
        w_ = gtake / np.maximum(z, 1.0)
    # scale from the ROBUST fit, so the outliers do not set their own bar
    if _ar is None:
        keep_arc = np.ones(len(ai), dtype=bool)
    else:
        f_h, f_v = _SIGMA_FLOOR * _mad(r_h), _SIGMA_FLOOR * _mad(r_v)
        s_h = _node_sigma(r_h, ai, aj, N, max(f_h, 1e-12), _ma or 1)
        s_v = _node_sigma(r_v, ai, aj, N, max(f_v, 1e-12), _ma or 1)
        keep_arc = np.maximum(
            np.abs(r_h) / np.minimum(s_h[ai], s_h[aj]),
            np.abs(r_v) / np.minimum(s_v[ai], s_v[aj])) <= _ar
    if keep_arc.sum() < 3:
        return lab_out, vel_out, hgt_out, sea_out, coh_out
    rejected = int((~keep_arc).sum())
    if debug:
        print(f'DEBUG: IRLS     {rejected:,} of {len(ai):,} arcs rejected '
              f'beyond {_ar} sigma  ({100 * rejected / max(len(ai), 1):.1f}%)',
              flush=True)
    # how much of each node's own support the rejection took away
    drej = np.bincount(np.r_[ai[~keep_arc], aj[~keep_arc]], minlength=N)
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
    # Whether a pixel IS a PS is settled by the PS test; how many network arcs
    # it happens to receive depends only on how well connected its neighbours
    # are. Requiring more than one here let that decide how many PS exist.
    live = dcount >= 1
    if debug:
        # NODES LOST BEFORE ANY COMPONENT EXISTS. A node whose every arc was
        # rejected holds no datum and cannot be reported. Counted separately
        # from the component floor below: the two are different losses at
        # different stages, and a single "kept" total hides which is which.
        print(f'DEBUG: solve     {int((~live).sum())} of {len(live)} nodes '
              f'left with no surviving arc', flush=True)
    if not live.any():
        return lab_out, vel_out, hgt_out, sea_out, coh_out

    # ---- integrate, one free datum per component -----------------------
    # EXACT on the arcs that survived, weighted by their own coherence. The
    # robust pass above decided WHICH arcs; it does not get to blur the ones
    # it kept.
    m_ = len(ai)
    G = _incidence(ai, aj, m_)
    hgt = _wsolve(G, dh, gtake)
    vel = _wsolve(G, dv, gtake)
    anr = _wsolve(G, ds_.real, gtake)
    ani = _wsolve(G, ds_.imag, gtake)
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
        return lab_out, vel_out, hgt_out, sea_out, coh_out
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
        lab_out[sy_[k], sx_[k]] = label_of[z]
        vel_out[sy_[k], sx_[k]] = vel[sel][k].astype(np.float32)   # rad/yr
        hgt_out[sy_[k], sx_[k]] = hgt[sel][k].astype(np.float32)   # rad
        coh_out[sy_[k], sx_[k]] = gnode[sel][k].astype(np.float32)
        sea_out[sy_[k], sx_[k]] = (anr[sel][k]
                                   + 1j * ani[sel][k]).astype(np.complex64)
        keep.append(k)
    kk = np.concatenate(keep)
    lab_all = np.array([label_of[z] for z in order_prio
                        for _ in range(len(comps[z][2]))], dtype=np.int8)
    k_mm = meter2rad * 1e-3

    _3d_fit_ps_array.stats = dict(
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
              f'{int(dropped)} component(s) dropped past the int8 label limit',
              flush=True)
        print(f'DEBUG:          arcs/node p50 {np.median(_dn):.0f}  '
              f'min {_dn.min()}  max {_dn.max()}', flush=True)
        if len(_gn):
            print(f'DEBUG:          node gamma p50 {np.median(_gn):.3f}  '
                  f'p10 {np.percentile(_gn, 10):.3f}', flush=True)

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
    if densify:
        cand_ds = np.isfinite(q) & (q >= float(threshold))
        cand_ds[iy[sel][kk], ix[sel][kk]] = False        # nodes are not DS here
        dy_, dx_ = np.where(cand_ds)
        if len(dy_):
            ny_ps, nx_ps = iy[sel][kk], ix[sel][kk]
            hy_d, hx_d = max(pey // 2, 1), max(pex // 2, 1)
            tre = cKDTree(np.c_[ny_ps / hy_d, nx_ps / hx_d])
            near = tre.query_ball_point(np.c_[dy_ / hy_d, dx_ / hx_d],
                                        1.0, p=np.inf)
            src = np.repeat(np.arange(len(dy_)),
                            [len(v) for v in near]).astype(np.int64)
            tgt = (np.concatenate(near).astype(np.int64) if len(src)
                   else np.zeros(0, np.int64))
            _3d_fit_ps_array.stats['ds_candidates'] = int(len(dy_))
            _3d_fit_ps_array.stats['ds_reached'] = int(len(np.unique(src)))
            _3d_fit_ps_array.stats['ds_arcs'] = int(len(src))
            if len(src):
                ga = np.empty(len(src), np.float32)
                dha = np.empty(len(src)); dva = np.empty(len(src))
                dsa = np.empty(len(src), np.complex128)
                step2 = max(1, int(_3d_budget_mb(budget) * 1024 * 1024
                                   // max(n * 16, 1)))
                for b0 in range(0, len(src), step2):
                    sl = slice(b0, min(b0 + step2, len(src)))
                    ad = np.abs(S[:, dy_[src[sl]], dx_[src[sl]]])
                    ud = np.where(ad > 0, S[:, dy_[src[sl]], dx_[src[sl]]]
                                  / np.where(ad > 0, ad, 1), 0)
                    arc2 = np.ascontiguousarray(
                        (ud * np.conj(Un[:, sel][:, kk][:, tgt[sl]])
                         ).astype(np.complex64))
                    # NO DIFFERENTIAL ANNUAL ON THE ATTACHING ARC. The DS
                    # sits inside its partner's own DS window, tens of metres
                    # away, and there is no seasonal GRADIENT at that scale:
                    # stratified and thermal delay vary with elevation and over
                    # kilometres, not across a courtyard. So the differential
                    # annual is physically ~0 and fitting it is fitting noise
                    # -- two free parameters on a marginal arc buy enough
                    # coherence to carry the rate a whole sideband away, and
                    # attachments then alias at many times the rate they do
                    # without it.
                    #
                    # The DS still GETS an annual -- it inherits its partner's,
                    # which was fitted on the network arcs where the baseline is
                    # long enough for a real differential to exist. That is the
                    # physically right split: the network measures the seasonal,
                    # the attachment transfers it.
                    # RANK FIRST, REFINE THE SHORTLIST. Only `min_agreeing`
                    # partners are used, so refining all of them spends the
                    # larger half of the fit on candidates that are discarded a
                    # few lines below. Ranking needs the model only well
                    # enough to order it, and the refinement cannot leave its
                    # own lattice cell, so it reorders neighbours at most.
                    #
                    # ONE PASS, AND THAT IS A CORRECTNESS FLOOR RATHER THAN
                    # A SETTING. The lattice scores every candidate at a
                    # QUANTISED model, so what is compared is not each
                    # candidate's own fit: neighbours can share a grid point
                    # and tie exactly, and even apart they are scored at
                    # whatever the grid rounded them to. One pass takes each
                    # off the grid and onto its own optimum, which is what
                    # makes the comparison mean anything -- it removes the
                    # lattice artefact, no more. Further passes converge inside
                    # a cell whose ordering is already settled, so they cannot
                    # change which candidates are chosen.
                    ga[sl], dha[sl], dva[sl], dsa[sl] = _3d_arc_fit(
                        arc2, ele2phase, t, meter2rad, max_dh, max_dv, step_dh, step_dv,
                        budget, 0.0, device, iterations=1)
                # ONE COMPONENT PER DS. Every component carries its own
                # free datum, so partners drawn from two of them disagree by
                # that offset however good each arc is. A DS at a seam would
                # then see its best partners contradict one another and be
                # rejected for a disagreement that is bookkeeping rather than
                # measurement -- and a seam is precisely where the network is
                # thinnest and the extra reach matters most. So the candidates
                # are cut to a single component first and the consensus is
                # sought inside it, which is what makes the surviving votes
                # comparable at all.
                #
                # WHICH component is decided by the quantity the answer will
                # rest on: the coherence of its `min_agreeing`-th best arc,
                # the weakest partner the consensus would be built from. One
                # that cannot field that many cannot answer, and is not asked.
                # With a single component -- what an uncapped network gives on
                # connected ground -- none of this runs.
                if _ma is not None and len(comps) > 1:
                    # THE BEST PARTNER NAMES THE COMPONENT, AND THEN ONLY ITS
                    # NODES ARE USED. A DS reaches every node inside its
                    # extent regardless of which component that node belongs
                    # to; the single most coherent arc says which network this
                    # pixel belongs to, and the consensus is then sought among
                    # that component's nodes alone.
                    #
                    # It is the ONE criterion available before any value is
                    # read. Scoring a component by whether it can field
                    # `min_agreeing` good partners -- what this did first --
                    # picks the component most likely to PASS, which is
                    # choosing to suit the answer: a DS whose best arc by far
                    # sits in a small component would be handed to a larger
                    # one it agrees with less, purely because the larger one
                    # could fill the shortlist. Arc coherence is settled
                    # before any velocity is looked at, so the best arc cannot
                    # be picked to produce a result.
                    #
                    # If the named component then cannot field `min_agreeing`
                    # partners, or they disagree, the DS is unmeasured. That
                    # is the honest outcome -- not a reason to go back and
                    # choose a different network.
                    #
                    # Spatial position plays no part. Components are not
                    # regions; an island and a far river bank interleave with
                    # the main network across the raster.
                    _lab = lab_all[tgt].astype(np.int64)
                    _gf = np.where(np.isfinite(ga), ga, -np.inf)
                    _top1 = np.full(len(dy_), -np.inf)
                    np.maximum.at(_top1, src, _gf)
                    _is1 = (_gf >= _top1[src]) & (_gf > -np.inf)
                    _win = np.full(len(dy_), -1, dtype=np.int64)
                    _win[src[_is1]] = _lab[_is1]
                    if debug:
                        # WHAT THE RESTRICTION FORBIDS, counted before it
                        # acts: shortlists that would have spanned two
                        # components. Not "saw more than one component" --
                        # nearly every DS does once a second one exists in
                        # reach. Only a straddling SHORTLIST would have
                        # differenced partners against unrelated datums.
                        _so = np.lexsort((-_gf, src))
                        _sn = np.bincount(src[_so], minlength=len(dy_))
                        _sf = np.r_[0, np.cumsum(_sn)[:-1]]
                        _tp = _so[(np.arange(len(_so))
                                   - np.repeat(_sf, _sn)) < _ma]
                        _lo = np.full(len(dy_), np.iinfo(np.int64).max, np.int64)
                        _hi = np.full(len(dy_), -1, dtype=np.int64)
                        np.minimum.at(_lo, src[_tp], _lab[_tp])
                        np.maximum.at(_hi, src[_tp], _lab[_tp])
                        _3d_fit_ps_array.stats['ds_shortlist_straddled'] = int(
                            np.count_nonzero((_hi >= 0) & (_hi != _lo)))
                        del _so, _sn, _sf, _tp, _lo, _hi
                    _3d_fit_ps_array.stats['ds_multi_component'] = int(
                        np.count_nonzero(
                            np.bincount(src[_lab != _win[src]],
                                        minlength=len(dy_)) > 0))
                    ga[_lab != _win[src]] = np.nan
                    del _lab, _gf, _top1, _is1, _win

                # REFINE EXACTLY WHAT WILL BE USED. `consensus` says how many
                # partners the answer rests on, so that is how many are worth
                # refining -- no separate width to choose or keep in step. The
                # lattice ranking is what picks them, and it can do that because
                # the refinement never leaves its own lattice cell: it reorders
                # neighbours rather than moving a candidate across the field.
                _short = int(_ma) if _ma is not None else len(src)
                _ord = np.lexsort((-np.where(np.isfinite(ga), ga, -np.inf), src))
                _cnt0 = np.bincount(src[_ord], minlength=len(dy_))
                _off0 = np.r_[0, np.cumsum(_cnt0)[:-1]]
                _col0 = np.arange(len(_ord)) - np.repeat(_off0, _cnt0)
                # ADMISSIBLE ONLY. Ranking puts NaN last, but taking the
                # first `_short` POSITIONS still reaches them when a DS has
                # fewer admissible candidates than that -- and refitting one
                # hands it a fresh finite coherence, so an arc ruled out
                # before the ranking would come back holding a vote. That is
                # how a DS assigned to one component could still be scored
                # against another's nodes, which is the whole thing the
                # restriction above is for. A DS left with too few is left
                # with too few; the consensus rejects it for that.
                _keep = _ord[(_col0 < _short) & np.isfinite(ga[_ord])]
                if len(_keep):
                    ad = np.abs(S[:, dy_[src[_keep]], dx_[src[_keep]]])
                    ud = np.where(ad > 0, S[:, dy_[src[_keep]], dx_[src[_keep]]]
                                  / np.where(ad > 0, ad, 1), 0)
                    arc3 = np.ascontiguousarray(
                        (ud * np.conj(Un[:, sel][:, kk][:, tgt[_keep]])
                         ).astype(np.complex64))
                    ga[_keep], dha[_keep], dva[_keep], dsa[_keep] = _3d_arc_fit(
                        arc3, ele2phase, t, meter2rad, max_dh, max_dv,
                        step_dh, step_dv, budget, 0.0, device,
                        iterations=iterations)
                    del arc3, ud, ad
                # anything not refined cannot be used, whatever its lattice value
                _unref = np.ones(len(src), dtype=bool)
                _unref[_keep] = False
                ga[_unref] = np.nan
                good = np.isfinite(ga) & (ga >= float(threshold)) & np.isfinite(dha)
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
                v_abs = vel[sel][kk][tgt] + dva
                o2 = np.lexsort((-np.where(good, ga, -np.inf), src))
                o2 = o2[good[o2]]
                nds = len(dy_)
                cnt = np.bincount(src[o2], minlength=nds)
                if cnt.max() if len(cnt) else 0:
                    off = np.r_[0, np.cumsum(cnt)[:-1]]
                    col = np.arange(len(o2)) - np.repeat(off, cnt)
                    # ONLY THE BEST `_ma` PARTNERS ENTER, AND NOTHING ELSE DOES.
                    # At the PS extent a DS has tens of candidates spanning
                    # every quality from just above `threshold` upwards, and a
                    # median across that mixture estimates nothing: it is a
                    # summary of several populations, so it neither describes
                    # the good arcs nor the bad ones. Robustness is not what
                    # makes it meaningless -- the mixture is.
                    #
                    # So the partners are CHOSEN first, by arc coherence, which
                    # is settled before any value is read and so cannot be
                    # picked to suit the answer. Everything downstream -- the
                    # centre, the scale, the rejection -- then sees one
                    # homogeneous set of comparable measurements, which is the
                    # only situation where a robust scale means anything.
                    #
                    # Columns are gamma-descending, so the first `_ma` ARE the
                    # best `_ma`; a row with fewer admissible arcs cannot fill
                    # them and is rejected for having too few.
                    if _ma is not None:
                        take = col < _ma
                        o2, col = o2[take], col[take]
                        cnt = np.minimum(cnt, _ma)
                    kmax = int(cnt.max())
                    row = src[o2]
                    V = np.full((nds, kmax), np.nan)
                    G = np.zeros((nds, kmax))
                    IDX = np.full((nds, kmax), -1, dtype=np.int64)
                    V[row, col] = v_abs[o2]
                    G[row, col] = ga[o2]
                    IDX[row, col] = o2
                    fin = np.isfinite(V)
                    # rows for a DS with no admissible arc are all NaN and are
                    # rejected below; taking a median of one would only warn
                    anyf = fin.any(axis=1)
                    e = np.zeros(nds)
                    if anyf.any():
                        e[anyf] = np.nanmedian(
                            np.where(fin, V, np.nan)[anyf], axis=1)
                    d0 = np.abs(V - e[:, None])
                    floor = max(_SIGMA_FLOOR * float(np.nanmedian(d0[fin])), 1e-9) \
                        if fin.any() else 1e-9
                    for _ in range(_ii):
                        w = np.where(fin, G / np.maximum(np.abs(V - e[:, None]),
                                                         floor), 0.0)
                        sw = w.sum(axis=1)
                        e = np.where(sw > 0, (w * np.where(fin, V, 0.0)).sum(axis=1)
                                     / np.maximum(sw, 1e-30), e)
                    r = np.abs(V - e[:, None])
                    # PER DS, from its own partners: they are what say how much
                    # this pixel's measurements scatter. The floor keeps a row
                    # whose partners happen to land identically from rejecting
                    # everything else on a zero scale.
                    sig = np.full(nds, floor)
                    if fin.any():
                        rr = np.where(fin, r, np.nan)
                        enough = fin.sum(axis=1) >= (_ma if _ma is not None else 1)
                        if enough.any():
                            sig[enough] = np.maximum(
                                1.4826 * np.nanmedian(rr[enough], axis=1), floor)
                    keep = (fin & (r <= _ar * sig[:, None])
                            if _ar is not None else fin)
                    nkeep = keep.sum(axis=1)
                    # THE BEST `_ma` PARTNERS MUST AGREE -- not any `_ma` of
                    # them. At the PS extent a DS sees tens of candidates, and
                    # "some three of them agree" is close to vacuous: the
                    # weighted median finds the densest cluster of whatever is
                    # there, and any unimodal scatter has three near its mode.
                    # The test would then pass on the SHAPE of the sample
                    # rather than on the quality of the pixel, and raising the
                    # count does not repair it -- it asks the same weak
                    # question of a bigger subset.
                    #
                    # Naming WHICH partners have to agree is what makes it a
                    # test. They are chosen by arc coherence, which is settled
                    # before any value is looked at, so the choice cannot be
                    # made to suit the answer. Columns are gamma-descending, so
                    # the first `_ma` of them ARE the best `_ma`, and a row
                    # holding fewer than `_ma` admissible arcs cannot fill them.
                    #
                    # It also stops the mixing of partners of very different
                    # quality: a strong arc and a barely-admitted one carried
                    # equal standing in a count, and the count was what decided.
                    if _ma is not None:
                        # ALL `min_agreeing` OF THEM, UNANIMOUSLY. The
                        # selection names WHICH partners are asked -- that is
                        # what stops the test being "some few of many" and
                        # therefore vacuous -- and every one of them has to
                        # agree.
                        #
                        # Allowing a single dissenter is not the mild
                        # relaxation it reads as. There are only `_ma` columns,
                        # so a DS holding `_ma - 1` admissible partners fills
                        # every column it has and passes: the tolerance meant
                        # for one partner DISAGREEING is spent instead on one
                        # partner being ABSENT, and a pixel that never had the
                        # asked-for redundancy is reported as though it did.
                        # Those two failures are not interchangeable -- one is
                        # a measurement that was contradicted, the other is a
                        # measurement that was never made.
                        #
                        # Requiring the full count keeps them apart. A row with
                        # too few partners cannot reach `_ma` and is rejected
                        # for being too few, which is what it is.
                        okds = keep[:, :_ma].sum(axis=1) == _ma
                    else:
                        okds = nkeep >= 1
                    # rows are gamma-descending, so the first survivor is the
                    # best arc of the consistent set
                    sel_col = np.argmax(keep, axis=1)
                    first = IDX[np.flatnonzero(okds), sel_col[okds]]
                    if debug:
                        # THE INVARIANT, CHECKED RATHER THAN ARGUED. The
                        # SHORTLIST behind an attached DS -- the partners
                        # whose agreement was tested -- must lie in ONE
                        # component. Components carry unrelated datums, so a
                        # decision taken across two is a numerical error, not
                        # a noisier answer. Must read 0 on any scene with any
                        # parameters; it is an assertion, not a quality
                        # measure. It exists because the restriction HAS been
                        # reversed downstream once already, silently, by a
                        # shortlist that selected by position and let nulled
                        # arcs be refitted back into votes.
                        #
                        # On `fin`, NOT on `keep`. A cross-component partner
                        # differs by the datum offset, which is precisely what
                        # makes it an outlier, so the sigma rejection discards
                        # it before `keep` exists -- a check there asks a
                        # question the preceding code already answered, and
                        # reads 0 even with the restriction switched off.
                        _rw = np.flatnonzero(okds)
                        _ix2 = IDX[_rw]
                        _lb2 = np.where(fin[_rw] & (_ix2 >= 0),
                                        lab_all[tgt[np.clip(_ix2, 0, None)]
                                                ].astype(np.int64), -1)
                        _hi2 = _lb2.max(axis=1)
                        _lo2 = np.where(_lb2 >= 0, _lb2,
                                        np.iinfo(np.int64).max).min(axis=1)
                        _3d_fit_ps_array.stats['ds_cross_component_votes'] = int(
                            np.count_nonzero((_hi2 >= 0) & (_hi2 != _lo2)))
                        del _rw, _ix2, _lb2, _hi2, _lo2
                    votes = nkeep[okds].astype(np.int32)
                    reach = int(np.count_nonzero(cnt))
                    # DS holding at least one admissible arc -- the population
                    # the consensus is actually asked about. `ds_no_consensus`
                    # is the whole shortfall from it and CONTAINS
                    # `ds_too_few`; the two are nested, not side by side.
                    _3d_fit_ps_array.stats['ds_admissible'] = reach
                    _3d_fit_ps_array.stats['ds_no_consensus'] = reach - len(first)
                    _3d_fit_ps_array.stats['ds_too_few'] = int(
                        reach - np.count_nonzero(
                            fin[:, :_ma].sum(axis=1) >= _ma)
                        if _ma is not None else 0)
                else:
                    first = np.zeros(0, dtype=np.int64)
                    votes = np.zeros(0, dtype=np.int32)
                if len(first):
                    ds_i, ps_i = src[first], tgt[first]
                    yy2, xx2 = dy_[ds_i], dx_[ds_i]
                    h_ds = hgt[sel][kk][ps_i] + dha[first]
                    lab_ds = lab_all[ps_i]
                    lab_out[yy2, xx2] = lab_ds
                    vel_out[yy2, xx2] = (vel[sel][kk][ps_i]
                                         + dva[first]).astype(np.float32)
                    hgt_out[yy2, xx2] = h_ds.astype(np.float32)
                    coh_out[yy2, xx2] = ga[first].astype(np.float32)
                    sea_out[yy2, xx2] = ((anr[sel][kk][ps_i] + dsa[first].real)
                                         + 1j * (ani[sel][kk][ps_i]
                                                 + dsa[first].imag)
                                         ).astype(np.complex64)
                    _3d_fit_ps_array.stats.update(
                        ds_attached=int(len(first)),
                        ds_partners=np.bincount(
                            src[good], minlength=len(dy_))[src[first]],
                        ds_gamma=ga[first].copy(),
                        ds_votes=np.asarray(votes, dtype=np.int32),
                        ds_iy=yy2.copy(), ds_ix=xx2.copy(), ds_label=lab_ds,
                        ds_height_rad=h_ds.astype(np.float32),
                        ds_velocity_rad_yr=(vel[sel][kk][ps_i]
                                            + dva[first]).astype(np.float32),
                        # the annual travels with the DS as well: it is fitted
                        # on the attaching arc and made absolute by its
                        # partner's own value, exactly as height and rate are
                        ds_seasonal_rad=((anr[sel][kk][ps_i] + dsa[first].real)
                                         + 1j * (ani[sel][kk][ps_i]
                                                 + dsa[first].imag)
                                         ).astype(np.complex64))
    if debug:
        _s = _3d_fit_ps_array.stats
        _att = int(_s.get('ds_attached', 0))
        _cnd = int(_s.get('ds_candidates', 0))
        if not densify:
            print('DEBUG: DS       densify=False -- nodes only', flush=True)
        elif _cnd:
            print(f'DEBUG: DS       {_cnd:,} candidates, '
                  f'{int(_s.get("ds_reached", 0)):,} reached a node over '
                  f'{int(_s.get("ds_arcs", 0)):,} arcs', flush=True)
            _adm = int(_s.get('ds_admissible', 0))
            _no = int(_s.get('ds_no_consensus', 0))
            _few = int(_s.get('ds_too_few', 0))
            _g = _s.get('ds_gamma')
            print(f'DEBUG:          {_adm:,} held an admissible arc: '
                  f'{_att:,} attached, {_no:,} did not'
                  + (f'   gamma p50 {np.median(_g):.3f}'
                     if _g is not None and len(_g) else ''), flush=True)
            print(f'DEBUG:          of those {_no:,}: {_few:,} had too few '
                  f'partners, {_no - _few:,} had enough and disagreed',
                  flush=True)
            print(f'DEBUG:          {int(_s.get("ds_multi_component", 0)):,} '
                  f'candidates saw more than one component; '
                  f'{int(_s.get("ds_shortlist_straddled", 0)):,} would have '
                  f'drawn a shortlist spanning two', flush=True)
            _xv = _s.get('ds_cross_component_votes')
            if _xv is not None:
                print(f'DEBUG:          cross-component votes among attached '
                      f'DS: {int(_xv):,}' + ('' if _xv == 0 else '   <-- BUG'),
                      flush=True)
        else:
            print('DEBUG: DS       no candidates cleared the threshold',
                  flush=True)
        print(f'DEBUG: total    {len(_s.get("iy", ())):,} PS + {_att:,} DS '
              f'= {len(_s.get("iy", ())) + _att:,} measured pixels', flush=True)
    return lab_out, vel_out, hgt_out, sea_out, coh_out
