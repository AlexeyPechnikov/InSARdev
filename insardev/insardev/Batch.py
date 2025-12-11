# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------
from __future__ import annotations
from .BatchCore import BatchCore
import numpy as np
import xarray as xr
from . import utils_xarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Stack import Stack
    import inspect

class Batch(BatchCore):
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off only the real 2D vars from Stack
        if isinstance(mapping, Stack):
            #print ('Batch __init__: Stack')
            real_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                # keep only non-complex data_vars that live on the ('y','x') grid
                real_vars = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind != 'c'
                    and tuple(ds[v].dims) == ('y', 'x')
                ]
                real_dict[key] = ds[real_vars]
            mapping = real_dict
        #print('Batch __init__ mapping', mapping or {}, '\n')
        # delegate to your base class for the actual init
        super().__init__(mapping or {})
    
    def clip(self, min=None, max=None, **kwargs):
        """
        used for correlation in [0,1] range
        """
        return BatchUnit(super().clip(min=min, max=max, **kwargs))
    
    def plot(
        self,
        cmap = 'turbo',
        alpha = 0.5,
        caption='Phase, [rad]',
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["alpha"] = alpha
        kwargs["caption"] = caption
        return super().plot(*args, **kwargs)

class BatchWrap(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores wrapped phase (real values).
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None, wrap: bool = True):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchComplex)):
            raise ValueError(f'ERROR: BatchWrap does not support Stack or BatchComplex objects.')
        # skip wrapping for intermediate objects like DatasetCoarsen
        if not wrap:
            dict.__init__(self, mapping or {})
        else:
            wrapped = {k: self.wrap(v) for k, v in (mapping or {}).items()}
            dict.__init__(self, wrapped)

    @staticmethod
    def wrap(data):
        return np.mod(data + np.pi, 2 * np.pi) - np.pi

    def __add__(self, other: Batch):
        keys = self.keys()
        #& other.keys()
        return type(self)({k: self[k] + other[k] if k in other else self[k] for k in keys})

    def __sub__(self, other: Batch):
        import xarray as xr
        keys = self.keys()
        result = {}
        for k in keys:
            if k not in other:
                result[k] = self[k]
            else:
                val = other[k]
                ds = self[k]
                # Handle per-pair coefficients from burst_polyfit
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    sample_var = list(ds.data_vars)[0]
                    sample_da = ds[sample_var]
                    has_pair_dim = 'pair' in sample_da.dims
                    n_pairs = sample_da.sizes.get('pair', 1)
                    first_elem = val[0]

                    if isinstance(first_elem, (list, tuple)):
                        # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                        result[k] = ds - self[[k]].polyval({k: val})[k]
                    elif has_pair_dim and len(val) == n_pairs:
                        # Multi-pair degree=0: [off0, off1, ...]
                        offsets = xr.DataArray(val, dims=['pair'],
                                               coords={'pair': sample_da.coords['pair']})
                        result[k] = ds - offsets
                    elif len(val) == 1:
                        # Single value wrapped in list: [offset]
                        result[k] = ds - val[0]
                    else:
                        # Single pair degree=1: [ramp, offset]
                        result[k] = ds - self[[k]].polyval({k: val})[k]
                else:
                    result[k] = ds - val
        return type(self)(result)

    def __mul__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self[k] * other[k] if k in other else self[k] for k in keys})

    def __rmul__(self, other):
        # scalar * batch  → map scalar * each dataset
        return type(self)({k: other * v for k, v in self.items()})

    def __truediv__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self[k] / other[k] if k in other else self[k] for k in keys})

    def sin(self, **kwargs) -> Batch:
        """
        Return a Batch of the sin(theta) DataArrays, preserving attrs if requested.
        """
        return Batch(self.map_da(lambda da: xr.ufuncs.sin(da), **kwargs))

    def cos(self, **kwargs) -> Batch:
        """
        Return a Batch of the cos(theta) DataArrays, preserving attrs if requested.
        """
        return Batch(self.map_da(lambda da: xr.ufuncs.cos(da), **kwargs))
    
    def iexp(self, sign: int = -1, **kwargs):
        """
        Apply exp(sign * 1j * da) like np.exp(-1j * intfs)
        
        - If sign = -1 (the default), this is exp(-1j * da).
        - If sign = +1, this is exp(+1j * da).
        """
        from .Batch import BatchComplex
        return BatchComplex(self.map_da(lambda da: xr.ufuncs.exp(sign * 1j * da), **kwargs))

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Converts wrapped phase to complex numbers before aggregation and back to wrapped phase after.
        """
        print ('wrap _agg')
        import inspect
        import xarray as xr
        import pandas as pd
        out = {}
        for key, obj in self.items():
            # get the aggregation function
            fn = getattr(obj, name)
            sig = inspect.signature(fn)
            
            # perform aggregation in complex domain
            if 'dim' in sig.parameters:
                # intfs.mean('pair').isel(0)
                #agg_result = fn(dim=dim, **kwargs)
                complex_obj = xr.ufuncs.exp(1j * obj.astype('float32'))
                #fn_complex = getattr(complex_obj, name)
                #agg_result = fn_complex(dim=dim, **kwargs)
                if name in ('var', 'std'):
                    # |E[e^(iθ)]|
                    R = xr.ufuncs.abs(complex_obj.mean(dim=dim, **kwargs))
                    if name == 'var':
                        # 1 - |E[e^(iθ)]|
                        agg_result = (1 - R)
                    else:  # std
                        # √(-2 ln|E[e^(iθ)]|)
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    fn_complex = getattr(complex_obj, name)
                    agg_result = fn_complex(dim=dim, **kwargs)
                    # convert back to wrapped phase
                    agg_result = xr.ufuncs.angle(agg_result)
            else:
                # intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()
                # already in complex domain, see coarsen()
                if name in ('var', 'std'):
                    R = xr.ufuncs.abs(obj.mean(**kwargs))
                    if name == 'var':
                        agg_result = (1 - R)
                    else:  # std
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    agg_result = fn(**kwargs)
                    agg_result = xr.ufuncs.angle(agg_result)
            
            # Convert back to wrapped phase
            out[key] = agg_result.astype('float32')

            # xarray coarsen + aggregate do not preserve multiindex pair
            if all(coord in out[key].coords for coord in ('pair', 'ref','rep')) \
                   and not isinstance(out[key].coords['pair'], pd.MultiIndex):
                out[key] = out[key].set_index(pair=['ref', 'rep'])
            
        #print ('wrap _agg self.chunks', self.chunks)
        #return type(self)(out).chunk(self.chunks)
        print ('wrap _agg self.chunks', self.chunks)
        # filter out collapsed dimensions
        sample = next(iter(out.values()), None)
        dims = (sample.dims or []) if hasattr(sample, 'dims') else []
        chunks = {d: size for d, size in self.chunks.items() if d in dims}
        print ('wrap chunks', chunks)
        result = type(self)(out)
        if chunks:
            return result.chunk(chunks)
        return result

    def coarsen(self, window: dict[str, int], **kwargs) -> Batch:
        """
        Coarsen each DataSet in the batch by integer factors and align the 
        blocks so that they fall on "nice" grid boundaries.

        Parameters
        ----------
        window : dict[str,int]
            e.g. {'y': 2, 'x': 8}
        **kwargs
            extra args forwarded into the reduction, e.g. skipna=True.

        Returns
        -------
        Batch
            A new Batch where each Dataset has been sliced for alignment,
            coarsened by `window`, then reduced by `.mean()` (or whichever
            `func` you chose).
        """
        #print ('wrap coarsen')
        chunks = self.chunks
        #print ('self.chunks', chunks)
        out = {}
        # produce unified grid and chunks for all datasets in the batch
        for key, ds in self.items():
            # convert to complex numbers for proper circular statistics
            ds2 = xr.ufuncs.exp(1j * ds.astype('float32'))
            # align each dimension
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds2, dim, factor)
                #print ('start', start)
                if start is not None:
                    # rechunk to the original chunk sizes
                    ds2 = ds2.isel({dim: slice(start, None)}).chunk(chunks)
                    # or allow a bit different chunks for coarsening
                    #ds2 = ds2.isel({dim: slice(start, None)})
            # coarsen
            out[key] = ds2.coarsen(window, **kwargs)

        # wrap=False since these are DatasetCoarsen objects, not actual data
        return type(self)(out, wrap=False)

    def plot(
        self,
        cmap = 'gist_rainbow_r',
        alpha = 0.7,
        caption='Phase, [rad]',
        vmin=-np.pi,
        vmax=np.pi,
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["alpha"] = alpha
        kwargs["caption"] = caption
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
        return super().plot(*args, **kwargs)

    # def gaussian(self, *args, **kwargs):
    #     """
    #     Phase-aware Gaussian smoothing for wrapped phase data.
    #     """
    #     return self.iexp().gaussian(*args, **kwargs).angle()

    # def gaussian(self, *args, **kwargs):
    #     """
    #     Phase-aware Gaussian smoothing by filtering sin(θ) and cos(θ) separately,
    #     then recombining via atan2.  No complex dtype ever created.
    #     """
    #     from .Batch import Batch
    #     import xarray as xr

    #     keep_attrs = kwargs.pop('keep_attrs', None)
    #     # build two Batches of the real sin and cos components and filter them
    #     sin = self.sin(keep_attrs=keep_attrs).gaussian(*args, **kwargs)
    #     cos = self.cos(keep_attrs=keep_attrs).gaussian(*args, **kwargs)

    #     # compute wrapped phase using np.arctan2
    #     out = {k: xr.Dataset({
    #         var: xr.ufuncs.arctan2(sin[k][var], cos[k][var]).astype('float32')
    #         for var in sin[k].data_vars
    #     }) for k in self.keys()}

    #     return BatchWrap(out)

    def gaussian(self, *args, **kwargs):
        """
        Phase-aware Gaussian smoothing by filtering sin(θ) and cos(θ) separately,
        then recombining via arctan2.
        """
        from .Batch import Batch
        import xarray as xr

        keep_attrs = kwargs.pop('keep_attrs', False)
        data_vars = next(iter(self.values())).data_vars

        # build two Batches of the real sin and cos components and filter them
        sin = self.sin(keep_attrs=keep_attrs).gaussian(*args, **kwargs)
        cos = self.cos(keep_attrs=keep_attrs).gaussian(*args, **kwargs)

        # compute wrapped phase using arctan2
        out: dict[str, xr.Dataset] = {}
        for k in self.keys():
            phase_vars = {}
            for var in data_vars:
                phase = xr.ufuncs.arctan2(sin[k][var], cos[k][var]).astype('float32')
                if keep_attrs:
                    phase.attrs = self[k][var].attrs.copy()
                phase_vars[var] = phase
            ds = xr.Dataset(phase_vars)
            if keep_attrs:
                ds.attrs = self[k].attrs.copy()
            out[k] = ds

        return BatchWrap(out)

    def residuals(self, debug: bool = False) -> float | list[float]:
        """
        Measure phase offset discrepancy across all burst overlaps.

        Computes the weighted mean of absolute median phase differences
        across all overlapping regions. After offset correction with fit(),
        these median differences should be close to zero.

        Parameters
        ----------
        debug : bool, optional
            Print debug information for each overlap. Default is False.

        Returns
        -------
        float or list[float]
            Single pair: Weighted mean absolute median phase discrepancy in radians.
            Multiple pairs: List of discrepancies, one per pair.
            0.0 = perfect alignment, π = maximum discrepancy.

            Practical interpretation:

            - < 0.1 rad: Excellent alignment
            - 0.1 - 0.5 rad: Good alignment
            - 0.5 - 1.0 rad: Moderate misalignment
            - > 1.0 rad: Poor alignment

        Examples
        --------
        >>> # Compare before and after alignment
        >>> before = intfs.residuals()
        >>> aligned = intfs.align()
        >>> after = aligned.residuals()
        >>> print(f'Discrepancy reduced from {before} to {after}')
        """
        import numpy as np
        import dask

        def circ_wrap(x):
            """Wrap to [-π, π)"""
            return (x + np.pi) % (2*np.pi) - np.pi

        # Collect burst extents and detect pair dimension
        ids = sorted(self.keys())

        sample_ds = self[ids[0]]
        pol = list(sample_ds.data_vars)[0] if hasattr(sample_ds, 'data_vars') else None
        sample_da = sample_ds[pol] if pol else sample_ds
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        # Extract pathNumber and subswath attributes for each burst (required for stats)
        burst_subswath = {}
        burst_track = {}  # pathNumber + subswath for detailed debug output
        for bid in ids:
            ds = self[bid]
            if 'subswath' not in ds.attrs:
                raise ValueError(f"Burst '{bid}' missing required 'subswath' attribute")
            if 'pathNumber' not in ds.attrs:
                raise ValueError(f"Burst '{bid}' missing required 'pathNumber' attribute")
            burst_subswath[bid] = ds.attrs['subswath']
            burst_track[bid] = f"{ds.attrs['pathNumber']}{ds.attrs['subswath']}"

        extents = {}
        for bid in ids:
            ds = self[bid]
            pol = list(ds.data_vars)[0] if hasattr(ds, 'data_vars') else None
            da = ds[pol] if pol else ds
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            y_coords = da.coords['y'].values
            x_coords = da.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())

        def extents_overlap(e1, e2):
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap

        # Find all overlapping burst pairs
        overlap_pairs = []
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                e2 = extents[id2]
                if extents_overlap(e1, e2):
                    overlap_pairs.append((id1, id2))

        if not overlap_pairs:
            return [0.0] * n_pairs if has_pair_dim else 0.0

        if debug:
            print(f'residuals: found {len(overlap_pairs)} overlap pairs, {n_pairs} pair(s)', flush=True)

        # Build all lazy phase differences (dask graphs)
        jobs = []
        lazy_diffs = []
        for id1, id2 in overlap_pairs:
            i1 = self[id1]
            i2 = self[id2]

            pol = list(i1.data_vars)[0] if hasattr(i1, 'data_vars') else None
            if pol:
                i1 = i1[pol]
                i2 = i2[pol]

            for pair_idx in range(n_pairs):
                i1_p = i1.isel(pair=pair_idx) if 'pair' in i1.dims else i1
                i2_p = i2.isel(pair=pair_idx) if 'pair' in i2.dims else i2
                phase_diff = i2_p - i1_p
                jobs.append((id1, id2, pair_idx))
                lazy_diffs.append(phase_diff)

        # Compute all phase differences at once - dask schedules efficiently
        if debug:
            print(f'Computing {len(lazy_diffs)} phase differences...', flush=True)
        computed_diffs = dask.compute(*lazy_diffs)

        # Process computed results
        results = []
        for (id1, id2, pair_idx), phase_diff in zip(jobs, computed_diffs):
            valid = phase_diff.values.ravel()
            valid = valid[np.isfinite(valid)]

            if len(valid) == 0:
                continue

            valid = circ_wrap(valid)
            median_diff = np.median(valid)
            abs_discrepancy = np.abs(circ_wrap(median_diff))
            weight = len(valid)

            results.append((pair_idx, abs_discrepancy, weight, id1, id2, median_diff))

        # Aggregate results per pair and per subswath
        total_weights = [0.0] * n_pairs
        weighted_sums = [0.0] * n_pairs

        # Per-subswath tracking for debug
        subswath_stats = {}  # {(subswath, pair_idx): {'sum': float, 'weight': float, 'count': int, 'values': []}}
        per_overlap_discrepancies = {p: [] for p in range(n_pairs)}  # For computing std

        for result in results:
            if result is None:
                continue
            pair_idx, abs_discrepancy, weight, id1, id2, median_diff = result
            weighted_sums[pair_idx] += abs_discrepancy * weight
            total_weights[pair_idx] += weight
            per_overlap_discrepancies[pair_idx].append(abs_discrepancy)

            # Extract track info for debug stats
            if debug:
                track1 = burst_track[id1]
                track2 = burst_track[id2]

                # Categorize: same track or cross-track
                if track1 == track2:
                    track_key = track1
                else:
                    track_key = f'{track1}-{track2}'

                key = (track_key, pair_idx)
                if key not in subswath_stats:
                    subswath_stats[key] = {'sum': 0.0, 'weight': 0.0, 'count': 0, 'values': []}
                subswath_stats[key]['sum'] += abs_discrepancy * weight
                subswath_stats[key]['weight'] += weight
                subswath_stats[key]['count'] += 1
                subswath_stats[key]['values'].append(abs_discrepancy)

        discrepancies = []
        for p in range(n_pairs):
            if total_weights[p] == 0:
                discrepancies.append(0.0)
            else:
                discrepancies.append(round(weighted_sums[p] / total_weights[p], 3))

        if debug:
            # Compute std for overall discrepancy
            for p in range(n_pairs):
                vals = per_overlap_discrepancies[p]
                if len(vals) > 1:
                    std = np.std(vals)
                    print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} ± {std:.3f} rad ({len(vals)} overlaps)', flush=True)
                else:
                    print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} rad ({len(vals)} overlaps)', flush=True)

            # Print per-track stats (only for pair_idx=0 to avoid clutter)
            print('Per-track discrepancy (pair 0):', flush=True)
            for (track, pair_idx), stats in sorted(subswath_stats.items()):
                if pair_idx == 0 and stats['weight'] > 0:
                    track_disc = stats['sum'] / stats['weight']
                    vals = stats['values']
                    if len(vals) > 1:
                        track_std = np.std(vals)
                        print(f'  {track}: {track_disc:.3f} ± {track_std:.3f} rad ({stats["count"]} overlaps)', flush=True)
                    else:
                        print(f'  {track}: {track_disc:.3f} rad ({stats["count"]} overlaps)', flush=True)

        # Return single value for single pair, list for multiple
        if n_pairs == 1 and not has_pair_dim:
            return discrepancies[0]
        return discrepancies

    def fit(self,
            degree: int = 0,
            method: str = 'median',
            debug: bool = False,
            return_residuals: bool = False):
        """
        Estimate per-burst polynomial coefficients using overlap-based least-squares.

        Fits polynomial corrections (offset or offset+ramp) to each burst by analyzing
        phase differences in overlapping regions. Uses global least-squares optimization
        to find consistent coefficients across all bursts.

        Parameters
        ----------
        degree : int, optional
            Polynomial degree:
            - 0 (default): Estimate offsets only.
            - 1: Estimate linear ramp (in x/range direction).
        method : str, optional
            Estimation method: 'median' (robust) or 'mean' (faster).
        debug : bool, optional
            Print debug information. Default is False.
        return_residuals : bool, optional
            If True, also return input residuals (before correction). Default is False.

        Returns
        -------
        dict or tuple
            If return_residuals is False:
                For single pair (no pair dimension):
                    degree=0: {burst_id: offset}
                    degree=1: {burst_id: [ramp, intercept]}
                For multiple pairs:
                    degree=0: {burst_id: [offset_pair0, offset_pair1, ...]}
                    degree=1: {burst_id: [[ramp0, intercept0], [ramp1, intercept1], ...]}
            If return_residuals is True:
                (coefficients_dict, residuals) where residuals is float or list[float]

        Examples
        --------
        >>> # 3-step alignment for best results (0.028 rad discrepancy):
        >>> # Step 1: Estimate offsets
        >>> offsets1 = intfs.fit(degree=0)
        >>> intfs1 = intfs - offsets1
        >>> # Step 2: Estimate ramps
        >>> ramps = intfs1.fit(degree=1)
        >>> intfs2 = intfs1 - intfs1.polyval(ramps)
        >>> # Step 3: Re-estimate offsets
        >>> offsets2 = intfs2.fit(degree=0)
        >>> # Combine coefficients (for single pair)
        >>> coeffs = {b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]] for b in offsets1}
        >>> aligned = intfs - intfs.polyval(coeffs)
        """
        import numpy as np
        import dask
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        from scipy.sparse.csgraph import connected_components

        # Constants
        MIN_OVERLAP_PIXELS = 50
        MIN_ROW_PIXELS = 10
        MIN_VALID_ROWS = 5
        MIN_INLIER_SAMPLES = 10
        MAD_OUTLIER_THRESHOLD = 2.5
        OUTPUT_PRECISION = 3
        RAMP_PRECISION = 9

        def circ_wrap(x):
            return (x + np.pi) % (2*np.pi) - np.pi

        def circ_diff(a, center):
            return circ_wrap(a - center)

        def circ_mean(a):
            return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))

        def circ_mad(a, center):
            return np.median(np.abs(circ_diff(a, center)))

        # Collect burst extents and x-centers
        ids = sorted(self.keys())
        n_bursts = len(ids)
        id_to_idx = {bid: i for i, bid in enumerate(ids)}

        # Detect number of pairs
        sample_ds = self[ids[0]]
        pol = list(sample_ds.data_vars)[0] if hasattr(sample_ds, 'data_vars') else None
        sample_da = sample_ds[pol] if pol else sample_ds
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        if debug:
            print(f'fit(degree={degree}): {n_bursts} bursts, {n_pairs} pair(s)', flush=True)

        # Extract pathNumber and subswath attributes for each burst (required for degree=1)
        # Use pathNumber + subswath as track key to prevent inter-path same-subswath overlaps
        # in near-polar regions from being used for ramp estimation
        burst_track = {}  # pathNumber + subswath (e.g., '33IW3')
        burst_subswath = {}  # just subswath for debug output
        for bid in ids:
            ds = self[bid]
            if degree == 1:
                if 'subswath' not in ds.attrs:
                    raise ValueError(f"Burst '{bid}' missing required 'subswath' attribute for ramp estimation")
                if 'pathNumber' not in ds.attrs:
                    raise ValueError(f"Burst '{bid}' missing required 'pathNumber' attribute for ramp estimation")
                burst_track[bid] = f"{ds.attrs['pathNumber']}{ds.attrs['subswath']}"
                burst_subswath[bid] = ds.attrs['subswath']

        extents = {}
        x_centers = {}

        for bid in ids:
            ds = self[bid]
            pol = list(ds.data_vars)[0] if hasattr(ds, 'data_vars') else None
            da = ds[pol] if pol else ds
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            y_coords = da.coords['y'].values
            x_coords = da.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())
            x_centers[bid] = float(np.mean(x_coords))

        def extents_overlap(e1, e2):
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap

        def process_phase_diff(phase, id1, id2, pair_idx):
            """Process a computed phase difference to extract offset and optionally ramp."""
            all_valid = phase.values.ravel()
            all_valid = all_valid[np.isfinite(all_valid)]

            if len(all_valid) < MIN_OVERLAP_PIXELS:
                return None

            if 'y' not in phase.dims or 'x' not in phase.dims:
                return None

            x_coords = phase.coords['x'].values

            # Row-wise processing
            row_phases = []
            row_x_centroids = []
            row_weights = []

            for y_idx in range(phase.shape[0]):
                row = phase.values[y_idx, :]
                valid_mask = np.isfinite(row)
                n_valid = np.sum(valid_mask)
                if n_valid >= MIN_ROW_PIXELS:
                    x_valid = x_coords[valid_mask]
                    phase_valid = row[valid_mask]
                    phase_unwrapped = np.unwrap(phase_valid)
                    row_mean = circ_wrap(np.mean(phase_unwrapped))
                    row_phases.append(row_mean)
                    row_x_centroids.append(np.mean(x_valid))
                    row_weights.append(n_valid)

            if len(row_phases) < MIN_VALID_ROWS:
                return None

            a = np.array(row_phases)
            x_row = np.array(row_x_centroids)
            weights = np.array(row_weights)
            a = circ_wrap(a)

            # Outlier rejection
            if method == 'median':
                offset_initial = np.median(a)
                mad = circ_mad(a, offset_initial)
                if mad > 0:
                    inliers = np.abs(circ_diff(a, offset_initial)) <= MAD_OUTLIER_THRESHOLD * mad
                    if np.sum(inliers) >= MIN_INLIER_SAMPLES:
                        a = a[inliers]
                        x_row = x_row[inliers]
                        weights = weights[inliers]

            n_valid = int(np.sum(weights))
            x_centroid = float(np.average(x_row, weights=weights))

            # Compute offset
            if method == 'median':
                sorted_idx = np.argsort(a)
                cumsum = np.cumsum(weights[sorted_idx])
                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                offset = a[sorted_idx[median_idx]]
            else:
                offset = circ_mean(a)

            # Compute ramp if degree=1
            ramp_val = None
            if degree == 1 and len(a) >= MIN_VALID_ROWS:
                x_centered = x_row - x_centroid
                x_range = np.max(x_row) - np.min(x_row)
                if x_range > 100:
                    residuals = a - offset
                    Swxx = np.sum(weights * x_centered**2)
                    Swxr = np.sum(weights * x_centered * residuals)
                    if Swxx > 1e-10:
                        ramp_val = Swxr / Swxx

            return (id1, id2, pair_idx, circ_wrap(offset), ramp_val, x_centroid, n_valid)

        # Find overlapping burst pairs
        # For degree=1 (ramp), only use same-subswath overlaps (along-track)
        # For degree=0 (offset), use all overlaps including cross-subswath
        all_overlap_pairs = []
        cross_subswath_skipped = 0
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                if extents_overlap(e1, extents[id2]):
                    if degree == 1:
                        # For ramp estimation, only use same-track overlaps (same path + subswath)
                        # This prevents inter-path same-subswath overlaps in near-polar regions
                        track1, track2 = burst_track[id1], burst_track[id2]
                        if track1 != track2:
                            cross_subswath_skipped += 1
                            continue
                    all_overlap_pairs.append((id1, id2))

        if debug:
            print(f'Found {len(all_overlap_pairs)} overlapping burst pairs', flush=True)
            if degree == 1 and cross_subswath_skipped > 0:
                print(f'  (skipped {cross_subswath_skipped} cross-track pairs for ramp estimation)', flush=True)

        # Build all lazy phase differences (dask graphs)
        jobs = []
        lazy_diffs = []
        for id1, id2 in all_overlap_pairs:
            i1 = self[id1]
            i2 = self[id2]

            pol = list(i1.data_vars)[0] if hasattr(i1, 'data_vars') else None
            if pol:
                i1 = i1[pol]
                i2 = i2[pol]

            for pair_idx in range(n_pairs):
                i1_p = i1.isel(pair=pair_idx) if 'pair' in i1.dims else i1
                i2_p = i2.isel(pair=pair_idx) if 'pair' in i2.dims else i2
                phase_diff = i2_p - i1_p
                jobs.append((id1, id2, pair_idx))
                lazy_diffs.append(phase_diff)

        if debug:
            print(f'Computing {len(lazy_diffs)} phase differences...', flush=True)

        # Compute all phase differences at once - dask schedules efficiently
        computed_diffs = dask.compute(*lazy_diffs)

        # Process computed results
        results = []
        rejection_counts = {'too_few_pixels': 0, 'missing_dims': 0, 'too_few_rows': 0, 'no_ramp': 0}
        for (id1, id2, pair_idx), phase in zip(jobs, computed_diffs):
            result = process_phase_diff(phase, id1, id2, pair_idx)
            if result is not None:
                results.append(result)
            else:
                # Count rejections (only for pair_idx==0 to avoid double counting)
                if pair_idx == 0:
                    all_valid = phase.values.ravel()
                    all_valid = all_valid[np.isfinite(all_valid)]
                    if len(all_valid) < MIN_OVERLAP_PIXELS:
                        rejection_counts['too_few_pixels'] += 1
                    elif 'y' not in phase.dims or 'x' not in phase.dims:
                        rejection_counts['missing_dims'] += 1
                    else:
                        rejection_counts['too_few_rows'] += 1

        if debug:
            n_valid = len([r for r in results if r[2] == 0])  # count for pair_idx=0
            n_rejected = len(all_overlap_pairs) - n_valid
            print(f'  Rejections: {rejection_counts}', flush=True)
            if degree == 1:
                # Count how many had no ramp computed
                no_ramp = sum(1 for r in results if r[2] == 0 and r[4] is None)
                print(f'  Valid pairs with no ramp (x_range too small): {no_ramp}', flush=True)

        # Organize results by pair_idx
        pairs_by_pair_idx = {p: [] for p in range(n_pairs)}
        for result in results:
            if result is None:
                continue
            id1, id2, pair_idx, offset, ramp_val, x_centroid, n_used = result
            weight = np.sqrt(n_used)
            if degree == 0:
                pairs_by_pair_idx[pair_idx].append((id1, id2, offset, weight))
            else:
                if ramp_val is not None:
                    pairs_by_pair_idx[pair_idx].append((id1, id2, offset, ramp_val, x_centroid, weight))

        def solve_for_pair(pair_idx):
            """Solve least-squares for a single pair index."""
            pairs = pairs_by_pair_idx[pair_idx]

            if len(pairs) == 0:
                if degree == 0:
                    return {bid: 0.0 for bid in ids}
                else:
                    return {bid: [0.0, 0.0] for bid in ids}

            # Build connectivity graph
            adjacency = sparse.lil_matrix((n_bursts, n_bursts))
            for p in pairs:
                id1, id2 = p[0], p[1]
                i, j = id_to_idx[id1], id_to_idx[id2]
                adjacency[i, j] = 1
                adjacency[j, i] = 1

            n_components, labels = connected_components(adjacency.tocsr(), directed=False)

            if debug and pair_idx == 0:  # Only print for first pair to avoid spam
                print(f'  Found {n_components} connected component(s) for {len(pairs)} valid pairs', flush=True)
                for comp in range(n_components):
                    comp_indices = np.where(labels == comp)[0]
                    comp_ids = [ids[i] for i in comp_indices]
                    # Try to extract subswath info from burst IDs
                    subswaths = set()
                    paths = set()
                    for bid in comp_ids:
                        if '_IW' in bid:
                            sw = bid.split('_IW')[1][0]
                            subswaths.add(f'IW{sw}')
                        parts = bid.split('_')
                        if len(parts) >= 1 and parts[0].isdigit():
                            paths.add(parts[0])
                    print(f'    Component {comp}: {len(comp_ids)} bursts, paths={paths}, subswaths={subswaths}', flush=True)

            if degree == 0:
                offsets_out = {}

                for comp in range(n_components):
                    comp_indices = np.where(labels == comp)[0]
                    comp_ids = [ids[i] for i in comp_indices]
                    comp_id_to_local = {bid: i for i, bid in enumerate(comp_ids)}
                    n_comp = len(comp_ids)

                    if n_comp == 1:
                        offsets_out[comp_ids[0]] = 0.0
                        continue

                    comp_pairs = [(id1, id2, off, w) for id1, id2, off, w in pairs
                                  if id1 in comp_id_to_local and id2 in comp_id_to_local]

                    if len(comp_pairs) == 0:
                        for bid in comp_ids:
                            offsets_out[bid] = 0.0
                        continue

                    n_pairs_comp = len(comp_pairs)
                    A = sparse.lil_matrix((n_pairs_comp + 1, n_comp))
                    b = np.zeros(n_pairs_comp + 1)
                    W = np.zeros(n_pairs_comp + 1)

                    for k, (id1, id2, off, w) in enumerate(comp_pairs):
                        i = comp_id_to_local[id1]
                        j = comp_id_to_local[id2]
                        A[k, i] = -1
                        A[k, j] = +1
                        b[k] = off
                        W[k] = w

                    constraint_weight = np.sum(W[:-1]) * 100 if np.sum(W[:-1]) > 0 else 1e6
                    A[n_pairs_comp, 0] = 1
                    b[n_pairs_comp] = 0
                    W[n_pairs_comp] = constraint_weight

                    sqrt_W = np.sqrt(W)
                    result = lsqr(sparse.diags(sqrt_W) @ A.tocsr(), sqrt_W * b)

                    for i, bid in enumerate(comp_ids):
                        offsets_out[bid] = round(float(circ_wrap(result[0][i])), OUTPUT_PRECISION)

                return offsets_out

            else:  # degree == 1
                ramps_out = {}

                for comp in range(n_components):
                    comp_indices = np.where(labels == comp)[0]
                    comp_ids = [ids[i] for i in comp_indices]
                    comp_id_to_local = {bid: i for i, bid in enumerate(comp_ids)}
                    n_comp = len(comp_ids)

                    if n_comp == 1:
                        ramps_out[comp_ids[0]] = [0.0, 0.0]
                        continue

                    comp_pairs = [(id1, id2, off, r, xc, w) for id1, id2, off, r, xc, w in pairs
                                  if id1 in comp_id_to_local and id2 in comp_id_to_local]

                    if len(comp_pairs) == 0:
                        for bid in comp_ids:
                            ramps_out[bid] = [0.0, 0.0]
                        continue

                    n_pairs_comp = len(comp_pairs)
                    A = sparse.lil_matrix((n_pairs_comp + 1, n_comp))
                    b = np.zeros(n_pairs_comp + 1)
                    W = np.zeros(n_pairs_comp + 1)

                    for k, (id1, id2, off, ramp_diff, xc, w) in enumerate(comp_pairs):
                        i = comp_id_to_local[id1]
                        j = comp_id_to_local[id2]
                        A[k, i] = -1
                        A[k, j] = +1
                        b[k] = ramp_diff
                        W[k] = w

                    constraint_weight = np.sum(W[:-1]) * 100 if np.sum(W[:-1]) > 0 else 1e6
                    A[n_pairs_comp, 0] = 1
                    b[n_pairs_comp] = 0
                    W[n_pairs_comp] = constraint_weight

                    sqrt_W = np.sqrt(W)
                    result = lsqr(sparse.diags(sqrt_W) @ A.tocsr(), sqrt_W * b)

                    for i, bid in enumerate(comp_ids):
                        ramp = round(float(result[0][i]), RAMP_PRECISION)
                        intercept = round(-ramp * x_centers[bid], OUTPUT_PRECISION)
                        ramps_out[bid] = [ramp, intercept]

                return ramps_out

        # Solve for each pair
        results_per_pair = [solve_for_pair(p) for p in range(n_pairs)]

        # Compute residuals from the pairwise offsets (before correction)
        # This uses the same overlap data we already computed
        residuals_out = None
        if return_residuals:
            # Calculate weighted mean absolute offset per pair
            disc_per_pair = []
            for p in range(n_pairs):
                pairs_p = pairs_by_pair_idx[p]
                if len(pairs_p) == 0:
                    disc_per_pair.append(0.0)
                    continue

                # Extract offsets and weights
                if degree == 0:
                    # pairs_p = [(id1, id2, offset, weight), ...]
                    offsets = [abs(circ_wrap(t[2])) for t in pairs_p]
                    weights = [t[3] for t in pairs_p]
                else:
                    # pairs_p = [(id1, id2, offset, ramp, x_centroid, weight), ...]
                    offsets = [abs(circ_wrap(t[2])) for t in pairs_p]
                    weights = [t[5] for t in pairs_p]

                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_sum = sum(o * w for o, w in zip(offsets, weights))
                    disc_per_pair.append(round(weighted_sum / total_weight, 3))
                else:
                    disc_per_pair.append(0.0)

            if n_pairs == 1 and not has_pair_dim:
                residuals_out = disc_per_pair[0]
            else:
                residuals_out = disc_per_pair

            if debug:
                print(f'Input residuals: {residuals_out}', flush=True)

        # If single pair, return simple dict
        if n_pairs == 1 and not has_pair_dim:
            coeffs = results_per_pair[0]
            return (coeffs, residuals_out) if return_residuals else coeffs

        # Multiple pairs: combine into list per burst
        combined = {}
        for bid in ids:
            if degree == 0:
                combined[bid] = [results_per_pair[p][bid] for p in range(n_pairs)]
            else:
                combined[bid] = [results_per_pair[p][bid] for p in range(n_pairs)]

        return (combined, residuals_out) if return_residuals else combined

    def align(self,
              degree: int = 0,
              method: str = 'median',
              debug: bool = False,
              return_residuals: bool = False):
        """
        Align burst interferograms by removing phase offsets and optionally ionospheric ramps.

        Uses a multi-step approach for optimal alignment:
        - degree=0: Single-step offset correction
        - degree=1: 3-step correction (offset → ramp → re-offset) for ionospheric ramp removal

        The 3-step approach produces consistent fringes across bursts by removing
        per-track ionospheric ramps, which is essential for deformation analysis.

        Parameters
        ----------
        degree : int, optional
            Correction degree:
            - 0 (default): Offset-only correction (faster, good overlap alignment)
            - 1: Offset + linear ramp correction (better fringe continuity)
        method : str, optional
            Estimation method: 'median' (robust, default) or 'mean' (faster).
        debug : bool, optional
            Print debug information. Default is False.
        return_residuals : bool, optional
            If True, also return final residuals. Default is False.

        Returns
        -------
        BatchWrap or tuple
            If return_residuals is False:
                Aligned interferograms with phase corrections applied.
            If return_residuals is True:
                (aligned_intfs, residuals) where residuals is float or list[float]

        Examples
        --------
        >>> # Simple offset-only alignment (default)
        >>> aligned = intfs.align()
        >>>
        >>> # Alignment with ramp correction
        >>> aligned = intfs.align(degree=1)
        >>>
        >>> # With coherence filtering
        >>> aligned = intfs.where(corr >= 0.3).align()
        >>>
        >>> # Get alignment quality with result
        >>> aligned, res = intfs.align(return_residuals=True)
        >>> print('Residuals:', res)

        Notes
        -----
        For degree=1, the function performs:
        1. Estimate and remove offsets
        2. Estimate and remove ramps (using same-track overlaps only)
        3. Re-estimate offsets on ramp-corrected data
        4. Combine into final [ramp, offset] coefficients

        This 3-step approach achieves better fringe continuity than single-step
        methods because it separates the offset and ramp estimation, avoiding
        cross-contamination between the two.

        Use ``align(degree=0)`` (offset-only) when cross-subswath consistency
        matters most. Use ``align(degree=1)`` when per-track fringe continuity
        is more important than cross-subswath boundaries.
        """
        if degree == 0:
            # Single-step offset correction
            if debug:
                print('align(degree=0): single-step offset correction', flush=True)
                res_in = self.residuals()
                print(f'Input residuals: {res_in}', flush=True)

            offsets = self.fit(degree=0, method=method, debug=debug)
            aligned = self - offsets

            if debug or return_residuals:
                res_out = aligned.residuals()
                if debug:
                    print(f'Output residuals: {res_out}', flush=True)

            if return_residuals:
                return aligned, res_out
            return aligned

        elif degree == 1:
            # 3-step offset-ramp-offset correction
            if debug:
                print('align(degree=1): 3-step offset-ramp-offset correction', flush=True)
                res_in = self.residuals()
                print(f'Input residuals: {res_in}', flush=True)

            # Step 1: Estimate offsets
            if debug:
                print('\nStep 1: Estimate offsets...', flush=True)
            offsets1 = self.fit(degree=0, method=method, debug=debug)
            intfs1 = self - offsets1
            if debug:
                res1 = intfs1.residuals()
                print(f'Residuals after step 1: {res1}', flush=True)

            # Step 2: Estimate ramps (uses same-track overlaps only)
            if debug:
                print('\nStep 2: Estimate ramps...', flush=True)
            ramps = intfs1.fit(degree=1, method=method, debug=debug)
            intfs2 = intfs1 - intfs1.polyval(ramps)
            if debug:
                res2 = intfs2.residuals()
                print(f'Residuals after step 2: {res2}', flush=True)

            # Step 3: Re-estimate offsets
            if debug:
                print('\nStep 3: Re-estimate offsets...', flush=True)
            offsets2 = intfs2.fit(degree=0, method=method, debug=debug)

            # Combine coefficients: [ramp, offset1 + ramp_intercept + offset2]
            # Detect if multi-pair
            sample_bid = list(offsets1.keys())[0]
            is_multi_pair = isinstance(offsets1[sample_bid], list)

            if is_multi_pair:
                n_pairs = len(offsets1[sample_bid])
                coeffs = {
                    b: [[ramps[b][p][0], ramps[b][p][1] + offsets1[b][p] + offsets2[b][p]]
                        for p in range(n_pairs)]
                    for b in offsets1
                }
            else:
                coeffs = {
                    b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]]
                    for b in offsets1
                }

            aligned = self - self.polyval(coeffs)

            if debug or return_residuals:
                res_out = aligned.residuals()
                if debug:
                    print(f'Final residuals: {res_out}', flush=True)

            if return_residuals:
                return aligned, res_out
            return aligned

        else:
            raise ValueError(f"degree must be 0 or 1, got {degree}")

class BatchUnit(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores correlation in the range [0,1].
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchWrap, BatchComplex)):
            raise ValueError(f'ERROR: BatchUnit does not support Stack, BatchWrap or BatchComplex objects.')
        dict.__init__(self, mapping or {})

    def plot(
        self,
        cmap = 'auto',
        caption='Correlation',
        alpha=1,
        vmin=0,
        vmax=1,
        *args,
        **kwargs
    ):
        import matplotlib.colors as mcolors
        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
        kwargs["cmap"] = cmap
        kwargs["caption"] = caption
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
        kwargs["alpha"] = alpha
        return super().plot(*args, **kwargs)

class BatchComplex(BatchCore):
    """
    This class has 'data' stack variable for the datasets in the dict.
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off only the complex vars from Stack
        if isinstance(mapping, Stack):
            complex_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                # keep only complex data_vars
                complex_vars = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c'
                ]
                complex_dict[key] = ds[complex_vars]
            mapping = complex_dict

        #print('BatchComplex __init__ mapping', mapping or {}, '\n')
        # delegate to your base class for the actual init
        super().__init__(mapping or {})

    def real(self, **kwargs):
        """
        Return the real part of each complex data variable,
        producing a Batch of real-valued Datasets.
        """
        out = {}
        for key, ds in self.items():
            # ds.map() applies the lambda to each DataArray in the Dataset
            ds_real = ds.map(lambda da: da.real, **kwargs)
            out[key] = ds_real
        return Batch(out)

    def imag(self, **kwargs):
        """
        Return the imaginary part of each complex data variable,
        producing a Batch of real-valued Datasets.
        """
        out = {}
        for key, ds in self.items():
            ds_imag = ds.map(lambda da: da.imag, **kwargs)
            out[key] = ds_imag
        return Batch(out)

    def abs(self, **kwargs):
        print ('BatchComplex abs')
        return Batch(self.map_da(lambda da: xr.ufuncs.abs(da), **kwargs))

    def power(self, **kwargs):
        """ element-wise |x|², i.e. signal intensity """
        return Batch(self.map_da(lambda da: xr.ufuncs.abs(da)**2, **kwargs))

    def conj(self, **kwargs):
        """intfs.iexp().conj() for np.exp(-1j * intfs)"""
        return self.map_da(lambda da: xr.ufuncs.conj(da), **kwargs)

    def angle(self, **kwargs):
        """
        Compute element-wise phase (angle), returning a BatchWrap of float32 DataArrays in [-π, π].
        """
        return BatchWrap(self.map_da(lambda da: np.arctan2(da.imag, da.real).astype(np.float32), **kwargs))

    def angle(self, **kwargs):
        """
        Compute element-wise phase (angle) for the complex variables only,
        returning a BatchWrap of float32 DataArrays in [-π, π].
        """
        out = {}
        for k, ds in self.items():
            # select only the vars whose dtype is complex
            complex_vars = [
                var for var in ds.data_vars
                if ds[var].dtype.kind == 'c'
            ]
            if not complex_vars:
                # no complex vars → skip
                continue

            # subset to just those, then map over each DataArray
            ds_complex = ds[complex_vars]
            ds_phase = ds_complex.map(
                lambda da: xr.ufuncs.angle(da).astype('float32'),
                **kwargs
            )

            out[k] = ds_phase

        # package up as a BatchWrap (real, wrapped-phase)
        return BatchWrap(out)

    def plot(self, *args, **kwargs):
        """
        Plotting is not supported on raw complex batches.
        Convert to real values first (e.g. with .angle() or .abs()).
        """
        raise NotImplementedError(
            "BatchComplex objects do not support plot().\n"
            "Convert to a real-valued batch first, e.g.:\n"
            "  • use `.angle()` to get wrapped phase → BatchWrap\n"
            "  • use `.abs()` or `.power()` to get magnitude → Batch"
        )

    @staticmethod
    def _goldstein(phase, corr, psize=32, debug=False):
        import xarray as xr
        import numpy as np
        import dask
        from numbers import Real
        from collections.abc import Mapping
        import warnings
        # Ignore *any* RuntimeWarning coming from dask/_task_spec.py
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            module=r'dask\._task_spec'
        )
        # …and just in case you want to match by message too:
        warnings.filterwarnings(
            'ignore',
            message='invalid value encountered in divide',
            category=RuntimeWarning,
            module=r'dask\._task_spec'
        )

        if debug:
            print ('DEBUG: goldstein')

        if psize is None:
            # miss the processing
            return phase
        
        if not isinstance(psize, (Real, Mapping)):
            raise ValueError('ERROR: psize should be an integer, float, or dictionary')

        if isinstance(psize, Real):
            psize = {'y': psize, 'x': psize}

        # Handle Dataset objects by extracting the first DataArray
        if isinstance(phase, xr.Dataset):
            phase = next(iter(phase.data_vars.values()))
        if isinstance(corr, xr.Dataset):
            corr = next(iter(corr.data_vars.values()))

        def apply_pspec(data, alpha):
            # NaN is allowed value
            assert not(alpha < 0), f'Invalid parameter value {alpha} < 0'
            wgt = np.power(np.abs(data)**2, alpha / 2)
            data = wgt * data
            return data

        def make_wgt(psize):
            nyp, nxp = psize['y'], psize['x']
            # Create arrays of horizontal and vertical weights
            wx = 1.0 - np.abs(np.arange(nxp // 2) - (nxp / 2.0 - 1.0)) / (nxp / 2.0 - 1.0)
            wy = 1.0 - np.abs(np.arange(nyp // 2) - (nyp / 2.0 - 1.0)) / (nyp / 2.0 - 1.0)
            # Compute the outer product of wx and wy to create the top-left quadrant of the weight matrix
            quadrant = np.outer(wy, wx)
            # Create a full weight matrix by mirroring the quadrant along both axes
            wgt = np.block([[quadrant, np.flip(quadrant, axis=1)],
                            [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]])
            return wgt

        def patch_goldstein_filter(data, corr, wgt, psize):
            """
            Apply the Goldstein adaptive filter to the given data.

            Args:
                data: 2D numpy array of complex values representing the data to be filtered.
                corr: 2D numpy array of correlation values. Must have the same shape as `data`.

            Returns:
                2D numpy array of filtered data.
            """
            # Calculate alpha
            alpha = 1 - (wgt * corr).sum() / wgt.sum()
            data = np.fft.fft2(data, s=(psize['y'], psize['x']))
            data = apply_pspec(data, alpha)
            data = np.fft.ifft2(data, s=(psize['y'], psize['x']))
            return wgt * data

        def apply_goldstein_filter(data, corr, psize, wgt_matrix):
            # Create an empty array for the output
            out = np.zeros(data.shape, dtype=np.complex64)
            # ignore processing for empty chunks 
            if np.all(np.isnan(data)):
                return out
            # Create the weight matrix
            #wgt_matrix = make_wgt(psize)
            # Iterate over windows of the data
            for i in range(0, data.shape[0] - psize['y'], psize['y'] // 2):
                for j in range(0, data.shape[1] - psize['x'], psize['x'] // 2):
                    # Create proocessing windows
                    data_window = data[i:i+psize['y'], j:j+psize['x']]
                    corr_window = corr[i:i+psize['y'], j:j+psize['x']]
                    # do not process NODATA areas filled with zeros
                    fraction_valid = np.count_nonzero(data_window != 0) / data_window.size
                    if fraction_valid >= 0.5:
                        wgt_window = wgt_matrix[:data_window.shape[0],:data_window.shape[1]]
                        # Apply the filter to the window
                        filtered_window = patch_goldstein_filter(data_window, corr_window, wgt_window, psize)
                        # Add the result to the output array
                        slice_i = slice(i, min(i + psize['y'], out.shape[0]))
                        slice_j = slice(j, min(j + psize['x'], out.shape[1]))
                        out[slice_i, slice_j] += filtered_window[:slice_i.stop - slice_i.start, :slice_j.stop - slice_j.start]
            return out

        assert phase.shape == corr.shape, f'ERROR: phase and correlation variables have different shape \
                                          ({phase.shape} vs {corr.shape})'

        stack =[]
        for ind in range(len(phase)):
            # Apply function with overlap; psize//2 overlap is not enough (some empty lines produced)
            # use complex data and real correlation
            # fill NaN values in correlation by zeroes to prevent empty output blocks
            block = dask.array.map_overlap(apply_goldstein_filter,
                                           phase[ind].fillna(0).data,
                                           corr[ind].fillna(0).data,
                                           depth=(psize['y'] // 2 + 2, psize['x'] // 2 + 2),
                                           dtype=np.complex64, 
                                           meta=np.array(()),
                                           psize=psize,
                                           wgt_matrix = make_wgt(psize))
            # Calculate the phase
            stack.append(block)
            del block

        # Create DataArray with proper coordinates and attributes
        ds = xr.DataArray(
            dask.array.stack(stack),
            coords=phase.coords,
            dims=phase.dims,
            name=phase.name,
            attrs=phase.attrs
        )
        del stack
        # replace zeros produces in NODATA areas
        return ds.where(np.isfinite(phase))

    def goldstein(self, corr: BatchUnit, psize: int | dict[str, int] = 32, debug: bool = False):
        """
        Apply Goldstein adaptive filter to each dataset in the batch.
        
        Parameters
        ----------
        corr : BatchUnit
            Batch of correlation values to use for filtering.
        psize : int or dict[str, int], optional
            Patch size for the filter. If int, same size used for both dimensions.
            If dict, specify {'y': size_y, 'x': size_x}. Default is 32.
        debug : bool, optional
            Print debug information. Default is False.
            
        Returns
        -------
        BatchComplex
            New batch with filtered phase values
        """
        # Check if correlation is a BatchUnit by checking its class name
        if corr.__class__.__name__ != 'BatchUnit':
            raise ValueError("corr must be a BatchUnit")
            
        if set(corr.keys()) != set(self.keys()):
            raise ValueError("corr must have the same keys as self")
            
        # Apply Goldstein filter to each dataset
        result = {}
        for k in self.keys():
            ds = self[k]
            filtered_vars = {}
            
            # Process each complex data variable in the dataset
            for var_name, var_data in ds.data_vars.items():
                if var_data.dtype.kind == 'c':  # Only process complex variables
                    filtered_data = self._goldstein(
                        phase=var_data,
                        corr=corr[k],
                        psize=psize,
                        debug=debug
                    )
                    filtered_vars[var_name] = filtered_data
                else:
                    filtered_vars[var_name] = var_data
            
            # Create a new dataset with the filtered variables
            result[k] = xr.Dataset(
                filtered_vars,
                coords=ds.coords,
                attrs=ds.attrs
            )
            
        return type(self)(result)
