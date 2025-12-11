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
from .Stack_export import Stack_export
from insardev_toolkit import progressbar

class Stack_plot(Stack_export):
    import xarray as xr
    import numpy as np
    import pandas as pd
    import matplotlib

    def plot(self, cmap='turbo', alpha=1):
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import patheffects

        df = self.to_dataframe().reset_index()
        df['date'] = df['startTime'].dt.date

        df['label'] = df.apply(lambda rec: f"{rec['flightDirection'].replace('E','')[:3]} {rec['date']} [{rec['pathNumber']}]", axis=1)
        unique_labels = sorted(df['label'].unique())
        unique_paths = sorted(df['pathNumber'].astype(str).unique())
        #colors = {label[-4:-1]: 'orange' if label[0] == 'A' else 'cyan' for i, label in enumerate(unique_labels)}
        n = len(unique_labels)
        colormap = matplotlib.cm.get_cmap(cmap, n)
        color_map = {label[-4:-1]: colormap(i) for i, label in enumerate(unique_labels)}
        fig, ax = plt.subplots()
        for label, group in df.groupby('label'):
            group.plot(ax=ax, edgecolor=color_map[label[-4:-1]], facecolor='none', lw=0.25, label=label)
        handles = [matplotlib.lines.Line2D([0], [0], color=color_map[label[-4:-1]], lw=1, label=label) for label in unique_labels]
        ax.legend(handles=handles, loc='upper right')

        col = df.columns[0]
        for _, row in df.drop_duplicates(subset=[col]).iterrows():
            # compute centroid
            x, y = row.geometry.centroid.coords[0]
            ax.annotate(
                str(row[col]),
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                color=color_map[row['label'][-4:-1]],
                path_effects=[patheffects.withStroke(linewidth=0.25, foreground='black')],
                alpha=1
            )

        ax.set_title('Sentinel-1 Burst Footprints')
        ax.set_xlabel('easting [m]')
        ax.set_ylabel('northing [m]')

    # def plot_dataset(self,
    #                  data: xr.Dataset | xr.DataArray,
    #                  polarizations: list[str] | tuple[str] | str | None,
    #                  cmap: matplotlib.colors.Colormap | str | None,
    #                  vmin: float | None,
    #                  vmax: float | None,
    #                  quantile: float | None,
    #                  symmetrical: bool,
    #                  caption: str,
    #                  cols: int,
    #                  rows: int,
    #                  size: float,
    #                  nbins: int,
    #                  aspect: float,
    #                  y: float,
    #                  wrap: bool,
    #                  _size: tuple[int, int] | None,
    #                  ):
    #     import xarray as xr
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt

    #     # no data means no plot and no error
    #     if data is None:
    #         return

    #     assert isinstance(data, (dict, list, tuple, xr.Dataset, xr.DataArray)), 'ERROR: data should be a dict or list or tuple or Dataset or DataArray'

    #     # screen size in pixels (width, height) to estimate reasonable number pixels per plot
    #     # this is quite large to prevent aliasing on 600dpi plots without additional processing
    #     if _size is None:
    #         _size = (8000,4000)

    #     def plot_polarization(data, polarization):

    #         if isinstance(data, dict):
    #             data = list(data.values())

    #         if isinstance(data, xr.Dataset):
    #             stackvar = list(data[polarization].dims)[0]
    #             da = data[polarization].isel({stackvar: slice(0, rows)})
    #         else:
    #             stackvar = list(data[0].dims)[0]
    #             das = [da[polarization].isel({stackvar: slice(0, rows)}) for da in data]
    #             da = self.to_dataset(das)
    #             del das

    #         if 'stack' in da.dims and isinstance(da.coords['stack'].to_index(), pd.MultiIndex):
    #             da = da.unstack('stack')
            
    #         # there is no reason to plot huge arrays much larger than screen size for small plots
    #         #print ('screen_size', screen_size)
    #         size_y, size_x = da.shape[1:]
    #         #print ('size_x, size_y', size_x, size_y)
    #         factor_y = int(np.round(size_y / (_size[1] / rows)))
    #         factor_x = int(np.round(size_x / (_size[0] / cols)))
    #         #print ('factor_x, factor_y', factor_x, factor_y)
    #         # decimate for faster plot, do not coarsening without antialiasing
    #         # maybe data is already smoothed and maybe not, decimation is the only safe option
    #         da = da[:,::max(1, factor_y), ::max(1, factor_x)]
    #         # materialize for all the calculations and plotting
    #         progressbar(da := da.persist(), desc=f'Computing {polarization} Plot'.ljust(25))

    #         # calculate min, max when needed
    #         if quantile is not None:
    #             _vmin, _vmax = np.nanquantile(da, quantile)
    #         else:
    #             _vmin, _vmax = vmin, vmax
    #         # define symmetrical boundaries
    #         if symmetrical is True and _vmax > 0:
    #             minmax = max(abs(_vmin), _vmax)
    #             _vmin = -minmax
    #             _vmax =  minmax
            
    #         # multi-plots ineffective for linked lazy data
    #         fg = (self.wrap(da) if wrap else da).rename(caption)\
    #             .plot.imshow(
    #             col=stackvar,
    #             col_wrap=min(cols, da[stackvar].size), size=size, aspect=aspect,
    #             vmin=_vmin, vmax=_vmax, cmap=cmap
    #         )
    #         #fg.set_axis_labels('Range', 'Azimuth')
    #         fg.set_ticks(max_xticks=nbins, max_yticks=nbins)
    #         fg.fig.suptitle(f'{polarization} {caption}', y=y)
    #         return fg

    #     if quantile is not None:
    #         assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

    #     if not isinstance(data, (xr.Dataset, xr.DataArray)):
    #         raise ValueError(f'ERROR: invalid data type {type(data)}. Should be xr.Dataset or xr.DataArray')

    #     if isinstance(data, xr.DataArray):
    #         # convert DataArray to Dataset to plot a single polarization
    #         data = data.to_dataset()

    #     if polarizations is None:
    #         polarizations = list(data.data_vars) if isinstance(data, xr.Dataset) else list(data[0].data_vars)
    #     elif isinstance(polarizations, str):
    #         polarizations = [polarizations]
    #     #print ('polarizations', polarizations)

    #     # process polarizations one by one
    #     fgs = []
    #     for pol in polarizations:
    #         fg = plot_polarization(data, polarization=pol)
    #         fgs.append(fg)
    #     return fgs

    def plot_displacement_mm(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Displacement, [mm]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        data_los_mm = self.los_displacement_mm(data)
        return self.plot_dataset(data_los_mm, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, _size=_size)

    def plot_displacement(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Displacement, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        return self.plot_dataset(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, _size=_size)

    def plot_phase(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Phase, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        return self.plot_dataset(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, _size=_size)

    def plot_interferogram(self, data, polarizations=None,
                           cmap='gist_rainbow_r',
                           caption='Phase, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        import numpy as np
        return self.plot_dataset(data, polarizations,
                        cmap=cmap, vmin=-np.pi, vmax=np.pi, quantile=None, symmetrical=False,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, _size=_size)

    def plot_correlation(self, data, polarizations=None,
                         cmap='auto', vmin=0, vmax=1, quantile=None, symmetrical=False,
                         caption='Correlation', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None):
        import matplotlib.colors as mcolors
        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
        return self.plot_dataset(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=False,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, _size=_size)

    def plot_stack_correlation(self, data, threshold=None, caption='Correlation Stack', bins=100, cmap='auto'):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            data = data.unstack('stack')

        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
    
        data_flatten = data.values.ravel()
    
        fig, axs = plt.subplots(1, 2)
    
        ax2 = axs[0].twinx()
        axs[0].hist(data_flatten, range=(0, 1), bins=bins, density=False, cumulative=False, color='gray', edgecolor='black', alpha=0.5)
        ax2.hist(data_flatten, range=(0, 1), bins=bins, density=False, cumulative=True, color='orange', edgecolor='black', alpha=0.25)
    
        mean_value = np.nanmean(data_flatten)
        axs[0].axvline(mean_value, color='b', label=f'Average {mean_value:0.3f}')
        median_value = np.nanmedian(data_flatten)
        axs[0].axvline(median_value, color='g', label=f'Median {median_value:0.3f}')
        axs[0].set_xlim([0, 1])
        axs[0].grid()
        axs[0].set_xlabel('Correlation')
        axs[0].set_ylabel('Count')
        ax2.set_ylabel('Cumulative Count', color='orange')
    
        axs[0].set_title('Histogram')
        if threshold is not None:
            data.where(data > threshold).plot.imshow(cmap=cmap, vmin=0, vmax=1, ax=axs[1])
            axs[1].set_title(f'Threshold = {threshold:0.3f}')
            axs[0].axvline(threshold, linestyle='dashed', color='black', label=f'Threshold {threshold:0.3f}')
        else:
            data.where(data).plot.imshow(cmap=cmap, vmin=0, vmax=1, ax=axs[1])
        axs[0].legend()
        plt.suptitle(caption)
        plt.tight_layout()


    @staticmethod
    def burst_offset(intfs: 'BatchWrap',
                     method: str = 'median',
                     debug: bool = False) -> dict[str, float]:
        """
        Estimate phase offsets for all bursts using global least-squares optimization.

        Finds all overlapping burst pairs based on their y,x tile extents, computes
        pairwise phase differences in overlap regions, then solves a global linear
        system to find consistent offsets for all bursts. Handles disconnected groups
        of bursts separately.

        Uses row-wise gradient-based alignment: computes mean phase per row (range direction),
        then takes median across rows. This is robust to ionospheric ramps which vary along
        azimuth (y) but maintain consistent fringes along range (x).

        Parameters
        ----------
        intfs : BatchWrap
            Batch of wrapped interferograms, optionally pre-filtered by coherence.
            Example: dss.where(corrs.sel(dss).reindex_like(dss, method='nearest') > 0.5)
        method : str, optional
            Estimation method for pairwise offsets:
            - 'median' (default): Robust median with MAD-based outlier rejection
            - 'mean': Circular mean - faster but less robust
        debug : bool, optional
            Print debug information for each burst pair. Default is False.

        Returns
        -------
        dict[str, float]
            Dictionary mapping burst IDs to phase offsets (radians).
            Can be used directly with BatchWrap arithmetic: intfs - offsets

        Examples
        --------
        >>> offsets = Stack.burst_offset(dss_filtered)
        >>> intfs_aligned = dss - offsets

        Notes
        -----
        Works for any set of bursts: sequential, cross-subswath, cross-scene, or
        spatially disconnected groups. Each disconnected group is solved independently
        with its own reference (first burst in sorted order has offset=0).
        """
        import numpy as np
        import dask
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        from scipy.sparse.csgraph import connected_components

        # Minimum valid pixels for reliable phase offset estimation
        MIN_OVERLAP_PIXELS = 50

        # Minimum pixels per row for row-wise phase estimation
        MIN_ROW_PIXELS = 10

        # Minimum valid rows for row-wise processing
        MIN_VALID_ROWS = 5

        # Minimum samples after outlier rejection to accept the result
        MIN_INLIER_SAMPLES = 10

        # MAD multiplier for outlier rejection (2.5 = ~99% of normal distribution)
        MAD_OUTLIER_THRESHOLD = 2.5

        # Output precision (decimal places for rounding)
        OUTPUT_PRECISION = 3

        # Circular statistics helper functions
        def circ_wrap(x):
            """Wrap to [-π, π)"""
            return (x + np.pi) % (2*np.pi) - np.pi

        def circ_diff(a, center):
            """Circular difference: shortest arc from center to each point"""
            return circ_wrap(a - center)

        def circ_mean(a):
            """Circular mean"""
            return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))

        def circ_mad(a, center):
            """Circular MAD: median of absolute circular deviations from given center"""
            return np.median(np.abs(circ_diff(a, center)))

        # Step 1: Collect burst extents for fast overlap lookup
        ids = sorted(intfs.keys())
        n_bursts = len(ids)
        id_to_idx = {bid: i for i, bid in enumerate(ids)}

        if debug:
            print(f'Collecting extents for {n_bursts} bursts...', flush=True)

        extents = {}  # burst_id -> (y_min, y_max, x_min, x_max)

        for bid in ids:
            ds = intfs[bid]
            pol = list(ds.data_vars)[0] if hasattr(ds, 'data_vars') else None
            da = ds[pol] if pol else ds
            if 'pair' in da.dims:
                da = da.isel(pair=0)

            y_coords = da.coords['y'].values
            x_coords = da.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())

        def extents_overlap(e1, e2):
            """Check if two extents overlap"""
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap

        def process_phase_offset(phase, id1, id2):
            """Process a computed phase difference to extract offset.

            Returns: (id1, id2, offset, n_used, reject_reason)
            """
            # Get all valid pixels first
            all_valid = phase.values.ravel()
            all_valid = all_valid[np.isfinite(all_valid)]
            n_total = len(all_valid)

            if n_total < MIN_OVERLAP_PIXELS:
                return (id1, id2, None, 0, f"too few pixels ({n_total} < {MIN_OVERLAP_PIXELS})")

            # Row-wise alignment for robustness to ionospheric ramps
            if 'y' not in phase.dims or 'x' not in phase.dims:
                return (id1, id2, None, 0, "missing y/x dimensions")

            x_coords = phase.coords['x'].values

            # Collect row data: mean phase per row
            row_phases = []
            row_weights = []

            for y_idx in range(phase.shape[0]):
                row = phase.values[y_idx, :]
                valid_mask = np.isfinite(row)
                n_valid_in_row = np.sum(valid_mask)
                if n_valid_in_row >= MIN_ROW_PIXELS:
                    phase_valid = row[valid_mask]
                    # Unwrap within row for mean calculation
                    phase_unwrapped = np.unwrap(phase_valid)
                    row_mean = circ_wrap(np.mean(phase_unwrapped))
                    row_phases.append(row_mean)
                    row_weights.append(n_valid_in_row)

            n_valid_rows = len(row_phases)

            if n_valid_rows < MIN_VALID_ROWS:
                return (id1, id2, None, 0, f"too few valid rows ({n_valid_rows} < {MIN_VALID_ROWS})")

            a = np.array(row_phases)
            weights = np.array(row_weights)

            # Wrap to [-π, π)
            a = circ_wrap(a)

            # Outlier rejection using MAD
            if method == 'median':
                offset_initial = np.median(a)
                mad = circ_mad(a, offset_initial)
                if mad > 0:
                    inliers = np.abs(circ_diff(a, offset_initial)) <= MAD_OUTLIER_THRESHOLD * mad
                    n_inliers = int(np.sum(inliers))
                    if n_inliers >= MIN_INLIER_SAMPLES:
                        a = a[inliers]
                        weights = weights[inliers]

            n_valid = int(np.sum(weights))

            # Compute offset
            if method == 'median':
                # Weighted median
                sorted_idx = np.argsort(a)
                cumsum = np.cumsum(weights[sorted_idx])
                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                offset = a[sorted_idx[median_idx]]
            else:
                offset = circ_mean(a)

            return (id1, id2, circ_wrap(offset), n_valid, None)

        # Step 2: Find all overlapping pairs
        if debug:
            print(f'Finding overlapping pairs...', flush=True)

        all_overlap_pairs = []
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                e2 = extents[id2]
                if extents_overlap(e1, e2):
                    all_overlap_pairs.append((id1, id2))

        # Build all lazy phase differences (dask graphs)
        lazy_diffs = []
        for id1, id2 in all_overlap_pairs:
            i1 = intfs[id1]
            i2 = intfs[id2]

            pol = list(i1.data_vars)[0] if hasattr(i1, 'data_vars') else None
            if pol:
                i1 = i1[pol]
                i2 = i2[pol]

            if 'pair' in i1.dims:
                i1 = i1.isel(pair=0)
            if 'pair' in i2.dims:
                i2 = i2.isel(pair=0)

            phase_diff = i2 - i1
            lazy_diffs.append(phase_diff)

        if debug:
            print(f'Computing {len(all_overlap_pairs)} phase differences...', flush=True)

        # Compute all phase differences at once - dask schedules efficiently
        computed_diffs = dask.compute(*lazy_diffs)

        # Process computed results
        pairs_offset = []
        for (id1, id2), phase in zip(all_overlap_pairs, computed_diffs):
            id1, id2, offset, n_used, reject_reason = process_phase_offset(phase, id1, id2)
            if offset is None:
                if debug:
                    print(f'{id1} <-> {id2}: REJECTED - {reject_reason}', flush=True)
                continue

            weight = np.sqrt(n_used)
            pairs_offset.append((id1, id2, offset, weight))
            if debug:
                print(f'{id1} <-> {id2}: offset={offset:.3f}, weight={weight:.1f}', flush=True)

        if debug:
            print(f'Found {len(pairs_offset)} valid overlapping pairs', flush=True)

        if len(pairs_offset) == 0:
            return {bid: 0.0 for bid in ids}

        # Step 3: Build connectivity graph and find connected components
        adjacency = sparse.lil_matrix((n_bursts, n_bursts))
        for id1, id2, _, _ in pairs_offset:
            i, j = id_to_idx[id1], id_to_idx[id2]
            adjacency[i, j] = 1
            adjacency[j, i] = 1

        n_components, labels = connected_components(adjacency.tocsr(), directed=False)

        if debug:
            print(f'Found {n_components} connected component(s)', flush=True)

        # Step 4: Solve for offsets in each connected component
        offsets_out = {}

        for comp in range(n_components):
            comp_indices = np.where(labels == comp)[0]
            comp_ids = [ids[i] for i in comp_indices]
            comp_id_to_local = {bid: i for i, bid in enumerate(comp_ids)}
            n_comp = len(comp_ids)

            if debug:
                print(f'Component {comp}: {n_comp} bursts', flush=True)

            if n_comp == 1:
                offsets_out[comp_ids[0]] = 0.0
                continue

            comp_pairs = [(id1, id2, off, w) for id1, id2, off, w in pairs_offset
                          if id1 in comp_id_to_local and id2 in comp_id_to_local]

            if len(comp_pairs) == 0:
                for bid in comp_ids:
                    offsets_out[bid] = 0.0
                continue

            n_pairs = len(comp_pairs)
            A = sparse.lil_matrix((n_pairs + 1, n_comp))
            b = np.zeros(n_pairs + 1)
            W = np.zeros(n_pairs + 1)

            for k, (id1, id2, off, w) in enumerate(comp_pairs):
                i = comp_id_to_local[id1]
                j = comp_id_to_local[id2]
                A[k, i] = -1
                A[k, j] = +1
                b[k] = off
                W[k] = w

            constraint_weight = np.sum(W[:-1]) * 100 if np.sum(W[:-1]) > 0 else 1e6
            A[n_pairs, 0] = 1
            b[n_pairs] = 0
            W[n_pairs] = constraint_weight

            sqrt_W = np.sqrt(W)
            result = lsqr(sparse.diags(sqrt_W) @ A.tocsr(), sqrt_W * b)

            for i, bid in enumerate(comp_ids):
                offsets_out[bid] = round(float(circ_wrap(result[0][i])), OUTPUT_PRECISION)

            if debug:
                for bid in comp_ids:
                    print(f'  {bid}: offset={offsets_out[bid]:.{OUTPUT_PRECISION}f}', flush=True)

        return offsets_out

    @staticmethod
    def burst_polyfit(intfs: 'BatchWrap',
                      degree: int = 0,
                      method: str = 'median',
                      debug: bool = False) -> dict:
        """
        Estimate per-burst polynomial coefficients using overlap-based least-squares.

        Fits polynomial corrections (offset or offset+ramp) to each burst by analyzing
        phase differences in overlapping regions. Uses global least-squares optimization
        to find consistent coefficients across all bursts.

        Parameters
        ----------
        intfs : BatchWrap
            Batch of wrapped interferograms, optionally pre-filtered by coherence.
            Can have multiple pairs (pair dimension).
        degree : int, optional
            Polynomial degree:
            - 0 (default): Estimate offsets only.
            - 1: Estimate linear ramp (in x/range direction).
        method : str, optional
            Estimation method: 'median' (robust) or 'mean' (faster).
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        dict
            For single pair (no pair dimension):
                degree=0: {burst_id: offset}
                degree=1: {burst_id: [ramp, intercept]}
            For multiple pairs:
                degree=0: {burst_id: [offset_pair0, offset_pair1, ...]}
                degree=1: {burst_id: [[ramp0, intercept0], [ramp1, intercept1], ...]}

        Examples
        --------
        >>> # 3-step alignment for best results (0.028 rad discrepancy):
        >>> # Step 1: Estimate offsets
        >>> offsets1 = Stack.burst_polyfit(dss, degree=0)
        >>> intfs1 = dss - offsets1
        >>> # Step 2: Estimate ramps
        >>> ramps = Stack.burst_polyfit(intfs1, degree=1)
        >>> intfs2 = intfs1 - intfs1.polyval(ramps)
        >>> # Step 3: Re-estimate offsets
        >>> offsets2 = Stack.burst_polyfit(intfs2, degree=0)
        >>> # Combine coefficients (for single pair)
        >>> coeffs = {b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]] for b in offsets1}
        >>> aligned = dss - dss.polyval(coeffs)
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
        ids = sorted(intfs.keys())
        n_bursts = len(ids)
        id_to_idx = {bid: i for i, bid in enumerate(ids)}

        # Detect number of pairs
        sample_ds = intfs[ids[0]]
        pol = list(sample_ds.data_vars)[0] if hasattr(sample_ds, 'data_vars') else None
        sample_da = sample_ds[pol] if pol else sample_ds
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        if debug:
            print(f'burst_polyfit(degree={degree}): {n_bursts} bursts, {n_pairs} pair(s)', flush=True)

        extents = {}
        x_centers = {}

        for bid in ids:
            ds = intfs[bid]
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
        all_overlap_pairs = []
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                if extents_overlap(e1, extents[id2]):
                    all_overlap_pairs.append((id1, id2))

        if debug:
            print(f'Found {len(all_overlap_pairs)} overlapping burst pairs', flush=True)

        # Build all lazy phase differences (dask graphs)
        jobs = []
        lazy_diffs = []
        for id1, id2 in all_overlap_pairs:
            i1 = intfs[id1]
            i2 = intfs[id2]

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

        # If single pair, return simple dict
        if n_pairs == 1 and not has_pair_dim:
            return results_per_pair[0]

        # Multiple pairs: combine into list per burst
        combined = {}
        for bid in ids:
            if degree == 0:
                combined[bid] = [results_per_pair[p][bid] for p in range(n_pairs)]
            else:
                combined[bid] = [results_per_pair[p][bid] for p in range(n_pairs)]

        return combined

    @staticmethod
    def burst_discrepancy(intfs: 'BatchWrap', debug: bool = False, n_jobs: int = -1) -> float | list[float]:
        """
        Measure phase offset discrepancy across all burst overlaps.

        Computes the weighted mean of absolute median phase differences
        across all overlapping regions. After offset correction with burst_polyfit(),
        these median differences should be close to zero.

        Parameters
        ----------
        intfs : BatchWrap
            Batch of wrapped interferograms, optionally pre-filtered by coherence.
            Can have multiple pairs (pair dimension).
        debug : bool, optional
            Print debug information for each overlap. Default is False.
        n_jobs : int, optional
            Number of parallel jobs for overlap processing. Default is -1 (all CPUs).

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
        >>> before = Stack.burst_discrepancy(intfs)
        >>> offsets = Stack.burst_polyfit(intfs, degree=0)
        >>> after = Stack.burst_discrepancy(intfs - intfs.polyval(offsets))
        >>> print(f'Discrepancy reduced from {before} to {after}')
        """
        import numpy as np
        import dask

        def circ_wrap(x):
            """Wrap to [-π, π)"""
            return (x + np.pi) % (2*np.pi) - np.pi

        # Collect burst extents and detect pair dimension
        ids = sorted(intfs.keys())

        sample_ds = intfs[ids[0]]
        pol = list(sample_ds.data_vars)[0] if hasattr(sample_ds, 'data_vars') else None
        sample_da = sample_ds[pol] if pol else sample_ds
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        extents = {}
        for bid in ids:
            ds = intfs[bid]
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
            print(f'burst_discrepancy: found {len(overlap_pairs)} overlap pairs, {n_pairs} pair(s)', flush=True)

        # Build all lazy phase differences (dask graphs)
        jobs = []
        lazy_diffs = []
        for id1, id2 in overlap_pairs:
            i1 = intfs[id1]
            i2 = intfs[id2]

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
        subswath_stats = {}  # {(subswath, pair_idx): {'sum': float, 'weight': float, 'count': int}}

        for result in results:
            if result is None:
                continue
            pair_idx, abs_discrepancy, weight, id1, id2, median_diff = result
            weighted_sums[pair_idx] += abs_discrepancy * weight
            total_weights[pair_idx] += weight

            # Extract subswath info for debug stats
            if debug:
                # Determine subswath from burst IDs
                sw1 = sw2 = 'unknown'
                if '_IW' in id1:
                    sw1 = 'IW' + id1.split('_IW')[1][0]
                if '_IW' in id2:
                    sw2 = 'IW' + id2.split('_IW')[1][0]

                # Categorize: same subswath or cross-subswath
                if sw1 == sw2:
                    sw_key = sw1
                else:
                    sw_key = f'{sw1}-{sw2}'

                key = (sw_key, pair_idx)
                if key not in subswath_stats:
                    subswath_stats[key] = {'sum': 0.0, 'weight': 0.0, 'count': 0}
                subswath_stats[key]['sum'] += abs_discrepancy * weight
                subswath_stats[key]['weight'] += weight
                subswath_stats[key]['count'] += 1

        discrepancies = []
        for p in range(n_pairs):
            if total_weights[p] == 0:
                discrepancies.append(0.0)
            else:
                discrepancies.append(round(weighted_sums[p] / total_weights[p], 3))

        if debug:
            print(f'Overall discrepancy: {discrepancies}', flush=True)
            # Print per-subswath stats (only for pair_idx=0 to avoid clutter)
            print('Per-subswath discrepancy (pair 0):', flush=True)
            for (sw, pair_idx), stats in sorted(subswath_stats.items()):
                if pair_idx == 0 and stats['weight'] > 0:
                    sw_disc = stats['sum'] / stats['weight']
                    print(f'  {sw}: {sw_disc:.3f} rad ({stats["count"]} overlaps)', flush=True)

        # Return single value for single pair, list for multiple
        if n_pairs == 1 and not has_pair_dim:
            return discrepancies[0]
        return discrepancies
