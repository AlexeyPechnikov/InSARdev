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
            Reference burst(s) in each connected group have offset 0.
            Can be used directly with BatchWrap arithmetic: intfs - offsets
        
        Examples
        --------
        >>> # Filter by coherence and estimate offsets
        >>> dss_filtered = dss.where(corrs.sel(dss).reindex_like(dss, method='nearest') > 0.5)
        >>> offsets = Stack.burst_offset(dss_filtered)
        >>> intfs_aligned = dss - offsets
        
        Notes
        -----
        Works for any set of bursts: sequential, cross-subswath, cross-scene, or
        spatially disconnected groups. Each disconnected group is solved independently
        with its own reference (first burst in sorted order has offset 0).
        """
        import numpy as np
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        from scipy.sparse.csgraph import connected_components
                
        # Minimum valid pixels for reliable phase offset estimation
        # 50 pixels is enough for statistical reliability
        MIN_OVERLAP_PIXELS = 50
        
        # Minimum pixels per row for row-wise phase estimation
        # 10 pixels provides enough samples for reliable row mean calculation
        MIN_ROW_PIXELS = 10
        
        # Minimum valid rows for row-wise processing
        # Need enough rows for median to be robust (5 rows minimum)
        # If fewer rows qualify, fall back to flat pixel approach
        MIN_VALID_ROWS = 5
        
        # Minimum samples after outlier rejection to accept the result
        MIN_INLIER_SAMPLES = 10
        
        # MAD multiplier for outlier rejection (2.5 = ~99% of normal distribution)
        MAD_OUTLIER_THRESHOLD = 2.5
        
        # Output precision (decimal places for rounding)
        # 3 decimal places = 0.001 rad =~ 0.01mm
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
            # Get first data variable
            pol = list(ds.data_vars)[0] if hasattr(ds, 'data_vars') else None
            if pol:
                da = ds[pol]
            else:
                da = ds
            # Select first pair if pair dimension exists
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            
            y_coords = da.coords['y'].values
            x_coords = da.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())
        
        # Step 2: Find all overlapping pairs and compute pairwise offsets
        if debug:
            print(f'Finding overlapping pairs...', flush=True)
        
        def extents_overlap(e1, e2):
            """Check if two extents overlap"""
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap
        
        def compute_pairwise_offset(id1, id2):
            """Compute phase offset between two bursts in their overlap region.
            
            Returns: (offset, n_used, debug_info) where debug_info is a dict with:
                - n_total: total valid pixels in overlap
                - n_rows: total rows in overlap
                - n_valid_rows: rows with >= MIN_ROW_PIXELS
                - n_rowwise: pixels captured by valid rows
                - n_inliers: rows after outlier rejection (if median method)
                - n_inlier_pixels: pixels in inlier rows
                - reject_reason: why overlap was rejected (if offset is None)
            """
            i1 = intfs[id1]
            i2 = intfs[id2]
            
            # Get first data variable
            pol = list(i1.data_vars)[0] if hasattr(i1, 'data_vars') else None
            if pol:
                i1 = i1[pol]
                i2 = i2[pol]
            
            # Select first pair if pair dimension exists
            if 'pair' in i1.dims:
                i1 = i1.isel(pair=0)
            if 'pair' in i2.dims:
                i2 = i2.isel(pair=0)
            
            # Compute phase difference in overlap region
            phase = i2 - i1
            if hasattr(phase, 'compute'):
                phase = phase.compute()
            
            # Initialize debug info
            info = {
                'n_total': 0,
                'n_rows': 0,
                'n_valid_rows': 0,
                'n_rowwise': 0,
                'n_inliers': 0,
                'n_inlier_pixels': 0,
                'reject_reason': None
            }
            
            # Get all valid pixels first
            all_valid = phase.values.ravel()
            all_valid = all_valid[np.isfinite(all_valid)]
            info['n_total'] = len(all_valid)
            
            if info['n_total'] < MIN_OVERLAP_PIXELS:
                info['reject_reason'] = f"too few pixels ({info['n_total']} < {MIN_OVERLAP_PIXELS})"
                return None, 0, info
            
            # Row-wise alignment for robustness to ionospheric ramps
            if 'y' not in phase.dims or 'x' not in phase.dims:
                info['reject_reason'] = "missing y/x dimensions"
                return None, 0, info
            
            info['n_rows'] = phase.shape[0]
            row_means = []
            row_weights = []
            for y_idx in range(phase.shape[0]):
                row = phase.values[y_idx, :]
                row = row[np.isfinite(row)]
                if len(row) >= MIN_ROW_PIXELS:
                    row_unwrapped = np.unwrap(row)
                    row_mean = circ_wrap(np.mean(row_unwrapped))
                    row_means.append(row_mean)
                    row_weights.append(len(row))
            
            info['n_valid_rows'] = len(row_means)
            info['n_rowwise'] = sum(row_weights)
            
            # Require enough valid rows for robust median
            if info['n_valid_rows'] < MIN_VALID_ROWS:
                info['reject_reason'] = f"too few valid rows ({info['n_valid_rows']} < {MIN_VALID_ROWS})"
                return None, 0, info
            
            a = np.array(row_means)
            weights = np.array(row_weights)
            n_valid = np.sum(row_weights)
            
            # Wrap to [-π, π)
            a = circ_wrap(a)
            
            if method == 'median':
                offset = np.median(a)
                mad = circ_mad(a, offset)
                if mad > 0:
                    inliers = np.abs(circ_diff(a, offset)) <= MAD_OUTLIER_THRESHOLD * mad
                    info['n_inliers'] = int(np.sum(inliers))
                    if info['n_inliers'] >= MIN_INLIER_SAMPLES:
                        a_inliers = a[inliers]
                        w_inliers = weights[inliers]
                        info['n_inlier_pixels'] = int(np.sum(w_inliers))
                        sorted_idx = np.argsort(a_inliers)
                        cumsum = np.cumsum(w_inliers[sorted_idx])
                        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                        offset = a_inliers[sorted_idx[median_idx]]
                        n_valid = info['n_inlier_pixels']
                    else:
                        # Not enough inliers but still use all rows
                        info['n_inliers'] = len(a)
                        info['n_inlier_pixels'] = int(np.sum(weights))
                else:
                    # MAD is 0, all values identical - use all
                    info['n_inliers'] = len(a)
                    info['n_inlier_pixels'] = int(np.sum(weights))
            else:
                offset = circ_mean(a)
                info['n_inliers'] = len(a)
                info['n_inlier_pixels'] = int(np.sum(weights))
            
            return circ_wrap(offset), n_valid, info
        
        # Find all pairs with extent overlap, then check for valid pixel overlap
        pairs = []  # List of (id1, id2, offset, weight)
        analyzed = set()
        
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                pair_key = (id1, id2)
                if pair_key in analyzed:
                    continue
                analyzed.add(pair_key)
                
                e2 = extents[id2]
                if not extents_overlap(e1, e2):
                    continue
                
                # Check valid pixel overlap and compute offset
                offset, n_used, info = compute_pairwise_offset(id1, id2)
                
                if offset is None:
                    if debug:
                        print(f'{id1} <-> {id2}: REJECTED - {info["reject_reason"]}', flush=True)
                        print(f'  pipeline: {info["n_total"]} pixels -> {info["n_rows"]} rows -> '
                              f'{info["n_valid_rows"]} valid rows (>={MIN_ROW_PIXELS} px) -> '
                              f'{info["n_rowwise"]} pixels', flush=True)
                    continue
                
                # Weight by overlap quality (sqrt of used pixels)
                weight = np.sqrt(n_used)
                pairs.append((id1, id2, offset, weight))
                
                if debug:
                    print(f'{id1} <-> {id2}: offset={offset:.3f}, weight={weight:.1f}', flush=True)
                    print(f'  pipeline: {info["n_total"]} pixels -> {info["n_rows"]} rows -> '
                          f'{info["n_valid_rows"]} valid rows (>={MIN_ROW_PIXELS} px) -> '
                          f'{info["n_rowwise"]} pixels -> {info["n_inliers"]} inlier rows -> '
                          f'{info["n_inlier_pixels"]} pixels', flush=True)
        
        if debug:
            print(f'Found {len(pairs)} overlapping pairs', flush=True)
        
        if len(pairs) == 0:
            # No overlaps found - return all zeros
            return {bid: 0.0 for bid in ids}
        
        # Step 3: Build connectivity graph and find connected components
        adjacency = sparse.lil_matrix((n_bursts, n_bursts))
        for id1, id2, _, _ in pairs:
            i, j = id_to_idx[id1], id_to_idx[id2]
            adjacency[i, j] = 1
            adjacency[j, i] = 1
        
        n_components, labels = connected_components(adjacency.tocsr(), directed=False)
        
        if debug:
            print(f'Found {n_components} connected component(s)', flush=True)
        
        # Step 4: Solve for offsets in each connected component
        offsets = {}
        
        for comp in range(n_components):
            # Get burst indices in this component
            comp_indices = np.where(labels == comp)[0]
            comp_ids = [ids[i] for i in comp_indices]
            comp_id_to_local = {bid: i for i, bid in enumerate(comp_ids)}
            n_comp = len(comp_ids)
            
            if debug:
                print(f'Component {comp}: {n_comp} bursts', flush=True)
            
            if n_comp == 1:
                # Single burst - offset is 0
                offsets[comp_ids[0]] = 0.0
                continue
            
            # Get pairs within this component
            comp_pairs = [(id1, id2, off, w) for id1, id2, off, w in pairs
                          if id1 in comp_id_to_local and id2 in comp_id_to_local]
            
            if len(comp_pairs) == 0:
                # No pairs - all offsets are 0
                for bid in comp_ids:
                    offsets[bid] = 0.0
                continue
            
            # Build linear system: A @ x = b
            # For each pair (i, j) with offset d_ij: x_j - x_i = d_ij
            # Design matrix A: row k has +1 at j, -1 at i
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
            
            # Add constraint: first burst has offset 0 (reference)
            # Weight should dominate other observations to enforce the constraint
            # Using sum of all pair weights * 100 ensures constraint is effectively exact
            constraint_weight = np.sum(W[:-1]) * 100 if np.sum(W[:-1]) > 0 else 1e6
            A[n_pairs, 0] = 1
            b[n_pairs] = 0
            W[n_pairs] = constraint_weight
            
            # Weighted least squares: minimize ||sqrt(W) @ (A @ x - b)||^2
            A_csr = A.tocsr()
            sqrt_W = np.sqrt(W)
            A_weighted = sparse.diags(sqrt_W) @ A_csr
            b_weighted = sqrt_W * b
            
            # Solve using LSQR
            result = lsqr(A_weighted, b_weighted)
            x = result[0]
            
            # Wrap offsets to [-π, π) and round to avoid floating point noise
            for i, bid in enumerate(comp_ids):
                offsets[bid] = round(float(circ_wrap(x[i])), OUTPUT_PRECISION)
            
            if debug:
                for bid in comp_ids:
                    print(f'  {bid}: offset={offsets[bid]:.{OUTPUT_PRECISION}f}', flush=True)
        
        return offsets

    @staticmethod
    def burst_discrepancy(intfs: 'BatchWrap', debug: bool = False) -> float:
        """
        Measure phase offset discrepancy across all burst overlaps.
        
        Computes the weighted mean of absolute median phase differences
        across all overlapping regions. After offset correction with burst_offset(),
        these median differences should be close to zero.
        
        Parameters
        ----------
        intfs : BatchWrap
            Batch of wrapped interferograms, optionally pre-filtered by coherence.
        debug : bool, optional
            Print debug information for each overlap. Default is False.
        
        Returns
        -------
        float
            Weighted mean absolute median phase discrepancy in radians.
            0.0 = perfect alignment, π = maximum discrepancy.
            Returns 0.0 if no valid overlaps found.
            
            Practical interpretation:
            
            - < 0.1 rad: Excellent alignment
            - 0.1 - 0.5 rad: Good alignment
            - 0.5 - 1.0 rad: Moderate misalignment
            - > 1.0 rad: Poor alignment
        
        Examples
        --------
        >>> # Compare before and after alignment
        >>> before = Stack.burst_discrepancy(intfs)
        >>> offsets = Stack.burst_offset(intfs)
        >>> after = Stack.burst_discrepancy(intfs - offsets)
        >>> print(f'Discrepancy reduced from {before:.3f} to {after:.3f} rad')
        """
        import numpy as np
        
        def circ_wrap(x):
            """Wrap to [-π, π)"""
            return (x + np.pi) % (2*np.pi) - np.pi
        
        def circ_mean(phases):
            """Circular mean"""
            return np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))
        
        # Collect burst extents
        ids = sorted(intfs.keys())
        
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
        
        # Compute weighted mean absolute phase difference across all overlaps
        total_weight = 0.0
        weighted_discrepancy_sum = 0.0
        
        analyzed = set()
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                pair_key = (id1, id2)
                if pair_key in analyzed:
                    continue
                analyzed.add(pair_key)
                
                e2 = extents[id2]
                if not extents_overlap(e1, e2):
                    continue
                
                # Get phase difference in overlap
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
                if hasattr(phase_diff, 'compute'):
                    phase_diff = phase_diff.compute()
                
                valid = phase_diff.values.ravel()
                valid = valid[np.isfinite(valid)]
                
                if len(valid) == 0:
                    continue
                
                # Wrap to [-π, π)
                valid = circ_wrap(valid)
                
                # median phase difference is the offset estimation
                median_diff = np.median(valid)
                abs_discrepancy = np.abs(circ_wrap(median_diff))
                weight = len(valid)
                
                weighted_discrepancy_sum += abs_discrepancy * weight
                total_weight += weight
                
                if debug:
                    print(f'{id1} <-> {id2}: median_diff={median_diff:.3f} rad, weight={weight}', flush=True)
        
        if total_weight == 0:
            return 0.0
        
        discrepancy = weighted_discrepancy_sum / total_weight
        
        if debug:
            print(f'Total discrepancy: {discrepancy:.4f} rad (from {total_weight:.0f} pixels)', flush=True)
        
        return round(discrepancy, 3)
