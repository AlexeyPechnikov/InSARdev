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
                     debug: bool = False) -> tuple[dict[str, float], dict[str, float]]:
        """
        Estimate cumulative phase offsets for all bursts to align them.
        
        Uses row-wise gradient-based alignment: computes mean phase per row (range direction),
        then takes median across rows. This is robust to ionospheric ramps which vary along
        azimuth (y) but maintain consistent fringes along range (x).
        
        Parameters
        ----------
        intfs : BatchWrap
            Batch of wrapped interferograms, optionally pre-filtered by coherence.
            Example: dss.where(corrs.sel(dss).reindex_like(dss, method='nearest') > 0.5)
        method : str, optional
            Estimation method:
            - 'median' (default): Robust median with MAD-based outlier rejection
            - 'mean': Circular mean - faster but less robust
        debug : bool, optional
            Print debug information for each burst pair. Default is False.
        
        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            Two dictionaries:
            - offsets: burst IDs to cumulative offsets (radians). First burst has offset 0.
            - confidence: burst IDs to MAD (Median Absolute Deviation) in radians. Lower is better.
              First burst has confidence 0 (reference).
            Can be used directly with BatchWrap arithmetic: intfs - offsets
        
        Examples
        --------
        >>> # Filter by coherence and estimate offsets
        >>> dss_filtered = dss.where(corrs.sel(dss).reindex_like(dss, method='nearest') > 0.5)
        >>> offsets, confidence = Stack.burst_offset(dss_filtered)
        >>> intfs_aligned = dss - offsets
        >>> # Check quality - lower std means more reliable offset
        >>> print(confidence)
        """
        import numpy as np
        
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
        
        def circ_std(a, center):
            """Circular standard deviation"""
            diff = circ_diff(a, center)
            R = np.sqrt(np.mean(np.sin(diff))**2 + np.mean(np.cos(diff))**2)
            return np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1.0)))
        
        # Get sorted burst IDs
        ids = sorted(intfs.keys())
        
        # First burst has zero offset and zero std (reference)
        offsets = {ids[0]: 0.0}
        confidence = {ids[0]: 0.0}
        cumulative = 0.0
        
        for id1, id2 in zip(ids[:-1], ids[1:]):
            # Get phase data for both bursts
            i1 = intfs[id1]
            i2 = intfs[id2]
            
            # Get first data variable (VV, VH, etc.)
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
            # Force computation if lazy
            if hasattr(phase, 'compute'):
                phase = phase.compute()
            
            # Row-wise alignment: compute mean phase per row (along x/range),
            # then take median across rows. Robust to ionospheric y-ramps.
            if 'y' in phase.dims and 'x' in phase.dims:
                row_means = []
                row_weights = []
                for y_idx in range(phase.shape[0]):
                    row = phase.values[y_idx, :]
                    row = row[np.isfinite(row)]
                    if len(row) > 10:
                        # Unwrap row to handle phase wrapping within the row
                        row_unwrapped = np.unwrap(row)
                        # Take mean of unwrapped values, then wrap back
                        row_mean = circ_wrap(np.mean(row_unwrapped))
                        row_means.append(row_mean)
                        row_weights.append(len(row))
                
                if len(row_means) > 0:
                    a = np.array(row_means)
                    weights = np.array(row_weights)
                else:
                    a = np.array([])
                    weights = np.array([])
            else:
                a = phase.values.ravel()
                a = a[np.isfinite(a)]
                weights = None
            
            if a.size == 0:
                offset, conf = np.nan, np.nan
                if debug:
                    print(f'{id1} -> {id2}: NO OVERLAP (0 valid rows)', flush=True)
            else:
                # Wrap to [-π, π)
                a = circ_wrap(a)
                
                if debug:
                    print(f'{id1} -> {id2}: {a.size} valid rows, range [{a.min():.3f}, {a.max():.3f}]', flush=True)
                
                if method == 'median':
                    # Initial estimate
                    offset = np.median(a)
                    
                    # Outlier rejection using MAD
                    mad = circ_mad(a, offset)
                    if mad > 0:
                        inliers = np.abs(circ_diff(a, offset)) <= 2.5 * mad
                        if np.sum(inliers) > 10:
                            if debug:
                                print(f'  Outlier rejection: {a.size} -> {np.sum(inliers)} rows, MAD={mad:.3f}', flush=True)
                            a_inliers = a[inliers]
                            if weights is not None:
                                w_inliers = weights[inliers]
                                # Weighted median
                                sorted_idx = np.argsort(a_inliers)
                                cumsum = np.cumsum(w_inliers[sorted_idx])
                                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                                offset = a_inliers[sorted_idx[median_idx]]
                            else:
                                offset = np.median(a_inliers)
                            # Compute MAD on inliers as confidence (lower = better)
                            a = a_inliers
                    # Use MAD as confidence metric (more robust than std)
                    conf = circ_mad(a, offset)
                else:
                    offset = circ_mean(a)
                    conf = circ_mad(a, offset)
            
            # Wrap offset to [-π, π] and accumulate
            offset = circ_wrap(offset)
            cumulative += offset
            offsets[id2] = cumulative
            confidence[id2] = round(conf, 2)
            
            if debug:
                print(f'{id1} -> {id2}: offset={offset:.3f}, MAD={conf:.3f}, cumulative={cumulative:.3f}', flush=True)
        
        return offsets, confidence

    @staticmethod
    def burst_offsets(intfs_list: list['BatchWrap'],
                      method: str = 'median',
                      debug: bool = False) -> tuple[dict[str, float], dict[str, float]]:
        """
        Estimate cumulative phase offsets using multiple coherence thresholds and select the best.
        
        Runs burst_offset on each BatchWrap in the list and selects the complete result
        from the threshold with the lowest total MAD (sum of MAD across all transitions).
        
        Parameters
        ----------
        intfs_list : list[BatchWrap]
            List of BatchWrap objects, typically filtered at different coherence thresholds.
            Example: [dss.where(corr > 0.5), dss.where(corr > 0.6), dss.where(corr > 0.7)]
        method : str, optional
            Estimation method passed to burst_offset. Default is 'median'.
        debug : bool, optional
            Print debug information. Default is False.
        
        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            Two dictionaries:
            - offsets: burst IDs to cumulative offsets (radians) from the best threshold.
            - confidence: burst IDs to MAD (radians) for each transition.
        
        Examples
        --------
        >>> intfs1 = dss.where(corrs.sel(dss).reindex_like(dss, method='nearest') > 0.5)
        >>> intfs2 = dss.where(corrs.sel(dss).reindex_like(dss, method='nearest') > 0.6)
        >>> intfs3 = dss.where(corrs.sel(dss).reindex_like(dss, method='nearest') > 0.7)
        >>> offsets, confidence = Stack.burst_offsets([intfs1, intfs2, intfs3])
        >>> intfs_aligned = dss - offsets
        """
        import numpy as np
        from .Stack_plot import Stack_plot as Stack
        
        if not intfs_list:
            raise ValueError("intfs_list cannot be empty")
        
        # Run burst_offset for each input, also track number of valid rows
        all_results = []
        all_nrows = []
        for idx, intfs in enumerate(intfs_list):
            if debug:
                print(f'=== Processing threshold {idx + 1}/{len(intfs_list)} ===', flush=True)
            offsets, confidence = Stack.burst_offset(intfs, method=method, debug=debug)
            all_results.append((offsets, confidence))
            if debug:
                print(flush=True)
        
        # Get all burst IDs (sorted)
        ids = list(all_results[0][0].keys())
        
        # Filter out results with NaN offsets (insufficient data)
        valid_results = []
        for idx, (offsets, confidence) in enumerate(all_results):
            has_nan = any(np.isnan(offsets[burst_id]) for burst_id in ids)
            if not has_nan:
                valid_results.append((idx, offsets, confidence))
        
        if not valid_results:
            # All results have NaN, return the first one
            best_offsets, best_confidence = all_results[0]
            best_idx = 0
        elif len(valid_results) == 1:
            # Only one valid result
            best_idx, best_offsets, best_confidence = valid_results[0]
        else:
            # Use median voting: for each burst, take median offset across all valid thresholds
            # This is robust to outliers from both too-low and too-high coherence thresholds
            median_offsets = {ids[0]: 0.0}
            median_confidence = {ids[0]: 0.0}
            
            for burst_id in ids[1:]:
                # Collect offsets from all valid thresholds
                offsets_list = [offsets[burst_id] for _, offsets, _ in valid_results]
                # Collect corresponding MADs
                mads_list = [confidence[burst_id] for _, _, confidence in valid_results]
                
                # Use median offset
                median_offsets[burst_id] = float(np.median(offsets_list))
                # Use median MAD as confidence
                median_confidence[burst_id] = round(float(np.median(mads_list)), 2)
            
            best_offsets = median_offsets
            best_confidence = median_confidence
            best_idx = -1  # Indicates median voting was used
        
        # Compute total MAD for reporting
        total_mads = []
        for offsets, confidence in all_results:
            mads = [confidence[burst_id] for burst_id in ids[1:]]
            valid_mads = [m for m in mads if not np.isnan(m)]
            total = np.sum(valid_mads) if valid_mads else np.inf
            total_mads.append(total)
        
        if debug:
            print(f'=== Summary ===', flush=True)
            print(f'Total MAD per threshold: {[f"{m:.2f}" for m in total_mads]}', flush=True)
            print(f'Valid thresholds (no NaN): {[i+1 for i, _, _ in valid_results]}', flush=True)
            for idx, (offsets, confidence) in enumerate(all_results):
                marker = ' <-- SELECTED' if idx == best_idx else ''
                print(f'Threshold {idx + 1}{marker}:', flush=True)
                for burst_id in ids:
                    off = offsets[burst_id]
                    mad = confidence[burst_id]
                    off_str = f'{off:.3f}' if not np.isnan(off) else 'nan'
                    print(f'  {burst_id}: offset={off_str}, MAD={mad}', flush=True)
            if best_idx == -1:
                print(f'MEDIAN VOTING result:', flush=True)
                for burst_id in ids:
                    print(f'  {burst_id}: offset={best_offsets[burst_id]:.3f}, MAD={best_confidence[burst_id]}', flush=True)
        
        return best_offsets, best_confidence
