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

    # def plot_phase(self, data, caption='Phase, [rad]',
    #                quantile=None, vmin=None, vmax=None, symmetrical=False,
    #                cmap='turbo', aspect=None, **kwargs):
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt

    #     if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
    #         data = data.unstack('stack')

    #     if quantile is not None:
    #         assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

    #     if quantile is not None:
    #         vmin, vmax = np.nanquantile(data, quantile)

    #     # define symmetrical boundaries
    #     if symmetrical is True and vmax > 0:
    #         minmax = max(abs(vmin), vmax)
    #         vmin = -minmax
    #         vmax =  minmax

    #     plt.figure()
    #     data.plot.imshow(vmin=vmin, vmax=vmax, cmap=cmap)
    #     #self.plot_AOI(**kwargs)
    #     #self.plot_POI(**kwargs)
    #     if aspect is not None:
    #         plt.gca().set_aspect(aspect)
    #     plt.title(caption)

    # def plot_phases(self, data, caption='Phase, [rad]', cols=4, size=4, nbins=5, aspect=1.2, y=1.05,
    #                 quantile=None, vmin=None, vmax=None, symmetrical=False, **kwargs):
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt

    #     if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
    #         data = data.unstack('stack')

    #     if quantile is not None:
    #         assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

    #     if quantile is not None:
    #         vmin, vmax = np.nanquantile(data, quantile)

    #     # define symmetrical boundaries
    #     if symmetrical is True and vmax > 0:
    #         minmax = max(abs(vmin), vmax)
    #         vmin = -minmax
    #         vmax =  minmax

    #     # multi-plots ineffective for linked lazy data
    #     fg = data.plot.imshow(
    #         col='pair',
    #         col_wrap=cols, size=size, aspect=aspect,
    #         vmin=vmin, vmax=vmax, cmap='turbo'
    #     )
    #     #fg.set_axis_labels('Range', 'Azimuth')
    #     fg.set_ticks(max_xticks=nbins, max_yticks=nbins)
    #     fg.fig.suptitle(caption, y=y)
        
    #     #self.plots_AOI(fg, **kwargs)
    #     #self.plots_POI(fg, **kwargs)

    # def plot_interferogram(self, data, caption='Phase, [rad]', cmap='gist_rainbow_r', aspect=None, **kwargs):
    #     import xarray as xr
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt

    #     if isinstance(data, xr.Dataset):
    #         data = data.phase

    #     if data.dims == ('pair', 'y', 'x'):
    #         data = data.isel(pair=0)

    #     if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
    #         data = data.unstack('stack')

    #     plt.figure()
    #     self.wrap(self.interferogram(data) if np.issubdtype(data.dtype, np.complexfloating) else data)\
    #         .plot.imshow(vmin=-np.pi, vmax=np.pi, cmap=cmap)
    #     #self.plot_AOI(**kwargs)
    #     #self.plot_POI(**kwargs)
    #     if aspect is not None:
    #         plt.gca().set_aspect(aspect)
    #     plt.title(caption)

    def plot_stack(self, data, polarizations,
                   cmap, vmin, vmax, quantile, symmetrical,
                   caption, cols, rows, size, nbins, aspect, y, wrap, _size=None, **kwargs):
        import xarray as xr
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import warnings
        # supress Dask warning "RuntimeWarning: invalid value encountered in divide"
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # no data means no plot and no error
        if data is None:
            return

        assert isinstance(data, (dict, list, tuple, xr.Dataset, xr.DataArray)), 'ERROR: data should be a dict or list or tuple or Dataset or DataArray'

        # screen size in pixels (width, height) to estimate reasonable number pixels per plot
        # this is quite large to prevent aliasing on 600dpi plots without additional processing
        if _size is None:
            _size = (8000,4000)

        def plot_polarization(data, polarization):

            if isinstance(data, dict):
                data = list(data.values())

            if isinstance(data, xr.Dataset):
                stackvar = list(data.dims)[0]
                da = data[polarization].isel({stackvar: slice(0, rows)})
            else:
                stackvar = list(data[0].dims)[0]
                das = [da[polarization].isel({stackvar: slice(0, rows)}) for da in data]
                da = self.to_dataset(das)
                del das

            if 'stack' in da.dims and isinstance(da.coords['stack'].to_index(), pd.MultiIndex):
                da = da.unstack('stack')
            
            # there is no reason to plot huge arrays much larger than screen size for small plots
            #print ('screen_size', screen_size)
            size_y, size_x = da.shape[1:]
            #print ('size_x, size_y', size_x, size_y)
            factor_y = int(np.round(size_y / (_size[1] / rows)))
            factor_x = int(np.round(size_x / (_size[0] / cols)))
            #print ('factor_x, factor_y', factor_x, factor_y)
            # coarsen and materialize data for all the calculations and plotting
            progressbar(da := da[:,::max(1, factor_y), ::max(1, factor_x)].persist(), desc=f'Computing {polarization}'.ljust(25))

            # calculate min, max when needed
            if quantile is not None:
                _vmin, _vmax = np.nanquantile(da, quantile)
            else:
                _vmin, _vmax = vmin, vmax
            # define symmetrical boundaries
            if symmetrical is True and _vmax > 0:
                minmax = max(abs(_vmin), _vmax)
                _vmin = -minmax
                _vmax =  minmax

            # multi-plots ineffective for linked lazy data
            fg = (self.wrap(da) if wrap else da).rename(caption)\
                .plot.imshow(
                col=stackvar,
                col_wrap=min(cols, da[stackvar].size), size=size, aspect=aspect,
                vmin=_vmin, vmax=_vmax, cmap=cmap
            )
            #fg.set_axis_labels('Range', 'Azimuth')
            fg.set_ticks(max_xticks=nbins, max_yticks=nbins)
            fg.fig.suptitle(f'{polarization} {caption}', y=y)            
            #self.plots_AOI(fg, **kwargs)
            #self.plots_POI(fg, **kwargs)

        if quantile is not None:
            assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

        if not isinstance(data, (xr.Dataset, xr.DataArray, (dict, list, tuple))):
            raise ValueError(f'ERROR: invalid data type {type(data)}. Should be xr.Dataset or xr.DataArray or list of xr.Dataset or xr.DataArray')

        if isinstance(data, xr.DataArray):
            # convert DataArray to Dataset to plot a single polarization
            data = data.to_dataset()
        elif isinstance(data, (list, tuple)) and not isinstance(data[0], xr.Dataset):
            # convert list of DataArray to list of Dataset to plot a single polarization
            data = [da.to_dataset() for da in data]
        elif isinstance(data, dict):
            # convert dict of DataArray to dict of Dataset to plot a single polarization
            data = {k: v.to_dataset() if not isinstance(v, xr.Dataset) else v for k, v in data.items()}

        if polarizations is None:
            polarizations = list(data.data_vars) if isinstance(data, xr.Dataset) else list(data[0].data_vars)
        elif isinstance(polarizations, str):
            polarizations = [polarizations]
        #print ('polarizations', polarizations)

        # process polarizations one by one
        for pol in polarizations:
            plot_polarization(data, polarization=pol)

    def plot_displacement_mm(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Displacement, [mm]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None, **kwargs):
        data_los_mm = self.los_displacement_mm(data)
        self.plot_stack(data_los_mm, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, _size=_size, **kwargs)

    def plot_displacement(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Displacement, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None, **kwargs):
        self.plot_stack(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, _size=_size, **kwargs)

    def plot_phase(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Phase, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None, **kwargs):
        self.plot_stack(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, _size=_size, **kwargs)

    def plot_interferogram(self, data, polarizations=None,
                           cmap='gist_rainbow_r',
                           caption='Phase, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None, **kwargs):
        import numpy as np
        self.plot_stack(data, polarizations,
                        cmap=cmap, vmin=-np.pi, vmax=np.pi, quantile=None, symmetrical=False,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, _size=_size, **kwargs)

    def plot_correlation(self, data, polarizations=None,
                         cmap='auto', vmin=0, vmax=1, quantile=None, symmetrical=False,
                         caption='Correlation', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, _size=None, **kwargs):
        import matplotlib.colors as mcolors
        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
        self.plot_stack(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=False,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, _size=_size, **kwargs)

    # def plot_correlation(self, data, caption='Correlation', cmap='gray', aspect=None, **kwargs):
    #     import xarray as xr
    #     import pandas as pd
    #     import matplotlib.pyplot as plt

    #     if isinstance(data, xr.Dataset):
    #         data = data.correlation

    #     if data.dims == ('pair', 'y', 'x'):
    #         data = data.isel(pair=0)

    #     if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
    #         data = data.unstack('stack')

    #     plt.figure()
    #     data.plot.imshow(vmin=0, vmax=1, cmap=cmap)
    #     #self.plot_AOI(**kwargs)
    #     #self.plot_POI(**kwargs)
    #     if aspect is not None:
    #         plt.gca().set_aspect(aspect)
    #     plt.title(caption)

    def plot_stack_correlation(self, data, threshold=None, caption='Correlation Stack', bins=100, cmap='auto', **kwargs):
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
        #self.plot_AOI(ax=axs[1], **kwargs)
        #self.plot_POI(ax=axs[1], **kwargs)
        plt.suptitle(caption)
        plt.tight_layout()
