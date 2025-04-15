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
from .Stack_base import Stack_base
from insardev_toolkit import progressbar

class Stack_phasediff(Stack_base):

    def interferogram(self, pairs, datas, weight, phase,
                              resolution, wavelength, gaussian_threshold, psize, coarsen,
                              stack, polarizations, compute, debug):
        import xarray as xr
        import numpy as np
        import dask

        if datas is None:
            # list of datasets to process
            datas = self.dss
        
        if not isinstance(datas, (list, tuple)):
            datas = [datas]

        if polarizations is None:
            polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in datas[0].data_vars]
        elif isinstance(polarizations, str):
            polarizations = [polarizations]
        
        if len(polarizations) > 1:
            intfs_total = []
            corrs_total = []
            for pol in polarizations:
                intfs, corrs = self.interferogram(pairs, datas=datas, weight=weight, phase=phase,
                                            resolution=resolution, wavelength=wavelength, gaussian_threshold=gaussian_threshold,
                                            psize=psize, coarsen=coarsen,
                                            stack=stack, polarizations=pol, compute=compute, debug=debug)
                intfs_total.append(intfs)
                corrs_total.append(corrs)
                del intfs, corrs

            intfs_total = [xr.merge([das[idx] for das in intfs_total]) for idx in range(len(intfs_total[0]))]
            corrs_total = [xr.merge([das[idx] for das in corrs_total]) for idx in range(len(corrs_total[0]))]
            return (intfs_total, corrs_total)
        
        # process single polarization
        polarization = polarizations[0] if isinstance(polarizations, (list, tuple)) else polarizations

        # define anti-aliasing filter for the specified output resolution
        if wavelength is None:
            wavelength = resolution

        if weight is not None:
           # convert to lazy data
           weight = weight.astype(np.float32).chunk(-1 if weight.chunks is None else weight.chunks)
    
        # Initialize decimators to None by default
        decimator_intf = None
        decimator_corr = None
        
        # decimate the 1:4 multilooking grids to specified resolution
        if resolution is not None:
            decimator_intf = self.decimator(resolution=resolution, grid=datas, coarsen=coarsen, wrap=True,  debug=debug)
            decimator_corr = self.decimator(resolution=resolution, grid=datas, coarsen=coarsen, wrap=False, debug=debug)
        
        intfs = []
        corrs = []
        for data in datas:
            # copy id from the data to the result
            id = data.attrs.get('id', None)

            if weight is not None:
                data = data.reindex_like(weight, fill_value=np.nan)
            intensity = np.square(np.abs(data[polarization]))
            # Gaussian filtering with cut-off wavelength and optional multilooking on amplitudes
            intensity_look = self.multilooking(intensity, weight=weight,
                                               wavelength=wavelength, coarsen=coarsen, gaussian_threshold=gaussian_threshold, debug=debug)
            del intensity
            # calculate phase difference with topography correction
            phasediff = self.phasediff(pairs, data[polarization], phase=phase, debug=debug)
            # Gaussian filtering with cut-off wavelength and optional multilooking on phase difference
            phasediff_look = self.multilooking(phasediff, weight=weight,
                                               wavelength=wavelength, coarsen=coarsen, gaussian_threshold=gaussian_threshold, debug=debug)
            del phasediff
            # correlation with optional range decimation
            corr_look = self.correlation(phasediff_look, intensity_look, debug=debug)
            del intensity_look
            if psize is not None:
                # Goldstein filter in psize pixel patch size on square grid cells produced using 1:4 range multilooking
                phasediff_look_goldstein = self.goldstein(phasediff_look, corr_look, psize, debug=debug)
                del phasediff_look
                # convert complex phase difference to interferogram
                #intf_look = self.interferogram(phasediff_look_goldstein, debug=debug)
                phasediff_look = phasediff_look_goldstein
                del phasediff_look_goldstein

            # filter out not valid pixels
            if weight is not None:
                weight_look = self.multilooking(weight, wavelength=None, coarsen=coarsen, debug=debug)
                phasediff_look = phasediff_look.where(np.isfinite(weight_look))
                corr_look = corr_look.where(np.isfinite(weight_look))
                del weight_look
                
            # convert complex phase difference to interferogram
            intf_look = self.phase2interferogram(phasediff_look, debug=debug)
            del phasediff_look
    
            # compute together because correlation depends on phase, and filtered phase depends on correlation.
            #progressbar(result := dask.persist(decimator(corr15m), decimator(intf15m)), desc='Compute Phase and Correlation')
            # unpack results for a single interferogram
            #corr90m, intf90m = [grid[0] for grid in result]
            # anti-aliasing filter for the output resolution is applied above
            if decimator_intf is not None and decimator_corr is not None:
                #ds = xr.merge([intf_dec, corr_dec])
                das = (decimator_intf(intf_look),  decimator_corr(corr_look))
            else:
                #ds = xr.merge([intf_look, corr_look])
                das = (intf_look,  corr_look)
            del corr_look, intf_look

            if isinstance(stack, xr.DataArray):
                #ds = ds.interp(y=stack.y, x=stack.x, method='nearest')
                intfs.append(das[0].interp(y=stack.y, x=stack.x, method='nearest').to_dataset(name=polarization).assign_attrs(id=id))
                corrs.append(das[1].interp(y=stack.y, x=stack.x, method='nearest').to_dataset(name=polarization).assign_attrs(id=id))
            else:
                intfs.append(das[0].to_dataset(name=polarization).assign_attrs(id=id))
                corrs.append(das[1].to_dataset(name=polarization).assign_attrs(id=id))
            del das

        # clean up decimators after all iterations are complete
        del decimator_intf, decimator_corr

        if compute:
            progressbar(result := dask.persist(intfs, corrs), desc=f'Compute {polarization} Interferogram'.ljust(25))
            del intfs, corrs
            return result
        return (intfs, corrs)


    # single-look interferogram processing has a limited set of arguments
    # resolution and coarsen are not applicable here
    def interferogram_singlelook(self, pairs, datas=None, weight=None, phase=None,
                                         wavelength=None, gaussian_threshold=0.5, psize=None,
                                         stack=None, polarizations=None, compute=False, debug=False):
        return self.interferogram(pairs, datas=datas, weight=weight, phase=phase,
                                   wavelength=wavelength, gaussian_threshold=gaussian_threshold, psize=psize,
                                   stack=stack, polarizations=polarizations, compute=compute, debug=debug)

    # Goldstein filter requires square grid cells means 1:4 range multilooking.
    # For multilooking interferogram we can use square grid always using coarsen = (1,4)
    def interferogram_multilook(self, pairs, datas=None, weight=None, phase=None,
                                        resolution=None, wavelength=None, gaussian_threshold=0.5, psize=None, coarsen=(1,4),
                                        stack=None, polarizations=None, compute=False, debug=False):
        return self.interferogram(pairs, datas=datas, weight=weight, phase=phase,
                                   resolution=resolution, wavelength=wavelength, gaussian_threshold=gaussian_threshold, psize=psize, coarsen=coarsen,
                                   stack=stack, polarizations=polarizations, compute=compute, debug=debug)

    @staticmethod
    def phase2interferogram(phase, debug=False):
        import numpy as np

        if debug:
            print ('DEBUG: interferogram')

        if np.issubdtype(phase.dtype, np.complexfloating):
            return np.arctan2(phase.imag, phase.real).rename('phase')
        return phase

#     @staticmethod
#     def correlation(I1, I2, amp):
#         import xarray as xr
#         import numpy as np
#         # constant from GMTSAR code
#         thresh = 5.e-21
#         i = I1 * I2
#         corr = xr.where(i > 0, amp / np.sqrt(i), 0)
#         corr = xr.where(corr < 0, 0, corr)
#         corr = xr.where(corr > 1, 1, corr)
#         # mask too low amplitude areas as invalid
#         # amp1 and amp2 chunks are high for SLC, amp has normal chunks for NetCDF
#         return xr.where(i >= thresh, corr, np.nan).chunk(a.chunksizes).rename('phase')

    def correlation(self, phase, intensity, debug=False):
        """
        Example:
        data_200m = stack.multilooking(np.abs(sbas.open_data()), wavelength=200, coarsen=(4,16))
        intf2_200m = stack.multilooking(intf2, wavelength=200, coarsen=(4,16))
        stack.correlation(intf2_200m, data_200m)

        Note:
        Multiple interferograms require the same data grids, allowing us to speed up the calculation
        by saving filtered data to a disk file.
        """
        import pandas as pd
        import dask
        import xarray as xr
        import numpy as np

        if debug:
            print ('DEBUG: correlation')

        # convert pairs (list, array, dataframe) to 2D numpy array
        pairs, dates = self.get_pairs(phase, dates=True)
        pairs = pairs[['ref', 'rep']].astype(str).values

        # check correctness for user-defined data arguments
        assert np.issubdtype(phase.dtype, np.complexfloating), 'ERROR: Phase should be complex-valued data.'
        assert not np.issubdtype(intensity.dtype, np.complexfloating), 'ERROR: Intensity cannot be complex-valued data.'

        stack = []
        for stack_idx, pair in enumerate(pairs):
            date1, date2 = pair
            # calculate correlation
            corr = (np.abs(phase.sel(pair=' '.join(pair)) / np.sqrt(intensity.sel(date=date1) * intensity.sel(date=date2)))).clip(0, 1)
            # modify values in place
            #corr = xr.where(corr < 0, 0, corr)
            #corr = xr.where(corr > 1, 1, corr)
            #corr = corr.where(corr.isnull() | (corr >= 0), 0)
            #corr = corr.where(corr.isnull() | (corr <= 1), 1)
            # add to stack
            stack.append(corr)
            del corr

        return xr.concat(stack, dim='pair').rename('correlation')

    def phasediff(self, pairs, data, phase=None, debug=False):
        import dask.array as da
        import xarray as xr
        import numpy as np
        import pandas as pd

        if debug:
            print ('DEBUG: phasediff')

        assert phase is None or \
               np.issubdtype(phase.dtype, np.floating) or \
               np.issubdtype(phase.dtype, np.complexfloating)

        # convert pairs (list, array, dataframe) to 2D numpy array
        pairs, dates = self.get_pairs(pairs, dates=True)
        pairs = pairs[['ref', 'rep']].astype(str).values
        # append coordinates which usually added from topo phase dataarray
        coord_pair = [' '.join(pair) for pair in pairs]
        coord_ref = xr.DataArray(pd.to_datetime(pairs[:,0]), coords={'pair': coord_pair})
        coord_rep = xr.DataArray(pd.to_datetime(pairs[:,1]), coords={'pair': coord_pair})

        # calculate phase difference
        data1 = data.sel(date=pairs[:,0]).drop_vars('date').rename({'date': 'pair'})
        data2 = data.sel(date=pairs[:,1]).drop_vars('date').rename({'date': 'pair'})

        if phase is None:
            phase_correction = 1
        else:
            # convert real phase values to complex if needed 
            phase_correction = np.exp(-1j * phase) if np.issubdtype(phase.dtype, np.floating) else phase

        da = (phase_correction * data1 * data2.conj())\
               .assign_coords(ref=coord_ref, rep=coord_rep, pair=coord_pair)
        del phase_correction, data1, data2
        
        return da.rename('phase')

    def goldstein(self, phase, corr, psize=32, debug=False):
        import xarray as xr
        import numpy as np
        import dask
        import warnings
        # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
        warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', module='dask')
        warnings.filterwarnings('ignore', module='dask.core')

        if debug:
            print ('DEBUG: goldstein')

        if psize is None:
            # miss the processing
            return phase
        
        if not isinstance(psize, (list, tuple)):
            psize = (psize, psize)

        def apply_pspec(data, alpha):
            # NaN is allowed value
            assert not(alpha < 0), f'Invalid parameter value {alpha} < 0'
            wgt = np.power(np.abs(data)**2, alpha / 2)
            data = wgt * data
            return data

        def make_wgt(psize):
            nyp, nxp = psize
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
            data = np.fft.fft2(data, s=psize)
            data = apply_pspec(data, alpha)
            data = np.fft.ifft2(data, s=psize)
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
            for i in range(0, data.shape[0] - psize[0], psize[0] // 2):
                for j in range(0, data.shape[1] - psize[1], psize[1] // 2):
                    # Create proocessing windows
                    data_window = data[i:i+psize[0], j:j+psize[1]]
                    corr_window = corr[i:i+psize[0], j:j+psize[1]]
                    wgt_window = wgt_matrix[:data_window.shape[0],:data_window.shape[1]]
                    # Apply the filter to the window
                    filtered_window = patch_goldstein_filter(data_window, corr_window, wgt_window, psize)
                    # Add the result to the output array
                    slice_i = slice(i, min(i + psize[0], out.shape[0]))
                    slice_j = slice(j, min(j + psize[1], out.shape[1]))
                    out[slice_i, slice_j] += filtered_window[:slice_i.stop - slice_i.start, :slice_j.stop - slice_j.start]
            return out

        assert phase.shape == corr.shape, f'ERROR: phase and correlation variables have different shape \
                                          ({phase.shape} vs {corr.shape})'

        if len(phase.dims) == 2:
            stackvar = None
        else:
            stackvar = phase.dims[0]
    
        stack =[]
        for ind in range(len(phase) if stackvar is not None else 1):
            # Apply function with overlap; psize//2 overlap is not enough (some empty lines produced)
            # use complex data and real correlation
            # fill NaN values in correlation by zeroes to prevent empty output blocks
            block = dask.array.map_overlap(apply_goldstein_filter,
                                           (phase[ind] if stackvar is not None else phase).fillna(0).data,
                                           (corr[ind]  if stackvar is not None else corr ).fillna(0).data,
                                           depth=(psize[0] // 2 + 2, psize[1] // 2 + 2),
                                           dtype=np.complex64, 
                                           meta=np.array(()),
                                           psize=psize,
                                           wgt_matrix = make_wgt(psize))
            # Calculate the phase
            stack.append(block)
            del block

        if stackvar is not None:
            ds = xr.DataArray(dask.array.stack(stack), coords=phase.coords)
        else:
            ds = xr.DataArray(stack[0], coords=phase.coords)
        del stack
        # replace zeros produces in NODATA areas
        return ds.where(ds).rename('phase')

    def concat(self, datas=None, polarizations=None, wrap=False, compute=False):
        """
        This function is a faster implementation for the standalone function combination of xr.concat and xr.align:
        xr.concat(xr.align(*datas, join='outer'), dim='stack_dim').mean('stack_dim').compute()
        """
        import xarray as xr
        import numpy as np
        import dask

        #print ('datas', datas, 'polarizations', polarizations, 'wrap', wrap)
        #print ()
        # if datas is None:
        #     datas = self.dss

        # not iterable data cannot be concatenated
        if isinstance(datas, (xr.Dataset, xr.DataArray)):
            if compute:
                progressbar(result := datas.persist(), desc=f'Compute Data'.ljust(25))
                return result
            return datas
        elif isinstance(datas, (list, tuple)):
            # empty list or single dataarray do not need to be concatenated
            if len(datas) == 0:
                return None
            if len(datas) == 1:
                if compute:
                    progressbar(result := datas[0].persist(), desc=f'Compute Data'.ljust(25))
                    return result
                return datas[0]
        else:
            raise ValueError(f'ERROR: datas is not a list, tuple, Dataset or DataArray: {type(datas)}')

        # process list of datasets with multiple polarizations
        if isinstance(datas[0], xr.Dataset):
            if polarizations is None:
                polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in datas[0].data_vars]
            elif isinstance(polarizations, str):
                polarizations = [polarizations]
            #print ('polarizations', polarizations)
            
            das_total = []
            for pol in polarizations:
                das = self.concat([ds[pol] for ds in datas], wrap=wrap)
                das_total.append(das)
                del das
            das_total = xr.merge(das_total)
            
            if compute:
                progressbar(result := das_total.persist(), desc=f'Compute Concatenated Data'.ljust(25))
                del das_total
                return result
            return das_total

        # process single polarization (variable)

        # define unified grid
        y_min = min(ds.y.min().item() for ds in datas)
        y_max = max(ds.y.max().item() for ds in datas)
        x_min = min(ds.x.min().item() for ds in datas)
        x_max = max(ds.x.max().item() for ds in datas)
        #print (y_min, y_max, x_min, x_max, y_max-y_min, x_max-x_min)
        stackvar = list(datas[0].dims)[0]
        # workaround for dask.array.blockwise
        stackval = datas[0][stackvar].astype(str)
        dy = datas[0].y.diff('y').item(0)
        dx = datas[0].x.diff('x').item(0)
        #print ('dy, dx', dy, dx)
        ys = xr.DataArray(np.arange(y_min, y_max + dy/2, dy), dims=['y']).chunk({'y': self.chunksize})
        xs = xr.DataArray(np.arange(x_min, x_max + dx/2, dx), dims=['x']).chunk({'x': self.chunksize})
        #print ('pair', pair)
        #print ('ys', ys)
        #print ('xs', xs)
        
        # use outer variable datas
        def block_dask(stack, y_chunk, x_chunk):
            #print ('pair', pair)
            das_slice = [da.sel(y=slice(y_chunk.min(), y_chunk.max()), x=slice(x_chunk.min(), x_chunk.max())).compute(num_workers=1) for da in datas]
            das_block = [da.reindex({'y': y_chunk, 'x': x_chunk}, fill_value=np.nan, copy=False) for da in das_slice if da.y.size > 0 and da.x.size > 0]
            del das_slice
            if len(das_block) == 0:
                # return empty block
                return np.full((stack.size, y_chunk.size, x_chunk.size), np.nan, dtype=datas[0].dtype)
            if len(das_block) == 1:
                # return single block as is
                return das_block[0].values
            if not wrap:
                # calculate arithmetic mean for phase and correlation data
                return xr.concat(das_block, dim='stack_dim', join='inner').mean('stack_dim', skipna=True).values
            else:
                # calculate circular mean for interferogram data
                block_complex = xr.concat([np.exp(1j * da) for da in das_block], dim='stack_dim').mean('stack_dim').values
                return np.arctan2(block_complex.imag, block_complex.real)

        data = dask.array.blockwise(
            block_dask,
            'zyx',
            stackval, 'z',
            ys, 'y',
            xs, 'x',
            dtype=datas[0].dtype
        )
        da = xr.DataArray(data, coords={stackvar: stackval, 'y': ys, 'x': xs})\
            .rename(datas[0].name)\
            .assign_attrs(datas[0].attrs)
        del data
        return self.spatial_ref(da, datas)

    def concat_interferogram(self, datas=None, polarizations=None, compute=False):
        return self.concat(datas, polarizations, wrap=True, compute=compute)

    def concat_correlation(self, datas=None, polarizations=None, compute=False):
        return self.concat(datas, polarizations, wrap=False, compute=compute)

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
                   caption, cols, rows, size, nbins, aspect, y, wrap, screen=None, **kwargs):
        import xarray as xr
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # screen size in pixels (width, height) to estimate reasonable number pixels per plot
        # this is quite large to prevent aliasing on 600dpi plots without additional processing
        if screen is None:
            screen = (8000,4000)

        def plot_polarization(data, polarization):

            stackvar = list(data.dims)[0]

            if isinstance(data, xr.Dataset):
                da = data[polarization].isel({stackvar: slice(0, rows)})
            else:
                das = [da[polarization].isel({stackvar: slice(0, rows)}) for da in data]
                da = self.concat(das, wrap=wrap)
                del das

            if 'stack' in da.dims and isinstance(da.coords['stack'].to_index(), pd.MultiIndex):
                da = da.unstack('stack')
            
            # there is no reason to plot huge arrays much larger than screen size for small plots
            #print ('screen_size', screen_size)
            size_y, size_x = da.shape[1:]
            #print ('size_x, size_y', size_x, size_y)
            factor_y = int(np.round(size_y / (screen[1] / rows)))
            factor_x = int(np.round(size_x / (screen[0] / cols)))
            #print ('factor_x, factor_y', factor_x, factor_y)
            # decimate and materialize data for all the calculations and plotting
            da = da[:,::max(1, factor_y),::max(1, factor_x)].compute()

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

        if not isinstance(data, (xr.Dataset, xr.DataArray, (list, tuple))):
            raise ValueError(f'ERROR: invalid data type {type(data)}. Should be xr.Dataset or xr.DataArray or list of xr.Dataset or xr.DataArray')

        # convert DataArray to Dataset to plot a single polarization
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        # convert list of DataArray to list of Dataset to plot a single polarization
        if isinstance(data, (list, tuple)):
            data = [da.to_dataset() for da in data]

        if polarizations is None:
            polarizations = list(data.data_vars) if isinstance(data, xr.Dataset) else list(data[0].data_vars)
        elif isinstance(polarizations, str):
            polarizations = [polarizations]

        # process polarizations one by one
        for pol in polarizations:
            plot_polarization(data, polarization=pol)

    def plot_displacement_mm(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Displacement, [mm]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, screen=None, **kwargs):
        data_los_mm = self.los_displacement_mm(data)
        self.plot_stack(data_los_mm, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, screen=screen, **kwargs)

    def plot_displacement(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Displacement, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, screen=None, **kwargs):
        self.plot_stack(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, screen=screen, **kwargs)

    def plot_phase(self, data, polarizations=None,
                   cmap='turbo', vmin=None, vmax=None, quantile=None, symmetrical=False,
                   caption='Phase, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, screen=None, **kwargs):
        self.plot_stack(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=symmetrical,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, screen=screen, **kwargs)

    def plot_interferogram(self, data, polarizations=None,
                           cmap='gist_rainbow_r',
                           caption='Phase, [rad]', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, screen=None, **kwargs):
        import numpy as np
        self.plot_stack(data, polarizations,
                        cmap=cmap, vmin=-np.pi, vmax=np.pi, quantile=None, symmetrical=False,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=True, **kwargs)

    def plot_correlation(self, data, polarizations=None,
                         cmap='auto', vmin=0, vmax=1, quantile=None, symmetrical=False,
                         caption='Correlation', cols=4, rows=4, size=4, nbins=5, aspect=1.2, y=1.05, screen=None, **kwargs):
        import matplotlib.colors as mcolors
        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
        self.plot_stack(data, polarizations,
                        cmap=cmap, vmin=vmin, vmax=vmax, quantile=quantile, symmetrical=False,
                        caption=caption, cols=cols, rows=rows, size=size, nbins=nbins, aspect=aspect, y=y, wrap=False, screen=screen, **kwargs)

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
