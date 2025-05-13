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
    import xarray as xr
    import numpy as np
    import pandas as pd

    # internal method to compute interferogram on single polarization data array(s)
    def _interferogram(self,
                       pairs:list[tuple[str|int,str|int]]|np.ndarray|pd.DataFrame,
                       data:xr.DataArray,
                       weight:xr.DataArray|None=None,
                       phase:xr.DataArray|None=None,
                       wavelength:float|None=None,
                       gaussian_threshold:float=0.5,
                       multilook:bool=False,
                       goldstein_window:int|list[int,int]|None=None,
                       debug:bool=False
                       ) -> tuple[xr.DataArray,xr.DataArray]:
        import xarray as xr
        import numpy as np
        import dask
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

        assert isinstance(data, xr.DataArray), 'ERROR: data should be a DataArray'

        # wrap simplified single pair argument in a list
        pairs = pairs if isinstance(pairs[0], (list, tuple)) else [pairs]

        if weight is not None:
           # convert to lazy data
           weight = weight.astype(np.float32).chunk(-1 if weight.chunks is None else weight.chunks)
        
        # convert numeric pairs to date pairs
        _pairs = [pair if isinstance(pair[0], str) else (data.date.isel(date=pair[0]).item(), data.date.isel(date=pair[1]).item()) for pair in pairs]
        
        # data = data[polarization]

        if weight is not None:
            # unify shape of data and weight
            data = data.reindex_like(weight, fill_value=np.nan)
        intensity = np.square(np.abs(data))
        # Gaussian filtering with cut-off wavelength and optional multilooking on amplitudes
        intensity_look = self._multilooking(intensity, weight=weight,
                                            wavelength=wavelength, gaussian_threshold=gaussian_threshold, debug=debug)
        del intensity
        # calculate phase difference with topography correction
        phasediff = self._phasediff(_pairs, data, debug=debug)
        if phase is not None:
            phasediff = phasediff * (np.exp(-1j * phase) if np.issubdtype(phase.dtype, np.floating) else phase)
        # Gaussian filtering with cut-off wavelength and optional multilooking on phase difference
        phasediff_look = self._multilooking(phasediff, weight=weight,
                                            wavelength=wavelength, gaussian_threshold=gaussian_threshold, debug=debug)
        # correlation requires multilooking to detect influence between pixels
        # hint: use multilook=False argument to keep phase difference without multilooking
        corr_look = self._correlation(phasediff_look, intensity_look, debug=debug)
        if not multilook:
            # keep phase difference without multilooking
            phasediff_look = phasediff
        del phasediff
        del intensity_look
        if goldstein_window is not None:
            # Goldstein filter in "psize" patch size, pixels
            phasediff_look = self._goldstein(phasediff_look, corr_look, psize=goldstein_window, debug=debug)

        # filter out not valid pixels
        if weight is not None:
            # apply coarsening to weight to match the phase difference grid
            # no multilooking for weight because wavelength=None
            weight_coarsen = self._multilooking(weight, wavelength=None, debug=debug)
            phasediff_look = phasediff_look.where(np.isfinite(weight_coarsen))
            corr_look = corr_look.where(np.isfinite(weight_coarsen))
            del weight_coarsen
            
        # convert complex phase difference to interferogram
        intf_look = self._phase2interferogram(phasediff_look, debug=debug)
        del phasediff_look

        intf = intf_look.assign_attrs(data.attrs)
        corr = corr_look.assign_attrs(data.attrs)
        del corr_look, intf_look

        return (intf, corr)

        # if compute:
        #     progressbar(result := dask.persist(intfs, corrs), desc=f'Computing {data.name} Interferogram'.ljust(25))
        #     del intfs, corrs
        #     return result
        # return (intfs, corrs)

    def interferogram(self,
                      pairs:list[tuple[str|int,str|int]]|np.ndarray|pd.DataFrame,
                      datas:dict[str,xr.Dataset],
                      weights:dict[str,xr.DataArray]|None=None,
                      phases:dict[str,xr.DataArray]|None=None,
                      wavelength:float|None=None,
                      gaussian_threshold:float=0.5,
                      multilook:bool=False,
                      goldstein_window:int|list[int,int]|None=None,
                      compute:bool=False,
                      debug:bool=False
                      ) -> dict[str,xr.Dataset]:
        import xarray as xr
        import numpy as np
        import dask

        assert isinstance(datas, dict), 'ERROR: datas should be a dict of xarray.Dataset'
        assert isinstance(weights, dict) or weights is None, 'ERROR: weights should be a dict of xarray.DataArray'
        assert isinstance(phases, dict) or phases is None, 'ERROR: phases should be a dict of xarray.DataArray'
        # workaround for previous code

        polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in next(iter(datas.values())).data_vars]
        #print ('polarizations', polarizations)

        results = []
        for polarization in polarizations:
            results_pol = [self._interferogram(pairs,
                                       data=datas[k][polarization],
                                       weight=weights[k] if weights is not None else None,
                                       phase=phases[k] if phases is not None else None,
                                       wavelength=wavelength,
                                       gaussian_threshold=gaussian_threshold,
                                       multilook=multilook,
                                       goldstein_window=goldstein_window,
                                       debug=debug) for k in datas.keys()]
            if compute:
                progressbar(results_pol := dask.persist(*results_pol), desc=f'Computing {polarization} Intfs...'.ljust(25))
            results.append(results_pol)
            del results_pol

        # unpack the results
        intfs = {k:xr.merge([results[pidx][idx][0] for pidx in range(len(polarizations))]) for idx,k in enumerate(datas.keys())}
        corrs = {k:xr.merge([results[pidx][idx][1] for pidx in range(len(polarizations))]) for idx,k in enumerate(datas.keys())}
        return intfs, corrs

    # single-look interferogram processing has a limited set of arguments
    # resolution and coarsen are not applicable here
    def interferogram_singlelook(self,
                                pairs,
                                datas=None,
                                weights=None,
                                phases=None,
                                wavelength=None,
                                gaussian_threshold=0.5,
                                goldstein_window=None,
                                compute=False,
                                debug=False):
        return self.interferogram(pairs,
                                datas=datas,
                                weights=weights,
                                phases=phases,
                                wavelength=wavelength,
                                gaussian_threshold=gaussian_threshold,
                                multilook=False,
                                goldstein_window=goldstein_window,
                                compute=compute,
                                debug=debug)

    # Goldstein filter requires square grid cells means 1:4 range multilooking.
    # For multilooking interferogram we can use square grid always using coarsen = (1,4)
    def interferogram_multilook(self,
                              pairs,
                              datas=None,
                              weights=None,
                              phases=None,
                              wavelength=None,
                              gaussian_threshold=0.5,
                              goldstein_window=None,
                              compute=False,
                              debug=False):
        return self.interferogram(pairs,
                                datas=datas,
                                weights=weights,
                                phases=phases,
                                wavelength=wavelength,
                                gaussian_threshold=gaussian_threshold,
                                multilook=True,
                                goldstein_window=goldstein_window,
                                compute=compute,
                                debug=debug)

    @staticmethod
    def _phase2interferogram(phase, debug=False):
        import numpy as np

        if debug:
            print ('DEBUG: interferogram')

        if np.issubdtype(phase.dtype, np.complexfloating):
            return np.arctan2(phase.imag, phase.real)
        return phase

    def _correlation(self, phase, intensity, debug=False):
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

        # check correctness for user-defined data arguments
        assert np.issubdtype(phase.dtype, np.complexfloating), 'ERROR: Phase should be complex-valued data.'
        assert not np.issubdtype(intensity.dtype, np.complexfloating), 'ERROR: Intensity cannot be complex-valued data.'

        if debug:
            print ('DEBUG: correlation')

        # convert pairs (list, array, dataframe) to 2D numpy array
        pairs, dates = self._get_pairs(phase, dates=True)
        pairs = pairs[['ref', 'rep']].astype(str).values

        stack = []
        for stack_idx, pair in enumerate(pairs):
            date1, date2 = pair
            # calculate correlation
            corr = (np.abs(phase.sel(pair=' '.join(pair)) / np.sqrt(intensity.sel(date=date1) * intensity.sel(date=date2)))).clip(0, 1)
            # add to stack
            stack.append(corr)
            del corr

        return xr.concat(stack, dim='pair')

    def _phasediff(self, pairs, data, debug=False):
        import dask.array as da
        import xarray as xr
        import numpy as np
        import pandas as pd
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
            print ('DEBUG: phasediff')

        # convert pairs (list, array, dataframe) to 2D numpy array
        pairs, dates = self._get_pairs(pairs, dates=True)
        pairs = pairs[['ref', 'rep']].astype(str).values
        # append coordinates which usually added from topo phase dataarray
        coord_pair = [' '.join(pair) for pair in pairs]
        coord_ref = xr.DataArray(pd.to_datetime(pairs[:,0]), coords={'pair': coord_pair})
        coord_rep = xr.DataArray(pd.to_datetime(pairs[:,1]), coords={'pair': coord_pair})

        # calculate phase difference
        data1 = data.sel(date=pairs[:,0]).drop_vars('date').rename({'date': 'pair'})
        data2 = data.sel(date=pairs[:,1]).drop_vars('date').rename({'date': 'pair'})

        da = (data1 * data2.conj()).assign_coords(ref=coord_ref, rep=coord_rep, pair=coord_pair)
        del data1, data2
        
        return da

    def _goldstein(self, phase, corr, psize=32, debug=False):
        import xarray as xr
        import numpy as np
        import dask
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
                    # do not process NODATA areas filled with zeros
                    fraction_valid = np.count_nonzero(data_window != 0) / data_window.size
                    if fraction_valid >= 0.5:
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
        return ds.where(np.isfinite(phase)).rename(phase.name)
