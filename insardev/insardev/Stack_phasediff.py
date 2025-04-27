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
                       pairs:list[tuple[str,str]]|np.ndarray|pd.DataFrame,
                       datas:dict[str,xr.DataArray],
                       polarization: str,
                       weight:xr.DataArray|None=None,
                       phase:xr.DataArray|None=None,
                       resolution:float|None=None,
                       wavelength:float|None=None,
                       gaussian_threshold:float=0.5,
                       psize:int|list[int,int]|None=None,
                       coarsen:list[int,int]|None=None,
                       stack:xr.DataArray|None=None,
                       compute:bool=False,
                       debug:bool=False
                       ):
        import xarray as xr
        import numpy as np
        import dask
        import warnings
        # Ignore *any* RuntimeWarning coming from dask/_task_spec.py
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"dask\._task_spec"
        )
        # …and just in case you want to match by message too:
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
            module=r"dask\._task_spec"
        )

        assert isinstance(datas, (list, tuple, xr.DataArray)), 'ERROR: datas should be a list or tuple or DataArray'

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
        for data in (datas if isinstance(datas, (list, tuple)) else [datas]):
            #data_vars = data.drop(['VV','VH','HH','HV'], errors='ignore').data_vars
            #data_attrs = data.attrs
            #print (data_vars)
            data = data[polarization]

            if weight is not None:
                data = data.reindex_like(weight, fill_value=np.nan)
            intensity = np.square(np.abs(data))
            # Gaussian filtering with cut-off wavelength and optional multilooking on amplitudes
            intensity_look = self.multilooking(intensity, weight=weight,
                                               wavelength=wavelength, coarsen=coarsen, gaussian_threshold=gaussian_threshold, debug=debug)
            del intensity
            # calculate phase difference with topography correction
            phasediff = self.phasediff(pairs, data, phase=phase, debug=debug)
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
            # anti-aliasing filter for the output resolution is applied above
            if decimator_intf is not None and decimator_corr is not None:
                das = (decimator_intf(intf_look),  decimator_corr(corr_look))
            else:
                das = (intf_look,  corr_look)
            del corr_look, intf_look

            # append original data attributes to the result
            das = [da.assign_attrs(data.attrs) for da in das]

            if isinstance(stack, xr.DataArray):
                intfs.append(das[0].interp(y=stack.y, x=stack.x, method='nearest'))
                corrs.append(das[1].interp(y=stack.y, x=stack.x, method='nearest'))
            else:
                intfs.append(das[0])
                corrs.append(das[1])
            del das

        # clean up decimators after all iterations are complete
        del decimator_intf, decimator_corr

        if not isinstance(datas, (list, tuple)):
            intfs = intfs[0]
            corrs = corrs[0]

        if compute:
            progressbar(result := dask.persist(intfs, corrs), desc=f'Computing {data.name} Interferogram'.ljust(25))
            del intfs, corrs
            return result
        return (intfs, corrs)

    def interferogram(self,
                      pairs:list[tuple[str,str]]|np.ndarray|pd.DataFrame,
                      datas:dict[str,xr.DataArray]|xr.Dataset|xr.DataArray,
                      weight:xr.DataArray|None=None,
                      phase:xr.DataArray|None=None,
                      resolution:float|None=None,
                      wavelength:float|None=None,
                      gaussian_threshold:float=0.5,
                      psize:int|list[int,int]|None=None,
                      coarsen:list[int,int]|None=None,
                      stack:xr.DataArray|None=None,
                      compute:bool=False,
                      debug:bool=False
                      ):
        import xarray as xr
        import numpy as np
        import dask

        assert isinstance(datas, (dict, xr.Dataset, xr.DataArray)), 'ERROR: datas should be a dict, Dataset or DataArray'

        # if datas is None:
        #     datas = self.dss

        datas_iterable = isinstance(datas, dict)
        # workaround for previous code
        datas = list(datas.values()) if datas_iterable else datas
        #print ('datas_iterable', datas_iterable)
        datas_dataset = isinstance(datas[0], xr.Dataset) if datas_iterable else isinstance(datas, xr.Dataset)
        #print ('datas_dataset', datas_dataset)

        if not isinstance(datas, (list, tuple)):
            if isinstance(datas, xr.Dataset):
                datas = [datas]
            elif isinstance(datas, xr.DataArray):
                datas = [datas.to_dataset()]
            else:
                raise ValueError(f'ERROR: datas is not a Dataset and DataArray or list or tuple of them: {type(datas)}')
        else:
            if isinstance(datas[0], xr.DataArray):
                datas = [ds.to_dataset() for ds in datas]
        
        # copy id from the data to the result
        #ids = [ds.attrs.get('id', None) for ds in datas]

        polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in datas[0].data_vars]
        #print ('polarizations', polarizations)
        
        intfs_pols = []
        corrs_pols = []
        for pol in polarizations:
            intfs, corrs = self._interferogram(pairs,
                                               datas=datas,
                                               polarization=pol,
                                               weight=weight,
                                               phase=phase,
                                               resolution=resolution,
                                               wavelength=wavelength,
                                               gaussian_threshold=gaussian_threshold,
                                               psize=psize,
                                               coarsen=coarsen,
                                               stack=stack,
                                               compute=compute,
                                               debug=debug
                                               )
            intfs_pols.append(intfs)
            corrs_pols.append(corrs)
            del intfs, corrs

        # if not datas_iterable:
        #     intfs_pols = [intfs[0].assign_attrs(id=ids[0]) for intfs in intfs_pols]
        #     corrs_pols = [corrs[0].assign_attrs(id=ids[0]) for corrs in corrs_pols]
        #     return (intfs_pols[0], corrs_pols[0]) 
        #     #return (intfs_pols[0], corrs_pols[0]) if datas_dataset else (intfs_pols[0][polarizations[0]], corrs_pols[0][polarizations[0]])

        intfs_pols = [xr.merge([das[idx] for das in intfs_pols]) for idx in range(len(intfs_pols[0]))]
        corrs_pols = [xr.merge([das[idx] for das in corrs_pols]) for idx in range(len(corrs_pols[0]))]
        dss = (intfs_pols, corrs_pols) if datas_dataset else ([intf[polarizations[0]] for intf in intfs_pols], [corrs[polarizations[0]] for corrs in corrs_pols])
        # workaround for previous code, use attributes from the original data for keys to build a dict
        return self.to_dict(dss[0]), self.to_dict(dss[1]) if datas_iterable else dss
        #return dss

    # single-look interferogram processing has a limited set of arguments
    # resolution and coarsen are not applicable here
    def interferogram_singlelook(self,
                                pairs,
                                datas=None,
                                weight=None,
                                phase=None,
                                wavelength=None,
                                gaussian_threshold=0.5,
                                psize=None,
                                stack=None,
                                compute=False,
                                debug=False):
        return self.interferogram(pairs,
                                datas=datas,
                                weight=weight,
                                phase=phase,
                                wavelength=wavelength,
                                gaussian_threshold=gaussian_threshold,
                                psize=psize,
                                stack=stack,
                                compute=compute,
                                debug=debug)

    # Goldstein filter requires square grid cells means 1:4 range multilooking.
    # For multilooking interferogram we can use square grid always using coarsen = (1,4)
    def interferogram_multilook(self,
                              pairs,
                              datas=None,
                              weight=None,
                              phase=None,
                              resolution=None,
                              wavelength=None,
                              gaussian_threshold=0.5,
                              psize=None,
                              coarsen=(1,4),
                              stack=None,
                              compute=False,
                              debug=False):
        return self.interferogram(pairs,
                                datas=datas,
                                weight=weight,
                                phase=phase,
                                resolution=resolution,
                                wavelength=wavelength,
                                gaussian_threshold=gaussian_threshold,
                                psize=psize,
                                coarsen=coarsen,
                                stack=stack,
                                compute=compute,
                                debug=debug)

    @staticmethod
    def phase2interferogram(phase, debug=False):
        import numpy as np

        if debug:
            print ('DEBUG: interferogram')

        if np.issubdtype(phase.dtype, np.complexfloating):
            return np.arctan2(phase.imag, phase.real)
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
        import warnings
        # Ignore *any* RuntimeWarning coming from dask/_task_spec.py
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"dask\._task_spec"
        )
        # …and just in case you want to match by message too:
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
            module=r"dask\._task_spec"
        )

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

        return xr.concat(stack, dim='pair')

    def phasediff(self, pairs, data, phase=None, debug=False):
        import dask.array as da
        import xarray as xr
        import numpy as np
        import pandas as pd
        import warnings
        # Ignore *any* RuntimeWarning coming from dask/_task_spec.py
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"dask\._task_spec"
        )
        # …and just in case you want to match by message too:
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
            module=r"dask\._task_spec"
        )

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
        
        return da

    def goldstein(self, phase, corr, psize=32, debug=False):
        import xarray as xr
        import numpy as np
        import dask
        import warnings
        # Ignore *any* RuntimeWarning coming from dask/_task_spec.py
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"dask\._task_spec"
        )
        # …and just in case you want to match by message too:
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
            module=r"dask\._task_spec"
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
