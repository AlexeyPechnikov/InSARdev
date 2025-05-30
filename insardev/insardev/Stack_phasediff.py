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
from .Batch import Batch, BatchWrap, BatchUnit
from . import utils_xarray

class Stack_phasediff(Stack_base):
    import xarray as xr
    import numpy as np
    import pandas as pd

    # internal method to compute interferogram on single polarization data array(s)
    def _phasediff(self,
                       data:xr.DataArray|None=None,
                       weight:xr.DataArray|None=None,
                       phase:xr.DataArray|None=None,
                       pairs:list[tuple[str|int,str|int]]|np.ndarray|pd.DataFrame|None=None,
                       wavelength:float|None=None,
                       gaussian_threshold:float=0.5,
                       multilook:bool=False,
                       goldstein_window:int|list[int,int]|None=None,
                       complex:bool=False,
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

        #if weight is not None:
        #    # unify shape of data and weight
        #    data = data.reindex_like(weight, fill_value=np.nan)
        intensity = np.square(np.abs(data))
        # Gaussian filtering with cut-off wavelength on amplitudes
        intensity_look = self._multilooking(intensity, weight=weight,
                                            wavelength=wavelength, gaussian_threshold=gaussian_threshold, debug=debug)
        del intensity
        # calculate phase difference with topography correction
        phasediff = self._conj(_pairs, data, debug=debug)
        if phase is not None:
            phasediff = phasediff * (np.exp(-1j * phase) if np.issubdtype(phase.dtype, np.floating) else phase)
        # Gaussian filtering with cut-off wavelength on phase difference
        phasediff_look = self._multilooking(phasediff, weight=weight,
                                            wavelength=wavelength, gaussian_threshold=gaussian_threshold, debug=debug)
        # correlation requires multilooking to detect influence between pixels
        # hint: use multilook=False argument to keep phase difference without multilooking

        #print (phasediff)

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
        if not complex:
            intf_look = self._interferogram(phasediff_look, debug=debug)
        else:
            intf_look = phasediff_look
        del phasediff_look

        return (intf_look.assign_attrs(data.attrs), corr_look.assign_attrs(data.attrs))

    def phasediff_singlelook(self, pairs, weights=None, phases=None, compute=False, **kwarg):
        from .Batch import BatchComplex
        kwarg['multilook'] = False
        intfs, corrs = utils_xarray.apply_pol(BatchComplex(self), weights, phases, func=self._phasediff, compute=compute, pairs=pairs, **kwarg)
        return BatchWrap(intfs), BatchUnit(corrs)

    def phasediff_multilook(self, pairs, weights=None, phases=None, compute=False, **kwarg):
        from .Batch import BatchComplex
        kwarg['multilook'] = True
        intfs, corrs = utils_xarray.apply_pol(BatchComplex(self), weights, phases, func=self._phasediff, compute=compute, pairs=pairs, **kwarg)
        return BatchWrap(intfs), BatchUnit(corrs)
