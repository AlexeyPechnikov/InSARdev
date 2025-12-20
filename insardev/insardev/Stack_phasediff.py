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
from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex
from . import utils_xarray

class Stack_phasediff(Stack_base):
    import xarray as xr
    import numpy as np
    import pandas as pd

    # internal method to compute interferogram on single polarization data array(s)
    def phasediff(self,
                       pairs:list[tuple[str|int,str|int]]|np.ndarray|pd.DataFrame|None=None,
                       weight:xr.DataArray|None=None,
                       phase:xr.DataArray|None=None,
                       wavelength:float|None=None,
                       gaussian_threshold:float=0.5,
                       multilook:bool=True,
                       goldstein:int|list[int,int]|None=None,
                       complex:bool=False
                       ) -> tuple[xr.DataArray,xr.DataArray]:
        """
        hint: use multilook=False argument to keep phase difference without multilooking
        """
        import numpy as np

        # if wavelength is None:
        #     raise ValueError('wavelength is required to define spatial correlation')

        if goldstein is not None and wavelength is None:
            raise ValueError('wavelength is required to define spatial correlation for Goldstein filtering')

        pairs = np.array(pairs if isinstance(pairs[0], (list, tuple, np.ndarray)) else [pairs])
        ref_dates = pairs[:,0]
        rep_dates = pairs[:,1]

        data = BatchComplex(self)
        if weight is not None:
            data = data.reindex_like(weight, fill_value=np.nan)

        data1 = self.isel(date=ref_dates).rename(date='pair')
        data2 = self.isel(date=rep_dates).rename(date='pair')
        phasediff = BatchComplex(data1).drop_vars('pair') * BatchComplex(data2).drop_vars('pair').conj()
        if phase is not None:
            phasediff = phasediff * (phase.iexp(-1) if not isinstance(phase, BatchComplex) else phase)

        corr_look = None
        if wavelength is not None:
            # Gaussian filtering with cut-off wavelength on phase difference
            phasediff_look = phasediff.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)

            # Gaussian filtering with cut-off wavelength on amplitudes
            intensity_look = data.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)
            intensity_look1 = intensity_look.isel(date=ref_dates).drop_vars('date').rename(date='pair')
            intensity_look2 = intensity_look.isel(date=rep_dates).drop_vars('date').rename(date='pair')

            # correlation requires multilooking to detect influence between pixels
            corr_look = (phasediff_look.abs() / (intensity_look1 * intensity_look2).sqrt()).clip(0, 1)

        # keep phase difference without multilooking if multilook=False
        if not multilook or wavelength is None:
            phasediff_look = phasediff
        if goldstein is not None:
            phasediff_look = phasediff_look.goldstein(corr_look, goldstein)

        # filter out not valid pixels
        if weight is not None:
            phasediff_look = phasediff_look.where(np.isfinite(weight))
            corr_look = corr_look.where(np.isfinite(weight)) if corr_look is not None else None
        
        if not complex:
            phasediff_look = phasediff_look.angle()

        # BPR differences aligned with pair dimension: BPR(rep) - BPR(ref)
        bpr = data2[['BPR']].drop_vars('pair') - data1[['BPR']].drop_vars('pair')
        #print ('bpr', bpr.to_dict())

        def as_xarray(batch):
            # Add ref/rep/BPR as coordinates along pair dimension
            # BPR is passed as a Batch for per-burst assignment
            return batch.assign_coords(
                ref=('pair', data1.coords['pair'].values),
                rep=('pair', data2.coords['pair'].values),
                BPR=('pair', bpr)
            ).set_index(pair=['ref', 'rep'])
        
        if corr_look is None:
            return as_xarray(phasediff_look)
        return (as_xarray(phasediff_look), as_xarray(corr_look))

    def phasediff_singlelook(self, *args, **kwarg):
        from .Batch import BatchComplex
        kwarg['multilook'] = False
        return self.phasediff(*args, **kwarg)
        #intfs, corrs = self.phasediff(**kwarg)
        #return BatchWrap(intfs), BatchUnit(corrs)

    def phasediff_multilook(self, *args, **kwarg):
        from .Batch import BatchComplex
        kwarg['multilook'] = True
        return self.phasediff(*args, **kwarg)
