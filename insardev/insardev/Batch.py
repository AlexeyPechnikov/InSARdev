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
from insardev_toolkit import datagrid, progressbar
from . import utils_io
from . import utils_xarray

class Batch(dict):
    """
    intfs60_detrend = Batch(intfs60) - Batch(intfs60_trend)

    dss = intfs60_detrend.sel(['106_226487_IW2','106_226488_IW2','106_226489_IW2','106_226490_IW2','106_226491_IW2'])
    dss_fixed = dss + {'106_226490_IW2': 2.6, '106_226491_IW2': 3})

    intfs60_detrend.isel(1)
    intfs60_detrend.isel([0, 2])
    intfs60_detrend.isel(slice(1, None))
    """
    def __init__(self, mapping:dict|None=None):
        dict.__init__(self, mapping or {})

    def __repr__(self):
        if not self:
            return f"{self.__class__.__name__}(empty)"
        n = len(self)
        if n <= 1:
            # delegate to the underlying dict repr
            return dict.__repr__(self)
        sample = next(iter(self.values()))
        sample_len = f'{len(sample.date)} date' if 'date' in sample else f'{len(sample.pair)} pair'
        keys = list(self.keys())
        return f'{self.__class__.__name__} object containing {len(self)} items for {sample_len} ({keys[0]} ... {keys[-1]})'

    def __add__(self, other: 'Batch'):
        keys = self.keys()
        #& other.keys()
        return Batch({k: (self[k] + other[k] if k in other else self[k]) for k in keys})

    def __sub__(self, other: 'Batch'):
        keys = self.keys()
        return Batch({k: (self[k] - other[k] if k in other else self[k]) for k in keys})

    def __mul__(self, other: 'Batch'):
        keys = self.keys()
        return Batch({k: (self[k] * other[k] if k in other else self[k]) for k in keys})

    def __truediv__(self, other: 'Batch'):
        keys = self.keys()
        return Batch({k: (self[k] / other[k] if k in other else self[k]) for k in keys})
    
    def sel(self, keys: dict|list|str):
        if isinstance(keys, str):
            keys = [keys]
        return Batch({k: self[k] for k in (keys if isinstance(keys, list) else keys.keys())})

    def isel(self, indices):
        """Select by integer locations (like xarray .isel)."""
        import numpy as np

        keys = list(self.keys())
        # allow a single integer, a list of ints, or a slice
        if isinstance(indices, (int, np.integer)):
            idxs = [indices]
        elif isinstance(indices, slice):
            idxs = list(range(*indices.indices(len(keys))))
        else:
            idxs = list(indices)
        selected = {keys[i]: self[keys[i]] for i in idxs }
        return Batch(selected)

    def compute(self):
        import dask
        progressbar(result := dask.persist(dict(self))[0], desc=f'Computing Batch...'.ljust(25))
        return Batch(result)

    def save(self, store: str, storage_options: dict[str, str] | None = None, compat: bool = True, chunksize: int|str = 'auto',
                caption: str | None = 'Saving...', n_jobs: int = -1, debug=False):
        return utils_io.save(dict(self), store=store, storage_options=storage_options, compat=compat, chunksize=chunksize, caption=caption, n_jobs=n_jobs, debug=debug)

    def open(self, store: str, storage_options: dict[str, str] | None = None, compat: bool = True, chunksize: int|str = 'auto',
                n_jobs: int = -1, debug=False):
        data = utils_io.open(store=store, storage_options=storage_options, compat=compat, chunksize=chunksize, n_jobs=n_jobs, debug=debug)
        self.update(data)
        return self
    
    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                compat: bool = True, n_jobs: int = -1, debug=False):
        #return utils_io.snapshot(*args, store=store, storage_options=storage_options, compat=compat, n_jobs=n_jobs, debug=debug)
        self.save(store=store, storage_options=storage_options, compat=compat, n_jobs=n_jobs, debug=debug)
        return self.open(store=store, storage_options=storage_options, compat=compat, n_jobs=n_jobs, debug=debug)

class BatchWrap(Batch):
    def __init__(self, mapping:dict|None=None):
        dict.__init__(self, mapping or {})
