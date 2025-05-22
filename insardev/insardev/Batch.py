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
from __future__ import annotations
from insardev_toolkit import progressbar
from . import utils_io,  utils_xarray

class Batch(dict):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores real values (correlation and unwrapped phase).
    
    Examples:
    intfs60_detrend = Batch(intfs60) - Batch(intfs60_trend)

    dss = intfs60_detrend.sel(['106_226487_IW2','106_226488_IW2','106_226489_IW2','106_226490_IW2','106_226491_IW2'])
    dss_fixed = dss + {'106_226490_IW2': 2.6, '106_226491_IW2': 3})

    intfs60_detrend.isel(1)
    intfs60_detrend.isel([0, 2])
    intfs60_detrend.isel(slice(1, None))
    """
    import xarray as xr

    @staticmethod
    def wrap(data):
        import numpy as np
        return np.mod(data + np.pi, 2 * np.pi) - np.pi

    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | BatchComplex | None = None):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchComplex)):
            real_dict = {}
            for key, ds in mapping.items():
                # pick only the data_vars whose dtype is not complex
                real_vars = [v for v in ds.data_vars if ds[v].dtype.kind != 'c' and tuple(ds[v].dims) == ('y', 'x')]
                real_dict[key] = ds[real_vars]
            mapping = real_dict
        dict.__init__(self, mapping or {})

    def __repr__(self):
        if not self:
            return f"{self.__class__.__name__}(empty)"
        n = len(self)
        if n <= 1:
            # delegate to the underlying dict repr
            return dict.__repr__(self)
        sample = next(iter(self.values()))
        if not 'date' in sample and not 'pair' in sample:
            return f'{self.__class__.__name__} object containing {len(self)} items'
        sample_len = f'{len(sample.date)} date' if 'date' in sample else f'{len(sample.pair)} pair'
        keys = list(self.keys())
        return f'{self.__class__.__name__} object containing {len(self)} items for {sample_len} ({keys[0]} ... {keys[-1]})'

    def __getitem__(self, key):
        # like batch[['azi','rng']]
        if isinstance(key, (list, tuple)):
            return type(self)({
                burst_id: ds[key]
                for burst_id, ds in self.items()
            })
        # otherwise fall back to dictionary lookup: batch['033_069722_IW3']
        return super().__getitem__(key)

    def __add__(self, other: 'Batch'):
        keys = self.keys()
        #& other.keys()
        return type(self)({k: self.wrap(self[k] + other[k] if k in other else self[k]) for k in keys})

    def __sub__(self, other: 'Batch'):
        keys = self.keys()
        return type(self)({k: self.wrap(self[k] - other[k] if k in other else self[k]) for k in keys})

    def __mul__(self, other: 'Batch'):
        keys = self.keys()
        return type(self)({k: self.wrap(self[k] * other[k] if k in other else self[k]) for k in keys})

    def __truediv__(self, other: 'Batch'):
        keys = self.keys()
        return type(self)({k: self.wrap(self[k] / other[k] if k in other else self[k]) for k in keys})
    
    def sel(self, keys: dict|list|str):
        if isinstance(keys, str):
            keys = [keys]
        return type(self)({k: self[k] for k in (keys if isinstance(keys, list) else keys.keys())})

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
        return type(self)(selected)

    def compute(self):
        import dask
        progressbar(result := dask.persist(dict(self))[0], desc=f'Computing Batch...'.ljust(25))
        return type(self)(result)

    def save(self, store: str, storage_options: dict[str, str] | None = None, chunksize: int|str = 'auto',
                caption: str | None = 'Saving...', n_jobs: int = -1, debug=False):
        return utils_io.save(self, store=store, storage_options=storage_options, compat=False, chunksize=chunksize, caption=caption, n_jobs=n_jobs, debug=debug)

    def open(self, store: str, storage_options: dict[str, str] | None = None, chunksize: int|str = 'auto', n_jobs: int = -1, debug=False):
        data = utils_io.open(store=store, storage_options=storage_options, compat=False, chunksize=chunksize, n_jobs=n_jobs, debug=debug)
        if not isinstance(data, dict):
            raise ValueError(f'ERROR: open() returns multiple datasets, you need to use Stack class to open them.')
        return data
    
    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None, chunksize: int|str = 'auto',
                caption: str | None = 'Snapshotting...', n_jobs: int = -1, debug=False):
        self.save(store=store, storage_options=storage_options, chunksize=chunksize, caption=caption, n_jobs=n_jobs, debug=debug)
        return self.open(store=store, storage_options=storage_options, chunksize=chunksize, n_jobs=n_jobs, debug=debug)

class BatchWrap(Batch):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores wrapped phase (real values).
    """
    def __init__(self, mapping:dict|None=None):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchComplex)):
            raise ValueError(f'ERROR: BatchWrap does not support Stack or BatchComplex objects.')
        dict.__init__(self, mapping or {})

class BatchComplex(Batch):
    """
    This class has 'data' stack variable for the datasets in the dict.
    """
    def __init__(self, mapping:dict|None=None):
        dict.__init__(self, mapping or {})

    def sel(self, keys: dict|list|str):
        import pandas as pd
        import numpy as np

        if not isinstance(keys, pd.DataFrame):
            if isinstance(keys, str):
                keys = [keys]
            return type(self)({k: self[k] for k in (keys if isinstance(keys, list) else keys.keys())})

        dss = {}
        # iterate all burst groups
        for id in keys.index.get_level_values(0).unique():
            # select all records for the current burst group
            records = keys[keys.index.get_level_values(0)==id]
            # filter dates
            dates = records.startTime.dt.date.values.astype(str)
            ds = self[id].sel(date=dates)
            # filter polarizations
            pols = records.polarization.unique()
            if len(pols) > 1:
                raise ValueError(f'ERROR: Inconsistent polarizations found for the same burst: {id}')
            elif len(pols) == 0:
                raise ValueError(f'ERROR: No polarizations found for the burst: {id}')
            pols = pols[0]
            if ',' in pols:
                pols = pols.split(',')
            if isinstance(pols, str):
                pols = [pols]
            count = 0
            if np.unique(pols).size < len(pols):
                raise ValueError(f'ERROR: defined polarizations {pols} are not unique.')
            if len([pol for pol in pols if pol in ds.data_vars]) < len(pols):
                raise ValueError(f'ERROR: defined polarizations {pols} are not available in the dataset: {id}')
            for pol in [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in ds.data_vars]:
                if pol not in pols:
                    ds = ds.drop(pol)
                else:
                    count += 1
            if count == 0:
                raise ValueError(f'ERROR: No valid polarizations found for the burst: {id}')
            dss[id] = ds
        return type(self)(dss)
