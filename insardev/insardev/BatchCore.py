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
from . import utils_io,  utils_xarray
from typing import TYPE_CHECKING
import operator
import numpy as np
import xarray as xr
if TYPE_CHECKING:
    from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex
    from .Stack import Stack
    import matplotlib

class BatchCore(dict):
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

    @staticmethod
    def wrap(data):
        import numpy as np
        return np.mod(data + np.pi, 2 * np.pi) - np.pi

    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | BatchComplex | None = None):
        from .Stack import Stack
        from .Batch import BatchComplex
        if isinstance(mapping, (Stack, BatchComplex)):
            real_dict = {}
            for key, ds in mapping.items():
                # pick only the data_vars whose dtype is not complex
                real_vars = [v for v in ds.data_vars if ds[v].dtype.kind != 'c' and tuple(ds[v].dims) == ('y', 'x')]
                real_dict[key] = ds[real_vars]
            mapping = real_dict
        dict.__init__(self, mapping or {})

    # def __repr__(self):
    #     if not self:
    #         return f"{self.__class__.__name__}(empty)"
    #     n = len(self)
    #     if n <= 1:
    #         # delegate to the underlying dict repr
    #         return dict.__repr__(self)
    #     sample = next(iter(self.values()))
    #     if not 'date' in sample and not 'pair' in sample:
    #         return f'{self.__class__.__name__} object containing {len(self)} items'
    #     sample_len = f'{len(sample.date)} date' if 'date' in sample else f'{len(sample.pair)} pair'
    #     keys = list(self.keys())
    #     return f'{self.__class__.__name__} object containing {len(self)} items for {sample_len} ({keys[0]} ... {keys[-1]})'

    # def __repr__(self):
    #     if not self:
    #         return f"{self.__class__.__name__}(empty)"
    #     sample = next(iter(self.values()))  # pick any dataset
    #     # figure out which stack coord we have
    #     if 'date' in sample.coords:
    #         count = sample.coords['date'].size
    #         axis_name = 'date'
    #     elif 'pair' in sample.coords:
    #         count = sample.coords['pair'].size
    #         axis_name = 'pair'
    #     else:
    #         # fallback if neither coord is present
    #         return f"{self.__class__.__name__} containing {len(self)} items"
    #     keys = list(self.keys())
    #     return (
    #         f"{self.__class__.__name__} containing {len(self)} items "
    #         f"for {count} {axis_name} "
    #         f"({keys[0]} … {keys[-1]})"
    #     )

    def __repr__(self):
        # empty case
        if not self:
            return f"{self.__class__.__name__}(empty)"

        n = len(self)
        # single‐item: show the actual Dataset repr
        if n == 1:
            key, ds = next(iter(self.items()))
            return f"{self.__class__.__name__}['{key}']:\n{ds!r}"

        # multi‐item: show summary
        sample = next(iter(self.values()))
        if 'date' in sample.coords:
            count = sample.coords['date'].size
            axis = 'date'
        elif 'pair' in sample.coords:
            count = sample.coords['pair'].size
            axis = 'pair'
        else:
            return f"{self.__class__.__name__} containing {n} items"

        keys = list(self.keys())
        return (
            f"{self.__class__.__name__} containing {n} items "
            f"for {count} {axis} "
            f"({keys[0]} … {keys[-1]})"
        )

    @property
    def data(self) -> xr.Dataset:
        """
        Return the single Dataset in this Batch.

        Raises
        ------
        ValueError
            if the Batch has zero or more than one item.
        """
        n = len(self)
        if n != 1:
            raise ValueError(f'Batch.data is only available for single-item Batches, but this Batch has {n} items')
        # return the only Dataset
        return next(iter(self.values()))

    def __getitem__(self, key):
        # like batch[['azi','rng']]
        if isinstance(key, (list, tuple)):
            return type(self)({
                burst_id: ds[key]
                for burst_id, ds in self.items()
            })
        # otherwise fall back to dictionary lookup: batch['033_069722_IW3']
        return super().__getitem__(key)

    def __add__(self, other: Batch):
        keys = self.keys()
        #& other.keys()
        return type(self)({k: self.wrap(self[k] + other[k] if k in other else self[k]) for k in keys})

    def __sub__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self.wrap(self[k] - other[k] if k in other else self[k]) for k in keys})

    def __mul__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self.wrap(self[k] * other[k] if k in other else self[k]) for k in keys})

    def __rmul__(self, other):
        # scalar * batch  → map scalar * each dataset
        return type(self)({
            k: other * v
            for k, v in self.items()
        })

    def __truediv__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self.wrap(self[k] / other[k] if k in other else self[k]) for k in keys})
    
    def _binop(self, other, op):
        """
        generic helper for any binary operator `op(ds, other)` or `op(ds, other_ds)`
        """
        if isinstance(other, (int, float)):
            return type(self)({k: op(ds, other) for k, ds in self.items()})
        elif isinstance(other, Batch):
            common = set(self) & set(other)
            return type(self)({k: op(self[k], other[k]) for k in common})
        else:
            return NotImplemented

    def __gt__(self, other):   return self._binop(other, operator.gt)
    def __lt__(self, other):   return self._binop(other, operator.lt)
    def __ge__(self, other):   return self._binop(other, operator.ge)
    def __le__(self, other):   return self._binop(other, operator.le)
    def __eq__(self, other):   return self._binop(other, operator.eq)
    def __ne__(self, other):   return self._binop(other, operator.ne)

    # reversed ops
    __rgt__ = __gt__
    __rlt__ = __lt__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        np.exp(-1j * intfs)
        """
        from .Batch import BatchWrap
        # only handle the normal call
        if method != "__call__":
            return NotImplemented

        # find the first BatchWrap among inputs
        batch = next((x for x in inputs if isinstance(x, BatchWrap)), None)
        if batch is None:
            return NotImplemented

        result = {}
        for k in batch.keys():
            # build the argument list for this key
            args = [
                inp[k] if isinstance(inp, BatchWrap) else inp
                for inp in inputs
            ]
            result[k] = ufunc(*args, **kwargs)

        return type(self)(result)

    # def iexp(self):
    #     """
    #     np.exp(-1j * intfs)
    #     """
    #     import numpy as np
    #     return np.exp(1j * self)

    # def conj(self) -> BatchWrap:
    #     """
    #     Return a new BatchWrap in which each complex dataset has been
    #     replaced with its complex conjugate.

    #     Example:
    #     intfs.iexp().conj() for np.exp(-1j * intfs)
    #     """
    #     return type(self)({
    #         k: ds.conj()
    #         for k, ds in self.items()
    #     })

    def map_da(self, func, **kwargs):
        """Apply func(DataArray) → DataArray to every var in every dataset."""
        return type(self)({
            k: ds.map(func, **kwargs)
            for k, ds in self.items()
        })

    def astype(self, dtype, **kwargs):
        return self.map_da(lambda da: da.astype(dtype), **kwargs)
    
    def abs(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.abs(da), **kwargs)

    def square(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.square(da), **kwargs)

    def sqrt(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.sqrt(da), **kwargs)

    def clip(self, min=None, max=None, **kwargs):
        return self.map_da(lambda da: da.clip(min=min, max=max), **kwargs)

    def isfinite(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.isfinite(da), **kwargs)

    # def where(self, cond, other=0):
    #     # cond can be a BatchWrap of booleans
    #     if isinstance(cond, BatchWrap):
    #         return type(self)({
    #             k: ds.where(cond[k], other)
    #             for k, ds in self.items()
    #         })
    #     else:
    #         return self.map_da(lambda da: da.where(cond, other), keep_attrs=True)

    # def where(self, cond, other=0, **kwargs):
    #     """
    #     Batch‐wise .where:
        
    #     - if `cond` is a Batch (or BatchWrap) with exactly the same keys:
    #         * when other==0 → do ds * mask  (very fast, no alignment)
    #         * otherwise   → ds.where(mask, other, **kwargs)
    #     - else:
    #         broadcast a single mask or scalar/DataArray
    #         into every var via `map_da(lambda da: da.where(cond, other, **kwargs))`.
    #     """
    #     # per‐burst mask
    #     if hasattr(cond, 'keys') and set(cond.keys()) == set(self.keys()):
    #         print ('X')
    #         out = {}
    #         for k, ds in self.items():
    #             mask = cond[k]
    #             # if mask coords don't exactly match ds, you can
    #             # uncomment the next line to reindex first:
    #             # mask = mask.reindex_like(ds, method='nearest')
                
    #             if other == 0:
    #                 # blaze past .where with a simple multiply
    #                 out[k] = ds * mask
    #             else:
    #                 out[k] = ds.where(mask, other, **kwargs)
    #         return type(self)(out)

    #     # single‐mask/scalar-broadcast case:
    #     return self.map_da(lambda da: da.where(cond, other, **kwargs), **kwargs)


    # def where(self, cond, other=0, **kwargs):
    #     """
    #     Batch-wise .where: if cond is another Batch with exactly the same keys,
    #     do each ds.where(mask, other), otherwise fall back to per-DataArray broadcast.
    #     """
    #     # 1) fast path: cond is a Batch with the same bursts
    #     if isinstance(cond, Batch) and set(cond.keys()) == set(self.keys()):
    #         return type(self)({
    #             k: ds.where(cond[k], other, **kwargs)
    #             for k, ds in self.items()
    #         })

    #     # 2) broadcast a single mask/scalar to every var
    #     return self.map_da(lambda da: da.where(cond, other, **kwargs), **kwargs)

    def where(self, cond, other=np.nan, **kwargs):
        """
        Fast batch-wise .where:

        If `cond` is a Batch (or subclass) with exactly the same keys,
           and each cond[k] is a 1-variable Dataset or a DataArray,
           we extract the single DataArray mask and do either:
             - other==0 → simple multiply ds * mask_da
             - else       → ds.where(mask_da, other, **kwargs)

        Otherwise fall back to per-DataArray map_da (slower).

        keep_attrs=True argument can be used to preserve attributes of the original data.
        """
        # detect same key Batch-like mask
        if hasattr(cond, 'keys') and set(cond.keys()) == set(self.keys()):
            out = {}
            for k, ds in self.items():
                mask_obj = cond[k]
                # extract DataArray from a 1-var Dataset or use it direct
                if isinstance(mask_obj, xr.Dataset):
                    data_vars = list(mask_obj.data_vars)
                    if len(data_vars) != 1:
                        raise ValueError(f"Batch.where: expected 1 var in mask for '{k}', got {data_vars}")
                    mask_da = mask_obj[data_vars[0]]
                else:
                    mask_da = mask_obj
                out[k] = ds.where(mask_da, other, **kwargs)
            return type(self)(out)

        # fallback: single scalar or DataArray broadcast
        # DataArray case seems not usefull because Batch datasets differ in shape
        return self.map_da(lambda da: da.where(cond, other, **kwargs), **kwargs)

    def __pow__(self, exponent, **kwargs):
        return self.map_da(lambda da: da**exponent, **kwargs)
    
    def power(self, **kwargs):
        """ element-wise |x|², i.e. signal intensity """
        return self.map_da(lambda da: xr.ufuncs.abs(da)**2, **kwargs)

    # def abs(self):
    #     """ element-wise absolute value """
    #     return type(self)({k: ds.map(lambda da: da.abs()) for k, ds in self.items()})

    # def sqrt(self):
    #     """ element-wise square-root """
    #     return type(self)({k: ds.map(lambda da: da.sqrt()) for k, ds in self.items()})

    # def square(self):
    #     """ element-wise square """
    #     return type(self)({k: ds.map(lambda da: da**2) for k, ds in self.items()})

    # def clip(self, min_, max_):
    #     """ element-wise clip to [min_, max_] """
    #     return type(self)({k: ds.map(lambda da: da.clip(min_, max_)) for k, ds in self.items()})

    # def where(self, cond, other=np.nan):
    #     """
    #     like xarray.where: keep ds where cond is True, else fill with other.
    #     `cond` may be a scalar, a DataArray, or another Batch with the same keys.
    #     """
    #     if isinstance(cond, Batch):
    #         return type(self)({
    #             k: ds.where(cond[k], other)
    #             for k, ds in self.items()
    #         })
    #     else:
    #         return type(self)({
    #             k: ds.where(cond, other)
    #             for k, ds in self.items()
    #         })

    # def isfinite(self):
    #     """ element-wise finite mask """
    #     return type(self)({k: ds.map(lambda da: np.isfinite(da)) for k, ds in self.items()})

    # def sel(self, keys: dict|list|str):
    #     if isinstance(keys, str):
    #         keys = [keys]
    #     return type(self)({k: self[k] for k in (keys if isinstance(keys, list) else keys.keys())})

    def sel(self, keys: dict|list|str| pd.DataFrame):
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

    # def isel(self, indices):
    #     """Select by integer locations (like xarray .isel)."""
    #     import numpy as np

    #     keys = list(self.keys())
    #     # allow a single integer, a list of ints, or a slice
    #     if isinstance(indices, (int, np.integer)):
    #         idxs = [indices]
    #     elif isinstance(indices, slice):
    #         idxs = list(range(*indices.indices(len(keys))))
    #     else:
    #         idxs = list(indices)
    #     selected = {keys[i]: self[keys[i]] for i in idxs }
    #     return type(self)(selected)

    # def isel(self, indices=None, **indexers):
    #     """
    #     Select by integer locations, either by a single positional index/slice
    #     (applied over the *keys* of the batch) OR by keyword dimension selectors
    #     (delegated to each xarray.Dataset.isel).
    #     """
    #     # xarray‐style keyword isel
    #     if indexers:
    #         return type(self)({
    #             k: ds.isel(**indexers)
    #             for k, ds in self.items()
    #         })

    #     # positional isel over the batch keys (old behavior)
    #     import numpy as np
    #     keys = list(self.keys())
    #     if indices is None:
    #         return type(self)(dict(self))  # no selection
    #     if isinstance(indices, (int, np.integer)):
    #         idxs = [indices]
    #     elif isinstance(indices, slice):
    #         idxs = list(range(*indices.indices(len(keys))))
    #     else:
    #         idxs = list(indices)
    #     return type(self)({
    #         keys[i]: self[keys[i]]
    #         for i in idxs
    #     })

    def isel(self, indices=None, **indexers):
        """
        Select by integer locations, either by:
        keyword dimension selectors (delegated to each xarray.Dataset.isel)
        a single positional index/slice/list over the *keys* of the batch
        (NEW) a single dict positional argument of dimension indexers
        """
        import numpy as np

        # === NEW: if they passed a dict as the first argument, treat it like keyword indexers ===
        if isinstance(indices, dict):
            indexers = indices
            indices = None

        # xarray‐style keyword isel (including now dict-via-positional)
        if indexers:
            return type(self)({
                k: ds.isel(**indexers)
                for k, ds in self.items()
            })

        # fallback: positional isel over the batch keys (old behavior)
        keys = list(self.keys())
        if indices is None:
            return type(self)(dict(self))  # no selection
        if isinstance(indices, (int, np.integer)):
            idxs = [indices]
        elif isinstance(indices, slice):
            idxs = list(range(*indices.indices(len(keys))))
        else:
            idxs = list(indices)

        return type(self)({
            keys[i]: self[keys[i]]
            for i in idxs
        })

    def drop_vars(self, names):
        """Return a new Batch with those data-vars removed from each dataset."""
        if isinstance(names, str):
            names = [names]
        return type(self)({
            k: ds.drop_vars(names)
            for k, ds in self.items()
        })

    def rename_vars(self, **kw):
        return type(self)({k: ds.rename_vars(**kw) for k, ds in self.items()})

    def reindex(self, **kw):
        return type(self)({k: ds.reindex(**kw) for k, ds in self.items()})

    def interp(self, **kw):
        return type(self)({k: ds.interp(**kw) for k, ds in self.items()})

    def interp_like(self, other: Batch, **interp_kwargs):
        """Regrid each Dataset onto the coords of the *corresponding* Dataset in `other`."""
        return type(self)({
            k: ds.interp_like(other[k], **interp_kwargs)
            for k, ds in self.items() if k in other
        })

    def transpose(self, *dims, **kw):
        return type(self)({k: ds.transpose(*dims, **kw) for k, ds in self.items()})

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Internal helper for aggregation methods.
        If the target object's .<name>() accepts a `dim=` arg, we pass dim, otherwise we just call it without.
        """
        import inspect
        out = {}
        for key, obj in self.items():
            fn = getattr(obj, name)
            sig = inspect.signature(fn)
            if "dim" in sig.parameters:
                out[key] = fn(dim=dim, **kwargs)
            else:
                out[key] = fn(**kwargs)
        return type(self)(out)

    def mean(self, dim=None, **kwargs):
        return self._agg("mean", dim=dim, **kwargs)

    def sum(self, dim=None, **kwargs):
        return self._agg("sum", dim=dim, **kwargs)

    def min(self, dim=None, **kwargs):
        return self._agg("min", dim=dim, **kwargs)

    def max(self, dim=None, **kwargs):
        return self._agg("max", dim=dim, **kwargs)

    def median(self, dim=None, **kwargs):
        return self._agg("median", dim=dim, **kwargs)

    def std(self, dim=None, **kwargs):
        return self._agg("std", dim=dim, **kwargs)

    def var(self, dim=None, **kwargs):
        return self._agg("var", dim=dim, **kwargs)

    # def coarsen(self, window: dict[str,int], **kwargs):
    #     """
    #     intfs.coarsen({'y':2, 'x':8}, boundary='trim').mean().isel(0)
    #     """
    #     return type(self)({
    #         k: ds.coarsen(window, **kwargs)
    #         for k, ds in self.items()
    #     })

    def coarsen(self, window: dict[str, int], **kwargs) -> Batch:
        """
        Coarsen each DataSet in the batch by integer factors and align the 
        blocks so that they fall on "nice" grid boundaries.

        Parameters
        ----------
        window : dict[str,int]
            e.g. {'y': 2, 'x': 8}
        **kwargs
            extra args forwarded into the reduction, e.g. skipna=True.

        Returns
        -------
        Batch
            A new Batch where each Dataset has been sliced for alignment,
            coarsened by `window`, then reduced by `.mean()` (or whichever
            `func` you chose).
        """
        out = {}
        for key, ds in self.items():
            ds2 = ds
            # align each dimension
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds2, dim, factor)
                #print ('start', start)
                if start is not None:
                    ds2 = ds2.isel({dim: slice(start, None)})
            # coarsen
            out[key] = ds2.coarsen(window, **kwargs)

        return type(self)(out)

    def chunk(self, chunks):
        return type(self)({k: ds.chunk(chunks) for k, ds in self.items()})

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def map(self, func, *args, **kwargs):
        return type(self)({k: func(ds, *args, **kwargs) for k, ds in self.items()})

    def compute(self):
        import dask
        from insardev_toolkit import progressbar
        progressbar(result := dask.persist(dict(self))[0], desc=f'Computing Batch...'.ljust(25))
        return type(self)(result)

    @property
    def spacing(self) -> tuple[float, float]:
        """Return the (y, x) grid spacing."""
        sample = next(iter(self.values()))
        return sample.y.diff('y').item(0), sample.x.diff('x').item(0)
    
    def downsample(self, new_spacing: tuple[float, float] | float | int):
        """
        Update the Batch data onto a grid with the given (y, x) spacing.
        Like to coarsening but with cell size in meters instead of pixels:
        intfs.downsample(60)
        intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()
        """
        if isinstance(new_spacing, (int, float)):
            new_spacing = (new_spacing, new_spacing)
        dy, dx = self.spacing
        yscale, xscale = int(np.round(new_spacing[0]/dy)), int(np.round(new_spacing[1]/dx))
        print (f'DEBUG: cell size in meters: y={dy:.1f}, x={dx:.1f} -> y={new_spacing[0]:.1f}, x={new_spacing[1]:.1f}')
        return self.coarsen({'y': yscale, 'x': xscale}, boundary='trim').mean()

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

    def to_dataset(self, polarization=None, dissolve=True, compute: bool = False):
        """
        This function is a faster implementation for the standalone function combination of xr.concat and xr.align:
        xr.concat(xr.align(*intfs, join='outer'), dim='stack_dim').ffill('stack_dim').isel(stack_dim=-1).compute()
        #xr.concat(xr.align(*datas, join='outer'), dim='stack_dim').mean('stack_dim').compute()

        Parameters
        ----------
        datas: xr.Dataset | xr.DataArray | dict[str, xr.Dataset | xr.DataArray] | None
            The datasets to concatenate.
        wrap: bool | None
            There are three options:
            - None: return the topmost burst in chronological order for overlapping areas
            - True: compute the circular mean
            - False: compute the arithmetic mean
        compute: bool
            Whether to compute the result.
        """
        import xarray as xr
        import numpy as np
        import dask
        from insardev_toolkit import progressbar, datagrid
        from .Batch import BatchWrap

        if not len(self):
            return None

        wrap = True if type(self) == BatchWrap else False
        print ('dtype', type(self), 'wrap', wrap, 'dissolve', dissolve, 'polarization', polarization)

        #print (type(datas))
        
        sample = next(iter(self.values()))
        if len(self) == 1:
            if compute:
                progressbar(sample := sample.persist(), desc=f'Compute Dataset'.ljust(25))
                return sample
            return sample

        if polarization is None:
            # find all variables in the first dataset related to polarizations
            # TODO
            polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in sample.data_vars]

            # process list of datasets with one or multiple polarizations
            das = xr.merge([self.to_dataset(polarization=pol, dissolve=dissolve) for pol in polarizations])
            if compute:
                progressbar(das := das.persist(), desc=f'Computing Dataset...'.ljust(25))
                return das
            return das
        
        # process single polarization dataarrays
        datas = [self[k][polarization] for k in self.keys()]

        # process list of dataarrays with single polarization
        y_chunksize = datas[0].chunks[-2][0]
        x_chunksize = datas[0].chunks[-1][0]
        # define unified grid
        y_min = min(ds.y.min().item() for ds in datas)
        y_max = max(ds.y.max().item() for ds in datas)
        x_min = min(ds.x.min().item() for ds in datas)
        x_max = max(ds.x.max().item() for ds in datas)
        #print (y_min, y_max, x_min, x_max, y_max-y_min, x_max-x_min)
        stackvar = list(datas[0].dims)[0]
        # workaround for dask.array.blockwise
        stackval = datas[0][stackvar].astype(str)
        stackidx = xr.DataArray(np.arange(len(stackval), dtype=int), dims=('z',))
        dy = datas[0].y.diff('y').item(0)
        dx = datas[0].x.diff('x').item(0)
        #print ('dy, dx', dy, dx)
        ys = xr.DataArray(np.arange(y_min, y_max + dy/2, dy), dims=['y'])
        xs = xr.DataArray(np.arange(x_min, x_max + dx/2, dx), dims=['x'])
        #print ('stack', stackvar, stackval)
        #print ('ys', ys)
        #print ('xs', xs)
        # extract extents of all datasets once
        extents = [(float(da.y.min()), float(da.y.max()), float(da.x.min()), float(da.x.max())) for da in datas]
        
        # use outer variable datas
        def block_dask(stack, y_chunk, x_chunk, wrap):
            #print ('pair', pair)
            #print ('concat: block_dask', stackvar, stack)
            # extract extent of the current chunk once
            ymin0, ymax0 = float(y_chunk.min()), float(y_chunk.max())
            xmin0, xmax0 = float(x_chunk.min()), float(x_chunk.max())
            # select all datasets overlapping with the current chunk
            das_slice = [da.isel({stackvar: stackidx}).sel({'y': slice(ymin0, ymax0), 'x': slice(xmin0, xmax0)}).compute(num_workers=1)
                         for da, (ymin, ymax, xmin, xmax) in zip(datas, extents)
                         if ymin0 < ymax and ymax0 > ymin and xmin0 < xmax and xmax0 > xmin]
            #print ('concat: das_slice', len(das_slice), [da.shape for da in das_slice])
            
            fill_dtype = datas[0].dtype
            fill_nan = np.nan * np.ones((), dtype=fill_dtype)
            if len(das_slice) == 0:
                # return empty block
                return np.full((stack.size, y_chunk.size, x_chunk.size), fill_nan, dtype=fill_dtype)
            #das_block = [da.reindex({'y': y_chunk, 'x': x_chunk}, fill_value=fill_nan, copy=False) for da in das_slice if da.size > 0]
            das_block = [da.reindex({'y': y_chunk, 'x': x_chunk}, fill_value=fill_nan, copy=False) for da in das_slice]
            del das_slice
            if len(das_block) == 1:
                # return single block as is
                return das_block[0].values

            if not dissolve:
                #print ('wrap None')
                # ffill does not work correct on complex data and per-component ffill is faster
                # the magic trick is to use sorting to ensure burst overpapping order
                # bursts ends should be overlapped by bursts starts
                das_block_concat = xr.concat(das_block, dim="stack_dim", join="inner")
                if np.issubdtype(das_block_concat.dtype, np.complexfloating):
                    return (das_block_concat.real.ffill("stack_dim").isel(stack_dim=-1)
                            + 1j*das_block_concat.imag.ffill("stack_dim").isel(stack_dim=-1)).values
                else:
                    return das_block_concat.ffill("stack_dim").isel(stack_dim=-1).values
            elif wrap == True:
                #print ('wrap True')
                # calculate circular mean for interferogram data
                das_block_concat = xr.concat([np.exp(1j * da) for da in das_block], dim='stack_dim')
                block_complex = das_block_concat.mean('stack_dim', skipna=True).values
                return np.arctan2(block_complex.imag, block_complex.real)
            elif wrap == False:
                #print ('wrap False')
                das_block_concat = xr.concat(das_block, dim="stack_dim", join="inner")
                # calculate arithmetic mean for phase and correlation data
                return das_block_concat.mean('stack_dim', skipna=True).values
            else:
                raise ValueError(f'ERROR: wrap is not a boolean or None: {wrap}')

        # prevent warnings 'PerformanceWarning: Increasing number of chunks by factor of ...'
        import warnings
        from dask.array.core import PerformanceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerformanceWarning)
            # rechunk data for expected usage
            data = dask.array.blockwise(
                block_dask,
                'zyx',
                stackidx.chunk(1), 'z',
                ys.chunk({'y': y_chunksize}), 'y',
                xs.chunk({'x': x_chunksize}), 'x',
                meta = np.empty((0, 0, 0), dtype=datas[0].dtype),
                wrap=wrap
            )
        da = xr.DataArray(data, coords={stackvar: stackval, 'y': ys, 'x': xs})\
            .rename(datas[0].name)\
            .assign_attrs(datas[0].attrs)
        return datagrid.spatial_ref(da, datas)

    def plot(self,
            dissolve: bool = True,
            cmap: matplotlib.colors.Colormap | str | None = 'gist_rainbow_r',
            vmin: float | None = None,
            vmax: float | None = None,
            quantile: float | None = None,
            symmetrical: bool = False,
            caption: str = '',
            cols: int = 4,
            rows: int = 4,
            size: float = 4,
            nbins: int = 5,
            aspect: float = 1.02,
            y: float = 1.05,
            _size: tuple[int, int] | None = None,
            ):
        import xarray as xr
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from .Batch import BatchWrap
        from insardev_toolkit import progressbar

        # no data means no plot and no error
        if not len(self):
            return
        
        wrap = True if type(self) == BatchWrap else False
        print ('dtype', type(self), 'wrap', wrap, 'dissolve', dissolve)

        # screen size in pixels (width, height) to estimate reasonable number pixels per plot
        # this is quite large to prevent aliasing on 600dpi plots without additional processing
        if _size is None:
            _size = (8000,4000)

        # use outer variables
        def plot_polarization(polarization):
            stackvar = list(sample.dims)[0]
            #print ('stackvar', stackvar)
            #da = self[[polarization]]
            #da = self[[polarization]].isel({stackvar: slice(0, rows)})
            da = self[[polarization]].isel({stackvar: slice(0, rows)}).to_dataset(dissolve=dissolve)[polarization]
            #print ('da', da)
            if 'stack' in da.dims and isinstance(da.coords['stack'].to_index(), pd.MultiIndex):
                da = da.unstack('stack')
            
            # there is no reason to plot huge arrays much larger than screen size for small plots
            #print ('screen_size', screen_size)
            size_y, size_x = da.shape[1:]
            #print ('size_x, size_y', size_x, size_y)
            factor_y = int(np.round(size_y / (_size[1] / rows)))
            factor_x = int(np.round(size_x / (_size[0] / cols)))
            #print ('factor_x, factor_y', factor_x, factor_y)
            # decimate for faster plot, do not coarsening without antialiasing
            # maybe data is already smoothed and maybe not, decimation is the only safe option
            da = da[:,::max(1, factor_y), ::max(1, factor_x)]
            # materialize for all the calculations and plotting
            progressbar(da := da.persist(), desc=f'Computing {polarization} Plot'.ljust(25))

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
            return fg

        if quantile is not None:
            assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

        sample = next(iter(self.values()))
        # find all variables in the first dataset related to polarizations
        # TODO
        polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in sample.data_vars]
        print ('polarizations', polarizations)

        # process polarizations one by one
        fgs = []
        for pol in polarizations:
            fg = plot_polarization(polarization=pol)
            fgs.append(fg)
        return fgs

    def gaussian(
        self,
        weight: BatchUnit | None = None,
        wavelength: float | None = None,
        threshold: float = 0.5,
        debug: bool = False
    ) -> Batch:
        """
        2D (yx) Gaussian kernel smoothing (multilook) on each dataset in this Batch.

        Parameters
        ----------
        weight : BatchUnit or None
            A Batch of 2D DataArrays, one per key, matching this Batch's keys.
            If None, no weighting is applied.
        wavelength : float or None
            Gaussian sigma via 5.3 cutoff formula. Must be positive if provided.
        threshold : float
            Drop-off threshold for the kernel.
        debug : bool
            Print sigma values if True.

        Returns
        -------
        Batch
            A new Batch with the same keys, each smoothed by its corresponding weight.
        """
        import xarray as xr
        import numpy as np
        import dask
        from .utils_gaussian import nanconvolve2d_gaussian
        from .Batch import BatchUnit
        # constant 5.3 defines half-gain at filter_wavelength
        cutoff = 5.3

        # validate weight if provided
        if weight is not None:
            if not isinstance(weight, BatchUnit) or set(weight.keys()) != set(self.keys()):
                raise ValueError('`weight` must be a BatchUnit with the same keys as `self`')

        # validate wavelength if provided
        if wavelength is not None:
            if wavelength <= 0:
                raise ValueError('wavelength must be positive')
            # precompute dy, dx, σ‐factors
            dy, dx = self.spacing
            sig_y = wavelength / (dy * cutoff)
            sig_x = wavelength / (dx * cutoff)
            if debug:
                print(f'DEBUG: multilooking sigmas ({sig_y:.2f}, {sig_x:.2f}), wavelength {wavelength:.1f}')
            sigmas = (sig_y, sig_x)
        else:
            sigmas = None

        out = {}
        # loop over each key
        for key, ds in self.items():
            w = weight[key] if weight is not None else None

            def gaussian_da(da: xr.DataArray, w: xr.DataArray | None = None) -> xr.DataArray:
                # apply the 2D Gaussian convolver slice‐by‐slice
                def one_slice(arr):
                    return nanconvolve2d_gaussian(
                        arr, w, sigmas, threshold=threshold
                    )

                # handle an optional first dimension (e.g. 'pair' or 'time')
                if da.ndim == 3:
                    dim0 = da.dims[0]
                    pieces = [
                        one_slice(da.sel({dim0: v}))
                        for v in da.coords[dim0].values
                    ]
                    out_da = xr.concat(
                        [xr.DataArray(p, dims=da.dims[1:], coords={d: da.coords[d] for d in da.dims[1:]})
                         for p in pieces],
                        dim=dim0
                    ).assign_coords({dim0: da.coords[dim0]})
                else:
                    out_da = xr.DataArray(
                        one_slice(da),
                        dims=da.dims,
                        coords=da.coords
                    )

                # preserve original chunking
                return out_da.chunk({d: da.chunks[i][0] for i, d in enumerate(da.dims)})

            # apply to every 2D or 3D (yx) var in the Dataset
            out[key] = xr.Dataset({
                var: gaussian_da(ds[var], w[var] if w is not None else None)
                for var in ds.data_vars
                if (ds[var].ndim in (2,3) and ds[var].dims[-2:] == ('y','x'))
            })

        return type(self)(out)

