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
import operator
import numpy as np
import xarray as xr
from collections.abc import Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex
    from .Stack import Stack
    import rasterio as rio
    import pandas as pd
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

    class CoordCollection:
        def __init__(self, ds):
            self._ds = ds
        def __getitem__(self, key):
            return self._ds.coords[key]
        def __contains__(self, key):
            return key in self._ds.coords
        def get(self, key, default=None):
            return self._ds.coords.get(key, default)
        def keys(self):
            return self._ds.coords.keys()
        def values(self):
            return self._ds.coords.values()
        def items(self):
            return self._ds.coords.items()

    def __init__(self, mapping: Mapping[str, xr.Dataset] | Stack | BatchComplex | None = None):
        from .Stack import Stack
        from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex
        #print('BatchCore __init__', 0 if mapping is None else len(mapping))
        # Batch/etc. initialization won't filter out the data when it's a child class of BatchCore
        if isinstance(mapping, (Stack, BatchComplex)) and not isinstance(self, (Batch, BatchWrap, BatchUnit, BatchComplex)):
            real_dict = {}
            for key, ds in mapping.items():
                # pick only the data_vars whose dtype is not complex
                real_vars = [v for v in ds.data_vars if ds[v].dtype.kind != 'c' and tuple(ds[v].dims) == ('y', 'x')]
                real_dict[key] = ds[real_vars]
            mapping = real_dict
        #print('BatchCore __init__ mapping', mapping or {}, '\n')
        super().__init__(mapping or {})

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
        
        # Handle CoordCollection objects
        if isinstance(sample, self.CoordCollection):
            keys = list(self.keys())
            return f"{self.__class__.__name__} coords containing {n} items ({keys[0]} … {keys[-1]})"
        
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

    def __or__(self, other):
        # Batch | Mapping
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = dict(self)
        merged.update(other)
        return type(self)(merged)

    def __ror__(self, other):
        # Mapping | Batch
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = dict(other)
        merged.update(self)
        return type(self)(merged)

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

    # @property
    # def chunks(self) -> tuple[int, int, int]:
    #     sample = next(iter(self.values()))
    #     # for DatasetCoarsen extract the original Dataset
    #     if hasattr(sample, 'obj'):
    #         sample = sample.obj
    #     data_var = [var for var in sample.data_vars if (sample[var].ndim in (2,3) and sample[var].dims[-2:] == ('y','x'))][0]

    #     if sample[data_var].chunks is None:
    #         print ('WARNING: Batch.chunks undefined, i.e. the data is not lazy and parallel chunks processing is not possible.')
    #         return (1, -1, -1)
    #     else:
    #         return tuple(chunks[0] for chunks in sample[data_var].chunks)

    @property
    def crs(self) -> rio.crs.CRS:
        return next(iter(self.values())).rio.crs

    @property
    def chunks(self) -> dict[str, int]:
        try:
            sample = next(iter(self.values()))
        except StopIteration:
            return {}

        # for DatasetCoarsen extract the original Dataset
        if hasattr(sample, 'obj'):
            sample = sample.obj
        data_var = [var for var in sample.data_vars if (sample[var].ndim in (2,3) and sample[var].dims[-2:] == ('y','x'))][0]
        
        chunks = sample[data_var].chunks
        #print ('chunks', chunks)
        if chunks is None:
            print ('WARNING: Batch.chunks undefined, i.e. the data is not lazy and parallel chunks processing is not possible.')
            # use "common" chunking for 2D and 3D data
            return -1 if data_var.ndim == 2 else (1, -1, -1)

        # build dict of first‐chunk sizes, one chunk means chunk size 1 or -1
        return {dim: sizes[0] if len(sizes) > 1 else (1 if sizes[0] == 1 else -1) for dim, sizes in zip(sample[data_var].dims, chunks)}

    def __getitem__(self, key):
        """
        Access coordinates, data variables, or datasets in the batch.
        
        Parameters
        ----------
        key : str, list, or tuple
            If str: access coordinate or data variable across all datasets
            If list/tuple: select subset of datasets
            
        Returns
        -------
        Batch
            Batch of the requested coordinate/variable or selected datasets
        """
        # Handle list/tuple keys for dataset selection
        if isinstance(key, (list, tuple)):
            return type(self)({
                burst_id: ds[key]
                for burst_id, ds in self.items()
            })
            
        # Try to access as a dataset key first
        try:
            return super().__getitem__(key)
        except KeyError:
            # If not a dataset key, try to access as coordinate/variable
            return type(self)({
                k: ds[key] if not isinstance(ds, self.CoordCollection) else ds._ds.coords[key]
                for k, ds in self.items()
                if (isinstance(ds, self.CoordCollection) and key in ds._ds.coords) or 
                   (not isinstance(ds, self.CoordCollection) and (key in ds.coords or key in ds.data_vars))
            })

    def __add__(self, other: Batch):
        keys = self.keys()
        #& other.keys()
        return type(self)({k: (self[k] + other[k] if k in other else self[k]) for k in keys})

    def __sub__(self, other: Batch):
        keys = self.keys()
        result = {}
        for k in keys:
            if k not in other:
                result[k] = self[k]
            else:
                val = other[k]
                ds = self[k]
                # Handle per-pair coefficients from burst_polyfit
                if isinstance(val, (list, tuple)):
                    sample_var = list(ds.data_vars)[0]
                    sample_da = ds[sample_var]
                    has_pair_dim = 'pair' in sample_da.dims
                    n_pairs = sample_da.sizes.get('pair', 1)

                    if len(val) > 0 and isinstance(val[0], (list, tuple)):
                        # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                        # Use polyval for this case
                        result[k] = ds - self[[k]].polyval({k: val})[k]
                    elif has_pair_dim and len(val) == n_pairs:
                        # Multi-pair degree=0: [off0, off1, ...]
                        offsets = xr.DataArray(val, dims=['pair'],
                                               coords={'pair': sample_da.coords['pair']})
                        result[k] = ds - offsets
                    else:
                        # Single pair degree=1: [ramp, offset] or other
                        result[k] = ds - val
                else:
                    result[k] = ds - val
        return type(self)(result)

    def __mul__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: (self[k] * other[k] if k in other else self[k]) for k in keys})

    def __rmul__(self, other):
        # scalar * batch  → map scalar * each dataset
        return type(self)({k: other * v for k, v in self.items()})

    def __truediv__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: (self[k] / other[k] if k in other else self[k]) for k in keys})
    
    def _binop(self, other, op):
        """
        generic helper for any binary operator `op(ds, other)` or `op(ds, other_ds)`
        """
        if isinstance(other, (int, float)):
            return type(self)({k: op(ds, other) for k, ds in self.items()})
        elif isinstance(other, BatchCore):
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
                
                # Align mask to data coordinates (handles different x/y grids)
                # Get reference DataArray from ds for alignment
                if isinstance(ds, xr.Dataset):
                    ref_var = list(ds.data_vars)[0]
                    ref_da = ds[ref_var]
                else:
                    ref_da = ds
                mask_da = mask_da.reindex_like(ref_da, method='nearest')
                
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
            if isinstance(keys, list):
                return type(self)({k: self[k] for k in keys})
            
            # keys is dict-like (e.g., BatchWrap, BatchUnit)
            # Select matching burst IDs and align dimensions (like 'pair') per key
            result = {}
            for k in keys.keys():
                if k not in self:
                    continue
                ds = self[k]
                other_ds = keys[k]
                
                # Align 'pair' dimension if both have it (per burst key)
                if hasattr(other_ds, 'dims') and 'pair' in getattr(other_ds, 'dims', []):
                    if hasattr(ds, 'dims') and 'pair' in ds.dims:
                        # Get pairs from the other dataset for THIS key
                        other_pairs = set(other_ds.coords['pair'].values)
                        my_pairs = ds.coords['pair'].values
                        matching = [p for p in my_pairs if p in other_pairs]
                        if matching:
                            ds = ds.sel(pair=matching)
                        # If no matching pairs, keep ds as-is (will have 0 after filtering)
                
                result[k] = ds
            return type(self)(result)

        dss = {}
        # iterate all burst groups (fullBurstID is the first index level)
        for id in keys.index.get_level_values(0).unique():
            if id not in self:
                continue
            # select all records for the current burst group
            records = keys[keys.index.get_level_values(0)==id]
            ds = self[id]
            
            # Detect dimension type: date for Stack-like, pair for Batch-like
            if 'date' in ds.dims:
                # Stack-like: filter by dates
                dates = records.startTime.values.astype(str)
                ds = ds.sel(date=dates)
            # For pair-based data, we just select the burst if it exists
            # (pair filtering is handled elsewhere or not needed for simple selection)
            
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

        # dict as a keyword indexers
        if isinstance(indices, dict):
            indexers = indices
            indices = None

        # xarray‐style keyword isel (including dict-via-positional)
        if indexers:
            return type(self)({
                k: ds.isel(**indexers)
                for k, ds in self.items()
            })

        # fallback: positional isel over the batch keys (old behavior)
        keys = list(self.keys())
        if indices is None:
            # no selection, cast to dict to prevent special logic in the class constructor
            return type(self)(dict(self))
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

    @property
    def dims(self):
        return {k: self[k].dims for k in self.keys()}
    
    @property
    def coords(self):
        """Return a Batch of Coordinates for each dataset."""
        return type(self)({k: ds.coords.to_dataset() for k, ds in self.items()})

    def assign_coords(self, coords=None, **coords_kwargs):
        """
        Assign new coordinates to each dataset in the batch.
        Works like xarray.Dataset.assign_coords but handles batch operations.
        
        Parameters
        ----------
        coords : dict-like or Batch, optional
            Dictionary of coordinates to assign or Batch of coordinates
        **coords_kwargs : optional
            Coordinates to assign, specified as keyword arguments
        
        Returns
        -------
        Batch
            New batch with assigned coordinates
        """
        if coords is None:
            coords = {}
        coords = dict(coords, **coords_kwargs)
        
        def process_coord(coord):
            if not isinstance(coord, tuple) or len(coord) != 2:
                return coord
                
            dims, data = coord
            
            # Handle DataArray directly
            if isinstance(data, xr.DataArray):
                values = data.values
                return xr.DataArray(values if data.ndim > 0 else np.array([values]), dims=dims)
            
            # Handle BatchComplex
            if isinstance(data, type(self)):
                first_ds = next(iter(data.values()))
                if isinstance(first_ds, xr.DataArray):
                    values = first_ds.values
                    return xr.DataArray(values if first_ds.ndim > 0 else np.array([values]), dims=dims)
                elif isinstance(first_ds, xr.Dataset):
                    coord_name = first_ds.dims[0]
                    values = first_ds.coords[coord_name].values
                    return xr.DataArray(values if not np.isscalar(values) else np.array([values]), dims=dims)
            
            # Handle objects with values attribute
            if hasattr(data, 'values'):
                values = data.values() if callable(data.values) else data.values
                if hasattr(values, '__iter__'):
                    values = next(iter(values))
                    if isinstance(values, xr.DataArray):
                        values = values.values
                values = np.asarray(values)
                return xr.DataArray(values if values.ndim > 0 else np.array([values]), dims=dims)
            
            # Handle array-like inputs
            values = np.asarray(data)
            return xr.DataArray(values if values.ndim > 0 else np.array([values]), dims=dims)
        
        # Get target dimension size from first dataset
        first_ds = next(iter(self.values()))
        target_size = first_ds.dims[list(coords.values())[0][0]]
        
        # Process coordinates
        processed_coords = {name: process_coord(coord) for name, coord in coords.items()}
        
        # Ensure consistent dimension sizes
        for name, coord in processed_coords.items():
            if coord.size != target_size:
                if coord.size == 1 and target_size == 2:
                    processed_coords[name] = xr.DataArray([coord.values[0], coord.values[0]], dims=coord.dims)
                else:
                    raise ValueError(f"Coordinate {name} has size {coord.size} but expected size {target_size}")
        
        return type(self)({
            k: ds.assign_coords(processed_coords)
            for k, ds in self.items()
        })

    def set_index(self, indexes=None, **indexes_kwargs):
        """
        Set Dataset index(es) for each dataset in the batch.
        Works like xarray.Dataset.set_index but handles batch operations.
        
        Parameters
        ----------
        indexes : dict-like or Batch, optional
            Dictionary of indexes to set or Batch of indexes
        **indexes_kwargs : optional
            Indexes to set, specified as keyword arguments
        
        Returns
        -------
        Batch
            New batch with set indexes
        """
        if indexes is None:
            indexes = {}
        indexes = dict(indexes, **indexes_kwargs)
        
        # Handle both dict and Batch inputs
        if isinstance(indexes, type(self)):
            return type(self)({
                k: ds.set_index(indexes[k])
                for k, ds in self.items()
                if k in indexes
            })
        else:
            return type(self)({
                k: ds.set_index(indexes)
                for k, ds in self.items()
            })

    def expand_dims(self, *args, **kw):
        return type(self)({k: ds.expand_dims(*args, **kw) for k, ds in self.items()})

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
    
    def rename(self, **kw):
        return type(self)({k: ds.rename(**kw) for k, ds in self.items()})

    def reindex(self, **kw):
        return type(self)({k: ds.reindex(**kw) for k, ds in self.items()})

    def interp(self, **kw):
        return type(self)({k: ds.interp(**kw) for k, ds in self.items()})

    def interp_like(self, other: Batch, **interp_kwargs):
        """Regrid each Dataset onto the coords of the *corresponding* Dataset in `other`."""
        return type(self)({k: ds.interp_like(other[k], **interp_kwargs) for k, ds in self.items() if k in other})

    def reindex_like(self, other: Batch, **reindex_kwargs):
        return type(self)({k: ds.reindex_like(other[k], **reindex_kwargs) for k, ds in self.items() if k in other})

    def transpose(self, *dims, **kw):
        return type(self)({k: ds.transpose(*dims, **kw) for k, ds in self.items()})

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Internal helper for aggregation methods.
        If the target object's .<name>() accepts a `dim=` arg, we pass dim, otherwise we just call it without.
        """
        import inspect
        import pandas as pd
        out = {}
        for key, obj in self.items():
            fn = getattr(obj, name)
            sig = inspect.signature(fn)
            if "dim" in sig.parameters:
                out[key] = fn(dim=dim, **kwargs)
            else:
                out[key] = fn(**kwargs)
            
            # xarray coarsen + aggregate do not preserve multiindex pair
            if all(coord in out[key].coords for coord in ('pair', 'ref','rep')) \
                   and not isinstance(out[key].coords['pair'], pd.MultiIndex):
                out[key] = out[key].set_index(pair=['ref', 'rep'])

        #print ('_agg self.chunks', self.chunks)
        # filter out collapsed dimensions
        sample = next(iter(out.values()), None)
        dims = (sample.dims or []) if hasattr(sample, 'dims') else []
        chunks = {d: size for d, size in self.chunks.items() if d in dims}
        #print ('chunks', chunks)
        result = type(self)(out)
        if chunks:
            return result.chunk(chunks)
        return result

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

    def polyval(self, coeffs: dict[str, list | xr.DataArray], dim: str = 'x') -> BatchCore:
        """
        Evaluate polynomial coefficients for each burst.

        Applies xarray.polyval to evaluate polynomial corrections at each position
        along the specified dimension. Designed to work with polynomial coefficients
        returned by Stack.burst_polyfit.

        Parameters
        ----------
        coeffs : dict[str, list | xr.DataArray]
            Polynomial coefficients per burst. Can be either:
            - list[float]: [ramp, offset] for linear polynomial ramp*x + offset (single pair)
            - list[list[float]]: [[ramp0, offset0], [ramp1, offset1], ...] (multiple pairs)
            - list[float]: [offset0, offset1, ...] for degree=0 with multiple pairs
            - xr.DataArray: with 'degree' dimension [1, 0] (xarray.polyfit format)
            Following xarray.polyfit convention: highest degree first.
        dim : str, optional
            Coordinate dimension to evaluate polynomial on. Default is 'x' for range
            direction corrections.

        Returns
        -------
        BatchCore (or subclass)
            New batch with polynomial evaluated at each position. The result has
            the same structure as self, with polynomial values broadcast to match
            each dataset's shape.

        Examples
        --------
        >>> # Single pair: Estimate offsets and ramps
        >>> coeffs = Stack.burst_polyfit(intfs, degree=1)
        >>> corrections = intfs.polyval(coeffs, dim='x')
        >>> intfs_aligned = intfs - corrections

        >>> # Multiple pairs: coefficients are lists per pair
        >>> offsets = Stack.burst_polyfit(intfs_multi, degree=0)
        >>> # offsets = {'burst1': [off0, off1], 'burst2': [off0, off1]}
        >>> intfs_aligned = intfs_multi - offsets

        See Also
        --------
        xarray.polyval : Underlying polynomial evaluation function
        Stack.burst_polyfit : Function that produces compatible coefficients
        """
        result = {}
        for bid, ds in self.items():
            if bid not in coeffs:
                # No coefficients for this burst - zero correction
                sample_var = list(ds.data_vars)[0]
                result[bid] = xr.zeros_like(ds[sample_var]).to_dataset(name=sample_var)
                continue

            # Get coordinate for evaluation
            coord = ds.coords[dim]
            sample_var = list(ds.data_vars)[0]
            sample_da = ds[sample_var]

            coeff = coeffs[bid]

            # Check if we have per-pair coefficients (list of lists or list of scalars for multiple pairs)
            has_pair_dim = 'pair' in sample_da.dims
            n_pairs = sample_da.sizes.get('pair', 1)

            if isinstance(coeff, (list, tuple)) and len(coeff) > 0:
                first_elem = coeff[0]

                # Detect format:
                # - Single pair degree=1: [ramp, offset] where both are scalars
                # - Single pair degree=0: scalar (but wrapped in list by caller)
                # - Multi pair degree=0: [off0, off1, ...] list of scalars
                # - Multi pair degree=1: [[ramp0, off0], [ramp1, off1], ...] list of lists

                if isinstance(first_elem, (list, tuple)):
                    # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                    corrections = []
                    for pair_coeff in coeff:
                        corr = pair_coeff[0] * coord + pair_coeff[1]
                        corrections.append(corr)
                    # Stack along pair dimension
                    correction = xr.concat(corrections, dim='pair')
                    if has_pair_dim:
                        correction = correction.assign_coords(pair=sample_da.coords['pair'])

                elif has_pair_dim and len(coeff) == n_pairs and not isinstance(first_elem, (list, tuple)):
                    # Multi-pair degree=0: [off0, off1, ...] - all scalars matching pair count
                    # Check if it looks like [ramp, offset] for single pair (2 elements, no pair dim wouldn't reach here)
                    if len(coeff) == 2 and not has_pair_dim:
                        # Single pair degree=1: [ramp, offset]
                        correction = coeff[0] * coord + coeff[1]
                    else:
                        # Multi-pair degree=0
                        corrections = [xr.full_like(coord, off, dtype=float) for off in coeff]
                        correction = xr.concat(corrections, dim='pair')
                        correction = correction.assign_coords(pair=sample_da.coords['pair'])

                else:
                    # Single pair degree=1: [ramp, offset]
                    correction = coeff[0] * coord + coeff[1]

            elif isinstance(coeff, xr.DataArray):
                # General case using xr.polyval
                correction = xr.polyval(coord, coeff)
            else:
                # Single scalar (degree=0, single pair)
                correction = xr.full_like(coord, float(coeff), dtype=float)

            result[bid] = correction.to_dataset(name=sample_var)

        return type(self)(result)

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
        chunks = self.chunks
        out = {}
        # produce unified grid and chunks for all datasets in the batch
        for key, ds in self.items():
            # align each dimension
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds, dim, factor)
                #print ('start', start)
                if start is not None:
                    # rechunk to the original chunk sizes
                    ds = ds.isel({dim: slice(start, None)}).chunk(chunks)
                    # or allow a bit different chunks for coarsening
                    #ds = ds.isel({dim: slice(start, None)})
            # coarsen and revert original chunks
            out[key] = ds.coarsen(window, **kwargs)

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

    def to_dataframe(self,
                     crs: str | int | None = 'auto',
                     debug: bool = False) -> pd.DataFrame:
        """
        Return a Pandas/GeoPandas DataFrame for all Batch scenes.
        
        Extracts attributes from each Dataset in the Batch (from .attrs or dim-indexed data vars)
        and combines them into a single DataFrame, matching the Stack.to_dataframe format
        with additional ref/rep columns for pair information.

        Parameters
        ----------
        crs : str | int | None, optional
            Coordinate reference system for the output GeoDataFrame.
            If 'auto', uses CRS from the data. If None, returns without CRS conversion.
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            The DataFrame containing Batch scenes with their attributes.
            Index is (fullBurstID, burst) matching Stack.to_dataframe.
            For pair-based data, ref and rep columns are added after the index.

        Examples
        --------
        >>> df = batch.to_dataframe()
        >>> df = batch.to_dataframe(crs=4326)
        """
        import geopandas as gpd
        from shapely import wkt
        import pandas as pd

        if not self:
            return pd.DataFrame()

        # Detect CRS from data if auto
        if crs is not None and isinstance(crs, str) and crs == 'auto':
            sample = next(iter(self.values()))
            crs = sample.attrs.get('crs', 4326)

        # Detect polarizations
        sample = next(iter(self.values()))
        polarizations = [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in sample.data_vars]

        # Detect dimension: 'date' for BatchComplex, 'pair' for others
        dim = 'date' if 'date' in sample.dims else 'pair'

        # Define the attribute order matching Stack.to_dataframe
        # Order: fullBurstID, burst, startTime, polarization, flightDirection, pathNumber, subswath, mission, beamModeType, BPR, geometry
        attr_order = ['fullBurstID', 'burst', 'startTime', 'polarization', 'flightDirection', 
                      'pathNumber', 'subswath', 'mission', 'beamModeType', 'BPR', 'geometry']

        # Make attributes dataframe from data
        processed_attrs = []
        for key, ds in self.items():
            for idx in range(ds.dims[dim]):
                processed_attr = {}
                
                # Get ref/rep for pair dimension
                if dim == 'pair':
                    if 'ref' in ds.coords:
                        processed_attr['ref'] = pd.Timestamp(ds['ref'].values[idx])
                    if 'rep' in ds.coords:
                        processed_attr['rep'] = pd.Timestamp(ds['rep'].values[idx])
                else:
                    processed_attr['date'] = pd.Timestamp(ds[dim].values[idx])
                
                # Extract attributes from ds.attrs
                for attr_name in attr_order:
                    if attr_name in ds.attrs and attr_name not in processed_attr:
                        value = ds.attrs[attr_name]
                        if attr_name == 'geometry' and isinstance(value, str):
                            processed_attr[attr_name] = wkt.loads(value)
                        elif attr_name == 'startTime':
                            processed_attr[attr_name] = pd.Timestamp(value)
                        else:
                            processed_attr[attr_name] = value
                
                processed_attrs.append(processed_attr)

        if not processed_attrs:
            return pd.DataFrame()

        # Check if we have geometry column for GeoDataFrame
        has_geometry = 'geometry' in processed_attrs[0]
        
        if has_geometry:
            df = gpd.GeoDataFrame(processed_attrs, crs=4326)
        else:
            df = pd.DataFrame(processed_attrs)

        # Add polarization info if not already present
        if 'polarization' not in df.columns and polarizations:
            df['polarization'] = ','.join(map(str, polarizations))

        # Round BPR for readability
        if 'BPR' in df.columns:
            df['BPR'] = df['BPR'].round(1)

        # Reorder columns to match Stack.to_dataframe format
        # For pair data: fullBurstID, burst (index), then ref, rep, then rest
        if 'fullBurstID' in df.columns and 'burst' in df.columns:
            # Build column order
            if dim == 'pair':
                # ref, rep first after index, then startTime, polarization, etc.
                first_cols = ['fullBurstID', 'burst', 'ref', 'rep']
            else:
                first_cols = ['fullBurstID', 'burst', 'date']
            
            # Rest of columns in attr_order, excluding index columns and ref/rep/date
            other_cols = [c for c in attr_order if c not in first_cols and c in df.columns]
            
            # Reorder
            ordered_cols = [c for c in first_cols if c in df.columns] + other_cols
            df = df[ordered_cols]
            
            # Sort and set index
            df = df.sort_values(by=['fullBurstID', 'burst']).set_index(['fullBurstID', 'burst'])
            
            # Move geometry to end if present
            if has_geometry and 'geometry' in df.columns:
                df = df.loc[:, df.columns.drop("geometry").tolist() + ["geometry"]]

        # Convert CRS if requested and we have a GeoDataFrame
        if has_geometry and crs is not None:
            return df.to_crs(crs)
        return df
    
    def persist(self):
        return type(self)({
            k: ds.chunk(ds.chunks).persist()
            for k, ds in self.items()
        })

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

    def save(self, store: str, storage_options: dict[str, str] | None = None,
                caption: str | None = 'Saving...', n_jobs: int = -1, debug=False):
        return utils_io.save(self, store=store, storage_options=storage_options, compat=False, caption=caption, n_jobs=n_jobs, debug=debug)

    def open(self, store: str, storage_options: dict[str, str] | None = None, n_jobs: int = -1, debug=False):
        data = utils_io.open(store=store, storage_options=storage_options, compat=False, n_jobs=n_jobs, debug=debug)
        if not isinstance(data, dict):
            raise ValueError(f'ERROR: open() returns multiple datasets, you need to use Stack class to open them.')
        return data
    
    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str | None = 'Snapshotting...', n_jobs: int = -1, debug=False):
        # Only save if this batch has data; otherwise just open existing store
        if len(self) > 0:
            self.save(store=store, storage_options=storage_options, caption=caption, n_jobs=n_jobs, debug=debug)
        return utils_io.open(store=store, storage_options=storage_options, compat=False, n_jobs=n_jobs, debug=debug)

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
        #print ('dtype', type(self), 'wrap', wrap, 'dissolve', dissolve, 'polarization', polarization)

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
            #polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in sample.data_vars]
            # exclude spatial_ref as it's a metadata variable, not a data variable
            polarizations = [v for v in sample.data_vars if v != 'spatial_ref']
            #print ('polarizations', polarizations)

            # process list of datasets with one or multiple polarizations
            das = xr.merge([self.to_dataset(polarization=pol, dissolve=dissolve) for pol in polarizations])
            if compute:
                progressbar(das := das.persist(), desc=f'Computing Dataset...'.ljust(25))
                return das
            return das
        
        # process single polarization dataarrays
        datas = [self[k][polarization] for k in self.keys()]

        # process list of dataarrays with single polarization
        # define unified grid
        y_min = min(ds.y.min().item() for ds in datas)
        y_max = max(ds.y.max().item() for ds in datas)
        x_min = min(ds.x.min().item() for ds in datas)
        x_max = max(ds.x.max().item() for ds in datas)
        #print (y_min, y_max, x_min, x_max, y_max-y_min, x_max-x_min)
        dims = datas[0].dims
        #print ('dims', dims, len(dims))
        stackvar = list(dims)[0] if len(dims) > 2 else None
        #print ('stackvar', stackvar)
        # workaround for dask.array.blockwise
        if stackvar == 'pair':
            # multiindex pair
            stackval = [(str(ref)[:10] +' '+ str(rep)[:10]) for ref, rep in datas[0][stackvar].values]
        elif stackvar is not None:
            stackval = datas[0][stackvar].astype(str)
        else:
            stackvar = 'fake'
            stackval = [0]
            datas = [da.expand_dims({stackvar: [0]}) for da in datas]
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
        def block_dask(stack, y_chunk, x_chunk, wrap, fill_dtype):
            #fill_dtype = datas[0].dtype
            fill_nan = np.nan * np.ones((), dtype=fill_dtype)

            # TEST: return empty block
            #fill_nan = np.nan * np.ones((), dtype=fill_dtype)
            #return np.full((stack.size, y_chunk.size, x_chunk.size), fill_nan, dtype=fill_dtype)

            #print ('pair', pair)
            #print ('concat: block_dask', stackvar, stack)
            # extract extent of the current chunk once
            ymin0, ymax0 = float(y_chunk.min()), float(y_chunk.max())
            xmin0, xmax0 = float(x_chunk.min()), float(x_chunk.max())
            # select all datasets overlapping with the current chunk
            # das_slice = [da.isel({stackvar: stackidx}).sel({'y': slice(ymin0, ymax0), 'x': slice(xmin0, xmax0)}).compute()
            #              for da, (ymin, ymax, xmin, xmax) in zip(datas, extents)
            #              if ymin0 < ymax and ymax0 > ymin and xmin0 < xmax and xmax0 > xmin]
            # print ('concat: das_slice', len(das_slice), [da.shape for da in das_slice])
            das_slice = [(idx, {'y': slice(ymin0, ymax0), 'x': slice(xmin0, xmax0)})
                           for idx, (ymin, ymax, xmin, xmax) in enumerate(extents)
                           if ymin0 < ymax and ymax0 > ymin and xmin0 < xmax and xmax0 > xmin]
            if len(das_slice) == 0:
                # return empty block
                return np.full((stack.size, y_chunk.size, x_chunk.size), fill_nan, dtype=fill_dtype)
            das_block = [datas[idx].isel({stackvar: stackidx}).sel(slice) for idx, slice in das_slice]
            del das_slice
            das_block = dask.compute(*das_block)
            #print ('concat: das_block', len(das_block))

            # TEST: return empty block
            #return np.full((stack.size, y_chunk.size, x_chunk.size), fill_nan, dtype=fill_dtype)
            
            #das_block = [da.reindex({'y': y_chunk, 'x': x_chunk}, fill_value=fill_nan, copy=False) for da in das_slice if da.size > 0]
            das_block = [da.reindex({'y': y_chunk, 'x': x_chunk}, fill_value=fill_nan, copy=False) for da in das_block]
            
            if len(das_block) == 1:
               # return single block as is
               return das_block[0].values

            if not dissolve:
                #print ('wrap None')
                # ffill does not work correct on complex data and per-component ffill is faster
                # the magic trick is to use sorting to ensure burst overpapping order
                # bursts ends should be overlapped by bursts starts
                das_block_concat = xr.concat(das_block, dim='stack_dim', join='inner')
                if np.issubdtype(das_block_concat.dtype, np.complexfloating):
                    return (das_block_concat.real.ffill('stack_dim').isel(stack_dim=-1)
                            + 1j*das_block_concat.imag.ffill('stack_dim').isel(stack_dim=-1)).values
                else:
                    return das_block_concat.ffill('stack_dim').isel(stack_dim=-1).values
            elif wrap == True:
                #print ('wrap True')
                # calculate circular mean for interferogram data
                das_block_concat = xr.concat([np.exp(1j * da) for da in das_block], dim='stack_dim')
                block_complex = das_block_concat.mean('stack_dim', skipna=True).values
                return np.arctan2(block_complex.imag, block_complex.real)
            elif wrap == False:
                #print ('wrap False')
                das_block_concat = xr.concat(das_block, dim='stack_dim', join='outer')
                # calculate arithmetic mean for phase and correlation data
                return das_block_concat.mean('stack_dim', skipna=True).reindex({'y': y_chunk, 'x': x_chunk}, fill_value=fill_nan, copy=False).values
            else:
                raise ValueError(f'ERROR: wrap is not a boolean or None: {wrap}')

        # prevent warnings 'PerformanceWarning: Increasing number of chunks by factor of ...'
        import warnings
        from dask.array.core import PerformanceWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=PerformanceWarning)
            # rechunk data for expected usage using Dask machinery
            #print('Default target chunk size:', dask.config.get('array.chunk-size'))
            # control chunk size using Dask config:
            # with dask.config.set({'array.chunk-size': '256MiB'}):
            # ....to_dataset()
            # TODO: fix the hard coded value 16
            chunks = dask.array.core.normalize_chunks('auto', (ys.size, xs.size, 16), dtype=datas[0].dtype)
            #print ('chunks', chunks)
            data = dask.array.blockwise(
                block_dask,
                'zyx',
                stackidx.chunk(1), 'z',
                ys.chunk({'y': chunks[0]}), 'y',
                xs.chunk({'x': chunks[1]}), 'x',
                meta = np.empty((0, 0, 0), dtype=datas[0].dtype),
                wrap=wrap,
                fill_dtype=datas[0].dtype
            )
        da = xr.DataArray(data, coords={stackvar: stackval, 'y': ys, 'x': xs})\
            .rename(datas[0].name)\
            .assign_attrs(datas[0].attrs)
        da = datagrid.spatial_ref(da, datas)
        return da if stackvar != 'fake' else da.isel({stackvar: 0})

    def plot(self,
            dissolve: bool = False,
            cmap: matplotlib.colors.Colormap | str | None = 'viridis',
            alpha: float = 0.7,
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
        import matplotlib.ticker as mticker
        import matplotlib.pyplot as plt
        from .Batch import BatchWrap
        from insardev_toolkit import progressbar

        # no data means no plot and no error
        if not len(self):
            return
        
        wrap = True if type(self) == BatchWrap else False
        #print ('dtype', type(self), 'wrap', wrap, 'dissolve', dissolve)

        # screen size in pixels (width, height) to estimate reasonable number pixels per plot
        # this is quite large to prevent aliasing on 600dpi plots without additional processing
        if _size is None:
            _size = (8000,4000)

        # use outer variables
        def plot_polarization(polarization):
            stackvar = list(sample[polarization].dims)[0] if len(sample[polarization].dims) > 2 else None
            #print ('stackvar', stackvar)
            if stackvar is None:
                stackvar = 'fake'
                da = self[[polarization]].to_dataset(dissolve=dissolve)[polarization].expand_dims({stackvar: [0]})
            else:
                #da = self[[polarization]]
                #da = self[[polarization]].isel({stackvar: slice(0, rows)})
                da = self[[polarization]].isel({stackvar: slice(0, rows)}).to_dataset(dissolve=dissolve)[polarization]
            #print ('da', da)
            if 'stack' in da.dims and isinstance(da.coords['stack'].to_index(), pd.MultiIndex):
                da = da.unstack('stack')
            
            # there is no reason to plot huge arrays much larger than screen size for small plots
            #print ('screen_size', screen_size)
            size_y, size_x = da.shape[-2:]
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
                vmin=_vmin, vmax=_vmax,
                cmap=cmap, alpha=alpha,
            )
            fg.set_axis_labels('easting [m]', 'northing [m]')
            fg.set_ticks(max_xticks=nbins, max_yticks=nbins)
            fg.fig.suptitle(f'{polarization} {caption}', y=y)

            # fg is the FacetGrid returned by xarray.plot.imshow
            for ax in fg.axes.flatten():
                # disable the offset text (like "1e6")
                # force plain formatting (no scientific notation) on the y‐axis
                ax.ticklabel_format(style='plain', axis='y', useOffset=False)
                if stackvar == 'fake':
                    # remove 'fake = 0' title
                    ax.set_title('')
                
            return fg

        if quantile is not None:
            assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

        sample = next(iter(self.values()))
        # find all variables in the first dataset related to polarizations
        # TODO
        #polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in sample.data_vars]
        polarizations = list(sample.data_vars)
        #print ('polarizations', polarizations)

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

    def residuals(self, debug: bool = False) -> float | list[float]:
        """
        Measure phase offset discrepancy across all burst overlaps.

        Computes the weighted mean of absolute median phase differences
        across all overlapping regions. After offset correction with fit(),
        these median differences should be close to zero.

        Parameters
        ----------
        debug : bool, optional
            Print debug information for each overlap. Default is False.

        Returns
        -------
        float or list[float]
            Single pair: Weighted mean absolute median phase discrepancy in radians.
            Multiple pairs: List of discrepancies, one per pair.
            0.0 = perfect alignment, π = maximum discrepancy (for wrapped phase).

            Practical interpretation:

            - < 0.1 rad: Excellent alignment
            - 0.1 - 0.5 rad: Good alignment
            - 0.5 - 1.0 rad: Moderate misalignment
            - > 1.0 rad: Poor alignment

        Examples
        --------
        >>> # Compare before and after alignment
        >>> before = intfs.residuals()
        >>> aligned = intfs.align()
        >>> after = aligned.residuals()
        >>> print(f'Discrepancy reduced from {before} to {after}')
        """
        from .Batch import Batch, BatchWrap
        import dask

        # Determine if we need circular statistics based on class type
        if isinstance(self, BatchWrap):
            use_circular = True
        elif isinstance(self, Batch):
            use_circular = False
        else:
            raise TypeError(f"residuals() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")

        def maybe_wrap(x):
            """Wrap to [-π, π) for circular stats, identity otherwise."""
            if use_circular:
                return (x + np.pi) % (2*np.pi) - np.pi
            return x

        # Collect burst extents and detect pair dimension
        ids = sorted(self.keys())

        sample_ds = self[ids[0]]
        pol = list(sample_ds.data_vars)[0] if hasattr(sample_ds, 'data_vars') else None
        sample_da = sample_ds[pol] if pol else sample_ds
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        # Extract pathNumber and subswath attributes for each burst (required for stats)
        burst_subswath = {}
        burst_track = {}  # pathNumber + subswath for detailed debug output
        for bid in ids:
            ds = self[bid]
            if 'subswath' not in ds.attrs:
                raise ValueError(f"Burst '{bid}' missing required 'subswath' attribute")
            if 'pathNumber' not in ds.attrs:
                raise ValueError(f"Burst '{bid}' missing required 'pathNumber' attribute")
            burst_subswath[bid] = ds.attrs['subswath']
            burst_track[bid] = f"{ds.attrs['pathNumber']}{ds.attrs['subswath']}"

        extents = {}
        for bid in ids:
            ds = self[bid]
            pol = list(ds.data_vars)[0] if hasattr(ds, 'data_vars') else None
            da = ds[pol] if pol else ds
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            y_coords = da.coords['y'].values
            x_coords = da.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())

        def extents_overlap(e1, e2):
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap

        # Find all overlapping burst pairs
        overlap_pairs = []
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                e2 = extents[id2]
                if extents_overlap(e1, e2):
                    overlap_pairs.append((id1, id2))

        if not overlap_pairs:
            return [0.0] * n_pairs if has_pair_dim else 0.0

        if debug:
            print(f'residuals: found {len(overlap_pairs)} overlap pairs, {n_pairs} pair(s)', flush=True)

        # Build all lazy phase differences (dask graphs)
        jobs = []
        lazy_diffs = []
        for id1, id2 in overlap_pairs:
            i1 = self[id1]
            i2 = self[id2]

            pol = list(i1.data_vars)[0] if hasattr(i1, 'data_vars') else None
            if pol:
                i1 = i1[pol]
                i2 = i2[pol]

            for pair_idx in range(n_pairs):
                i1_p = i1.isel(pair=pair_idx) if 'pair' in i1.dims else i1
                i2_p = i2.isel(pair=pair_idx) if 'pair' in i2.dims else i2
                phase_diff = i2_p - i1_p
                jobs.append((id1, id2, pair_idx))
                lazy_diffs.append(phase_diff)

        # Compute all phase differences at once - dask schedules efficiently
        if debug:
            print(f'Computing {len(lazy_diffs)} phase differences...', flush=True)
        computed_diffs = dask.compute(*lazy_diffs)

        # Process computed results
        results = []
        for (id1, id2, pair_idx), phase_diff in zip(jobs, computed_diffs):
            valid = phase_diff.values.ravel()
            valid = valid[np.isfinite(valid)]

            if len(valid) == 0:
                continue

            valid = maybe_wrap(valid)
            median_diff = np.median(valid)
            abs_discrepancy = np.abs(maybe_wrap(median_diff))
            weight = len(valid)

            results.append((pair_idx, abs_discrepancy, weight, id1, id2, median_diff))

        # Aggregate results per pair and per subswath
        total_weights = [0.0] * n_pairs
        weighted_sums = [0.0] * n_pairs

        # Per-subswath tracking for debug
        subswath_stats = {}  # {(subswath, pair_idx): {'sum': float, 'weight': float, 'count': int, 'values': []}}
        per_overlap_discrepancies = {p: [] for p in range(n_pairs)}  # For computing std

        for result in results:
            if result is None:
                continue
            pair_idx, abs_discrepancy, weight, id1, id2, median_diff = result
            weighted_sums[pair_idx] += abs_discrepancy * weight
            total_weights[pair_idx] += weight
            per_overlap_discrepancies[pair_idx].append(abs_discrepancy)

            # Extract track info for debug stats
            if debug:
                track1 = burst_track[id1]
                track2 = burst_track[id2]

                # Categorize: same track or cross-track
                if track1 == track2:
                    track_key = track1
                else:
                    track_key = f'{track1}-{track2}'

                key = (track_key, pair_idx)
                if key not in subswath_stats:
                    subswath_stats[key] = {'sum': 0.0, 'weight': 0.0, 'count': 0, 'values': []}
                subswath_stats[key]['sum'] += abs_discrepancy * weight
                subswath_stats[key]['weight'] += weight
                subswath_stats[key]['count'] += 1
                subswath_stats[key]['values'].append(abs_discrepancy)

        discrepancies = []
        for p in range(n_pairs):
            if total_weights[p] == 0:
                discrepancies.append(0.0)
            else:
                discrepancies.append(round(weighted_sums[p] / total_weights[p], 3))

        if debug:
            # Compute std for overall discrepancy
            for p in range(n_pairs):
                vals = per_overlap_discrepancies[p]
                if len(vals) > 1:
                    std = np.std(vals)
                    print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} ± {std:.3f} rad ({len(vals)} overlaps)', flush=True)
                else:
                    print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} rad ({len(vals)} overlaps)', flush=True)

            # Print per-track stats (only for pair_idx=0 to avoid clutter)
            print('Per-track discrepancy (pair 0):', flush=True)
            for (track, pair_idx), stats in sorted(subswath_stats.items()):
                if pair_idx == 0 and stats['weight'] > 0:
                    track_disc = stats['sum'] / stats['weight']
                    vals = stats['values']
                    if len(vals) > 1:
                        track_std = np.std(vals)
                        print(f'  {track}: {track_disc:.3f} ± {track_std:.3f} rad ({stats["count"]} overlaps)', flush=True)
                    else:
                        print(f'  {track}: {track_disc:.3f} rad ({stats["count"]} overlaps)', flush=True)

        # Return single value for single pair, list for multiple
        if n_pairs == 1 and not has_pair_dim:
            return discrepancies[0]
        return discrepancies

    def fit(self,
            degree: int = 0,
            method: str = 'median',
            debug: bool = False,
            return_residuals: bool = False):
        """
        Estimate per-burst polynomial coefficients using overlap-based least-squares.

        Fits polynomial corrections (offset or offset+ramp) to each burst by analyzing
        phase differences in overlapping regions. Uses global least-squares optimization
        to find consistent coefficients across all bursts.

        Parameters
        ----------
        degree : int, optional
            Polynomial degree:
            - 0 (default): Estimate offsets only.
            - 1: Estimate linear ramp (in x/range direction).
        method : str, optional
            Estimation method: 'median' (robust) or 'mean' (faster).
        debug : bool, optional
            Print debug information. Default is False.
        return_residuals : bool, optional
            If True, also return input residuals (before correction). Default is False.

        Returns
        -------
        dict or tuple
            If return_residuals is False:
                For single pair (no pair dimension):
                    degree=0: {burst_id: offset}
                    degree=1: {burst_id: [ramp, intercept]}
                For multiple pairs:
                    degree=0: {burst_id: [offset_pair0, offset_pair1, ...]}
                    degree=1: {burst_id: [[ramp0, intercept0], [ramp1, intercept1], ...]}
            If return_residuals is True:
                (coefficients_dict, residuals) where residuals is float or list[float]

        Examples
        --------
        >>> # 3-step alignment for best results (0.028 rad discrepancy):
        >>> # Step 1: Estimate offsets
        >>> offsets1 = intfs.fit(degree=0)
        >>> intfs1 = intfs - offsets1
        >>> # Step 2: Estimate ramps
        >>> ramps = intfs1.fit(degree=1)
        >>> intfs2 = intfs1 - intfs1.polyval(ramps)
        >>> # Step 3: Re-estimate offsets
        >>> offsets2 = intfs2.fit(degree=0)
        >>> # Combine coefficients (for single pair)
        >>> coeffs = {b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]] for b in offsets1}
        >>> aligned = intfs - intfs.polyval(coeffs)
        """
        from .Batch import Batch, BatchWrap
        import dask
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        from scipy.sparse.csgraph import connected_components

        # Determine if we need circular statistics based on class type
        if isinstance(self, BatchWrap):
            use_circular = True
        elif isinstance(self, Batch):
            use_circular = False
        else:
            raise TypeError(f"fit() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")

        # Constants
        MIN_OVERLAP_PIXELS = 50
        MIN_ROW_PIXELS = 10
        MIN_VALID_ROWS = 5
        MIN_INLIER_SAMPLES = 10
        MAD_OUTLIER_THRESHOLD = 2.5
        OUTPUT_PRECISION = 3
        RAMP_PRECISION = 9

        def maybe_wrap(x):
            """Wrap to [-π, π) for circular stats, identity otherwise."""
            if use_circular:
                return (x + np.pi) % (2*np.pi) - np.pi
            return x

        def phase_diff(a, center):
            """Circular or linear difference from center."""
            if use_circular:
                return maybe_wrap(a - center)
            return a - center

        def phase_mean(a):
            """Circular or linear mean."""
            if use_circular:
                return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
            return np.mean(a)

        def phase_mad(a, center):
            """Circular or linear MAD."""
            return np.median(np.abs(phase_diff(a, center)))

        # Collect burst extents and x-centers
        ids = sorted(self.keys())
        n_bursts = len(ids)
        id_to_idx = {bid: i for i, bid in enumerate(ids)}

        # Detect number of pairs
        sample_ds = self[ids[0]]
        pol = list(sample_ds.data_vars)[0] if hasattr(sample_ds, 'data_vars') else None
        sample_da = sample_ds[pol] if pol else sample_ds
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        if debug:
            print(f'fit(degree={degree}): {n_bursts} bursts, {n_pairs} pair(s)', flush=True)

        # Extract pathNumber and subswath attributes for each burst (required for degree=1)
        # Use pathNumber + subswath as track key to prevent inter-path same-subswath overlaps
        # in near-polar regions from being used for ramp estimation
        burst_track = {}  # pathNumber + subswath (e.g., '33IW3')
        burst_subswath = {}  # just subswath for debug output
        for bid in ids:
            ds = self[bid]
            if degree == 1:
                if 'subswath' not in ds.attrs:
                    raise ValueError(f"Burst '{bid}' missing required 'subswath' attribute for ramp estimation")
                if 'pathNumber' not in ds.attrs:
                    raise ValueError(f"Burst '{bid}' missing required 'pathNumber' attribute for ramp estimation")
                burst_track[bid] = f"{ds.attrs['pathNumber']}{ds.attrs['subswath']}"
                burst_subswath[bid] = ds.attrs['subswath']

        extents = {}
        x_centers = {}

        for bid in ids:
            ds = self[bid]
            pol = list(ds.data_vars)[0] if hasattr(ds, 'data_vars') else None
            da = ds[pol] if pol else ds
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            y_coords = da.coords['y'].values
            x_coords = da.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())
            x_centers[bid] = float(np.mean(x_coords))

        def extents_overlap(e1, e2):
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap

        def process_phase_diff(phase, id1, id2, pair_idx):
            """Process a computed phase difference to extract offset and optionally ramp."""
            all_valid = phase.values.ravel()
            all_valid = all_valid[np.isfinite(all_valid)]

            if len(all_valid) < MIN_OVERLAP_PIXELS:
                return None

            if 'y' not in phase.dims or 'x' not in phase.dims:
                return None

            x_coords = phase.coords['x'].values

            # Row-wise processing
            row_phases = []
            row_x_centroids = []
            row_weights = []

            for y_idx in range(phase.shape[0]):
                row = phase.values[y_idx, :]
                valid_mask = np.isfinite(row)
                n_valid = np.sum(valid_mask)
                if n_valid >= MIN_ROW_PIXELS:
                    x_valid = x_coords[valid_mask]
                    phase_valid = row[valid_mask]
                    # Unwrap for row mean computation (needed for both circular and linear)
                    if use_circular:
                        phase_unwrapped = np.unwrap(phase_valid)
                        row_mean = maybe_wrap(np.mean(phase_unwrapped))
                    else:
                        row_mean = np.mean(phase_valid)
                    row_phases.append(row_mean)
                    row_x_centroids.append(np.mean(x_valid))
                    row_weights.append(n_valid)

            if len(row_phases) < MIN_VALID_ROWS:
                return None

            a = np.array(row_phases)
            x_row = np.array(row_x_centroids)
            weights = np.array(row_weights)
            a = maybe_wrap(a)

            # Outlier rejection
            if method == 'median':
                offset_initial = np.median(a)
                mad = phase_mad(a, offset_initial)
                if mad > 0:
                    inliers = np.abs(phase_diff(a, offset_initial)) <= MAD_OUTLIER_THRESHOLD * mad
                    if np.sum(inliers) >= MIN_INLIER_SAMPLES:
                        a = a[inliers]
                        x_row = x_row[inliers]
                        weights = weights[inliers]

            n_valid = int(np.sum(weights))
            x_centroid = float(np.average(x_row, weights=weights))

            # Compute offset
            if method == 'median':
                sorted_idx = np.argsort(a)
                cumsum = np.cumsum(weights[sorted_idx])
                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                offset = a[sorted_idx[median_idx]]
            else:
                offset = phase_mean(a)

            # Compute ramp if degree=1
            ramp_val = None
            if degree == 1 and len(a) >= MIN_VALID_ROWS:
                x_centered = x_row - x_centroid
                x_range = np.max(x_row) - np.min(x_row)
                if x_range > 100:
                    residuals = a - offset
                    Swxx = np.sum(weights * x_centered**2)
                    Swxr = np.sum(weights * x_centered * residuals)
                    if Swxx > 1e-10:
                        ramp_val = Swxr / Swxx

            return (id1, id2, pair_idx, maybe_wrap(offset), ramp_val, x_centroid, n_valid)

        # Find overlapping burst pairs
        # For degree=1 (ramp), only use same-subswath overlaps (along-track)
        # For degree=0 (offset), use all overlaps including cross-subswath
        all_overlap_pairs = []
        cross_subswath_skipped = 0
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                if extents_overlap(e1, extents[id2]):
                    if degree == 1:
                        # For ramp estimation, only use same-track overlaps (same path + subswath)
                        # This prevents inter-path same-subswath overlaps in near-polar regions
                        track1, track2 = burst_track[id1], burst_track[id2]
                        if track1 != track2:
                            cross_subswath_skipped += 1
                            continue
                    all_overlap_pairs.append((id1, id2))

        if debug:
            print(f'Found {len(all_overlap_pairs)} overlapping burst pairs', flush=True)
            if degree == 1 and cross_subswath_skipped > 0:
                print(f'  (skipped {cross_subswath_skipped} cross-track pairs for ramp estimation)', flush=True)

        # Build all lazy phase differences (dask graphs)
        jobs = []
        lazy_diffs = []
        for id1, id2 in all_overlap_pairs:
            i1 = self[id1]
            i2 = self[id2]

            pol = list(i1.data_vars)[0] if hasattr(i1, 'data_vars') else None
            if pol:
                i1 = i1[pol]
                i2 = i2[pol]

            for pair_idx in range(n_pairs):
                i1_p = i1.isel(pair=pair_idx) if 'pair' in i1.dims else i1
                i2_p = i2.isel(pair=pair_idx) if 'pair' in i2.dims else i2
                phase_diff_val = i2_p - i1_p
                jobs.append((id1, id2, pair_idx))
                lazy_diffs.append(phase_diff_val)

        if debug:
            print(f'Computing {len(lazy_diffs)} phase differences...', flush=True)

        # Compute all phase differences at once - dask schedules efficiently
        computed_diffs = dask.compute(*lazy_diffs)

        # Process computed results
        results = []
        rejection_counts = {'too_few_pixels': 0, 'missing_dims': 0, 'too_few_rows': 0, 'no_ramp': 0}
        for (id1, id2, pair_idx), phase in zip(jobs, computed_diffs):
            result = process_phase_diff(phase, id1, id2, pair_idx)
            if result is not None:
                results.append(result)
            else:
                # Count rejections (only for pair_idx==0 to avoid double counting)
                if pair_idx == 0:
                    all_valid = phase.values.ravel()
                    all_valid = all_valid[np.isfinite(all_valid)]
                    if len(all_valid) < MIN_OVERLAP_PIXELS:
                        rejection_counts['too_few_pixels'] += 1
                    elif 'y' not in phase.dims or 'x' not in phase.dims:
                        rejection_counts['missing_dims'] += 1
                    else:
                        rejection_counts['too_few_rows'] += 1

        if debug:
            n_valid = len([r for r in results if r[2] == 0])  # count for pair_idx=0
            n_rejected = len(all_overlap_pairs) - n_valid
            print(f'  Rejections: {rejection_counts}', flush=True)
            if degree == 1:
                # Count how many had no ramp computed
                no_ramp = sum(1 for r in results if r[2] == 0 and r[4] is None)
                print(f'  Valid pairs with no ramp (x_range too small): {no_ramp}', flush=True)

        # Organize results by pair_idx
        pairs_by_pair_idx = {p: [] for p in range(n_pairs)}
        for result in results:
            if result is None:
                continue
            id1, id2, pair_idx, offset, ramp_val, x_centroid, n_used = result
            weight = np.sqrt(n_used)
            if degree == 0:
                pairs_by_pair_idx[pair_idx].append((id1, id2, offset, weight))
            else:
                if ramp_val is not None:
                    pairs_by_pair_idx[pair_idx].append((id1, id2, offset, ramp_val, x_centroid, weight))

        def solve_for_pair(pair_idx):
            """Solve least-squares for a single pair index."""
            pairs = pairs_by_pair_idx[pair_idx]

            if len(pairs) == 0:
                if degree == 0:
                    return {bid: 0.0 for bid in ids}
                else:
                    return {bid: [0.0, 0.0] for bid in ids}

            # Build connectivity graph
            adjacency = sparse.lil_matrix((n_bursts, n_bursts))
            for p in pairs:
                id1, id2 = p[0], p[1]
                i, j = id_to_idx[id1], id_to_idx[id2]
                adjacency[i, j] = 1
                adjacency[j, i] = 1

            n_components, labels = connected_components(adjacency.tocsr(), directed=False)

            if debug and pair_idx == 0:  # Only print for first pair to avoid spam
                print(f'  Found {n_components} connected component(s) for {len(pairs)} valid pairs', flush=True)
                for comp in range(n_components):
                    comp_indices = np.where(labels == comp)[0]
                    comp_ids = [ids[i] for i in comp_indices]
                    # Try to extract subswath info from burst IDs
                    subswaths = set()
                    paths = set()
                    for bid in comp_ids:
                        if '_IW' in bid:
                            sw = bid.split('_IW')[1][0]
                            subswaths.add(f'IW{sw}')
                        parts = bid.split('_')
                        if len(parts) >= 1 and parts[0].isdigit():
                            paths.add(parts[0])
                    print(f'    Component {comp}: {len(comp_ids)} bursts, paths={paths}, subswaths={subswaths}', flush=True)

            if degree == 0:
                offsets_out = {}

                for comp in range(n_components):
                    comp_indices = np.where(labels == comp)[0]
                    comp_ids = [ids[i] for i in comp_indices]
                    comp_id_to_local = {bid: i for i, bid in enumerate(comp_ids)}
                    n_comp = len(comp_ids)

                    if n_comp == 1:
                        offsets_out[comp_ids[0]] = 0.0
                        continue

                    comp_pairs = [(id1, id2, off, w) for id1, id2, off, w in pairs
                                  if id1 in comp_id_to_local and id2 in comp_id_to_local]

                    if len(comp_pairs) == 0:
                        for bid in comp_ids:
                            offsets_out[bid] = 0.0
                        continue

                    n_pairs_comp = len(comp_pairs)
                    A = sparse.lil_matrix((n_pairs_comp + 1, n_comp))
                    b = np.zeros(n_pairs_comp + 1)
                    W = np.zeros(n_pairs_comp + 1)

                    for k, (id1, id2, off, w) in enumerate(comp_pairs):
                        i = comp_id_to_local[id1]
                        j = comp_id_to_local[id2]
                        A[k, i] = -1
                        A[k, j] = +1
                        b[k] = off
                        W[k] = w

                    constraint_weight = np.sum(W[:-1]) * 100 if np.sum(W[:-1]) > 0 else 1e6
                    A[n_pairs_comp, 0] = 1
                    b[n_pairs_comp] = 0
                    W[n_pairs_comp] = constraint_weight

                    sqrt_W = np.sqrt(W)
                    result = lsqr(sparse.diags(sqrt_W) @ A.tocsr(), sqrt_W * b)

                    for i, bid in enumerate(comp_ids):
                        offsets_out[bid] = round(float(maybe_wrap(result[0][i])), OUTPUT_PRECISION)

                return offsets_out

            else:  # degree == 1
                ramps_out = {}

                for comp in range(n_components):
                    comp_indices = np.where(labels == comp)[0]
                    comp_ids = [ids[i] for i in comp_indices]
                    comp_id_to_local = {bid: i for i, bid in enumerate(comp_ids)}
                    n_comp = len(comp_ids)

                    if n_comp == 1:
                        ramps_out[comp_ids[0]] = [0.0, 0.0]
                        continue

                    comp_pairs = [(id1, id2, off, r, xc, w) for id1, id2, off, r, xc, w in pairs
                                  if id1 in comp_id_to_local and id2 in comp_id_to_local]

                    if len(comp_pairs) == 0:
                        for bid in comp_ids:
                            ramps_out[bid] = [0.0, 0.0]
                        continue

                    n_pairs_comp = len(comp_pairs)
                    A = sparse.lil_matrix((n_pairs_comp + 1, n_comp))
                    b = np.zeros(n_pairs_comp + 1)
                    W = np.zeros(n_pairs_comp + 1)

                    for k, (id1, id2, off, ramp_diff, xc, w) in enumerate(comp_pairs):
                        i = comp_id_to_local[id1]
                        j = comp_id_to_local[id2]
                        A[k, i] = -1
                        A[k, j] = +1
                        b[k] = ramp_diff
                        W[k] = w

                    constraint_weight = np.sum(W[:-1]) * 100 if np.sum(W[:-1]) > 0 else 1e6
                    A[n_pairs_comp, 0] = 1
                    b[n_pairs_comp] = 0
                    W[n_pairs_comp] = constraint_weight

                    sqrt_W = np.sqrt(W)
                    result = lsqr(sparse.diags(sqrt_W) @ A.tocsr(), sqrt_W * b)

                    for i, bid in enumerate(comp_ids):
                        ramp = round(float(result[0][i]), RAMP_PRECISION)
                        intercept = round(-ramp * x_centers[bid], OUTPUT_PRECISION)
                        ramps_out[bid] = [ramp, intercept]

                return ramps_out

        # Solve for each pair
        results_per_pair = [solve_for_pair(p) for p in range(n_pairs)]

        # Compute residuals from the pairwise offsets (before correction)
        # This uses the same overlap data we already computed
        residuals_out = None
        if return_residuals:
            # Calculate weighted mean absolute offset per pair
            disc_per_pair = []
            for p in range(n_pairs):
                pairs_p = pairs_by_pair_idx[p]
                if len(pairs_p) == 0:
                    disc_per_pair.append(0.0)
                    continue

                # Extract offsets and weights
                if degree == 0:
                    # pairs_p = [(id1, id2, offset, weight), ...]
                    offsets = [abs(maybe_wrap(t[2])) for t in pairs_p]
                    weights = [t[3] for t in pairs_p]
                else:
                    # pairs_p = [(id1, id2, offset, ramp, x_centroid, weight), ...]
                    offsets = [abs(maybe_wrap(t[2])) for t in pairs_p]
                    weights = [t[5] for t in pairs_p]

                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_sum = sum(o * w for o, w in zip(offsets, weights))
                    disc_per_pair.append(round(weighted_sum / total_weight, 3))
                else:
                    disc_per_pair.append(0.0)

            if n_pairs == 1 and not has_pair_dim:
                residuals_out = disc_per_pair[0]
            else:
                residuals_out = disc_per_pair

            if debug:
                print(f'Input residuals: {residuals_out}', flush=True)

        # If single pair, return simple dict
        if n_pairs == 1 and not has_pair_dim:
            coeffs = results_per_pair[0]
            return (coeffs, residuals_out) if return_residuals else coeffs

        # Multiple pairs: combine into list per burst
        combined = {}
        for bid in ids:
            if degree == 0:
                combined[bid] = [results_per_pair[p][bid] for p in range(n_pairs)]
            else:
                combined[bid] = [results_per_pair[p][bid] for p in range(n_pairs)]

        return (combined, residuals_out) if return_residuals else combined

    def align(self,
              degree: int = 0,
              method: str = 'median',
              debug: bool = False,
              return_residuals: bool = False):
        """
        Align burst interferograms by removing phase offsets and optionally ionospheric ramps.

        Uses a multi-step approach for optimal alignment:
        - degree=0: Single-step offset correction
        - degree=1: 3-step correction (offset → ramp → re-offset) for ionospheric ramp removal

        The 3-step approach produces consistent fringes across bursts by removing
        per-track ionospheric ramps, which is essential for deformation analysis.

        Parameters
        ----------
        degree : int, optional
            Correction degree:
            - 0 (default): Offset-only correction (faster, good overlap alignment)
            - 1: Offset + linear ramp correction (better fringe continuity)
        method : str, optional
            Estimation method: 'median' (robust, default) or 'mean' (faster).
        debug : bool, optional
            Print debug information. Default is False.
        return_residuals : bool, optional
            If True, also return final residuals. Default is False.

        Returns
        -------
        BatchCore or tuple
            If return_residuals is False:
                Aligned interferograms with phase corrections applied.
            If return_residuals is True:
                (aligned_intfs, residuals) where residuals is float or list[float]

        Examples
        --------
        >>> # Simple offset-only alignment (default)
        >>> aligned = intfs.align()
        >>>
        >>> # Alignment with ramp correction
        >>> aligned = intfs.align(degree=1)
        >>>
        >>> # With coherence filtering
        >>> aligned = intfs.where(corr >= 0.3).align()
        >>>
        >>> # Get alignment quality with result
        >>> aligned, res = intfs.align(return_residuals=True)
        >>> print('Residuals:', res)

        Notes
        -----
        For degree=1, the function performs:
        1. Estimate and remove offsets
        2. Estimate and remove ramps (using same-track overlaps only)
        3. Re-estimate offsets on ramp-corrected data
        4. Combine into final [ramp, offset] coefficients

        This 3-step approach achieves better fringe continuity than single-step
        methods because it separates the offset and ramp estimation, avoiding
        cross-contamination between the two.

        Use ``align(degree=0)`` (offset-only) when cross-subswath consistency
        matters most. Use ``align(degree=1)`` when per-track fringe continuity
        is more important than cross-subswath boundaries.
        """
        from .Batch import Batch, BatchWrap

        # Validate class type
        if not isinstance(self, (Batch, BatchWrap)):
            raise TypeError(f"align() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")

        if degree == 0:
            # Single-step offset correction
            if debug:
                print('align(degree=0): single-step offset correction', flush=True)
                res_in = self.residuals()
                print(f'Input residuals: {res_in}', flush=True)

            offsets = self.fit(degree=0, method=method, debug=debug)
            aligned = self - offsets

            if debug or return_residuals:
                res_out = aligned.residuals()
                if debug:
                    print(f'Output residuals: {res_out}', flush=True)

            if return_residuals:
                return aligned, res_out
            return aligned

        elif degree == 1:
            # 3-step offset-ramp-offset correction
            if debug:
                print('align(degree=1): 3-step offset-ramp-offset correction', flush=True)
                res_in = self.residuals()
                print(f'Input residuals: {res_in}', flush=True)

            # Step 1: Estimate offsets
            if debug:
                print('\nStep 1: Estimate offsets...', flush=True)
            offsets1 = self.fit(degree=0, method=method, debug=debug)
            intfs1 = self - offsets1
            if debug:
                res1 = intfs1.residuals()
                print(f'Residuals after step 1: {res1}', flush=True)

            # Step 2: Estimate ramps (uses same-track overlaps only)
            if debug:
                print('\nStep 2: Estimate ramps...', flush=True)
            ramps = intfs1.fit(degree=1, method=method, debug=debug)
            intfs2 = intfs1 - intfs1.polyval(ramps)
            if debug:
                res2 = intfs2.residuals()
                print(f'Residuals after step 2: {res2}', flush=True)

            # Step 3: Re-estimate offsets
            if debug:
                print('\nStep 3: Re-estimate offsets...', flush=True)
            offsets2 = intfs2.fit(degree=0, method=method, debug=debug)

            # Combine coefficients: [ramp, offset1 + ramp_intercept + offset2]
            # Detect if multi-pair
            sample_bid = list(offsets1.keys())[0]
            is_multi_pair = isinstance(offsets1[sample_bid], list)

            if is_multi_pair:
                n_pairs = len(offsets1[sample_bid])
                coeffs = {
                    b: [[ramps[b][p][0], ramps[b][p][1] + offsets1[b][p] + offsets2[b][p]]
                        for p in range(n_pairs)]
                    for b in offsets1
                }
            else:
                coeffs = {
                    b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]]
                    for b in offsets1
                }

            aligned = self - self.polyval(coeffs)

            if debug or return_residuals:
                res_out = aligned.residuals()
                if debug:
                    print(f'Final residuals: {res_out}', flush=True)

            if return_residuals:
                return aligned, res_out
            return aligned

        else:
            raise ValueError(f"degree must be 0 or 1, got {degree}")
