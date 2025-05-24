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
from .BatchCore import BatchCore
import xarray as xr
from . import utils_xarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Stack import Stack
    import xarray as xr
    import numpy as np
    import inspect

class Batch(BatchCore):
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off only the real 2D vars from Stack
        if isinstance(mapping, Stack):
            real_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                # keep only non-complex data_vars that live on the ('y','x') grid
                real_vars = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind != 'c'
                    and tuple(ds[v].dims) == ('y', 'x')
                ]
                real_dict[key] = ds[real_vars]
            mapping = real_dict

        # delegate to your base class for the actual init
        super().__init__(mapping or {})
    
    def plot(
        self,
        cmap = 'turbo',
        caption='Phase, [rad]',
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["caption"] = caption
        return super().plot(*args, **kwargs)

class BatchWrap(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores wrapped phase (real values).
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchComplex)):
            raise ValueError(f'ERROR: BatchWrap does not support Stack or BatchComplex objects.')
        dict.__init__(self, mapping or {})

    # def _agg(self, name: str, dim=None, **kwargs):
    #     """
    #     Converts wrapped phase to complex numbers before aggregation and back to wrapped phase after.
    #     """
    #     print ('wrap _agg')
    #     import inspect
    #     import numpy as np
    #     out = {}
    #     for key, obj in self.items():
    #         # get the aggregation function
    #         fn = getattr(obj, name)
    #         sig = inspect.signature(fn)
            
    #         # perform aggregation in complex domain
    #         if 'dim' in sig.parameters:
    #             # intfs.mean('pair').isel(0)
    #             #agg_result = fn(dim=dim, **kwargs)
    #             complex_obj = np.exp(1j * obj.astype(np.float32))
    #             fn_complex = getattr(complex_obj, name)
    #             agg_result = fn_complex(dim=dim, **kwargs)
    #         else:
    #             #intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()
    #             agg_result = fn(**kwargs)
            
    #         # Convert back to wrapped phase
    #         out[key] = np.arctan2(agg_result.imag, agg_result.real).astype(np.float32)
            
    #     return type(self)(out)




    def _agg(self, name: str, dim=None, **kwargs):
        """
        Converts wrapped phase to complex numbers before aggregation and back to wrapped phase after.
        """
        print ('wrap _agg')
        import inspect
        import xarray as xr
        out = {}
        for key, obj in self.items():
            # get the aggregation function
            fn = getattr(obj, name)
            sig = inspect.signature(fn)
            
            # perform aggregation in complex domain
            if 'dim' in sig.parameters:
                # intfs.mean('pair').isel(0)
                #agg_result = fn(dim=dim, **kwargs)
                complex_obj = xr.ufuncs.exp(1j * obj.astype('float32'))
                #fn_complex = getattr(complex_obj, name)
                #agg_result = fn_complex(dim=dim, **kwargs)
                if name in ('var', 'std'):
                    # |E[e^(iθ)]|
                    R = xr.ufuncs.abs(complex_obj.mean(dim=dim, **kwargs))
                    if name == 'var':
                        # 1 - |E[e^(iθ)]|
                        agg_result = (1 - R)
                    else:  # std
                        # √(-2 ln|E[e^(iθ)]|)
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    fn_complex = getattr(complex_obj, name)
                    agg_result = fn_complex(dim=dim, **kwargs)
                    # convert back to wrapped phase
                    agg_result = xr.ufuncs.angle(agg_result)
            else:
                # intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()
                # already in complex domain, see coarsen()
                if name in ('var', 'std'):
                    R = xr.ufuncs.abs(obj.mean(**kwargs))
                    if name == 'var':
                        agg_result = (1 - R)
                    else:  # std
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    agg_result = fn(**kwargs)
                    agg_result = xr.ufuncs.angle(agg_result)
            
            # Convert back to wrapped phase
            out[key] = agg_result.astype('float32')
            
        return type(self)(out)



    # def _agg(self, name: str, dim=None, **kwargs):
    #     """
    #     1) If obj has an .obj attribute → it's a Coarsen wrapper on complex data.
    #        We call obj.mean() to get E[e^{iθ}], then use R=|E| for var/std or arg(E) for mean.
    #     2) Otherwise we lift θ → e^{iθ}, call the same reduction, then angle().
    #     """
    #     import numpy as np
    #     import xarray as xr
    #     import inspect

    #     out = {}
    #     for key, obj in self.items():
    #         # ───── Are we dealing with a Coarsen wrapper on complex data? ─────
    #         if hasattr(obj, "obj") and not hasattr(obj, "data_vars"):
    #             # obj is e.g. ds2.coarsen(window, …)
    #             # the underlying ds2 holds exp(1j*θ) already.
    #             co = obj

    #             if name in ("mean", "var", "std"):
    #                 # E[e^{iθ}] over each block:
    #                 zbar = co.mean(dim=dim, **kwargs)

    #                 # magnitude & angle:
    #                 R = np.abs(zbar)
    #                 if name == "mean":
    #                     agg = np.angle(zbar).astype(np.float32)
    #                 elif name == "var":
    #                     agg = (1 - R).astype(np.float32)
    #                 else:  # std
    #                     agg = np.sqrt(-2 * np.log(R)).astype(np.float32)

    #             else:
    #                 # generic: apply the same reduction on the complex blocks
    #                 fnc = getattr(co, name)
    #                 # pass dim if supported
    #                 sig = inspect.signature(fnc)
    #                 if "dim" in sig.parameters and dim is not None:
    #                     zred = fnc(dim=dim, **kwargs)
    #                 else:
    #                     zred = fnc(**kwargs)

    #                 agg = np.angle(zred).astype(np.float32)

    #         else:
    #             # ───── Not a Coarsen wrapper → fall back to lift→reduce→angle ─────
    #             # obj is a real DataArray or Dataset of θ
    #             ds = obj
    #             # map every var into complex
    #             tmp = {}
    #             for var, da in ds.data_vars.items():
    #                 da_c = np.exp(1j * da.astype(np.float32))
    #                 fnc = getattr(da_c, name)
    #                 sig = inspect.signature(fnc)
    #                 if "dim" in sig.parameters and dim is not None:
    #                     zred = fnc(dim=dim, **kwargs)
    #                 else:
    #                     zred = fnc(**kwargs)
    #                 tmp[var] = np.angle(zred).astype(np.float32)
    #             agg = xr.Dataset(tmp)

    #         out[key] = agg

    #     return type(self)(out)

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
        print ('wrap coarsen')
        import numpy as np
        out = {}
        for key, ds in self.items():
            # convert to complex numbers for proper circular statistics
            ds2 = xr.ufuncs.exp(1j * ds.astype(np.float32))
            # align each dimension
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds2, dim, factor)
                #print ('start', start)
                if start is not None:
                    ds2 = ds2.isel({dim: slice(start, None)})
            # coarsen
            out[key] = ds2.coarsen(window, **kwargs)

        return type(self)(out)

    def plot(
        self,
        cmap = 'gist_rainbow_r',
        caption='Phase, [rad]',
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["caption"] = caption
        return super().plot(*args, **kwargs)

class BatchUnit(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores correlation in the range [0,1].
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchWrap, BatchComplex)):
            raise ValueError(f'ERROR: BatchUnit does not support Stack, BatchWrap or BatchComplex objects.')
        dict.__init__(self, mapping or {})

    def plot(
        self,
        cmap = 'auto',
        caption='Correlation',
        *args,
        **kwargs
    ):
        import matplotlib.colors as mcolors
        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
        kwargs["cmap"] = cmap
        kwargs["caption"] = caption
        return super().plot(*args, **kwargs)

class BatchComplex(BatchCore):
    """
    This class has 'data' stack variable for the datasets in the dict.
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off only the complex vars from Stack
        if isinstance(mapping, Stack):
            complex_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                # keep only complex data_vars
                complex_vars = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c'
                ]
                complex_dict[key] = ds[complex_vars]
            mapping = complex_dict

        # delegate to your base class for the actual init
        super().__init__(mapping or {})

    def abs(self, **kwargs):
        return Batch(self.map_da(lambda da: xr.ufuncs.abs(da), **kwargs))

    def power(self, **kwargs):
        """ element-wise |x|², i.e. signal intensity """
        return Batch(self.map_da(lambda da: xr.ufuncs.abs(da)**2, **kwargs))

    def conj(self, **kwargs):
        """intfs.iexp().conj() for np.exp(-1j * intfs)"""
        return self.map_da(lambda da: xr.ufuncs.conj(da), **kwargs)

    def angle(self, **kwargs):
        """
        Compute element-wise phase (angle), returning a BatchWrap of float32 DataArrays in [-π, π].
        """
        return BatchWrap(self.map_da(lambda da: np.arctan2(da.imag, da.real).astype(np.float32), **kwargs))

    def angle(self, **kwargs):
        """
        Compute element-wise phase (angle) for the complex variables only,
        returning a BatchWrap of float32 DataArrays in [-π, π].
        """
        out = {}
        for k, ds in self.items():
            # select only the vars whose dtype is complex
            complex_vars = [
                var for var in ds.data_vars
                if ds[var].dtype.kind == 'c'
            ]
            if not complex_vars:
                # no complex vars → skip
                continue

            # subset to just those, then map over each DataArray
            ds_complex = ds[complex_vars]
            ds_phase = ds_complex.map(
                lambda da: xr.ufuncs.angle(da).astype('float32'),
                **kwargs
            )

            out[k] = ds_phase

        # package up as a BatchWrap (real, wrapped-phase)
        return BatchWrap(out)

    def plot(self, *args, **kwargs):
        """
        Plotting is not supported on raw complex batches.
        Convert to real values first (e.g. with .angle() or .abs()).
        """
        raise NotImplementedError(
            "BatchComplex objects do not support plot().\n"
            "Convert to a real-valued batch first, e.g.:\n"
            "  • use `.angle()` to get wrapped phase → BatchWrap\n"
            "  • use `.abs()` or `.power()` to get magnitude → Batch"
        )