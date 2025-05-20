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

def regression1d(data, dim='auto', degree=1, wrap=False):
    import xarray as xr
    import pandas as pd
    import numpy as np

    def wrap(data):
        return np.mod(data + np.pi, 2 * np.pi) - np.pi

    multi_index = None
    if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
        multi_index = data['stack']
        # detect unused coordinates
        unused_coords = [d for d in multi_index.coords if not d in multi_index.dims and not d in multi_index.indexes]
        # cleanup multiindex to merge it with the processed dataset later
        multi_index = multi_index.drop_vars(unused_coords)
        data = data.reset_index('stack')

    stackdim = [_dim for _dim in ['date', 'pair'] if _dim in data.dims]
    if len(stackdim) != 1:
        raise ValueError("The 'data' argument must include a 'date' or 'pair' dimension to detect trends.")
    stackdim = stackdim[0]

    if isinstance(dim, str) and dim == 'auto':
        dim = stackdim

    # add new coordinate using 'dim' values
    if not isinstance(dim, str):
        if isinstance(dim, (xr.DataArray, pd.DataFrame, pd.Series)):
            dim_da = xr.DataArray(dim.values, dims=[stackdim])
        else:
            dim_da = xr.DataArray(dim, dims=[stackdim])
        data_dim = data.assign_coords(polyfit_coord=dim_da).swap_dims({'pair': 'polyfit_coord'})
        
    if wrap:
        # wrap to prevent outrange
        data = wrap(data)
        # fit sine/cosine
        trend_sin = regression1d(np.sin(data), dim, degree=degree, wrap=False)
        trend_cos = regression1d(np.cos(data), dim, degree=degree, wrap=False)
        # define the angle offset at zero baseline
        trend_sin0 = xr.polyval(xr.DataArray(0, dims=[]), trend_sin.coefficients)
        trend_cos0 = xr.polyval(xr.DataArray(0, dims=[]), trend_cos.coefficients)
        fit = np.arctan2(trend_sin, trend_cos) - np.arctan2(trend_sin0, trend_cos0)
        del trend_sin, trend_cos, trend_sin0, trend_cos0
        # wrap to prevent outrange
        return wrap(fit)

    # add new coordinate using 'dim' values
    if not isinstance(dim, str):
        # fit the specified values
        # Polynomial coefficients, highest power first, see numpy.polyfit
        fit_coeff = data_dim.polyfit('polyfit_coord', degree).polyfit_coefficients.astype(np.float32)
        fit = xr.polyval(data_dim['polyfit_coord'], fit_coeff)\
            .swap_dims({'polyfit_coord': stackdim}).drop_vars('polyfit_coord').astype(np.float32).rename('trend')
        out = xr.merge([fit, fit_coeff]).rename(polyfit_coefficients='coefficients')
        if multi_index is not None:
            return out.assign_coords(stack=multi_index)
        return out

    # fit existing coordinate values
    # Polynomial coefficients, highest power first, see numpy.polyfit
    fit_coeff = data.polyfit(dim, degree).polyfit_coefficients.astype(np.float32)
    fit = xr.polyval(data[dim], fit_coeff).astype(np.float32).rename('trend')
    out = xr.merge([fit, fit_coeff]).rename(polyfit_coefficients='coefficients')
    if multi_index is not None:
        return out.assign_coords(stack=multi_index)
    return out
