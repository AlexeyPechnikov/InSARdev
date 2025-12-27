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

def regression2d(data, variables, weight=None, algorithm='linear', degree=1, wrap=False, valid_pixels_threshold=1000, **kwargs):
    """
    topo = sbas.get_topo().coarsen({'x': 4}, boundary='trim').mean()
    yy, xx = xr.broadcast(topo.y, topo.x)
    strat_sbas = sbas.regression(unwrap_sbas.phase,
            [topo,    topo*yy,    topo*xx,    topo*yy*xx,
                topo**2, topo**2*yy, topo**2*xx, topo**2*yy*xx,
                topo**3, topo**3*yy, topo**3*xx, topo**3*yy*xx,
                yy, xx,
                yy**2, xx**2, yy*xx,
                yy**3, xx**3, yy**2*xx, xx**2*yy], corr_sbas)

    topo = sbas.interferogram(topophase)
    inc = decimator(sbas.incidence_angle())
    yy, xx = xr.broadcast(topo.y, topo.x)
    variables = [topo,  topo*yy,  topo*xx, topo*yy*xx]
    trend_sbas = sbas.regression(intf_sbas, variables, corr_sbas)
    """
    import numpy as np
    import xarray as xr
    import dask
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import SGDRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    
    # find stack dim
    stackvar = data.dims[0] if len(data.dims) >= 3 else 'stack'
    #print ('stackvar', stackvar)
    shape2d = data.shape[1:] if len(data.dims) == 3 else data.shape
    #print ('shape2d', shape2d)
    chunk2d = data.chunks[1:] if len(data.dims) == 3 else data.chunks
    # Handle non-chunked data (e.g., when called inside dask.delayed)
    if chunk2d is None:
        chunk2d = shape2d
    #print ('chunk2d', chunk2d)

    def regression_block(data, weight, *args, **kwargs):
        data_values  = data.ravel()
        # manage variable number of variables
        variables_stack = np.stack([arg[0] if arg.ndim==3 else arg for arg in args])
        #variables_values = variables_stack.reshape(-1, variables_stack.shape[0]).T
        variables_values = variables_stack.reshape(variables_stack.shape[0], -1)
        del variables_stack
        #assert 0, f'TEST: {data_values.shape}, {variables_values.shape}'

        nanmask_data = ~np.isfinite(data_values)
        nanmask_values = np.any(~np.isfinite(variables_values), axis=0)
        if weight.size > 1:
            weight_values = weight.ravel().astype(np.float64)
            nanmask_weight = ~np.isfinite(weight_values)
            nanmask = nanmask_data | nanmask_values | nanmask_weight
            #assert 0, f'TEST weight: {data_values.shape}, {variables_values.shape}, {weight_values.shape}'
        else:
            weight_values = None
            nanmask_weight = None
            nanmask = nanmask_data | nanmask_values
        del nanmask_data, nanmask_weight

        # regression requires enough amount of valid pixels
        if data_values.size - np.sum(nanmask) < valid_pixels_threshold:
            del data_values, variables_values, weight_values, nanmask
            return np.nan * np.zeros(data.shape)

        # prepare target Y for regression
        if wrap:
            # convert angles to sine and cosine as (N,2) -> (sin, cos) matrix
            #Y = np.column_stack([np.sin(data_values), np.cos(data_values)]).astype(np.float64)
            P = np.exp(1j * data_values)
            # log(P) = ln|P| + i·arg(P); since |P|=1, the real part is zero
            #  maginary part is wrapped phase in (–π,π]
            Y = np.imag(np.log(P)).astype(np.float64).ravel()
        else:
            # just use data values as (N) matrix
            Y = data_values.reshape(-1).astype(np.float64)
        del data_values

        # build prediction model
        if algorithm == 'linear':
            #regr = make_pipeline(StandardScaler(), LinearRegression(copy_X=False, n_jobs=1, **kwargs))
            regr = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                StandardScaler(),
                LinearRegression(copy_X=False, n_jobs=1, **kwargs)
            )
            fit_params = {'linearregression__sample_weight': weight_values[~nanmask]} if weight_values is not None else {}
        elif algorithm == 'sgd':
            #regr = make_pipeline(StandardScaler(), SGDRegressor(**kwargs))
            regr = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                StandardScaler(),
                SGDRegressor(**kwargs)
            )
            fit_params = {'sgdregressor__sample_weight': weight_values[~nanmask]} if weight_values is not None else {}
        elif algorithm == 'hgb':
            regr = make_pipeline(
                StandardScaler(),
                HistGradientBoostingRegressor(**kwargs)
            )
            fit_params = {'histgradientboostingregressor__sample_weight': weight_values[~nanmask]} if weight_values is not None else {}
        else:
            raise ValueError(f"Unsupported algorithm {algorithm}. Should be 'linear', 'sgd', or 'hgb'.")
        del weight_values

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            regr.fit(variables_values[:, ~nanmask].T, Y[~nanmask], **fit_params)
        del Y, nanmask

        # Predict for all valid pixels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            model_pred = regr.predict(variables_values[:, ~nanmask_values].T)
        del regr, variables_values

        model = np.full_like(data, np.nan).ravel()
        if wrap:
            # (N,2) -> (sin, cos)
            #model[~nanmask_values] = np.arctan2(model_pred[:, 0], model_pred[:, 1])
            # wrap to [–π,π)
            model[~nanmask_values] = (model_pred.ravel() + np.pi) % (2*np.pi) - np.pi
        else:
            # (N,1), just flatten
            model[~nanmask_values] = model_pred.ravel()
        del model_pred, nanmask_values
        
        return model.reshape(data.shape).astype(np.float32)

    dshape = data[0].shape if data.ndim==3 else data.shape
    if isinstance(variables, (list, tuple)):
        vshapes = [v[0].shape if v.ndim==3 else v.shape for v in variables]
        equals = np.all([vshape == dshape for vshape in vshapes])
        if not equals:
            print (f'NOTE: shapes of variables slices {vshapes} and data slice {dshape} differ.')
        #assert equals, f'{dshape} {vshapes}, {equals}'
        variables_stack = [v.reindex_like(data).chunk(dict(y=chunk2d[0], x=chunk2d[1]))  for v in variables]
    else:
        vshape = variables[0].shape if variables.ndim==3 else variables.shape
        if not {vshape} == {dshape}:
            print (f'NOTE: shapes of variables slice {vshape} and data slice {dshape} differ.')
        variables_stack = [variables.reindex_like(data).chunk(dict(y=chunk2d[0], x=chunk2d[1]))]

    if weight is not None:
        if not weight.shape == data.shape:
            print (f'NOTE: shapes of weight {weight.shape} and data {data.shape} differ.')
        weight_stack = weight.reindex_like(data).chunk(dict(y=chunk2d[0], x=chunk2d[1]))
    else:
        weight_stack = None

    # xarray wrapper
    model = xr.apply_ufunc(
        regression_block,
        data,
        weight_stack,
        *variables_stack,
        dask='parallelized',
        vectorize=False,
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={**kwargs},
    )
    del variables_stack

    return model.rename(data.name)

# def _regression2d_linear(self, data, variables, weight=None, degree=1, wrap=False, valid_pixels_threshold=1000, fit_intercept=True):
#     """   
#     topo = sbas.get_topo().coarsen({'x': 4}, boundary='trim').mean()
#     yy, xx = xr.broadcast(topo.y, topo.x)
#     strat_sbas = sbas.regression_linear(unwrap_sbas.phase,
#             [topo,    topo*yy,    topo*xx,    topo*yy*xx,
#              topo**2, topo**2*yy, topo**2*xx, topo**2*yy*xx,
#              topo**3, topo**3*yy, topo**3*xx, topo**3*yy*xx,
#              yy, xx,
#              yy**2, xx**2, yy*xx,
#              yy**3, xx**3, yy**2*xx, xx**2*yy], corr_sbas)
#     """
#     return self._regression2d(
#         data,
#         variables,
#         weight=weight,
#         wrap=wrap,
#         valid_pixels_threshold=valid_pixels_threshold,
#         algorithm='linear',
#         degree=degree,
#         fit_intercept=fit_intercept
#     )

# def _regression2d_sgd(self, data, variables, weight=None, degree=1, wrap=False, valid_pixels_threshold=1000,
#                   penalty='elasticnet', max_iter=1000, tol=1e-3, alpha=0.0001, l1_ratio=0.15):
#     """
#     Perform regression on a dataset using the SGDRegressor model from scikit-learn.

#     This function applies Stochastic Gradient Descent (SGD) regression to fit the given 'data' against a set of 'variables'.
#     It's suitable for large datasets and handles high-dimensional features efficiently.

#     Parameters:
#     data (xarray.DataArray): The target data array to fit.
#     variables (xarray.DataArray or list of xarray.DataArray): Predictor variables. It can be a single 3D DataArray or a list of 2D DataArrays.
#     weight (xarray.DataArray, optional): Weights for each data point. Useful if certain data points are more important. Defaults to None.
#     valid_pixels_threshold (int, optional): Minimum number of valid pixels required for the regression to be performed. Defaults to 10000.
#     max_iter (int, optional): Maximum number of passes over the training data (epochs). Defaults to 1000.
#     tol (float, optional): Stopping criterion. If not None, iterations will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs. Defaults to 1e-3.
#     alpha (float, optional): Constant that multiplies the regularization term. Higher values mean stronger regularization. Defaults to 0.0001.
#     l1_ratio (float, optional): The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.

#     Returns:
#     xarray.DataArray: The predicted values as an xarray DataArray, fitted by the SGDRegressor model.

#     Notes:
#     - SGDRegressor is well-suited for large datasets due to its efficiency in handling large-scale and high-dimensional data.
#     - Proper tuning of parameters (max_iter, tol, alpha, l1_ratio) is crucial for optimal performance and prevention of overfitting.

#     Example:
#     decimator = sbas.decimator(resolution=15, grid=(1,1))
#     topo = decimator(sbas.get_topo())
#     inc = decimator(sbas.incidence_angle())
#     yy, xx = xr.broadcast(topo.y, topo.x)
#     trend_sbas = sbas.regression(unwrap_sbas.phase,
#             [topo,    topo*yy,    topo*xx,    topo*yy*xx,
#              topo**2, topo**2*yy, topo**2*xx, topo**2*yy*xx,
#              topo**3, topo**3*yy, topo**3*xx, topo**3*yy*xx,
#              inc,     inc**yy,    inc*xx,     inc*yy*xx,
#              yy, xx,
#              yy**2, xx**2, yy*xx,
#              yy**3, xx**3, yy**2*xx, xx**2*yy], corr_sbas)
#     """
#     return self._regression2d(
#         data,
#         variables,
#         weight=weight,
#         wrap=wrap,
#         valid_pixels_threshold=valid_pixels_threshold,
#         algorithm='sgd',
#         degree=degree,
#         penalty=penalty,
#         max_iter=max_iter,
#         tol=tol,
#         alpha=alpha,
#         l1_ratio=l1_ratio
#     )

# def _regression2d_xgboost(self, data, variables, weight=None, wrap=False, valid_pixels_threshold=1000,
#                        n_estimators=100, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=1, **kwargs):
#     """
#     Perform regression on a dataset using XGBoost (XGBRegressor).
#     https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
#     https://xgboost.readthedocs.io/en/stable/parameter.html

#     Parameters:
#     -----------
#     data : xarray.DataArray
#         The target data array to fit.
#     variables : list[xarray.DataArray] or xarray.DataArray
#         Predictor variables.
#     weight : xarray.DataArray, optional
#         Sample weights per data point. Defaults to None.
#     valid_pixels_threshold : int, optional
#         Minimum valid pixels required for the regression to run. Defaults to 1000.
#     n_estimators : int, optional
#         Number of boosting rounds. Defaults to 100.
#     learning_rate : float, optional
#         Step size shrinkage used in update to prevents overfitting. Defaults to 0.05.
#     max_depth : int, optional
#         Maximum depth of a tree. Defaults to 6.
#     kwargs :
#         Additional keyword arguments passed to XGBRegressor.

#     Returns:
#     --------
#     xarray.DataArray
#         The predicted values as an xarray DataArray, fitted by XGBRegressor.
#     """
#     return self._regression2d(
#         data,
#         variables,
#         weight=weight,
#         wrap=wrap,
#         valid_pixels_threshold=valid_pixels_threshold,
#         algorithm='xgb',
#         degree=None,
#         n_estimators=n_estimators,
#         learning_rate=learning_rate,
#         max_depth=max_depth,
#         subsample=subsample,
#         colsample_bytree=colsample_bytree,
#         **kwargs
#     )
