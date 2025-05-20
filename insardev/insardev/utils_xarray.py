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
import xarray as xr

def get_spacing(da, coarsen=(1, 1)):
    import numpy as np
    if coarsen is None:
        coarsen = (1, 1)
    if not isinstance(coarsen, (list, tuple, np.ndarray)):
        coarsen = (coarsen, coarsen)
    dy = da.y.diff('y').item(0)
    dx = da.x.diff('x').item(0)
    if coarsen is not None:
        dy *= coarsen[0]
        dx *= coarsen[1]
    return (dy, dx)

def coarsen_start(da, name, spacing, grid_factor=1):
    """
    Calculate start coordinate to align coarsened grids.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    name : str
        Coordinate name to align
    spacing : int
        Coarsening spacing
    grid_factor : int, optional
        Grid factor for alignment, default is 1
        
    Returns
    -------
    int or None
        Start index for optimal alignment, or None if no good alignment found
    """
    import numpy as np
    
    # get coordinate values
    coords = da[name].values
    if len(coords) < spacing:
        print(f'_coarsen_start: Not enough points for spacing {spacing}')
        return None
        
    # calculate coordinate differences
    diffs = np.diff(coords)
    if not np.allclose(diffs, diffs[0], rtol=1e-5):
        print(f'_coarsen_start: Non-uniform spacing detected for {name}')
        return None
        
    # calculate target spacing
    target_spacing = diffs[0] * spacing * grid_factor
    
    # find best alignment point
    best_offset = None
    min_error = float('inf')
    
    for i in range(spacing):
        # get coarsened coordinates
        coarse_coords = coords[i::spacing]
        if len(coarse_coords) < 2:
            continue
            
        # calculate alignment error
        error = np.abs(coarse_coords[0] % target_spacing)
        if error < min_error:
            min_error = error
            best_offset = i
            
    if best_offset is not None:
        #print(f'_coarsen_start: {name} spacing={spacing} grid_factor={grid_factor} => {best_offset} (error={min_error:.2e})')
        return best_offset
        
    print(f'_coarsen_start: No good alignment found for {name}')
    return None

#decimator = lambda da: da.coarsen({'y': 2, 'x': 2}, boundary='trim').mean()
def downsampler(grid, coarsen=None, resolution=60, func='mean', wrap=False, debug=False):
    """
    Return function for pixel decimation to the specified output resolution.

    Parameters
    ----------
    grid : xarray object
        Grid to define the spacing.
    resolution : int, optional
        DEM grid resolution in meters. The same grid is used for geocoded results output.
    debug : bool, optional
        Boolean flag to print debug information.

    Returns
    -------
    callable
        Post-processing lambda function.

    Examples
    --------
    Decimate computed interferograms to default DEM resolution 60 meters:
    decimator = stack.decimator()
    stack.intf(pairs, func=decimator)
    """
    import numpy as np
    import dask
    import warnings
    # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', module='dask')
    warnings.filterwarnings('ignore', module='dask.core')

    dy, dx = get_spacing(grid, coarsen)
    yscale, xscale = int(np.round(resolution/dy)), int(np.round(resolution/dx))
    if debug:
        print (f'DEBUG: ground pixel size in meters: y={dy:.1f}, x={dx:.1f}')
    if yscale <= 1 and xscale <= 1:
        # decimation impossible
        if debug:
            print (f'DEBUG: decimator = lambda da: da')
        return lambda da: da
    if debug:
        print (f"DEBUG: decimator = lambda da: da.coarsen({{'y': {yscale}, 'x': {xscale}}}, boundary='trim').{func}()")

    # decimator function
    def decimator(datas):
        def decimator_dataset(ds):
            y_chunksize = ds.chunks[-2][0]
            x_chunksize = ds.chunks[-1][0]
            print ('ds.chunks', y_chunksize, x_chunksize)
            coarsen_args = {'y': yscale, 'x': xscale}
            # calculate coordinate offsets to align coarsened grids
            y0 = coarsen_start(ds, 'y', yscale)
            x0 = coarsen_start(ds, 'x', xscale)
            ds = ds.isel({'y': slice(y0, None), 'x': slice(x0, None)})
            if wrap:
                da_complex = np.exp(1j * ds.astype(np.float32))
                da_complex_agg = getattr(da_complex\
                        .coarsen(coarsen_args, boundary='trim'), func)()\
                        .astype(np.complex64)\
                        .chunk({'y': y_chunksize, 'x': x_chunksize})
                da_decimated = np.arctan2(da_complex_agg.imag, da_complex_agg.real).astype(np.float32)
                del da_complex, da_complex_agg
                return da_decimated
            else:
                return getattr(ds\
                        .coarsen(coarsen_args, boundary='trim'), func)()\
                        .astype(np.float32)\
                        .chunk({'y': y_chunksize, 'x': x_chunksize})
        # avoid creating the large chunks
        #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        if isinstance(datas, dict):
            return {k:decimator_dataset(v) for k,v in datas.items()}
        else:
            return decimator_dataset(datas)

    # return callback function and set common chunk size
    return lambda datas: decimator(datas)

def downsampler_interferogram(grid, coarsen=None, resolution=60, func='mean', debug=False):
    return downsampler(grid, coarsen, resolution, func, wrap=True, debug=debug)

def downsampler_correlation(grid, coarsen=None, resolution=60, func='mean', debug=False):
    return downsampler(grid, coarsen, resolution, func, wrap=False, debug=debug)


def to_dict(datas: dict[str, xr.Dataset | xr.DataArray] | None = None):
    """
    Convert a list of datasets or dictionaries of datasets to a dictionary.
    """
    if isinstance(datas, xr.Dataset):
        return {'default': datas}
    if isinstance(datas, xr.DataArray):
        return {'default': datas.to_dataset()}
    return datas

def apply(*args, **kwarg):
    """
    Apply a function to multiple datasets or dictionaries of datasets with the same keys.

    Parameters
    ----------
    *args : list of datasets or dictionaries of datasets
        The datasets to apply the function to.
    **kwarg : dict
        The keyword arguments to pass to the function.

    Returns
    -------
    dict or dataset
        The result of applying the function to the datasets.
        If the input is a dictionary or a list of dictionaries, the result is a dictionary.
        If the input is a dataset or a list of datasets, the result is a dataset.
    
    Examples
    --------
    >>> sbas.apply(func=lambda a, b, **kwargs: (a, b))
    >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: (a, b))
    >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: a)
    >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: (a,b))
    >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: a)
    """
    from insardev_toolkit import progressbar
    import dask

    func = kwarg.pop('func', None)
    if func is None:
        raise ValueError('`func` argument is required')
    compute = kwarg.pop('compute', False)
    #print ('compute', compute)
    add_key = kwarg.pop('add_key', False)
    if not args:
        return
    datas = [to_dict(arg) if arg is not None else None for arg in args]
    keys = list(datas[0].keys())
    if add_key:
        dss = {key: func(*(d[key] if d is not None else None for d in datas), **(kwarg | {'key': key})) for key in keys}
    else:
        dss = {key: func(*(d[key] if d is not None else None for d in datas), **kwarg) for key in keys}
    if compute:
        progressbar(dss := dask.persist(dss)[0], desc=f'Computing...'.ljust(25))
    # detect output type
    sample = next(iter(dss.values()))
    # multiple datasets or dictionaries
    if (isinstance(sample, (tuple, list))):
        n = len(sample)
        dicts = [{key: dss[key][i] for key in keys} for i in range(n)]
        if isinstance(args[0], dict):
            return tuple(dicts)
        return tuple(d['default'] for d in dicts)
    # single dataset or dictionary
    if isinstance(args[0], dict):
        return dss
    return dss['default']

def apply_pol(*args, **kwarg):
    """
    Apply a function to multiple datasets or dictionaries of datasets with the same keys.
    The function process a single polarization at a time and then the results are merged.

    Parameters
    ----------
    *args : list of datasets or dictionaries of datasets
        The datasets to apply the function to.
    **kwarg : dict
        The keyword arguments to pass to the function.

    Returns
    -------
    dict or dataset
        The result of applying the function to the datasets.
        If the input is a dictionary or a list of dictionaries, the result is a dictionary.
        If the input is a dataset or a list of datasets, the result is a dataset.
    
    Examples
    --------
    >>> sbas.apply(func=lambda a, b, **kwargs: (a, b))
    >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: (a, b))
    >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: a)
    >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: (a,b))
    >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: a)
    """
    from insardev_toolkit import progressbar
    import xarray as xr
    import dask

    func = kwarg.pop('func', None)
    if func is None:
        raise ValueError('`func` argument is required')
    compute = kwarg.pop('compute', False)
    #print ('compute', compute)
    add_key = kwarg.pop('add_key', False)
    #print ('kwarg', kwarg)
    if not args:
        return
    datas = [to_dict(arg) if arg is not None else None for arg in args]
    # detect input type
    sample = next(iter(datas[0].values()))
    polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in sample.data_vars]
    #print ('polarizations', polarizations)
    keys = list(datas[0].keys())
    dss = []
    for polarization in polarizations:
        if add_key:
            dss_pol = {key: func(*(d[key][polarization] if d is not None else None for d in datas), **(kwarg | {'key': key})) for key in keys}
        else:
            dss_pol = {key: func(*(d[key][polarization] if d is not None else None for d in datas), **kwarg) for key in keys}
        if compute:
            progressbar(dss_pol := dask.persist(dss_pol)[0], desc=f'Computing {polarization}...'.ljust(25))
        dss.append(dss_pol)
        del dss_pol
    # detect output type
    sample = next(iter(dss[0].values()))
    #print ('sample', sample)
    # multiple datasets or dictionaries
    if (isinstance(sample, (tuple, list))):
        n = len(sample)
        dicts = [{key: xr.merge([dss[pidx][key][i] for pidx in range(len(polarizations))]) for key in keys} for i in range(n)]
        if isinstance(args[0], dict):
            return tuple(dicts)
        return tuple(d['default'] for d in dicts)
    # single dataset or dictionary
    # unpack polarizations
    dss = {key: xr.merge([dss[pidx][key] for pidx in range(len(polarizations))]) for key in keys}
    if isinstance(args[0], dict):
        return dss
    return dss['default']
