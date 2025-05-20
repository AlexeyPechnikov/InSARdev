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

def binary_erosion(data, *args, **kwargs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html
    """
    import xarray as xr
    from scipy.ndimage import binary_erosion
    array = binary_erosion(data.values, *args, **kwargs)
    return xr.DataArray(array, coords=data.coords, dims=data.dims, attrs=data.attrs)

def binary_dilation(data, *args, **kwargs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html
    """
    import xarray as xr
    from scipy.ndimage import binary_dilation
    array = binary_dilation(data.values, *args, **kwargs)
    return xr.DataArray(array, coords=data.coords, dims=data.dims, attrs=data.attrs)

def binary_opening(data, *args, **kwargs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_opening.html
    
    corrmask = utils.binary_closing(corrmask, structure=np.ones((10,10)))
    corrmask = utils.binary_opening(corrmask, structure=np.ones((10,10)))
    """
    import xarray as xr
    from scipy.ndimage import binary_opening
    array = binary_opening(data.values, *args, **kwargs)
    return xr.DataArray(array, coords=data.coords, dims=data.dims, attrs=data.attrs)

def binary_closing(data, *args, **kwargs):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_opening.html
    
    corrmask = utils.binary_closing(corrmask, structure=np.ones((10,10)))
    corrmask = utils.binary_opening(corrmask, structure=np.ones((10,10)))
    """
    import xarray as xr
    from scipy.ndimage import binary_closing
    array = binary_closing(data.values, *args, **kwargs)
    return xr.DataArray(array, coords=data.coords, dims=data.dims, attrs=data.attrs)

