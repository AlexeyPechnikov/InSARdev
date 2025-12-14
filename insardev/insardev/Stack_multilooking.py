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
from .Stack_phasediff import Stack_phasediff
from .utils_gaussian import nanconvolve2d_gaussian
from .utils_xarray import get_spacing

class Stack_multilooking(Stack_phasediff):

    def _multilooking(self, data, weight=None, wavelength=None, gaussian_threshold=0.5, debug=False):
        import xarray as xr
        import numpy as np
        import dask
    
        # GMTSAR constant 5.3 defines half-gain at filter_wavelength
        cutoff = 5.3

        # no-op, processing is needed
        if wavelength is None:
            return data
    
        # calculate sigmas based on wavelength
        dy, dx = get_spacing(data)
        sigmas = [wavelength / dy / cutoff, wavelength / dx / cutoff]
        if debug:
            print(f'DEBUG: multilooking sigmas ({sigmas[0]:.2f}, {sigmas[1]:.2f}), wavelength {wavelength:.1f}')

        if isinstance(data, xr.Dataset):
            dims = data[list(data.data_vars)[0]].dims
        else:
            dims = data.dims

        if len(dims) == 2:
            stackvar = None
        else:
            stackvar = dims[0]
        #print ('stackvar', stackvar)

        if weight is not None:
            # for InSAR processing expect 2D weights
            assert isinstance(weight, xr.DataArray) and len(weight.dims)==2, \
                'ERROR: multilooking weight should be 2D DataArray'
        
        if weight is not None and len(data.dims) == len(weight.dims):
            #print ('2D check shape weighted')
            # single 2D grid processing
            if isinstance(data, xr.Dataset):
                for varname in data.data_vars:
                    assert data[varname].shape == weight.shape, \
                        f'ERROR: multilooking data[{varname}] and weight variables have different shape'
            else:
                assert data.shape == weight.shape, 'ERROR: multilooking data and weight variables have different shape'
        elif weight is not None and len(data.dims) == len(weight.dims) + 1:
            #print ('3D check shape weighted')
            # stack of 2D grids processing
            if isinstance(data, xr.Dataset):
                for varname in data.data_vars:
                    assert data[varname].shape[1:] == weight.shape, \
                        f'ERROR: multilooking data[{varname}] slice and weight variables have different shape \
                        ({data[varname].shape[1:]} vs {weight.shape})'
            else:
                assert data.shape[1:] == weight.shape, f'ERROR: multilooking data slice and weight variables have different shape \
                ({data.shape[1:]} vs {weight.shape})'

        # process a slice of dataarray
        def process_slice(slice_data):
            conv = nanconvolve2d_gaussian(slice_data, weight, sigmas, threshold=gaussian_threshold)
            return xr.DataArray(conv, dims=slice_data.dims, name=slice_data.name)

        # process stack of dataarray slices
        def process_slice_var(dataarray):    
            if stackvar:
                stack = [process_slice(dataarray[ind]) for ind in range(len(dataarray[stackvar]))]
                return xr.concat(stack, dim=stackvar).assign_coords(dataarray.coords)
            else:
                return process_slice(dataarray).assign_coords(dataarray.coords)

        if isinstance(data, xr.Dataset):
            print ('X')
            ds = xr.Dataset({varname: process_slice_var(data[varname]) for varname in data.data_vars})
        else:
            ds = process_slice_var(data)

        # coordinates need to be numpy arrays
        return ds
