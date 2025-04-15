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
from .utils import utils

class Stack_multilooking(Stack_phasediff):

    @staticmethod
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
            print(f'calculate_coarsen_start: Not enough points for spacing {spacing}')
            return None
            
        # calculate coordinate differences
        diffs = np.diff(coords)
        if not np.allclose(diffs, diffs[0], rtol=1e-5):
            print(f'calculate_coarsen_start: Non-uniform spacing detected for {name}')
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
            #print(f'calculate_coarsen_start: {name} spacing={spacing} grid_factor={grid_factor} => {best_offset} (error={min_error:.2e})')
            return best_offset
            
        print(f'calculate_coarsen_start: No good alignment found for {name}')
        return None

    @staticmethod
    def nanconvolve2d_gaussian(data,
                        weight=None,
                        sigma=None,
                        mode='reflect',
                        truncate=4.0,
                        threshold=0.5):
        """
        Convolve a data array with a Gaussian kernel.

        Parameters
        ----------
        data : xarray.DataArray
            The data array to convolve.
        weight : xarray.DataArray, optional
            The weight array to use for the convolution.
        sigma : float or tuple of floats, optional
            The standard deviation of the Gaussian kernel.
        mode : str, optional
            The mode to use for the convolution.
        truncate : float, optional
            The truncation factor for the Gaussian kernel.
        threshold : float, optional
            The threshold for the convolution.

        We use a threshold defined as a fraction of the weight as an indicator that the Gaussian window
        covers enough valid (non-NaN) pixels. When the accumulated weight is below this threshold, we replace
        the output with NaN, since the result is unreliable due to insufficient data within the window.
        This is a simple way to prevent border effects when most of the filter window is empty.
        """
        import numpy as np
        import xarray as xr
    
        if sigma is None:
            return data
    
        if not isinstance(sigma, (list, tuple, np.ndarray)):
            sigma = (sigma, sigma)
        depth = [np.ceil(_sigma * truncate).astype(int) for _sigma in sigma]
        #print ('sigma', sigma, 'depth', depth)
    
        # weighted Gaussian filtering for real floats with NaNs
        def nanconvolve2d_gaussian_floating_dask_chunk(data, weight=None, **kwargs):
            import numpy as np
            from scipy.ndimage import gaussian_filter
            assert not np.issubdtype(data.dtype, np.complexfloating)
            assert np.issubdtype(data.dtype, np.floating)
            if weight is not None:
                assert not np.issubdtype(weight.dtype, np.complexfloating)
                assert np.issubdtype(weight.dtype, np.floating)
            # all other arguments are passed to gaussian_filter
            threshold = kwargs.pop('threshold')
            # replace nan + 1j to to 0.+0.j
            data_complex  = (1j + data) * (weight if weight is not None else 1)
            conv_complex = gaussian_filter(np.nan_to_num(data_complex, 0), **kwargs)
            #conv = conv_complex.real/conv_complex.imag
            # to prevent "RuntimeWarning: invalid value encountered in divide" even when warning filter is defined
            conv = np.where(conv_complex.imag <= threshold*(weight if weight is not None else 1), np.nan, conv_complex.real/(conv_complex.imag + 1e-17))
            del data_complex, conv_complex
            return conv
    
        def nanconvolve2d_gaussian_dask_chunk(data, weight=None, **kwargs):
            import numpy as np
            if np.issubdtype(data.dtype, np.complexfloating):
                #print ('complexfloating')
                real = nanconvolve2d_gaussian_floating_dask_chunk(data.real, weight, **kwargs)
                imag = nanconvolve2d_gaussian_floating_dask_chunk(data.imag, weight, **kwargs)
                conv = real + 1j*imag
                del real, imag
            else:
                #print ('floating')
                conv = nanconvolve2d_gaussian_floating_dask_chunk(data.real, weight, **kwargs)
            return conv
    
        # weighted Gaussian filtering for real or complex floats
        def nanconvolve2d_gaussian_dask(data, weight, **kwargs):
            import dask.array as da
            # ensure both dask arrays have the same chunk structure
            # use map_overlap with the custom function to handle both arrays
            return da.map_overlap(
                nanconvolve2d_gaussian_dask_chunk,
                *([data, weight] if weight is not None else [data]),
                depth={0: depth[0], 1: depth[1]},
                boundary='none',
                dtype=data.dtype,
                meta=data._meta,
                **kwargs
            )

        return xr.DataArray(nanconvolve2d_gaussian_dask(data.data,
                                     weight.data if weight is not None else None,
                                     threshold=threshold,
                                     sigma=sigma,
                                     mode=mode,
                                     truncate=truncate),
                            coords=data.coords,
                            name=data.name)

    #decimator = lambda da: da.coarsen({'y': 2, 'x': 2}, boundary='trim').mean()
    def decimator(self, grid, coarsen=None, resolution=60, func='mean', wrap=False, debug=False):
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

        dy, dx = self.get_spacing(grid, coarsen)
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

        # decimate function
        def decimator(da):
            import warnings
            # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
            warnings.filterwarnings('ignore')
            warnings.filterwarnings('ignore', module='dask')
            warnings.filterwarnings('ignore', module='dask.core')
            # unstack data if needed
            if 'stack' in da.dims:
                # .unstack() is too slow on lazy grids in some of Xarray/Dask versions
                da = da.compute().unstack('stack')
            # workaround for Google Colab when we cannot save grids with x,y coordinate names
            # also supports geographic coordinates
            yname = [varname for varname in ['y', 'lat', 'a'] if varname in da.dims][0]
            xname = [varname for varname in ['x', 'lon', 'r'] if varname in da.dims][0]
            coarsen_args = {yname: yscale, xname: xscale}
            # calculate coordinate offsets to align coarsened grids
            y0 = self.coarsen_start(da, yname, yscale)
            x0 = self.coarsen_start(da, xname, xscale)
            # avoid creating the large chunks
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                #if func not in ['mean', 'min', 'max', 'count', 'sum']:
                #    raise ValueError(f"Unsupported function {func}. Should be 'mean','min','max','count', or 'sum'")
                # return getattr(da.coarsen(coarsen_args, boundary='trim'), func)()\
                #        .chunk({yname: self.chunksize, xname: self.chunksize})
                if wrap:

                    # complex_das = [np.exp(1j * da) for da in das]
                    # da_complex = xr.concat(xr.align(*complex_das, join='outer'), dim='stack_dim').mean('stack_dim')
                    # da = np.arctan2(da_complex.imag, da_complex.real)
                    da_complex = np.exp(1j * da.isel({yname: slice(y0, None), xname: slice(x0, None)}))
                    da_complex_agg = getattr(da_complex\
                           .coarsen(coarsen_args, boundary='trim'), func)()\
                           .chunk({yname: self.chunksize, xname: self.chunksize})
                    da_decimated = np.arctan2(da_complex_agg.imag, da_complex_agg.real)
                    del da_complex, da_complex_agg
                    return da_decimated
                else:
                    return getattr(da.isel({yname: slice(y0, None), xname: slice(x0, None)})\
                           .coarsen(coarsen_args, boundary='trim'), func)()\
                           .chunk({yname: self.chunksize, xname: self.chunksize})

        # return callback function and set common chunk size
        return lambda da: decimator(da)

    def multilooking(self, data, weight=None, wavelength=None, coarsen=None, gaussian_threshold=0.5, debug=False):
        import xarray as xr
        import numpy as np
        import dask
    
        # GMTSAR constant 5.3 defines half-gain at filter_wavelength
        cutoff = 5.3
    
        # Expand simplified definition of coarsen
        coarsen = (coarsen, coarsen) if coarsen is not None and not isinstance(coarsen, (list, tuple, np.ndarray)) else coarsen
    
        # no-op, processing is needed
        if wavelength is None and coarsen is None:
            return data
    
        # calculate sigmas based on wavelength or coarsen
        if wavelength is not None:
            dy, dx = self.get_spacing(data)
            sigmas = [wavelength / cutoff / dy, wavelength / cutoff / dx]
            if debug:
                print(f'DEBUG: multilooking sigmas ({sigmas[0]:.2f}, {sigmas[1]:.2f}), wavelength {wavelength:.1f}')
        else:
            sigmas = [coarsen[0] / cutoff, coarsen[1] / cutoff]
            if debug:
                print(f'DEBUG: multilooking sigmas ({sigmas[0]:.2f}, {sigmas[1]:.2f}), coarsen {coarsen}')

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
            conv = self.nanconvolve2d_gaussian(slice_data, weight, sigmas, threshold=gaussian_threshold)
            return xr.DataArray(conv, dims=slice_data.dims, name=slice_data.name)

        # process stack of dataarray slices
        def process_slice_var(dataarray):    
            if stackvar:
                stack = [process_slice(dataarray[ind]) for ind in range(len(dataarray[stackvar]))]
                return xr.concat(stack, dim=stackvar).assign_coords(dataarray.coords)
            else:
                return process_slice(dataarray).assign_coords(dataarray.coords)

        if isinstance(data, xr.Dataset):
            ds = xr.Dataset({varname: process_slice_var(data[varname]) for varname in data.data_vars})
        else:
            ds = process_slice_var(data)
    
        # Set chunk size
        chunksizes = {'y': self.chunksize, 'x': self.chunksize}

        if coarsen:
            # calculate coordinate offsets to align coarsened grids
            y0 = self.coarsen_start(ds, 'y', coarsen[0])
            x0 = self.coarsen_start(ds, 'x', coarsen[1])
            ds = ds.isel({'y': slice(y0, None), 'x': slice(x0, None)})\
                     .coarsen({'y': coarsen[0], 'x': coarsen[1]}, boundary='trim')\
                     .mean()

        return ds.chunk(chunksizes)
