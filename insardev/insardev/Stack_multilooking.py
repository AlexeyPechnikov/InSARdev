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

    #decimator = lambda da: da.coarsen({'y': 2, 'x': 2}, boundary='trim').mean()
    def get_decimator(self, grid, coarsen=None, resolution=60, func='mean', wrap=False, debug=False):
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

        # decimator function
        def decimator(datas):
            def decimator_dataset(ds):
                coarsen_args = {'y': yscale, 'x': xscale}
                # calculate coordinate offsets to align coarsened grids
                y0 = self._coarsen_start(ds, 'y', yscale)
                x0 = self._coarsen_start(ds, 'x', xscale)
                ds = ds.isel({'y': slice(y0, None), 'x': slice(x0, None)})
                if wrap:
                    da_complex = np.exp(1j * ds)
                    da_complex_agg = getattr(da_complex\
                           .coarsen(coarsen_args, boundary='trim'), func)()\
                           .chunk({'y': self.chunksize, 'x': self.chunksize})
                    da_decimated = np.arctan2(da_complex_agg.imag, da_complex_agg.real)
                    del da_complex, da_complex_agg
                    return da_decimated
                else:
                    return getattr(ds\
                           .coarsen(coarsen_args, boundary='trim'), func)()\
                           .chunk({'y': self.chunksize, 'x': self.chunksize})
            # avoid creating the large chunks
            #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            if isinstance(datas, dict):
                return {k:decimator_dataset(v) for k,v in datas.items()}
            else:
                return decimator_dataset(datas)

        # return callback function and set common chunk size
        return lambda datas: decimator(datas)

    def get_decimator_interferogram(self, grid, coarsen=None, resolution=60, func='mean', debug=False):
        return self.get_decimator(grid, coarsen, resolution, func, wrap=True, debug=debug)

    def get_decimator_correlation(self, grid, coarsen=None, resolution=60, func='mean', debug=False):
        return self.get_decimator(grid, coarsen, resolution, func, wrap=False, debug=debug)

    def _multilooking(self, data, weight=None, wavelength=None, coarsen=None, gaussian_threshold=0.5, debug=False):
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
    
        # TODO: create function get_sigma(data, wavelength=None, coarsen=None)
        # calculate sigmas based on wavelength or coarsen
        if wavelength is not None:
            dy, dx = self.get_spacing(data)
            sigmas = [wavelength / dy / cutoff, wavelength / dx / cutoff]
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
            conv = self._nanconvolve2d_gaussian(slice_data, weight, sigmas, threshold=gaussian_threshold)
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
            y0 = self._coarsen_start(ds, 'y', coarsen[0])
            x0 = self._coarsen_start(ds, 'x', coarsen[1])
            ds = ds.isel({'y': slice(y0, None), 'x': slice(x0, None)})\
                     .coarsen({'y': coarsen[0], 'x': coarsen[1]}, boundary='trim')\
                     .mean()

        return ds.chunk(chunksizes)
