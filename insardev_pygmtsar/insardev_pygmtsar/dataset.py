# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from insardev_toolkit import tqdm_dask
from insardev_toolkit import datagrid

class dataset(datagrid):

    # work directory
    basedir = '.'

    def get_filename(self, name, basedir='auto'):
        """
        Generate a NetCDF filename by appending .nc extension.

        Parameters
        ----------
        name : str
            Base name for the file without extension.
        basedir : str, optional
            Base directory for the file. If 'auto', uses default directory. Default is 'auto'.

        Returns
        -------
        str
            Full path to the NetCDF file.
        """
        import os

        if isinstance(basedir, str) and basedir == 'auto':
            basedir = self.basedir

        filename = os.path.join(basedir, f'{name}.nc')
        return filename

    def open_cube(self, name, basedir='auto'):
        """
        Opens an xarray 2D/3D Dataset or DataArray from a NetCDF file.

        This function takes the name of the model to be opened, reads the NetCDF file, and re-chunks
        the dataset according to the provided chunksize or the default value from the 'stack' object.
        The 'date' dimension is always chunked with a size of 1.

        Parameters
        ----------
        name : str
            The name of the model file to be opened.
        basedir : str, optional
            Base directory path. If 'auto', uses default directory. Default is 'auto'.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Data read from the specified NetCDF file. Returns a Dataset unless the original data
            was a DataArray with a name stored in attributes.

        Raises
        ------
        AssertionError
            If the specified NetCDF file does not exist.
        """
        import xarray as xr
        import pandas as pd
        import numpy as np
        import os

        filename = self.get_filename(name, basedir=basedir)
        assert os.path.exists(filename), f'ERROR: The NetCDF file is missed: {filename}'

        # Workaround: open the dataset without chunking
        data = xr.open_dataset(filename,
                               engine=self.netcdf_engine_read,
                               format=self.netcdf_format)
        
        if 'stack' in data.dims:
            if 'y' in data.coords and 'x' in data.coords:
                multi_index_names = ['y', 'x']
            elif 'lat' in data.coords and 'lon' in data.coords:
                multi_index_names = ['lat', 'lon']
            multi_index = pd.MultiIndex.from_arrays([data.y.values, data.x.values], names=multi_index_names)
            data = data.assign_coords(stack=multi_index).set_index({'stack': ['y', 'x']})
            chunksize = self.chunksize1d
        else:
            chunksize = self.chunksize

        # set the proper chunk sizes
        chunks = {dim: 1 if dim in ['pair', 'date'] else chunksize for dim in data.dims}
        data = data.chunk(chunks)

        # attributes are empty when dataarray is prezented as dataset
        # revert dataarray converted to dataset
        data_vars = list(data.data_vars)
        if len(data_vars) == 1 and 'dataarray' in data.attrs:
            assert data.attrs['dataarray'] == data_vars[0]
            data = data[data_vars[0]]

        # convert string dates to dates
        for dim in ['date', 'ref', 'rep']:
            if dim in data.dims:
                data[dim] = pd.to_datetime(data[dim])

        return data

    def save_cube(self, data, name=None, spatial_ref=None, caption='Saving NetCDF 2D/3D Dataset', basedir='auto'):
        """
        Save a lazy or non-lazy 2D/3D xarray Dataset or DataArray to a NetCDF file.

        The 'date' or 'pair' dimension is always chunked with a size of 1.

        Parameters
        ----------
        data : xarray.Dataset or xarray.DataArray
            The data to be saved. Can be either lazy (dask array) or non-lazy (numpy array).
        name : str, optional
            The name for the output NetCDF file. If None and data is a DataArray,
            will use data.name. Required if data is a Dataset.
        spatial_ref : str, optional
            The spatial reference system. Default is None.
        caption : str, optional
            The text caption for the saving progress bar. Default is 'Saving NetCDF 2D/3D Dataset'.
        basedir : str, optional
            Base directory for saving the file. If 'auto', uses the default directory. Default is 'auto'.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If name is None and data is not a named DataArray.
        AssertionError
            If name is None and data is a DataArray without a name.

        Examples
        --------
        # Save lazy 3D dataset/dataarray
        stack.save_cube(intf90m, 'intf90m')                              
        stack.save_cube(intf90m.phase, 'intf90m')                        

        # Save lazy 2D dataset/dataarray
        stack.save_cube(intf90m.isel(pair=0), 'intf90m')                 
        stack.save_cube(intf90m.isel(pair=0).phase, 'intf90m')           

        # Save non-lazy (computed) 3D dataset/dataarray
        stack.save_cube(intf90m.compute(), 'intf90m')                    
        stack.save_cube(intf90m.phase.compute(), 'intf90m')              

        # Save non-lazy (computed) 2D dataset/dataarray
        stack.save_cube(intf90m.isel(pair=0).compute(), 'intf90m')       
        stack.save_cube(intf90m.isel(pair=0).phase.compute(), 'intf90m') 
        """
        import xarray as xr
        import pandas as pd
        import dask
        import os
        import warnings
        # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
        warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', module='dask')
        warnings.filterwarnings('ignore', module='dask.core')
        import logging
        # prevent warnings "RuntimeWarning: All-NaN slice encountered"
        logging.getLogger('distributed.nanny').setLevel(logging.ERROR)
        # disable "distributed.utils_perf - WARNING - full garbage collections ..."
        try:
            from dask.distributed import utils_perf
            utils_perf.disable_gc_diagnosis()
        except ImportError:
            from distributed.gc import disable_gc_diagnosis
            disable_gc_diagnosis()

        if name is None and isinstance(data, xr.DataArray):
            assert data.name is not None, 'Define data name or use "name" argument for the NetCDF filename'
            name = data.name
        elif name is None:
            raise ValueError('Specify name for the output NetCDF file')

        chunksize = None
        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            # replace multiindex by sequential numbers 0,1,...
            data = data.reset_index('stack')
            # single-dimensional data compression required
            chunksize = self.netcdf_chunksize1d

        if isinstance(data, xr.DataArray):
            if data.name is None:
                data = data.rename(name)
            data = data.to_dataset().assign_attrs({'dataarray': data.name})

        is_dask = isinstance(data[list(data.data_vars)[0]].data, dask.array.Array)
        encoding = {varname: self._compression(data[varname].shape, chunksize=chunksize) for varname in data.data_vars}
        #print ('save_cube encoding', encoding)
        #print ('is_dask', is_dask, 'encoding', encoding)

        # save to NetCDF file
        filename = self.get_filename(name, basedir=basedir)
        if os.path.exists(filename):
            os.remove(filename)
        delayed = self.spatial_ref(data, spatial_ref).to_netcdf(filename,
                                 engine=self.netcdf_engine_write,
                                 format=self.netcdf_format,
                                 encoding=encoding,
                                 compute=not is_dask)
        if is_dask:
            if caption is None:
                dask.compute(delayed)
            else:
                tqdm_dask(result := dask.persist(delayed), desc=caption)
                del result
            # cleanup - sometimes writing NetCDF handlers are not closed immediately and block reading access
            del delayed
            import gc; gc.collect()

    def delete_cube(self, name, basedir='auto'):
        """
        Delete a NetCDF cube file.

        Parameters
        ----------
        name : str
            Name of the cube file to delete
        basedir : str, optional
            Base directory path. If 'auto', uses default directory. Default is 'auto'.
        """
        import os

        filename = self.get_filename(name, basedir=basedir)
        #print ('filename', filename)
        if os.path.exists(filename):
            os.remove(filename)
