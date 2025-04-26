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
from insardev_toolkit import progressbar
from insardev_toolkit import datagrid

class dataset(datagrid):

    # work directory
    basedir = '.'

    def _glob_re(self, name, basedir='auto'):
        """
        Find files matching a regular expression pattern in a directory.

        Parameters
        ----------
        name : str
            Regular expression pattern to match filenames against.
        basedir : str, optional
            Base directory to search in. If 'auto', uses default directory. Default is 'auto'.

        Returns
        -------
        list
            Sorted list of full paths to matching files.
        """
        import os
        import re

        if isinstance(basedir, str) and basedir == 'auto':
            basedir = self.basedir

        filenames = filter(re.compile(name).match, os.listdir(basedir))
        return sorted([os.path.join(basedir, filename) for filename in filenames])

    def _get_filename(self, name, basedir='auto'):
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

    def _get_filenames(self, pairs, name, basedir='auto'):
        """
        Get the filenames of the data grids based on pairs and name parameters.

        Parameters
        ----------
        pairs : np.ndarray or pd.DataFrame or None
            An array or DataFrame of pairs. Can be:
            - 1D array of dates for single date files
            - 2D array of date pairs for interferogram files 
            - DataFrame with 'ref' and 'rep' columns for interferogram files
        name : str
            Base name for the grid files. Will be prefixed to dates/pairs in filenames.
            If empty or None, no prefix will be added.
        basedir : str, optional
            Base directory path. If 'auto', uses default directory.
            Default is 'auto'.

        Returns
        -------
        list of str
            List of full paths to NetCDF files. Filenames are constructed as:
            - For single dates: {basedir}/{name}_{date}.nc 
            - For pairs: {basedir}/{name}_{ref_date}_{rep_date}.nc
            Dates are formatted without hyphens.
        """
        import pandas as pd
        import numpy as np
        import os

        if isinstance(basedir, str) and basedir == 'auto':
            basedir = self.basedir

        if isinstance(pairs, pd.DataFrame):
            # convert to standalone DataFrame first
            pairs = self.get_pairs(pairs)[['ref', 'rep']].astype(str).values
        else:
            pairs = np.asarray(pairs)

        if name == '' or name is None:
            name = ''
        else:
            name = name + '_'
 
        filenames = []
        if len(pairs.shape) == 1:
            # read all the grids from files
            for date in sorted(pairs):
                filename = os.path.join(basedir, f'{name}{date}.nc'.replace('-',''))
                filenames.append(filename)
        elif len(pairs.shape) == 2:
            # read all the grids from files
            for pair in pairs:
                filename = os.path.join(basedir, f'{name}{pair[0]}_{pair[1]}.nc'.replace('-',''))
                filenames.append(filename)
        return filenames

    def cube_open(self, name, basedir='auto'):
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

        filename = self._get_filename(name, basedir=basedir)
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

    def cube_sync(self, data, name=None, spatial_ref=None, caption='Syncing NetCDF 2D/3D Dataset', basedir='auto'):
        """
        Save and reload a 2D/3D xarray Dataset or DataArray to/from a NetCDF file.

        This is a convenience method that combines save_cube() and open_cube() operations.

        Parameters
        ----------
        data : xarray.Dataset or xarray.DataArray
            The data to be saved and reloaded.
        name : str, optional
            The name for the output NetCDF file. If None and data is a DataArray,
            will use data.name. Default is None.
        spatial_ref : str, optional
            The spatial reference system. Default is None.
        caption : str, optional
            The text caption for the saving progress bar. Default is 'Syncing NetCDF 2D/3D Dataset'.
        basedir : str, optional
            Base directory for saving/loading the file. If 'auto', uses the default directory. Default is 'auto'.

        Returns
        -------
        xarray.Dataset
            The reloaded data from the saved NetCDF file.

        Raises
        ------
        ValueError
            If name is None and data is not a named DataArray.
        """
        import xarray as xr
        if name is None and isinstance(data, xr.DataArray):
            assert data.name is not None, 'Define data name or use "name" argument for the NetCDF filename'
            name = data.name
        elif name is None:
            raise ValueError('Specify name for the output NetCDF file')
        self.cube_save(data, name, spatial_ref, caption, basedir=basedir)
        return self.cube_open(name, basedir=basedir)

    def cube_save(self, data, name=None, spatial_ref=None, caption='Saving NetCDF 2D/3D Dataset', basedir='auto'):
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
        encoding = {varname: self.get_encoding_netcdf(data[varname].shape, chunksize=chunksize) for varname in data.data_vars}
        #print ('save_cube encoding', encoding)
        #print ('is_dask', is_dask, 'encoding', encoding)

        # save to NetCDF file
        filename = self._get_filename(name, basedir=basedir)
        if os.path.exists(filename):
            os.remove(filename)
        delayed = self.spatial_ref(data, spatial_ref).to_netcdf(filename,
                                 engine=self.netcdf_engine_write,
                                 format=self.netcdf_format,
                                 auto_complex=True,
                                 encoding=encoding,
                                 compute=not is_dask)
        if is_dask:
            progressbar(result := dask.persist(delayed), desc=caption)
            # cleanup - sometimes writing NetCDF handlers are not closed immediately and block reading access
            del delayed, result
            import gc; gc.collect()

    def cube_drop(self, name, basedir='auto'):
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

        filename = self._get_filename(name, basedir=basedir)
        #print ('filename', filename)
        if os.path.exists(filename):
            os.remove(filename)

    def stack_sync(self, data, name=None, spatial_ref=None, caption='Saving 2D Stack', basedir='auto', queue=None, timeout=300):
        """
        Synchronize stack data by deleting existing files and saving new data.

        Parameters
        ----------
        data : xarray.Dataset or xarray.DataArray
            Data to synchronize
        name : str, optional
            Name for the output files. If None, uses data.name. Default is None.
        spatial_ref : str, optional
            Spatial reference system. Default is None.
        caption : str, optional
            Progress bar caption. Default is 'Saving 2D Stack'.
        basedir : str, optional
            Base directory path. If 'auto', uses default directory. Default is 'auto'.
        queue : int, optional
            Number of files to process at once. Default is None.
        timeout : int, optional
            Timeout in seconds for worker restart. Default is 300.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Opened synchronized stack data
        """
        import xarray as xr
        if name is None and isinstance(data, xr.DataArray):
            assert data.name is not None, 'Define data name or use "name" argument for the NetCDF filenames'
            name = data.name
        elif name is None:
            raise ValueError('Specify name for the output NetCDF files')
        self.stack_drop(name, basedir=basedir)
        self.stack_save(data, name, spatial_ref, caption, basedir=basedir, queue=queue, timeout=timeout)
        return self.stack_open(name, basedir=basedir)

    def stack_open(self, name, stack=None, basedir='auto'):
        """
        Open a stack of NetCDF files.

        Parameters
        ----------
        name : str
            Base name of the stack files
        stack : list or array-like, optional
            Dates or pairs to open. If None, opens all files. Default is None.
        basedir : str, optional
            Base directory path. If 'auto', uses default directory. Default is 'auto'.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Opened stack data

        Examples
        --------
        stack.open_stack('data')
        stack.open_stack('data', ['2018-03-23'])
        stack.open_stack('data', ['2018-03-23', '2018-03-11'])
        stack.open_stack('phase15m')
        stack.open_stack('intf90m',[['2018-02-21','2018-03-11']])
        stack.open_stack('intf90m', stack.get_pairs([['2018-02-21','2018-03-11']]))
        """
        import xarray as xr
        import pandas as pd
        import numpy as np
        import glob
        import os
    
        if name == '' or name is None:
            name = ''
        else:
            name = name + '_'

        if stack is None:
            # look for all stack files
            #filenames = self._get_filenames(['*'], name)[0]
            #filenames = self._get_filename(f'{name}_????????_????????')
            # like data_20180323.nc or intf60m_20230114_20230219.nc
            filenames = self._glob_re(name + '[0-9]{8}(_[0-9]{8})*.nc', basedir=basedir)
        elif isinstance(stack, (list, tuple, np.ndarray)) and len(np.asarray(stack).shape) == 1:
            # dates
            filenames = self._get_filenames(np.asarray(stack), name, basedir=basedir)
        else:
            # pairs
            filenames = self._get_filenames(stack, name, basedir=basedir)
        #print ('filenames', filenames)

        data = xr.open_mfdataset(
            filenames,
            engine=self.netcdf_engine_read,
            format=self.netcdf_format,
            parallel=True,
            concat_dim='stackvar',
            chunks={"stackvar": 1},
            combine='nested'
        )
        
        if 'stack' in data.dims:
            if 'y' in data.coords and 'x' in data.coords:
                multi_index_names = ['y', 'x']
            elif 'lat' in data.coords and 'lon' in data.coords:
                multi_index_names = ['lat', 'lon']
            multi_index = pd.MultiIndex.from_arrays([data.y.values, data.x.values], names=multi_index_names)
            data = data.assign_coords(stack=multi_index).set_index({'stack': ['y', 'x']}).chunk({'stack': self.chunksize1d})
        else:
            dims = list(data.dims)
            data = data.chunk({dims[0]: 1, dims[1]: self.chunksize, dims[2]: self.chunksize})

        # revert dataarray converted to dataset
        data_vars = list(data.data_vars)
        if 'dataarray' in data.attrs:
            data = data[data.attrs['dataarray']]

        for dim in ['pair', 'date']:
            #if dim in (data.data_vars if isinstance(data, xr.Dataset) else data.coords):
            if dim in data.coords:
                if data[dim].shape == () or 'stack' in data.dims:
                    if data[dim].shape == ():
                        data = data.assign_coords(pair=('stackvar', [data[dim].values]))
                    data = data.rename({'stackvar': dim}).set_index({dim: dim})
                else:
                    data = data.swap_dims({'stackvar': dim})
    
        # convert string (or already timestamp) dates to dates
        for dim in ['date', 'ref', 'rep']:
            if dim in data.dims:
                if not data[dim].shape == ():
                    data[dim] = pd.to_datetime(data[dim])
                else:
                    data[dim].values = pd.to_datetime(data['date'].values)
    
        return data

    def stack_save(self, data, name, spatial_ref=None, caption='Saving 2D Stack', basedir='auto', queue=None, timeout=None):
        """
        Save stack data to multiple 2D NetCDF files.

        Parameters
        ----------
        data : xarray.Dataset or xarray.DataArray
            Data to save
        name : str
            Base name for the output files
        spatial_ref : str, optional
            Spatial reference system. Default is None.
        caption : str, optional
            Progress bar caption. Default is 'Saving 2D Stack'.
        basedir : str, optional
            Base directory path. If 'auto', uses default directory. Default is 'auto'.
        queue : int, optional
            Number of files to process at once. Default is None.
        timeout : int, optional
            Timeout in seconds for worker restart. Default is None.
        """
        import numpy as np
        import xarray as xr
        import pandas as pd
        import dask
        import os
        from dask.distributed import get_client
        import warnings
        # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
        warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', module='dask')
        warnings.filterwarnings('ignore', module='dask.core')
        # Filter out Dask "Restarting worker" warnings
        warnings.filterwarnings("ignore", module="distributed.nanny")
        import logging
        # Suppress Dask "Restarting worker" warnings
        logging.getLogger('distributed.nanny').setLevel(logging.ERROR)
        # disable "distributed.utils_perf - WARNING - full garbage collections ..."
        try:
            from dask.distributed import utils_perf
            utils_perf.disable_gc_diagnosis()
        except ImportError:
            from distributed.gc import disable_gc_diagnosis
            disable_gc_diagnosis()
    
        # Dask cluster client
        client = get_client()
        
        if isinstance(data, xr.Dataset):
            stackvar = data[list(data.data_vars)[0]].dims[0]
            is_dask = isinstance(data[list(data.data_vars)[0]].data, dask.array.Array)
        elif isinstance(data, xr.DataArray):
            stackvar = data.dims[0]
            is_dask = isinstance(data.data, dask.array.Array)
        else:
            raise Exception('Argument grid is not xr.Dataset or xr.DataArray object')
        #print ('is_dask', is_dask, 'stackvar', stackvar)
        stacksize = data[stackvar].size
    
        if queue is None:
            queue = self.netcdf_queue
        if queue is None:
            # process all the stack items in a single operation
            queue = stacksize
    
        if 'stack' in data.dims and isinstance(data.coords['stack'].to_index(), pd.MultiIndex):
            # replace multiindex by sequential numbers 0,1,...
            data = data.reset_index('stack')

        if isinstance(data, xr.DataArray):
            data = data.to_dataset().assign_attrs({'dataarray': data.name})
        encoding = {varname: self.get_encoding_netcdf(data[varname].shape[1:]) for varname in data.data_vars}
        #print ('save_stack encoding', encoding)
    
        # Applying iterative processing to prevent Dask scheduler deadlocks.
        counter = 0
        digits = len(str(stacksize))
        # Splitting all the pairs into chunks, each containing approximately queue pairs.
        n_chunks = stacksize // queue if stacksize > queue else 1
        for chunk in np.array_split(range(stacksize), n_chunks):
            dss = [data.isel({stackvar: ind}) for ind in chunk]
            if stackvar == 'date':
                stackvals = [ds[stackvar].dt.date.values for ds in dss]
            else:
                stackvals = [ds[stackvar].item().split(' ') for ds in dss]
            # save to NetCDF file
            filenames = self._get_filenames(stackvals, name, basedir=basedir)
            #[os.remove(filename) for filename in filenames if os.path.exists(filename)]
            delayeds = xr.save_mfdataset(self.spatial_ref(dss, spatial_ref),
                                         filenames,
                                         encoding=encoding,
                                         engine=self.netcdf_engine_write,
                                         format=self.netcdf_format,
                                         auto_complex=True,
                                         compute=not is_dask)
            # process lazy chunk
            if is_dask:
                if n_chunks > 1:
                    chunk_caption = f'{caption}: {(counter+1):0{digits}}...{(counter+len(chunk)):0{digits}} from {stacksize}'
                else:
                    chunk_caption = caption
                progressbar(result := dask.persist(delayeds), desc=chunk_caption)
                del delayeds, result
                # cleanup - sometimes writing NetCDF handlers are not closed immediately and block reading access
                import gc; gc.collect()
                # cleanup - release all workers memory, call garbage collector before to prevent heartbeat errors
                if timeout is not None:
                    client.restart(timeout=timeout, wait_for_workers=True)
#                 # more granular control
#                 n_workers = len(client.nthreads())
#                 client.restart(wait_for_workers=False)
#                 client.wait_for_workers(n_workers, timeout=timeout)
            # update chunks counter
            counter += len(chunk)

    def stack_drop(self, name, basedir='auto'):
        """
        Delete all 2D NetCDF files in a stack.

        Parameters
        ----------
        name : str
            Base name of the stack files to delete
        basedir : str, optional
            Base directory path. If 'auto', uses default directory. Default is 'auto'.
        """
        import os

        if name == '' or name is None:
            name = ''
        else:
            name = name + '_'

        filenames = self._glob_re(name + '[0-9]{8}(_[0-9]{8})*.nc', basedir=basedir)
        #print ('filenames', filenames)
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)
