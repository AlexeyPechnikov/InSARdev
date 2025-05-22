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
from .Stack_plot import Stack_plot
from .Batch import Batch, BatchWrap, BatchComplex
from collections.abc import Mapping
from . import utils_io
from . import utils_xarray
from insardev_toolkit import datagrid

class Stack(Stack_plot, Mapping):
    import rasterio as rio
    import pandas as pd
    import xarray as xr
    #import geopandas as gpd
    #import zarr

    def __init__(self, dss:dict[str, xr.Dataset] | None = None):
        self.dss = BatchComplex(dss)

    def __repr__(self):
        if not getattr(self, 'dss', {}):
            return f"{self.__class__.__name__}(empty)"
        n = len(self)
        if n <= 1:
            # delegate to the underlying dict repr
            return dict.__repr__(self.dss)
        sample = next(iter(self.dss.values()))
        keys = list(self.dss.keys())
        return f'{self.__class__.__name__} object containing {len(self.dss)} items for {len(sample.date)} date ({keys[0]} ... {keys[-1]})' 

    def __len__(self):
        return len(getattr(self, 'dss', {}))

    def __getitem__(self, key):
        # so stack[key] → self.dss[key]
        return self.dss[key]

    def __iter__(self):
        # iteration yields the same keys as the Batch
        return iter(self.dss)

    def __bool__(self):
        return bool(self.__len__())

    def __add__(self, other):
        """
        Add two stacks together.

        s3 = s1 + s2
        """
        import copy
        if not isinstance(other, Stack):
            return NotImplemented
        # make a shallow copy of self
        new = copy.copy(self)
        # merge the dicts
        new.dss = self.dss | other.dss
        return new

    def __iadd__(self, other):
        """
        Add two stacks together in place.

        s1 += s2
        """
        if not isinstance(other, Stack):
            return NotImplemented
        # update in‐place
        self.dss.update(other.dss)
        return self

    def sel(self, *args, **kwargs) -> 'Stack':
        # call through to the Batch.sel
        dss = self.dss.sel(*args, **kwargs)
        # build a fresh Stack without re-running __init__
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.dss = dss
        return new

    def isel(self, *args, **kwargs) -> 'Stack':
        dss = self.dss.isel(*args, **kwargs)
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.dss = dss
        return new
        
    def PRM(self, key:str) -> str|float|int:
        """
        Use as stack.PRM('radar_wavelength') to get the radar wavelength from the first burst.
        """
        return next(iter(self.dss.values())).attrs[key]

    def crs(self) -> rio.crs.CRS:
        return next(iter(self.dss.values())).rio.crs

    def epsg(self) -> int:
        return next(iter(self.dss.values())).rio.crs.to_epsg()

    def snapshot(self, *args, store: str | None = None, storage_options: dict[str, str] | None = None, chunksize: int|str = 'auto',
                caption: str = 'Snapshotting...', n_jobs: int = -1, debug=False):
        if len(args) > 2:
            raise ValueError(f'ERROR: snapshot() accepts only one or two Batch/BatchWrap/dict objects or no arguments.')
        datas = utils_io.snapshot(*args, store=store, storage_options=storage_options, compat=True, chunksize=chunksize, caption=caption, n_jobs=n_jobs, debug=debug)
        return datas

    def downsample(self, *args, coarsen=None, resolution=60, func='mean', debug:bool=False):
        datas = []
        for arg in args:
            print ('type(arg)', type(arg))
            if isinstance(arg, (Stack, BatchComplex)):
                arg = Batch(arg)
                print (arg.isel(0)['033_069722_IW3'].data_vars)
            wrap = True if isinstance(arg, BatchWrap) else False
            print ('\ttype(arg)', type(arg), 'wrap', wrap)
            sample = next(iter(arg.values()))
            callback = utils_xarray.downsampler(sample, coarsen=coarsen, resolution=resolution, func=func, wrap=wrap, debug=debug)
            data = callback(arg)
            datas.append(BatchWrap(data) if wrap else Batch(data))
        return datas

    def to_dataframe(self,
                     datas: dict[str, xr.Dataset | xr.DataArray] | None = None,
                     crs:str|None='auto',
                     attr_start:str='BPR',
                     debug:bool=False
                     ) -> pd.DataFrame:
        """
        Return a Pandas DataFrame for all Stack scenes.

        Returns
        -------
        pandas.DataFrame
            The DataFrame containing Stack scenes.

        Examples
        --------
        df = stack.to_dataframe()
        """
        import geopandas as gpd
        from shapely import wkt
        import pandas as pd
        import numpy as np

        if datas is not None and not isinstance(datas, dict):
            raise ValueError(f'ERROR: datas is not None or a dict: {type(datas)}')
    
        if crs is not None and isinstance(crs, str) and crs == 'auto':
            crs = self.crs()

        if datas is None:
            datas = self.dss

        polarizations = [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in next(iter(datas.values())).data_vars]
        #print ('polarizations', polarizations)

        # make attributes dataframe from datas
        processed_attrs = []
        for ds in datas.values():
            #print (data.id)
            attrs = [data_var for data_var in ds if ds[data_var].dims==('date',)][::-1]
            attr_start_idx = attrs.index(attr_start)
            for date_idx, date in enumerate(ds.date.values):
                processed_attr = {}
                for attr in attrs[:attr_start_idx+1]:
                    #NotImplementedError: 'item' is not yet a valid method on dask arrays
                    value = ds[attr].item(date_idx)
                    #value = ds[attr].values[date_idx]
                    #print (attr, date_idx, date, value)
                    #processed_attr['date'] = date
                    if hasattr(value, 'item'):
                        processed_attr[attr] = value.item()
                    elif attr == 'geometry':
                        processed_attr[attr] = wkt.loads(value)
                    else:
                        processed_attr[attr] = value
                processed_attrs.append(processed_attr)
                #print (processed_attr)
        df = gpd.GeoDataFrame(processed_attrs, crs=4326)
        #del df['date']
        #df['polarization'] = ','.join(polarizations)
        # convert polarizations to strings like "VV,VH" to pevent confusing with tuples in the dataframe
        df = df.assign(polarization=','.join(map(str, polarizations)))
        # reorder columns to the same order as preprocessor uses
        pol = df.pop("polarization")
        df.insert(3, "polarization", pol)
        # round for human readability
        df['BPR'] = df['BPR'].round(1)

        group_col = df.columns[0]
        burst_col = df.columns[1]
        #print ('df.columns[0]', df.columns[0])
        #print ('df.columns[:2][::-1].tolist()', df.columns[:2][::-1].tolist())
        df['startTime'] = pd.to_datetime(df['startTime'])
        #df['date'] = df['startTime'].dt.date.astype(str)
        df = df.sort_values(by=[group_col, burst_col]).set_index([group_col, burst_col])
        # move geometry to the end of the dataframe to be the most similar to insar_pygmtsar output
        df = df.loc[:, df.columns.drop("geometry").tolist() + ["geometry"]]
        
        return df.to_crs(crs) if crs is not None else df

    def load(self, urls:str | list | dict[str, str], storage_options:dict[str, str]|None=None, chunksize: int|str = 'auto', attr_start:str='BPR', debug:bool=False):
        import numpy as np
        import xarray as xr
        import pandas as pd
        import geopandas as gpd
        import zarr
        from shapely import wkt
        import os
        from insardev_toolkit import progressbar_joblib
        from tqdm.auto import tqdm
        import joblib
        import warnings
        # suppress the "Sending large graph of size …"
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            module=r'distributed\.client',
            message=r'Sending large graph of size .*'
        )
        from distributed import get_client, WorkerPlugin
        class IgnoreDaskDivide(WorkerPlugin):
            def setup(self, worker):
                # suppress the "RuntimeWarning: invalid value encountered in divide"
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=r'dask\._task_spec'
                )
        client = get_client()
        client.register_plugin(IgnoreDaskDivide(), name='ignore_divide')

        def burst_preprocess(ds, attr_start:str='BPR', debug:bool=False):
            import xarray as xr
            import numpy as np
            #print ('ds_preprocess', ds)
            process_attr = True if debug else False
            for key in ds.attrs:
                if key==attr_start:
                    process_attr = True
                if not process_attr and not key in ['SLC_scale']:
                    continue
                #print ('key', key)
                if key not in ['Conventions', 'spatial_ref']:
                    # Create a new DataArray with the original value
                    ds[key] = xr.DataArray(ds.attrs[key], dims=[])
                    # remove the attribute
                    del ds.attrs[key]
            
            # remove attributes for repeat bursts to unify the attributes
            BPR = ds['BPR'].values.item(0)
            if BPR != 0:
                ds.attrs = {}

            ds['data'] = (ds.re + 1j*ds.im).astype(np.complex64)
            if not debug:
                del ds['re'], ds['im']
            date = pd.to_datetime(ds['startTime'].item())
            return ds.expand_dims({'date': np.array([date.date()], dtype='U10')})

        def _bursts_transform_preprocess(bursts, transform, chunksize):
            import xarray as xr
            import numpy as np

            # in case of multiple polarizations, merge them into a single dataset
            polarizations = np.unique(bursts.polarization)
            if len(polarizations) > 1:
                datas = []
                for polarization in polarizations:
                    data = bursts.isel(date=bursts.polarization == polarization)\
                                .rename({'data': polarization})
                    # cannot combine in a single value VV and VH polarizations and corresponding burst names
                    data.burst.values = [
                        v.replace(polarization, 'XX') for v in data.burst.values
                    ]
                    del data['polarization']
                    datas.append(data.chunk(chunksize))
                ds = xr.merge(datas)
                del datas
            else:
                ds = ds.rename({'data': polarizations[0]})

            for var in transform.data_vars:
                #if var not in ['re', 'im']:
                ds[var] = transform[var].chunk(chunksize)

            ds.rio.write_crs(bursts.attrs['spatial_ref'], inplace=True)
            return ds

        def bursts_transform_preprocess(dss, transform):
            """
            Combine bursts and transform into a single dataset.
            Only reference burst for every polarization has attributes (see burst_preprocess)
            """
            import xarray as xr
            import numpy as np

            polarizations = np.unique([ds.polarization for ds in dss])
            #print ('polarizations', polarizations)

            # convert generic 'data' variable for all polarizations to VV, VH,... variables
            datas = []
            for polarization in polarizations:
                data = [ds for ds in dss if ds.polarization==polarization]
                data = xr.concat(data, dim='date', combine_attrs='no_conflicts').rename({'data': polarization})
                # cannot combine in a single value VV and VH polarizations and corresponding burst names
                data.burst.values = [v.replace(polarization, 'XX') for v in data.burst.values]
                del data['polarization']
                datas.append(data)
                del data
            ds = xr.merge(datas)
            # only reference burst has spatial_ref attribute, concat bursts before getting spatial_ref
            spatial_ref = ds.attrs['spatial_ref']
            del datas

            # add transform variables
            for var in transform.data_vars:
                ds[var] = transform[var]

            # set the coordinate reference system
            ds.rio.write_crs(spatial_ref, inplace=True)
            return ds

        # if isinstance(urls, str):
        #     print ('NOTE: urls is a string, convert to dict with burst as key and list of URLs as value.')
        # elif isinstance(urls, dict):
        #     print ('NOTE: urls is a dict, using it as is.')
        #     groups = urls
        # elif isinstance(urls.index, pd.MultiIndex) and urls.index.nlevels == 2:
        #     print ('NOTE: Detected Pandas Dataframe with MultiIndex, using first level as fullBurstID and the first column as URLs.')
        #     #groups = {key: group.index.get_level_values(1).tolist() for key, group in urls.groupby(level=0)}
        #     groups = {key: group[urls.columns[0]].tolist() for key, group in urls.groupby(level=0)}
        # elif isinstance(urls, list):
        #     print ('NOTE: urls is a list, convert to dict with burst as key and list of URLs as value.')
        #     groups = {}
        #     for url in urls:
        #         parent = url.rsplit('/', 2)[1]
        #         groups.setdefault(parent, []).append(url)
        # else:
        #     raise ValueError(f'ERROR: urls is not a dict, list, or Pandas Dataframe: {type(urls)}')

        # def store_open_burst(grp):
        #     #ds = xr.open_zarr(root.store, group=f'021_043788_IW1/{burst}', consolidated=True, zarr_format=3)
        #     #grp = root['021_043788_IW1'][burst]
        #     ds = xr.open_zarr(grp.store, group=grp.path, consolidated=True, zarr_format=3)
        #     return burst_preprocess(ds)
        
        def store_open_group(root, group):
            # open group (fullBurstID)
            grp = root[group]
            # get all subgroups (bursts) except transform
            grp_bursts = [grp[k] for k in grp.keys() if k!='transform']
            dss = [xr.open_zarr(grp.store, group=grp.path, consolidated=True, zarr_format=3) for grp in grp_bursts]
            dss = [burst_preprocess(ds) for ds in dss]
            # get transform subgroup
            grp_transform = grp['transform']
            transform = xr.open_zarr(grp_transform.store, group=grp_transform.path, consolidated=True, zarr_format=3)
            # combine bursts and transform
            ds = bursts_transform_preprocess(dss, transform)
            del dss, transform
            return group, ds

        if isinstance(urls, str):
            # note: isinstance(urls, zarr.storage.ZipStore) can be loaded too but it is less efficient
            urls = os.path.expanduser(urls)
            root = zarr.open_consolidated(urls, zarr_format=3, mode='r')
            with progressbar_joblib.progressbar_joblib(tqdm(desc='Loading Dataset...'.ljust(25), total=len(list(root.group_keys())))) as progress_bar:
                dss = joblib.Parallel(n_jobs=-1, backend='loky')\
                    (joblib.delayed(store_open_group)(root, group) for group in list(root.group_keys()))
            # list of key - dataset converted to dict and appended to the existing dict
            self.dss.update(dss)
        # elif isinstance(urls, FsspecStore):
        #     root = zarr.open_consolidated(urls, zarr_format=3, mode='r')
        #     dss = []
        #     for group in tqdm(list(root.group_keys()), desc='Loading Store'):
        #         dss.append(store_open_group(root, group))
        #     self.dss = dict(dss)
        #     del dss
        elif isinstance(urls, list) or isinstance(urls, pd.DataFrame):
            # load bursts and transform specified by URLs
            # this allows to load from multiple locations with precise control of the data
            if isinstance(urls, list):
                print ('NOTE: urls is a list, using it as is.')
                df = pd.DataFrame(urls, columns=['url'])
                df['fullBurstID'] = df['url'].str.rsplit('/', n=2).str[1]
                df['burst'] = df["url"].str.rsplit("/", n=2).str[2]
                urls = df.sort_values(by=['fullBurstID', 'burst']).set_index(['fullBurstID', 'burst'])
                print (urls.head())
            elif isinstance(urls.index, pd.MultiIndex) and urls.index.nlevels == 2 and len(urls.columns) == 1:
                print ('NOTE: Detected Pandas Dataframe with MultiIndex, using first level as fullBurstID and the first column as URLs.')
                #groups = {key: group.index.get_level_values(1).tolist() for key, group in urls.groupby(level=0)}
                #groups = {key: group[urls.columns[0]].tolist() for key, group in urls.groupby(level=0)}
            else:
                raise ValueError(f'ERROR: urls is not a list, or Pandas Dataframe with multiindex: {type(urls)}')

            dss = {}
            for fullBurstID in tqdm(urls.index.get_level_values(0).unique(), desc='Loading Datasets...'.ljust(25)):
                #print ('fullBurstID', fullBurstID)
                df = urls[urls.index.get_level_values(0) == fullBurstID]
                bases = df[df.index.get_level_values(1) != 'transform'].iloc[:,0].values
                #print ('fullBurstID', fullBurstID, '=>', bases)
                base = df[df.index.get_level_values(1) == 'transform'].iloc[:,0].values[0]
                #print ('fullBurstID', fullBurstID, '=>', base)
                bursts = xr.open_mfdataset(
                    bases,
                    engine='zarr',
                    zarr_format=3,
                    consolidated=True,
                    parallel=True,
                    concat_dim='date',
                    combine='nested',
                    preprocess=lambda ds: burst_preprocess(ds, attr_start=attr_start, debug=False),
                    storage_options=storage_options,
                )
                # some variables are stored as int32 with scale factor, convert to float32 instead of default float64
                transform = xr.open_dataset(base, engine='zarr', zarr_format=3, consolidated=True, storage_options=storage_options).astype('float32')

                ds = _bursts_transform_preprocess(bursts, transform, chunksize)
                dss[fullBurstID] = ds
                del ds, bursts, transform

            #assert len(np.unique([ds.rio.crs.to_epsg() for ds in dss])) == 1, 'All datasets must have the same coordinate reference system'
            self.dss.update(dss)

    def to_dataset(self,
              datas: xr.Dataset | xr.DataArray | dict[str, xr.Dataset | xr.DataArray] | None = None,
              wrap: bool | None = None,
              compute: bool = False):
        """
        This function is a faster implementation for the standalone function combination of xr.concat and xr.align:
        xr.concat(xr.align(*intfs, join='outer'), dim='stack_dim').ffill('stack_dim').isel(stack_dim=-1).compute()
        #xr.concat(xr.align(*datas, join='outer'), dim='stack_dim').mean('stack_dim').compute()

        Parameters
        ----------
        datas: xr.Dataset | xr.DataArray | dict[str, xr.Dataset | xr.DataArray] | None
            The datasets to concatenate.
        wrap: bool | None
            There are three options:
            - None: return the topmost burst in chronological order for overlapping areas
            - True: compute the circular mean
            - False: compute the arithmetic mean
        compute: bool
            Whether to compute the result.
        """
        import xarray as xr
        import numpy as np
        import dask
        from insardev_toolkit import progressbar

        # in case of dict wrap is not defined, also it should be preserved for recursive calls
        dtype = type(datas)
        if dtype == BatchWrap:
            wrap = True
        elif dtype == Batch:
            wrap = False
        print ('dtype', dtype, 'wrap', wrap)

        #print (type(datas))
        if datas is None:
            datas = self.dss
        elif isinstance(datas, xr.Dataset):
            return datas
        elif isinstance(datas, xr.DataArray):
            return datas.to_dataset()
        elif not isinstance(datas, (dict, list, tuple)):
            raise ValueError(f'ERROR: datas is not a dict, list, or tuple: {type(datas)}')

        # all the grids will be unified to a single grid, we don't need the dict keys
        if isinstance(datas, dict):
            datas = list(datas.values())
        
        if len(datas) == 0:
            return None

        if len(datas) == 1:
            datas = datas[0]
            if compute:
                progressbar(result := datas.persist(), desc=f'Compute Dataset'.ljust(25))
                return result
            return datas

        # find all variables in the first dataset related to polarizations
        data_vars = datas[0].data_vars if isinstance(datas[0], xr.Dataset) else datas[0].name
        polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in data_vars]

        # process list of datasets with one or multiple polarizations
        if isinstance(datas[0], xr.Dataset):
            das_total = []
            for pol in polarizations:
                # TODO: workaround to preserve crs for variables
                #das = self.to_dataset([ds[pol].rio.set_crs(datas[0].rio.crs) for ds in datas])
                das = self.to_dataset([ds[pol] for ds in datas], wrap=wrap)
                das_total.append(das)
                del das
            das_total = xr.merge(das_total)
            
            if compute:
                progressbar(result := das_total.persist(), desc=f'Compute Unified Dataset'.ljust(25))
                del das_total
                return result
            return das_total

        # process list of dataarrays with single polarization
        y_chunksize = datas[0].chunks[-2][0]
        x_chunksize = datas[0].chunks[-1][0]
        # define unified grid
        y_min = min(ds.y.min().item() for ds in datas)
        y_max = max(ds.y.max().item() for ds in datas)
        x_min = min(ds.x.min().item() for ds in datas)
        x_max = max(ds.x.max().item() for ds in datas)
        #print (y_min, y_max, x_min, x_max, y_max-y_min, x_max-x_min)
        stackvar = list(datas[0].dims)[0]
        # workaround for dask.array.blockwise
        stackval = datas[0][stackvar].astype(str)
        stackidx = xr.DataArray(np.arange(len(stackval), dtype=int), dims=('z',))
        dy = datas[0].y.diff('y').item(0)
        dx = datas[0].x.diff('x').item(0)
        #print ('dy, dx', dy, dx)
        ys = xr.DataArray(np.arange(y_min, y_max + dy/2, dy), dims=['y'])
        xs = xr.DataArray(np.arange(x_min, x_max + dx/2, dx), dims=['x'])
        #print ('stack', stackvar, stackval)
        #print ('ys', ys)
        #print ('xs', xs)
        # extract extents of all datasets once
        extents = [(float(da.y.min()), float(da.y.max()), float(da.x.min()), float(da.x.max())) for da in datas]
        
        # use outer variable datas
        def block_dask(stack, y_chunk, x_chunk, wrap):
            #print ('pair', pair)
            #print ('concat: block_dask', stackvar, stack)
            # extract extent of the current chunk once
            ymin0, ymax0 = float(y_chunk.min()), float(y_chunk.max())
            xmin0, xmax0 = float(x_chunk.min()), float(x_chunk.max())
            # select all datasets overlapping with the current chunk
            das_slice = [da.isel({stackvar: stackidx}).sel({'y': slice(ymin0, ymax0), 'x': slice(xmin0, xmax0)}).compute(num_workers=1)
                         for da, (ymin, ymax, xmin, xmax) in zip(datas, extents)
                         if ymin0 < ymax and ymax0 > ymin and xmin0 < xmax and xmax0 > xmin]
            #print ('concat: das_slice', len(das_slice), [da.shape for da in das_slice])
            
            fill_dtype = datas[0].dtype
            fill_nan = np.nan * np.ones((), dtype=fill_dtype)
            if len(das_slice) == 0:
                # return empty block
                return np.full((stack.size, y_chunk.size, x_chunk.size), fill_nan, dtype=fill_dtype)
            #das_block = [da.reindex({'y': y_chunk, 'x': x_chunk}, fill_value=fill_nan, copy=False) for da in das_slice if da.size > 0]
            das_block = [da.reindex({'y': y_chunk, 'x': x_chunk}, fill_value=fill_nan, copy=False) for da in das_slice]
            del das_slice
            if len(das_block) == 1:
                # return single block as is
                return das_block[0].values

            if wrap is None:
                #print ('wrap None')
                # ffill does not work correct on complex data and per-component ffill is faster
                # the magic trick is to use sorting to ensure burst overpapping order
                # bursts ends should be overlapped by bursts starts
                das_block_concat = xr.concat(das_block, dim="stack_dim", join="inner")
                if np.issubdtype(das_block_concat.dtype, np.complexfloating):
                    return (das_block_concat.real.ffill("stack_dim").isel(stack_dim=-1)
                            + 1j*das_block_concat.imag.ffill("stack_dim").isel(stack_dim=-1)).values
                else:
                    return das_block_concat.ffill("stack_dim").isel(stack_dim=-1).values
            elif wrap == True:
                #print ('wrap True')
                # calculate circular mean for interferogram data
                das_block_concat = xr.concat([np.exp(1j * da) for da in das_block], dim='stack_dim')
                block_complex = das_block_concat.mean('stack_dim', skipna=True).values
                return np.arctan2(block_complex.imag, block_complex.real)
            elif wrap == False:
                #print ('wrap False')
                das_block_concat = xr.concat(das_block, dim="stack_dim", join="inner")
                # calculate arithmetic mean for phase and correlation data
                return das_block_concat.mean('stack_dim', skipna=True).values
            else:
                raise ValueError(f'ERROR: wrap is not a boolean or None: {wrap}')

        # prevent warnings 'PerformanceWarning: Increasing number of chunks by factor of ...'
        import warnings
        from dask.array.core import PerformanceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerformanceWarning)
            # rechunk data for expected usage
            data = dask.array.blockwise(
                block_dask,
                'zyx',
                stackidx.chunk(1), 'z',
                ys.chunk({'y': y_chunksize}), 'y',
                xs.chunk({'x': x_chunksize}), 'x',
                meta = np.empty((0, 0, 0), dtype=datas[0].dtype),
                wrap=wrap
            )
        da = xr.DataArray(data, coords={stackvar: stackval, 'y': ys, 'x': xs})\
            .rename(datas[0].name)\
            .assign_attrs(datas[0].attrs)
        del data
        return datagrid.spatial_ref(da, datas)
