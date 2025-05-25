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
from __future__ import annotations
from .Stack_plot import Stack_plot
from .BatchCore import BatchCore
from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex
from collections.abc import Mapping
from . import utils_io
from . import utils_xarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import rasterio as rio
    import pandas as pd
    import xarray as xr

class Stack(Stack_plot, Mapping):
    
    def __init__(self, dss:dict[str, xr.Dataset] | None = None):
        self.dss = BatchCore(dss)

    def __repr__(self):
        if not getattr(self, 'dss', {}):
            return f"{self.__class__.__name__}(empty)"
        n = len(self)
        if n <= 1:
            # delegate to the underlying dict repr
            #return dict.__repr__(self.dss)
            key, ds = next(iter(self.items()))
            return f"{self.__class__.__name__}['{key}']:\n{ds!r}"
        sample = next(iter(self.dss.values()))
        keys = list(self.dss.keys())
        return f'{self.__class__.__name__} containing {len(self.dss)} items for {len(sample.date)} date ({keys[0]} … {keys[-1]})' 

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

    def snapshot(self, *args, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str = 'Snapshotting...', n_jobs: int = -1, debug=False):
        if len(args) > 2:
            raise ValueError(f'ERROR: snapshot() accepts only one or two Batch or dict objects or no arguments.')
        datas = utils_io.snapshot(*args, store=store, storage_options=storage_options, compat=True, caption=caption, n_jobs=n_jobs, debug=debug)
        return datas

    # def downsample(self, *args, coarsen=None, resolution=60, func='mean', debug:bool=False):
    #     datas = []
    #     for arg in args:
    #         print ('type(arg)', type(arg))
    #         if isinstance(arg, (Stack, BatchComplex)):
    #             arg = Batch(arg)
    #             print (arg.isel(0)['033_069722_IW3'].data_vars)
    #         wrap = True if isinstance(arg, BatchWrap) else False
    #         print ('\ttype(arg)', type(arg), 'wrap', wrap)
    #         sample = next(iter(arg.values()))
    #         callback = utils_xarray.downsampler(sample, coarsen=coarsen, resolution=resolution, func=func, wrap=wrap, debug=debug)
    #         data = callback(arg)
    #         datas.append(BatchWrap(data) if wrap else Batch(data))
    #     return datas

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
