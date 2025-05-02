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
from insardev_toolkit import progressbar

class Stack(Stack_plot):
    import rasterio as rio
    import pandas as pd
    import xarray as xr
    import geopandas as gpd
    from zarr.storage._fsspec import FsspecStore
    import zarr

    def __repr__(self):
        return f"Object {self.__class__.__name__} with {len(self.dss)} bursts for {len(next(iter(self.dss.values())).date)} dates"

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

    def PRM(self, key:str) -> str|float|int:
        """
        Use as stack.PRM('radar_wavelength') to get the radar wavelength from the first burst.
        """
        return next(iter(self.dss.values())).attrs[key]

    def crs(self) -> rio.crs.CRS:
        return next(iter(self.dss.values())).rio.crs

    def epsg(self) -> int:
        return next(iter(self.dss.values())).rio.crs.to_epsg()

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
                    value = ds[attr].item(date_idx)
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

    def to_dict(self,
                datas: dict[str, xr.Dataset | xr.DataArray] | list[xr.Dataset | xr.DataArray] | pd.DataFrame | None = None):
        """
        Return the full dictionary of datasets or convert speciied list of datasets or dataarrays to a dictionary.
        """
        import pandas as pd
        import xarray as xr
        import numpy as np

        if datas is None:
            return self.dss
        elif isinstance(datas, pd.DataFrame):
            dss = {}
            # iterate all burst groups
            for id in datas.index.get_level_values(0).unique():
                # select all records for the current burst group
                records = datas[datas.index.get_level_values(0)==id]
                # filter dates
                dates = records.startTime.dt.date.values.astype(str)
                ds = self.dss[id].sel(date=dates)
                # filter polarizations
                pols = records.polarization.unique()
                if len(pols) > 1:
                    raise ValueError(f'ERROR: Inconsistent polarizations found for the same burst: {id}')
                elif len(pols) == 0:
                    raise ValueError(f'ERROR: No polarizations found for the burst: {id}')
                pols = pols[0]
                if ',' in pols:
                    pols = pols.split(',')
                if isinstance(pols, str):
                    pols = [pols]
                count = 0
                if np.unique(pols).size < len(pols):
                    raise ValueError(f'ERROR: defined polarizations {pols} are not unique.')
                if len([pol for pol in pols if pol in ds.data_vars]) < len(pols):
                    raise ValueError(f'ERROR: defined polarizations {pols} are not available in the dataset: {id}')
                for pol in [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in ds.data_vars]:
                    if pol not in pols:
                        ds = ds.drop(pol)
                    else:
                        count += 1
                if count == 0:
                    raise ValueError(f'ERROR: No valid polarizations found for the burst: {id}')
                dss[id] = ds
            return dss
        elif isinstance(datas, dict):
            return datas
        elif isinstance(datas, (list, tuple)):
            if 'fullBurstID' in datas[0].data_vars:
                return {ds.fullBurstID.item(0):ds for ds in datas}
            else:
                import random, string
                chars = string.ascii_lowercase
                random_prefix = ''.join(random.choices(chars, k=8))
                print (f'NOTE: No fullBurstID variable found in the dataset, using random prefix {random_prefix} with order as the key.')
                return {f'{random_prefix}_{i+1}':ds for i, ds in enumerate(datas)}
        else:
            raise ValueError(f'ERROR: datas is not None or dataframe or a dict, list, or tuple of xr.Dataset or xr.DataArray: {type(datas)}')

    # def to_dataset(self, records=None):
    #     dss = self.to_datasets(records)
    #     if len(dss) > 1:
    #         return self.to_datasets(records)[0]

    def __init__(self, urls:str | list | dict[str, str], storage_options:dict[str, str]|None=None, attr_start:str='BPR', debug:bool=False):
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

        def _bursts_transform_preprocess(bursts, transform):
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
                    datas.append(data)
                ds = xr.merge(datas)
                del datas
            else:
                ds = ds.rename({'data': polarizations[0]})

            for var in transform.data_vars:
                #if var not in ['re', 'im']:
                ds[var] = transform[var]

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
            with progressbar_joblib.progressbar_joblib(tqdm(desc='Loading Dataset', total=len(list(root.group_keys())))) as progress_bar:
                dss = joblib.Parallel(n_jobs=-1, backend='loky')\
                    (joblib.delayed(store_open_group)(root, group) for group in list(root.group_keys()))
            self.dss = dict(dss)
            del dss
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
            for fullBurstID in tqdm(urls.index.get_level_values(0).unique(), desc='Loading Datasets'):
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
                    chunks=self.chunksize,
                    concat_dim='date',
                    combine='nested',
                    preprocess=lambda ds: burst_preprocess(ds, attr_start=attr_start, debug=False),
                    storage_options=storage_options,
                )
                # some variables are stored as int32 with scale factor, convert to float32 instead of default float64
                transform = xr.open_dataset(base, engine='zarr', zarr_format=3, chunks=self.chunksize,consolidated=True, storage_options=storage_options).astype('float32')

                ds = _bursts_transform_preprocess(bursts, transform)
                dss[fullBurstID] = ds
                del ds, bursts, transform

            #assert len(np.unique([ds.rio.crs.to_epsg() for ds in dss])) == 1, 'All datasets must have the same coordinate reference system'
            self.dss = dss
            del dss

    # def baseline_table(self):
    #     import xarray as xr
    #     return xr.concat([ds.BPR for ds in self.ds], dim='burst').mean('burst').to_dataframe()[['BPR']]

    def baseline_pairs(self, days:int|None=None, meters:int|None=None, invert:bool=False) -> pd.DataFrame:
        """
        Generates a sorted list of baseline pairs.
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the sorted list of baseline pairs with reference and repeat dates,
            timelines, and baselines.
    
        """
        import numpy as np
        import pandas as pd
        
        if days is None:
            # use large number for unlimited time interval in days
            days = 1e6
    
        tbl = self.baseline_table()
        data = []
        for line1 in tbl.itertuples():
            counter = 0
            for line2 in tbl.itertuples():
                #print (line1, line2)
                if not (line1.Index < line2.Index and (line2.Index - line1.Index).days < days + 1):
                    continue
                if meters is not None and not (abs(line1.BPR - line2.BPR)< meters + 1):
                    continue
    
                counter += 1
                if not invert:
                    data.append({'ref':line1.Index, 'rep': line2.Index,
                                 'ref_baseline': np.round(line1.BPR, 2),
                                 'rep_baseline': np.round(line2.BPR, 2)})
                else:
                    data.append({'ref':line2.Index, 'rep': line1.Index,
                                 'ref_baseline': np.round(line2.BPR, 2),
                                 'rep_baseline': np.round(line1.BPR, 2)})
    
        df = pd.DataFrame(data).sort_values(['ref', 'rep'])
        return df.assign(pair=[f'{ref} {rep}' for ref, rep in zip(df['ref'].dt.date, df['rep'].dt.date)],
                         baseline=df.rep_baseline - df.ref_baseline,
                         duration=(df['rep'] - df['ref']).dt.days,
                         rel=np.datetime64('nat'))

    def plot(self, records:pd.DataFrame|None=None, cmap='turbo'):
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import patheffects

        if records is None:
            records = self.to_dataframe()

        df = records.reset_index()
        df['date'] = df['startTime'].dt.date

        df['label'] = df.apply(lambda rec: f"{rec['flightDirection'].replace('E','')[:3]} {rec['date']} [{rec['pathNumber']}]", axis=1)
        unique_labels = sorted(df['label'].unique())
        unique_paths = sorted(df['pathNumber'].astype(str).unique())
        #colors = {label[-4:-1]: 'orange' if label[0] == 'A' else 'cyan' for i, label in enumerate(unique_labels)}
        n = len(unique_labels)
        colormap = matplotlib.cm.get_cmap(cmap, n)
        color_map = {label[-4:-1]: colormap(i) for i, label in enumerate(unique_labels)}
        fig, ax = plt.subplots(figsize=(10, 8))
        for label, group in df.groupby('label'):
            group.plot(ax=ax, edgecolor=color_map[label[-4:-1]], facecolor='none', lw=0.25, alpha=1, label=label)
        handles = [matplotlib.lines.Line2D([0], [0], color=color_map[label[-4:-1]], lw=1, label=label) for label in unique_labels]
        ax.legend(handles=handles, loc='upper right')

        col = df.columns[0]
        for _, row in df.iterrows():
            # compute centroid
            x, y = row.geometry.centroid.coords[0]
            ax.annotate(
                str(row[col]),
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                color=color_map[row['label'][-4:-1]],
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')]
            )

        ax.set_title('Sentinel-1 Burst Footprints')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    def to_dataset(self,
              datas: xr.Dataset | xr.DataArray | dict[str, xr.Dataset | xr.DataArray] | None = None,
              compute: bool = False):
        """
        This function is a faster implementation for the standalone function combination of xr.concat and xr.align:
        xr.concat(xr.align(*intfs, join='outer'), dim='stack_dim').ffill('stack_dim').isel(stack_dim=-1).compute()
        #xr.concat(xr.align(*datas, join='outer'), dim='stack_dim').mean('stack_dim').compute()
        """
        import xarray as xr
        import numpy as np
        import dask

        #print ('datas', datas, 'polarizations', polarizations)
        #print ()
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
                das = self.to_dataset([ds[pol] for ds in datas])
                das_total.append(das)
                del das
            das_total = xr.merge(das_total)
            
            if compute:
                progressbar(result := das_total.persist(), desc=f'Compute Unified Dataset'.ljust(25))
                del das_total
                return result
            return das_total

        # process list of dataarrays with single polarization

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
        def block_dask(stack, y_chunk, x_chunk):
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

            das_block_concat = xr.concat(das_block, dim="stack_dim", join="inner")
            # ffill does not work correct on complex data and per-component ffill is faster
            # the magic trick is to use sorting to ensure burst overpapping order
            # bursts ends should be overlapped by bursts starts
            if np.issubdtype(das_block_concat.dtype, np.complexfloating):
                return (das_block_concat.real.ffill("stack_dim").isel(stack_dim=-1)
                        + 1j*das_block_concat.imag.ffill("stack_dim").isel(stack_dim=-1)).values
            else:
                return das_block_concat.ffill("stack_dim").isel(stack_dim=-1).values
            # if not wrap:
            #     # calculate arithmetic mean for phase and correlation data
            #     return xr.concat(das_block, dim='stack_dim', join='inner').mean('stack_dim', skipna=True).values
            # else:
            #     # calculate circular mean for interferogram data
            #     block_complex = xr.concat([np.exp(1j * da) for da in das_block], dim='stack_dim').mean('stack_dim').values
            #     return np.arctan2(block_complex.imag, block_complex.real)

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
                ys.chunk({'y': self.chunksize}), 'y',
                xs.chunk({'x': self.chunksize}), 'x',
                meta = np.empty((0, 0, 0), dtype=datas[0].dtype)
            )
        da = xr.DataArray(data, coords={stackvar: stackval, 'y': ys, 'x': xs})\
            .rename(datas[0].name)\
            .assign_attrs(datas[0].attrs)
        del data
        return self.spatial_ref(da, datas)
