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

class Stack(Stack_plot):

    # redefine for fast caching
    netcdf_complevel = -1

    def __repr__(self):
        return f"Object {self.__class__.__name__} with {len(self.dss)} bursts for {len(self.dss[0].date)} dates"

    def PRM(self, key):
        """
        Use as stack.PRM('radar_wavelength') to get the radar wavelength from the first burst.
        """
        return self.dss[0].attrs[key]

    def to_dataframe(self, id=None, date=None):
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

        df = self.df
        if id is not None:
            df = df[df.index.get_level_values(0).isin([id] if isinstance(id, str) else id)]
        if date is not None:
            df = df[df.index.get_level_values(2).isin([date] if isinstance(date, str) else date)]
        return df

    def to_dataset(self, records=None, polarizations=None, ids=None):
        import pandas as pd

        print ('records', records, 'polarizations', polarizations, 'ids', ids)
        if records is None and polarizations is None and ids is None:
            return self.dss

        if ids is not None and isinstance(ids, str):
            ids = [ids]
        
        if polarizations is not None and isinstance(polarizations, str):
            polarizations = [polarizations]
        polarizations_all = [pol for pol in ['VV','VH','HH','HV'] if pol in self.dss[0].data_vars]

        if records is None:
            records = self.df
            #if dates is None else [ds.sel(date=dates) for ds in self.dss]
        
        assert isinstance(records, pd.DataFrame)
        
        dss = []
        for rid in records.index.get_level_values(0).unique():
            if ids is not None and not rid in ids:
                continue
            dates = records[records.index.get_level_values(0)==rid].index.get_level_values(2).astype(str)
            ds = [ds for ds in self.dss if ds.id==rid][0].sel(date=dates)
            if polarizations is not None:
                for pol in polarizations_all:
                    if pol not in polarizations:
                        ds = ds.drop(pol)
            dss.append(ds)
        return dss

    # def to_dataset(self, records=None):
    #     dss = self.to_datasets(records)
    #     if len(dss) > 1:
    #         return self.to_datasets(records)[0]

    def __init__(self, basedir, pattern_burst='*_*_?W?',
                 pattern_date = 'S1_[0-9]+_IW[0-9]_[0-9]{8}T[0-9]{6}_[HV]{2}_.*-BURST\.nc',
                 attr_start='BPR', debug=False):
        import numpy as np
        import xarray as xr
        import pandas as pd
        import geopandas as gpd
        from shapely import wkt
        import glob
        import os

        # use oter variables attr_start, debug
        def ds_preprocess(ds):
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
            
            # remove attributes for repeat bursts
            BPR = ds['BPR'].values.item(0)
            if BPR != 0:
                ds.attrs = {}

            #polarization = ds['polarization'].values.item(0)
            scale = ds['SLC_scale'].values.item(0)
            #print ('scale', scale)
            ds['data'] = (scale*(ds.re + 1j*ds.im)).astype(np.complex64).where(ds.re != 0)
            if not debug:
                del ds['re'], ds['im'], ds['SLC_scale']
            date = pd.to_datetime(ds['startTime'].item())
            return ds.expand_dims({'date': np.array([date.date()], dtype='U10')})
        
        self.basedir = basedir

        dss = []
        bursts = sorted(glob.glob(pattern_burst, root_dir=self.basedir))
        #print ('bursts', bursts)
        for burst in bursts:
            basedir = os.path.join(self.basedir, burst)
            filenames = self._glob_re(pattern_date, basedir=basedir)
            #print ('\tfilenames', filenames)
            filename = os.path.join(basedir, f'trans.nc')

            ds = xr.open_mfdataset(
                filenames,
                engine=self.netcdf_engine_read,
                format=self.netcdf_format,
                parallel=True,
                concat_dim='date',
                chunks={'date': 1, 'y': self.chunksize, 'x': self.chunksize},
                combine='nested',
                preprocess=ds_preprocess,
            )

            # in case of multiple polarizations, merge them into a single dataset
            polarizations = np.unique(ds.polarization)
            if len(polarizations) > 1:
                datas = []
                for polarization in polarizations:
                    data = ds.isel(date=ds.polarization == polarization).rename({'data': polarization})
                    del data['polarization'], data['burst']
                    datas.append(data)
                ds = xr.merge(datas)
            else:
                ds = ds.rename({'data': polarizations[0]})

            trans = xr.open_dataset(filename, engine=self.netcdf_engine_read, format=self.netcdf_format)

            #scale = data.attrs['SLC_scale']
            #scale = ds['SLC_scale'].values.item(0)
            #print ('scale', scale)
            # Create complex data while preserving attributes
            #ds_complex = self.spatial_ref(
            #    (scale*(ds.re + 1j*ds.im)).astype(np.complex64).where(ds.re != 0).rename(burst),
            #    ds
            #).to_dataset(name='data')

            #ds_complex.attrs.update(ds.attrs)

            # for var in ds.data_vars:
            #     if var not in ['re', 'im']:
            #         ds_complex[var] = ds[var]
            for var in trans.data_vars:
                if var not in ['re', 'im']:
                    ds[var] = trans[var]
            del trans

            #ds_complex.attrs = ds_complex.data.attrs
            #ds_complex.data.attrs = {}
            dss.append(ds.assign_attrs({'id': burst}))
        self.dss = dss

        # make attributes dataframe from datas
        processed_attrs = []
        for ds in dss:
            #print (data.id)
            attrs = [data_var for data_var in ds if ds[data_var].dims==('date',)][::-1]
            attr_start_idx = attrs.index(attr_start)
            for date_idx, date in enumerate(ds.date.values):
                processed_attr = {}
                for attr in attrs[:attr_start_idx+1]:
                    value = ds[attr].item(date_idx)
                    #print (attr, date_idx, date, value)
                    processed_attr['date'] = date
                    if hasattr(value, 'item'):
                        processed_attr[attr] = value.item()
                    elif attr == 'geometry':
                        processed_attr[attr] = wkt.loads(value)
                    else:
                        processed_attr[attr] = value
                processed_attrs.append(processed_attr)
                #print (processed_attr)
        df = gpd.GeoDataFrame(processed_attrs)
        del df['date']
        df['polarization'] = ','.join(polarizations)
        burst_col = df.columns[0]
        #print ('df.columns[0]', df.columns[0])
        #print ('df.columns[:2][::-1].tolist()', df.columns[:2][::-1].tolist())
        df['startTime'] = pd.to_datetime(df['startTime'])
        df['date'] = df['startTime'].dt.date.astype(str)
        df = df.sort_values(by=[burst_col, 'polarization', 'date']).set_index([burst_col, 'polarization', 'date'])
        
        self.df = df

    # def baseline_table(self):
    #     import xarray as xr
    #     return xr.concat([ds.BPR for ds in self.ds], dim='burst').mean('burst').to_dataframe()[['BPR']]

    def baseline_pairs(self, days=None, meters=None, invert=False):
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

    def plot(self, records=None):
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt

        if records is None:
            records = self.df

        df = records.reset_index()

        df['label'] = df.apply(lambda rec: f"{rec['flightDirection'].replace('E','')[:3]} {rec['date']} [{rec['pathNumber']}]", axis=1)
        unique_labels = sorted(df['label'].unique())
        unique_paths = sorted(df['pathNumber'].astype(str).unique())
        colors = {label[-4:-1]: 'orange' if label[0] == 'A' else 'cyan' for i, label in enumerate(unique_labels)}
        fig, ax = plt.subplots(figsize=(10, 8))
        for label, group in df.groupby('label'):
            group.plot(ax=ax, edgecolor=colors[label[-4:-1]], facecolor='none', linewidth=1, alpha=1, label=label)
        handles = [matplotlib.lines.Line2D([0], [0], color=colors[label[-4:-1]], lw=1, label=label) for label in unique_labels]
        ax.legend(handles=handles, loc='upper right')
        ax.set_title('Sentinel-1 Burst Footprints')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
