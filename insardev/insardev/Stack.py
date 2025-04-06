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
from .Stack_export import Stack_export

class Stack(Stack_export):

    # redefine for fast caching
    netcdf_complevel = -1

    def __repr__(self):
        return 'Object %s %d bursts %d dates' % (self.__class__.__name__, len(self.ds), len(self.ds[0].date))

    def to_dataset(self):
        import numpy as np
        import xarray as xr
        data = xr.concat(xr.align(*self.ds, join='outer'), dim='stack_dim').mean('stack_dim')
        return data

    def __init__(self, basedir, pattern_burst='*_*_?W?', pattern_date = '[0-9]{8}\.nc'):
        """
        Initialize an instance of the Stack class.
        """
        #import numpy as np
        import xarray as xr
        import geopandas as gpd
        import pandas as pd
        from shapely import wkt
        from tqdm.auto import tqdm
        import joblib
        import glob
        import os

        def read_netcdf_attributes(filename, attr_start='BPR'):
            with xr.open_dataset(filename, engine=self.netcdf_engine_read, format=self.netcdf_format) as ds:
                attrs = dict(ds.attrs)
            # remove attributes before geometry
            keys = list(attrs.keys())
            idx = keys.index(attr_start)
            attrs = {k: attrs[k] for k in keys[idx:]}
            return attrs

        self.basedir = basedir
        
        bursts = glob.glob(pattern_burst, root_dir=self.basedir)
        filenames = []
        for burst in bursts:
            basedir = os.path.join(self.basedir, burst)
            fnames = self._glob_re(pattern_date, basedir=basedir)
            filenames.extend(fnames)
            #print ('fnames', fnames)

        with self.tqdm_joblib(tqdm(desc='TODO fnames', total=len(filenames))) as progress_bar:
            attrs = joblib.Parallel(n_jobs=-1, backend=None)\
                (joblib.delayed(read_netcdf_attributes)(filename) for filename in filenames)
        #print ('attrs', attrs)

        processed_attrs = []
        for attr in attrs:
            processed_attr = {}
            for key, value in list(attr.items())[::-1]:
                #print (key, value)
                if hasattr(value, 'item'):
                    processed_attr[key] = value.item()
                elif key == 'geometry':
                    processed_attr[key] = wkt.loads(value)
                else:
                    processed_attr[key] = value
            processed_attrs.append(processed_attr)
        df = gpd.GeoDataFrame(processed_attrs)
        #df = df.sort_values(by=['fullBurstID', 'date']).set_index(['fullBurstID', 'date'])
        df = df.sort_values(by=df.columns[:2].tolist()).set_index(df.columns[:2].tolist())
        df['datetime'] = pd.to_datetime(df['datetime'])
        self.df = df

    def to_dataframe(self):
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
        return self.df

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
