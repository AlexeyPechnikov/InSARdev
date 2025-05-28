# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from insardev_toolkit import progressbar_joblib
from insardev_toolkit import datagrid

class S1_base(progressbar_joblib, datagrid):
    import pandas as pd

    def __repr__(self):
        return 'Object %s %d items\n%r' % (self.__class__.__name__, len(self.df), self.df)

    def to_dataframe(self, crs: int=4326, ref: str=None) -> pd.DataFrame:
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
        if ref is None:
            df = self.df
        else:
            path_number = self.df[self.df.startTime.dt.date.astype(str)==ref].pathNumber.unique()
            if len(path_number) == 0:
                return self.df
            df = self.df[self.df.pathNumber==path_number[0]]
        return df.set_crs(4326).to_crs(crs)

    def fullBurstId(self, burst: str) -> str:
        df = self.get_record(burst)
        return df.index.get_level_values(0)[0]

    def get_burstfile(self, burst: str, basedir: str, ext: str='nc', clean: bool=False) -> str:
        import os
        prefix = self.fullBurstId(burst)
        filename = os.path.join(basedir, prefix, f'{burst}.{ext}')
        #print ('get_burstfile', filename)
        if clean:
            if os.path.exists(filename):
                os.remove(filename)
        else:
            if not os.path.exists(filename):
                assert os.path.exists(filename), f'ERROR: The file is missed: {filename}'
        return filename

    def get_filename(self, burst: str, basedir: str, name: str, ext: str='nc', clean: bool=False) -> str:
        import os
        filename = os.path.join(basedir, f'{name}.{ext}' if ext is not None else name)
        #print ('get_filename', filename)
        if clean:
            if os.path.exists(filename):
                os.remove(filename)
        else:
            if not os.path.exists(filename):
                assert os.path.exists(filename), f'ERROR: The file is missed: {filename}'
        return filename
   
    # def get_basename(self, workdir: str, burst: str) -> str:
    #     import os
    #     prefix = self.fullBurstId(burst)
    #     basename = os.path.join(workdir, prefix, burst)
    #     return basename
    
    # def get_dirname(self, workdir: str, burst: str) -> str:
    #     import os
    #     prefix = self.fullBurstId(burst)
    #     dirname = os.path.join(workdir, prefix)
    #     return dirname

    def get_record(self, burst: str) -> pd.DataFrame:
        """
        Return dataframe record.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            The DataFrame containing reference record.
        """
        df = self.df[self.df.index.get_level_values(2)==burst]
        if len(df) == 0:
            df = self.df[self.df.index.get_level_values(0)==burst]
        assert len(df) > 0, f'Record not found'
        return df

    def get_repref(self, ref: str, records: pd.DataFrame=None) -> dict:
        """
        Get the reference and repeat bursts for a given reference date.

        Parameters
        ----------
        ref : str
            The reference date.
        records : pd.DataFrame, optional
            The DataFrame containing the records.

        Returns
        -------
        dict
            A dictionary with the reference and repeat burst lists.
        """
        if records is None:
            records = self.to_dataframe(ref=ref)
        
        recs_ref = records[records.startTime.dt.date.astype(str)==ref]
        refs_dict = {}
        for rec in recs_ref.itertuples():
            refs_dict.setdefault(rec.Index[0], []).append(rec.Index)
        
        recs_rep = records[records.startTime.dt.date.astype(str)!=ref]
        reps_dict = {}
        for rec in recs_rep.itertuples():
            reps_dict.setdefault(rec.Index[0], []).append(rec.Index)

        for key in refs_dict:
            if key not in reps_dict:
                print (f'NOTE: {key} has no repeat bursts, ignore.')
        for key in reps_dict:
            if key not in refs_dict:
                print (f'NOTE: {key} has no reference bursts, ignore.')

        # return only pairs with both reference and repeat bursts
        return {key: (refs_dict[key], reps_dict[key]) for key in refs_dict if key in reps_dict}

    def julian_to_datetime(self, julian_timestamp: float) -> pd.Timestamp:
        """
        Convert Julian timestamp to datetime.
        
        Parameters
        ----------
        julian_timestamp : float
            Timestamp in format YYYYDOY.FRACTION, e.g., 2023040.1484139557
            where YYYY is year, DOY is day of year, and FRACTION is fractional day
            
        Returns
        -------
        pd.Timestamp
            Converted datetime
        """
        import pandas as pd
        import numpy as np
        
        # Split into year, DOY, and fraction
        year = int(julian_timestamp / 1000)  # Get year from first 4 digits
        doy = int(julian_timestamp % 1000)   # Get DOY from next 3 digits
        fraction = julian_timestamp - int(julian_timestamp)  # Get fractional day
        
        # Convert to datetime
        base_date = pd.Timestamp(f"{year}-01-01")
        date = base_date + pd.Timedelta(days=doy) + pd.Timedelta(days=fraction)
        
        return date