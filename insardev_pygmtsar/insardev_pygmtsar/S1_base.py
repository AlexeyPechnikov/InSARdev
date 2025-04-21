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

    def get_prefix(self, burst: str) -> str:
        df = self.get_record(burst)
        return df.index.get_level_values(0)[0]

    def get_burstfile(self, burst: str, ext: str='nc', clean: bool=False) -> str:
        import os
        prefix = self.get_prefix(burst)
        filename = os.path.join(self.basedir, prefix, f'{burst}.{ext}')
        #print ('get_burstfile', filename)
        if clean:
            if os.path.exists(filename):
                os.remove(filename)
        else:
            if not os.path.exists(filename):
                assert os.path.exists(filename), f'ERROR: The file is missed: {filename}'
        return filename

    def get_filename(self, burst: str, name: str, ext: str='nc', clean: bool=False) -> str:
        import os
        prefix = self.get_prefix(burst)
        filename = os.path.join(self.basedir, prefix, f'{name}.{ext}')
        #print ('get_filename', filename)
        if clean:
            if os.path.exists(filename):
                os.remove(filename)
        else:
            if not os.path.exists(filename):
                assert os.path.exists(filename), f'ERROR: The file is missed: {filename}'
        return filename
   
    def get_basename(self, burst: str) -> str:
        import os
        prefix = self.get_prefix(burst)
        basename = os.path.join(self.basedir, prefix, burst)
        return basename
    
    def get_dirname(self, burst: str) -> str:
        import os
        prefix = self.get_prefix(burst)
        dirname = os.path.join(self.basedir, prefix)
        return dirname

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

        return {key: (refs_dict[key], reps_dict[key]) for key in refs_dict}
