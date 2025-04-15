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

    def __repr__(self):
        return 'Object %s %d items\n%r' % (self.__class__.__name__, len(self.df), self.df)

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

    def get_prefix(self, burst):
        df = self.get_record(burst)
        return df.index.get_level_values(0)[0]

    def get_burstfile(self, burst, ext='nc', clean=False):
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

    def get_filename(self, burst, name, ext='nc', clean=False):
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
   
    def get_basename(self, burst):
        import os
        prefix = self.get_prefix(burst)
        basename = os.path.join(self.basedir, prefix, burst)
        return basename
    
    def get_dirname(self, burst):
        import os
        prefix = self.get_prefix(burst)
        dirname = os.path.join(self.basedir, prefix)
        return dirname

    def get_reference(self):
        """
        Get the reference date for the Stack object.
        """
        return self.reference

    def set_reference(self, reference):
        """
        Define reference date for Stack object.

        Parameters
        ----------
        reference : str
            Date string representing the reference scene.

        Returns
        -------
        Stack
            Modified instance of the Stack class.

        Examples
        --------
        Set the reference scene to '2022-01-20':
        stack.set_reference('2022-01-20')
        """
        if reference is None:
            if self.reference is None:
                self.reference = self.df.startTime.dt.date.iloc[0]
                print (f'NOTE: auto set reference date {self.reference}. You can change it like set_reference("{self.reference}")')
            return self
        assert reference in self.df.startTime.dt.date.astype(str).values, f'Reference burst(s) not found: {reference}'
        self.reference = reference
        return self

    def get_record(self, burst):
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

    def get_records_ref(self, records=None):
            if records is None:
                records = self.df
            records_ref = records[records.startTime.dt.date.astype(str)==self.reference]
            return records_ref

    def get_records_rep(self, records=None):
            if records is None:
                records = self.df
            records_rep = records[records.startTime.dt.date.astype(str)!=self.reference]
            return records_rep
    
    def get_records_rep_ref(self, records=None):
        records_ref = self.get_records_ref(records)
        refs_dict = {}
        for record in records_ref.itertuples():
            refs_dict[record.Index[:2]] = record.Index[2]
        #print ('refs_dict', refs_dict)
        records_rep = self.get_records_rep(records)
        reps_dict = {}
        for record in records_rep.itertuples():
            reps_dict[record.Index[2]] = record.Index[:2]
        
        return {burst_rep: refs_dict[reps_dict[burst_rep]] for burst_rep in reps_dict.keys()}