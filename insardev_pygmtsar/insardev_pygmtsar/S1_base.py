# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from insardev_toolkit import tqdm_joblib, tqdm_dask
from .dataset import dataset

class S1_base(tqdm_joblib, dataset):

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

#     def get_prefix(self, burst, date=None):
#         """
#         Get the path prefix for a burst and date combination.

#         Parameters
#         ----------
#         burst : str
#             The burst identifier.
#         date : str, optional
#             Date string in yyyy-mm-dd format. If None, uses reference date.
#             If False, returns only burst path.

#         Returns
#         -------
#         str
#             Path prefix combining burst and date information.

#         Examples
#         --------
#         Get prefix for burst '173_370325_IW1' and date '2022-01-20':
#         prefix = stack.get_prefix('173_370325_IW1', '2022-01-20')
#         """
#         import os

#         print ('get_prefix', burst, date)

#         assert date is None or date is False or len(date)==10, f'ERROR: date format is not yyyy-mm-dd (burst={burst} date={date})'
#         # TODO
#         assert len(burst)!=10, 'ERROR: mixed burst and date arguments (burst={burst} date={date})'

#         path = os.path.join(self.basedir, burst)
#         if not os.path.isdir(path):
#             os.makedirs(path)

#         if date == False:
#             return os.path.join(burst, '')
            
#         # use reference datetime if not defined
# #         if date is None or date  == self.reference:
# #             df = self.get_reference(burst)
# #         else:
# #             df = self.get_repeat(burst, date)    
# #         name = df.burst.iloc[0]
#         if date is None:
#             date = self.reference
#         df = self.get_record(burst, date)
#         prefix = df.burst.iloc[0]
#         return os.path.join(burst, prefix)

    def get_prefix(self, burst):
        df = self.get_record(burst)
        return df.index.get_level_values(0)[0]

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

    # def get_reference(self, burst):
    #     """
    #     Return dataframe reference record.

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     -------
    #     pd.DataFrame
    #         The DataFrame containing reference record.
    #     """
    #     df = self.df.loc[[(burst, self.reference)]]
    #     assert len(df) > 0, f'Reference record not found'
    #     return df
        
    # def get_repeat(self, burst, date=None):
    #     """
    #     Return dataframe repeat records (excluding reference).

    #     Parameters
    #     ----------
    #     date : datetime, optional
    #         The date for which to return repeat records. If None, all dates are considered. Default is None.

    #     Returns
    #     -------
    #     pd.DataFrame
    #         The DataFrame containing repeat records for the specified date.
    #     """
    #     if date is None:
    #         df_filtered = self.df[self.df.index.get_level_values(0) == burst]
    #         idx_reference = self.df.index[self.df.index.get_level_values(1) == self.reference]
    #         return df_filtered.loc[df_filtered.index.difference(idx_reference)]

    #     assert not date == self.reference, f'ERROR: repeat date cannot be equal to reference date "{date}"'
    #     return self.df.loc[[(burst, date)]]

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