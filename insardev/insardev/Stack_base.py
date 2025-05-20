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
from .Batch import Batch

class Stack_base():

    # def apply(self, *args, **kwarg):
    #     """
    #     Apply a function to multiple datasets or dictionaries of datasets with the same keys.

    #     Parameters
    #     ----------
    #     *args : list of datasets or dictionaries of datasets
    #         The datasets to apply the function to.
    #     **kwarg : dict
    #         The keyword arguments to pass to the function.

    #     Returns
    #     -------
    #     dict or dataset
    #         The result of applying the function to the datasets.
    #         If the input is a dictionary or a list of dictionaries, the result is a dictionary.
    #         If the input is a dataset or a list of datasets, the result is a dataset.
        
    #     Examples
    #     --------
    #     >>> sbas.apply(func=lambda a, b, **kwargs: (a, b))
    #     >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: (a, b))
    #     >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: a)
    #     >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: (a,b))
    #     >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: a)
    #     """
    #     from insardev_toolkit import progressbar
    #     import dask

    #     func = kwarg.pop('func', None)
    #     if func is None:
    #         raise ValueError('`func` argument is required')
    #     compute = kwarg.pop('compute', False)
    #     #print ('compute', compute)
    #     add_key = kwarg.pop('add_key', False)
    #     if not args:
    #         return
    #     datas = [self.to_dict(arg) if arg is not None else None for arg in args]
    #     keys = list(datas[0].keys())
    #     if add_key:
    #         dss = {key: func(*(d[key] if d is not None else None for d in datas), **(kwarg | {'key': key})) for key in keys}
    #     else:
    #         dss = {key: func(*(d[key] if d is not None else None for d in datas), **kwarg) for key in keys}
    #     if compute:
    #         progressbar(dss := dask.persist(dss)[0], desc=f'Computing...'.ljust(25))
    #     # detect output type
    #     sample = next(iter(dss.values()))
    #     # multiple datasets or dictionaries
    #     if (isinstance(sample, (tuple, list))):
    #         n = len(sample)
    #         dicts = [{key: dss[key][i] for key in keys} for i in range(n)]
    #         if isinstance(args[0], dict):
    #             return tuple(dicts)
    #         return tuple(d['default'] for d in dicts)
    #     # single dataset or dictionary
    #     if isinstance(args[0], dict):
    #         return dss
    #     return dss['default']

    # def apply_pol(self, *args, **kwarg):
    #     """
    #     Apply a function to multiple datasets or dictionaries of datasets with the same keys.
    #     The function process a single polarization at a time and then the results are merged.

    #     Parameters
    #     ----------
    #     *args : list of datasets or dictionaries of datasets
    #         The datasets to apply the function to.
    #     **kwarg : dict
    #         The keyword arguments to pass to the function.

    #     Returns
    #     -------
    #     dict or dataset
    #         The result of applying the function to the datasets.
    #         If the input is a dictionary or a list of dictionaries, the result is a dictionary.
    #         If the input is a dataset or a list of datasets, the result is a dataset.
        
    #     Examples
    #     --------
    #     >>> sbas.apply(func=lambda a, b, **kwargs: (a, b))
    #     >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: (a, b))
    #     >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: a)
    #     >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: (a,b))
    #     >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: a)
    #     """
    #     import xarray as xr
    #     import dask
    #     from insardev_toolkit import progressbar

    #     func = kwarg.pop('func', None)
    #     if func is None:
    #         raise ValueError('`func` argument is required')
    #     compute = kwarg.pop('compute', False)
    #     #print ('compute', compute)
    #     add_key = kwarg.pop('add_key', False)
    #     #print ('kwarg', kwarg)
    #     if not args:
    #         return
    #     datas = [self.to_dict(arg) if arg is not None else None for arg in args]
    #     # detect input type
    #     sample = next(iter(datas[0].values()))
    #     polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in sample.data_vars]
    #     #print ('polarizations', polarizations)
    #     keys = list(datas[0].keys())
    #     dss = []
    #     for polarization in polarizations:
    #         if add_key:
    #             dss_pol = {key: func(*(d[key][polarization] if d is not None else None for d in datas), **(kwarg | {'key': key})) for key in keys}
    #         else:
    #             dss_pol = {key: func(*(d[key][polarization] if d is not None else None for d in datas), **kwarg) for key in keys}
    #         if compute:
    #             progressbar(dss_pol := dask.persist(dss_pol)[0], desc=f'Computing {polarization}...'.ljust(25))
    #         dss.append(dss_pol)
    #         del dss_pol
    #     # detect output type
    #     sample = next(iter(dss[0].values()))
    #     #print ('sample', sample)
    #     # multiple datasets or dictionaries
    #     if (isinstance(sample, (tuple, list))):
    #         n = len(sample)
    #         dicts = [{key: xr.merge([dss[pidx][key][i] for pidx in range(len(polarizations))]) for key in keys} for i in range(n)]
    #         if isinstance(args[0], dict):
    #             return tuple(dicts)
    #         return tuple(d['default'] for d in dicts)
    #     # single dataset or dictionary
    #     # unpack polarizations
    #     dss = {key: xr.merge([dss[pidx][key] for pidx in range(len(polarizations))]) for key in keys}
    #     if isinstance(args[0], dict):
    #         return dss
    #     return dss['default']

    def _get_pairs(self, pairs, dates=False):
        """
        Get pairs as DataFrame and optionally dates array.

        Parameters
        ----------
        pairs : np.ndarray, optional
            An array of pairs. If None, all pairs are considered. Default is None.
        dates : bool, optional
            Whether to return dates array. Default is False.
        name : str, optional
            The name of the phase filter. Default is 'phasefilt'.

        Returns
        -------
        pd.DataFrame or tuple
            A DataFrame of pairs. If dates is True, also returns an array of dates.
        """
        import xarray as xr
        import pandas as pd
        import numpy as np
        from glob import glob

        if isinstance(pairs, pd.DataFrame):
            # workaround for baseline_pairs() output
            pairs = pairs.rename(columns={'ref_date': 'ref', 'rep_date': 'rep'})
        elif isinstance(pairs, (xr.DataArray, xr.Dataset)):
            # pairs = pd.DataFrame({
#                 'ref': pairs.coords['ref'].values,
#                 'rep': pairs.coords['rep'].values
#             })
            refs = pairs.coords['ref'].values
            reps = pairs.coords['rep'].values
            pairs = pd.DataFrame({
                'ref': refs if isinstance(refs, np.ndarray) else [refs],
                'rep': reps if isinstance(reps, np.ndarray) else [reps]
            })
        else:
            # Convert numpy array to DataFrame
            # in case of 1d array with 2 items convert to a single pair
            pairs_2d = [pairs] if np.asarray(pairs).shape == (2,) else pairs
            pairs = pd.DataFrame(pairs_2d, columns=['ref', 'rep'])

        # Convert ref and rep columns to datetime format
        pairs['ref'] = pd.to_datetime(pairs['ref'])
        pairs['rep'] = pd.to_datetime(pairs['rep'])
        pairs['pair'] = [f'{ref} {rep}' for ref, rep in zip(pairs['ref'].dt.date, pairs['rep'].dt.date)]
        # Calculate the duration in days and add it as a new column
        #pairs['duration'] = (pairs['rep'] - pairs['ref']).dt.days

        if dates:
            # pairs is DataFrame
            dates = np.unique(pairs[['ref', 'rep']].astype(str).values.flatten())
            return (pairs, dates)
        return pairs

    def _get_pairs_matrix(self, pairs):
        """
        Create a matrix based on interferogram dates and pairs.

        Parameters
        ----------
        pairs : pandas.DataFrame or xarray.DataArray or xarray.Dataset
            DataFrame or DataArray containing interferogram date pairs.
        
        Returns
        -------
        numpy.ndarray
            A matrix with one row for every interferogram and one column for every date.
            Each element in the matrix is a float, with 1 indicating the start date,
            -1 indicating the end date, 0 if the date is covered by the corresponding 
            interferogram timeline, and NaN otherwise.

        """
        import numpy as np
        import pandas as pd

        # also define image capture dates from interferogram date pairs
        pairs, dates = self._get_pairs(pairs, dates=True)
        pairs = pairs[['ref', 'rep']].astype(str).values

        # here are one row for every interferogram and one column for every date
        matrix = []
        for pair in pairs:
            #mrow = [date>=pair[0] and date<=pair[1] for date in dates]
            mrow = [(-1 if date==pair[0] else (1 if date==pair[1] else (0 if date>pair[0] and date<pair[1] else np.nan))) for date in dates]
            matrix.append(mrow)
        matrix = np.stack(matrix).astype(np.float32)
        return matrix
