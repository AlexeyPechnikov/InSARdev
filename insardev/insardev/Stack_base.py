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
from insardev_toolkit import progressbar_joblib
from .dataset import dataset

class Stack_base(progressbar_joblib, dataset):

    def checkout(self, *args, name: str | None = None, n_jobs: int = -1, compat: bool = True, interleave: bool = False):
        self.save(*args, name=name, n_jobs=n_jobs, compat=compat, interleave=interleave)
        return self.open(name=name, n_jobs=n_jobs, compat=compat, interleave=interleave)

    def save(self, *args, name: str | None = None, caption: str | None = 'Saving...', n_jobs: int = -1, compat: bool = True, interleave: bool = False):
        """
        Save multiple xarray.Datasets into one Zarr store, each under its own subgroup.

        Parameters
        ----------
        *args : xr.Dataset or dict[str, xr.Dataset]
            Multiple datasets to save. Can be:
            - Individual xarray.Datasets
            - Dictionary of xarray.Datasets
            - List of xarray.Datasets
        name : str, optional
            base name for the store directory.
        compat : bool, optional
            If True, automatically pack datasets saved as a list or a single dataset into a dictionary.
            If False, only allow a dictionary of datasets. Default is True.
        n_jobs : int, optional
            Number of parallel jobs to use for saving. Default is -1 (use all available cores).
        Notes
        -----
        This implementation is slightly inaccurate since some data processing occurs outside 
        the progressbar section.
        """
        import xarray as xr
        import dask
        import zarr
        import os
        import shutil
        import joblib
        from tqdm.auto import tqdm

        if name is None:
            raise ValueError('name is required')
        store_path = f'{name}.zarr'

        # remove any existing store
        if os.path.exists(store_path):
            shutil.rmtree(store_path)

        # process all arguments into a single dictionary
        datas = {}
        if interleave:
            if len(args) == 2 and (isinstance(args[0], dict) or isinstance(args[1], dict)):
                for (k0, v0), (k1, v1) in zip(args[0].items(), args[1].items()):
                    datas[f'i0_{k0}'] = v0
                    datas[f'i1_{k1}'] = v1
            elif len(args) == 2 and (isinstance(args[0], xr.Dataset) and isinstance(args[1], xr.Dataset)):
                datas['i0_default'] = args[0]
                datas['i1_default'] = args[1]
            else:
                raise ValueError('Arguments must be two xarray.Datasets or dictionaries of xarray.Datasets when interleave is True')
        elif compat and len(args) == 1 and isinstance(args[0], xr.Dataset):
            datas = {'default': args[0]}
        else:
            for i, arg in enumerate(args):
                if isinstance(arg, xr.Dataset):
                    if not compat:
                        raise ValueError('Arguments must be dictionaries of xarray.Datasets when compat is False')
                    datas[f'default_{i}'] = arg
                elif isinstance(arg, dict):
                    datas.update(arg)
                else:
                    raise ValueError('Arguments must be xarray.Datasets or dictionaries of xarray.Datasets when compat is True')

        def _save_grp(grp, ds):
            # silently drop problematic attributes
            ds_clean = ds.copy()
            for v in ds_clean.data_vars:
                ds_clean[v].attrs.pop('grid_mapping', None)
            # save to subdirectory
            ds_clean.to_zarr(
                store=f'{store_path}/{grp}',
                mode='w',
                consolidated=True,
                zarr_format=3
            )

        with self.progressbar_joblib(tqdm(desc='Saving...'.ljust(25), total=len(datas))) as progress_bar:
            joblib.Parallel(n_jobs=n_jobs, backend='threading')(joblib.delayed(_save_grp) (grp, ds) for grp, ds in datas.items())

        # consolidate metadata for the whole store
        root_store = zarr.storage.LocalStore(store_path)
        root_group = zarr.group(store=root_store, zarr_format=3, overwrite=False)
        zarr.consolidate_metadata(root_store, zarr_format=3)
        del datas

    def open(self, name: str, n_jobs: int = -1, compat: bool = True, interleave: bool = False):
        """
        Load a Zarr store created by save(...).

        Parameters
        ----------
        name : str
            Base name of the store (directory '<name>.zarr').
        compat : bool, optional
            If True, automatically unpack datasets saved as a list or a single dataset.
            If False, return a dictionary of datasets. Default is True.

        Returns
        -------
        Stack
        """
        import os
        import zarr
        import xarray as xr
        import joblib
        from tqdm.auto import tqdm

        store_path = f'{name}.zarr'
        if not os.path.isdir(store_path):
            raise FileNotFoundError(f'No such Zarr store: {store_path!r}')

        # open store
        # store = zarr.storage.LocalStore(store_path)
        # root = zarr.open_consolidated(store)
        # dss = {grp: xr.open_zarr(store, group=grp) for grp in root.group_keys()}

        store = zarr.storage.LocalStore(store_path)
        root = zarr.open_consolidated(store)
        groups = list(root.group_keys())
        print(len(groups))

        def _load_grp(grp):
            #ds = xr.open_zarr(store, group=grp, consolidated=True, zarr_format=3)
            ds = xr.open_zarr(f'{store_path}/{grp}', consolidated=True, zarr_format=3)
            return grp, ds
        with self.progressbar_joblib(tqdm(desc='Opening...'.ljust(25), total=len(groups))) as progress_bar:
            results = joblib.Parallel(n_jobs=n_jobs, backend='threading')(joblib.delayed(_load_grp) (grp) for grp in groups)
        dss = dict(results)
        del results

        if interleave:
            # unpack interleaved datasets
            if len(dss) == 2 and 'i0_default' in dss and 'i1_default' in dss:
                # special case for two datasets
                return dss['i0_default'], dss['i1_default']
            dss0 = {k[len('i0_'):]: v for k, v in dss.items() if k.startswith('i0_')}
            dss1 = {k[len('i1_'):]: v for k, v in dss.items() if k.startswith('i1_')}
            return dss0, dss1
        elif compat and len(dss) == 1 and 'default' in dss:
            # unpack single dataset
            return dss['default']
        elif compat and 'default_0' in dss:
            # default_0, default_1, etc. means that the datasets were saved as a list, unpack them
            return [dss[f'default_{i}'] for i in range(len(dss))]

        return dss

    def get_pairs(self, pairs, dates=False):
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

    def get_pairs_matrix(self, pairs):
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
        pairs, dates = self.get_pairs(pairs, dates=True)
        pairs = pairs[['ref', 'rep']].astype(str).values

        # here are one row for every interferogram and one column for every date
        matrix = []
        for pair in pairs:
            #mrow = [date>=pair[0] and date<=pair[1] for date in dates]
            mrow = [(-1 if date==pair[0] else (1 if date==pair[1] else (0 if date>pair[0] and date<pair[1] else np.nan))) for date in dates]
            matrix.append(mrow)
        matrix = np.stack(matrix).astype(np.float32)
        return matrix

    @staticmethod
    def phase_to_positive_range(phase):
        """
        Convert phase from the range [-pi, pi] to [0, 2pi].
    
        Parameters
        ----------
        phase : array_like
            Input phase values in the range [-pi, pi].
    
        Returns
        -------
        ndarray
            Phase values converted to the range [0, 2pi].
        
        Examples
        --------
        >>> phase_to_positive_range(np.array([-np.pi, -np.pi/2, np.pi, 2*-np.pi-1e-6, 2*-np.pi]))
        array([3.14159265, 4.71238898, 3.14159265, 6.28318431, 0.        ])
        """
        import numpy as np
        return (phase + 2 * np.pi) % (2 * np.pi)
    
    @staticmethod
    def phase_to_symmetric_range(phase):
        """
        Convert phase from the range [0, 2pi] to [-pi, pi].
    
        Parameters
        ----------
        phase : array_like
            Input phase values in the range [0, 2pi].
    
        Returns
        -------
        ndarray
            Phase values converted to the range [-pi, pi].
        
        Examples
        --------
        >>> phase_to_symmetric_range(np.array([0, np.pi, 3*np.pi/2, 2*np.pi]))
        array([ 0.        ,  3.14159265, -1.57079633,  0.        ])
        """
        import numpy as np
        return (phase + np.pi) % (2 * np.pi) - np.pi
