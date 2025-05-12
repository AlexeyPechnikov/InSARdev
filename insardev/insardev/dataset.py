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
from insardev_toolkit import datagrid

class dataset(datagrid):

    # import zarr
    # zarr_clevel: int = 0
    # zarr.config['array.v3_default_compressors.numeric'][0]['configuration']['level'] = zarr_clevel
    # zarr.config['array.v3_default_compressors.bytes'][0]['configuration']['level']   = zarr_clevel
    # zarr.config['array.v3_default_compressors.string'][0]['configuration']['level'] = zarr_clevel

    def snapshot_interleave(self, *args, store: str | None = None, storage_options: dict[str, str] | None = None,
                            compat: bool = True, n_jobs: int = -1, debug=False):
        """
        Save and open a Zarr store or just open it when no data arguments are provided.
        This function is a shortcut for snapshot(...) with interleave=True.
        """
        return self.snapshot(*args, store=store, storage_options=storage_options, compat=compat, interleave=True, n_jobs=n_jobs, debug=debug)

    def snapshot(self, *args, store: str | None = None, storage_options: dict[str, str] | None = None,
                 compat: bool = True, interleave: bool = False, n_jobs: int = -1, debug=False):
        """
        Save and open a Zarr store or just open it when no data arguments are provided.
        This function wraps save(...) and open(...) functions.

        Parameters
        ----------
        *args : xr.Dataset or dict[str, xr.Dataset]
            Multiple datasets to save. Can be:
            - Individual xarray.Datasets
            - Dictionary of xarray.Datasets
        store : str, optional
            Path to the store (directory name).
        storage_options : dict[str, str], optional
            Storage options for the store.
        compat : bool, optional
            If True, automatically pack datasets saved as a single dataset into a dictionary.
            If False, only allow a dictionary of datasets. Default is True.
        interleave : bool, optional
            If True, save two datasets as interleaved variables. Default is False.
        n_jobs : int, optional
            Number of parallel jobs to use for saving. Default is -1 (use all available cores).
        debug : bool, optional
            If True, print debug information and use sequential joblib backend. Default is False.
        """
        # call the function without data arguments to only open existing store
        if len(args) > 0:
            self.save(*args, store=store, storage_options=storage_options, compat=compat, interleave=interleave, n_jobs=n_jobs, debug=debug)
        return self.open(store=store, storage_options=storage_options, compat=compat, interleave=interleave, n_jobs=n_jobs, debug=debug)

    def save(self, *args, store: str, storage_options: dict[str, str] | None = None, compat: bool = True, interleave: bool = False,
             caption: str | None = 'Saving...', n_jobs: int = -1, debug=False):
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
        interleave : bool, optional
            If True, save two datasets as interleaved variables. Default is False.
        n_jobs : int, optional
            Number of parallel jobs to use for saving. Default is -1 (use all available cores).
        debug : bool, optional
            If True, print debug information and use sequential backend. Default is False.
        Notes
        -----
        The empty consolidated metadata created at the start and the groups one at the end.
        In case the data are not saved correctly, the consolidated metadata is valid but empty.
        When the data are saved correctly, the consolidated metadata lists groups names but not all groups variables.
        """
        import xarray as xr
        import dask
        import zarr
        import os
        import shutil
        import joblib
        from tqdm.auto import tqdm
        import gc
        from dask.distributed import get_client

        joblib_backend = 'sequential' if debug else 'threading'

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

        def _save_grp(grp, ds, chunks):
            # silently drop problematic attributes
            ds_clean = ds.copy()
            for v in ds_clean.data_vars:
                ds_clean[v].attrs.pop('grid_mapping', None)
            # save to subdirectory
            #print (ds_clean)
            ds_clean.chunk(chunks).to_zarr(
                store=f'{store}/{grp}',
                mode='w',
                consolidated=True,
                zarr_format=3
            )
            del ds_clean

        # open to drop existing store
        root_group = zarr.group(store=store, zarr_format=3, overwrite=True)
        with progressbar_joblib.progressbar_joblib(tqdm(desc='Saving Zarr...'.ljust(25), total=len(datas))) as progress_bar:
            joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(_save_grp) (grp, ds, chunks=self.zarr_chunksize) for grp, ds in datas.items())

        # consolidate metadata for the groups in the store
        zarr.consolidate_metadata(store, zarr_format=3)
        del datas
        # clear memory
        gc.collect()
        try:
            client = get_client()
            client.run(lambda: __import__('gc').collect())
        except ValueError:
            pass


    def open(self, store: str, storage_options: dict[str, str] | None = None, compat: bool = True, interleave: bool = False,
             n_jobs: int = -1, debug=False):
        """
        Load a Zarr store created by save(...).

        Parameters
        ----------
        store : str | zarr.storage.Store
            Path to the store (directory '<name>.zarr') or a Zarr storage object.
        storage_options : dict[str, str], optional
            Storage options for the store.
        compat : bool, optional
            If True, automatically unpack datasets saved as a list or a single dataset.
            If False, return a dictionary of datasets. Default is True.
        interleave : bool, optional
            If True, return two dictionaries of datasets saved as interleaved variables. Default is False.
        n_jobs : int, optional
            Number of parallel jobs to use for opening. Default is -1 (use all available cores).
        debug : bool, optional
            If True, print debug information and use sequential backend. Default is False.

        Returns
        -------
        Stack
        """
        import os
        import zarr
        import xarray as xr
        import joblib
        from tqdm.auto import tqdm

        root = zarr.open_consolidated(store, storage_options=storage_options, zarr_format=3, mode='r')
        groups = list(root.group_keys())
        print(len(groups))

        joblib_backend = 'sequential' if debug else 'threading'

        def _load_grp(grp, chunks):
            #ds = xr.open_zarr(store, group=grp, consolidated=True, zarr_format=3)
            ds = xr.open_zarr(f'{store}/{grp}', storage_options=storage_options, consolidated=True, zarr_format=3, chunks=chunks)
            return grp, ds
        with progressbar_joblib.progressbar_joblib(tqdm(desc='Opening Zarr...'.ljust(25), total=len(groups))) as progress_bar:
            results = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(_load_grp) (grp, chunks=self.chunksize) for grp in groups)
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

