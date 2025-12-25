# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_topo import S1_topo

class S1_transform(S1_topo):
    import pandas as pd
    import xarray as xr
    import numpy as np

    def consolidate_metadata(self, target: str, resolution: tuple[int, int]=(20, 5), burst: str=None):
        """
        Consolidate metadata for a given resolution and burst.

        Parameters
        ----------
        target : str
            The output directory where the results are saved.
        burst : str
            The burst to use.
        """
        import zarr
        import os
        root_dir = target
        if burst:
            root_dir = os.path.join(target, self.fullBurstId(burst))
        #print ('root_dir', root_dir)
        root_store = zarr.storage.LocalStore(root_dir)
        root_group = zarr.group(store=root_store, zarr_format=3, overwrite=False)
        zarr.consolidate_metadata(root_store)

    def transform(self,
                  target: str,
                  ref: str,
                  records: pd.DataFrame|None=None,
                  epsg: str|int|None='auto',
                  resolution: tuple[int, int]=(20, 5),
                  remove_topo_phase: bool = True,
                  dem_vertical_accuracy: float=0.5,
                  alignment_spacing: float=12.0/3600,
                  overwrite: bool=False,
                  append: bool=False,
                  n_jobs: int|None=None,
                  tmpdir: str|None=None,
                  debug: bool=False):
        """
        Transform SLC data to geographic coordinates.

        Parameters
        ----------
        target : str
            The output directory where the results are saved.
        ref : str
            The reference burst data. For multi-path processing only the path with this data is processed.
        records : pd.DataFrame, optional
            The records to use. By default, all records are used.
        epsg : str|int|None, optional
            The EPSG code to use for the output data. By default, the EPSG code is computed automatically for each burst.
        resolution : tuple[int, int], optional
            The resolution to use in meters per pixel in the projected coordinate system.
        remove_topo_phase : bool, optional
            Remove the topographic phase from SLC data for interferometric processing. Set to False
            when creating a DEM from interferograms so the topo phase remains.
        dem_vertical_accuracy : float, optional
            The DEM vertical accuracy in meters.
        alignment_spacing : float, optional
            The alignment spacing in decimal degrees.
        overwrite : bool, optional
            Overwrite existing results and process all bursts.
        append : bool, optional
            Append new burstID processed with the same parameters to the existing results.
        n_jobs : int, optional
            The number of jobs to run in parallel. Default is os.cpu_count().
        tmpdir : str, optional
            Directory for temporary files. Use fast local storage (e.g., '/mnt' on Google Colab)
            for better performance. Default is system temp directory.
        debug : bool, optional
            Whether to print debug information.

        Notes
        -----
        The processing is parallelized using joblib. GMTSAR files are saved in the temp directory.
        """
        from tqdm.auto import tqdm
        import joblib
        import os
        import tempfile
        import shutil
        import sys

        if self.DEM is None:
            raise ValueError('ERROR: DEM is not set. Please create a new instance of S1 with a DEM.')

        if records is None:
            records = self.to_dataframe(ref=ref)

        if epsg is None:
            print('NOTE: EPSG code will be computed automatically for each burst. These projections can be different.')
        elif isinstance(epsg, str) and epsg == 'auto':
            epsgs = self.to_dataframe().centroid.apply(lambda geom: self.get_utm_epsg(geom.y, geom.x)).unique()
            if len(epsgs) > 1:
                raise ValueError(f'ERROR: Multiple UTM zones found: {", ".join(map(str, epsgs))}. Specify the EPSG code manually.')
            epsg = epsgs[0]
            print(f'NOTE: EPSG code is computed automatically for all bursts: {epsg}.')

        # add asserts for the obvious expectations
        assert not os.path.exists(target) or os.path.isdir(target), f'ERROR: target exists but is not a directory'
        if overwrite and os.path.exists(target):
            # remove all previous results and process all bursts
            print(f'NOTE: Removing all previous results and processing all bursts.')
            shutil.rmtree(target)
        # consolidated metadata file zarr.json is saved at the end of the processing
        metafile = os.path.join(target, 'zarr.json')
        assert not os.path.exists(metafile) or os.path.isfile(metafile), f'ERROR: target metadata is not a file'
        # check if the processing is completed
        if os.path.exists(target):
            if not os.path.exists(metafile) or os.path.getsize(metafile) == 0:
                print(f'NOTE: target processing is not completed before. Continuing...')
            elif not append:
                # processing is completed before, nothing to do
                print(f'NOTE: target processing is completed before. Skipping...')
                return
        # remove the consolidated metadata file when appending
        if os.path.exists(metafile):
            os.remove(metafile)

        # Use user-specified tmpdir or fall back to system temp directory
        tmpdir_base = tmpdir if tmpdir is not None else tempfile.gettempdir()

        def process_refrep(bursts, target, tmpdir_base, debug=False):
            with tempfile.TemporaryDirectory(prefix=bursts[0][0][0], dir=tmpdir_base) as tmpdir:             
                #print('working in:', basedir)
                # polarization does not matter for geometry alignment, any polarization reference burst can be used
                # burst format is like ('021_043788_IW1', 'S1_043788_IW1_20230129T033343_VH_DAAA-BURST')
                burst_refs = bursts[0]
                burst_reps = bursts[1]
                # output directory
                fullBurstId = self.fullBurstId(burst_refs[0][-1])
                outdir = os.path.join(target, fullBurstId)
                # check if the directory is already exists and processing completed
                # consolidated metadata file zarr.json is saved at the end of the processing
                metafile = os.path.join(outdir, 'zarr.json')

                # check the specific case when the directory exists and the processing is performed before, completed or not
                if os.path.exists(outdir):
                    # add asserts for the obvious expectations
                    assert os.path.isdir(outdir), f'ERROR: {fullBurstId} exists but is not a directory'
                    assert not os.path.exists(metafile) or os.path.isfile(metafile), f'ERROR: {fullBurstId} metadata is not a file'
                    # check if the processing is completed
                    if os.path.exists(metafile) and os.path.getsize(metafile) > 0:
                        # file exists and is not empty, the burstID successfully processed before
                        return
                    else:
                        # remove the directory when the consolidated metadata file is missing
                        # processing is not completed before, data are corrupted
                        print(f'NOTE: {fullBurstId} directory exists but metadata file is missing. Removing...')
                        shutil.rmtree(outdir)

                # prepare reference burst for all polarizations
                for burst_ref in burst_refs:
                    #print ('burst_ref', burst_ref)
                    self.align_ref(burst_ref[-1], tmpdir, debug=debug)
                #print ('compute_transform')
                # transform is saved in the output directory to be used for the future analysis
                self.compute_transform(outdir, burst_refs[0][-1], basedir=tmpdir, resolution=resolution, scale_factor=1/dem_vertical_accuracy, epsg=epsg)
                # transformation matrix
                transform = self.get_transform(outdir, burst_refs[0][-1])
                # for topo phase calculation
                #print ('compute_transform_inverse')
                
                if remove_topo_phase:
                    # topo in radar coordinate saved in the temp directory for the processing time only
                    self.compute_topo(target, transform, burst_refs[0][-1], basedir=tmpdir)
                    topo = self.get_topo(burst_ref, tmpdir)
                else:
                    # keep topo phase intact (e.g., for DEM creation by interferogram)
                    topo = None

                # use sequential processing as it is well parallelized internally
                #print ('transform_slc')
                for burst_rep in burst_reps + burst_refs:
                    burst_ref = [burst for burst in burst_refs if burst[:2]==burst_rep[:2]][0]
                    #print (burst_rep[-1], '->', burst_ref[-1])
                    # align repeat bursts to the reference burst
                    if burst_rep not in burst_refs:
                        #print ('align_rep', burst_rep, '=>', burst_ref)
                        self.align_rep(burst_rep[-1], burst_ref=burst_refs[0][-1], basedir=tmpdir, degrees=alignment_spacing, debug=debug)
                    # processed bursts saved in the output directory to be used for the future analysis
                    self.transform_slc_int16(outdir, transform, topo, burst_rep[-1], burst_ref[-1], basedir=tmpdir, epsg=epsg)
                #print ()

                # cleanup
                del topo, transform
            # consolidate zarr metadata for the bursts directory
            self.consolidate_metadata(target, burst=burst_rep[-1])

        # get reference and repeat bursts as groups
        refrep_dict = self.get_repref(ref=ref)
        refreps = [v for v in refrep_dict.values()]

        # Default n_jobs to cpu_count() for reliable parallelization (joblib's -1 can undercount on some systems)
        if n_jobs is None:
            n_jobs = os.cpu_count()

        joblib_backend = None if not debug else 'sequential'
        # for Google Colab NetCDF compatibility use threading backend
        if joblib_backend is None and 'google.colab' in sys.modules:
            joblib_backend = 'threading'
        with self.progressbar_joblib(tqdm(desc='Transforming SLC...'.ljust(25), total=len(refreps))) as progress_bar:
            joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(joblib.delayed(process_refrep)(refrep, target, tmpdir_base, debug=debug) for refrep in refreps)

        # consolidate zarr metadata for the resolution directory
        self.consolidate_metadata(target, resolution=resolution)

    def transform_slc_int16(self,
                            outdir: str,
                            transform: xr.Dataset,
                            topo: xr.DataArray | None,
                            burst_rep: str,
                            burst_ref: str,
                            basedir: str,
                            epsg: int,
                            scale: float=2.5e-07
                            ):
        """
        Perform geocoding from radar to geographic coordinates.

        Parameters
        ----------
        burst_rep : str
            The repeat burst name.
        burst_ref : str
            The reference burst name.
        resolution : tuple[int, int]
            The resolution to use.
        scale : float, optional
            The scale to use. Default is 2.5e-07.
        """
        import pandas as pd
        import numpy as np
        import xarray as xr
        import os
        #print(f'transform_slc {burst} {date}')
        # get record
        df = self.get_record(burst_rep)

        # get PRM parameters
        prm_rep = self.PRM(burst_rep, basedir=basedir)
        prm_ref = self.PRM(burst_ref, basedir=basedir)

        #print ('transform_slc', burst_rep, burst_ref, basedir, resolution, epsg, scale)

        # read SLC data
        slc = prm_rep.read_SLC_int()
        # scale as complex values
        slc_complex = scale*(slc.re.astype(np.float32) + 1j*slc.im.astype(np.float32)).rename('data')
        # mask empty borders
        slc_complex = slc_complex\
                .where((slc_complex!=0+0j).sum('a') > 0.8 * slc_complex.a.size)\
                .where((slc_complex!=0+0j).sum('r') > 0.8 * slc_complex.r.size)
        # compute the topo phase
        #print ('burst_rep, burst_ref', burst_rep, burst_ref)
        phase = self.flat_earth_topo_phase(topo, burst_rep, burst_ref, basedir=basedir)
        # reproject as a single complex variable
        complex_proj = self.geocode(transform, slc_complex * np.exp(-1j * phase))
        # unify the order of dimensions
        complex_proj = complex_proj.transpose('y', 'x')
        del phase, slc_complex
        
        # do not apply scale to complex_proj to preserve projection attributes
        data_proj = xr.merge([
                        (complex_proj.real / scale).round().astype(np.int16).where(np.isfinite(complex_proj.real), np.iinfo(np.int16).max).rename('re'),
                        (complex_proj.imag / scale).round().astype(np.int16).where(np.isfinite(complex_proj.imag), np.iinfo(np.int16).max).rename('im')
                        ])
        del complex_proj

        # add PRM attributes
        for name, value in prm_rep.df.itertuples():
            if name not in ['input_file', 'SLC_file', 'led_file']:
                data_proj.attrs[name] = value

        # add TOPS-specific parameters for phase ramp computation (already read above)
        for name, value in prm_rep.read_tops_params().items():
            data_proj.attrs[name] = value

        # add calculated attributes
        BPL, BPR = prm_ref.SAT_baseline(prm_rep).get('B_parallel', 'B_perpendicular')
        # prevent confusing -0.0
        data_proj.attrs['BPR'] = BPR + 0
        # workaround for the hard-coded attribute
        #data_proj.attrs['SLC_scale'] = scale
        
        # add record attributes
        for _, row in df.reset_index().iterrows():
            # reverse the items order within each row
            for name, value in ((n, v) for n, v in list(row.items())[::-1] if n not in ['orbit', 'path']):
                if isinstance(value, (pd.Timestamp, np.datetime64)):
                    value = pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S')
                if name == 'geometry':
                    value = value.wkt
                data_proj.attrs[name] = value

        # add georeference attributes
        data_proj = self.spatial_ref(data_proj, epsg)
        data_proj.attrs['spatial_ref'] = data_proj.spatial_ref.attrs['spatial_ref']
        # remove spatial_ref variable to limit zarray files count
        data_proj = data_proj.drop_vars('spatial_ref')

        # add storage specific attributes to variables only
        for varname in ['re', 'im']:
            data_proj[varname].attrs['scale_factor'] = scale
            data_proj[varname].attrs['add_offset'] = 0
            data_proj[varname].attrs['_FillValue'] = np.iinfo(np.int16).max

        # use transfrom coordinates
        data_proj = data_proj.drop_vars(['x','y'])
        # use a single chunk per burst for efficient storage
        shape = data_proj.re.shape
        encoding = {var: {'chunks': shape} for var in ['re', 'im']}
        data_proj.to_zarr(
            store=os.path.join(outdir, burst_rep),
            mode='w',
            zarr_format=3,
            consolidated=True,
            encoding=encoding
        )
        slc.close()
        del data_proj, slc

        for ext in ['SLC', 'LED', 'PRM']:
            self.get_burstfile(burst_rep, basedir, ext=ext, clean=True)
