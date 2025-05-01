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

    def consolidate_metadata(self, workdir: str, resolution: tuple[int, int]=(20, 5), burst: str=None):
        """
        Consolidate metadata for a given resolution and burst.

        Parameters
        ----------
        workdir : str
            The work directory.
        resolution : tuple[int, int]
            The resolution to use.
        burst : str
            The burst to use.
        """
        import zarr
        import os
        root_dir = os.path.join(workdir, f'{resolution[0]}x{resolution[1]}')
        if burst:
            root_dir = os.path.join(root_dir, self.fullBurstId(burst))
        #print ('root_dir', root_dir)
        root_store = zarr.storage.LocalStore(root_dir)
        root_group = zarr.group(store=root_store, overwrite=False)
        zarr.consolidate_metadata(root_store)

    def transform(self,
                  workdir: str,
                  ref: str,
                  records: pd.DataFrame|None=None,
                  epsg: str|int|None='auto',
                  resolution: tuple[int, int]=(20, 5),
                  topo_correction: bool = True,
                  dem_vertical_accuracy: float=0.5,
                  alignment_spacing: float=12.0/3600,
                  debug: bool=False):
        import dask
        from tqdm.auto import tqdm
        import joblib
        import os
        import tempfile
        from distributed import get_client
        import gc

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

        def process_refrep(bursts, debug=False):
            with tempfile.TemporaryDirectory(prefix=bursts[0][0][0]) as tmpdir:             
                #print('working in:', basedir)
                # polarization does not matter for geometry alignment, any polarization reference burst can be used
                # burst format is like ('021_043788_IW1', 'S1_043788_IW1_20230129T033343_VH_DAAA-BURST')
                burst_refs = bursts[0]
                burst_reps = bursts[1]
                # output directory
                outdir = os.path.join(workdir, f'{resolution[0]}x{resolution[1]}', self.fullBurstId(burst_refs[0][-1]))

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
                
                if topo_correction:
                    # topo in radar coordinate saved in the temp directory for the processing time only
                    self.compute_topo(workdir, transform, burst_refs[0][-1], basedir=tmpdir)
                    topo = self.get_topo(burst_ref, tmpdir)
                else:
                    # topographic phase is not needed when topo correction is not applied
                    topo = 0

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
            self.consolidate_metadata(workdir, resolution=resolution, burst=burst_rep[-1])

        # get reference and repeat bursts as groups
        refrep_dict = self.get_repref(ref=ref)
        refreps = [v for v in refrep_dict.values()]
        
        # restart Dask cluster to avoid issues with the previous workers
        client = get_client()
        client.restart(wait_for_workers=True, timeout=60)
        gc.collect()
        for refrep in tqdm(refreps, desc='Transforming SLC'):
            # single worker is used to avoid Dask issues
            joblib.Parallel(n_jobs=1, backend='threading')(
                [joblib.delayed(process_refrep)(refrep, debug=debug)]
            )
            # cleanup workers and the main process
            client.run(lambda: __import__('gc').collect())
            gc.collect()
        # consolidate zarr metadata for the resolution directory
        self.consolidate_metadata(workdir, resolution=resolution)

    def transform_slc_int16(self,
                            outdir: str,
                            transform: xr.Dataset,
                            topo: xr.DataArray,
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
        import dask
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

        # transfer attributes to the output variables
        #print ('data_proj.attrs', data_proj.attrs)
        data_proj.im.attrs = data_proj.attrs
        data_proj.re.attrs = data_proj.attrs

        # add storage specific attributes
        for varname in ['re', 'im']:
            data_proj[varname].attrs['scale_factor'] = scale
            data_proj[varname].attrs['add_offset'] = 0
            data_proj[varname].attrs['_FillValue'] = np.iinfo(np.int16).max
        
        encoding_vars = {var: self.get_encoding_zarr(chunks=(data_proj.x.size,),
                                                     dtype=data_proj[var].dtype,
                                                     shuffle='noshuffle'
                                                     ) for var in data_proj.data_vars}
        #print ('encoding_vars', encoding_vars)
        # use transfrom coordinates
        data_proj = data_proj.drop_vars(['x','y'])
        data_proj.to_zarr(
            store=os.path.join(outdir, burst_rep),
            encoding=encoding_vars,
            mode='w',
            consolidated=True
        )
        slc.close()
        del data_proj, slc

        for ext in ['SLC', 'LED', 'PRM']:                
            self.get_burstfile(burst_rep, ext, clean=True)
