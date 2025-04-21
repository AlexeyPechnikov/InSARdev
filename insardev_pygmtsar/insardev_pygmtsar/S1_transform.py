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

    def transform(self, ref: str, records: pd.DataFrame|None=None, degrees: float=12.0/3600,
            resolution: tuple[int, int]=(20, 5), epsg: str|int|None='auto', n_jobs: int=-1, debug: bool=False):
        from tqdm.auto import tqdm
        import joblib
        from tqdm.auto import tqdm
        import os

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

        if n_jobs is None or debug == True:
            print ('Note: sequential joblib processing is applied when n_jobs is None or debug is True.')
            joblib_backend = 'sequential'
        else:
            joblib_backend = None

        def process_refrep(bursts, debug=False):
            # polarization does not matter for geometry alignment, any polarization reference burst can be used
            # burst format is like ('021_043788_IW1', 'VH', 'S1_043788_IW1_20230129T033343_VH_DAAA-BURST')
            burst_refs = bursts[0]
            burst_reps = bursts[1]

            prefix = self.get_prefix(burst_refs[0][-1])
            path_prefix = os.path.join(self.basedir, prefix)
            if not os.path.isdir(path_prefix):
                os.makedirs(path_prefix)
                
            #print (ref, burst_refs, '->', burst_reps)
            # align reference burst for all polarizations
            joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)\
                (joblib.delayed(self.align_ref)(burst_ref[-1], debug=debug) for burst_ref in burst_refs)
            # align repeat bursts for all polarizations using any reference burst
            joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)\
                (joblib.delayed(self.align_rep)(burst_rep[-1], burst_ref=burst_refs[0][-1], degrees=degrees, debug=debug) for burst_rep in burst_reps)

            #print ('compute_transform')
            self.compute_transform(burst_refs[0][-1], resolution=resolution, epsg=epsg)
            # for topo phase calculation
            #print ('compute_transform_inverse')
            self.compute_transform_inverse(burst_refs[0][-1], resolution=resolution)
            # compute the transform data for the same polarization reference burst
            # this allows to easily detect when topo phase is not applicable
            # use sequential processing as it is well parallelized internally
            #print ('transform_slc')
            for burst_rep in burst_reps + burst_refs:
                burst_ref = [burst for burst in burst_refs if burst[:2]==burst_rep[:2]][0]
                #print (burst_rep[-1], '->', burst_ref[-1])
                self.transform_slc(burst_rep[-1], burst_ref[-1], resolution=resolution)
            
            # cleanup
            filename = self.get_filename(burst_refs[0][-1], f'transform_inverse.{resolution[0]}x{resolution[1]}')
            if os.path.exists(filename):
                os.remove(filename)

        # get reference and repeat bursts as groups
        refrep_dict = self.get_repref(ref=ref)
        refreps = [v for v in refrep_dict.values()]
        for refrep in tqdm(refreps, desc="Transforming SLC"):
            process_refrep(refrep, debug=debug)

    def transform_slc(self, burst_rep: str, burst_ref: str, resolution: tuple[int, int], scale: float=2.5e-07, interactive: bool=False):
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
        interactive : bool, optional
            If True, the computation will be performed interactively and the result will be returned as a delayed object.
            Default is False.
        """
        import pandas as pd
        import numpy as np
        import xarray as xr
        import dask

        phase = self.topo_phase(burst_rep, burst_ref, resolution)
        #dates = pd.DatetimeIndex(data.date).strftime('%Y-%m-%d')

        #print(f'transform_slc {burst} {date}')
        # get record
        df = self.get_record(burst_rep)

        # get PRM parameters
        prm_rep = self.PRM(burst_rep)
        prm_ref = self.PRM(burst_ref)

        # read SLC data
        slc = prm_rep.read_SLC_int()
        # scale as complex values, zero in np.int16 type means NODATA
        slc_complex = scale*(slc.re.astype(np.float32) + 1j*slc.im.astype(np.float32)).where(slc.re != 0).rename('data')
        # zero in np.int16 type means NODATA
        #slc_complex = slc_complex.where(slc_complex != 0)
        del slc
        # reproject as a single complex variable
        complex_proj = self.geocode(burst_rep, slc_complex * np.exp(-1j * phase), resolution)
        
        # do not apply scale to complex_proj to preserve projection attributes
        data_proj = self.spatial_ref(
                         xr.merge([
                            (complex_proj.real / scale).round().astype(np.int16).rename('re'),
                            (complex_proj.imag / scale).round().astype(np.int16).rename('im')
                         ]),
                         complex_proj
                    )
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
        data_proj.attrs['SLC_scale'] = scale
        
        # add record attributes
        for _, row in df.reset_index().iterrows():
            # reverse the items order within each row
            for name, value in ((n, v) for n, v in list(row.items())[::-1] if n not in ['orbit', 'path']):
                if isinstance(value, (pd.Timestamp, np.datetime64)):
                    value = pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S')
                if name == 'geometry':
                    value = value.wkt
                data_proj.attrs[name] = value

        if interactive:
            return data_proj

        dy = int(data_proj.y.diff('y').round(0).values[0])
        dx = int(data_proj.x.diff('x').round(0).values[0])
        filename = self.get_burstfile(burst_rep, ext=f'{dy}x{dx}.nc', clean=True)
        #encoding = {'data': self.get_compression(data_proj.shape)}
        encoding = {varname: self.get_compression(data_proj[varname].shape) for varname in data_proj.data_vars}
        #print ('encoding', encoding)
        data_proj.to_netcdf(filename,
                            encoding=encoding,
                            engine=self.netcdf_engine_write,
                            format=self.netcdf_format)

        for ext in ['SLC', 'LED', 'PRM']:                
            self.get_burstfile(burst_rep, ext, clean=True)
