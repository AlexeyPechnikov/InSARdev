# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_align import S1_align
from insardev_toolkit import tqdm_dask

class S1_geocode(S1_align):

    def geocode(self, records=None, dem='auto', resolution=(15, 5), epsg='auto'):
        """
        Perform geocoding from radar to projected coordinates.

        Parameters
        ----------
        records : pandas.DataFrame, optional
            The records to process. If None, all records will be processed.
        dem : str, optional
            The DEM to use. If 'auto', the DEM will be computed.
            Default is 'auto'.
        resolution : tuple, optional
            The resolution in the azimuth and range direction.
            Default is (15, 5).
        epsg : str, optional
            The EPSG code to use. If 'auto', the EPSG code will be computed.
            Default is 'auto'.
        """
        import warnings
        # suppress Dask warning "RuntimeWarning: All-NaN slice encountered"
        warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', module='dask')
        warnings.filterwarnings('ignore', module='dask.core')
        from tqdm.auto import tqdm
        #import joblib

        #print ('bursts', bursts)
        def burst_geocode(burst_ref, dem, resolution, epsg):
            #print ('burst', burst)
            self.compute_trans(burst_ref, dem=dem, resolution=resolution, epsg=epsg)
            # do not save the grid
            #trans_inv = self.compute_trans_inv(burst, interactive=True)
            #topo = self.get_topo(burst, trans_inv)
            #self.compute_trans_slc(burst, topo=topo)
            # save the grid (2 times faster)
            self.compute_trans_inv(burst_ref)

        bursts_ref = self.get_records_ref(records).index.get_level_values(2)
        # use sequential processing as geocoding is well parallelized internally
        for burst_ref in tqdm(bursts_ref, desc='Geocoding transform'):
            burst_geocode(burst_ref, dem=dem, resolution=resolution, epsg=epsg)

    def transform(self, records=None, clean=True):
        """
        Perform geocoding from radar to projected coordinates.

        Parameters
        ----------
        records : pandas.DataFrame, optional
            The records to process. If None, all records will be processed.
        clean : bool, optional
            If True, the source SLC, LED and PRM files will be deleted.
            Default is True.
        """
        from tqdm.auto import tqdm
        #import joblib

        # # use parallel processing for simple bursts conversion
        # with self.tqdm_joblib(tqdm(desc='Geocoding bursts', total=len(bursts)*len(dates))) as progress_bar:
        #    joblib.Parallel(n_jobs=-1, backend=None)\
        #        (joblib.delayed(self.transform_slc)(burst, date) for burst in bursts for date in dates)

        #bursts_rep = self.get_records_rep(records).index.get_level_values(2)
        #bursts = self.df.index.get_level_values(2)
        rep_ref_dict = self.get_records_rep_ref(records)
        # process as repeat as reference bursts
        rep_ref_dict = rep_ref_dict | dict(zip(rep_ref_dict.values(), rep_ref_dict.values()))
        for burst_rep, burst_ref in tqdm(rep_ref_dict.items(), desc='Geocoding bursts'):
            self.transform_slc(burst_rep, burst_ref, clean=clean)

    def transform_slc(self, burst_rep, burst_ref, topo='auto', phase='auto', scale=2.5e-07, clean=True, interactive=False):
        """
        Perform geocoding from radar to geographic coordinates.

        Parameters
        ----------
        burst_rep : str
            The repeat burst name.
        burst_ref : str
            The reference burst name.
        topo : str, optional
            The topographic data to use. If 'auto', the topographic data will be computed.
            Default is 'auto'.
        phase : str, optional
            The topographic phase to use. If 'auto', the topographic phase will be computed.
            Default is 'auto'.
        scale : float, optional
            The scale to use. Default is 2.5e-07.
        clean : bool, optional
            If True, the source SLC, LED and PRM files will be deleted.
            Default is True.
        interactive : bool, optional
            If True, the computation will be performed interactively and the result will be returned as a delayed object.
            Default is False.
        """
        import pandas as pd
        import numpy as np
        import xarray as xr
        import dask

        if isinstance(phase, str) and phase == 'auto':
            phase = self.topo_phase(burst_rep, burst_ref, topo=topo)
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
        complex_proj = self.project(burst_rep, slc_complex * np.exp(-1j * phase))
        
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

        filename = self.get_burstfile(burst_rep, clean=True)
        #encoding = {'data': self._compression(data_proj.shape)}
        encoding = {varname: self._compression(data_proj[varname].shape) for varname in data_proj.data_vars}
        #print ('encoding', encoding)
        data_proj.to_netcdf(filename,
                            encoding=encoding,
                            engine=self.netcdf_engine_write,
                            format=self.netcdf_format)

        if clean:
            for ext in ['SLC', 'LED', 'PRM']:                
                self.get_burstfile(burst_rep, ext, clean=True)

    def project(self, burst, data, trans='auto'):
        """
        Perform geocoding from radar to geographic coordinates.

        Parameters
        ----------
        grid : xarray.DataArray
            Grid(s) representing the interferogram(s) in radar coordinates.
        trans : xarray.DataArray
            Geocoding transform matrix in radar coordinates.

        Returns
        -------
        xarray.DataArray
            The inverse geocoded grid(s) in geographic coordinates.

        Examples
        --------
        Geocode 3D unwrapped phase grid stack:
        unwraps_ll = stack.intf_ra2ll(stack.open_grids(pairs, 'unwrap'))
        # or use "geocode" option for open_grids() instead:
        unwraps_ll = stack.open_grids(pairs, 'unwrap', geocode=True)
        """
        import dask
        import dask.array as da
        import xarray as xr
        import numpy as np
        import warnings
        warnings.filterwarnings('ignore')

        if isinstance(trans, str) and trans == 'auto':
            trans = self.get_trans(burst)

        # use outer data variable
        def trans_block(trans_block_azi, trans_block_rng):
            from scipy.interpolate import RegularGridInterpolator

            coord_a = data.a
            coord_r = data.r

            # check if the data block exists
            if not trans_block_azi.size:
                return np.nan * np.zeros(trans_block_azi.shape, dtype=data.dtype)

            # use trans table subset
            azis = trans_block_azi.ravel()
            rngs = trans_block_rng.ravel()
            points = np.column_stack([azis, rngs])
            
            # calculate trans grid subset extent
            amin, amax = np.nanmin(azis), np.nanmax(azis)
            rmin, rmax = np.nanmin(rngs), np.nanmax(rngs)
            coord_a = coord_a[(coord_a>amin-1)&(coord_a<amax+1)]
            coord_r = coord_r[(coord_r>rmin-1)&(coord_r<rmax+1)]
            del amin, amax, rmin, rmax
            # when no valid pixels for the processing
            if coord_a.size == 0 or coord_r.size == 0:
                del coord_a, coord_r, points
                return np.nan * np.zeros(trans_block_azi.shape, dtype=data.dtype)

            data_block = data.sel(a=coord_a, r=coord_r).compute(n_workers=1)
            values = data_block.data
            del data_block

            interp = RegularGridInterpolator((coord_a, coord_r), values, method='nearest', bounds_error=False)
            grid_proj = interp(points).reshape(trans_block_azi.shape)
            del coord_a, coord_r, points, values
            return grid_proj

        out = da.blockwise(
            trans_block,
            'yx',
            trans.azi, 'yx',
            trans.rng, 'yx',
            dtype=data.dtype
        )

        da = xr.DataArray(out, trans.coords).rename(data.name)
        del out
        return self.spatial_ref(da, trans)

    @staticmethod
    def get_utm_epsg(lat, lon):
        zone_num = int((lon + 180) // 6) + 1
        if lat >= 0:
            return 32600 + zone_num
        else:
            return 32700 + zone_num
    
    @staticmethod
    def proj(yy, xx, to_epsg, from_epsg):
        from pyproj import CRS, Transformer
        from_crs = CRS.from_epsg(from_epsg)
        to_crs = CRS.from_epsg(to_epsg)
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        xx_new, yy_new = transformer.transform(xx, yy)
        del transformer, from_crs, to_crs
        return yy_new, xx_new

    def get_trans(self, burst):
        """
        Retrieve the transform data.

        This function opens a NetCDF dataset, which contains data mapping from radar
        coordinates to geographical coordinates (from azimuth-range to latitude-longitude domain).

        Parameters
        ----------
        burst : str
            The burst name.

        Returns
        -------
        xarray.Dataset or list of xarray.Dataset
            An xarray dataset(s) with the transform data.

        Examples
        --------
        Get the inverse transform data:
        get_trans()
        """
        import xarray as xr
        filename = self.get_filename(burst, 'trans')
        return xr.open_dataset(filename,
                               engine=self.netcdf_engine_read,
                               format=self.netcdf_format,
                               chunks=self.chunksize)
        #return self.open_cube(filename)
        #.dropna(dim='y', how='all')
        #.dropna(dim='x', how='all')

    def compute_trans(self, burst_ref, dem='auto', resolution=(10, 2.5), epsg='auto', interactive=False):
        """
        Retrieve or calculate the transform data. This transform data is then saved as
        a NetCDF file for future use.

        This function generates data mapping from geographical coordinates to radar coordinates (azimuth-range domain).
        The function uses a Digital Elevation Model (DEM) to derive the geographical coordinates, and then uses the
        `SAT_llt2rat` function to map these to radar coordinates.

        Parameters
        ----------
        burst_ref : str
            The reference burst name.
        dem : str, optional
            The DEM to use. If 'auto', the DEM will be computed.
            Default is 'auto'.
        resolution : tuple, optional
            The resolution in the azimuth and range direction.
            Default is (10, 2.5).

        Returns
        -------
        None

        Examples
        --------
        Calculate and get the transform data:
        >>> Stack.compute_trans_dat(1)
        """
        import dask
        import xarray as xr
        import numpy as np
        import os
        from tqdm.auto import tqdm
        import joblib
        import cv2
        import warnings
        warnings.filterwarnings('ignore')

        # range, azimuth, elevation(ref to radius in PRM), look_E, look_N, look_U
        llt2ratlook_map = {0: 'rng', 1: 'azi', 2: 'ele', 3: 'look_E', 4: 'look_N', 5: 'look_U'}
        #llt2ratlook_map = {0: 'rng', 1: 'azi', 2: 'ele'}

        prm = self.PRM(burst_ref)
        def SAT_llt2ratlook(lats, lons, zs):
            # for binary=True values outside of the scene missed and the array is not complete
            # 4th and 5th coordinates are the same as input lat, lon
            #print (f'SAT_llt2rat: lats={lats}, lons={lons}, zs={zs} ({lats.shape}, {lons.shape}, {zs.shape})')
            coords3d = np.column_stack([lons, lats, np.nan_to_num(zs)])
            rae = prm.SAT_llt2rat(coords3d, precise=1, binary=False).astype(np.float32).reshape(zs.size, 5)[...,:3]
            #rae[~np.isfinite(zs), :] = np.nan
            #return rae
            # look_E look_N look_U
            look = prm.SAT_look(coords3d, binary=True).astype(np.float32).reshape(zs.size, 6)[...,3:]
            out = np.concatenate([rae, look], axis=-1)
            del rae, look
            out[~np.isfinite(zs), :] = np.nan
            return out

        # exclude latitude and longitude columns as redundant
        def trans_block(ys, xs, coarsen, epsg, amin=-np.inf, amax=np.inf, rmin=-np.inf, rmax=np.inf, filename=None):
            # disable "distributed.utils_perf - WARNING - full garbage collections ..."
            try:
                from dask.distributed import utils_perf
                utils_perf.disable_gc_diagnosis()
            except ImportError:
                from distributed.gc import disable_gc_diagnosis
                disable_gc_diagnosis()
            import warnings
            warnings.filterwarnings('ignore')

            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            lats, lons = self.proj(yy, xx, from_epsg=epsg, to_epsg=4326)

            dlat = dem.lat.diff('lat')[0]
            dlon = dem.lon.diff('lon')[0]
            elev = dem.sel(lat=slice(np.nanmin(lats)-dlat, np.nanmax(lats)+dlat), lon=slice(np.nanmin(lons)-dlon, np.nanmax(lons)+dlon))\
                      .compute(n_workers=1)

            if not elev.size:
                del lats, lons, elev
                return np.nan * np.zeros((6, ys.size, xs.size), np.float32)

            #print ('topo.shape', topo.shape, 'lats.size', lats.size, 'lons', lons.size)
            if np.isfinite(amin):
                # check if the elev block is empty or not
                lts = elev.lat.values
                lls = elev.lon.values
                border_lts = np.concatenate([lts, lts, np.repeat(lts[0], lls.size), np.repeat(lts[-1], lls.size)])                
                border_lls = np.concatenate([np.repeat(lls[0], lts.size), np.repeat(lls[-1], lts.size), lls, lls])
                border_zs  = np.concatenate([elev.values[:,0], elev.values[:,-1], elev.values[0,:], elev.values[-1,:]])
                rae = SAT_llt2ratlook(border_lts, border_lls, border_zs)[...,:3]
                del lts, lls, border_lts, border_lls, border_zs
                # this mask does not work for a single chunk
                #mask = (rae[:,0]>=rmin) & (rae[:,0]<=rmax) & (rae[:,1]>=amin) & (rae[:,1]<=amax)
                invalid_mask = ((rae[:,0]<rmin) | (rmax<rae[:,0])) & ((rae[:,1]<amin) | (amax<rae[:,1]))
                del rae
                valid_pixels = invalid_mask[~invalid_mask].size > 0
                del invalid_mask
            else:
                # continue the processing without empty block check
                valid_pixels = True

            if not valid_pixels:
                del lats, lons, elev
                return np.nan * np.zeros((6, ys.size, xs.size), np.float32)

            # apply coarsen when needed
            lats_coarsen = lats[::coarsen[0], ::coarsen[1]]
            lons_coarsen = lons[::coarsen[0], ::coarsen[1]]
            elev_coarsen = elev.interp({'lat': xr.DataArray(lats_coarsen), 'lon': xr.DataArray(lons_coarsen)}).values
            shape = elev_coarsen.shape
            del elev

            # compute 3D radar coordinates for all the geographical 3D points
            rae = SAT_llt2ratlook(lats_coarsen.astype(np.float32).ravel(),
                                  lons_coarsen.astype(np.float32).ravel(),
                                  elev_coarsen.astype(np.float32).ravel())
            del elev_coarsen, lats_coarsen, lons_coarsen

            # mask invalid values for better compression
            # extend for interpolation on boundaries
            mask = (rae[...,0]>=rmin - 2*coarsen[1]) & (rae[...,0]<=rmax + 2*coarsen[1]) \
                 & (rae[...,1]>=amin - 2*coarsen[0]) & (rae[...,1]<=amax + 2*coarsen[0])
            rae[~mask] = np.nan
            del mask
            rae_coarsen = rae.reshape(shape[0], shape[1], -1)

            if coarsen[0] > 1 or coarsen[1] > 1:
                src_grid_y, src_grid_x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            
                src_y_coords = np.interp(yy, ys[::coarsen[0]], np.arange(shape[0])).astype(np.float32)
                src_x_coords = np.interp(xx, xs[::coarsen[1]], np.arange(shape[1])).astype(np.float32)

                rae = np.stack([
                    cv2.remap(
                        rae_coarsen[...,i],
                        src_x_coords,
                        src_y_coords,
                        interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT
                    )
                    for i in range(6)
                ], axis=0)   
            else:
                rae = rae_coarsen.transpose(2,0,1)
            del rae_coarsen

            return rae.astype(np.float32)

        def trans_blocks(ys, xs, chunksize):
            #print ('ys', ys, 'xs', xs, 'sizes', ys.size, xs.size)
            # split to equal chunks and rest
            ys_blocks = np.array_split(ys, np.arange(0, ys.size, chunksize)[1:])
            xs_blocks = np.array_split(xs, np.arange(0, xs.size, chunksize)[1:])
            #print ('ys_blocks.size', len(ys_blocks), 'xs_blocks.size', len(xs_blocks))
            #print ('ys_blocks[0]', xs_blocks[0])
    
            blocks_total = []
            for ys_block in ys_blocks:
                blocks = []
                for xs_block in xs_blocks:
                    block = dask.array.from_delayed(
                        dask.delayed(trans_block)(ys_block, xs_block, coarsen, epsg, **borders),
                        shape=(6, ys_block.size, xs_block.size),
                        dtype=np.float32
                    )
                    blocks.append(block)
                    del block
                blocks_total.append(blocks)
                del blocks
            rae = dask.array.block(blocks_total)
            del blocks_total, ys_blocks, xs_blocks
            # transform to separate variables
            trans = xr.Dataset({val: xr.DataArray(rae[key].round(4), coords={'y': ys,'x': xs})
                              for (key, val) in llt2ratlook_map.items()})
            del rae
            return trans

        if isinstance(dem, str) and dem == 'auto':
            # do not use coordinate names lat,lon because the output grid saved as (lon,lon) in this case...
            dem = self.get_dem()

        if isinstance(epsg, str) and epsg == 'auto':
            epsg = self.get_utm_epsg(dem.lat.mean(), dem.lon.mean())

        a_max, r_max = prm.bounds()
        borders = {'amin': 0, 'amax': a_max, 'rmin': 0, 'rmax': r_max}
        #print ('borders', borders)
        
        # check DEM corners
        dem_corners = dem[::dem.lat.size-1, ::dem.lon.size-1].compute()
        lats, lons = xr.broadcast(dem_corners.lat, dem_corners.lon)
        yy, xx = self.proj(lats, lons, from_epsg=4326, to_epsg=epsg)
        dem_y_min = np.min(resolution[0] * ((yy/resolution[0]).round() + 0.5))
        dem_y_max = np.max(resolution[0] * ((yy/resolution[0]).round() - 0.5))
        dem_x_min = np.min(resolution[1] * ((xx/resolution[1]).round() + 0.5))
        dem_x_max = np.max(resolution[1] * ((xx/resolution[1]).round() - 0.5))
        #print ('dem_y_min', dem_y_min, 'dem_y_max', dem_y_max, 'dem_x_min', dem_x_min, 'dem_x_max', dem_x_max)
        ys = np.arange(dem_y_min, dem_y_max + resolution[0], resolution[0])
        xs = np.arange(dem_x_min, dem_x_max + resolution[1], resolution[1])
        #print ('ys', ys, 'xs', xs, 'sizes', ys.size, xs.size)
        
        dem_spacing = ((dem_y_max - dem_y_min)/dem.lat.size, (dem_x_max - dem_x_min)/dem.lon.size)
        #print (f'DEM spacing: {dem_spacing}')

        # transform user-specified grid resolution to proccoarsen factor
        coarsen = (
            max(1, int(np.round(dem_spacing[0]/resolution[0]))),
            max(1, int(np.round(dem_spacing[1]/resolution[1])))
        )
        #print ('coarsen', coarsen)
        
        # estimate the radar extent on decimated grid
        decimation = 10
        trans_est = trans_blocks(ys[::decimation], xs[::decimation], self.netcdf_chunksize).compute()
        trans_est = trans_est.ele.dropna(dim='y', how='all').dropna(dim='x', how='all')
        y_min = trans_est.y.min().item() - 2*decimation*resolution[0]*coarsen[0]
        y_max = trans_est.y.max().item() + 2*decimation*resolution[0]*coarsen[0]
        x_min = trans_est.x.min().item() - 2*decimation*resolution[1]*coarsen[1]
        x_max = trans_est.x.max().item() + 2*decimation*resolution[1]*coarsen[1]
        ys = ys[(ys>=y_min)&(ys<=y_max)]
        xs = xs[(xs>=x_min)&(xs<=x_max)]
        #print ('ys', ys, 'xs', xs, 'sizes', ys.size, xs.size)
        #print ('ys[0]', ys[0], 'ys[-1]', ys[-1], 'xs[0]', xs[0], 'xs[-1]', xs[-1])
        #print ('y pixels offset', (dem_y_min-ys[0])/resolution[0], 'x pixels offset', (dem_x_min-xs[0])/resolution[1])
        del trans_est

        # compute for the radar extent
        trans = trans_blocks(ys, xs, self.chunksize)

        if interactive:
            return trans

        filename = self.get_filename(burst_ref, 'trans', clean=True)
        encoding = {varname: self._compression(trans[varname].shape) for varname in trans.data_vars}
        self.spatial_ref(trans, epsg).to_netcdf(filename,
                        encoding=encoding,
                        engine=self.netcdf_engine_write,
                        format=self.netcdf_format)
        del trans

    def get_trans_inv(self, burst):
        """
        Retrieve the inverse transform data.

        This function opens a NetCDF dataset, which contains data mapping from radar
        coordinates to geographical coordinates (from azimuth-range to latitude-longitude domain).

        Parameters
        ----------
        burst : str
            The burst name.

        Returns
        -------
        xarray.Dataset
            An xarray dataset with the transform data.

        Examples
        --------
        Get the inverse transform data:
        get_trans_inv()
        """
        import xarray as xr
        filename = self.get_filename(burst, 'trans_inv')
        return xr.open_dataset(filename,
                               engine=self.netcdf_engine_read,
                               format=self.netcdf_format,
                               chunks=self.chunksize)

    def compute_trans_inv(self, burst_ref, trans='auto', interactive=False):
        """
        Retrieve or calculate the transform data. This transform data is then saved as
            a NetCDF file for future use.

            This function generates data mapping from radar coordinates to geographical coordinates.
            The function uses the direct transform data.

        Parameters
        ----------
        burst_ref : str
            The reference burst name.
        trans : str, optional
            The transform data to use. If 'auto', the transform data will be computed.
            Default is 'auto'.
        interactive : bool, optional
            If True, the computation will be performed interactively and the result will be returned as a delayed object.
            Default is False.

        Note
        ----
        This function operates on the 'trans' grid using NetCDF chunks (specified by 'netcdf_chunksize') rather than
        larger processing chunks. This approach is effective due to on-the-fly index creation for the NetCDF chunks.

        """
        import dask
        import xarray as xr
        import numpy as np
        import os
        import warnings
        warnings.filterwarnings('ignore')

        def trans_inv_block(azis, rngs, tolerance, chunksize):
            from scipy.spatial import cKDTree
            # disable "distributed.utils_perf - WARNING - full garbage collections ..."
            try:
                from dask.distributed import utils_perf
                utils_perf.disable_gc_diagnosis()
            except ImportError:
                from distributed.gc import disable_gc_diagnosis
                disable_gc_diagnosis()
            import warnings
            warnings.filterwarnings('ignore')

            # required one delta around for nearest interpolation and two for linear
            azis_min = azis.min() - 1
            azis_max = azis.max() + 1
            rngs_min = rngs.min() - 1
            rngs_max = rngs.max() + 1
            #print ('azis_min', azis_min, 'azis_max', azis_max)

            # define valid coordinate blocks 
            block_mask = ((trans_amin<=azis_max)&(trans_amax>=azis_min)&(trans_rmin<=rngs_max)&(trans_rmax>=rngs_min)).values
            block_azi, block_rng = trans_amin.shape
            blocks_ys, blocks_xs = np.meshgrid(range(block_azi), range(block_rng), indexing='ij')
            #assert 0, f'blocks_ys, blocks_xs: {blocks_ys[block_mask]}, {blocks_xs[block_mask]}'
            # extract valid coordinates from the defined blocks
            blocks_trans = []
            blocks_lt = []
            blocks_ll = []
            for block_y, block_x in zip(blocks_ys[block_mask], blocks_xs[block_mask]):
                # coordinates
                block_lt, block_ll = [block.ravel() for block in np.meshgrid(lt_blocks[block_y], ll_blocks[block_x], indexing='ij')]
                # variables
                block_trans = trans.isel(y=slice(chunksize*block_y,chunksize*(block_y+1)),
                                         x=slice(chunksize*block_x,chunksize*(block_x+1)))[['azi', 'rng', 'ele']]\
                                   .compute(n_workers=1).to_array().values.reshape(3,-1)
                # select valuable coordinates only
                mask = (block_trans[0,:]>=azis_min)&(block_trans[0,:]<=azis_max)&\
                       (block_trans[1,:]>=rngs_min)&(block_trans[1,:]<=rngs_max)
                # ignore block without valid pixels
                if mask[mask].size > 0:
                    # append valid pixels to accumulators
                    blocks_lt.append(block_lt[mask])
                    blocks_ll.append(block_ll[mask])
                    blocks_trans.append(block_trans[:,mask])
                del block_lt, block_ll, block_trans, mask
            del block_mask, block_azi, block_rng, blocks_ys, blocks_xs

            if len(blocks_lt) == 0:
                # this case is possible when DEM is incomplete, and it is not an error
                return np.nan * np.zeros((3, azis.size, rngs.size), np.float32)

            # TEST
            #return np.nan * np.zeros((3, azis.size, rngs.size), np.float32)

            # valid coordinates
            block_lt = np.concatenate(blocks_lt)
            block_ll = np.concatenate(blocks_ll)
            block_trans = np.concatenate(blocks_trans, axis=1)
            del blocks_lt, blocks_ll, blocks_trans

            # perform index search on radar coordinate grid for the nearest geographic coordinates grid pixel
            grid_azi, grid_rng = np.meshgrid(azis, rngs, indexing='ij')
            tree = cKDTree(np.column_stack([block_trans[0], block_trans[1]]), compact_nodes=False, balanced_tree=False)
            distances, indices = tree.query(np.column_stack([grid_azi.ravel(), grid_rng.ravel()]), k=1, workers=1)
            del grid_azi, grid_rng, tree, cKDTree

            # take the nearest pixels coordinates and elevation
            # the only one index search is required to define all the output variables
            grid_lt = block_lt[indices]
            grid_lt[distances>tolerance] = np.nan
            del block_lt
            grid_ll = block_ll[indices]
            grid_ll[distances>tolerance] = np.nan
            del block_ll
            grid_ele = block_trans[2][indices]
            grid_ele[distances>tolerance] = np.nan
            #print ('distance range', distances.min().round(2), distances.max().round(2))
            #assert distances.max() < 2, f'Unexpectedly large distance between radar and geographic coordinate grid pixels (>=2): {distances.max()}'
            del block_trans, indices, distances

            # pack all the outputs into one 3D array
            return np.asarray([grid_lt, grid_ll, grid_ele]).reshape((3, azis.size, rngs.size))

        if isinstance(trans, str) and trans == 'auto':
            # trans.dat - file generated by llt_grid2rat (r a topo lon lat)"
            trans = self.get_trans(burst_ref)

        # calculate indices on the fly
        trans_blocks = trans[['azi', 'rng']].coarsen(y=self.netcdf_chunksize, x=self.netcdf_chunksize, boundary='pad')
        #block_min, block_max = dask.compute(trans_blocks.min(), trans_blocks.max())
        # materialize without progress bar indication
        #trans_blocks_persist = dask.persist(trans_blocks.min(), trans_blocks.max()
        # only convert structure
        block_min, block_max = dask.compute(trans_blocks.min(), trans_blocks.max())
        trans_amin = block_min.azi
        trans_amax = block_max.azi
        trans_rmin = block_min.rng
        trans_rmax = block_max.rng
        del trans_blocks, block_min, block_max
        #print ('trans_amin', trans_amin)

        # split geographic coordinate grid to equal chunks and rest
        #chunks = trans.azi.data.chunks
        #lt_blocks = np.array_split(trans['lat'].values, np.cumsum(chunks[0])[:-1])
        #ll_blocks = np.array_split(trans['lon'].values, np.cumsum(chunks[1])[:-1])
        lt_blocks = np.array_split(trans['y'].values, np.arange(0, trans['y'].size, self.netcdf_chunksize)[1:])
        ll_blocks = np.array_split(trans['x'].values, np.arange(0, trans['x'].size, self.netcdf_chunksize)[1:])

        # split radar coordinate grid to equal chunks and rest
        prm = self.PRM(burst_ref)
        a_max, r_max = prm.bounds()
        azis = np.arange(0.5, a_max, 1)
        rngs = np.arange(0.5, r_max, 1)
        #print ('azis', azis, 'rngs', rngs, 'sizes', azis.size, rngs.size)
        
        azis_blocks = np.array_split(azis, np.arange(0, azis.size, self.netcdf_chunksize)[1:])
        rngs_blocks = np.array_split(rngs, np.arange(0, rngs.size, self.netcdf_chunksize)[1:])
        #print ('azis_blocks.size', len(azis_blocks), 'rngs_blocks.size', len(rngs_blocks))

        blocks_total = []
        for azis_block in azis_blocks:
            blocks = []
            for rngs_block in rngs_blocks:
                block = dask.array.from_delayed(dask.delayed(trans_inv_block, traverse=False)
                                               (azis_block, rngs_block, 2, self.netcdf_chunksize),
                                               shape=(3, azis_block.size, rngs_block.size), dtype=np.float32)
                blocks.append(block)
                del block
            blocks_total.append(blocks)
            del blocks

        trans_inv_dask = dask.array.block(blocks_total)
        del blocks_total
        coords = {'a': azis, 'r': rngs}
        trans_inv = xr.Dataset({key: xr.DataArray(trans_inv_dask[idx],  coords=coords)
                                for idx, key in enumerate(['y', 'x', 'ele'])})
        del trans_inv_dask

        if interactive:
            return trans_inv
        
        filename = self.get_filename(burst_ref, 'trans_inv', clean=True)
        encoding = {varname: self._compression(trans_inv[varname].shape) for varname in trans_inv.data_vars}
        trans_inv.to_netcdf(filename,
                        encoding=encoding,
                        engine=self.netcdf_engine_write,
                        format=self.netcdf_format)
        del trans_inv
