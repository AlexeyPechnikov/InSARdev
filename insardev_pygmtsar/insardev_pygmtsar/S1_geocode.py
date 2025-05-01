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

class S1_geocode(S1_align):
    import pandas as pd
    import xarray as xr
    import numpy as np

    def geocode(self, transform: xr.Dataset, data: xr.DataArray) -> xr.DataArray:
        """
        Perform geocoding from radar to projected coordinates.

        Parameters
        ----------
        transform : xarray.Dataset
            The transform matrix.
        data : xarray.DataArray
            Grid(s) representing the interferogram(s) in radar coordinates.

        Returns
        -------
        xarray.DataArray
            The geocoded grid(s) in projected coordinates.

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
            transform.azi, 'yx',
            transform.rng, 'yx',
            dtype=data.dtype
        )

        da = xr.DataArray(out, transform.ele.coords).rename(data.name)
        del out
        return da

    @staticmethod
    def get_utm_epsg(lat: float, lon: float) -> int:
        zone_num = int((lon + 180) // 6) + 1
        if lat >= 0:
            return 32600 + zone_num
        else:
            return 32700 + zone_num
    
    @staticmethod
    def proj(ys: np.ndarray, xs: np.ndarray, to_epsg: int, from_epsg: int) -> tuple[np.ndarray, np.ndarray]:
        from pyproj import CRS, Transformer
        from_crs = CRS.from_epsg(from_epsg)
        to_crs = CRS.from_epsg(to_epsg)
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        xs_new, ys_new = transformer.transform(xs, ys)
        del transformer, from_crs, to_crs
        return ys_new, xs_new

    def get_transform(self, outdir: str, burst: str) -> xr.Dataset:
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
        import os
        ds = xr.open_zarr(store=os.path.join(outdir, 'transform'),
                         consolidated=True,
                         chunks='auto')
        # variables are stored as int32, convert to float32 instead of default float64
        for v in ('azi','rng','ele'):
            ds[v] = ds[v].astype('float32')
        return ds
        #.dropna(dim='y', how='all')
        #.dropna(dim='x', how='all')

    def compute_transform(self,
                          outdir: str,
                          burst_ref: str,
                          basedir: str,
                          resolution: tuple[int, int]=(10, 2.5),
                          scale_factor: float=2.0,
                          epsg: int=None):
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

        prm = self.PRM(burst_ref, basedir)
        #timestamp = self.julian_to_datetime(prm.get('SC_clock_start'))
        #print ('timestamp geocode', timestamp)

        def SAT_llt2ratlook(lats, lons, zs):
            # for binary=True values outside of the scene missed and the array is not complete
            # 4th and 5th coordinates are the same as input lat, lon
            #print (f'SAT_llt2rat: lats={lats}, lons={lons}, zs={zs} ({lats.shape}, {lons.shape}, {zs.shape})')
            coords3d = np.column_stack([lons, lats, np.nan_to_num(zs)])
            #print ('coords3d', coords3d)
            rae = prm.SAT_llt2rat(coords3d, precise=1, binary=False).astype(np.float32)
            #print ('rae', rae)
            rae = rae.reshape(zs.size, 5)[...,:3]
            #rae[~np.isfinite(zs), :] = np.nan
            #return rae
            # look_E look_N look_U
            look = prm.SAT_look(coords3d, binary=True).astype(np.float32).reshape(zs.size, 6)[...,3:]
            out = np.concatenate([rae, look], axis=-1)
            del rae, look
            out[~np.isfinite(zs), :] = np.nan
            return out

        # exclude latitude and longitude columns as redundant
        def trans_block(ys, xs, coarsen, epsg, scale_factor, amin=-np.inf, amax=np.inf, rmin=-np.inf, rmax=np.inf):
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

            return rae
            #return np.where(np.isfinite(rae), (scale_factor*rae).round(), np.iinfo(np.int32).max).astype(np.int32)

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
                        dask.delayed(trans_block)(ys_block, xs_block, coarsen, epsg, scale_factor, **borders),
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

        # do not use coordinate names lat,lon because the output grid saved as (lon,lon) in this case...
        record = self.get_record(burst_ref)
        dem = self.get_dem_wgs84ellipsoid(geometry=record.geometry)

        if epsg is None:
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

        # transform user-specified grid resolution to coarsen factor
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

        # scale to integers for better compression
        for varname in ['azi', 'rng', 'ele']:
            trans[varname] = xr.where(np.isfinite(trans[varname]), (scale_factor*trans[varname]).round(), np.iinfo(np.int32).max).astype(np.int32)
            trans[varname].attrs['scale_factor'] = 1/scale_factor
            trans[varname].attrs['add_offset'] = 0
            trans[varname].attrs['_FillValue'] = np.iinfo(np.int32).max

        # add add georeference attributes
        trans = self.spatial_ref(trans, epsg)
        trans.attrs['spatial_ref'] = trans.spatial_ref.attrs['spatial_ref']
        trans = trans.drop_vars('spatial_ref')

        encoding_vars = {var: self.get_encoding_zarr(dtype=trans[var].dtype) for var in trans.data_vars}
        #print ('encoding_vars', encoding_vars)
        encoding_coords = {coord: self.get_encoding_zarr(chunks=(trans[coord].size,), dtype=trans[coord].dtype) for coord in trans.coords}
        #print ('encoding_coords', encoding_coords)
        trans.to_zarr(
            store=os.path.join(outdir, 'transform'),
            encoding=encoding_vars | encoding_coords,
            mode='w',
            consolidated=True
        )
        del trans
