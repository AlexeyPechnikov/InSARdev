# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_tidal import S1_tidal

class S1_dem(S1_tidal):
    import geopandas as gpd
    import xarray as xr
    import pandas as pd

    def get_geoid(self, grid: xr.DataArray|xr.Dataset=None) -> xr.DataArray:
        """
        Get EGM96 geoid heights.

        Parameters
        ----------
        grid : xarray array or dataset, optional
            Interpolate geoid heights on the grid. Default is None.

        Returns
        -------
        None

        Examples
        --------
        stack.get_geoid()

        Notes
        -----
        See EGM96 geoid heights on http://icgem.gfz-potsdam.de/tom_longtime
        """
        import xarray as xr
        import os
        import importlib.resources as resources

        with resources.as_file(resources.files('insardev_pygmtsar.data') / 'geoid_egm96_icgem.grd') as geoid_filename:
            geoid = xr.open_dataarray(geoid_filename, engine=self.netcdf_engine_read, chunks=self.netcdf_chunksize).rename({'y': 'lat', 'x': 'lon'})
        if grid is not None:
            return self.interp2d_like(geoid, grid)
        return geoid

    # buffer required to get correct (binary) results from SAT_llt2rat tool
    # small buffer produces incomplete area coverage and restricted NaNs
    # 0.02 degrees works well worldwide but not in Siberia
    # minimum buffer size: 8 arc seconds for 90 m DEM
    def get_dem_wgs84ellipsoid(self, geometry: gpd.GeoDataFrame=None, buffer_degrees: float=0.04):
        """
        Load and preprocess digital elevation model (DEM) data from specified datafile or variable.

        Parameters
        ----------
        geometry : geopandas.GeoDataFrame, optional
            The geometry of the area to crop the DEM.
        buffer_degrees : float, optional
            The buffer in degrees to add to the geometry.

        Returns
        -------
        None

        Examples
        --------
        Load and crop from local NetCDF file:
        stack.load_dem('GEBCO_2020/GEBCO_2020.nc')

        Load and crop from local GeoTIF file:
        stack.load_dem('GEBCO_2019.tif')

        Load from Xarray DataArray or Dataset:
        stack.set_dem(None).load_dem(dem)
        stack.set_dem(None).load_dem(dem.to_dataset())

        Notes
        -----
        This method loads DEM from the user specified file. The data is then preprocessed by removing
        the EGM96 geoid to make the heights relative to the WGS84 ellipsoid.
        """
        import xarray as xr
        import numpy as np
        import dask
        import rioxarray as rio
        import geopandas as gpd
        import pandas as pd
        import os

        if self.DEM is None:
            raise ValueError('ERROR: DEM is not specified.')

        if geometry is None:
            geometry = self.df

        if isinstance(self.DEM, (xr.Dataset)):
            ortho = self.DEM[list(self.DEM.data_vars)[0]]
        elif isinstance(self.DEM, (xr.DataArray)):
            ortho = self.DEM
        elif isinstance(self.DEM, str) and os.path.splitext(self.DEM)[-1] in ['.tiff', '.tif', '.TIF']:
            ortho = rio.open_rasterio(self.DEM, chunks=self.chunksize).squeeze(drop=True)\
                .rename({'y': 'lat', 'x': 'lon'})\
                .drop('spatial_ref')
            if ortho.lat.diff('lat')[0].item() < 0:
                ortho = ortho.reindex(lat=ortho.lat[::-1])
        elif isinstance(self.DEM, str) and os.path.splitext(self.DEM)[-1] in ['.nc', '.netcdf', '.grd']:
            ortho = xr.open_dataarray(self.DEM, engine=self.netcdf_engine_read, chunks=self.chunksize)
        elif isinstance(self.DEM, str):
            print ('ERROR: filename extension is not recognized. Should be one from .tiff, .tif, .TIF, .nc, .netcdf, .grd')
        else:
            print ('ERROR: argument is not an Xarray object and it is not a file name')
        ortho = ortho.transpose('lat','lon')
        
        # unique indices required for interpolation
        lat_index = pd.Index(ortho.coords['lat'])
        lon_index = pd.Index(ortho.coords['lon'])
        duplicates = lat_index[lat_index.duplicated()].tolist() + lon_index[lon_index.duplicated()].tolist()
        assert len(duplicates) == 0, 'ERROR: DEM grid includes duplicated coordinates, possibly on merged tiles edges'

        # crop to the geometry extent
        bounds = self.get_bounds(geometry.buffer(buffer_degrees))
        ortho = ortho.sel(lat=slice(bounds[1], bounds[3]), lon=slice(bounds[0], bounds[2]))

        # heights correction
        geoid = self.get_geoid(ortho)
        ds = (ortho + geoid).astype(np.float32).transpose('lat','lon').rename("dem")
        return self.spatial_ref(ds, 4326)
