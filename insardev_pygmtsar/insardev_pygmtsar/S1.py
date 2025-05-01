# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_transform import S1_transform
from .PRM import PRM

class S1(S1_transform):
    import pandas as pd
    import xarray as xr

    # class variables
    datadir: str|None = None
    DEM: str|xr.DataArray|xr.Dataset|None = None
    df: pd.DataFrame|None = None

    def plot(self, records: pd.DataFrame=None, ref: str=None,
             alpha: float=0.7, caption: str='Estimated Bursts Locations', cmap: str='turbo', aspect: float=None, _size: tuple[int, int]=None):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib

        # screen size in pixels (width, height) to estimate reasonable number pixels per plot
        # this is quite large to prevent aliasing on 600dpi plots without additional processing
        if _size is None:
            _size = (2000,1000)

        if records is None:
            records = self.to_dataframe(ref=ref)
        
        plt.figure()
        if self.DEM is not None:
            dem = self.get_dem_wgs84ellipsoid()
            # there is no reason to plot huge arrays much larger than screen size for small plots
            #print ('screen_size', screen_size)
            size_y, size_x = dem.shape
            #print ('size_x, size_y', size_x, size_y)
            factor_y = int(np.round(size_y / _size[1]))
            factor_x = int(np.round(size_x / _size[0]))
            #print ('factor_x, factor_y', factor_x, factor_y)
            # coarsen and materialize data for all the calculations and plotting
            dem = dem[::max(1, factor_y), ::max(1, factor_x)].load()
            dem.plot.imshow(cmap='gray', alpha=alpha, add_colorbar=True)
            dem.close()
            del dem
        cmap = matplotlib.colormaps[cmap]
        colors = dict([(v, cmap(k)) for k, v in enumerate(records.index.unique())])

        # Calculate overlaps including self-overlap
        overlap_count = [sum(1 for geom2 in records.geometry if geom1.intersects(geom2)) for geom1 in records.geometry]
        _alpha=max(1/max(overlap_count), 0.002)
        _alpha = min(_alpha, alpha/2)
        # define transparency for the calculated overlaps and apply minimum transparency threshold
        records.reset_index().plot(color=[colors[k] for k in records.index], alpha=_alpha, edgecolor='black', ax=plt.gca())
        if aspect is not None:
            plt.gca().set_aspect(aspect)
        plt.title(caption)
