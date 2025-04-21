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

    # redefine to save disk space
    netcdf_complevel = 1

    df = None
    basedir = None
    reference = None
    dem_filename = None
    landmask_filename = None
    
    # def set_workdir(self, workdir, drop_if_exists=False):
    #     import os
    #     import shutil

    #     # (re)create basedir only when force=True
    #     if os.path.exists(workdir):
    #         if drop_if_exists:
    #             shutil.rmtree(workdir, ignore_errors=True)
    #         else:
    #             #raise ValueError('ERROR: The base directory already exists. Use drop_if_exists=True to delete it and start new processing.')
    #             print('WARNING: The base directory already exists. Use drop_if_exists=True to delete it and start new processing.')
    #     os.makedirs(workdir, exist_ok=True)
    #     self.basedir = workdir

    # def set_bursts(self, scenes, reference=None):
    #     assert len(scenes), 'ERROR: the scenes list is empty.'
    #     assert len(scenes[scenes.orbit.isna()])==0, 'ERROR: orbits missed, check "orbit" column.'
    #     self.df = scenes
    #     self.set_reference(reference)
    #     return self

#    def make_gaussian_filter(self, range_dec, azi_dec, wavelength, debug=False):
#        """
#        Wrapper for PRM.make_gaussian_filter() and sonamed command line tool. Added for development purposes only.
#        """
#        import numpy as np
#
#        gauss_dec, gauss_string = self.PRM().make_gaussian_filter(range_dec, azi_dec, wavelength, debug=debug)
#        coeffs = [item for line in gauss_string.split('\n') for item in line.split('\t') if item != '']
#        # x,y dims order
#        shape = np.array(coeffs[0].split(' ')).astype(int)
#        # y,x dims order
#        matrix = np.array(coeffs[1:]).astype(float).reshape((shape[1],shape[0]))
#        return (gauss_dec, matrix)

    def plot(self, records: pd.DataFrame=None, dem: str='auto', image: xr.DataArray=None, ref: str=None,
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
        if image is not None:
            image.plot.imshow(cmap='gray', alpha=alpha, add_colorbar=False)
        if isinstance(dem, str) and dem == 'auto':
            if self.dem_filename is not None:
                dem = self.get_dem()
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
        elif dem is not None:
            dem.plot.imshow(cmap='gray', alpha=alpha, add_colorbar=True)
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
