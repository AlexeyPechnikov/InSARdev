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
from .Stack_multilooking import Stack_multilooking

class Stack_unwrap_snaphu(Stack_multilooking):

    # -s for SMOOTH mode and -d for DEFO mode when DEFOMAX_CYCLE should be defined in the configuration
    # DEFO mode (-d) and DEFOMAX_CYCLE=0 is equal to SMOOTH mode (-s)
    # https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu_man1.html
    def snaphu(self, phase, corr=None, conf=None, conncomp=False, debug=False):
        """
        Unwraps phase using SNAPHU with the given phase and correlation data.

        This function unwraps the phase of an interferogram using the Statistical-cost, Network-flow Algorithm
        for Phase Unwrapping (SNAPHU) with user-defined parameters. The unwrapped phase is saved as a grid file
        in the working directory.

        Parameters
        ----------
        phase : xarray.DataArray
            The phase data as a string or xarray.DataArray, default is 'phasefilt'.

        corr : xarray.DataArray, optional
            The correlation data as a string or xarray.DataArray, default is 'corr'.

        conf : str, optional
            The SNAPHU configuration string, default is None (use the snaphu_config method).

        conncomp : bool, optional
            If True, return connection components map, default is False.

        debug : bool, optional
            If True, print debugging information during the unwrapping process, default is False.

        Returns
        -------
        xarray.Dataset
            Return the unwrapped phase and optional connection components as an xarray.Dataset.

        """
        import xarray as xr
        import numpy as np
        import pandas as pd
        import os
        import subprocess
        from datetime import datetime
        import uuid
        import warnings
        # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
        warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', module='dask')
        warnings.filterwarnings('ignore', module='dask.core')
        # Filter out Dask "Restarting worker" warnings
        warnings.filterwarnings("ignore", module="distributed.nanny")
        # disable "distributed.utils_perf - WARNING - full garbage collections ..."
        try:
            from dask.distributed import utils_perf
            utils_perf.disable_gc_diagnosis()
        except ImportError:
            from distributed.gc import disable_gc_diagnosis
            disable_gc_diagnosis()

        if conf is None:
            conf = self.snaphu_config()
        # set unique processing subdirectory
        conf += f'    TILEDIR snaphu_tiledir_{str(uuid.uuid4())}'

        # define basename for SNAPHU temp files
        basename = os.path.join(self.basedir, f'snaphu_{str(uuid.uuid4())}')
        #print ('basename', basename)

        # SNAPHU input files
        phase_in = basename + '.phase'
        corr_in = basename + '.corr'
        mask_in = basename + '.bytemask'
        # SNAPHU output files
        unwrap_out = basename + 'unwrap.out'
        conncomp_out = basename + 'conncomp.out'

        # prepare SNAPHU input files
        # NaN values are not allowed for SNAPHU phase input file
        # interpolate when exist valid values around and fill zero pixels far away from valid ones
        # convert to lazy array setting the chunk size
        phase.fillna(0).compute(n_workers=1).values.astype(np.float32).tofile(phase_in)
        # SNAPHU masks out 0 and uses only valid pixels with mask 1
        xr.where(np.isnan(phase), 0, 1).values.astype(np.ubyte).tofile(mask_in)
    
        if corr is not None:
            # NaN values are not allowed for SNAPHU correlation input file
            # just fill NaNs by zeroes because the main trick is phase filling
            corr.fillna(0).compute(n_workers=1).values.astype(np.float32).tofile(corr_in)

        # launch SNAPHU binary (NaNs are not allowed for input but returned in output)
        argv = ['snaphu', phase_in, str(phase.shape[1]), '-M', mask_in, '-f', '/dev/stdin', '-o', unwrap_out, '-d']
        # output connection componetets map
        if conncomp:
            argv.append('-g')
            argv.append(conncomp_out)
        # add optional correlation grid
        if corr is not None:
            argv.append('-c')
            argv.append(corr_in)
        if debug:
            argv.append('-v')
            print ('DEBUG: argv', argv)
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             encoding='utf8', bufsize=10*1000*1000)
        stdout_data, stderr_data = p.communicate(input=conf)

        outs = []
        # check for expected SNAPHU output files
        if os.path.exists(unwrap_out) and (not conncomp or os.path.exists(conncomp_out)):
            # convert to grid unwrapped phase from SNAPHU output applying postprocessing
            values = np.fromfile(unwrap_out, dtype=np.float32).reshape(phase.shape)
            #values = np.frombuffer(stdout_data, dtype=np.float32).reshape(phase.shape)
            # revert NaNs in output because SNAPNU does not support them
            unwrap = xr.DataArray(values, phase.coords, name='phase').chunk(self.chunksize).where(~np.isnan(phase))
            outs.append(unwrap)
            del values, unwrap

            if conncomp:
                # convert to grid the connected components from SNAPHU output as is (UCHAR)
                values = np.fromfile(conncomp_out, dtype=np.ubyte).reshape(phase.shape)
                conn = xr.DataArray(values, phase.coords, name='conncomp').chunk(self.chunksize)
                outs.append(conn)
                del values, conn
        else:
            # return the same data structure as expected but NaN-filled
            outs.append(xr.full_like(phase, np.nan).rename('phase'))
            if conncomp:
                outs.append(xr.full_like(phase, np.nan).rename('conncomp'))
        out = xr.merge(outs)
        del outs

        # add processing log
        out.attrs['snaphu'] = stdout_data + '\n' + stderr_data

        # the output files deleted immediately
        # but these are accessible while open descriptors persist
        for tmp_file in [phase_in, corr_in, mask_in, unwrap_out, conncomp_out]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        return out

    def snaphu_config(self, defomax=0, **kwargs):
        """
        Generate a Snaphu configuration file.

        Parameters
        ----------
        defomax : int, optional
            Maximum deformation value. Default is 0.
        **kwargs : dict, optional
            Additional parameters to include in the configuration file.

        Returns
        -------
        str
            The Snaphu configuration file content.

        Examples
        --------
        Generate a Snaphu configuration file with defomax=10:
        snaphu_config(defomax=10)

        Generate a Snaphu configuration file with defomax=5 and additional parameters:
        snaphu_config(defomax=5, param1=10, param2=20)
        """
        import os
        import joblib

        tiledir = self.basedir
        n_jobs = joblib.cpu_count()

        conf_basic = f"""
        # basic config
        INFILEFORMAT   FLOAT_DATA
        OUTFILEFORMAT  FLOAT_DATA
        AMPFILEFORMAT  FLOAT_DATA
        CORRFILEFORMAT FLOAT_DATA
        ALTITUDE       693000.0
        EARTHRADIUS    6378000.0
        NEARRANGE      831000
        DR             18.4
        DA             28.2
        RANGERES       28
        AZRES          44
        LAMBDA         0.0554658
        NLOOKSRANGE    1
        NLOOKSAZ       1
        TILEDIR        {tiledir}_snaphu_tiledir
        NPROC          {n_jobs}
        """
        conf_custom = '# custom config\n'
        # defomax can be None
        keyvalues = ([('DEFOMAX_CYCLE', defomax)] if defomax is not None else []) + list(kwargs.items())
        for key, value in keyvalues:
            if isinstance(value, bool):
                value = 'TRUE' if value else 'FALSE'
            conf_custom += f'        {key} {value}\n'
        return conf_basic + conf_custom


    # Backward compatibility wrapper
    def nearest_grid(self, in_grid, search_radius_pixels=None):
        print('WARNING: nearest_grid() is deprecated. Use fill_nan_nearest() instead.')
        return self.fill_nan_nearest(in_grid, search_radius_pixels)

    def fill_nan_nearest(self, in_grid, search_radius_pixels=None):
        """
        Perform nearest neighbor interpolation on a 2D grid.

        Parameters
        ----------
        in_grid : xarray.DataArray
            The input 2D grid to be interpolated.
        search_radius_pixels : int, optional
            The interpolation distance in pixels. If not provided, the default is set to the chunksize of the Stack object.

        Returns
        -------
        xarray.DataArray
            The interpolated 2D grid.

        Examples
        --------
        Fill gaps in the specified grid using nearest neighbor interpolation:
        stack.fill_nan_nearest(grid)

        Notes
        -----
        This method performs nearest neighbor interpolation on a 2D grid. It replaces the NaN values in the input grid with
        the nearest non-NaN values. The interpolation is performed within a specified search radius in pixels.
        If a search radius is not provided, the default search radius is set to the chunksize of the Stack object.
        """
        from scipy.spatial import cKDTree
        import xarray as xr
        import numpy as np

        assert in_grid.chunks is not None, 'fill_nan_nearest() input grid chunks are not defined'

        if search_radius_pixels is None:
            search_radius_pixels = self.chunksize
        elif search_radius_pixels <= 0:
            print (f'NOTE: interpolation ignored for search_radius_pixels={search_radius_pixels}')
            return in_grid
        else:
            assert search_radius_pixels <= self.chunksize, \
                f'ERROR: apply fill_nan_nearest() multiple times to fill gaps more than {self.chunksize} pixels chunk size'

        def func(grid, y, x, distance, scaley, scalex):

            grid1d = grid.reshape(-1).copy()
            nanmask0 = np.isnan(grid1d)
            # all the pixels already defined
            if np.all(~nanmask0):
                return grid

            # crop full grid subset to search for missed values neighbors
            ymin = y.min()-scaley*distance-1
            ymax = y.max()+scaley*distance+1
            xmin = x.min()-scalex*distance-1
            xmax = x.max()+scalex*distance+1
            data = in_grid.sel(y=slice(ymin, ymax), x=slice(xmin, xmax))
            ys, xs = data.y, data.x
            # compute dask arrays to prevent ineffective index lookup
            ys, xs = [vals.values.reshape(-1) for vals in xr.broadcast(ys, xs)]
            data1d = data.values.reshape(-1)
            nanmask = np.isnan(data1d)
            # all the subset pixels are empty, the search is useless
            if np.all(nanmask):
                return grid

            # build index tree for all the valid subset values
            source_yxs = np.stack([ys[~nanmask]/scaley, xs[~nanmask]/scalex], axis=1)
            tree = cKDTree(source_yxs, compact_nodes=False, balanced_tree=False)

            # query the index tree for all missed values neighbors
            target_yxs = np.stack([(y/scaley).reshape(-1)[nanmask0], (x/scalex).reshape(-1)[nanmask0]], axis=1)
            #assert 0, target_yxs
            d, inds = tree.query(target_yxs, k = 1, distance_upper_bound=distance, workers=1)
            # fill missed values using neighbors when these ones are found
            inds = np.where(np.isinf(d), 0, inds)
            grid1d[nanmask0] = np.where(np.isinf(d), np.nan, data1d[~nanmask][inds])
            return grid1d.reshape(grid.shape)

        coords = ['y', 'x']
        scale = [in_grid[coord].diff(coord).item(0) for coord in coords]
        yy = xr.DataArray(in_grid[coords[0]]).chunk(-1)
        xx = xr.DataArray(in_grid[coords[1]]).chunk(-1)
        ys, xs = xr.broadcast(yy,xx)

        # xarray wrapper
        grid = xr.apply_ufunc(
            func,
            in_grid,
            ys.chunk(in_grid.chunks),
            xs.chunk(in_grid.chunks),
            dask='parallelized',
            vectorize=False,
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={'distance': search_radius_pixels, 'scaley': scale[0], 'scalex': scale[1]},
        )
        assert grid.chunks is not None, 'fill_nan_nearest() output grid chunks are not defined'
        return grid
