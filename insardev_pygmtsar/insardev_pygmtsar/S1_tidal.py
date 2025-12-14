# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_gmtsar import S1_gmtsar

class S1_tidal(S1_gmtsar):

    def tidal_los_rad(self, stack):
        """
        Calculate tidal LOS displacement [rad] for data dates and spatial extent
        """
        return 1000*self.tidal_los(stack)/self.los_displacement_mm(1)

    def tidal_los(self, stack):
        """
        Interpolate pre-calculated tidal displacement for data pairs dates on the specified grid
        and convert to LOS displacement in meters
        """
        import pandas as pd
        import xarray as xr
        import numpy as np

        # extract pairs
        if len(stack.dims) == 3:
            pairs, dates = self.get_pairs(stack, dates=True)
            pairs = pairs[['ref', 'rep']].astype(str).values
            grid = stack[0]
        else:
            dates = [stack[key].dt.date.astype(str).item() for key in ['ref', 'rep']]
            pairs = [dates]
            grid = stack

        solid_tide = self.get_tidal().sel(date=dates)
        # satellite look vector
        sat_look = self.get_satellite_look_vector()

        def interp_pair(pair):
            # use outer variables
            date1, date2 = pair
            # interpolate on the grid
            coords = {'y': grid.y, 'x': grid.x}
            tidal1 = solid_tide.sel(date=date1).interp(coords, method='linear', assume_sorted=True)
            tidal2 = solid_tide.sel(date=date2).interp(coords, method='linear', assume_sorted=True)
            look = sat_look.interp(coords, method='linear', assume_sorted=True)
            tidal_diff = tidal2 - tidal1
            los = xr.dot(xr.concat([look.look_E, look.look_N, look.look_U], dim='dim'),
                      xr.concat([tidal_diff.dx, tidal_diff.dy, tidal_diff.dz], dim='dim'),
                      dims=['dim'])
            return los.values.astype(np.float32)

        # process all pairs
        results = [interp_pair(pair) for pair in pairs]
        result = np.stack(results, axis=0)

        if len(stack.dims) == 3:
            out = xr.DataArray(result, coords=stack.coords)
        else:
            out = xr.DataArray(result[0], coords=stack.coords)
        return out.rename(stack.name)

    def tidal_los_rad(self, stack):
        """
        Calculate tidal LOS displacement [rad] for data_pairs pairs and spatial extent
        """
        return 1000*self.tidal_los(stack)/self.los_displacement_mm(1)

    def tidal_correction_wrap(self, stack):
        """
        Apply tidal correction to wrapped phase pairs [rad] and wrap the result.
        """
        return self.wrap(stack - self.tidal_los_rad(stack)).rename(stack.name)
    
    # def get_tidal(self):
    #     return self.open_cube('tidal')

    def _tidal(self, date, grid):
        import xarray as xr
        import pandas as pd
        import numpy as np
        from io import StringIO, BytesIO
        import subprocess

        coords = np.column_stack([grid.ll.values.ravel(), grid.lt.values.ravel()])
        buffer = BytesIO()
        np.savetxt(buffer, coords, delimiter=' ', fmt='%.6f')
        stdin_data = buffer.getvalue()
        #print ('stdin_data', stdin_data)

        SC_clock_start, SC_clock_stop = self.PRM(date).get('SC_clock_start', 'SC_clock_stop')
        dt = (SC_clock_start + SC_clock_stop)/2
        argv = ['solid_tide', str(dt)]
        #cwd = os.path.dirname(self.filename) if self.filename is not None else '.'
        cwd = self.workdir
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=cwd, bufsize=10*1000*1000)
        stdout_data, stderr_data = p.communicate(input=stdin_data)
        stderr_data = stderr_data.decode('utf8')
        if stderr_data is not None and len(stderr_data):
            #print ('DEBUG: solid_tide', stderr_data)
            assert 0, f'DEBUG: solid_tide: {stderr_data}'
        out = np.fromstring(stdout_data, dtype=np.float32, sep=' ').reshape(grid.y.size, grid.x.size, 5)[None,]
        coords = {'date': pd.to_datetime([date]), 'y': grid.y, 'x': grid.x}
        das = {v: xr.DataArray(out[...,idx], coords=coords) for (idx, v) in enumerate(['lon', 'lat', 'dx', 'dy', 'dz'])}
        ds = xr.Dataset(das)
        return ds

    # def compute_tidal(self, dates=None, coarsen=32, n_jobs=-1, interactive=False):
    #     import xarray as xr
    #     import numpy as np
    #     from tqdm.auto import tqdm
    #     import joblib

    #     if dates is None:
    #         dates = self.df.index.unique()

    #     # expand simplified definition
    #     if not isinstance(coarsen, (list,tuple, np.ndarray)):
    #         coarsen = (coarsen, coarsen)

    #     trans_inv = self.get_trans_inv()
    #     dy, dx = np.diff(trans_inv.y)[0], np.diff(trans_inv.x)[0]
    #     #print ('dy, dx', dy, dx)
    #     #step_y, step_x = int(np.round(coarsen[0]*dy)), int(np.round(coarsen[1]*dx))
    #     # define target grid spacing
    #     step_y, step_x = int(coarsen[0]/dy), int(coarsen[1]/dx)
    #     #print ('step_y, step_x', step_y, step_x)
    #     # fix zero step when specified coarsen is larger than the transform grid coarsen
    #     if step_y < 1:
    #         step_y = 1
    #     if step_x < 1:
    #         step_x = 1
    #     grid = trans_inv.sel(y=trans_inv.y[step_y//2::step_y], x=trans_inv.x[step_x//2::step_x])

    #     def tidal(date):
    #         return self._tidal(date, grid)

    #     with self.progressbar_joblib(tqdm(desc='Tidal Computation', total=len(dates))) as progress_bar:
    #         outs = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(tidal)(date) for date in dates)

    #     ds = xr.concat(outs, dim='date')
    #     if interactive:
    #         return ds
    #     self.save_cube(ds, 'tidal', 'Solid Earth Tides Saving')
