# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_geocode import S1_geocode

class S1_topo(S1_geocode):
    import xarray as xr

    def flat_earth_topo_phase(self, topo: xr.DataArray, burst_rep: str, burst_ref: str, basedir: str) -> xr.DataArray:
        """
        np.arctan2(np.sin(topo_phase), np.cos(topo_phase))[0].plot.imshow()
        """
        import pandas as pd
        import dask
        import dask.array as da
        import xarray as xr
        import numpy as np
        import joblib
        import warnings
        # suppress Dask warning "RuntimeWarning: invalid value encountered in divide"
        warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', module='dask')
        warnings.filterwarnings('ignore', module='dask.core')

        # calculate the combined earth curvature and topography correction
        def calc_drho(rho, topo, earth_radius, height, b, alpha, Bx):
            sina = np.sin(alpha)
            cosa = np.cos(alpha)
            c = earth_radius + height
            # compute the look angle using equation (C26) in Appendix C
            # GMTSAR uses long double here
            #ret = earth_radius + topo.astype(np.longdouble)
            ret = earth_radius + topo
            cost = ((rho**2 + c**2 - ret**2) / (2. * rho * c))
            #if (cost >= 1.)
            #    die("calc_drho", "cost >= 0");
            sint = np.sqrt(1. - cost**2)
            # Compute the offset effect from non-parallel orbit
            term1 = rho**2 + b**2 - 2 * rho * b * (sint * cosa - cost * sina) - Bx**2
            drho = -rho + np.sqrt(term1)
            del term1, sint, cost, ret, c, cosa, sina
            return drho
            
        #def block_phase(prm1, prm2, ylim, xlim):
        def block_phase_dask(block_topo, y_chunk, x_chunk, prm1, prm2):
            from scipy import constants

            if prm1 is None or prm2 is None:
                return np.ones_like(block_topo).astype(np.complex64)

            # get full dimensions
            xdim = prm1.get('num_rng_bins')
            ydim = prm1.get('num_patches') * prm1.get('num_valid_az')

            # get heights
            htc = prm1.get('SC_height')
            ht0 = prm1.get('SC_height_start')
            htf = prm1.get('SC_height_end')

            # compute the time span and the time spacing
            tspan = 86400 * abs(prm2.get('SC_clock_stop') - prm2.get('SC_clock_start'))
            assert (tspan >= 0.01) and (prm2.get('PRF') >= 0.01), \
                f"ERROR in sc_clock_start={prm2.get('SC_clock_start')}, sc_clock_stop={prm2.get('SC_clock_stop')}, or PRF={prm2.get('PRF')}"

            # setup the default parameters
            drange = constants.speed_of_light / (2 * prm2.get('rng_samp_rate'))
            alpha = prm2.get('alpha_start') * np.pi / 180
            # a constant that converts drho into a phase shift
            cnst = -4 * np.pi / prm2.get('radar_wavelength')

            # calculate initial baselines
            Bh0 = prm2.get('baseline_start') * np.cos(prm2.get('alpha_start') * np.pi / 180)
            Bv0 = prm2.get('baseline_start') * np.sin(prm2.get('alpha_start') * np.pi / 180)
            Bhf = prm2.get('baseline_end')   * np.cos(prm2.get('alpha_end')   * np.pi / 180)
            Bvf = prm2.get('baseline_end')   * np.sin(prm2.get('alpha_end')   * np.pi / 180)
            Bx0 = prm2.get('B_offset_start')
            Bxf = prm2.get('B_offset_end')

            # first case is quadratic baseline model, second case is default linear model
            if prm2.get('baseline_center') != 0 or prm2.get('alpha_center') != 0 or prm2.get('B_offset_center') != 0:
                Bhc = prm2.get('baseline_center') * np.cos(prm2.get('alpha_center') * np.pi / 180)
                Bvc = prm2.get('baseline_center') * np.sin(prm2.get('alpha_center') * np.pi / 180)
                Bxc = prm2.get('B_offset_center')

                dBh = (-3 * Bh0 + 4 * Bhc - Bhf) / tspan
                dBv = (-3 * Bv0 + 4 * Bvc - Bvf) / tspan
                ddBh = (2 * Bh0 - 4 * Bhc + 2 * Bhf) / (tspan * tspan)
                ddBv = (2 * Bv0 - 4 * Bvc + 2 * Bvf) / (tspan * tspan)

                dBx = (-3 * Bx0 + 4 * Bxc - Bxf) / tspan
                ddBx = (2 * Bx0 - 4 * Bxc + 2 * Bxf) / (tspan * tspan)
            else:
                dBh = (Bhf - Bh0) / tspan
                dBv = (Bvf - Bv0) / tspan
                dBx = (Bxf - Bx0) / tspan
                ddBh = ddBv = ddBx = 0

            # calculate height increment
            dht = (-3 * ht0 + 4 * htc - htf) / tspan
            ddht = (2 * ht0 - 4 * htc + 2 * htf) / (tspan * tspan)

            near_range = (prm1.get('near_range') + \
                x_chunk.reshape(1,-1) * (1 + prm1.get('stretch_r')) * drange) + \
                y_chunk.reshape(-1,1) * prm1.get('a_stretch_r') * drange

            # calculate the change in baseline and height along the frame
            time = y_chunk * tspan / (ydim - 1)        
            Bh = Bh0 + dBh * time + ddBh * time**2
            Bv = Bv0 + dBv * time + ddBv * time**2
            Bx = Bx0 + dBx * time + ddBx * time**2
            B = np.sqrt(Bh * Bh + Bv * Bv)
            alpha = np.arctan2(Bv, Bh)
            height = ht0 + dht * time + ddht * time**2

            # calculate the combined earth curvature and topography correction
            drho = calc_drho(near_range, block_topo, prm1.get('earth_radius'),
                             height.reshape(-1, 1), B.reshape(-1, 1), alpha.reshape(-1, 1), Bx.reshape(-1, 1))

            #phase_shift = np.exp(1j * (cnst * drho))
            phase_shift = cnst * drho
            del near_range, drho, height, B, alpha, Bx, Bv, Bh, time

            #return phase_shift.astype(np.complex64)
            return phase_shift.astype(np.float32)

        def prepare_prms(burst_rep, burst_ref):
            if burst_rep == burst_ref:
                return (None, None)
            prm_ref = self.PRM(burst_ref, basedir=basedir)
            prm_rep = self.PRM(burst_rep, basedir=basedir)
            prm_rep.set(prm_ref.SAT_baseline(prm_rep, tail=9)).fix_aligned()
            prm_ref.set(prm_ref.SAT_baseline(prm_ref).sel('SC_height','SC_height_start','SC_height_end')).fix_aligned()
            return (prm_ref, prm_rep)
        
        prms = prepare_prms(burst_rep, burst_ref)
        # fill NaNs by 0 and expand to 3d
        topo_dask = da.where(da.isnan(topo.data), 0, topo.data)
        out = da.blockwise(
            block_phase_dask,
            'yx',
            topo_dask, 'yx',
            topo.a, 'y',
            topo.r, 'x',
            prm1=prms[0],
            prm2=prms[1],
            #dtype=np.complex64
            dtype=np.float32
        )
        del topo_dask, prms

        topo_phase = xr.DataArray(out, topo.coords).where(da.isfinite(topo)).rename('phase')
        del out
        return topo_phase

    def get_topo(self, burst, basedir: str):
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
        import os
        return xr.open_zarr(os.path.join(basedir, 'topo'),
                            consolidated=True,
                            chunks="auto")['topo']

    def compute_topo(self, workdir: str, transform: xr.Dataset, burst_ref: str, basedir: str):
        """
        Retrieve or calculate the transform data. This transform data is then saved as
            a NetCDF file for future use.

            This function generates data mapping from radar coordinates to geographical coordinates.
            The function uses the direct transform data.

        Parameters
        ----------
        workdir : str
            The work directory.
        burst_ref : str
            The reference burst name.
        basedir : str
            The basedir directory.
        resolution : tuple[int, int]
            The resolution of the transform data.

        Note
        ----
        This function operates on the 'transform' grid using chunks (specified by 'chunksize') rather than
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
                block_trans = transform.isel(y=slice(chunksize*block_y,chunksize*(block_y+1)),
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
            grid_ele = block_trans[2][indices]
            grid_ele[distances>tolerance] = np.nan
            #print ('distance range', distances.min().round(2), distances.max().round(2))
            #assert distances.max() < 2, f'Unexpectedly large distance between radar and geographic coordinate grid pixels (>=2): {distances.max()}'
            del block_trans, indices, distances

            # pack all the outputs into one 3D array
            return np.asarray([grid_ele]).reshape((1, azis.size, rngs.size))

        # calculate indices on the fly
        trans_blocks = transform[['azi', 'rng']].coarsen(y=self.chunksize, x=self.chunksize, boundary='pad')
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
        lt_blocks = np.array_split(transform['y'].values, np.arange(0, transform['y'].size, self.chunksize)[1:])
        ll_blocks = np.array_split(transform['x'].values, np.arange(0, transform['x'].size, self.chunksize)[1:])

        # split radar coordinate grid to equal chunks and rest
        prm = self.PRM(burst_ref, basedir)
        a_max, r_max = prm.bounds()
        azis = np.arange(0.5, a_max, 1)
        rngs = np.arange(0.5, r_max, 1)
        #print ('azis', azis, 'rngs', rngs, 'sizes', azis.size, rngs.size)
        
        azis_blocks = np.array_split(azis, np.arange(0, azis.size, self.chunksize)[1:])
        rngs_blocks = np.array_split(rngs, np.arange(0, rngs.size, self.chunksize)[1:])
        #print ('azis_blocks.size', len(azis_blocks), 'rngs_blocks.size', len(rngs_blocks))

        blocks_total = []
        for azis_block in azis_blocks:
            blocks = []
            for rngs_block in rngs_blocks:
                block = dask.array.from_delayed(dask.delayed(trans_inv_block, traverse=False)
                                               (azis_block, rngs_block, 2, self.chunksize),
                                               shape=(1, azis_block.size, rngs_block.size), dtype=np.float32)
                blocks.append(block)
                del block
            blocks_total.append(blocks)
            del blocks

        trans_inv_dask = dask.array.block(blocks_total)
        del blocks_total
        coords = {'a': azis, 'r': rngs}
        trans_inv = xr.Dataset({key: xr.DataArray(trans_inv_dask[idx],  coords=coords) for idx, key in enumerate(['ele'])})
        del trans_inv_dask
        
        topo = trans_inv.ele.rename('topo')
        del trans_inv
        #print ('topo', topo)

        encoding = {'topo': self.get_encoding_zarr()}
        #print ('encoding', encoding)
        topo.to_zarr(
            store=os.path.join(basedir, 'topo'),
            encoding=encoding,
            mode='w',
            consolidated=True
        )
        del topo
