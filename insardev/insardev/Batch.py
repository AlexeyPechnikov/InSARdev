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
from __future__ import annotations
from .utils_torch import serialize_gpu
from .BatchCore import BatchCore
import numpy as np
import xarray as xr
from . import utils_xarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Stack import Stack
    import inspect


def _apply_goldstein_for_dask(phase_block, corr_block, psize, threshold, device):
    """Module-level function for Goldstein filter blockwise operation (DEPRECATED).

    Defined at module level to avoid dask serialization issues with nested functions.
    Closures capturing variables can cause memory explosions in dask workers.
    """
    return BatchComplex._goldstein(phase_block, corr_block, psize=psize,
                                   threshold=threshold, device=device)


def _apply_goldstein_2d_for_dask(phase_block, corr_block, psize=32, threshold=0.5, device='cpu'):
    """Module-level function for Goldstein filter map_overlap operation.

    Defined at module level to avoid dask serialization issues with nested functions.
    Handles both 2D (y, x) and 3D (1, y, x) blocks - _goldstein() handles squeeze/unsqueeze.

    Parameters
    ----------
    phase_block : np.ndarray
        Complex array from dask, shape (y, x) or (1, y, x)
    corr_block : np.ndarray
        Real array from dask, shape (y, x) or (1, y, x)
    psize : int or dict
        Patch size for the filter
    threshold : float
        Minimum fraction of valid pixels
    device : str
        PyTorch device

    Returns
    -------
    np.ndarray
        Filtered complex array with same shape as input
    """
    # _goldstein handles (1, y, x) -> squeeze -> process -> unsqueeze
    return BatchComplex._goldstein(phase_block, corr_block, psize=psize,
                                   threshold=threshold, device=device)


def _apply_velocity_pairs_block(data_block, dt_years, min_valid, device, weight_block=None):
    """Module-level so dask can serialise it. One weighted regression per pixel.

    Returns (2, chunk_y, chunk_x): [0] velocity per year, [1] RMSE of the fit.
    """
    import numpy as np
    dt_years = np.asarray(dt_years, dtype=np.float32)
    vel, rmse = Batch._velocity_pairs_torch(data_block, dt_years, min_valid=min_valid,
                                            weight=weight_block, device=device)
    return np.stack([vel, rmse], axis=0).astype(np.float32)


class Batch(BatchCore):
    _velocity_note_shown = False

    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off only the real 2D vars from Stack
        if isinstance(mapping, Stack):
            #print ('Batch __init__: Stack')
            real_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                # keep only non-complex data_vars that live on the ('y','x') grid
                # and include 1D non-complex variables (e.g., per-axis metadata)
                # the 2D grid, plus every non-gridded var whatever its rank:
                # per-date POLYNOMIAL COEFFICIENTS are ('date', '<name>_coef'),
                # and the old `len(dims) == 1` test dropped them, taking
                # `incidence`, `d_drho_dh`, `baseline_model` and the orbit
                # polynomials out of the transform view that exists to carry them
                real_vars = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind != 'c'
                    and (
                        tuple(ds[v].dims) == ('y', 'x')
                        or len(ds[v].dims) == 1
                        or not ({'y', 'x'} & set(ds[v].dims))
                    )
                ]
                real_dict[key] = ds[real_vars]
            mapping = real_dict
        #print('Batch __init__ mapping', mapping or {}, '\n')
        # delegate to your base class for the actual init
        super().__init__(mapping or {})
    
    def clip(self, min=None, max=None, **kwargs):
        """
        used for correlation in [0,1] range
        """
        return BatchUnit(super().clip(min=min, max=max, **kwargs))

    @staticmethod
    def _compute_rgb(copol: np.ndarray, xpol: np.ndarray,
                     gamma: float = 1.0, brightness: float = 2.0,
                     quantile: list = None) -> np.ndarray:
        """
        Compute RGB composite from co-pol and cross-pol arrays.

        Parameters
        ----------
        copol : np.ndarray
            Co-polarization data (HH or VV), shape (..., y, x)
        xpol : np.ndarray
            Cross-polarization data (HV or VH), shape (..., y, x)
        gamma : float
            Gamma correction (>1 brightens dark areas)
        brightness : float
            Linear brightness multiplier
        quantile : list
            Quantile range for normalization, default [0.02, 0.98]

        Returns
        -------
        np.ndarray
            RGB array as float32 [0-1], shape (..., y, x, 3)
        """
        # Normalize each channel to [0, 1] using quantile stretch
        def normalize_channel(data):
            valid = data[np.isfinite(data)]
            if len(valid) == 0:
                return np.zeros_like(data)
            q_vals = quantile if quantile is not None else [0.02, 0.98]
            if np.isscalar(q_vals):
                q_vals = [q_vals, 1 - q_vals] if q_vals < 0.5 else [1 - q_vals, q_vals]
            q = np.nanquantile(valid, q_vals)
            vmin_ch, vmax_ch = q[0], q[-1]
            if vmax_ch <= vmin_ch:
                vmax_ch = vmin_ch + 1e-10
            normalized = (data - vmin_ch) / (vmax_ch - vmin_ch)
            return np.clip(normalized, 0, 1)

        # R=copol, G=xpol, B=copol
        r_norm = normalize_channel(copol)
        g_norm = normalize_channel(xpol)
        b_norm = normalize_channel(copol)

        # Apply gamma correction
        if gamma != 1.0:
            r_norm = np.power(r_norm, 1.0 / gamma)
            g_norm = np.power(g_norm, 1.0 / gamma)
            b_norm = np.power(b_norm, 1.0 / gamma)

        # Handle NaN (set to 0)
        nan_mask = ~np.isfinite(copol) | ~np.isfinite(xpol)
        r_norm = np.where(nan_mask, 0, r_norm)
        g_norm = np.where(nan_mask, 0, g_norm)
        b_norm = np.where(nan_mask, 0, b_norm)

        # Stack to RGB
        rgb_float = np.stack([r_norm, g_norm, b_norm], axis=-1)

        # Apply brightness
        if brightness != 1.0:
            rgb_float = rgb_float * brightness
            rgb_float = np.clip(rgb_float, 0, 1)

        return rgb_float.astype(np.float32)

    def plot(
        self,
        cmap = 'turbo',
        alpha = 0.5,
        caption = None,
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["alpha"] = alpha
        kwargs["caption"] = caption
        return super().plot(*args, **kwargs)

    def plot2(self, *args, **kwargs):
        """
        Plot dual-pol RGB composite (shortcut for plot(composite=True)).

        This is a convenience method for dual-polarization data that creates
        an RGB composite where R=co-pol, G=cross-pol, B=co-pol.

        All arguments are passed to plot() with composite=True.

        See Also
        --------
        plot : Full plotting method with all options.
        """
        kwargs["composite"] = True
        return self.plot(*args, **kwargs)

    def rgb(self, gamma: float = 1.0, brightness: float = 2.0, quantile: list = None):
        """
        Create RGB composite from dual-pol data as xarray DataArray.

        Standard dual-pol RGB decomposition: R=co-pol, G=cross-pol, B=co-pol
        - Magenta/pink: high co-pol, low cross-pol (surface scattering, urban)
        - Green: high cross-pol (volume scattering, vegetation)
        - White/gray: both high (mixed scattering)
        - Dark: both low (smooth surfaces, water)

        Parameters
        ----------
        gamma : float, optional
            Gamma correction for brightness. Default 1.0.
            Values > 1 brighten dark areas, < 1 increase contrast.
        brightness : float, optional
            Linear brightness multiplier. Default 2.0.
        quantile : list, optional
            Quantile range for normalization. Default [0.02, 0.98].

        Returns
        -------
        xr.DataArray
            RGB array with dims (band, y, x) or (date/pair, band, y, x).
            Values are uint8 [0-255]. NaN pixels have value 0.

        Examples
        --------
        >>> model = stack.fit3d()
        >>> # modelled displacement per date, unwrapped
        >>> disp  = stack.predict(model=model).displacement_los(stack.transform())
        >>> # remove the modelled ground motion, keeping topography
        >>> ground = stack * stack.predict(model=model).iexp(-1)
        >>> # remove the whole model, topography included
        >>> resid  = stack * stack.predict(model=model, baseline='BPR').iexp(-1)
        """
        import numpy as np
        import xarray as xr
        import dask
        from insardev_toolkit import progressbar

        # Check for exactly 2 polarizations
        sample = next(iter(self.values()))
        polarizations = [v for v in sample.data_vars
                        if sample[v].dims[-2:] == ('y', 'x')]
        if len(polarizations) != 2:
            raise ValueError(f"rgb() requires exactly 2 polarizations, found {len(polarizations)}: {polarizations}")

        pol1, pol2 = polarizations[0], polarizations[1]

        # Get stack variable (date or pair)
        stackvar = list(sample[pol1].dims)[0] if len(sample[pol1].dims) > 2 else None

        # Merge to single dataset
        ds = self.to_dataset()
        da_copol = ds[pol1]
        da_xpol = ds[pol2]

        if stackvar is None:
            stackvar = 'fake'
            da_copol = da_copol.expand_dims({stackvar: [0]})
            da_xpol = da_xpol.expand_dims({stackvar: [0]})

        # Materialize
        da_copol, da_xpol = dask.persist(da_copol, da_xpol)
        progressbar([da_copol, da_xpol], desc='Computing RGB composite'.ljust(25))

        # Compute RGB using shared method from BatchCore
        copol = da_copol.values
        xpol = da_xpol.values
        rgb_float = Batch._compute_rgb(copol, xpol, gamma=gamma,
                                       brightness=brightness, quantile=quantile)
        rgb_uint8 = (rgb_float * 255).astype(np.uint8)

        # Create DataArray with band-first order for rasterio compatibility
        if stackvar == 'fake':
            # Remove fake dimension: (1, y, x, 3) -> (y, x, 3) -> (3, y, x)
            rgb_uint8 = rgb_uint8[0]
            rgb_da = xr.DataArray(
                rgb_uint8,
                dims=['y', 'x', 'band'],
                coords={'y': da_copol.y, 'x': da_copol.x, 'band': ['R', 'G', 'B']}
            ).transpose('band', 'y', 'x')
        else:
            rgb_da = xr.DataArray(
                rgb_uint8,
                dims=[stackvar, 'y', 'x', 'band'],
                coords={stackvar: da_copol[stackvar], 'y': da_copol.y, 'x': da_copol.x, 'band': ['R', 'G', 'B']}
            ).transpose(stackvar, 'band', 'y', 'x')

        rgb_da.attrs['crs'] = self.crs
        return rgb_da

    def lee(self, *args, **kwargs):
        """
        Apply Enhanced Lee speckle filter to reduce noise while preserving edges.

        This method requires the insardev_backscatter extension package.
        """
        raise ImportError(
            "lee() requires insardev_backscatter extension"
        )

    @staticmethod
    def _solve_spd(A, b, eps=1e-12):
        """Batched small symmetric-positive-definite solve, elementwise Cholesky.

        A: (n, k, k), b: (n, k) -> (n, k). Singular systems yield NaN rather than a
        plausible wrong value; those pixels are masked by valid_count downstream.
        """
        import torch
        n_cols = A.shape[-1]
        L = [[None] * n_cols for _ in range(n_cols)]
        bad = None
        for i in range(n_cols):
            for j in range(i + 1):
                acc = A[:, i, j].clone()
                for k in range(j):
                    acc = acc - L[i][k] * L[j][k]
                if i == j:
                    neg = acc <= eps
                    bad = neg if bad is None else (bad | neg)
                    L[i][j] = torch.sqrt(acc.clamp_min(eps))
                else:
                    L[i][j] = acc / L[j][j]
        y = []
        for i in range(n_cols):
            acc = b[:, i].clone()
            for k in range(i):
                acc = acc - L[i][k] * y[k]
            y.append(acc / L[i][i])
        x = [None] * n_cols
        for i in reversed(range(n_cols)):
            acc = y[i].clone()
            for k in range(i + 1, n_cols):
                acc = acc - L[k][i] * x[k]
            x[i] = acc / L[i][i]
        out = torch.stack(x, dim=1)
        if bad is not None:
            out = torch.where(bad.unsqueeze(1), torch.full_like(out, float('nan')), out)
        return out

    @staticmethod
    def _velocity_note():
        """Say once that this is an estimator. It is not a measurement."""
        if Batch._velocity_note_shown:
            return
        Batch._velocity_note_shown = True
        print('NOTE: velocity() is a fast ESTIMATOR, not a precise measurement.',
              flush=True)

    @staticmethod
    @serialize_gpu
    def _velocity_pairs_torch(data, dt_years, min_valid=5, weight=None,
                              device='auto', debug=False):
        """Velocity from per-PAIR unwrapped displacement, one parameter per pixel.

        A pair value is a DIFFERENCE, d_ij = v * (t_j - t_i), so there is no
        constant to fit: this is a weighted regression through the origin,

            v = sum(w * d * dt) / sum(w * dt^2)

        which is better conditioned than the per-date [1, t] form and leaves
        nothing for a seasonal or a DEM term to trade against.

        It is as accurate as the full per-date route without the network
        inversion, and a coherence weight improves it further. The weight is
        per-PAIR, exactly like the correlation this pipeline already produces,
        so it lines up 1:1 with the data.
        """
        import torch
        import numpy as np

        dev = Batch._get_torch_device(device)
        original_shape = data.shape
        n_pairs = original_shape[0]
        data_2d = data.reshape(n_pairs, -1) if len(original_shape) == 3 else data

        y = torch.from_numpy(np.ascontiguousarray(data_2d, dtype=np.float32)).to(dev)
        dt = torch.from_numpy(np.asarray(dt_years, np.float32)).to(dev).unsqueeze(1)

        nan_mask = torch.isnan(y)
        valid_count = (~nan_mask).sum(dim=0)
        if weight is None:
            w = (~nan_mask).float()
        else:
            g = np.asarray(weight, np.float32)
            if g.ndim == 3:
                g = g.reshape(g.shape[0], -1)
            g = torch.from_numpy(np.ascontiguousarray(g)).to(dev)
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 0.999)
            # coherence -> phase precision; the standard 1/sigma^2 weight
            w = (~nan_mask).float() * (g * g) / (1.0 - g * g + 1e-6)
        y_filled = torch.where(nan_mask, torch.zeros_like(y), y)

        den = (w * dt * dt).sum(dim=0)
        num = (w * dt * y_filled).sum(dim=0)
        # A pixel whose pairs carry no time spread cannot yield a rate.
        solvable = (valid_count >= max(int(min_valid), 1)) & (den > 0)
        velocity = torch.where(solvable, num / torch.where(den > 0, den,
                                                           torch.ones_like(den)),
                               torch.full_like(den, float('nan')))

        resid = (y_filled - velocity.unsqueeze(0) * dt) * (~nan_mask).float()
        # one parameter consumed, so divide by n-1, not n. Dividing by n
        # under-reports the residual, badly so at small sample counts.
        dof = (valid_count - 1).clamp(min=1).float()
        rmse = torch.sqrt((resid ** 2).sum(dim=0) / dof)
        rmse = torch.where(solvable, rmse, torch.full_like(rmse, float('nan')))

        vel_np = velocity.cpu().numpy()
        rmse_np = rmse.cpu().numpy()
        if len(original_shape) == 3:
            vel_np = vel_np.reshape(original_shape[1], original_shape[2])
            rmse_np = rmse_np.reshape(original_shape[1], original_shape[2])
        if dev.type == 'mps':
            torch.mps.empty_cache()
        elif dev.type == 'cuda':
            torch.cuda.empty_cache()
        return vel_np, rmse_np


    def velocity(self, weight=None, min_valid=5, device='auto', debug=False) -> "Batches":
        """
        FAST velocity estimate from per-PAIR unwrapped displacement.

        This is a preview, deliberately cheap: it runs BEFORE lstsq, so it costs
        one weighted regression per pixel instead of a network inversion. Once
        lstsq has run there is nothing left to estimate quickly -- use
        detrend1d()/fit3d() for an accurate rate on inverted data.

        A pair value is a difference, d_ij = v*(t_j - t_i), so the fit has no
        intercept: v = sum(w*d*dt)/sum(w*dt^2).

        Parameters
        ----------
        weight : BatchUnit or None
            Per-pair correlation. Improves the rate materially, and it aligns
            1:1 with the data because both are per-pair.
        min_valid : int
            Minimum valid pairs; below it the pixel is NaN.

        Returns
        -------
        Batches
            Batches[velocity, rmse]; velocity per year in the input units,
            rmse in the input units.

        Examples
        --------
        >>> displacement = phase_detrend.displacement_los(stack.transform())
        >>> velocity, rmse = displacement.velocity(weight=corr)
        """
        import dask
        import dask.array as da
        import numpy as np
        import pandas as pd
        import xarray as xr

        BatchCore._require_lazy(self, 'velocity')
        vel_results, rmse_results = {}, {}
        for key in self.keys():
            ds = self[key]
            vel_vars, rmse_vars = {}, {}
            for var in [v for v in ds.data_vars
                        if 'y' in ds[v].dims and 'x' in ds[v].dims]:
                da_ = ds[var]
                if 'pair' not in da_.dims:
                    raise TypeError(
                        f"velocity() operates on per-PAIR displacement, but '{var}' "
                        f"has dims {tuple(da_.dims)}. Call it BEFORE lstsq; an "
                        "already-inverted per-date series needs detrend1d() instead.")
                ref = pd.to_datetime(da_.coords['ref'].values)
                rep = pd.to_datetime(da_.coords['rep'].values)
                dt_years = np.array([(b - a).total_seconds() / (365.25 * 86400)
                                     for a, b in zip(ref, rep)], dtype=np.float32)

                Batch._velocity_note()

                mem_per_pixel = len(dt_years) * 4 * 3
                ay, ax = dask.array.core.normalize_chunks(
                    'auto', (da_.y.size, da_.x.size),
                    dtype=np.dtype(f'V{mem_per_pixel}'))
                cy, cx = ay[0], ax[0]
                da_ = da_.chunk({'pair': -1, 'y': cy, 'x': cx})
                d_dask = da_.data

                if weight is None:
                    def _blk(b, _dt=dt_years):
                        return _apply_velocity_pairs_block(b, _dt, min_valid, device)
                    res = da.map_blocks(_blk, d_dask, dtype=np.float32,
                                        drop_axis=0, new_axis=0,
                                        chunks=(2,) + d_dask.chunks[1:])
                else:
                    w_dask = weight[key][var].chunk(
                        {'pair': -1, 'y': cy, 'x': cx}).data
                    def _blk(b, wb, _dt=dt_years):
                        return _apply_velocity_pairs_block(b, _dt, min_valid,
                                                           device, weight_block=wb)
                    res = da.map_blocks(_blk, d_dask, w_dask, dtype=np.float32,
                                        drop_axis=0, new_axis=0,
                                        chunks=(2,) + d_dask.chunks[1:])

                coords = {'y': da_.y, 'x': da_.x}
                vel_vars[var] = xr.DataArray(res[0], dims=['y', 'x'], coords=coords)
                rmse_vars[var] = xr.DataArray(res[1], dims=['y', 'x'], coords=coords)

            for out, src in ((vel_results, vel_vars), (rmse_results, rmse_vars)):
                out_ds = xr.Dataset(src)
                out_ds.attrs = ds.attrs
                import rioxarray
                if ds.rio.crs is not None:
                    out_ds = out_ds.rio.write_crs(ds.rio.crs)
                out[key] = out_ds

        return Batches((Batch(vel_results), Batch(rmse_results)))

    @staticmethod
    def _ref_index(ref, dates):
        """Which acquisition a `ref` names, as an index into `dates`.

        Accepts what S1.transform(ref=...) accepts and one thing more:

          None                 the model's own anchor -- nothing is subtracted
          int                  position in the stack, Python-style, so 0 is the
                               first acquisition and -1 the last
          str / datetime       an acquisition date, matched to the day

        A date that is not in the stack raises rather than silently picking a
        neighbour: predict() re-references the output to it, and a near miss
        would shift every plane by a constant nobody asked for.
        """
        import numpy as np
        import pandas as pd
        d = np.asarray(dates).astype('datetime64[D]')
        if isinstance(ref, (bool, np.bool_)):
            raise TypeError(
                "ref takes a date or a stack position, not a bool. Use ref=0 "
                "for the first acquisition and ref=None for the model's anchor.")
        if isinstance(ref, (int, np.integer)):
            i = int(ref)
            if not -len(d) <= i < len(d):
                raise IndexError(
                    f"ref={ref} is out of range for {len(d)} acquisitions.")
            return i % len(d)
        try:
            want = np.datetime64(pd.to_datetime(ref), 'D')
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"ref must be None, an int position or a date, got {ref!r}") from e
        hit = np.nonzero(d == want)[0]
        if not len(hit):
            raise KeyError(
                f"ref={ref!r} is not one of this stack's acquisitions "
                f"({str(d[0])} .. {str(d[-1])}, {len(d)} dates).")
        return int(hit[0])

    @staticmethod
    @serialize_gpu
    def _fit1d_pairs_torch(data, A, weight=None, device='auto',
                           dh_prior=None, dh_col=None):
        """Weighted least squares of a per-pixel model on per-PAIR unwrapped phase.

        A is (n_pairs, k) and IDENTICAL for every pixel -- only the weights vary
        with the data -- so the normal equations are two matrix products rather
        than a loop:

            AtWA = (outer(a_p, a_p) flattened)^T @ w      (k*k, N)
            AtWy = A^T @ (w * y)                         (k, N)

        and the whole fit is those two GEMMs plus a batched k x k solve. There is
        no lattice and no refinement: unwrapped phase makes the objective convex,
        which is the entire reason this is cheaper than the wrapped route it
        replaces.

        NO 2 PI AMBIGUITY REFIT. Removing integer cycles by
        round(residual / 2 pi) is exact when the residual is dominated by an
        unwrapping error and destructive when it is dominated by noise, and
        which one holds is decided by the coherence of the stack, not by the
        estimator. At low coherence it chases noise past pi rather than cycles.
        Unwrapping errors belong to the unwrapper and to phase closure over
        triplets, which sees them without a model in the way.

        Columns are scaled to unit norm before the solve and the solution scaled
        back: dt is O(1) years while ele2phase is O(1e-4), and a Cholesky on the
        raw normal equations of two columns eight orders of magnitude apart is
        not a solve, it is a coin toss.

        Returns (theta (k, N), gamma (N,), rmse (N,)).
        """
        import torch
        import numpy as np

        dev = Batch._get_torch_device(device)
        shape = data.shape
        n_p = shape[0]
        n_pix = int(np.prod(shape[1:]))
        y = torch.from_numpy(np.ascontiguousarray(
            data.reshape(n_p, -1), dtype=np.float32)).to(dev)
        Am = torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)).to(dev)
        k = int(Am.shape[1])

        nan_mask = torch.isnan(y)
        valid_count = (~nan_mask).sum(dim=0)
        if weight is None:
            w = (~nan_mask).float()
        else:
            g = np.asarray(weight, np.float32)
            if g.ndim == 3:
                g = g.reshape(g.shape[0], -1)
            g = torch.from_numpy(np.ascontiguousarray(g)).to(dev)
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 0.999)
            # gamma^2, BOUNDED. The Cramer-Rao weight gamma^2/(1-gamma^2) is
            # the right one for a well-observed pixel and the wrong one here: it
            # is unbounded, so at low correlation it spans orders of magnitude
            # within a pixel and the rate is decided by that pixel's single best
            # pair. Clamping does not fix it, because the damage is the RATIO
            # between pairs, not the maximum. Weighting earns its keep where
            # correlation is high and uniform; where it is low, leaving it out
            # does better, which is why the default is None rather than the
            # correlation the caller happens to have.
            w = (~nan_mask).float() * (g * g)
        y = torch.where(nan_mask, torch.zeros_like(y), y)
        finite = (~nan_mask).float()

        scale = Am.norm(dim=0).clamp(min=1e-30)
        An = Am / scale
        # A PRIOR ON THE HEIGHT, not a robust loss. Adding the DEM column cost
        # held-out accuracy at EVERY quantile alike, which is variance, not
        # outliers. L1-IRLS and Huber were tried and both came back worse,
        # confirming there is no tail to trim.
        #
        # What a poorly-observed nuisance parameter needs is shrinkage. DEM
        # error is bounded by the DEM, not by this stack, so sigma_dh =
        # max_dh/3 is a property of the terrain model rather than a number to
        # tune, and
        #
        #     lambda = sigma_phi^2 / (meter2rad * sigma_dh)^2
        #
        # is the Bayesian weight it implies. It adapts by construction: where
        # the baselines resolve the height the prior is inert, and where they do
        # not it shrinks the column away.
        T = (An[:, :, None] * An[:, None, :]).reshape(n_p, k * k)
        # ALL PAIRS OR NOTHING. A pixel the unwrapper dropped from some pairs
        # is a degraded network, not a smaller one: the surviving subset is
        # whatever happened to stay coherent, so the model is fitted to a
        # different experiment at every such pixel and the result is not
        # comparable with its neighbours.
        #
        # The matrix is still tested below: linalg.solve raises on the WHOLE
        # batch when one element is singular, and a full pair set can still be
        # singular if every weight is zero.
        solvable = (valid_count == n_p) & (w.sum(dim=0) > 0)
        eye = torch.eye(k, device=dev, dtype=torch.float32).expand(n_pix, k, k)

        # WHICH column carries the height. It is index -1 only when the
        # seasonal is off; with the annual in the model the last column is its
        # sine, and shrinking that instead would leave the height untouched.
        _hc = k - 1 if dh_col is None else int(dh_col)
        _use_prior = bool(dh_prior) and 0 <= _hc < k
        lam_h = None
        # L1 / IRLS, the same reweighting lstsq() applies to the same input:
        # unwrapping error arrives as a whole cycle on one pair, which least
        # squares spreads over every parameter.
        w0 = w
        n_irls = 5
        eps_irls = 0.1
        for _pass in range(2 if _use_prior else 1):
          for _it in range(n_irls):
            AtWA = (T.T @ w).T.reshape(n_pix, k, k)
            AtWy = (An.T @ (w * y)).T
            if lam_h is not None:
                AtWA = AtWA.clone()
                AtWA[:, _hc, _hc] = AtWA[:, _hc, _hc] + lam_h
            AtWA_safe = torch.where(solvable[:, None, None], AtWA, eye)
            ok = solvable
            if dev.type == 'mps':
                # _solve_spd returns NaN for a singular element instead of
                # raising, so it needs no pre-test
                th = Batch._solve_spd(AtWA_safe, AtWy)
            else:
                # cholesky_ex REPORTS failure in `info` rather than raising, and
                # succeeding is exactly the property linalg.solve requires here,
                # so the un-solvable elements are swapped for the identity
                # before the solver ever sees them
                _, info = torch.linalg.cholesky_ex(AtWA_safe)
                ok = solvable & (info == 0)
                AtWA_safe = torch.where(ok[:, None, None], AtWA_safe, eye)
                th = torch.linalg.solve(AtWA_safe, AtWy.unsqueeze(-1)).squeeze(-1)
            ok = ok & torch.isfinite(th).all(dim=1)
            th = torch.where(ok[:, None], th, torch.full_like(th, 0.0))
            resid = (y - An @ th.T) * finite
            if _it + 1 < n_irls:
                # reweight and go round again; the last pass keeps the weights it
                # converged with so `resid` below is the one the model reports
                w = w0 / torch.clamp(torch.abs(resid), min=eps_irls)
                w = torch.where(finite > 0, w, torch.zeros_like(w))
            if _use_prior and _pass == 0:
                # sigma_phi per pixel from the unregularised pass, then the
                # Bayesian weight the height prior implies. The prior lives on
                # the NORMALISED column, so it means the same thing whatever
                # the baselines span.
                dof = (valid_count - k).clamp(min=1).float()
                s2 = (resid ** 2).sum(dim=0) / dof
                # An = A/s and theta_n = s*theta, so a penalty lambda on the
                # UNNORMALISED height becomes lambda/s^2 on the normalised one
                lam_h = s2 / float(dh_prior) ** 2 / (scale[_hc] ** 2)
                w = w0          # second pass re-runs IRLS from the base weights

        # LINEAR RESIDUALS, not circular. gamma used to be the resultant
        # |sum w e^{i r}| / sum w, which is right for wrapped phase and wrong
        # here: this phase is UNWRAPPED, so a residual of 2 pi is a whole cycle
        # of error and cos(2 pi) = 1 counts it as a perfect sample. With cycle
        # slips present it reports near-perfect agreement while the true
        # residual RMS is large.
        #
        # Unwrapping error is the dominant error mode in multilooked pairs, so
        # a quality number that cannot see it is worse than none: it certifies
        # exactly the pixels a caller most needs to drop. rmse is now the
        # weighted RMS of the residual itself, inflated by n/(n-k) for the
        # parameters spent, and coherence is its exact inverse transform
        # exp(-rmse^2/2) -- the same relation fit3d's pair satisfies, so the
        # two remain comparable while both now respond to a cycle slip.
        # The estimate is robust; the diagnostic must not be. rmse/coherence are
        # computed from the BASE weights w0, never the IRLS weights: those were
        # chosen to suppress the samples that fit worst, so scoring against them
        # would report the fit the reweighting engineered.
        wsum = w0.sum(dim=0)
        infl = valid_count.float() / (valid_count - k).clamp(min=1).float()
        mse = (w0 * resid * resid).sum(dim=0) / wsum.clamp(min=1e-12)
        rmse = torch.sqrt(torch.clamp(mse, min=0.0) * infl)
        gam = torch.exp(-0.5 * rmse * rmse).clamp(0.0, 1.0)

        # A box bound cannot help a convex fit -- it can only distort it. On
        # unwrapped phase the optimum is unique, so a constrained answer is a
        # boundary point and the residual is forced into the other parameters.
        # The bound therefore REJECTS here (see the caller).

        nanv = torch.full_like(gam, float('nan'))
        gam = torch.where(ok, gam, nanv)
        rmse = torch.where(ok, rmse, nanv)
        th = torch.where(ok[:, None], th / scale[None, :],
                         torch.full_like(th, float('nan')))

        out = (th.T.cpu().numpy(), gam.cpu().numpy(), rmse.cpu().numpy())
        if dev.type == 'mps':
            torch.mps.empty_cache()
        elif dev.type == 'cuda':
            torch.cuda.empty_cache()
        return out

    def fit1d(self, weight=None, baseline: str = 'BPR', transform=None,
              max_dh: float = 30.0, max_dv: float = 100.0,
              min_dv: 'float | None' = None,
              max_seasonal: float = 0.0, device: str = 'auto',
              debug: bool = False) -> 'Batch':
        """
        Full per-pixel model on UNWRAPPED per-PAIR phase -- NO network.

        The unwrapped twin of BatchComplex.fit1d(): the same model and the same
        output, fitted by weighted least squares instead of by coherence
        maximisation, because unwrapped phase makes the objective convex. Returns
        the MODEL ONLY, named and scaled as fit3d() names and scales it, so
        predict(model) is the one inverse for all three.

        WHY PAIRS AND NOT AN INVERTED SERIES. Multilooked phase has no zero
        closure, so a per-date series is not recoverable without committing to
        an inversion first. It does not need to be: every pair is the SAME
        absolute-time model differenced at two epochs,

            phi_p = -velocity * dt_p - height * ele2phase_p
                    - Re(seasonal) * dcos_p - Im(seasonal) * dsin_p

        with dt = t_rep - t_ref, ele2phase_p = dBPR_p / median(R sin(incidence)),
        and dcos_p = cos(2 pi t_rep) - cos(2 pi t_ref). The network the lstsq()
        inversion would solve is already IN those columns. Fitting here rather
        than after an inversion avoids a rank-deficient solve, a datum choice,
        and the propagation of one bad pair down a whole cumulative series --
        it is one bad ROW instead.

        The signs are the interferometric ones and are not a convention this
        function chose: interferogram() forms ref * conj(rep) while pairs() forms
        dt and dBPR as rep - ref, so a pair carries MINUS the model difference.

        IT DOES NOT FILTER. A long-wavelength screen in the pairs biases every
        parameter, and removing it is the caller's decision and the caller's
        scale:

        >>> model = (phase - phase.gaussian(wavelength=40000)).fit1d(weight=corr)

        Parameters
        ----------
        weight : BatchUnit or None
            Per-pair correlation, aligned 1:1 with the pairs. Enters as
            gamma^2/(1-gamma^2), the phase-precision weight.
        baseline : str
            Per-pair perpendicular baseline coordinate, default 'BPR'. Without
            it the height column is dropped and `height` comes back NaN -- which
            is only right when the topographic phase is already removed.
        transform : Batch or None
            Carries the 2D `rng` grid needed for median(R sin(incidence)). The
            grid does not ride with pair data; without it the range is taken
            from num_rng_bins/2, which is +0.43% here and biases height by that
            and the rate not at all.
        min_dv : float or None
            Lower limit of the rate in mm/yr of LOS displacement -- the sign
            `displacement_los()` reports, so NEGATIVE IS SUBSIDENCE and
            `min_dv=-70, max_dv=0` admits subsidence only. None (default) keeps the
            symmetric gate |v| <= max_dv, whose meaning does not depend on sign.
            A pixel solving outside the range returns NaN: the fit is convex, so
            there is no better solution inside the range to fall back on, and
            clamping to the bound would push the error into the other parameters.
        max_dh, max_dv, max_seasonal : float
            The same bounds BatchComplex.fit1d() takes, in the same units --
            metres, mm/yr, and mm of LOS half-amplitude -- and meaning the same
            thing: a pixel solving outside them comes back NaN rather than a
            plausible wrong number. `max_seasonal=0` leaves the annual out of
            the model entirely, exactly as it does there.

            They are BOUNDS here and not also a search range: the objective is
            convex on unwrapped phase, so there is no lattice to size and no
            guard band to keep a peak off a boundary -- which is why step_dh
            and step_dv have no counterpart in this signature. Nothing is
            clipped to the bound; it is reported or it is NaN.

            The annual is identifiable only when the pairs sample different
            times of year. Over a span well short of one, the annual columns
            collapse onto the rate column; the design report below states by
            how much rather than letting the split look decided.

        Returns
        -------
        Batch
            The model, named and scaled exactly as fit3d() and
            BatchComplex.fit1d():

              `velocity`   rad/yr
              `height`     rad per unit ele2phase, NaN with no baseline
              `seasonal`   complex rad, 0 when max_seasonal=0
              `coherence`  |sum w exp(i r)| / sum w about the reported model
              `rmse`       sqrt(-2 ln coherence), radians, n/(n-k) inflated

            plus a scalar `date` coordinate: the epoch the model is
            referenced to, which is what every fit here carries so predict()
            never has to guess. A pairs batch holds only baseline DIFFERENCES,
            so the master that fit3d() anchors on is not recoverable from it and
            the median acquisition is used instead -- recorded rather than
            assumed. Rate and height are indifferent to the origin; the annual
            is not, and a model fitted on one origin and removed on another
            leaves a residual annual of 2|sin(pi delta)| |seasonal|.

            NO `conncomp`: every pixel is solved alone.

            A pixel rejected by a bound has velocity, height and seasonal NaN
            while `coherence` and `rmse` survive, which is _3d_arc_fit()'s
            convention: they describe the fit that was attempted, and a caller
            diagnosing why a pixel was refused needs them.

        Examples
        --------
        >>> model = (phase - phase.gaussian(wavelength=40000)).fit1d(weight=corr)
        >>> noise = phase - stack.predict(model, baseline='BPR')
        """
        import numpy as np
        import pandas as pd
        import xarray as xr
        import dask
        import dask.array as da

        BatchCore._require_lazy(self, 'fit1d')
        model_result = {}
        for key in self.keys():
            ds = self[key]
            # a PAIR dim, not merely a grid: `rng` and friends are (y, x)
            # and would otherwise be counted as a second polarisation
            pols = [v for v in ds.data_vars
                    if 'pair' in ds[v].dims
                    and 'y' in ds[v].dims and 'x' in ds[v].dims]
            if not pols:
                raise TypeError(
                    f"fit1d() found no (pair, y, x) variables in burst {key}. "
                    "It operates on per-PAIR unwrapped phase; a per-date "
                    "complex stack needs BatchComplex.fit1d().")
            if len(pols) > 1:
                raise ValueError(
                    f"fit1d() fits ONE polarisation; burst '{key}' carries "
                    f"{len(pols)}: {pols}. The model variables are named by "
                    "quantity alone (velocity, height, ...), so two "
                    "polarisations would collide. Select one first, e.g. "
                    "batch[['VV']].")
            pol = pols[0]
            da_ = ds[pol]
            if da_.dims[0] != 'pair':
                da_ = da_.transpose('pair', ...)

            rd = (np.asarray(pd.to_datetime(da_.coords['ref'].values).values)
                  .astype('datetime64[D]').astype(np.float64))
            pd_ = (np.asarray(pd.to_datetime(da_.coords['rep'].values).values)
                   .astype('datetime64[D]').astype(np.float64))
            # the median acquisition, since the master is not recoverable from
            # differences; days, as every other origin in this library
            t0_day = float(np.median(np.unique(np.concatenate([rd, pd_]))))
            t_ref = (rd - t0_day) / 365.25
            t_rep = (pd_ - t0_day) / 365.25
            dt = t_rep - t_ref

            if 'radar_wavelength' not in ds:
                raise KeyError(
                    f"fit1d() needs 'radar_wavelength' on '{key}' to build the "
                    "height column and to read out in radians.")
            lam = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
            # only used to state the design's precision in physical units --
            # the columns and the solution are radians throughout
            meter2rad = 4.0 * np.pi / lam

            # ele2phase per PAIR: dBperp / (R sin(incidence)), from
            # elevation_phase() = 4 pi / (lambda R sin(inc))
            bp = None
            for src in (da_.coords, ds):
                if baseline in src:
                    bp = np.asarray(src[baseline].values, dtype=float).ravel()
                    break
            e2p = None
            if bp is not None and bp.shape == dt.shape:
                _fac = Batch._elevation_phase_approximate(
                    transform if transform is not None else self)[key]
                e2p = bp / ((4.0 * np.pi / lam) / _fac)
            if e2p is None:
                print(f"fit1d(): no {baseline!r} for '{key}' -- the height "
                      "column is dropped and the topographic phase stays in the "
                      "residual.", flush=True)

            # THE PAIR CONVENTION, which is this method's own -- pairs are not
            # a per-date stack and this fit is not BatchComplex.fit1d.
            #
            #   phi_p = +velocity*dt - height*ele2phase_p + annual
            #
            # `velocity` is the rate of LOS displacement expressed in radians:
            # the sense lstsq() works in, since it sums intervals rep-ref
            # (the network inversion sums intervals rep-ref), and the sense
            # displacement_los() converts,
            # since its -lambda/4pi is derived for a pair. An SLC phase is
            # -(4pi/lambda)*r, so a pair ref*conj(rep) carries +m2r*dr while a
            # date series carries -m2r*dr -- opposite. Fitting the date sense
            # here makes model.displacement_los() report subsidence as uplift.
            #
            # `height` keeps the other sign: it is metres of DEM error, not a
            # displacement, and nothing converts it with displacement_los().
            cols = [dt]
            has_h = e2p is not None
            if has_h:
                cols.append(-e2p)
            fit_seasonal = bool(max_seasonal and max_seasonal > 0)
            if fit_seasonal:
                cols.append(np.cos(2 * np.pi * t_rep) - np.cos(2 * np.pi * t_ref))
                cols.append(np.sin(2 * np.pi * t_rep) - np.sin(2 * np.pi * t_ref))
            A = np.stack(cols, axis=1).astype(np.float64)
            k = A.shape[1]
            if A.shape[0] < k + 1:
                raise ValueError(
                    f"fit1d() needs more than {k} pairs to fit {k} parameters; "
                    f"burst '{key}' has {A.shape[0]}.")
            # HOW WELL THE PAIR SET DETERMINES EACH PARAMETER, decided once,
            # because the design is the same for every pixel. This is not the
            # residual -- `rmse` is that, and the two are independent. A short
            # stack fitted with an annual returns an UNBIASED rate with a huge
            # variance while the fit looks excellent: the rate is meaningless
            # and no residual statistic can say so. Var(theta) = sigma^2 (A'A)^-1, so
            # sigma_velocity = rmse * `rate` below, per pixel, in mm/yr.
            try:
                Cinv = np.linalg.inv(A.T @ A)
                sig = np.sqrt(np.clip(np.diag(Cinv), 0, None))
            except np.linalg.LinAlgError:
                sig = np.full(k, np.inf)
            s_v = float(sig[0]) / (meter2rad * 1e-3)
            s_h = (float(sig[1]) / meter2rad) if has_h else float('nan')
            # what the annual costs the rate: the same design without it
            vif = 1.0
            if fit_seasonal and k > (2 if has_h else 1):
                kk_ = 2 if has_h else 1
                A0 = A[:, :kk_]
                try:
                    vif = float(np.sqrt(np.linalg.inv(A0.T @ A0)[0, 0])
                                / max(float(sig[0]), 1e-30))
                    vif = 1.0 / max(vif, 1e-30)
                except np.linalg.LinAlgError:
                    vif = float('inf')
            terms = 'velocity' + (', height' if has_h else '') \
                + (', seasonal' if fit_seasonal else '')
            msg = (f"fit1d('{key}'): {A.shape[0]} pairs, {k} parameters "
                   f"({terms}), date {np.datetime64(int(t0_day), 'D')}\n"
                   f"  per radian of residual the design gives "
                   f"sigma_velocity {s_v:.3f} mm/yr"
                   + (f", sigma_height {s_h:.3f} m" if has_h else "")
                   + (f"; the annual inflates the rate by x{vif:.2f}"
                      if fit_seasonal else ""))
            if debug or not np.isfinite(s_v) or vif > 3.0:
                print(msg, flush=True)
            if vif > 3.0:
                print("  the annual is NOT separable from the rate on this "
                      "pair set -- too short a span for them to differ. The "
                      "rate stays unbiased but its error grows by that factor; "
                      "pass max_seasonal=0 if the annual is not wanted.",
                      flush=True)

            mem_per_pixel = A.shape[0] * 4 * 4
            ay, ax = dask.array.core.normalize_chunks(
                'auto', (da_.y.size, da_.x.size),
                dtype=np.dtype(f'V{mem_per_pixel}'))
            cy, cx = ay[0], ax[0]
            da_ = da_.chunk({'pair': -1, 'y': cy, 'x': cx})
            d_dask = da_.data
            w_dask = None
            if weight is not None:
                w_dask = weight[key][pol].transpose('pair', ...).chunk(
                    {'pair': -1, 'y': cy, 'x': cx}).data
                if w_dask.chunks != d_dask.chunks:
                    w_dask = w_dask.rechunk(d_dask.chunks)

            # max_dh is a 3-sigma bound on a DEM error, so it also names the
            # PRIOR the height is shrunk under. One number, both jobs: what the
            # caller will accept, and what the terrain model is known to be.
            _dh_prior = meter2rad * float(max_dh) / 3.0 if has_h else None

            def _blk(b, wb=None, _A=A, _k=k, _hh=has_h, _se=fit_seasonal,
                     _dev=device, _dbg=bool(debug), _m2r=meter2rad,
                     _mh=float(max_dh), _mv=float(max_dv),
                     _mnv=(None if min_dv is None else float(min_dv)),
                     _ms=float(max_seasonal), _dp=_dh_prior):
                shp = b.shape[1:]
                th, gam, rms = Batch._fit1d_pairs_torch(
                    b, _A, weight=wb, device=_dev, dh_prior=_dp,
                    dh_col=(1 if _hh else None))
                nanp = np.full(shp, np.nan, np.float32)
                vel = th[0].reshape(shp).astype(np.float32)
                hgt = th[1].reshape(shp).astype(np.float32) if _hh else nanp
                if _se:
                    j = 2 if _hh else 1
                    sea = (th[j].reshape(shp)
                           + 1j * th[j + 1].reshape(shp)).astype(np.complex64)
                else:
                    # zero, not NaN: the model holds no annual, and that is a
                    # value rather than an absence -- the same convention fit3d
                    # uses at max_seasonal=0
                    sea = np.zeros(shp, np.complex64)
                gam = gam.reshape(shp).astype(np.float32)
                rms = rms.reshape(shp).astype(np.float32)
                # THE BOUNDS ARE THE GUARANTEE, not a hint: outside them the
                # pixel is NaN, never clipped to the edge and reported as if it
                # had solved there. coherence and rmse are kept -- they say how
                # the refused fit behaved, which is what a caller needs to see.
                bad = ~np.isfinite(vel)
                if _mnv is None:
                    bad |= np.abs(vel) > _mv * _m2r * 1e-3
                else:
                    # the range is in the units the caller reads -- mm/yr of LOS
                    # displacement, the sign displacement_los() shows -- while the
                    # internal `velocity` runs the other way, so negate on the way in
                    lo = -max(_mnv, _mv) * _m2r * 1e-3
                    hi = -min(_mnv, _mv) * _m2r * 1e-3
                    bad |= (vel < lo) | (vel > hi)
                if _hh:
                    bad |= np.abs(hgt) > _mh * _m2r
                if _se:
                    bad |= np.abs(sea) > _ms * _m2r * 1e-3
                vel = np.where(bad, np.nan, vel).astype(np.float32)
                hgt = np.where(bad, np.nan, hgt).astype(np.float32)
                sea = np.where(bad, np.nan + 0j, sea).astype(np.complex64)
                return np.stack([vel.astype(np.complex64),
                                 hgt.astype(np.complex64),
                                 sea,
                                 gam.astype(np.complex64),
                                 rms.astype(np.complex64)], axis=0)

            args = (d_dask,) if w_dask is None else (d_dask, w_dask)
            res = da.map_blocks(_blk, *args, dtype=np.complex64,
                                drop_axis=0, new_axis=0,
                                chunks=(5,) + d_dask.chunks[1:],
                                meta=np.empty((0, 0, 0), np.complex64))

            coords = {kk: vv for kk, vv in da_.coords.items()
                      if kk in ('y', 'x', 'spatial_ref')}
            mvars = {}
            for nm_, arr_ in (('velocity', res[0].real.astype(np.float32)),
                              ('height', res[1].real.astype(np.float32)),
                              ('seasonal', res[2].astype(np.complex64)),
                              ('coherence', res[3].real.astype(np.float32)),
                              ('rmse', res[4].real.astype(np.float32))):
                mvars[nm_] = xr.DataArray(arr_, dims=('y', 'x'), coords=coords)
            mds = xr.Dataset(mvars, attrs=ds.attrs)
            mds = mds.assign_coords(date=np.datetime64(int(t0_day), 'D'))
            if 'spatial_ref' in ds.coords:
                mds = mds.assign_coords(spatial_ref=ds.spatial_ref)
            model_result[key] = mds
        return Batch(model_result)


    def predict(self, model, baseline: str = 'BPR', transform=None,
                ref=None) -> 'Batch':
        """
        Phase predicted by a model, on THIS batch's own pairs or dates.

        The inverse of fit1d(), and the way to see what a fit did not explain:

        >>> model = (phase - phase.gaussian(wavelength=40000)).fit1d(
        ...     transform=stack.transform())
        >>> noise = phase - phase.predict(model, transform=stack.transform())

        A PAIRS batch gets PER-PAIR phase, rebuilt in the pair convention the
        model is published in, so it subtracts from the pairs directly:

            phi_p = velocity*dt_p - height*ele2phase_p
                    + Re(seasonal)*dcos_p + Im(seasonal)*dsin_p

        A per-DATE batch gets per-date phase, which runs OPPOSITE in velocity
        and seasonal -- an SLC phase is -(4pi/lambda)r while a pair
        ref*conj(rep) carries +m2r*dr -- and identical in height, whose
        per-date term +height*ele2phase_d already differences to
        -height*ele2phase_p. That asymmetry is why no global sign converts one
        into the other.

        The epoch comes from the model's scalar `date` coordinate, which every
        fit records, so a model fitted against one origin is never removed
        against another. Only the annual cares, and it cares completely: a
        mismatch of delta years leaves a residual annual of
        2|sin(pi delta)|*|seasonal|.

        Parameters
        ----------
        model : Batch
            Output of fit1d() or fit3d(): `velocity`, `height`, `seasonal`.
        baseline : str
            Per-pair or per-date perpendicular baseline, default 'BPR'. None,
            or absent, drops the height term.
        transform : Batch or None
            Carries the 2-D `rng` grid for median(R sin(incidence)). It does
            not ride with pair data, so without it the range falls back to
            num_rng_bins/2 -- +0.43% here, which biases the height by that and
            the rate not at all. Same argument as fit1d()'s.

        ref : None, int, str or datetime
            Which acquisition reads zero in the returned per-date series.
            None keeps the model's own anchor -- the master, where B_perp is
            smallest and the height term vanishes with the baseline that
            carries it. That is the right anchor for the FIT and an awkward one
            to look at, since the series then runs negative before the master
            and positive after; `ref=0` re-reads the same model from the first
            acquisition, and a date string or datetime picks any other. Same
            spelling as S1.transform(ref=...).

            IT IS A DISPLAY CHOICE AND NOTHING MORE. Re-referencing shifts
            every date by one per-pixel constant, so every DIFFERENCE is
            untouched -- pair predictions, removals and velocities come out
            bit-identical whatever `ref` says. The fit is not re-referenced,
            only the picture. Meaningless on a PAIRS batch, where a pair is
            already a difference and the reference cancels, so passing it there
            raises rather than being quietly ignored.
        Returns
        -------
        Batch
            Real radians, on the MODEL's grid, over this batch's pairs or
            dates. The scatterer's own constant is not part of any model, so
            the prediction is right up to one constant per pixel -- which is
            what you want for removal, and what a difference cancels anyway.
        """
        import numpy as np
        import pandas as pd
        import xarray as xr

        out = {}
        for key in self.keys():
            ds = self[key]
            if key not in model:
                raise KeyError(f"predict(): the model has no burst '{key}'.")
            mds = model[key]
            missing = [v for v in ('velocity', 'height', 'seasonal') if v not in mds]
            if missing:
                raise KeyError(
                    f"predict() needs {missing} in the model for '{key}'. "
                    "Pass the Batch returned by fit1d() or fit3d().")

            grids = [v for v in ds.data_vars
                     if 'y' in ds[v].dims and 'x' in ds[v].dims
                     and ('pair' in ds[v].dims or 'date' in ds[v].dims)]
            if not grids:
                raise TypeError(
                    f"predict() found no (pair|date, y, x) variables in '{key}'.")
            pol = grids[0]
            da_ = ds[pol]
            per_pair = 'pair' in da_.dims

            if 'radar_wavelength' not in ds:
                raise KeyError(f"predict() needs 'radar_wavelength' on '{key}'.")
            lam = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
            meter2rad = 4.0 * np.pi / lam

            # the epoch the model was fitted against
            t0 = None
            if 'date' in mds.coords and mds.coords['date'].ndim == 0:
                t0 = float(np.asarray(mds.coords['date'].values)
                           .astype('datetime64[D]').astype(np.float64))

            if per_pair:
                rd = (np.asarray(pd.to_datetime(da_.coords['ref'].values).values)
                      .astype('datetime64[D]').astype(np.float64))
                pd_ = (np.asarray(pd.to_datetime(da_.coords['rep'].values).values)
                       .astype('datetime64[D]').astype(np.float64))
                if t0 is None:
                    t0 = float(np.median(np.unique(np.concatenate([rd, pd_]))))
                t_ref = (rd - t0) / 365.25
                t_rep = (pd_ - t0) / 365.25
                bp = None
                for src in (da_.coords, ds):
                    if baseline and baseline in src:
                        bp = np.asarray(src[baseline].values, dtype=float).ravel()
                        break
            else:
                dday = (np.asarray(da_.coords['date'].values)
                        .astype('datetime64[D]').astype(np.float64))
                bp = None
                if baseline and baseline in ds:
                    bp = np.asarray(ds[baseline].values, dtype=float)
                    while bp.ndim > 1:
                        bp = np.nanmean(bp, axis=-1)
                if t0 is None:
                    _b = bp if (bp is not None and bp.shape == dday.shape) \
                        else np.zeros_like(dday)
                    t0 = float(dday[int(np.argmin(np.abs(_b)))])
                t_d = (dday - t0) / 365.25

            # ele2phase = B_perp / median(R sin(incidence)), from
            # elevation_phase() = 4 pi / (lambda R sin(inc))
            e2p = None
            n_obs = da_.sizes['pair' if per_pair else 'date']
            if bp is not None and bp.shape == (n_obs,):
                _fac = Batch._elevation_phase_approximate(
                    transform if transform is not None else self)[key]
                e2p = bp / ((4.0 * np.pi / lam) / _fac)

            import dask.array as da
            vel = mds['velocity'].data
            hgt = mds['height'].data
            sea = mds['seasonal'].data
            planes = []
            for i in range(n_obs):
                if per_pair:
                    # the pair convention, exactly fit1d's columns
                    phi = vel * float(t_rep[i] - t_ref[i])
                    if e2p is not None:
                        phi = phi - da.nan_to_num(hgt) * float(e2p[i])
                    dcos = float(np.cos(2 * np.pi * t_rep[i])
                                 - np.cos(2 * np.pi * t_ref[i]))
                    dsin = float(np.sin(2 * np.pi * t_rep[i])
                                 - np.sin(2 * np.pi * t_ref[i]))
                    phi = phi + (sea.real * dcos + sea.imag * dsin)
                else:
                    # per-date, published as master*conj(date) so it is the
                    # same radians displacement_los() converts -- identical in
                    # form to the pair branch, differenced against the epoch
                    phi = vel * float(t_d[i])
                    if e2p is not None:
                        phi = phi - da.nan_to_num(hgt) * float(e2p[i])
                    car = np.exp(2j * np.pi * float(t_d[i]))
                    phi = phi + (sea.real * car.real + sea.imag * car.imag)
                planes.append(phi.astype(np.float32))
            pred = da.stack(planes, axis=0)
            if ref is not None:
                if per_pair:
                    raise TypeError(
                        "ref re-references a per-DATE series, and a pair is "
                        "already a difference in which the reference cancels. "
                        "Call it on the per-date stack instead.")
                _i = Batch._ref_index(ref, da_.coords['date'].values)
                pred = pred - pred[_i]

            dim = 'pair' if per_pair else 'date'
            # BARE axes: the model's scalar `date` coordinate travels with any
            # DataArray taken from it and then collides with a `date` dimension
            coords = {dim: np.asarray(da_.coords[dim].values),
                      'y': np.asarray(mds.coords['y'].values),
                      'x': np.asarray(mds.coords['x'].values)}
            pds = xr.Dataset({pol: xr.DataArray(pred, dims=(dim, 'y', 'x'),
                                                coords=coords)}, attrs=ds.attrs)
            if per_pair:
                for c in ('ref', 'rep'):
                    if c in da_.coords:
                        pds = pds.assign_coords(
                            {c: ('pair', np.asarray(da_.coords[c].values))})
            sref = (mds['spatial_ref'] if 'spatial_ref' in mds.coords
                    else (ds['spatial_ref'] if 'spatial_ref' in ds.coords else None))
            if sref is not None:
                pds = pds.assign_coords(
                    spatial_ref=sref.drop_vars(list(sref.coords), errors='ignore'))
            out[key] = pds
        return Batch(out)


    def elevation_phase(self) -> "Batch":
        """Radians of phase per metre of elevation per metre of perpendicular baseline.

        `4 pi / (lambda R sin(incidence))`, per pixel. Multiply by B_perp and a
        height to get phase; divide a phase by it and B_perp to get a height.
        One definition of the geometry, so the forward and inverse conversions
        cannot drift apart.

        SLANT RANGE AND THE SINE OF THE INCIDENCE ANGLE, matching GMTSAR.
        `sbas.c:186` computes `scale = 4 pi / wl / rng / sin(theta)` with theta
        documented as "incidence angle of the radar wave", and `phase2topo.c`
        returns `topo = res * rho * c * sint / ret` where `c = re + height` is
        the satellite distance from the Earth CENTRE -- by the law of sines
        `c*sint/ret` is sin(incidence), so that is `res * rho * sin(incidence)`,
        the same relation inverted, and a numerical check agrees. Every `cos()`
        in GMTSAR's geometry is
        baseline projection (`bperp.c:151`, `B*cos(theta-alpha)`) or a look-
        vector rotation, never elevation from phase.

        This exists because `elevation()` inlined `SC_height * cos(incidence)`:
        the satellite HEIGHT (702 227 m here) where the slant range
        (856 485 - 860 637 m) belongs, and the cosine where the sine belongs.
        The errors nearly cancel -- 0.818 * 1.306 = 1.068 -- so the result read
        as plausible while biasing every height by +6.8%.

        Call on a transform batch (`stack.transform()`), which carries `rng`,
        `near_range`, `rng_samp_rate` and what `incidence()` needs.
        """
        import numpy as np

        c_light = 299792458.0
        inc_batch = self.incidence()
        out: dict[str, xr.Dataset] = {}
        for key, tfm in self.items():
            wavelength = tfm['radar_wavelength']
            wavelength = float(np.asarray(wavelength.values).ravel()[0]) if hasattr(wavelength, 'values') else float(wavelength)
            near_range = float(tfm['near_range'].mean().item())
            rng_samp_rate = float(tfm['rng_samp_rate'].mean().item())
            slant_range = near_range + tfm['rng'] * (c_light / (2.0 * rng_samp_rate))
            incidence = inc_batch[key]['incidence']
            fac = (4.0 * np.pi / wavelength) / (slant_range * xr.ufuncs.sin(incidence))
            res = xr.Dataset({'elevation_phase': fac.astype('float32')})
            res.attrs = tfm.attrs
            if 'spatial_ref' in tfm.coords:
                res = res.assign_coords(spatial_ref=tfm.spatial_ref)
            out[key] = res
        return Batch(out)

    def _elevation_phase_approximate(self) -> dict:
        """`elevation_phase()` at the centre of the PROCESSED area.

        `{burst_id: value}`, and it reads NO radar grids. `azi`, `rng` and
        `ele` are the special rasters `transform()` carries; a caller turning
        an ARGUMENT LIMIT into phase -- a height bound in metres into a lattice
        step -- must not depend on them, and PAIR data does not have them at
        all. Results are a different matter: they come out in radians and
        `displacement_los()` converts them with the full pixelwise geometry.

        WHERE THE CENTRE COMES FROM. `geometry` is the burst's exact radar
        extent as a polygon, and its four corners have known radar coordinates
        by construction -- (0.5, 0.5) to (num_lines-0.5, num_rng_bins-0.5).
        Inverting that mapping turns any map coordinate into (azi, rng), so the
        centre of the geocoded grid gives the centre of what was actually
        processed. That matters because `geometry` describes the WHOLE burst
        while a bbox may have cropped the data to a small part of it, and the
        burst centre would then sit outside the data entirely.

        The inverse mapping need not be accurate in range BINS -- the slant
        range barely moves across many of them -- only good enough to land in
        the processed area rather than at the far side of the burst.
        """
        import numpy as np

        c_light = 299792458.0
        out: dict = {}
        for key, ds in self.items():
            g = lambda n: float(np.asarray(ds[n].values).ravel().mean())
            wavelength = g('radar_wavelength')
            near_range = g('near_range')
            rng_samp_rate = g('rng_samp_rate')
            earth_radius = g('earth_radius')
            num_lines = g('num_lines')
            num_rng_bins = g('num_rng_bins')
            sc_height = 0.5 * (g('SC_height_start') + g('SC_height_end'))

            rng_c = num_rng_bins / 2.0
            try:
                from shapely import wkt as _wkt
                from pyproj import Transformer as _Tr
                poly = _wkt.loads(str(np.asarray(ds['geometry'].values).ravel()[0]))
                lon, lat = (np.asarray(v) for v in poly.exterior.coords.xy)
                crs = ds.rio.crs if hasattr(ds, 'rio') else None
                px, py = _Tr.from_crs('EPSG:4326', crs, always_xy=True).transform(
                    lon[:4], lat[:4])
                a_c = np.array([0.5, 0.5, num_lines - 0.5, num_lines - 0.5])
                r_c = np.array([0.5, num_rng_bins - 0.5,
                                num_rng_bins - 0.5, 0.5])
                M, b = [], []
                for x_, y_, a_, r_ in zip(px, py, a_c, r_c):
                    M.append([x_, y_, 1, 0, 0, 0, -a_ * x_, -a_ * y_]); b.append(a_)
                    M.append([0, 0, 0, x_, y_, 1, -r_ * x_, -r_ * y_]); b.append(r_)
                h = np.linalg.solve(np.asarray(M, float), np.asarray(b, float))
                xc = 0.5 * (float(ds.x.min()) + float(ds.x.max()))
                yc = 0.5 * (float(ds.y.min()) + float(ds.y.max()))
                den = h[6] * xc + h[7] * yc + 1.0
                cand = (h[3] * xc + h[4] * yc + h[5]) / den
                if np.isfinite(cand) and 0.0 <= cand <= num_rng_bins:
                    rng_c = float(cand)
            except Exception:
                # no `geometry`, no CRS, or a degenerate polygon: the burst
                # centre still answers, just less well on a cropped stack
                pass

            slant_range = near_range + rng_c * (c_light / (2.0 * rng_samp_rate))
            ground_dist = earth_radius + (
                float(np.asarray(ds['ref_height'].values).ravel().mean())
                if 'ref_height' in ds else 0.0)
            sat_dist = earth_radius + sc_height
            cos_earth = np.clip(
                (ground_dist ** 2 + sat_dist ** 2 - slant_range ** 2)
                / (2.0 * ground_dist * sat_dist), -1.0, 1.0)
            sin_inc = np.clip(
                sat_dist * np.sin(np.arccos(cos_earth)) / slant_range, -1.0, 1.0)
            out[key] = (4.0 * np.pi / wavelength) / (slant_range * sin_inc)
        return out

    def incidence(self) -> "Batch":
        """Compute incidence angle from azi, rng, ele, and radar geometry parameters.

        Uses spherical Earth geometry with per-pixel satellite height interpolation
        and terrain elevation correction. Matches GMTSAR look vector results within ~0.07%.

        Required vars: azi, rng, ele, near_range, SC_height_start, SC_height_end,
                       earth_radius, rng_samp_rate, num_lines
        """
        import numpy as np
        import rioxarray  # for .rio accessor

        c = 299792458.0  # speed of light

        # Get CRS from input batch
        crs = self.crs

        out: dict[str, xr.Dataset] = {}
        for key, tfm in self.items():
            # Get scalar parameters (mean if per-date)
            near_range = float(tfm['near_range'].mean().item())
            SC_height_start = float(tfm['SC_height_start'].mean().item())
            SC_height_end = float(tfm['SC_height_end'].mean().item())
            earth_radius = float(tfm['earth_radius'].mean().item())
            rng_samp_rate = float(tfm['rng_samp_rate'].mean().item())
            num_lines = float(tfm['num_lines'].mean().item())

            # Get per-pixel coordinates
            azi = tfm['azi']
            rng = tfm['rng']
            ele = tfm['ele']

            # Compute slant range to the actual elevated ground point
            range_pixel_size = c / (2 * rng_samp_rate)
            slant_range = near_range + rng * range_pixel_size

            # Interpolate satellite height based on azimuth position
            SC_height = SC_height_start + (SC_height_end - SC_height_start) * azi / (num_lines - 1)

            # Ground at earth_radius + ele from Earth center
            ground_dist = earth_radius + ele

            # Satellite at earth_radius + SC_height from Earth center
            sat_dist = earth_radius + SC_height

            # Spherical Earth geometry: law of cosines + law of sines
            cos_earth = (ground_dist**2 + sat_dist**2 - slant_range**2) / (2 * ground_dist * sat_dist)
            cos_earth = xr.where(cos_earth > 1, 1, xr.where(cos_earth < -1, -1, cos_earth))
            earth_angle = xr.ufuncs.arccos(cos_earth)
            sin_inc = sat_dist * xr.ufuncs.sin(earth_angle) / slant_range
            sin_inc = xr.where(sin_inc > 1, 1, xr.where(sin_inc < -1, -1, sin_inc))
            incidence = xr.ufuncs.arcsin(sin_inc).astype('float32')

            result_ds = xr.Dataset({"incidence": incidence})
            result_ds.attrs = tfm.attrs
            # Preserve CRS
            if crs is not None:
                result_ds = result_ds.rio.write_crs(crs)
            out[key] = result_ds
        return Batch(out)

    def iexp(self, sign: int = -1, **kwargs):
        """
        Apply exp(sign * 1j * da) to convert phase to complex phasor.

        Parameters
        ----------
        sign : int, optional
            Sign of the exponent. Default is -1 for exp(-1j * phase).

        Returns
        -------
        BatchComplex
            Complex phasor representation.
        """
        import xarray as xr
        return BatchComplex(self.map_da(lambda da: xr.ufuncs.exp(sign * 1j * da), **kwargs))

    def displacement_los(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute line-of-sight displacement (meters) from unwrapped phase.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch providing mission constants (radar_wavelength),
            or a Stack (will call .transform() internally).

        Returns
        -------
        Batch
            LOS displacement grids (meters), lazily scaled by the mission wavelength.

        Examples
        --------
        >>> disp_los = unwrapped.displacement_los(stack)
        >>> disp_los = unwrapped.displacement_los(stack.transform())
        """
        import numpy as np
        import xarray as xr
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        if not transform:
            raise ValueError('transform must contain at least one burst with radar_wavelength')

        transform_first = next(iter(transform.values()))

        def _scalar_from_ds(ds, name: str):
            if name in ds:
                var = ds[name]
                if var.ndim == 0:
                    return var.item()
                elif var.ndim >= 1:
                    values = var.values.flatten()
                    unique = np.unique(values)
                    if len(unique) != 1:
                        raise ValueError(f'{name} has multiple distinct values: {unique}')
                    return unique[0]
            return ds.attrs.get(name)

        wavelength = _scalar_from_ds(transform_first, 'radar_wavelength')
        if wavelength is None:
            raise KeyError('Missing radar_wavelength in transform')

        # scale factor from phase in radians to displacement in meters
        # constant is negative to make LOS = -1 * range change
        scale = -float(wavelength) / (4 * np.pi)

        out: dict[str, xr.Dataset] = {}
        for key, phase_ds in self.items():
            disp_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # radar_wavelength, burst ids and the rest ride along and are
                # carried through untouched; multiplying a <U43 burst name by
                # a float is the UFuncTypeError this prevents.
                if not ('y' in data.dims and 'x' in data.dims):
                    disp_vars[var_name] = data
                    continue
                disp = (data * scale).astype('float32')
                disp_vars[var_name] = disp
            out[key] = xr.Dataset(disp_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def _displacement_component(self, transform: 'Batch | Stack', func, suffix: str = '') -> 'Batch':
        """Internal helper to scale LOS displacement by an incidence-based function (e.g., cos/sin)."""
        import xarray as xr
        import numpy as np
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        # Decimate transform to match phase resolution for efficiency
        transform = Batch({k: transform[k].reindex(y=self[k].y, x=self[k].x, method='nearest')
                           for k in self.keys() if k in transform})

        los_batch = self.displacement_los(transform)
        incidence_batch = transform.incidence()

        out: dict[str, xr.Dataset] = {}

        for key, los_ds in los_batch.items():
            if key not in incidence_batch:
                raise KeyError(f'Missing incidence for key: {key}')

            inc_da = incidence_batch[key]['incidence']
            comp_vars: dict[str, xr.DataArray] = {}

            for var_name, data in los_ds.data_vars.items():
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # radar_wavelength, burst ids and the rest ride along and are
                # carried through untouched; multiplying a <U43 burst name by
                # a float is the UFuncTypeError this prevents.
                if not ('y' in data.dims and 'x' in data.dims):
                    comp_vars[var_name] = data
                    continue
                # align incidence to data grid
                if 'y' in data.coords and 'x' in data.coords:
                    incidence = inc_da.interp(y=data.y, x=data.x, method='linear')
                else:
                    incidence = inc_da.reindex_like(data, method='nearest')

                comp = (data / func(incidence)).astype('float32')

                if len(los_ds.data_vars) == 1:
                    name = suffix
                elif var_name.endswith('_los'):
                    name = var_name[:-4] + f'_{suffix}'
                else:
                    name = f'{var_name}_{suffix}'

                comp_vars[name] = comp

            out[key] = xr.Dataset(comp_vars, coords=los_ds.coords, attrs=los_ds.attrs)

        return Batch(out)

    def displacement_vertical(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute vertical displacement (meters) from unwrapped phase and incidence.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch providing incidence angle and mission constants,
            or a Stack (will call .transform() internally).

        Returns
        -------
        Batch
            Vertical displacement grids (meters).

        Examples
        --------
        >>> disp_v = unwrapped.displacement_vertical(stack)
        >>> disp_v = unwrapped.displacement_vertical(stack.transform())
        """
        import xarray as xr
        return self._displacement_component(transform, func=xr.ufuncs.cos, suffix='vertical')

    def displacement_eastwest(self, transform: 'Batch | Stack') -> 'Batch':
        """Compute east-west displacement (meters) from unwrapped phase and incidence.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch providing incidence angle and mission constants,
            or a Stack (will call .transform() internally).

        Returns
        -------
        Batch
            East-west displacement grids (meters).

        Examples
        --------
        >>> disp_ew = unwrapped.displacement_eastwest(stack)
        >>> disp_ew = unwrapped.displacement_eastwest(stack.transform())
        """
        import xarray as xr
        return self._displacement_component(transform, func=xr.ufuncs.sin, suffix='eastwest')

    def elevation(self, transform: 'Batch | Stack', baseline: float | None = None) -> 'Batch':
        """Compute elevation (meters) from unwrapped phase grids.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch containing look vectors for incidence calculation,
            or a Stack (will call .transform() internally).
        baseline : float | None, optional
            Perpendicular baseline in meters. If None, uses burst-specific BPR
            from phase coordinates.

        Returns
        -------
        Batch
            Elevation grids as float32 datasets.

        Examples
        --------
        >>> elev = unwrapped.elevation(stack)
        >>> elev = unwrapped.elevation(stack.transform())
        """
        import xarray as xr
        import numpy as np
        from .Stack import Stack

        # If Stack passed, get transform from it
        if isinstance(transform, Stack):
            transform = transform.transform()

        ep_batch = transform.elevation_phase()
        out: dict[str, xr.Dataset] = {}

        for key, phase_ds in self.items():
            if key not in ep_batch:
                raise KeyError(f'Missing geometry for key: {key}')

            tfm = transform[key]

            def _scalar_from_ds(ds, name: str):
                if name in ds:
                    var = ds[name]
                    if var.ndim == 0:
                        return float(var.item())
                    return float(var.mean().item())
                return ds.attrs.get(name)

            wavelength = _scalar_from_ds(tfm, 'radar_wavelength')
            if wavelength is None:
                raise KeyError(f"Missing radar_wavelength in transform for burst {key}")

            # Get BPR - either scalar or per-pair DataArray for broadcasting
            if baseline is not None:
                bpr = float(baseline)
            elif 'BPR' in phase_ds.coords:
                bpr = phase_ds.coords['BPR']
            else:
                raise KeyError(f"Missing baseline (BPR) for burst {key}")

            # ONE geometry, from elevation_phase(): radians per metre of height
            # per metre of baseline. This block used to inline
            # `SC_height * cos(incidence)` -- the satellite height where the
            # SLANT RANGE belongs and the cosine where the SINE belongs. The two
            # errors nearly cancel (0.818 * 1.306 = 1.068), so the answer looked
            # reasonable while every height came out 6.8% high. GMTSAR's
            # phase2topo.c and sbas.c both use slant range and sin(incidence);
            # see elevation_phase() for the references and the numbers.
            fac_da = ep_batch[key]['elevation_phase']

            ref_height = _scalar_from_ds(tfm, 'ref_height') or 0.0

            elev_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # radar_wavelength, burst ids and the rest ride along and are
                # carried through untouched; multiplying a <U43 burst name by
                # a float is the UFuncTypeError this prevents.
                if not ('y' in data.dims and 'x' in data.dims):
                    elev_vars[var_name] = data
                    continue
                if 'y' in data.coords and 'x' in data.coords:
                    fac = fac_da.interp(y=data.y, x=data.x, method='linear')
                else:
                    fac = fac_da.reindex_like(data, method='nearest')

                # phi = fac * B_perp * dh  ->  dh = phi / (fac * B_perp)
                elev = ref_height - data / (fac * bpr)
                elev_vars[var_name] = elev.astype('float32')

            out[key] = xr.Dataset(elev_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def stl(self, freq: str = 'W', periods: int = 52, robust: bool = False) -> 'Batch':
        """
        Perform Seasonal-Trend decomposition using LOESS (STL).

        Decomposes time series into trend, seasonal, and residual components.
        The Batch must have a 'date' dimension.

        Parameters
        ----------
        freq : str, optional
            Frequency string for resampling (default 'W' for weekly).
            Examples: '1W' for 1 week, '2W' for 2 weeks, '10d' for 10 days.
        periods : int, optional
            Number of periods for seasonal decomposition (default 52 for weekly data = 1 year).
        robust : bool, optional
            Whether to use robust fitting (slower but handles outliers better). Default False.

        Returns
        -------
        Batch
            Batch containing 'trend', 'seasonal', and 'resid' variables for each polarization.

        Examples
        --------
        >>> model = (unwrapped - unwrapped.gaussian(wavelength=40000)).fit1d(weight=corr)
        >>> stl_result = displacement.stl(freq='W', periods=52)
        >>> stl_result.plot()  # Shows trend, seasonal, resid components

        See Also
        --------
        statsmodels.tsa.seasonal.STL : Seasonal-Trend decomposition using LOESS
        """
        from .Stack import Stack

        return Stack.stl(Stack(), self, freq=freq, periods=periods, robust=robust)

class BatchWrap(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores wrapped phase (real values).
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None, wrap: bool = True):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchComplex)):
            raise ValueError(f'ERROR: BatchWrap does not support Stack or BatchComplex objects.')
        # skip wrapping for intermediate objects like DatasetCoarsen
        if not wrap:
            dict.__init__(self, mapping or {})
        else:
            wrapped = {k: self.wrap(v) for k, v in (mapping or {}).items()}
            dict.__init__(self, wrapped)

    @staticmethod
    def wrap(data):
        """Wrap the (y, x) planes to [-pi, pi]; leave everything else alone.

        Wrapping the whole Dataset would fold the radar metadata too --
        near_range 800000 m comes back as 2.3 rad, silently -- so only variables
        that actually carry phase are wrapped.
        """
        if isinstance(data, xr.Dataset):
            out = data.copy()
            for v in data.data_vars:
                da_ = data[v]
                if da_.ndim >= 2 and tuple(da_.dims[-2:]) == ('y', 'x'):
                    out[v] = np.mod(da_ + np.pi, 2 * np.pi) - np.pi
            out.attrs = data.attrs
            return out
        return np.mod(data + np.pi, 2 * np.pi) - np.pi

    def trend1d(self, *args, **kwargs):
        raise TypeError(
            "trend1d() does not support wrapped phase (BatchWrap). "
            "Use BatchComplex for complex phase fitting, or unwrap first for real polynomial fitting."
        )

    def trend2d(self, *args, **kwargs):
        raise TypeError(
            "trend2d() does not support wrapped phase (BatchWrap). "
            "Use BatchComplex for complex phase fitting, or unwrap first for real polynomial fitting."
        )

    def trend1d_pairs(self, *args, **kwargs):
        raise TypeError(
            "trend1d_pairs() requires BatchComplex (complex wrapped phase). "
            "Place fit3d() before unwrapping in the pipeline."
        )

    def __add__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: (self[k] + other[k] if k in other else self[k]) for k in keys})

    def __sub__(self, other: Batch):
        import xarray as xr
        import operator as _operator
        # SUBTRACT THE GRIDS, CARRY THE REST. A whole-Dataset `ds - val` reaches
        # the burst ids and the radar geometry too: numpy refuses to subtract
        # <U43 strings, so align() raised, and radar_wavelength minus itself is
        # 0, so meter2rad became infinite where it did not. _binary_vars is the
        # one place that rule lives.
        _sub_grids = lambda d, v: BatchCore._binary_vars(d, v, _operator.sub)
        keys = self.keys()
        result = {}
        for k in keys:
            if k not in other:
                result[k] = self[k]
            else:
                val = other[k]
                ds = self[k]
                # Handle per-pair coefficients from burst_polyfit
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    # Get a spatial variable (with y, x dims) to check for pair dimension
                    spatial_vars = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
                    sample_var = spatial_vars[0] if spatial_vars else list(ds.data_vars)[0]
                    sample_da = ds[sample_var]
                    has_pair_dim = 'pair' in sample_da.dims
                    n_pairs = sample_da.sizes.get('pair', 1)
                    first_elem = val[0]

                    if isinstance(first_elem, (list, tuple)):
                        # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                        result[k] = _sub_grids(ds, self[[k]].polyval({k: val})[k])
                    elif has_pair_dim and len(val) == n_pairs:
                        # Multi-pair degree=0: [off0, off1, ...]
                        # Use da.stack for dask 0-d arrays to avoid triggering .compute()
                        if any(hasattr(v, 'dask') for v in val):
                            import dask.array as _da
                            offsets = xr.DataArray(_da.stack(val), dims=['pair'])
                        else:
                            offsets = xr.DataArray(val, dims=['pair'])
                        result[k] = _sub_grids(ds, offsets)
                    elif len(val) == 1:
                        # Single value wrapped in list: [offset]
                        result[k] = _sub_grids(ds, val[0])
                    else:
                        # Single pair degree=1: [ramp, offset]
                        result[k] = _sub_grids(ds, self[[k]].polyval({k: val})[k])
                elif isinstance(val, (int, float)) \
                        or (hasattr(val, 'ndim') and val.ndim == 0):
                    # Scalar subtraction (concrete or dask 0-d array)
                    result[k] = _sub_grids(ds, val)
                else:
                    result[k] = _sub_grids(ds, val)
        return type(self)(result)

    def __mul__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self[k] * other[k] if k in other else self[k] for k in keys})

    def __rmul__(self, other):
        # scalar * batch  → map scalar * each dataset
        return type(self)({k: other * v for k, v in self.items()})

    def __truediv__(self, other: Batch):
        keys = self.keys()
        return type(self)({k: self[k] / other[k] if k in other else self[k] for k in keys})

    def sin(self, **kwargs) -> Batch:
        """
        Return a Batch of the sin(theta) DataArrays, preserving attrs if requested.
        """
        return Batch(self.map_da(lambda da, **kw: xr.ufuncs.sin(da), **kwargs))

    def cos(self, **kwargs) -> Batch:
        """
        Return a Batch of the cos(theta) DataArrays, preserving attrs if requested.
        """
        return Batch(self.map_da(lambda da, **kw: xr.ufuncs.cos(da), **kwargs))

    def iexp(self, sign: int = -1, **kwargs):
        """
        Apply exp(sign * 1j * da) like np.exp(-1j * intfs)

        - If sign = -1 (the default), this is exp(-1j * da).
        - If sign = +1, this is exp(+1j * da).
        """
        from .Batch import BatchComplex
        return BatchComplex(self.map_da(lambda da, **kw: xr.ufuncs.exp(sign * 1j * da), **kwargs))

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Converts wrapped phase to complex numbers before aggregation and back to wrapped phase after.
        """
        #print ('wrap _agg')
        import inspect
        import xarray as xr
        import pandas as pd
        out = {}
        for key, obj in self.items():
            # get the aggregation function
            fn = getattr(obj, name)
            sig = inspect.signature(fn)
            
            # perform aggregation in complex domain
            if 'dim' in sig.parameters:
                # intfs.mean('pair').isel(0)
                #agg_result = fn(dim=dim, **kwargs)
                complex_obj = xr.ufuncs.exp(1j * obj.astype('float32'))
                #fn_complex = getattr(complex_obj, name)
                #agg_result = fn_complex(dim=dim, **kwargs)
                if name in ('var', 'std'):
                    # |E[e^(iθ)]|
                    R = xr.ufuncs.abs(complex_obj.mean(dim=dim, **kwargs))
                    if name == 'var':
                        # 1 - |E[e^(iθ)]|
                        agg_result = (1 - R)
                    else:  # std
                        # √(-2 ln|E[e^(iθ)]|)
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    fn_complex = getattr(complex_obj, name)
                    agg_result = fn_complex(dim=dim, **kwargs)
                    # convert back to wrapped phase
                    agg_result = xr.ufuncs.angle(agg_result)
            else:
                # intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()
                # already in complex domain, see coarsen()
                if name in ('var', 'std'):
                    R = xr.ufuncs.abs(obj.mean(**kwargs))
                    if name == 'var':
                        agg_result = (1 - R)
                    else:  # std
                        agg_result = xr.ufuncs.sqrt(-2 * xr.ufuncs.log(R))
                else:
                    agg_result = fn(**kwargs)
                    agg_result = xr.ufuncs.angle(agg_result)
            
            # Convert back to wrapped phase
            out[key] = agg_result.astype('float32')
            
        #print ('wrap _agg self.chunks', self.chunks)
        #return type(self)(out).chunk(self.chunks)
        #print ('wrap _agg self.chunks', self.chunks)
        # filter out collapsed dimensions
        sample = next(iter(out.values()), None)
        dims = (sample.dims or []) if hasattr(sample, 'dims') else []
        chunks = {d: size for d, size in self.chunks.items() if d in dims}
        #print ('wrap chunks', chunks)
        result = type(self)(out)
        if chunks:
            return result.chunk(chunks)
        return result

    def coarsen(self, window: dict[str, int], **kwargs) -> Batch:
        """
        Coarsen each DataSet in the batch by integer factors and align the 
        blocks so that they fall on "nice" grid boundaries.

        Parameters
        ----------
        window : dict[str,int]
            e.g. {'y': 2, 'x': 8}
        **kwargs
            extra args forwarded into the reduction, e.g. skipna=True.

        Returns
        -------
        Batch
            A new Batch where each Dataset has been sliced for alignment,
            coarsened by `window`, then reduced by `.mean()` (or whichever
            `func` you chose).
        """
        #print ('wrap coarsen')
        chunks = self.chunks
        #print ('self.chunks', chunks)
        out = {}
        # produce unified grid and chunks for all datasets in the batch
        for key, ds in self.items():
            # convert to complex numbers for proper circular statistics
            ds2 = xr.ufuncs.exp(1j * ds.astype('float32'))
            # align each dimension
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds2, dim, factor)
                #print ('start', start)
                if start is not None:
                    # rechunk to the original chunk sizes
                    ds2 = ds2.isel({dim: slice(start, None)}).chunk(chunks)
                    # or allow a bit different chunks for coarsening
                    #ds2 = ds2.isel({dim: slice(start, None)})
            # coarsen
            out[key] = ds2.coarsen(window, **kwargs)

        # wrap=False since these are DatasetCoarsen objects, not actual data
        return type(self)(out, wrap=False)

    def plot(
        self,
        cmap = 'gist_rainbow_r',
        alpha = 0.7,
        caption='Phase, [rad]',
        vmin=-np.pi,
        vmax=np.pi,
        *args,
        **kwargs
    ):
        kwargs["cmap"] = cmap
        kwargs["alpha"] = alpha
        kwargs["caption"] = caption
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
        return super().plot(*args, **kwargs)

    # def gaussian(self, *args, **kwargs):
    #     """
    #     Phase-aware Gaussian smoothing for wrapped phase data.
    #     """
    #     return self.iexp().gaussian(*args, **kwargs).angle()

    # def gaussian(self, *args, **kwargs):
    #     """
    #     Phase-aware Gaussian smoothing by filtering sin(θ) and cos(θ) separately,
    #     then recombining via atan2.  No complex dtype ever created.
    #     """
    #     from .Batch import Batch
    #     import xarray as xr

    #     keep_attrs = kwargs.pop('keep_attrs', None)
    #     # build two Batches of the real sin and cos components and filter them
    #     sin = self.sin(keep_attrs=keep_attrs).gaussian(*args, **kwargs)
    #     cos = self.cos(keep_attrs=keep_attrs).gaussian(*args, **kwargs)

    #     # compute wrapped phase using np.arctan2
    #     out = {k: xr.Dataset({
    #         var: xr.ufuncs.arctan2(sin[k][var], cos[k][var]).astype('float32')
    #         for var in sin[k].data_vars
    #     }) for k in self.keys()}

    #     return BatchWrap(out)

    def gaussian(self, *args, **kwargs):
        """
        Phase-aware Gaussian smoothing by filtering sin(θ) and cos(θ) separately,
        then recombining via arctan2.
        """
        from .Batch import Batch
        import xarray as xr

        keep_attrs = kwargs.pop('keep_attrs', False)
        data_vars = next(iter(self.values())).data_vars

        # build two Batches of the real sin and cos components and filter them
        sin = self.sin(keep_attrs=keep_attrs).gaussian(*args, **kwargs)
        cos = self.cos(keep_attrs=keep_attrs).gaussian(*args, **kwargs)

        # compute wrapped phase using arctan2
        out: dict[str, xr.Dataset] = {}
        for k in self.keys():
            phase_vars = {}
            for var in data_vars:
                src = self[k][var]
                # A DATASET OPERATION APPLIES TO THE GRIDS AND NOTHING ELSE.
                # arctan2 on a <U43 burst id is a TypeError, and the geometry
                # riding beside the phase is not something to smooth.
                if not ('y' in src.dims and 'x' in src.dims):
                    phase_vars[var] = src
                    continue
                phase = xr.ufuncs.arctan2(sin[k][var], cos[k][var]).astype('float32')
                if keep_attrs:
                    phase.attrs = src.attrs.copy()
                phase_vars[var] = phase
            ds = xr.Dataset(phase_vars)
            if keep_attrs:
                ds.attrs = self[k].attrs.copy()
            out[k] = ds

        return BatchWrap(out)

    def unwrap2d(self, weight: 'BatchUnit | None' = None, conncomp: bool = False,
                 conncomp_size: int = 1000, conncomp_gap: int | None = None,
                 conncomp_linksize: int = 5, conncomp_linkcount: int = 30,
                 device: str = 'auto', debug: bool = False, **kwargs) -> 'Batch':
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L1 norm).

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        conncomp : bool
            If False (default), link disconnected components using ILP.
            If True, keep components separate and return conncomp labels.
        conncomp_size : int
            Minimum pixels for a connected component. Default 1000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon.

        Returns
        -------
        Batch or tuple
            If conncomp=False: Batch of unwrapped phase.
            If conncomp=True: tuple of (Batch unwrapped, BatchUnit conncomp).

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = phase.unwrap2d()  # Without weights
        >>> unwrapped = phase.unwrap2d(weight=corr)  # With weights
        """
        from .Stack import Stack

        return Stack.unwrap2d(Stack(), self, weight=weight,
                                       conncomp=conncomp, conncomp_size=conncomp_size,
                                       conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                                       conncomp_linkcount=conncomp_linkcount, device=device,
                                       debug=debug, **kwargs)

    def unwrap2d_chunk(self, weight: 'BatchUnit | None' = None, overlap=None,
                       device: str = 'auto', debug: bool = False, **kwargs) -> 'Batch':
        """
        Unwrap phase per spatial chunk with overlap using IRLS algorithm.

        Unlike unwrap2d() which requires a single spatial chunk (global unwrapping),
        this method unwraps each spatial chunk independently with overlap margins.
        Suitable for large rasters where global unwrapping would exceed memory.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        overlap : float, int, or tuple, optional
            Overlap size. Float = fraction of chunk size (0.25 = 25%).
            Int = pixels. Tuple (y, x) for different overlap per axis. Default 0.25.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon,
            conncomp_size.

        Returns
        -------
        Batch
            Batch of unwrapped phase.
        """
        from .Stack import Stack

        return Stack.unwrap2d_chunk(Stack(), self, weight=weight,
                                              overlap=overlap, device=device,
                                              debug=debug, **kwargs)

    def unwrap2d_irls(self, weight: 'BatchUnit | None' = None, device: str = 'auto',
                      max_iter: int = 50, tol: float = 1e-2, cg_max_iter: int = 10,
                      cg_tol: float = 1e-3, epsilon: float = 1e-2,
                      conncomp_size: int = 30, semaphore: int = 8, debug: bool = False) -> 'Batches':
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L1 norm).

        This is the core unwrapping algorithm. Disconnected components are
        unwrapped independently and aligned using per-component circular mean.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional weight for the unwrapping (typically correlation).
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        max_iter : int
            Maximum IRLS iterations. Default 50.
        tol : float
            Convergence tolerance. Default 1e-2.
        cg_max_iter : int
            Maximum conjugate gradient iterations. Default 10.
        cg_tol : float
            Conjugate gradient tolerance. Default 1e-3.
        epsilon : float
            Smoothing parameter for L1 approximation. Default 1e-2.
        conncomp_size : int
            Minimum connected component size in pixels. Components smaller than this
            are marked invalid (label 0). Default 30.
        semaphore : int
            Maximum concurrent CPU IRLS tasks per process. Default 8.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Tuple-like container with (Batch, BatchUnit):
            - unwrapped: Batch of unwrapped phase (float32)
            - conncomp: BatchUnit of component labels (uint16, 0=invalid, 1=largest, ...)

        Notes
        -----
        Uses a novel DCT+IRLS algorithm that combines DCT efficiency with IRLS
        robustness. See `utils_unwrap2d.irls_unwrap_2d` for algorithm details
        and references.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped, conncomp = phase.unwrap2d_irls(weight=corr)
        """
        from .Stack import Stack

        return Stack.unwrap2d_irls(Stack(), self, weight=weight,
                                            device=device, max_iter=max_iter, tol=tol,
                                            cg_max_iter=cg_max_iter, cg_tol=cg_tol,
                                            epsilon=epsilon, conncomp_size=conncomp_size,
                                            semaphore=semaphore, debug=debug)

    def unwrap2d_link(self, conncomp_size: int = 10_000, conncomp_gap: int | None = None,
                      conncomp_linksize: int = 5, conncomp_linkcount: int = 30,
                      debug: bool = False) -> 'Batch':
        """
        Link disconnected components in already unwrapped phase.

        This function applies component linking to already unwrapped phase data
        by finding optimal 2π offsets between disconnected components.
        Use this to correct phase jumps between components after unwrapping.

        Parameters
        ----------
        conncomp_size : int
            Minimum pixels for a connected component. Default 10,000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batch
            Batch of unwrapped phase with linked components.

        Examples
        --------
        >>> # First unwrap without linking
        >>> unwrapped = phase.unwrap2d_irls(weight=corr)
        >>>
        >>> # Then link components separately
        >>> linked = unwrapped.unwrap2d_link(conncomp_size=10_000, debug=True)
        """
        from .Stack import Stack

        return Stack.unwrap2d_link(Stack(), self,
                                            conncomp_size=conncomp_size,
                                            conncomp_gap=conncomp_gap,
                                            conncomp_linksize=conncomp_linksize,
                                            conncomp_linkcount=conncomp_linkcount,
                                            debug=debug)



class BatchUnit(BatchCore):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores correlation in the range [0,1].
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        if isinstance(mapping, (Stack, BatchWrap, BatchComplex)):
            raise ValueError(f'ERROR: BatchUnit does not support Stack, BatchWrap or BatchComplex objects.')
        dict.__init__(self, mapping or {})

    def plot(
        self,
        cmap = 'auto',
        caption=None,
        alpha=1,
        vmin=0,
        vmax=1,
        *args,
        **kwargs
    ):
        import matplotlib.colors as mcolors
        if isinstance(cmap, str) and cmap == 'auto':
            cmap = mcolors.LinearSegmentedColormap.from_list(
                name='custom_gray', 
                colors=['black', 'whitesmoke']
            )
        kwargs["cmap"] = cmap
        kwargs["caption"] = caption
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
        kwargs["alpha"] = alpha
        return super().plot(*args, **kwargs)



class BatchComplex(BatchCore):
    def fit1d(self, weight=None, baseline: str = 'BPR',
              max_dh: float = 200.0, max_dv: float = 100.0,
              step_dh: float = 4.0, step_dv: float = 2.0,
              max_seasonal: 'float | None' = None,
              budget: 'str | None' = None) -> 'Batch':
        """
        Full per-pixel model on the per-date complex stack -- NO network.

        The 1d twin of fit3d(): the SAME `_3d_arc_fit` kernel and the SAME
        model, fitted on each pixel's own time series instead of on arcs
        between neighbours. 1d is the time axis alone; 3d adds the two spatial
        ones. Returns the MODEL ONLY, named and scaled exactly as fit3d()
        names and scales it, so predict(model) is the single inverse for both.

        Velocity is the ROTATION RATE of the per-date phase vectors, so a pixel
        time series is exactly the object `_3d_arc_fit` already solves: a
        zero-centred lattice over (height error, rate) followed by a
        majorise-minimise refinement, with the constant scatterer phase
        profiled out by rotation and never estimated. Nothing is wrapped or
        unwrapped, and no reference date is needed.

        WHY THIS REPLACED THE MOVING-WINDOW ESTIMATOR. The previous version
        reported the constant term of a {1, cos, sin} fit to per-window rates,
        each estimated over a short window of few samples searching +-pi/dt.
        That path holds up only at high coherence and degrades where real
        pixels live: its neighbourhood disagreement came out FLAT with radius,
        which is what a noise field looks like, while this fit's grows with the
        box, like a real field. It was also far worse conditioned and more
        expensive.

        WHAT IT GIVES UP. An annual term of amplitude A rad leaves the model
        misspecified, and the coherence of the true rate is |J0(A)| while a
        sideband one cycle/yr away gets |J1(A)|; they cross at A = 1.435 rad,
        above which the sideband is genuinely the higher maximum and NO
        coherence-maximising estimator returns the truth. Whether a stack
        reaches that amplitude is a property of the stack, not of the fit;
        where it does, the rate is not identifiable from one pixel's phase
        alone, and the annual term belongs in the model.

        Parameters
        ----------
        weight : None
            Not supported: the fit normalises every sample to a unit phasor, so
            a magnitude weight cannot reach the objective. Passing one raises,
            rather than being silently ignored.
        baseline : str
            Variable holding the perpendicular baseline per date. With it the
            per-pixel DEM error is solved jointly with the rate, which matters:
            they are NOT separable one at a time, because the perpendicular
            baseline is not a smooth function of time. Without it (absent
            variable, or None) the height term is not estimated at all and the
            rate carries whatever the DEM error contributes.
        max_dh, max_dv : float
            Largest height error (m) and rate (mm/yr) to admit. A pixel solving
            outside them returns NaN rather than a plausible wrong number. The
            search runs wider than they say, so max_dv=100 detects 99 mm/yr on
            its merits and never against a boundary.
        step_dh, step_dv : float
            Lattice steps. They choose which basin is found, not the accuracy --
            the refinement is continuous and absorbs the quantisation.
        max_seasonal : float
            Largest annual amplitude to admit, in mm of LOS (HALF amplitude, so
            60 means a 120 mm peak-to-peak swing). 0 (default) leaves the annual
            term out of the model entirely.

            It is not a refinement: an annual term of amplitude A radians leaves
            coherence |J0(A)| at the true rate and |J1(A)| one cycle/yr away,
            and they cross at A = 1.435 rad. Above that the sideband IS the
            higher maximum, so a {height, rate} fit returns the sideband rather
            than the truth; with the term in the model the rate returns to its
            no-seasonal accuracy.

            It costs search time, and a little accuracy when there is no annual
            signal at all, so it is cheap to leave on. Large amplitudes are
            only partly recovered, but they fail LOUDLY -- NaN rather than
            silent wrong rates. Where a stack carries no seasonal signal the
            default 0 is right; zones with a real one are what this is for.
            ON ARCS, KEEP IT SMALL. A seasonal signal is long-wavelength, so
            an arc -- two pixels tens of metres apart -- sees only the small
            residue that does not cancel in the difference. A large
            max_seasonal there is wrong twice over, since it searches thousands
            of lattice points for an amplitude that cannot be present.

            Small, it earns its keep: marginal arcs are rescued and nodes
            isolated at any threshold join the network. Set too small, the arcs
            are rescued but fitted poorly, so the amplitude does need room to
            move.

            Judge any gain against a MATCHED-gamma null, not a raw one: two free
            parameters always raise gamma, and pure-noise arcs sit low enough
            that there is far more room to climb there than at a real arc, so an
            unmatched comparison understates the real gain.

            Whether the atmosphere is itself seasonal is a property of the
            stack and has to be checked there, against a permuted-date null
            rather than by eye. On a stack with genuinely seasonal delay the
            annual term would absorb it, and per pixel the two are not
            separable.

            What it does fix, where a real seasonal signal exists, is the
            contamination of dh and dv by leaving it out: an unmodelled annual
            term biases the height and can push the rate onto a whole sideband,
            while modelling it returns both to their clean values.
        budget : str or None
            Memory budget for the lattice product, e.g. '512MB'.

        Returns
        -------
        Batch
            ONE dataset of model parameters, named by quantity, identical in
            name, unit and convention to fit3d()'s:

              `velocity`   rad/yr
              `height`     rad per unit ele2phase
              `seasonal`   complex rad, the fitted annual
              `coherence`  gamma, the resultant length the fit maximised
              `rmse`       radians, circular deviation about that same model

            NO `conncomp`. fit3d() carries one because its network solves in
            connected components; every pixel here is solved alone, so there is
            no component to report and none is invented. predict() never reads
            it -- it is pixelwise.

            HEIGHT AND SEASONAL COST NOTHING. This is not a richer fit, it is
            the same fit reporting what it already had: the kernel must solve
            height jointly with rate (they do not separate) and the annual sits
            in the same objective, and the previous version bound two of the
            four returned values to `_dh` and `_sa` and dropped them. Same
            lattice, same 16 refinements, same runtime, four parameters out
            instead of one.

            THE RMSE IS EXACT. gamma is the resultant length of the residual
            about the model that was actually REPORTED -- it is the objective
            the fit maximised -- so

                sigma = sqrt(-2 ln gamma)

            is self-consistent by construction, equals the RMS for small
            residuals, and has no ceiling as the phase decorrelates. It is
            inflated by n/(n-p) for the parameters spent, and p now counts the
            annual's two whenever max_seasonal is non-zero -- it did not while
            the annual was fitted and discarded, which under-reported sigma on
            every pixel.

        Examples
        --------
        >>> model = stack.fit1d()
        >>> noise = stack * stack.predict(model, baseline='BPR').iexp()
        """
        import dask.array as da
        import numpy as np
        from .BatchCore import _parse_budget
        import xarray as xr
        from .Batch import Batch

        BatchCore._require_lazy(self, 'fit1d')
        if weight is not None:
            raise TypeError(
                'fit1d() does not accept a weight: every sample is normalised '
                'to a unit phasor before the fit, so a magnitude weight cannot '
                'affect the result. Mask the input instead.')

        from .utils_dask import get_dask_chunk_size_mb
        budget_mb = (_parse_budget(budget) if budget is not None
                     else get_dask_chunk_size_mb())

        # DELEGATE BY STACK TYPE. The same call fits the same model whether the
        # samples are dates or pairs; only the design columns differ, so the
        # caller writes fit1d() either way and never selects a variant by hand.
        # THE ANNUAL'S DEFAULT FOLLOWS THE STACK, because the model does. On
        # dates the kernel's cos(2 pi t) IS the basis, so 5 mm is the useful
        # default. On pairs the annual is a DIFFERENCE of two epochs, which is
        # the kernel's basis at the mean epoch rotated 90 degrees and scaled by
        # 2 sin(pi dt) -- a PER-PAIR scale no single t can carry. So pairs
        # default to no annual and raise only if one is actually asked for.
        _pairs = any('pair' in ds[v].dims
                     for ds in self.values() for v in ds.data_vars)
        if max_seasonal is None:
            max_seasonal = 5.0
        if _pairs:
            # the split is kept so a pair-domain fit has a home when one works
            raise NotImplementedError(
                'fit1d() does not support complex PAIRS. Use the per-DATE stack, '
                'or unwrap and call Batch.fit1d() on the unwrapped pairs.')

        model_result = {}
        for burst_id, ds in self.items():

            pols = [v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c'
                    and 'date' in ds[v].dims
                    and 'y' in ds[v].dims and 'x' in ds[v].dims]
            if not pols:
                raise TypeError(
                    f'fit1d() found no complex (date, y, x) variables in '
                    f'burst {burst_id}')
            # ONE polarisation, exactly as fit3d(): the model variables are
            # named by quantity alone so predict() can look them up directly,
            # and two polarisations would collide on those names.
            if len(pols) > 1:
                raise ValueError(
                    f"fit1d() fits ONE polarisation; burst '{burst_id}' carries "
                    f"{len(pols)}: {pols}. The model variables are named by "
                    "quantity alone (velocity, height, ...), so two "
                    "polarisations would collide. Select one first, e.g. "
                    "batch[['VV']].")
            pol = pols[0]

            # ele2phase = B_perp / (R sin theta), one value per burst: it
            # varies about a percent across it, which keeps the fit a matmul
            dates = np.asarray(ds.coords['date'].values)
            dday = dates.astype('datetime64[D]').astype(np.float64)
            bp = None
            ele2phase = None
            meter2rad = None
            if 'radar_wavelength' in ds:
                lam_ = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
                meter2rad = 4.0 * np.pi / lam_
                if baseline and baseline in ds:
                    bp = np.asarray(ds[baseline].values, dtype=float)
                    if bp.ndim > 1:
                        bp = np.nanmean(bp.reshape(len(dates), -1), axis=1)
                    # elevation_phase() = 4 pi / (lambda R sin(inc))
                    _fac = Batch._elevation_phase_approximate(self)[burst_id]
                    ele2phase = bp / (meter2rad / _fac)
            if meter2rad is None:
                raise TypeError(
                    f'fit1d() needs radar_wavelength in burst {burst_id} to '
                    f'turn a rotation rate into a velocity')

            # t = 0 AT THE MASTER, where B_perp is smallest -- the same origin
            # _3d_fit_ps_array and predict() use. Rate and height do not care
            # (a shift in t adds a constant and the constant is profiled out),
            # but the annual does: car = exp(2j*pi*t) rotates by
            # exp(2j*pi*delta), so a model fitted on one origin and removed on
            # another leaves a residual annual of 2|sin(pi*delta)|*|seasonal|.
            # This used to run from dates[0], which was invisible only because
            # the seasonal was discarded before anyone could subtract it.
            _b = bp if (bp is not None and bp.shape == dday.shape) \
                else np.zeros_like(dday)
            _master = int(np.argmin(np.abs(_b)))
            tyr = (dday - dday[_master]) / 365.25

            data_da = ds[pol]
            if data_da.dims[0] != 'date':
                data_da = data_da.transpose('date', ...)
            data_dask = data_da.data

            def _fit_block(block, _h=ele2phase, _t=tyr, _m=meter2rad,
                           _mh=float(max_dh), _mv=float(max_dv),
                           _sh=float(step_dh), _sv=float(step_dv),
                           _se=float(max_seasonal), _bm=int(budget_mb)):
                from .utils_arcs import _3d_arc_fit
                S = np.asarray(block)
                nd = S.shape[0]
                shape = S.shape[1:]
                Z = np.ascontiguousarray(S.reshape(nd, -1))
                if Z.shape[1] == 0:
                    e = np.empty(0, np.complex64).reshape(shape)
                    return np.stack([e, e, e, e, e], axis=0)
                # FOUR values, all of them kept. _3d_arc_fit already returns
                # rad/yr and rad per unit ele2phase: the library works in phase
                # and displacement_los() is the one place a length is made.
                gam, hgt, vel, sea = _3d_arc_fit(Z, _h, _t, _m,
                                                 _mh, _mv, _sh, _sv, _bm, _se)
                # ONE CONVENTION ACROSS EVERY FIT: displacement_los() must turn
                # this model into a negative rate where the ground subsides,
                # whichever fit produced it. _3d_arc_fit solves the per-DATE
                # phase, and a pair runs opposite to it -- an SLC phase is
                # -(4pi/lambda)r, so ref*conj(rep) carries +m2r*dr while a date
                # series carries -m2r*dr. displacement_los()'s -lambda/4pi is
                # derived for the pair, so the pair sense is the one the library
                # converts. Returning the date sense reports subsidence as uplift.
                #
                # HEIGHT IS NOT NEGATED. Its per-date term +hgt*e2p_d
                # differences to -hgt*e2p_pair, which already matches the pair
                # convention, and it is why a global sign flip on the
                # prediction does not work.
                vel = -vel
                sea = -sea
                # circular deviation about the REPORTED model, inflated for the
                # parameters the fit spent: phi0 and rate always, height when a
                # baseline was given, and the annual's real and imaginary parts
                # whenever it is in the model.
                nok = np.maximum((np.abs(Z) > 0).sum(axis=0), 1)
                npar = 2 + (0 if _h is None else 1) + (2 if _se > 0 else 0)
                infl = nok / np.maximum(nok - npar, 1)
                Rres = np.clip(gam.astype(np.float64), 1e-9, 1.0)
                rms = np.sqrt(np.maximum(-2.0 * np.log(Rres), 0.0) * infl)
                rms = np.where(np.isfinite(gam), rms, np.nan)
                # complex64 carries the real planes without rounding, so one
                # dtype ships all five and the seasonal needs no second pass
                return np.stack([vel.reshape(shape).astype(np.complex64),
                                 hgt.reshape(shape).astype(np.complex64),
                                 sea.reshape(shape).astype(np.complex64),
                                 gam.reshape(shape).astype(np.complex64),
                                 rms.reshape(shape).astype(np.complex64)],
                                axis=0)

            stacked = da.blockwise(
                _fit_block, 'nyx', data_dask, 'dyx',
                new_axes={'n': 5}, concatenate=True, dtype=np.complex64,
                meta=np.empty((0, 0, 0), dtype=np.complex64),
                name='fit1d_arcfit')

            coords = {kk: vv for kk, vv in data_da.coords.items()
                      if kk in ('y', 'x', 'spatial_ref')}
            mvars = {}
            for nm_, arr_ in (('velocity', stacked[0].real.astype(np.float32)),
                              ('height', stacked[1].real.astype(np.float32)),
                              ('seasonal', stacked[2].astype(np.complex64)),
                              ('coherence', stacked[3].real.astype(np.float32)),
                              ('rmse', stacked[4].real.astype(np.float32))):
                mvars[nm_] = xr.DataArray(arr_, dims=('y', 'x'), coords=coords)
            mds = xr.Dataset(mvars, attrs=ds.attrs)
            # the epoch the model is referenced to, named as every other date
            # in this library is named
            mds = mds.assign_coords(date=np.datetime64(int(dday[_master]), 'D'))
            if 'spatial_ref' in ds.coords:
                mds = mds.assign_coords(spatial_ref=ds.spatial_ref)
            model_result[burst_id] = mds
        return Batch(model_result)

    def predict(self, model, baseline: 'str | None' = 'BPR',
                transform=None, ref=None) -> 'Batch':
        """
        Predicted per-date phase from a fit3d() or fit1d() model.

        Reconstructs, for every date in this stack,

            phi_d = velocity * t_d + height * ele2phase_d + Re(car_d * conj(seasonal))

        with `car_d = exp(2j*pi*t_d)`, `t_d` in years from the first date, and
        `ele2phase_d = BPR_d / median(R sin(incidence))` -- the same geometry
        fit3d() fitted against, so the two are exact inverses.

        The scatterer's constant phase is NOT part of the model: fit3d() profiles
        it out by rotation rather than gauging it to an epoch, so the prediction
        is correct up to one constant per pixel. That is what you want for
        removal -- `stack * predict(model).iexp(sign=1)` cancels the modelled
        part and leaves the constant, which no interferometric measurement
        determines. SIGN=1, not the default: the prediction is published as
        master*conj(date), the same radians displacement_los() converts, and
        rotating it out of the stack therefore runs the other way.

        Parameters
        ----------
        model : Batch
            Output of fit3d() or fit1d(): variables `velocity`, `height`,
            `seasonal`
            (unprefixed, one polarisation).
        baseline : str or None
            Per-date perpendicular baseline variable. DEFAULT None, i.e. the
            topographic term is NOT projected onto dates, so the prediction is
            rate + seasonal only -- the object that matches phase which has
            already had topography removed. Pass 'BPR' to include it and predict
            the raw phase instead. Verified: predict('BPR') / predict(None) is
            exactly exp(1j*ele2phase*height) to 4e-07 rad.

        Returns
        -------
        Batch
            Modelled phase in RADIANS, one plane per date, on the same grid as
            this stack. REAL and UNWRAPPED: `velocity * t` is rad/yr times years
            and is never reduced, so 20 mm/yr over 3 years is 13.59 rad and 50
            mm/yr over 5 years is 56.6 rad.

            Radians are the primitive because the conversion runs ONE WAY.
            `.iexp(-1)` turns this into a phasor whenever one is wanted, but no
            operation turns a phasor back: .angle() only returns [-pi, pi], and
            at 20 mm/yr the rate it implies comes back as -1.52 mm/yr, the wrong
            sign. Returning phasors here would destroy the model's one advantage
            over the data -- that its prediction is unwrapped by construction.

        Examples
        --------
        >>> model  = stack.fit3d()
        >>> ground = stack * stack.predict(model=model).conj()
        >>> resid  = stack * stack.predict(model=model, baseline='BPR').conj()
        >>> # modelled displacement per date, unwrapped:
        >>> disp = (stack.predict(model=model, phasor=False)
        ...              .displacement_los(stack.transform()))
        """
        import numpy as np
        import xarray as xr
        import dask.array as da

        out = {}
        for key, ds in self.items():
            pols = [v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c' and 'date' in ds[v].dims
                    and 'y' in ds[v].dims and 'x' in ds[v].dims]
            if len(pols) > 1:
                raise ValueError(
                    f"predict() takes ONE polarisation, burst '{key}' carries "
                    f"{len(pols)}: {pols}. fit3d() has the same rule.")
            if not pols:
                raise TypeError(
                    f"predict() found no complex (date, y, x) variables in '{key}'.")
            pol = pols[0]
            da_xr = ds[pol]
            if da_xr.dims[0] != 'date':
                da_xr = da_xr.transpose('date', ...)

            mds = model[key]
            missing = [v for v in ('velocity', 'height', 'seasonal') if v not in mds]
            if missing:
                raise KeyError(
                    f"predict() needs {missing} in the model for '{key}'. "
                    "Pass the Batch returned by fit3d().")

            # TIME ORIGIN MUST MATCH THE FIT. utils_arcs zeroes t at the MASTER
            # -- argmin|B_perp| -- not at the first date, because there the phase
            # is zero by construction and the height term vanishes with the
            # baseline carrying it. Velocity and height absorb an origin shift
            # into the free constant, but car = exp(2j*pi*t) does NOT: getting
            # this wrong rotates the fitted annual by exp(2j*pi*delta) and leaves
            # a residual annual of amplitude 2|sin(pi*delta)|*|seasonal|, so
            # removing the model ADDS a seasonal. Days, as the kernel uses,
            # not seconds.
            dday = (np.asarray(ds.coords['date'].values)
                    .astype('datetime64[D]').astype(np.float64))
            # BPR is read for the ORIGIN even when baseline is None -- that
            # argument decides whether the height term is applied, not where
            # time starts. With no baseline the kernel has B = zeros, so
            # argmin|B| is index 0 and the origin is the first date; this
            # reproduces that too.
            _b = None
            if baseline and baseline in ds:
                _b = np.asarray(ds[baseline].values, dtype=float)
                while _b.ndim > 1:
                    _b = np.nanmean(_b, axis=-1)
            if _b is None or _b.shape != dday.shape:
                _b = np.zeros_like(dday)
            # THE MODEL CARRIES ITS OWN EPOCH, in a scalar `date` coordinate
            # every fit writes -- the master for the per-date fits, the median
            # acquisition for Batch.fit1d(), which sees only baseline
            # DIFFERENCES and cannot recover argmin|B| from them. Reading it
            # back is what makes predict() the exact inverse of whichever fit
            # produced the model rather than of one of them. The fallback stays
            # for a model written before the coordinate existed.
            t0 = None
            if 'date' in mds.coords and mds.coords['date'].ndim == 0:
                t0 = float(np.asarray(mds.coords['date'].values)
                           .astype('datetime64[D]').astype(np.float64))
            if t0 is None:
                t0 = float(dday[int(np.argmin(np.abs(_b)))])
            t = ((dday - t0) / 365.25).astype(np.float64)

            # ele2phase per date: BPR / median(R sin(inc)), and
            # elevation_phase() = 4 pi / (lambda R sin(inc))
            ele2phase = None
            if baseline and baseline in ds:
                _fac = Batch._elevation_phase_approximate(
                    transform if transform is not None else self)[key]
                bp = np.asarray(ds[baseline].values, dtype=float)
                while bp.ndim > 1:
                    bp = np.nanmean(bp, axis=-1)
                _lam = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
                ele2phase = bp / ((4.0 * np.pi / _lam) / _fac)

            vel = mds['velocity'].data
            hgt = mds['height'].data
            sea = mds['seasonal'].data
            planes = []
            for i in range(len(t)):
                # THE SAME RADIANS AS EVERYTHING ELSE: what displacement_los()
                # converts. A per-date series relates to displacement the
                # OPPOSITE way a pair does -- an SLC phase is -(4pi/lambda)r, so
                # d = +psi/m2r per date while d = -phi/m2r per pair -- and
                # displacement_los()'s -lambda/4pi is the pair relation. Emitting
                # the raw SLC phase here made a planted -20 mm/yr subsidence read
                # back as +20.00 mm/yr through displacement_los while the pair
                # branch gave -20.00. So the per-date prediction is published as
                # master*conj(date), which IS an interferometric phase and obeys
                # the one convention.
                #
                # REMOVAL THEREFORE ROTATES THE OTHER WAY:
                #     residual = stack * predict(model).iexp(sign=+1)
                # iexp() defaults to exp(-1j*phase), which is the removal
                # direction for the raw SLC phase this no longer returns.
                phi = vel * float(t[i])
                if ele2phase is not None:
                    # height is NaN where no baseline was available; a NaN would
                    # poison the whole date, so contribute only where it solved
                    phi = phi - da.nan_to_num(hgt) * float(ele2phase[i])
                car = np.exp(2j * np.pi * float(t[i]))
                phi = phi + (sea.real * car.real + sea.imag * car.imag)
                planes.append(phi.astype(np.float32))
            pred = da.stack(planes, axis=0)
            if ref is not None:
                # WHICH ACQUISITION READS ZERO -- see the docstring. Every
                # difference is invariant to this, so it cannot change a
                # removal or a rate, only which plane sits at zero.
                _i = Batch._ref_index(ref, ds.coords['date'].values)
                pred = pred - pred[_i]

            # THE GRID IS THE MODEL'S, THE DATES ARE THIS STACK'S. A model is
            # routinely fitted on multilooked phase and then predicted against
            # the stack it came from -- `downsample(30)` in the usual pipeline
            # -- so the two carry the same extent at different postings. Taking
            # y and x from the stack's own variable asserted they were equal
            # and raised as soon as they were not: "conflicting sizes for
            # dimension 'x': length 2915 on the data but length 23330 on
            # coordinate 'x'". The planes are built from the model's rasters,
            # so the model's axes are the ones that describe them.
            # BARE AXES, not the model's DataArrays. The model carries its epoch
            # as a SCALAR `date` coordinate, and that scalar travels with any
            # coordinate or variable taken from it -- which then collides with
            # the `date` DIMENSION this prediction has:
            # "dimension 'date' already exists as a scalar variable".
            coords = {'date': np.asarray(da_xr.coords['date'].values),
                      'y': np.asarray(mds.coords['y'].values),
                      'x': np.asarray(mds.coords['x'].values)}
            pds = xr.Dataset({pol: xr.DataArray(pred, dims=('date', 'y', 'x'),
                                                coords=coords)}, attrs=ds.attrs)
            sref = (mds['spatial_ref'] if 'spatial_ref' in mds.coords
                    else (ds['spatial_ref'] if 'spatial_ref' in ds.coords else None))
            if sref is not None:
                sref = sref.drop_vars(list(sref.coords), errors='ignore')
                pds = pds.assign_coords(spatial_ref=sref)
            out[key] = pds
        return Batch(out)

    @serialize_gpu
    def fit3d(self, threshold: float = 0.5, window: tuple = (32, 128),
                cell: tuple = (2, 8),
                baseline: str = 'BPR', budget: 'str | None' = None,
                level: int = 1,
                max_dh: float = 100.0, max_dv: float = 50.0,
                step_dh: float = 4.0, step_dv: float = 2.0,
                max_seasonal: float = 5.0,
                consensus: 'tuple | None' = (8, 5.0),
                device: str = 'auto', iterations: int = 8,
                debug: bool = False) -> 'Batch':
        """
        Fit a per-pixel (height, velocity, seasonal) model on a PS network.

        Returns the MODEL ONLY -- one dataset of named parameters, no phase.
        Use predict(model=...) to reconstruct phase from it, and subtract
        whatever you actually want removed; the caller decides, not the fit. Solved
        PER DASK CHUNK with no inter-chunk state:

          nodes    every pixel `arcs()` certifies; nothing is thinned, since
                   the cell constrains ARCS, not nodes
          arcs     each node's best partners by raw coherence inside four
                   half-offset windows, then a maximum spanning forest and a
                   second arc per node, every edge clearing the independence
                   cell -- a pair inside the cell is one ground sample, and
                   its coherence is the impulse response
          model    joint (height, velocity) per arc; they are NOT separable
                   one at a time, because the perpendicular baseline is not a
                   smooth function of time
          network  least squares onto per-node values; each component's free
                   datum is estimated from its own mean residual and removed,
                   so a network in several pieces costs nothing
          ground   node phase minus its own HEIGHT term only. Pixels that
                   carry neither a node nor an attached DS are NaN; nothing is
                   interpolated into them
          level 1  every DS is fitted against the PS nodes inside the PS
                   EXTENT -- the same reach the network arcs use, not the
                   smaller DS window, since a PS is by definition a scatterer
                   that holds a fitted arc that far. It inherits the chosen
                   node's height, rate, seasonal and component LABEL, so it
                   lands on the same datum. A DS with no arc clearing
                   `threshold` stays NaN: without a coherent path to the
                   network it has no datum
          level 2  the DS attached at level 1 are then offered as partners to
                   whatever is still unresolved, under the same rules and
                   inside the DS WINDOW -- a DS is certified only that far

        Returns ONE dataset carrying the solve, its variables named
        by quantity:

          `velocity`   rad/yr
          `height`     rad per unit ele2phase
          `seasonal`   complex rad, the fitted annual
          `coherence`  arc coherence tying the pixel to the network
          `rmse`       sqrt(-2 ln coherence), rad
          `conncomp`   int8, -1 nodata, 0 the largest component

        all NaN (or -1) where nothing was solved. Names rather than positions,
        so a caller never counts commas and adding a quantity moves nothing.
        There is NO polarisation prefix: fit3d() takes exactly one polarisation
        and raises otherwise, so nothing needs disambiguating and predict() can
        look the names up directly.

        RADIANS OUT, as everywhere else here. `displacement_los()` stays the
        single place a length is produced. `coherence` is the arc quality that
        ties each pixel in -- the mean over its own arcs for a node, the
        attaching arc for a densified DS -- and `rmse` is its exact inverse
        transform, carried for convenience rather than as new information.

        VELOCITY IS A RASTER, not a `.stats` entry. The kernel's stats dict is a
        function attribute written by whichever block ran last in the worker, so
        under dask it describes ONE chunk and misdescribes the others -- it
        reported 200 nodes for a raster carrying 29118. A product that cannot be
        rebuilt from what the method returns is not really returned; `.stats`
        stays for single-block diagnostics only.

        NO ATMOSPHERIC SCREEN IS COMPUTED, by measurement rather than
        omission. A per-node screen kriged to the ground lowered coherence at
        every separation: a node residual is dominated by its own noise rather
        than by correlated signal, so interpolation spreads mostly that error.
        Published kriging estimators were reproduced and behave the same way.
        The one term with a positive effect is a per-epoch stratified delay
        proportional to ELEVATION, and only marginally, concentrated at long
        range -- not enough to put in this path.

        Parameters
        ----------
        threshold : float
            Arc coherence for an arc to count and to be kept in the network.
        window : tuple of int
            `(wy, wx)` is the DS window in pixels, centred on each pixel: the
            neighbourhood the short arc test measures over. `(wy, wx, py, px)`
            sets the PS extent apart from it, otherwise the extent is derived
            from the window.

            The PS extent is the reach of everything that tests against a NODE
            -- the long arcs that prove a PS, the network arcs that carry the
            datum, and the DS attachment. It is therefore the caller's cost
            dial as well: widening it grows the candidate partners per pixel
            and the arc fits with them.
        baseline : str
            Variable holding the perpendicular baseline per date.
        iterations : int
            Refinement passes per arc, for the arcs that reach the final fit. The per-arc search is a lattice followed
            by a majorise-minimise refinement, and the refinement's step
            contracts by exactly `(1 - gamma)` per pass -- so the useful count
            follows from `threshold`, not from taste. At a 0.4 gate the
            contraction is 0.6 and eight passes leave 0.6**8, under two percent
            of a lattice cell: 0.07 m at `step_dh=4`, 0.03 mm/yr at
            `step_dv=2`. Refining far below the step it sits inside buys
            nothing.

            It is also the larger half of the attachment's cost, since stage 1
            is one product over the box while this runs on every arc this many
            times. A lower `threshold` contracts more slowly and wants more
            passes; a higher one wants fewer.
        consensus : tuple or None
            How much agreement is required before a value is reported, asked
            once for both halves of the solve: an arc must agree with the
            network, a DS's best partners must agree with each other. Same
            machinery in both places -- IRLS to find the consistent set and
            rejection beyond a robust sigma.

            `min_agreeing` names HOW MANY of a DS's partners must agree, and
            it is the BEST that many, ranked by arc coherence -- not any that
            many out of however many are in reach. Ranking is settled before
            any value is read, so it cannot be chosen to suit the answer, and
            the selected partners are the ONLY ones the centre, the scale and
            the rejection ever see: a robust scale across a mixture of arc
            qualities describes none of them.

            It therefore has to be large enough to CARRY that scale, which is a
            stricter requirement than having enough votes. With three residuals
            the robust scale is the smaller of the two gaps, so two partners
            landing close collapse it and the third is rejected however good it
            was. Below about five the gate rejects on the accident of spacing
            rather than on the pixel, and reported coverage is NOT monotonic in
            `min_agreeing` for that reason -- a larger value can return more
            points because its scale stops discarding good partners.

            The cost is paid by the network: `min_agreeing` also decides how
            many arcs a node needs to survive, so PS thin out as it rises.

            A bare `min_agreeing` uses the defaults for the rest;
            `(min_agreeing, reject_sigma)` is the default (8, 5.0), and
            `(min_agreeing, reject_sigma, irls_passes)` sets the pass count too.
            Note `(3)` is just the integer 3 in Python -- the one-element tuple
            is `(3,)` -- and both are accepted.
            A node the rejection leaves below `min_agreeing` arcs leaves the
            rasters, the stats and the DS attachment together; a DS whose
            partners cannot muster that many agreeing is not attached.

            None turns it off: the network is integrated by plain least squares
            and a DS takes its best arc. That is the right setting when the
            caller is tuning `threshold` themselves and does not want a second
            rule moving the answer underneath them -- the checks exist to make a
            LOW threshold usable, by removing what a low gate lets in.

        level : int
            How far to carry the solve, 0 to 2. Each level uses the one below
            it as its references, so they are cumulative rather than
            alternative.

            0   the PS network only. Nodes carry values; every other pixel is
                NaN. This is the answer when a caller wants only what the PS
                test certified.
            1   plus DS attached to PS, each by `consensus` agreeing PS
                partners inside the PS extent.
            2   plus DS attached to the DS of level 1, by `consensus` agreeing
                partners inside the DS window. A pixel can be plainly
                connectable and still fail level 1 where the PS are too sparse
                to field that many -- a property of the ground, not of the
                pixel -- and the level-1 DS are dense enough to ask instead.

            The default is 1. Level 2 costs the most arcs by far and is much
            the slowest stage, and every reference it uses is itself one hop
            from the network, so it adds coverage at a somewhat higher error
            rate. That is a trade worth making deliberately rather than by
            default.

            There is no level 3. Each hop's error adds, and level-2 pixels are
            certified by DS rather than by PS, so the argument that justifies
            level 2 does not survive another step.

        debug : bool
            Print a stage-by-stage account of the solve: how many nodes the PS
            test found, how many pairs were fitted and how many cleared
            `threshold`, what the robust pass rejected, the connected
            components and their sizes, and where the DS candidates went --
            too few partners, no consensus, or attached. Off by default.

            The counts are the ones that explain a disappointing result. A
            thin answer is either few nodes, few arcs, a network in pieces, or
            DS that reached a node and failed consensus, and those call for
            different changes -- the returned rasters alone cannot tell them
            apart.

        budget : str or None
            Memory budget for the arc-counting slabs, e.g. '512MB'.
        max_dh, max_dv : float
            Largest DIFFERENTIAL height (m) and rate (mm/yr) an arc may carry:
            the difference between two neighbours a few tens of metres apart,
            not an absolute elevation or velocity. Anything solving beyond
            them returns NaN rather than a plausible wrong number, so these
            are the guarantee, not a hint -- the search runs wider than they
            say so that max_dv=100 detects 99 mm/yr on its merits and never
            against a boundary. Narrow them for speed when the terrain
            allows, widen them for a rapidly deforming one.
        step_dh, step_dv : float
            Lattice step in height (m) and rate (mm/yr). These pick which
            BASIN is found, NOT the accuracy -- the refinement is continuous
            and absorbs the quantisation over a wide range of steps. Raise
            them to go faster; the failure when they are finally too coarse is
            detected, not silent.
        max_seasonal : float
            Largest annual amplitude to admit, in mm of LOS (HALF amplitude, so
            60 means a 120 mm peak-to-peak swing). 0 (default) leaves the annual
            term out of the model entirely.

            It is not a refinement: an annual term of amplitude A radians leaves
            coherence |J0(A)| at the true rate and |J1(A)| one cycle/yr away,
            and they cross at A = 1.435 rad. Above that the sideband IS the
            higher maximum, so a {height, rate} fit returns the sideband rather
            than the truth; with the term in the model the rate returns to its
            no-seasonal accuracy.

            It costs search time, and a little accuracy when there is no annual
            signal at all, so it is cheap to leave on. Large amplitudes are
            only partly recovered, but they fail LOUDLY -- NaN rather than
            silent wrong rates. Where a stack carries no seasonal signal the
            default 0 is right; zones with a real one are what this is for.
            ON ARCS, KEEP IT SMALL. A seasonal signal is long-wavelength, so
            an arc -- two pixels tens of metres apart -- sees only the small
            residue that does not cancel in the difference. A large
            max_seasonal there is wrong twice over, since it searches thousands
            of lattice points for an amplitude that cannot be present.

            Small, it earns its keep: marginal arcs are rescued and nodes
            isolated at any threshold join the network. Set too small, the arcs
            are rescued but fitted poorly, so the amplitude does need room to
            move.

            Judge any gain against a MATCHED-gamma null, not a raw one: two free
            parameters always raise gamma, and pure-noise arcs sit low enough
            that there is far more room to climb there than at a real arc, so an
            unmatched comparison understates the real gain.

            Whether the atmosphere is itself seasonal is a property of the
            stack and has to be checked there, against a permuted-date null
            rather than by eye. On a stack with genuinely seasonal delay the
            annual term would absorb it, and per pixel the two are not
            separable.

            What it does fix, where a real seasonal signal exists, is the
            contamination of dh and dv by leaving it out: an unmodelled annual
            term biases the height and can push the rate onto a whole sideband,
            while modelling it returns both to their clean values.

        Returns
        -------
        BatchCore
            complex64 unit phasors of the atmospheric phase, NaN where not
            solved.

        THE SCREEN COMES FROM PS AND FROM NOTHING ELSE. Distributed scatterers
        can receive it; they cannot source it, because their residuals are noise
        and averaging more of them converges to zero rather than to the delay.
        The nodes are therefore the PS raster -- a DS window's best pixel,
        verified against partners BEYOND the window -- and the network is bounded
        by the same PS window, so `window` is the only reach the caller sets.

        RETURNS TWO THINGS, AND THE SECOND IS NOT DECORATION. Each connected
        component of the network carries its own free constant per date, because
        arcs cancel exactly that, so two pixels are comparable only if they came
        from the same component. `labels` says which, exactly as the 2D
        unwrapping reports its own components, and a caller who ignores it will
        compare values that share no reference.

        Returns
        -------
        Batches of two Batch
            `model = stack.fit3d(...)`, then `stack.predict(model=model)`.

            screen  complex unit phasors per date, NaN where no component
                    reached the pixel. NaN is the answer there: away from the
                    nodes the field is extrapolated, not measured.
            labels  int8. 0 is the largest component, 1 the next, by node
                    count; -1 is nodata. A scene that needs more than 127 has
                    shattered rather than resolved, and says so rather than
                    folding one component's label onto another's. Where two
                    components reach one pixel the screen is taken from the one
                    with the more arcs per node, never averaged between --
                    their datums are unrelated, and mixing them is worse than
                    either alone.

        Examples
        --------
        >>> model = stack.fit3d()
        >>> predicted = stack.predict(model=model)
        >>> model['velocity'].plot(cmap='turbo')   # rad/yr
        >>> good = model.where(model['conncomp'] == 0)
        >>> main = velocity.where(labels == 0)   # one datum, comparable
        """
        # DELEGATE BY STACK TYPE, exactly as fit1d() does: pairs take the
        # pair branch (not implemented), dates take the PS network below.
        if any('pair' in ds[v].dims
               for ds in self.values() for v in ds.data_vars):
            # `window` is the PS network's (32, 128) tuple on the date path; on the
            # pair path it is the side of the box the covariance is estimated over, so
            # a scalar. A tuple is reduced to its smallest side rather than refused.
            # the split is kept so a pair-domain fit has a home when one works
            raise NotImplementedError(
                'fit3d() does not support complex PAIRS. Use the per-DATE stack, '
                'or unwrap and call Batch.fit1d() on the unwrapped pairs.')
        # validated HERE, not only in the kernel: fit3d() returns a lazy Batch,
        # so a bad value would otherwise surface at compute time far from where
        # it was written
        from . import utils_arcs as _ua
        _ua._3d_consensus(consensus)
        if int(level) not in (0, 1, 2):
            raise ValueError(f'level must be 0, 1 or 2; got {level!r}')
        return self._fit3d_ps_impl(
            threshold=threshold, window=window, cell=cell,
            baseline=baseline, budget=budget, level=level,
            max_dh=max_dh, max_dv=max_dv, step_dh=step_dh, step_dv=step_dv,
            max_seasonal=max_seasonal, consensus=consensus, device=device,
            debug=debug,
            iterations=iterations)



    def _fit3d_ps_impl(self, threshold, window, cell, baseline,
                         budget, max_dh, max_dv, step_dh, step_dv,
                         max_seasonal, level=1, consensus=(8, 5.0),
                         device='cpu', iterations=8, debug=False):
        """The PS screen and its component labels, per dask block.

        One block at a time and no inter-block state, like arcs(): a component
        is a property of the arcs inside a block, and stitching components
        across blocks would tie together datums that nothing in the data
        relates. A caller who needs one datum over a wider area asks for a
        wider chunk.
        """
        import numpy as np
        import xarray as xr
        import dask.array as da
        from . import utils_arcs
        from .Batch import Batch, Batches
        from .utils_dask import get_dask_chunk_size_mb

        # '4GB' -> 4096, as fit1d() does. Typed `str | None`, so a string has
        # to be parsed rather than handed to float() further down.
        from .BatchCore import _parse_budget
        budget_mb = (get_dask_chunk_size_mb() if budget is None
                     else _parse_budget(budget) if isinstance(budget, str)
                     else float(budget))
        wy, wx, pey, pex = utils_arcs._3d_windows(window)
        # the geometry, not re-derived here
        _ep_batch = Batch._elevation_phase_approximate(self)
        model_result = {}
        for key, ds in self.items():
            pols = [v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c' and 'date' in ds[v].dims
                    and 'y' in ds[v].dims and 'x' in ds[v].dims]
            if len(pols) > 1:
                raise ValueError(
                    f"fit3d() fits ONE polarisation; burst '{key}' carries "
                    f"{len(pols)}: {pols}. The model variables are named by "
                    "quantity alone (velocity, height, ...), so two polarisations "
                    "would collide. Select one first, e.g. batch[['VV']].")
            if not pols:
                raise TypeError(
                    f'fit3d() found no complex (date, y, x) variables in '
                    f'burst {key}')
            date_values = np.asarray(ds.coords['date'].values)
            bp = (np.asarray(ds[baseline].values)
                  if baseline and baseline in ds else None)
            lam_ = float(np.asarray(ds['radar_wavelength'].values).ravel()[0])
            # elevation_phase = 4 pi / (lambda R sin(inc))
            geom = (lam_, (4.0 * np.pi / lam_) / _ep_batch[key])
            yv = np.asarray(ds['y'].values, dtype=float)
            xv = np.asarray(ds['x'].values, dtype=float)
            spacing = (abs(float(yv[1] - yv[0])) if yv.size > 1 else 1.0,
                       abs(float(xv[1] - xv[0])) if xv.size > 1 else 1.0)

            mvars = {}
            for pol in pols:
                da_xr = ds[pol]
                if da_xr.dims[0] != 'date':
                    da_xr = da_xr.transpose('date', ...)
                dsk = da_xr.data.rechunk({0: -1})

                # ONE call per block, split afterwards. The screen and the
                # labels are two faces of the same solve -- the labels say
                # which component each pixel's screen came from -- so running
                # the kernel once per output would fit every arc twice. They
                # travel as one array with the labels riding in a trailing
                # plane, which costs one date's worth of memory and is exact:
                # a component index is a small integer and complex64 carries it
                # without rounding.

                def _blk(block, _d=date_values, _bp=bp, _w=(wy, wx, pey, pex),
                         _t=float(threshold), _c=tuple(cell), _g=geom,
                         _sp=spacing, _bm=budget_mb,
                         _mh=float(max_dh), _mv=float(max_dv),
                         _sh=float(step_dh), _sv=float(step_dv),
                         _se=float(max_seasonal), _dn=int(level),
                         _cs=(consensus if consensus is None
                              or isinstance(consensus, (int, float))
                              else tuple(consensus)),
                         _dv2=str(device), _it=int(iterations),
                         _db=bool(debug)):
                    l, v, h, sa, cg = utils_arcs._3d_fit_ps_array(
                        block, _d, spacing=_sp, bperp=_bp, window=_w,
                        threshold=_t, cell=_c, geometry=_g,
                        budget=_bm, level=_dn, max_dh=_mh, max_dv=_mv,
                        step_dh=_sh, step_dv=_sv, max_seasonal=_se,
                        consensus=_cs, device=_dv2, iterations=_it,
                        debug=_db)
                    # ONE CONVENTION ACROSS EVERY FIT: displacement_los() must turn
                    # this model into a negative rate where the ground subsides,
                    # whichever fit produced it. _3d_arc_fit solves the per-DATE
                    # phase, and a pair runs opposite to it -- an SLC phase is
                    # -(4pi/lambda)r, so ref*conj(rep) carries +m2r*dr while a date
                    # series carries -m2r*dr. displacement_los()'s -lambda/4pi is
                    # derived for the pair, so the pair sense is the one the
                    # library converts. Returning the date sense reports
                    # subsidence as uplift.
                    #
                    # HEIGHT IS NOT NEGATED. Its per-date term +hgt*e2p_d
                    # differences to -hgt*e2p_pair, which already matches the pair
                    # convention, and it is why a global sign flip on the
                    # prediction does not work.
                    v = -v
                    sa = -sa
                    return np.concatenate(
                        [l[None].astype(np.complex64),
                         v[None].astype(np.complex64),
                         h[None].astype(np.complex64),
                         sa[None].astype(np.complex64),
                         cg[None].astype(np.complex64)], axis=0)
                # FIVE planes out, never n_dates+5: the kernel no longer builds a
                # phase it would only have to throw away. predict(model)
                # reconstructs phase when a caller actually wants it.
                # THE SAME HALO arcs() USES, AND FOR THE SAME REASON. Each
                # block is still solved on its own and nothing is merged
                # across them -- a component is a property of the arcs inside
                # one block -- but a node at a block edge has to be able to
                # REACH its partners, and they lie out to the PS extent. With
                # no halo a narrow block held no complete network and returned
                # an empty raster, so the chunking silently decided what the
                # answer was.
                #
                # The depth, and the check that the chunks can carry it,
                # come from `_3d_depth` -- the one place that decides how far
                # the arc search reaches, so this and arcs() cannot disagree.
                from dask.array.overlap import overlap, trim_internal
                dep_y, dep_x = utils_arcs._3d_depth(dsk.chunks[1:], window)
                _dep = {0: 0, 1: int(dep_y), 2: int(dep_x)}
                _ov = overlap(dsk, depth=_dep, boundary='none')
                both = trim_internal(
                    da.map_blocks(
                        _blk, _ov, dtype=np.complex64,
                        chunks=((5,),) + _ov.chunks[1:],
                        meta=np.empty((0, 0, 0), np.complex64)),
                    _dep)
                lb = both[0].real.astype(np.int8)
                vv = both[1].real.astype(np.float32)
                hh_ = both[2].real.astype(np.float32)
                sa_ = both[3].astype(np.complex64)
                cg_ = both[4].real.astype(np.float32)
                # rmse is sqrt(-2 ln gamma), derived here rather than carried as
                # a sixth plane: the two are exact inverses, so shipping both
                # through the graph would move the same information twice
                rr = da.sqrt(da.maximum(
                    -2.0 * da.log(da.clip(cg_, 1e-9, 1.0)), 0.0)
                    ).astype(np.float32)
                coords = {k_: v for k_, v in da_xr.coords.items()
                          if k_ in ('y', 'x', 'spatial_ref')}
                # ONE dataset for the model, variables named by QUANTITY
                # alone. Positional outputs make a caller count commas; a name
                # does not move when the list grows. No polarisation prefix --
                # fit3d() takes exactly one polarisation, so nothing needs
                # disambiguating.
                for nm_, arr_ in (('velocity', vv), ('height', hh_),
                                  ('seasonal', sa_), ('coherence', cg_),
                                  ('rmse', rr), ('conncomp', lb)):
                    mvars[nm_] = xr.DataArray(arr_, dims=('y', 'x'), coords=coords)
            mds = xr.Dataset(mvars, attrs=ds.attrs)
            # the epoch the model is referenced to -- the MASTER, where B_perp
            # is smallest, which is where _3d_fit_ps_array zeroes its t. Every
            # fit here records it under the same name, so predict() reads the
            # origin instead of reconstructing it.
            _dd = (np.asarray(date_values).astype('datetime64[D]')
                   .astype(np.float64))
            _b3 = np.zeros_like(_dd) if bp is None else np.asarray(bp, float).ravel()
            if _b3.shape != _dd.shape:
                _b3 = np.zeros_like(_dd)
            mds = mds.assign_coords(
                date=np.datetime64(int(_dd[int(np.argmin(np.abs(_b3)))]), 'D'))
            if 'spatial_ref' in ds.coords:
                mds = mds.assign_coords(spatial_ref=ds.spatial_ref)
            model_result[key] = mds
        return Batch(model_result)

    """
    This class has 'data' stack variable for the datasets in the dict.
    """
    def __init__(self, mapping: dict[str, xr.Dataset] | Stack | None = None):
        from .Stack import Stack
        # pick off the complex vars from Stack, PLUS the 1D metadata that rides
        # with them. Keeping only dtype.kind=='c' stranded the radar geometry:
        # radar_wavelength, near_range, earth_radius, SC_height_start,
        # rng_samp_rate and BPR are per-date 1D variables, so pairs() ->
        # BatchComplex dropped every one of them and nothing downstream could
        # convert units or build ele2phase. Batch.__init__ already keeps 1D
        # non-complex variables for exactly this reason; this makes the two
        # agree. Grids stay excluded -- a (y,x) real variable is data, not
        # metadata, and belongs in a Batch.
        if isinstance(mapping, Stack):
            complex_dict: dict[str, xr.Dataset] = {}
            for key, ds in mapping.items():
                keep = [
                    v for v in ds.data_vars
                    if ds[v].dtype.kind == 'c' or ds[v].ndim <= 1
                ]
                complex_dict[key] = ds[keep]
            mapping = complex_dict

        #print('BatchComplex __init__ mapping', mapping or {}, '\n')
        # delegate to your base class for the actual init
        super().__init__(mapping or {})

    def real(self, **kwargs):
        """
        Return the real part of each complex data variable,
        producing a Batch of real-valued Datasets.
        """
        out = {}
        for key, ds in self.items():
            # ds.map() applies the lambda to each DataArray in the Dataset
            ds_real = ds.map(lambda da: da.real, **kwargs)
            out[key] = ds_real
        return Batch(out)

    def imag(self, **kwargs):
        """
        Return the imaginary part of each complex data variable,
        producing a Batch of real-valued Datasets.
        """
        out = {}
        for key, ds in self.items():
            ds_imag = ds.map(lambda da: da.imag, **kwargs)
            out[key] = ds_imag
        return Batch(out)

    def abs(self, **kwargs):
        return Batch(self.map_da(lambda da: xr.ufuncs.abs(da), **kwargs))

    def power(self, **kwargs):
        """ element-wise |x|², i.e. signal intensity """
        # Optimized: avoid sqrt in abs() by computing real² + imag² directly
        return Batch(self.map_da(lambda da: da.real**2 + da.imag**2, **kwargs))

    def threshold(self, weight=None, threshold=np.pi/2) -> "BatchComplex":
        """
        Filter pixels by circular standard deviation (cstd) of pair phases.

        Computes weighted cstd across all pairs per pixel. Pixels with
        cstd >= threshold are set to 0+0j (all pairs). Useful for rejecting
        incoherent pixels before velocity estimation or detrending.

        Parameters
        ----------
        weight : BatchUnit or None
            Optional correlation weight for weighted cstd.
        threshold : float
            Maximum cstd in radians. Default π/2. Use π/4 for stricter filtering.

        Returns
        -------
        BatchComplex
            Filtered copy with incoherent pixels zeroed.
        """
        import dask.array as da
        import xarray as xr
        from . import utils_detrend

        BatchCore._require_lazy(self, 'threshold')

        results = {}
        for burst_id, burst_ds in self.items():
            burst_weight = weight[burst_id] if weight is not None else None
            filtered_vars = {}
            for pol in [v for v in burst_ds.data_vars if v not in ['ref', 'rep', 'BPR', 'BPT']]:
                data_da = burst_ds[pol]
                weight_da = burst_weight[pol] if burst_weight is not None else None

                if data_da.dims[0] != 'pair':
                    data_da = data_da.transpose('pair', ...)

                data_dask = data_da.data
                weight_dask = weight_da.data if weight_da is not None else None
                n_pairs_val = data_da.shape[0]

                def _threshold_block(data_block, weight_block=None,
                                     _threshold=threshold):
                    return utils_detrend.threshold_pairs_array(
                        [data_block],
                        [weight_block] if weight_block is not None else None,
                        threshold=_threshold,
                    )

                if weight_dask is not None:
                    filtered_dask = da.blockwise(
                        _threshold_block, 'dyx',
                        data_dask, 'pyx',
                        weight_dask, 'pyx',
                        new_axes={'d': n_pairs_val},
                        concatenate=True,
                        dtype=data_dask.dtype,
                        meta=np.empty((0, 0, 0), dtype=data_dask.dtype),
                    )
                else:
                    filtered_dask = da.blockwise(
                        _threshold_block, 'dyx',
                        data_dask, 'pyx',
                        new_axes={'d': n_pairs_val},
                        concatenate=True,
                        dtype=data_dask.dtype,
                        meta=np.empty((0, 0, 0), dtype=data_dask.dtype),
                    )

                filtered_vars[pol] = xr.DataArray(filtered_dask, dims=data_da.dims,
                                                   coords=data_da.coords, name=pol)

            filtered_ds = burst_ds.assign(filtered_vars)
            results[burst_id] = filtered_ds

        return BatchComplex(results)

    def backscatter(self, *args, **kwargs):
        """
        Compute backscatter intensity (sigma0) from radiometrically calibrated SLC data.

        This method requires the insardev_backscatter extension package.
        """
        raise ImportError(
            "backscatter() requires insardev_backscatter extension"
        )

    def adi(self, *args, **kwargs):
        """
        Compute Amplitude Dispersion Index (ADI) for calibrated σ₀ data.

        ADI = std(amplitude) / mean(amplitude) over time.
        Lower ADI indicates more stable scatterers (PS candidates).

        This method requires the insardev_backscatter extension package.
        """
        raise ImportError(
            "adi() requires insardev_backscatter extension"
        )

    def conj(self, **kwargs):
        """intfs.iexp().conj() for np.exp(-1j * intfs)"""
        return self.map_da(lambda da: xr.ufuncs.conj(da), **kwargs)

    def pairs(self, pairs):
        """Select date pairs from per-date data, returning ref and rep stacks.

        Parameters
        ----------
        pairs : array-like (n_pairs, 2)
            Pairs as [[ref_date, rep_date], ...]. Dates as datetime64 or indices.

        Returns
        -------
        tuple (ref, rep)
            Two BatchComplex with 'pair' dimension instead of 'date'.
        """
        import numpy as np
        pairs = np.asarray(pairs)
        ref_dates = pairs[:, 0]
        rep_dates = pairs[:, 1]

        # Map dates to integer indices (match by day to handle precision differences)
        key0 = list(self.keys())[0]
        date_coords = self[key0].coords['date'].values
        # Truncate to day precision for matching
        date_days = np.array(date_coords, dtype='datetime64[D]')
        date_to_idx = {d: i for i, d in enumerate(date_days)}
        ref_idx = [date_to_idx[np.datetime64(d, 'D')] for d in ref_dates]
        rep_idx = [date_to_idx[np.datetime64(d, 'D')] for d in rep_dates]

        # Select, rename date→pair, and assign pair coords matching the caller
        n_pairs = len(ref_idx)
        pair_coords = np.arange(n_pairs)
        screen_ref = self.isel(date=ref_idx).rename(date='pair').map(
            lambda ds: ds.assign_coords(pair=pair_coords))
        screen_rep = self.isel(date=rep_idx).rename(date='pair').map(
            lambda ds: ds.assign_coords(pair=pair_coords))

        return screen_ref, screen_rep

    def angle(self, **kwargs):
        """
        Compute element-wise phase (angle) for the complex variables only,
        returning a BatchWrap of float32 DataArrays in [-π, π].
        """
        out = {}
        for k, ds in self.items():
            # select only the vars whose dtype is complex
            complex_vars = [
                var for var in ds.data_vars
                if ds[var].dtype.kind == 'c'
            ]
            if not complex_vars:
                # no complex vars → skip
                continue

            # subset to just those, then map over each DataArray. The 1D radar
            # metadata rides along afterwards -- angle() of near_range is
            # meaningless, but DROPPING it strands every downstream unit
            # conversion, which is how radar_wavelength went missing.
            ds_complex = ds[complex_vars]
            ds_phase = ds_complex.map(
                lambda da: xr.ufuncs.angle(da).astype('float32'),
                **kwargs
            )

            meta = [v for v in ds.data_vars
                    if v not in complex_vars and ds[v].ndim <= 1]
            if meta:
                ds_phase = ds_phase.assign({v: ds[v] for v in meta})
            ds_phase.attrs = ds.attrs
            out[k] = ds_phase

        # package up as a BatchWrap (real, wrapped-phase)
        return BatchWrap(out)

    def unwrap2d(self, *args, **kwargs):
        """Unwrap complex interferogram via .angle() conversion."""
        return self.angle().unwrap2d(*args, **kwargs)

    def unwrap2d_irls(self, *args, **kwargs):
        """Unwrap complex interferogram via .angle() conversion."""
        return self.angle().unwrap2d_irls(*args, **kwargs)

    def plot(self, *args, **kwargs):
        """
        Plot complex phase as wrapped phase via .angle() conversion.
        """
        return self.angle().plot(*args, **kwargs
        )

    @staticmethod
    @serialize_gpu
    def _goldstein(phase_np, corr_np, psize=32, threshold=0.5, device='auto'):
        """
        Apply Goldstein adaptive filter.

        Uses loop-based processing for CPU (constant memory) and
        PyTorch unfold/fold for GPU (vectorized).

        Parameters
        ----------
        phase_np : np.ndarray
            2D complex numpy array of phase data.
        corr_np : np.ndarray
            2D real numpy array of correlation values.
        psize : int or dict
            Patch size for the filter. Default is 32.
        threshold : float
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
        device : str, optional
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        np.ndarray
            Filtered complex array with same shape as input.
        """
        import numpy as np
        from .BatchCore import BatchCore
        from .utils_goldstein import goldstein_numpy, goldstein_pytorch

        if psize is None:
            return phase_np

        # Handle (1, y, x) arrays from apply_ufunc
        squeeze = False
        if phase_np.ndim == 3 and phase_np.shape[0] == 1:
            phase_np = phase_np[0]
            corr_np = corr_np[0] if corr_np.ndim == 3 else corr_np
            squeeze = True

        if isinstance(psize, dict):
            psize_y, psize_x = psize['y'], psize['x']
        else:
            psize_y, psize_x = int(psize), int(psize)

        # Ensure correct dtypes (goldstein functions require complex64/float32)
        if phase_np.dtype != np.complex64:
            phase_np = phase_np.astype(np.complex64)
        if corr_np.dtype != np.float32:
            corr_np = corr_np.astype(np.float32)

        # Dispatch based on device
        dev = BatchCore._get_torch_device(device)

        if dev.type == 'cpu':
            result = goldstein_numpy(phase_np, corr_np, psize_y, psize_x, threshold=threshold)
        else:
            import torch
            result = goldstein_pytorch(phase_np, corr_np, psize_y, psize_x, dev, threshold=threshold)
            if dev.type == 'mps':
                torch.mps.empty_cache()
            elif dev.type == 'cuda':
                torch.cuda.empty_cache()

        if squeeze:
            result = result[np.newaxis, ...]
        return result

    def goldstein(self, corr: BatchUnit, window: int | dict[str, int] = 32, threshold: float = 0.5,
                  device: str = 'auto', debug: bool = False):
        """
        Apply Goldstein adaptive filter to each dataset in the batch.

        Parameters
        ----------
        corr : BatchUnit
            Batch of correlation values to use for filtering.
        window : int or dict[str, int], optional
            Patch size for the filter. If int, same size used for both dimensions.
            If dict, specify {'y': size_y, 'x': size_x}. Default is 32.
        threshold : float, optional
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        BatchComplex
            New batch with filtered phase values
        """
        import numpy as np
        import dask.array as da

        if debug:
            print('DEBUG: goldstein')

        if window is None:
            return self

        # Check if correlation is a BatchUnit by checking its class name
        if corr.__class__.__name__ != 'BatchUnit':
            raise ValueError("corr must be a BatchUnit")

        if set(corr.keys()) != set(self.keys()):
            raise ValueError("corr must have the same keys as self")

        # Validate lazy data
        BatchCore._require_lazy(self, 'goldstein')

        if isinstance(window, int):
            window = {'y': window, 'x': window}
        elif isinstance(window, (tuple, list)):
            window = {'y': window[0], 'x': window[1]}

        # Resolve device ONCE here, not in every task
        if device == 'auto':
            resolved_device = BatchCore._get_torch_device(device, debug=debug)
            device = resolved_device.type  # 'cpu', 'cuda', or 'mps' as string

        # Apply Goldstein filter to each dataset
        result = {}
        for k in self.keys():
            ds = self[k]
            corr_ds = corr[k]
            filtered_vars = {}

            # Process each complex data variable in the dataset
            for var_name, var_data in ds.data_vars.items():
                if var_data.dtype.kind == 'c':  # Only process complex variables
                    corr_da = corr_ds[var_name]
                    phase_dask = var_data.data
                    corr_dask = corr_da.data

                    # Require first dimension chunked as 1 (avoid hidden rechunking overhead)
                    chunks = phase_dask.chunks
                    if var_data.ndim == 3 and chunks[0][0] != 1:
                        raise ValueError(
                            f"goldstein() requires first dimension chunked as 1, got chunks {chunks[0]}. "
                            f"Data should already have pair=1 chunks from load()."
                        )

                    # Calculate overlap depth: window//2 + 2 (PyGMTSAR formula)
                    depth_y = window['y'] // 2 + 2
                    depth_x = window['x'] // 2 + 2

                    if debug:
                        print(f'DEBUG: goldstein map_overlap depth=({depth_y}, {depth_x})')

                    depth_2d = {0: depth_y, 1: depth_x}
                    depth_3d = {0: 0, 1: depth_y, 2: depth_x}
                    filtered_dask = da.map_overlap(
                        _apply_goldstein_2d_for_dask,
                        phase_dask,
                        corr_dask,
                        depth= depth_3d if var_data.ndim == 3 else depth_2d,
                        boundary='none',
                        dtype=np.complex64,
                        psize=window,
                        threshold=threshold,
                        device=device,
                    )

                    filtered_vars[var_name] = xr.DataArray(
                        filtered_dask,
                        dims=var_data.dims,
                        coords=var_data.coords
                    )
                else:
                    filtered_vars[var_name] = var_data

            # Create a new dataset with the filtered variables
            result[k] = xr.Dataset(
                filtered_vars,
                coords=ds.coords,
                attrs=ds.attrs
            )

        return type(self)(result)


def _subtract_date_from_pair(first, second):
    """Subtract per-date atmospheric screens from per-pair data.

    Uses BatchComplex.pairs() to select ref/rep screens,
    then: result = data * conj(screen_ref) * screen_rep
    """
    import numpy as np

    key0 = list(first.keys())[0]
    ref_dates = first[key0].coords['ref'].values
    rep_dates = first[key0].coords['rep'].values
    pairs = np.column_stack([ref_dates, rep_dates])

    screen_ref, screen_rep = second.pairs(pairs)
    # Rechunk screens to match first's chunks — isel produces fragmented
    # dim0 chunks (63,63,...,31) while first has merged (1165,) chunks.
    # Without this, the multiply triggers an implicit rechunk of first
    # that re-reads the full upstream graph 19× per spatial tile.
    ref_var = next(v for v in first[key0].data_vars if first[key0][v].ndim >= 3)
    ref_chunks = first[key0][ref_var].data.chunks
    for batch in (screen_ref, screen_rep):
        for key in batch:
            ds = batch[key]
            rechunked = {}
            for v in ds.data_vars:
                da_xr = ds[v]
                if da_xr.ndim >= 3 and hasattr(da_xr.data, 'chunks') and da_xr.data.chunks != ref_chunks:
                    rechunked[v] = da_xr.chunk(dict(zip(da_xr.dims, ref_chunks)))
            if rechunked:
                batch[key] = ds.assign(rechunked)
    return first * screen_ref.conj() * screen_rep


class Batches(tuple):
    """
    A tuple-like container for multiple Batch objects that allows chained operations.

    Enables operations like:
        mintf, mcorr = stack.phasediff(...).downsample(20).compute()
        mintf, mcorr = stack.phasediff(...).downsample(20).snapshot('mintf_corr')
        mintf, mcorr = Batches.open('mintf_corr')

    Instead of:
        mintf, mcorr = stack.phasediff(...)
        mintf, mcorr = stack.compute(mintf.downsample(20), mcorr.downsample(20))
    """

    def __new__(cls, batches=()):
        return super().__new__(cls, batches)

    @staticmethod
    def _preserve_nonspatial(source, target):
        """Copy non-spatial variables (e.g. BPR) from source to target batch."""
        import dask.array as da
        for key in source:
            src_ds = source[key]
            tgt_ds = target[key]
            extra = {}
            for v in src_ds.data_vars:
                if v not in tgt_ds.data_vars:
                    var = src_ds[v]
                    if not isinstance(var.data, da.Array):
                        var = var.chunk()
                    extra[v] = var
            if extra:
                target[key] = tgt_ds.assign(extra)
        return target

    def phase(self) -> 'BatchComplex | BatchWrap | Batch | None':
        """Extract phase from Batches.

        Returns the first BatchComplex, or first BatchWrap,
        or first Batch found. Returns None if none found.
        """
        for b in self:
            if isinstance(b, BatchComplex):
                return b
        for b in self:
            if isinstance(b, BatchWrap):
                return b
        for b in self:
            if isinstance(b, Batch) and not isinstance(b, BatchUnit):
                return b
        return None

    def correlation(self) -> 'BatchUnit | None':
        """Extract correlation weights from Batches.

        Returns the first BatchUnit found, or None.
        """
        for b in self:
            if isinstance(b, BatchUnit):
                return b
        return None

    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                 caption: str | None = None,
                 debug: bool = False, **kwargs):
        """Save or open a Batches snapshot.

        When called on a Batches with data, saves all batches to Zarr store.
        When called on an empty Batches(), opens an existing store.

        Parameters
        ----------
        store : str
            Path to the Zarr store.
        storage_options : dict, optional
            Storage options for cloud stores.
        caption : str, optional
            Progress bar caption.
        debug : bool
            Print debug information.

        Returns
        -------
        tuple
            Tuple of Batch objects for unpacking.

        Examples
        --------
        >>> # Save
        >>> mintf, mcorr = stack.phasediff(...).downsample(20).snapshot('mintf_corr')
        >>> # Open
        >>> mintf, mcorr = Batches().snapshot('mintf_corr')
        """
        from . import utils_io

        if len(self) == 0:
            result = utils_io.snapshot(store=store, storage_options=storage_options,
                                       caption=caption or 'Opening...',
                                       debug=debug)
        else:
            result = utils_io.snapshot(*self, store=store, storage_options=storage_options,
                                       caption=caption or 'Snapshotting...',
                                       debug=debug, wrapper=Batches)

        if isinstance(result, Batches):
            return result
        return Batches((result,))  # fallback for stores without __wrapper__

    def archive(self, store: str, caption: str | None = None, compression: int = 6,
                debug: bool = False):
        """Save or open a Batches archive as a single ZIP file.

        Wrapper around snapshot() that uses ZipStore for single-file storage.
        Useful for downloading data from Google Colab or similar environments.

        Parameters
        ----------
        store : str
            Path to the ZIP file. Must end with '.zip'.
        caption : str, optional
            Progress bar caption.
        compression : int
            ZIP compression level 0-9 (0=no compression, 9=max). Default 6.
            Higher values produce smaller files but take longer.
        debug : bool
            Print debug information.

        Returns
        -------
        tuple
            Tuple of Batch objects for unpacking.

        Examples
        --------
        >>> # Save to zip
        >>> mintf, mcorr = stack.phasediff(...).downsample(20).archive('mintf_corr.zip')
        >>> # Save with max compression (for GitHub 100MB limit)
        >>> mintf, mcorr = stack.phasediff(...).archive('mintf_corr.zip', compression=9)
        >>> # Save to cloud storage (GCS, S3, etc.)
        >>> mintf, mcorr = stack.phasediff(...).archive('gs://bucket/mintf_corr.zip')
        >>> # Open from zip
        >>> mintf, mcorr = Batches().archive('mintf_corr.zip')
        """
        import zarr
        import zipfile
        import tempfile
        import os
        import fsspec

        if not store.endswith('.zip'):
            raise ValueError(f"Archive store must have '.zip' extension, got: {store}")

        # Check if cloud storage path
        is_cloud = '://' in store

        if len(self) == 0:
            # Open mode - check file exists first
            if is_cloud:
                fs, path = fsspec.core.url_to_fs(store)
                if not fs.exists(path):
                    raise FileNotFoundError(f"Archive not found: {store}")
            elif not os.path.exists(store):
                raise FileNotFoundError(f"Archive not found: {store}")
            # Use ZipStore directly for reading
            zip_store = zarr.storage.ZipStore(store, mode='r')
            result = self.snapshot(store=zip_store, caption=caption or 'Opening archive...', debug=debug)
            zip_store.close()
            return result
        else:
            # Save mode - write to temp directory, then zip
            # This avoids ZipStore's duplicate entry problem
            temp_dir = tempfile.mkdtemp()
            try:
                result = self.snapshot(store=temp_dir, caption=caption or 'Archiving...', debug=debug)
                # Create zip with specified compression level
                # Use fsspec for cloud storage support
                with fsspec.open(store, 'wb') as f:
                    with zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED, compresslevel=compression) as zf:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_dir)
                                zf.write(file_path, arcname)
            finally:
                import shutil
                shutil.rmtree(temp_dir)
            return result

    def downsample(self, *args, **kwargs):
        """Apply downsample to all batches."""
        return Batches([b.downsample(*args, **kwargs) for b in self])

    def chunk(self, *args, **kwargs):
        """Apply chunk to all batches."""
        return Batches([b.chunk(*args, **kwargs) for b in self])

    def chunk2d(self, *args, **kwargs):
        """Apply chunk2d to all batches."""
        return Batches([b.chunk2d(*args, **kwargs) for b in self])

    def chunk1d(self, *args, **kwargs):
        """Apply chunk1d to all batches."""
        return Batches([b.chunk1d(*args, **kwargs) for b in self])

    def where(self, cond, other=np.nan, **kwargs):
        """Apply where mask to all batches."""
        return Batches([b.where(cond, other, **kwargs) for b in self])

    def crop(self, *args, **kwargs):
        """Apply crop to all batches."""
        return Batches([b.crop(*args, **kwargs) for b in self])

    def sel(self, *args, **kwargs):
        """Apply sel to all batches."""
        return Batches([b.sel(*args, **kwargs) for b in self])

    def isel(self, *args, **kwargs):
        """Apply isel to all batches."""
        return Batches([b.isel(*args, **kwargs) for b in self])

    def filter(self, days=None, meters=None, date=None, pair=None, count=None,
               min_connections=2, cleanup=True):
        """Filter pairs in the baseline network.

        Selects pairs matching the given temporal/spatial criteria. By default,
        removes degraded dates (hanging or single-side connected).

        Parameters
        ----------
        days : int, optional
            Maximum temporal separation in days. If None, no temporal limit.
        meters : float, optional
            Maximum perpendicular baseline in meters. If None, no limit.
        date : str or list, optional
            Date(s) to exclude. Accepts a single date string or a list,
            any format parseable by ``pd.to_datetime``.
        pair : str or list, optional
            Pair(s) to exclude. Each pair is a string ``'YYYY-MM-DD YYYY-MM-DD'``.
            Accepts a single pair string or a list.
        count : int, optional
            Remove dates with fewer than this many connections.
        min_connections : int, optional
            Minimum pairs per date for cleanup. Default is 2.
        cleanup : bool, optional
            If True (default), iteratively remove hanging dates and dates
            connected only to predecessors or only to successors.
            Set to False to keep the raw network for testing.

        Returns
        -------
        Batches
            Filtered Batches with valid pairs only.

        Examples
        --------
        >>> stack.filter(days=100, meters=80).unwrap3d()
        >>> stack.filter(date='2024-12-30').unwrap3d()
        >>> intfcorr.filter(date=['2024-12-30', '2024-06-21'])
        >>> intfcorr.filter(pair='2024-06-21 2024-12-30')
        >>> intfcorr.filter(count=3)  # remove dates with < 3 connections
        >>> intfcorr.filter(days=100, meters=80, cleanup=False)  # raw network
        """
        import numpy as np
        import pandas as pd

        if days is None and meters is None and date is None and pair is None and count is None:
            return self

        # Get pair coordinates from the first batch element
        first_batch = self[0]
        first_key = next(iter(first_batch.keys()))
        ds = first_batch[first_key]
        ref = pd.DatetimeIndex(ds.coords['ref'].values).normalize()
        rep = pd.DatetimeIndex(ds.coords['rep'].values).normalize()
        bpr = ds.coords['BPR'].values
        n_pairs = len(ref)

        # Build mask of valid pairs
        mask = np.ones(n_pairs, dtype=bool)

        if days is not None:
            duration = (rep - ref).days
            mask &= duration <= days

        if meters is not None:
            mask &= np.abs(bpr) <= meters

        if date is not None:
            if isinstance(date, str):
                date = [date]
            exclude_dates = pd.to_datetime(date).normalize()
            mask &= ~ref.isin(exclude_dates) & ~rep.isin(exclude_dates)

        if pair is not None:
            if isinstance(pair, str):
                pair = [pair]
            exclude_pairs = set()
            for p in pair:
                parts = str(p).split()
                r, s = pd.Timestamp(parts[0]).normalize(), pd.Timestamp(parts[1]).normalize()
                exclude_pairs.add((r, s))
            for i in range(n_pairs):
                if (ref[i], rep[i]) in exclude_pairs:
                    mask[i] = False

        # Build DataFrame for pruning
        df = pd.DataFrame({'ref': ref[mask], 'rep': rep[mask],
                           'idx': np.where(mask)[0]})

        if len(df) > 0:
            from .Baseline import _cleanup_network
            min_conn = max(min_connections, count) if count is not None else min_connections
            if cleanup:
                df = _cleanup_network(df, min_connections=min_conn)
            elif count is not None:
                counts = pd.concat([df['ref'], df['rep']]).value_counts()
                low_dates = set(counts[counts < count].index)
                df = df[~df['ref'].isin(low_dates) & ~df['rep'].isin(low_dates)]

        if len(df) == 0:
            raise ValueError("No valid pairs remain after filtering. "
                             "Try increasing 'days' or 'meters'.")

        valid_idx = df['idx'].values
        return self.isel(pair=valid_idx)

    def coherent(self, threshold=0.5):
        """Mask low-coherence pixels using mean correlation.

        Computes mean correlation across pairs from the BatchUnit item
        and sets pixels with mean correlation below threshold to NaN
        in all batch items.

        Parameters
        ----------
        threshold : float
            Minimum mean correlation to keep. Default 0.5.

        Returns
        -------
        Batches
            Same structure with NaN where mean correlation < threshold.
        """
        corr = next((b for b in self if isinstance(b, BatchUnit)), None)
        if corr is None:
            raise ValueError('coherent() requires a BatchUnit (correlation) in Batches')
        results = []
        for b in self:
            out = {}
            for key in b:
                corr_ds = corr[key]
                corr_var = next(v for v in corr_ds.data_vars if 'y' in corr_ds[v].dims)
                corr_da = corr_ds[corr_var]
                mask = corr_da.mean('pair') >= threshold if 'pair' in corr_da.dims else corr_da >= threshold
                out[key] = b[key].where(mask)
            results.append(type(b)(out))
        return Batches(results)

    def angle(self):
        """Apply angle() to BatchComplex batches, return others unchanged.

        Returns
        -------
        Batches
            Batches with BatchComplex converted to BatchWrap (phase angles),
            other batch types unchanged.

        Examples
        --------
        >>> phase, corr = stack.phasediff2(pairs).angle()
        >>> # phase is now BatchWrap with angles, corr is unchanged BatchUnit
        """
        results = []
        for b in self:
            if isinstance(b, BatchComplex):
                results.append(b.angle())
            else:
                results.append(b)
        return Batches(results)

    def goldstein(self, window: int | list[int, int] = 32, threshold: float = 0.5, device: str = 'auto'):
        """Apply Goldstein filter to phase using correlation as weight.

        Expects Batches with [BatchComplex (phase), BatchUnit (correlation)].

        Parameters
        ----------
        window : int or list[int, int]
            Goldstein filter patch size, default 32.
        threshold : float
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batches
            Batches with Goldstein-filtered phase and unchanged correlation.

        Examples
        --------
        >>> phase, corr = stack.phasediff(pairs, wavelength=30).goldstein(32).angle()
        """
        if len(self) < 2:
            raise ValueError("goldstein() requires Batches with at least 2 elements: [phase, correlation]")

        phase, corr = self[0], self[1]

        if not isinstance(phase, BatchComplex):
            raise TypeError(f"First element must be BatchComplex, got {type(phase).__name__}")
        if not isinstance(corr, BatchUnit):
            raise TypeError(f"Second element must be BatchUnit, got {type(corr).__name__}")

        filtered_phase = phase.goldstein(corr, window, threshold=threshold, device=device)
        return Batches([filtered_phase, corr] + list(self[2:]))

    def interferogram(self,
                  weight: 'BatchUnit | None' = None,
                  phase: 'BatchComplex | None' = None,
                  wavelength: float | None = None,
                  gaussian_threshold: float = 0.5,
                  device: str = 'auto') -> 'Batches':
        """
        Compute phase difference from paired SLC data.

        Expects Batches from pairs() with [ref, rep] BatchComplex objects.

        Parameters
        ----------
        weight : BatchUnit or None
            Per-burst weights for Gaussian filtering and masking.
        phase : BatchComplex or None
            Optional phase to subtract (e.g., topographic phase).
        gaussian_threshold : float
            Threshold for Gaussian filter (default 0.5).
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batches
            Batches with [phase, correlation].

        Examples
        --------
        >>> ref, rep = stack.pairs(baseline.tolist())
        >>> phase, corr = ref.interferogram(rep, wavelength=30)
        >>> # Or chained:
        >>> phase, corr = stack.pairs(baseline.tolist()).interferogram(wavelength=30)
        """
        if len(self) != 2:
            raise ValueError("interferogram() requires Batches with exactly 2 elements: [ref, rep]")

        ref, rep = self[0], self[1]

        if not isinstance(ref, BatchComplex) or not isinstance(rep, BatchComplex):
            raise TypeError("Both elements must be BatchComplex")

        if weight is not None and not isinstance(weight, BatchUnit):
            raise TypeError(
                f'weight must be a BatchUnit, got {type(weight).__name__}. '
                'Use BatchUnit(stack.from_dataset(data)) to convert a single DataArray.'
            )

        intf = ref * rep.conj()
        if phase is not None:
            if isinstance(phase, BatchComplex):
                intf = intf * phase
            else:
                intf = intf * phase.iexp(-1)

        if wavelength is not None:
            intf_look = intf.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
            intensity_ref = ref.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
            intensity_rep = rep.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold, device=device)
            del ref, rep
            corr_look = (intf_look.abs() / (intensity_ref * intensity_rep).sqrt()).clip(0, 1)
            del intensity_ref, intensity_rep
        else:
            intf_look = intf
            corr_look = None
            del ref, rep
        del intf

        if weight is not None:
            intf_look = intf_look.where(weight.isfinite())
            corr_look = corr_look.where(weight.isfinite()) if corr_look else None

        if corr_look is None:
            return Batches([intf_look])
        return Batches([intf_look, corr_look])

    def interferogram2(self, *args, **kwargs):
        """
        Compute optimized interferogram using dual-polarization coherence optimization.

        This method requires the insardev_polsar extension package.
        """
        raise ImportError(
            "interferogram2() requires insardev_polsar extension"
        )

    def compute(self):
        """Compute all batches at once via dask.persist().

        Persists all bursts across all batches in a single scheduler
        submission. Data stays in worker memory. Preserves shared computation
        between dependent batches (e.g., phase and correlation). For
        memory-constrained sequential processing, use snapshot().

        Returns
        -------
        Batches
            Computed batches with data in memory.
        """
        import dask
        import numpy as np
        from insardev_toolkit.progressbar import progressbar

        # Get all burst keys (should be same across all batches)
        keys = list(self[0].keys())
        n_batches = len(self)

        # Save input chunk structure per batch per burst
        all_input_chunks = []  # list of {burst_key: {var_name: chunks_dict}}
        for batch in self:
            batch_chunks = {}
            for key, ds in batch.items():
                ic = {}
                for var_name in ds.data_vars:
                    arr = ds[var_name]
                    if hasattr(arr.data, 'chunks'):
                        ic[var_name] = dict(zip(arr.dims, arr.data.chunks))
                batch_chunks[key] = ic
            all_input_chunks.append(batch_chunks)

        # Persist all batches at once — single scheduler submission
        # progressbar extracts futures and blocks until completion
        all_dicts = [dict(batch) for batch in self]
        all_results = list(dask.persist(*all_dicts))
        progressbar(all_results, desc='Computing bursts'.ljust(25))

        # Finalize: materialize coordinates and rechunk to match input
        computed_batches = []
        for bi in range(n_batches):
            result = all_results[bi]
            computed = {}
            for key, ds in result.items():
                new_coords = {}
                for name, coord in ds.coords.items():
                    if hasattr(coord, 'data') and hasattr(coord.data, 'compute'):
                        new_coords[name] = (coord.dims, coord.compute().values)
                if new_coords:
                    ds = ds.assign_coords(new_coords)
                input_chunks = all_input_chunks[bi][key]
                rechunked_vars = {}
                for var_name in ds.data_vars:
                    arr = ds[var_name]
                    if var_name in input_chunks:
                        chunks = input_chunks[var_name]
                        if isinstance(arr.data, np.ndarray):
                            arr = arr.chunk(chunks)
                        elif hasattr(arr.data, 'chunks') and dict(zip(arr.dims, arr.data.chunks)) != chunks:
                            arr = arr.chunk(chunks)
                        rechunked_vars[var_name] = arr
                if rechunked_vars:
                    ds = ds.assign(rechunked_vars)
                computed[key] = ds
            computed_batches.append(type(self[bi])(computed))
        return Batches(computed_batches)

    def unwrap2d(self, conncomp=False, conncomp_size=1000, conncomp_gap=None,
                 conncomp_linksize=5, conncomp_linkcount=30, device='auto', debug=False, **kwargs):
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L¹ norm).

        Expects Batches with [BatchWrap or BatchComplex (phase), BatchUnit (weight, optional)].
        If the first element is BatchComplex, .angle() is called automatically.

        Parameters
        ----------
        conncomp : bool
            If False (default), link disconnected components using ILP.
            If True, keep components separate and return conncomp labels.
        conncomp_size : int
            Minimum pixels for a connected component. Default 1000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon.

        Returns
        -------
        Batch or tuple
            If conncomp=False: Batch of unwrapped phase.
            If conncomp=True: tuple of (Batch unwrapped, BatchUnit conncomp).

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline.tolist()).phasediff(wavelength=30).angle()
        >>> unwrapped = phase.unwrap2d()  # Without weights
        >>> unwrapped = Batches([phase, corr]).unwrap2d()  # With weights
        """
        if len(self) < 1:
            raise ValueError("unwrap2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Auto-convert complex phase to wrapped phase
        if isinstance(phase, BatchComplex):
            phase = phase.angle()

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap or BatchComplex, got {type(phase).__name__}")

        # Delegate to BatchWrap.unwrap2d
        return phase.unwrap2d(weight=weight, conncomp=conncomp, conncomp_size=conncomp_size,
                              conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                              conncomp_linkcount=conncomp_linkcount, device=device,
                              debug=debug, **kwargs)

    def unwrap2d_chunk(self, overlap=None, device='auto', debug=False, **kwargs):
        """
        Unwrap phase per spatial chunk with overlap using IRLS algorithm.

        Expects Batches with [BatchWrap or BatchComplex (phase), BatchUnit (weight, optional)].
        If the first element is BatchComplex, .angle() is called automatically.

        Unlike unwrap2d() which requires a single spatial chunk, this method
        unwraps each spatial chunk independently with overlap margins.

        Parameters
        ----------
        overlap : float, int, or tuple, optional
            Overlap size. Float = fraction of chunk size. Default 0.25.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.
        **kwargs
            Additional arguments: max_iter, tol, cg_max_iter, cg_tol, epsilon,
            conncomp_size.

        Returns
        -------
        Batches
            Batches with [unwrapped_phase, weight] preserving original types.

        Examples
        --------
        >>> phase, corr = stack.pairs(baseline).interferogram(wavelength=30).angle()
        >>> unwrapped, corr = phase.chunk2d('128MiB').unwrap2d_chunk()
        """
        if len(self) < 1:
            raise ValueError("unwrap2d_chunk() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        # Auto-convert complex phase to wrapped phase
        if isinstance(phase, BatchComplex):
            phase = phase.angle()

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap or BatchComplex, got {type(phase).__name__}")

        unwrapped = phase.unwrap2d_chunk(weight=weight, overlap=overlap,
                                          device=device, debug=debug, **kwargs)

        elements = [unwrapped] + list(self[1:])
        return Batches(elements)

    def trend2d(self, transform=None, degree=1, window=None, stride=1, device='auto', extrapolate=False, debug=False):
        """
        Compute 2D spatial trend and append it to Batches.

        Appends the per-pair trend as a new BatchComplex element, preserving
        original data unchanged. Use with subtract() to remove it.

        Parameters
        ----------
        transform : Batch
            Coordinate transform from stack.transform() containing 'azi', 'rng'.
        degree : int
            Polynomial degree (1=plane). Default 1.
        window : int, tuple, or None
            Window size in pixels. None = global fit.
        stride : int
            Subsample step for windowed fit. Default 1.
        device : str
            PyTorch device.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Original Batches with appended trend BatchComplex.

        Examples
        --------
        >>> # Append trend for later network-consistent detrending
        >>> intfcorr2d = intfcorr.trend2d(transform, window=(500,2000), stride=10)
        >>> # intfcorr2d = [intfs, corr, trend]  or  [intfs, trend]
        """
        if len(self) < 1:
            raise ValueError("trend2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, (Batch, BatchComplex)):
            raise TypeError(f"First element must be Batch or BatchComplex, got {type(phase).__name__}")

        if window is None:
            trend = phase.trend2d(transform, weight=weight, degree=degree,
                                  device=device, detrend=False,
                                  extrapolate=extrapolate, debug=debug)
        else:
            if degree != 1:
                raise ValueError("Windowed trend2d only supports degree=1.")
            if transform is not None:
                n_vars = len([v for v in transform[list(transform.keys())[0]].data_vars
                              if 'y' in transform[list(transform.keys())[0]][v].dims])
                if not (1 <= n_vars <= 3):
                    raise ValueError(f"Windowed trend2d requires 1-3 transform variables, got {n_vars}.")
            trend = phase.trend2d_window(transform, weight=weight,
                                         window=window, stride=stride,
                                         detrend=False, extrapolate=extrapolate,
                                         debug=debug)

        # Preserve non-spatial variables (e.g. BPR, ref, rep)
        trend = Batches._preserve_nonspatial(phase, trend)

        # Append trend to Batches
        elements = list(self) + [trend]
        return Batches(elements)

    def detrend2d(self, transform=None, degree=1, window=None, stride=1, device='auto', debug=False):
        """
        Detrend 2D polynomial trend and return Batches with detrended data.

        Two modes:
        - window=None (default): global polynomial fit across full spatial extent.
        - window=N or window=(Ny, Nx): local windowed fit with 4 half-overlapping
          grids averaged per pixel. Extent-independent. Window size in pixels.

        Parameters
        ----------
        transform : Batch
            Coordinate transform from stack.transform() containing 'azi', 'rng', 'ele'.
        degree : int
            Polynomial degree (1=plane, 2=quadratic). Default 1.
        window : int, tuple, or None
            Window size in pixels. None = global fit. int = square window.
            tuple (win_y, win_x) = rectangular window.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'.
        debug : bool
            Print diagnostic information.

        Returns
        -------
        Batches
            Batches with [detrended_phase, weight] preserving original types.

        Examples
        --------
        >>> # Global detrend (default)
        >>> intf = stack.pairs(bl).interferogram(wl=30).detrend2d(transform)
        >>> # Local windowed detrend (extent-independent)
        >>> intf = stack.pairs(bl).interferogram(wl=30).detrend2d(transform, window=250)
        """
        if len(self) < 1:
            raise ValueError("detrend2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, (Batch, BatchComplex)):
            raise TypeError(f"First element must be Batch or BatchComplex, got {type(phase).__name__}")

        if window is None:
            # Global polynomial fit (Pattern D: three-phase)
            detrended = phase.trend2d(transform, weight=weight, degree=degree,
                                      device=device, detrend=True, debug=debug)
        else:
            if degree != 1:
                raise ValueError("Windowed detrend2d only supports degree=1. "
                                 "Use window=None for higher-degree global fit.")
            if transform is not None:
                n_vars = len([v for v in transform[list(transform.keys())[0]].data_vars
                              if 'y' in transform[list(transform.keys())[0]][v].dims])
                if not (1 <= n_vars <= 3):
                    raise ValueError(f"Windowed detrend2d requires 1-3 transform variables "
                                     f"(e.g. ele, azi+rng, azi+rng+ele), got {n_vars}. "
                                     f"Use window=None for global fit with more variables.")
            # Local sliding window fit
            detrended = phase.trend2d_window(transform, weight=weight,
                                              window=window, stride=stride,
                                              detrend=True, debug=debug)

        # Preserve non-spatial variables (e.g. BPR) that may be dropped by arithmetic
        detrended = Batches._preserve_nonspatial(phase, detrended)

        # Rebuild Batches preserving all original elements except first
        elements = [detrended] + list(self[1:])
        return Batches(elements)

    def subtract(self):
        """
        Subtract the next same-type batch from the first batch.

        Finds the first batch element, then the next element of the same type,
        subtracts the second from the first, replaces the first with the result,
        and drops the second.

        Type-specific subtraction:
        - BatchComplex: first * conj(second) (phase subtraction on unit circle)
        - Batch: first - second (real subtraction)
        - BatchWrap: wrap(first - second) (wrapped phase subtraction)
        - BatchUnit: not supported (raises error)

        Returns
        -------
        Batches
            With first element replaced by subtracted result, second dropped.

        Examples
        --------
        >>> intfcorr = intfs.trend2d(transform, ...).subtract()
        """
        if len(self) < 2:
            raise ValueError("subtract() requires at least 2 elements")

        first_type = type(self[0])
        if isinstance(self[0], BatchUnit):
            raise TypeError("subtract() cannot be applied to BatchUnit")

        # Find next element of the same type
        second_idx = None
        for i in range(1, len(self)):
            if type(self[i]) is first_type:
                second_idx = i
                break
        if second_idx is None:
            raise ValueError(f"subtract() requires a second {first_type.__name__} element")

        first = self[0]
        second = self[second_idx]

        # Check if second is per-date and first is per-pair
        is_date_to_pair = False
        for key in first.keys():
            first_ds = first[key]
            second_ds = second[key]
            first_pol = [v for v in first_ds.data_vars if 'y' in first_ds[v].dims][0]
            second_pol = [v for v in second_ds.data_vars if 'y' in second_ds[v].dims][0]
            if 'pair' in first_ds[first_pol].dims and 'date' in second_ds[second_pol].dims:
                is_date_to_pair = True
            break

        if is_date_to_pair:
            result = _subtract_date_from_pair(first, second)
        elif isinstance(first, BatchComplex):
            result = first * second.conj()
        else:
            result = first - second

        result = Batches._preserve_nonspatial(first, result)

        elements = list(self)
        elements[0] = result
        elements.pop(second_idx)
        return Batches(elements)

    def displacement_los(self, transform):
        """
        Convert phase to line-of-sight displacement (meters).

        Applies Batch.displacement_los() to the first element.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch or Stack providing radar_wavelength.

        Returns
        -------
        Batches
            Batches with [displacement Batch] preserving other elements.

        Examples
        --------
        >>> disp = stack.unwrap3d().displacement_los(stack.transform())
        """
        data = self[0]
        result = data.displacement_los(transform)
        elements = [result] + list(self[1:])
        return Batches(elements)

    def regression1d_baseline(self, *args, **kwargs):
        raise NotImplementedError("Batches.regression1d_baseline() is removed. Use Batches.detrend1d() or Batch.trend1d() instead.")

    def threshold(self, threshold=np.pi/2):
        """
        Filter pixels by circular standard deviation (cstd) of pair phases.

        Pixels with cstd >= threshold are set to NaN. Uses correlation
        weights from the second element if available.

        Parameters
        ----------
        threshold : float
            Maximum cstd in radians. Default π/2.

        Returns
        -------
        Batches
            Batches with filtered phase, preserving other elements.
        """
        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        if not isinstance(phase, BatchComplex):
            raise TypeError(f"threshold() requires BatchComplex, got {type(phase).__name__}")

        filtered = phase.threshold(weight=weight, threshold=threshold)
        elements = [filtered] + list(self[1:])
        return Batches(elements)

    def velocity(self, **kwargs):
        """
        Fast rate estimate from the first element, with the weight from the
        second when it carries one.

        Returns
        -------
        Batches
            Batches[velocity, rmse].
        """
        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None
        if isinstance(phase, BatchComplex):
            # Not a rename away: the per-date complex fit returns a MODEL
            # (velocity, height, seasonal, coherence, rmse), not the
            # Batches[velocity, rmse] this method promises, so dispatching here
            # would change the return type on the basis of the input type.
            raise TypeError(
                'velocity() on a complex per-date stack is now fit1d(), which '
                'returns the full per-pixel model rather than a rate alone. '
                'Call .fit1d() and read model.velocity, or predict(model) to '
                'rebuild the phase.')
        return phase.velocity(**kwargs) if weight is None \
            else phase.velocity(weight=weight, **kwargs)

    def fit1d(self, **kwargs):
        """
        Per-pixel model from the first element, dispatching on its type.

        Returns
        -------
        Batch
            The model: velocity, height, seasonal, coherence, rmse. Identical
            in name and unit to fit3d()'s, so predict() consumes either.
        """
        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None
        # BatchComplex.fit1d() normalises every sample to a unit phasor and
        # rejects a weight outright, so only the unwrapped fit is offered one
        if weight is None or isinstance(phase, BatchComplex):
            return phase.fit1d(**kwargs)
        return phase.fit1d(weight=weight, **kwargs)

    def rmse(self, solution):
        """RMSE of phase vs solution, using correlation weight if present.

        Extracts phase from self[0] and optional weight from self[1] (BatchUnit).
        Weight is automatically passed to the RMSE calculation and reduced
        to (y, x) via mean over the temporal dimension.

        Parameters
        ----------
        solution : Batch
            Velocity (y, x), pair-based, or date-based solution.

        Returns
        -------
        Batches
            [RMSE Batch (y, x), mean weight BatchUnit (y, x)] when weight present,
            [RMSE Batch (y, x)] otherwise.
        """
        if len(self) < 1:
            raise ValueError("rmse() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        weight = self[1] if len(self) >= 2 and isinstance(self[1], BatchUnit) else None

        rmse_result = phase.rmse(solution, weight=weight)

        if weight is not None:
            # Reduce weight to (y, x) — detect temporal dimension
            w_sample_ds = next(iter(weight.values()))
            w_spatial = [v for v in w_sample_ds.data_vars if 'y' in w_sample_ds[v].dims]
            tdim = next((d for d in ('pair', 'date')
                         if w_spatial and d in w_sample_ds[w_spatial[0]].dims), None)
            reduced_weight = weight.mean(tdim) if tdim else weight
            elements = [rmse_result, reduced_weight]
        else:
            elements = [rmse_result]

        return Batches(elements)

    def regression1d_pairs(self, *args, **kwargs):
        raise NotImplementedError("Batches.regression1d_pairs() is removed. Use fit3d() instead.")

    def trend1d_pairs(self, *args, **kwargs):
        raise NotImplementedError("Batches.trend1d_pairs() is removed. Use fit3d() instead.")

    def stl(self, freq='W', periods=52, robust=False):
        """
        Perform Seasonal-Trend decomposition using LOESS (STL).

        Expects Batches with [Batch (time series data)]. No weight parameter needed.

        Parameters
        ----------
        freq : str
            Frequency string for resampling (default 'W' for weekly).
        periods : int
            Number of periods for seasonal decomposition (default 52).
        robust : bool
            Whether to use robust fitting. Default False.

        Returns
        -------
        Batch
            Batch containing 'trend', 'seasonal', and 'resid' variables.

        Examples
        --------
        >>> stl_result = Batches([displacement]).stl(freq='W', periods=52)
        """
        if len(self) < 1:
            raise ValueError("stl() requires Batches with at least 1 element: [data]")

        data = self[0]

        # Delegate to Batch.stl
        return data.stl(freq=freq, periods=periods, robust=robust)

    def __getattr__(self, name):
        """Proxy unknown attributes to all batches if they're callable."""
        if name.startswith('_'):
            raise AttributeError(f"Batches has no attribute '{name}'")

        # Check if all batches have this attribute and it's callable
        attrs = [getattr(b, name, None) for b in self]
        if all(callable(a) for a in attrs if a is not None):
            def method(*args, **kwargs):
                results = [getattr(b, name)(*args, **kwargs) for b in self]
                # If results are Batch-like, wrap in Batches
                if results and hasattr(results[0], 'keys') and callable(results[0].keys):
                    return Batches(results)
                return tuple(results)
            return method

        raise AttributeError(f"Batches has no attribute '{name}'")
