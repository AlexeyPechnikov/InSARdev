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
# required for function decorators
from numba import jit
# import directive is not compatible to numba
import numpy as np


class Stack_unwrap1d(Stack_multilooking):
    """1D phase unwrapping along the temporal dimension."""

    @staticmethod
    def wrap(data_pairs):
        import xarray as xr
        import numpy as np
        import dask

        if isinstance(data_pairs, xr.DataArray):
            return xr.DataArray(dask.array.mod(data_pairs.data + np.pi, 2 * np.pi) - np.pi, data_pairs.coords)\
                .rename(data_pairs.name)
        return np.mod(data_pairs + np.pi, 2 * np.pi) - np.pi

    @staticmethod
    @jit(nopython=True, nogil=True)
    def _unwrap1d_pairs(data      : np.ndarray,
                        weight    : np.ndarray = np.empty((0),   dtype=np.float32),
                        matrix    : np.ndarray = np.empty((0,0), dtype=np.float32),
                        tolerance : np.float32 = np.pi/2) -> np.ndarray:
        # import directive is not compatible to numba
        #import numpy as np

        assert data.ndim   == 1
        assert weight.ndim <= 1
        assert matrix.ndim == 2
        assert data.shape[0] == matrix.shape[0]

        if np.all(np.isnan(data)) or (weight.size != 0 and np.all(np.isnan(weight))):
            return np.full(data.size, np.nan, dtype=np.float32)

        nanmask = np.isnan(data)
        if weight.size != 0:
            nanmask = nanmask | np.isnan(weight)
        if np.all(nanmask):
            # no valid input data
            return np.full(data.size, np.nan, dtype=np.float32)

        # the buffer variable will be modified
        # buffer datatype is the same as input data datatype
        buffer = data[~nanmask].copy()
        if weight.size != 0:
            weight = weight[~nanmask]

        # exclude matrix records for not valid data
        matrix = matrix[~nanmask,:]
        pair_sum = matrix.sum(axis=1)

        # processed pairs, allow numba to recognize list type
        pairs_ok = [0][1:]

        # check all compound pairs vs single pairs: only detect all not wrapped
        # fill initial pairs_ok
        for ndate in np.unique(pair_sum)[1:]:
            pair_idxs = np.where(pair_sum==ndate)[0]
            if weight.size != 0:
                # get the sorted order for the specific weights corresponding to pair_idxs
                pair_idxs_order = np.argsort(weight[pair_idxs])[::-1]
                # Apply the sorted order to pair_idxs
                pair_idxs = pair_idxs[pair_idxs_order]
            for pair_idx in pair_idxs:
                matching_columns = matrix[pair_idx] == 1
                matching_rows = ((matrix[:, matching_columns] >= 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
                matching_matrix = matrix[:,matching_columns] * matching_rows[:,None]
                value = buffer[pair_idx]
                values = (matching_matrix * buffer[:,None])
                jump = int(np.round((values.sum() - value) / (2*np.pi)))
                is_tolerance = abs(value - values.sum()) < tolerance
                if jump == 0 and is_tolerance:
                    pairs_ok.append(pair_idx)
                    pairs_ok.extend(np.where(matching_rows)[0])

        # check all compound pairs vs single pairs: fix wrapped compound using not wrapped singles only
        # use pairs_ok to compare and add to pairs_ok
        for ndate in np.unique(pair_sum)[1:]:
            pair_idxs = np.where(pair_sum==ndate)[0]
            if weight.size != 0:
                pair_idxs_order = np.argsort(weight[pair_idxs])[::-1]
                pair_idxs = pair_idxs[pair_idxs_order]
            for pair_idx in pair_idxs:
                if pair_idx in pairs_ok:
                    continue
                matching_columns = matrix[pair_idx] == 1
                matching_rows = ((matrix[:, matching_columns] >= 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
                matching_matrix = matrix[:,matching_columns] * matching_rows[:,None]

                pairs_single_not_valid = [pair for pair in np.where(matching_rows)[0] if not pair in pairs_ok]
                if len(pairs_single_not_valid) > 0:
                    continue

                value = buffer[pair_idx]
                values = (matching_matrix * buffer[:,None])
                jump = int(np.round((values.sum() - value) / (2*np.pi)))
                buffer[pair_idx] += 2*np.pi*jump
                is_tolerance = abs(buffer[pair_idx] - values.sum()) < tolerance
                if is_tolerance:
                    pairs_ok.append(pair_idx)
                continue

        # check all compound pairs vs single pairs
        # complete pairs_ok always when possible
        for ndate in np.unique(pair_sum)[1:]:
            pair_idxs = np.where(pair_sum==ndate)[0]
            if weight.size != 0:
                pair_idxs_order = np.argsort(weight[pair_idxs])[::-1]
                pair_idxs = pair_idxs[pair_idxs_order]
            for pair_idx in pair_idxs:
                if pair_idx in pairs_ok:
                    continue
                matching_columns = matrix[pair_idx] == 1
                matching_rows = ((matrix[:, matching_columns] >= 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
                matching_matrix = matrix[:,matching_columns] * matching_rows[:,None]
                value = buffer[pair_idx]
                values = (matching_matrix * buffer[:,None])
                # unwrap if needed (jump=0 means no unwrapping)
                buffer[pair_idx] += 2*np.pi*jump
                is_tolerance = abs(buffer[pair_idx] - values.sum()) < tolerance
                if is_tolerance:
                    pairs_ok.append(pair_idx)
                    pairs_ok.extend(np.where(matching_rows)[0])

        # return original values when unwrapping is not possible at all
        if len(pairs_ok) == 0:
            return buffer.astype(np.float32)
        # return unwrapped values
        # validity mask
        mask = [idx in pairs_ok for idx in range(buffer.size)]
        out = np.full(data.size, np.nan, dtype=np.float32)
        out[~nanmask] = np.where(mask, buffer, np.nan)
        return out

    def unwrap1d_matrix(self, pairs):
        """
        Create a matrix for use in the least squares computation based on interferogram date pairs.

        Parameters
        ----------
        pairs : pandas.DataFrame or xarray.DataArray or xarray.Dataset
            DataFrame or DataArray containing interferogram date pairs.

        Returns
        -------
        numpy.ndarray
            A matrix with one row for every interferogram and one column for every date.
            Each element in the matrix is a float, with 1 indicating the start date,
            -1 indicating the end date, 0 if the date is covered by the corresponding
            interferogram timeline, and NaN otherwise.
        """
        return (self._get_pairs_matrix(pairs)>=0).astype(int)

    def unwrap1d(self, data, weight=None, tolerance=np.pi/2):
        """
        Perform 1D phase unwrapping along the temporal dimension.

        Parameters
        ----------
        data : xarray.DataArray
            Phase data with 'pair' dimension.
        weight : xarray.DataArray, optional
            Weights for each pair.
        tolerance : float, optional
            Tolerance for phase consistency check. Default is π/2.

        Returns
        -------
        xarray.DataArray
            Unwrapped phase data.
        """
        import xarray as xr
        import numpy as np

        pairs = self._get_pairs(data)
        matrix = self.unwrap1d_matrix(pairs)

        if not 'stack' in data.dims:
            chunks_z, chunks_y, chunks_x = data.chunks if data.chunks is not None else np.inf, np.inf, np.inf
            if np.max(chunks_y) > self.netcdf_chunksize or np.max(chunks_x) > self.netcdf_chunksize:
                print (f'Note: data chunk size ({np.max(chunks_y)}, {np.max(chunks_x)}) is too large for stack processing')
                chunks_y = chunks_x = self.netcdf_chunksize//2
                print (f'Note: auto tune data chunk size to a half of NetCDF chunk: ({chunks_y}, {chunks_x})')
            chunks = dict(pair=-1, y=chunks_y, x=chunks_x)
        else:
            chunks_z, chunks_stack = data.chunks if data.chunks is not None else np.inf, np.inf
            if np.max(chunks_stack) > self.chunksize1d:
                print (f'Note: data chunk size ({np.max(chunks_stack)} is too large for stack processing')
                chunks_stack = self.chunksize1d
                print (f'Note: auto tune data chunk size to 1D chunk: ({chunks_stack})')
            chunks = dict(pair=-1, stack=chunks_stack)

        # xarray wrapper
        input_core_dims = [['pair']]
        args = [self.wrap(data).chunk(chunks)]
        if weight is not None:
            input_core_dims.append(['pair'])
            args.append(weight.chunk(chunks))
        model = xr.apply_ufunc(
            self._unwrap1d_pairs,
            *args,
            dask='parallelized',
            vectorize=True,
            input_core_dims=input_core_dims,
            output_core_dims=[['pair']],
            output_dtypes=[np.float32],
            kwargs={'matrix': matrix, 'tolerance': tolerance}
        ).transpose('pair',...)
        del args

        return model.rename('unwrap')
