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

class Stack_unwrap(Stack_multilooking):

    @staticmethod
    def _get_2d_edges(shape):
        """
        Construct edges for a 2D grid graph with 4-connectivity.

        Parameters
        ----------
        shape : tuple
            Shape of the 2D grid (height, width).

        Returns
        -------
        np.ndarray
            Array of shape (n_edges, 2) containing pairs of connected node indices.
        """
        nodes = np.arange(np.prod(shape)).reshape(shape)
        edges = np.concatenate(
            (
                np.stack([nodes[:, :-1].ravel(), nodes[:, 1:].ravel()], axis=1),
                np.stack([nodes[:-1, :].ravel(), nodes[1:, :].ravel()], axis=1),
            ),
            axis=0,
        )
        return edges

    @staticmethod
    def _branch_cut(phase, correlation=None, max_jump=1, norm=1, scale=2**16 - 1, max_iters=100):
        """
        Phase unwrapping using branch-cut algorithm with max-flow optimization.

        This algorithm minimizes the total weighted phase discontinuities by finding
        optimal branch cuts using Google OR-Tools max-flow solver.

        Parameters
        ----------
        phase : np.ndarray
            2D array of wrapped phase values in radians.
        correlation : np.ndarray, optional
            2D array of correlation values for weighting edges.
        max_jump : int, optional
            Maximum phase jump step. Default is 1.
        norm : float, optional
            P-norm for energy calculation. Default is 1.
        scale : float, optional
            Scaling factor for integer conversion. Default is 2**16 - 1.
        max_iters : int, optional
            Maximum iterations per jump step. Default is 100.

        Returns
        -------
        np.ndarray
            2D array of unwrapped phase values in radians.
        """
        from ortools.graph.python import max_flow

        shape = phase.shape
        # flatten and normalize phase to cycles (divide by 2π)
        # so that jumps of 1 correspond to one cycle
        phase_flat = (phase.ravel().astype(np.float64)) / (2 * np.pi)

        # handle NaN values - create mask for valid pixels
        valid_mask = ~np.isnan(phase_flat)
        if not np.any(valid_mask):
            return np.full(shape, np.nan, dtype=np.float32)

        # get edges for valid pixels only
        all_edges = Stack_unwrap._get_2d_edges(shape)

        # filter edges to include only those between valid pixels
        valid_edges_mask = valid_mask[all_edges[:, 0]] & valid_mask[all_edges[:, 1]]
        edges = all_edges[valid_edges_mask]

        if len(edges) == 0:
            return np.full(shape, np.nan, dtype=np.float32)

        # prepare correlation weights if provided
        corr_flat = None
        if correlation is not None:
            corr_flat = correlation.ravel().astype(np.float64)

        def scale_phase(x):
            return np.round(scale * x).astype(np.int64)

        def p_norm(x):
            return np.abs(x) ** norm

        def safe_log(x, epsilon=1e-10):
            return np.log(np.maximum(x, epsilon))

        nodes = phase_flat.size
        source = nodes
        sink = nodes + 1
        jumps = np.zeros(nodes + 2, dtype=np.int64)

        def energy_estimate(jumps, phase_flat, i, j):
            return np.sum(p_norm(jumps[j] - jumps[i] - phase_flat[i] + phase_flat[j]))

        energy_prev = energy_estimate(jumps, phase_flat, edges[:, 0], edges[:, 1])
        jump_step = max_jump

        while jump_step >= 1:
            iter_count = 0
            while iter_count < max_iters:
                max_flow_solver = max_flow.SimpleMaxFlow()

                i, j = edges[:, 0], edges[:, 1]
                phase_diff = phase_flat[i] - phase_flat[j]
                phase_res = (jumps[j] - jumps[i]) - phase_diff
                energy_res = p_norm(phase_res)
                energy_res_down = p_norm(phase_res - jump_step)
                energy_res_up = p_norm(phase_res + jump_step)
                weight = np.maximum(0, (energy_res_up - energy_res) + (energy_res_down - energy_res))

                if corr_flat is not None:
                    correlation_ij = (corr_flat[i] + corr_flat[j]) / 2
                    # avoid division by zero for low correlation
                    weight = weight / np.maximum(-safe_log(correlation_ij), 0.1)

                # add forward and backward edges
                max_flow_solver.add_arcs_with_capacity(
                    edges[:, 0].astype(np.int32),
                    edges[:, 1].astype(np.int32),
                    scale_phase(weight)
                )
                max_flow_solver.add_arcs_with_capacity(
                    edges[:, 1].astype(np.int32),
                    edges[:, 0].astype(np.int32),
                    np.zeros(len(weight), dtype=np.int64)
                )

                # compute source/sink weights
                weight_source = np.zeros(nodes)
                weight_sink = np.zeros(nodes)

                diff_up_down = energy_res_up - energy_res
                diff_down_up = energy_res - energy_res_up
                positive_diff_up_down = np.maximum(0, diff_up_down)
                negative_diff_up_down = np.minimum(0, diff_up_down)
                positive_diff_down_up = np.maximum(0, diff_down_up)
                negative_diff_down_up = np.minimum(0, diff_down_up)

                np.add.at(weight_source, i, positive_diff_up_down)
                np.add.at(weight_source, j, positive_diff_down_up)
                np.add.at(weight_sink, i, -negative_diff_up_down)
                np.add.at(weight_sink, j, -negative_diff_down_up)

                # add source and sink edges
                node_indices = np.arange(nodes, dtype=np.int32)
                source_nodes = np.full(nodes, source, dtype=np.int32)
                sink_nodes = np.full(nodes, sink, dtype=np.int32)

                max_flow_solver.add_arcs_with_capacity(source_nodes, node_indices, scale_phase(weight_source))
                max_flow_solver.add_arcs_with_capacity(node_indices, sink_nodes, scale_phase(weight_sink))

                status = max_flow_solver.solve(source, sink)
                if status != max_flow.SimpleMaxFlow.OPTIMAL:
                    break

                source_nodes_cut = max_flow_solver.get_source_side_min_cut()
                jumps[source_nodes_cut] += jump_step

                energy = energy_estimate(jumps, phase_flat, edges[:, 0], edges[:, 1])

                if energy < energy_prev:
                    energy_prev = energy
                    iter_count += 1
                else:
                    jumps[source_nodes_cut] -= jump_step
                    break

            jump_step //= 2

        # compute unwrapped phase: original phase + jumps converted to radians
        # jumps are integer cycles, multiply by 2π to get radians
        unwrapped = phase.ravel() + jumps[:-2] * (2 * np.pi)
        unwrapped = unwrapped.reshape(shape).astype(np.float32)

        # restore NaN values
        unwrapped[np.isnan(phase)] = np.nan

        return unwrapped

    @staticmethod
    def _conncomp_2d(phase):
        """
        Compute connected components for a 2D phase array.

        Parameters
        ----------
        phase : np.ndarray
            2D array of phase values (NaN indicates invalid pixels).

        Returns
        -------
        np.ndarray
            2D array of connected component labels (0 for invalid pixels).
        """
        from scipy.ndimage import label

        valid_mask = ~np.isnan(phase)
        labeled_array, num_features = label(valid_mask)
        return labeled_array.astype(np.int32)

    @staticmethod
    def wrap(data_pairs):
        import xarray as xr
        import numpy as np
        import dask

        if isinstance(data_pairs, xr.DataArray):
            return xr.DataArray(dask.array.mod(data_pairs.data + np.pi, 2 * np.pi) - np.pi, data_pairs.coords)\
                .rename(data_pairs.name)
        return np.mod(data_pairs + np.pi, 2 * np.pi) - np.pi
# 
#     @staticmethod
#     @jit(nopython=True, nogil=True)
#     def unwrap_pairs(data   : np.ndarray,
#                      weight : np.ndarray = np.empty((0),   dtype=np.float32),
#                      matrix : np.ndarray = np.empty((0,0), dtype=np.float32)):
#         # import directive is not compatible to numba
#         #import numpy as np
#     
#         assert data.ndim   == 1
#         assert weight.ndim == 1 or weight.ndim == 0
#         assert matrix.ndim == 2
#         assert data.shape[0] == matrix.shape[0]
# 
#         if np.all(np.isnan(data)) or (weight.size != 0 and np.all(np.isnan(weight))):
#             return np.nan * np.zeros(data.size)
#         
#         # the data variable will be modified and returned as the function output
#         #data = data.copy()
#         nanmask = np.isnan(data)
#         if weight.size != 0:
#             nanmask = nanmask | np.isnan(weight)
#         if np.all(nanmask):
#             # no valid input data
#             return np.nan * np.zeros(data.size)
#         
#         # the buffer variable will be modified
#         buffer = data[~nanmask].copy()
#         if weight.size != 0:
#             weight = weight[~nanmask]
#         
#         # exclude matrix records for not valid data
#         matrix = matrix[~nanmask,:]
#         pair_sum = matrix.sum(axis=1)
#         
#         # pairs do not require unwrapping, allow numba to recognize list type
#         pairs_ok = [0][1:]
#     
#         # check all compound pairs vs single pairs: only detect all not wrapped
#         for ndate in np.unique(pair_sum)[1:]:
#             pair_idxs = np.where(pair_sum==ndate)[0]
#             if weight.size != 0:
#                 # get the sorted order for the specific weights corresponding to pair_idxs
#                 pair_idxs_order = np.argsort(weight[pair_idxs])[::-1]
#                 # Apply the sorted order to pair_idxs
#                 pair_idxs = pair_idxs[pair_idxs_order]
#             for pair_idx in pair_idxs:
#                 #print (pair_idx, matrix[pair_idx])
#                 matching_columns = matrix[pair_idx] == 1
#                 #print ('matching_columns', matching_columns)
#                 # works for dates = 2
#                 #matching_rows = ((matrix[:, matching_columns] == 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
#                 matching_rows = ((matrix[:, matching_columns] >= 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
#                 #print ('matching_rows', matching_rows)
#                 matching_matrix = matrix[:,matching_columns] * matching_rows[:,None]
#                 #row_indices = np.where(matching_rows)[0]
#                 #print ('row_indices', row_indices)
#                 #with np.printoptions(threshold=np.inf, linewidth=np.inf):
#                 #    print (matching_matrix.T)
#                 value = buffer[pair_idx]
#                 values = (matching_matrix * buffer[:,None])
#                 #print ('value', value, '?=', values.sum())    
#                 jump = int(np.round((values.sum() - value) / (2*np.pi)))
#                 if jump == 0:
#                     pairs_ok.append(pair_idx)
#                     pairs_ok.extend(np.where(matching_rows)[0])
#                     #print ()
#                     continue
#     
#                 #print ('jump', jump)
#                 if abs(value) > abs(value + 2*np.pi*jump):
#                     #print (f'JUMP {ndate}:', jump)
#                     #jump, phase.values[pair_idx] + 2*np.pi*jump
#                     #print ('value', value, '=>', value + 2*np.pi*jump, '=', values.sum())
#                     pass
#                 else:
#                     #print (f'JUMP singles:', jump)
#                     pass
#                 #print (values[matching_rows])
#                 # check for wrapping
#                 valid_values = values[matching_rows].ravel()
#                 #maxdiff = abs(np.diff(valid_values[valid_values!=0])).max()
#                 #print ('maxdiff', maxdiff, maxdiff >= np.pi)
#                 #print ()
#                 #break
#         #print ('pairs_ok', pairs_ok)
#         #print ('==================')
#     
#         # check all compound pairs vs single pairs: fix wrapped compound using not wrapped singles only
#         for ndate in np.unique(pair_sum)[1:]:
#             pair_idxs = np.where(pair_sum==ndate)[0]
#             if weight.size != 0:
#                 # get the sorted order for the specific weights corresponding to pair_idxs
#                 pair_idxs_order = np.argsort(weight[pair_idxs])[::-1]
#                 # Apply the sorted order to pair_idxs
#                 pair_idxs = pair_idxs[pair_idxs_order]
#             for pair_idx in pair_idxs:
#                 if pair_idx in pairs_ok:
#                     continue
#                 #print (pair_idx, matrix[pair_idx])
#                 matching_columns = matrix[pair_idx] == 1
#                 #print ('matching_columns', matching_columns)
#                 # works for dates = 2
#                 #matching_rows = ((matrix[:, matching_columns] == 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
#                 matching_rows = ((matrix[:, matching_columns] >= 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
#                 #print ('matching_rows', matching_rows)
#                 matching_matrix = matrix[:,matching_columns] * matching_rows[:,None]
#                 #row_indices = np.where(matching_rows)[0]
#                 #print ('row_indices', row_indices)
#                 #with np.printoptions(threshold=np.inf, linewidth=np.inf):
#                 #    print (matching_matrix.T)
#                 #print ('matching_matrix', matching_matrix.shape)
#     
#                 pairs_single_not_valid = [pair for pair in np.where(matching_rows)[0] if not pair in pairs_ok]
#                 if len(pairs_single_not_valid) > 0:
#                     # some of single-pairs requires unwrapping, miss the compound segment processing
#                     #print ('ERROR')
#                     #print ()
#                     continue
#     
#                 value = buffer[pair_idx]
#                 values = (matching_matrix * buffer[:,None])
#                 #print ('value', value, '?=', values.sum())    
#                 jump = int(np.round((values.sum() - value) / (2*np.pi)))
#                 #print (f'JUMP {ndate}:', jump)
#                 buffer[pair_idx] += 2*np.pi*jump
#                 pairs_ok.append(pair_idx)
#                 #print ()
#                 continue
#         #print ('==================')
#     
#         # check all compound pairs vs single pairs
#         for ndate in np.unique(pair_sum)[1:]:
#             pair_idxs = np.where(pair_sum==ndate)[0]
#             if weight.size != 0:
#                 # get the sorted order for the specific weights corresponding to pair_idxs
#                 pair_idxs_order = np.argsort(weight[pair_idxs])[::-1]
#                 # Apply the sorted order to pair_idxs
#                 pair_idxs = pair_idxs[pair_idxs_order]
#             for pair_idx in pair_idxs:
#                 if pair_idx in pairs_ok:
#                     continue
#                 #print (pair_idx, matrix[pair_idx])
#                 matching_columns = matrix[pair_idx] == 1
#                 #print ('matching_columns', matching_columns)
#                 # works for dates = 2
#                 #matching_rows = ((matrix[:, matching_columns] == 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
#                 matching_rows = ((matrix[:, matching_columns] >= 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
#                 #print ('matching_rows', matching_rows)
#                 matching_matrix = matrix[:,matching_columns] * matching_rows[:,None]
#                 #row_indices = np.where(matching_rows)[0]
#                 #print ('row_indices', row_indices)
#                 #with np.printoptions(threshold=np.inf, linewidth=np.inf):
#                 #    print (matching_matrix.T)
#                 value = buffer[pair_idx]
#                 values = (matching_matrix * buffer[:,None])
#                 #print ('value', value, '???=', values.sum())    
#                 jump = int(np.round((values.sum() - value) / (2*np.pi)))
#                 #print ('jump', jump)
#                 if jump != 0:
#                     #print (f'JUMP {ndate}:', jump)
#                     #jump, phase.values[pair_idx] + 2*np.pi*jump
#                     #print ('value', value, '=>', value + 2*np.pi*jump, '=', values.sum())
#                     pass
#                 #print (values[matching_rows])
#                 # unwrap
#                 buffer[pair_idx] += 2*np.pi*jump
#                 pairs_ok.append(pair_idx)
#                 pairs_ok.extend(np.where(matching_rows)[0])
#                 #print ()
#                 #break
#     
#         # validity mask
#         #mask = [idx in pairs_ok for idx in range(buffer.size)]
#         mask = np.isin(np.arange(buffer.size), pairs_ok)
#         # output is the same size as input
#         #out = np.nan * np.zeros(data.size)
#         out = np.full(data.size, np.nan, dtype=np.float32)
#         #out[~nanmask] = np.where(mask, buffer, np.nan)
#         out[~nanmask] = np.where(mask[~nanmask], buffer[~nanmask], np.nan)
#         return out

    @staticmethod
    @jit(nopython=True, nogil=True)
    def unwrap_pairs(data      : np.ndarray,
                     weight    : np.ndarray = np.empty((0),   dtype=np.float32),
                     matrix    : np.ndarray = np.empty((0,0), dtype=np.float32),
                     tolerance : np.float32 = np.pi/2) -> np.ndarray:
        # import directive is not compatible to numba
        #import numpy as np
    
        assert data.ndim   == 1
        assert weight.ndim <= 1
        assert matrix.ndim == 2
        assert data.shape[0] == matrix.shape[0]
        # asserts with error text are not compatible to numba
#         assert data.ndim == 1, f'ERROR: Data argument should have a single dimension, but it has {data.ndim}'
#         assert weight.ndim <= 1, f'ERROR: Weight argument should have zero or one dimension, but it has {weight.ndim}'
#         assert matrix.ndim == 2, f'ERROR: Matrix should be 2-dimensional, but it has shape {matrix.shape}'
#         assert data.shape[0] == matrix.shape[0], f'ERROR: Data and weight argument first dimension is not equal: {data.shape[0]} vs {matrix.shape[0]}'

        # for now, xr.apply_ufunc always call specified function with float64 arguments
        # this error should be resolved controlling the output datatype
        #assert data.dtype   == np.float32, f'ERROR: data argument should be float32 array but it is {data.dtype}'
        #assert weight.dtype == np.float32, f'ERROR: weight argument should be float32 array but it is {weight.dtype}'

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
                #print (pair_idx, matrix[pair_idx])
                matching_columns = matrix[pair_idx] == 1
                #print ('matching_columns', matching_columns)
                # works for dates = 2
                #matching_rows = ((matrix[:, matching_columns] == 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
                matching_rows = ((matrix[:, matching_columns] >= 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
                #print ('matching_rows', matching_rows)
                matching_matrix = matrix[:,matching_columns] * matching_rows[:,None]
                #row_indices = np.where(matching_rows)[0]
                #print ('row_indices', row_indices)
                #with np.printoptions(threshold=np.inf, linewidth=np.inf):
                #    print (matching_matrix.T)
                value = buffer[pair_idx]
                values = (matching_matrix * buffer[:,None])
                #print ('value', value, '?=', values.sum())    
                jump = int(np.round((values.sum() - value) / (2*np.pi)))
                is_tolerance = abs(value - values.sum()) < tolerance
                #print (f'1JUMP {ndate}:', jump, value, '?', values.sum(), 'tol', is_tolerance)
                if jump == 0 and is_tolerance:
                    pairs_ok.append(pair_idx)
                    pairs_ok.extend(np.where(matching_rows)[0])
    
        # check all compound pairs vs single pairs: fix wrapped compound using not wrapped singles only
        # use pairs_ok to compare and add to pairs_ok
        for ndate in np.unique(pair_sum)[1:]:
            pair_idxs = np.where(pair_sum==ndate)[0]
            if weight.size != 0:
                # get the sorted order for the specific weights corresponding to pair_idxs
                pair_idxs_order = np.argsort(weight[pair_idxs])[::-1]
                # Apply the sorted order to pair_idxs
                pair_idxs = pair_idxs[pair_idxs_order]
            for pair_idx in pair_idxs:
                if pair_idx in pairs_ok:
                    continue
                #print (pair_idx, matrix[pair_idx])
                matching_columns = matrix[pair_idx] == 1
                #print ('matching_columns', matching_columns)
                # works for dates = 2
                #matching_rows = ((matrix[:, matching_columns] == 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
                matching_rows = ((matrix[:, matching_columns] >= 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
                #print ('matching_rows', matching_rows)
                matching_matrix = matrix[:,matching_columns] * matching_rows[:,None]
                #row_indices = np.where(matching_rows)[0]
                #print ('row_indices', row_indices)
                #with np.printoptions(threshold=np.inf, linewidth=np.inf):
                #    print (matching_matrix.T)
                #print ('matching_matrix', matching_matrix.shape)
    
                pairs_single_not_valid = [pair for pair in np.where(matching_rows)[0] if not pair in pairs_ok]
                if len(pairs_single_not_valid) > 0:
                    # some of single-pairs requires unwrapping, miss the compound segment processing
                    #print ('ERROR')
                    #print ()
                    continue
    
                value = buffer[pair_idx]
                values = (matching_matrix * buffer[:,None])
                #print ('value', value, '?=', values.sum())    
                jump = int(np.round((values.sum() - value) / (2*np.pi)))
                #print (f'JUMP {ndate}:', jump, 'value', value, '?=', values.sum())
                buffer[pair_idx] += 2*np.pi*jump
                is_tolerance = abs(buffer[pair_idx] - values.sum()) < tolerance
                #print (f'2JUMP {ndate}:', jump, value, '=>', buffer[pair_idx], '?', values.sum(), 'tol', is_tolerance)
                if is_tolerance:
                    pairs_ok.append(pair_idx)
                #print ()
                continue
        #print ('==================')
    
        # check all compound pairs vs single pairs
        # complete pairs_ok always when possible
        for ndate in np.unique(pair_sum)[1:]:
            pair_idxs = np.where(pair_sum==ndate)[0]
            if weight.size != 0:
                # get the sorted order for the specific weights corresponding to pair_idxs
                pair_idxs_order = np.argsort(weight[pair_idxs])[::-1]
                # Apply the sorted order to pair_idxs
                pair_idxs = pair_idxs[pair_idxs_order]
            for pair_idx in pair_idxs:
                if pair_idx in pairs_ok:
                    continue
                #print (pair_idx, matrix[pair_idx])
                matching_columns = matrix[pair_idx] == 1
                #print ('matching_columns', matching_columns)
                # works for dates = 2
                #matching_rows = ((matrix[:, matching_columns] == 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
                matching_rows = ((matrix[:, matching_columns] >= 1).sum(axis=1) == 1)&((matrix[:, ~matching_columns] == 1).sum(axis=1) == 0)
                #print ('matching_rows', matching_rows)
                matching_matrix = matrix[:,matching_columns] * matching_rows[:,None]
                #row_indices = np.where(matching_rows)[0]
                #print ('row_indices', row_indices)
                #with np.printoptions(threshold=np.inf, linewidth=np.inf):
                #    print (matching_matrix.T)
                value = buffer[pair_idx]
                values = (matching_matrix * buffer[:,None])
                #print ('value', value, '???=', values.sum())    
                #jump = int(np.round((values.sum() - value) / (2*np.pi)))
                #print ('\t3jump', jump)
                # unwrap if needed (jump=0 means no unwrapping)
                buffer[pair_idx] += 2*np.pi*jump
                is_tolerance = abs(buffer[pair_idx] - values.sum()) < tolerance
                #print (f'3JUMP {ndate}:', jump, value, '=>', buffer[pair_idx], '?', values.sum(), 'tol', is_tolerance)
                if is_tolerance:
                    pairs_ok.append(pair_idx)
                    pairs_ok.extend(np.where(matching_rows)[0])
                #print ()
                #break
    
        # return original values when unwrapping is not possible at all
        if len(pairs_ok) == 0:
            # buffer datatype is the same as input data datatype
            return buffer.astype(np.float32)
        # return unwrapped values
        # validity mask
        mask = [idx in pairs_ok for idx in range(buffer.size)]
        #mask = np.isin(np.arange(buffer.size), pairs_ok)
        # output is the same size as input
        #out = np.nan * np.zeros(data.size)
        out = np.full(data.size, np.nan, dtype=np.float32)
        out[~nanmask] = np.where(mask, buffer, np.nan)
        #out[~nanmask] = np.where(mask[~nanmask], buffer[~nanmask], np.nan)
        return out

    def unwrap_matrix(self, pairs):
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
        #return self._get_pairs_matrix(pairs)
        # revert temporally for backward compatibility
        return (self._get_pairs_matrix(pairs)>=0).astype(int)

    def unwrap1d(self, data, weight=None, tolerance=np.pi/2):
        import xarray as xr
        import numpy as np

        pairs = self._get_pairs(data)
        matrix = self.unwrap_matrix(pairs)
    
        if not 'stack' in data.dims:
            chunks_z, chunks_y, chunks_x = data.chunks if data.chunks is not None else np.inf, np.inf, np.inf
            if np.max(chunks_y) > self.netcdf_chunksize or np.max(chunks_x) > self.netcdf_chunksize:
                print (f'Note: data chunk size ({np.max(chunks_y)}, {np.max(chunks_x)}) is too large for stack processing')
                # 1/4 NetCDF chunk is the smallest reasonable processing chunk 
                chunks_y = chunks_x = self.netcdf_chunksize//2
                print (f'Note: auto tune data chunk size to a half of NetCDF chunk: ({chunks_y}, {chunks_x})')
            #else:
            #    # use the existing data chunks size
            #    chunks_y = np.max(chunks_y)
            #    chunks_x = np.max(chunks_x)
            chunks = dict(pair=-1, y=chunks_y, x=chunks_x)
        else:
            chunks_z, chunks_stack = data.chunks if data.chunks is not None else np.inf, np.inf
            if np.max(chunks_stack) > self.chunksize1d:
                print (f'Note: data chunk size ({np.max(chunks_stack)} is too large for stack processing')
                # 1D chunk size can be defined straightforward
                chunks_stack = self.chunksize1d
                print (f'Note: auto tune data chunk size to 1D chunk: ({chunks_stack})')
            #else:
            #    # use the existing data chunks size
            #    chunks_stack = np.max(chunks_stack)
            chunks = dict(pair=-1, stack=chunks_stack)
        
        # xarray wrapper
        input_core_dims = [['pair']]
        args = [self.wrap(data).chunk(chunks)]
        if weight is not None:
            # sdd another 'pair' dimension for weight
            input_core_dims.append(['pair'])
            # add weight to the arguments
            args.append(weight.chunk(chunks))
        model = xr.apply_ufunc(
            self.unwrap_pairs,
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

    def _unwrap_2d(self, phase_da, weight_da=None, conncomp_flag=False):
        """
        Process a single 3D DataArray (pair, y, x) for unwrapping.

        Parameters
        ----------
        phase_da : xr.DataArray
            3D DataArray with dimensions (pair, y, x).
        weight_da : xr.DataArray, optional
            3D DataArray of correlation weights.
        conncomp_flag : bool
            Whether to compute connected components.

        Returns
        -------
        tuple
            (unwrapped DataArray, conncomp DataArray or None)
        """
        import xarray as xr

        stackvar = phase_da.dims[0]  # 'pair'

        # save original chunks for restoring after processing (None for numpy-backed data)
        original_chunks = phase_da.chunks

        # rechunk to single chunk per y,x for processing
        # this also converts numpy-backed data to dask-backed data
        chunk_single = {stackvar: 1, 'y': -1, 'x': -1}
        phase_da = phase_da.chunk(chunk_single)
        if weight_da is not None:
            weight_da = weight_da.chunk(chunk_single)

        def _unwrap_single(phase_2d, corr_2d=None):
            """Unwrap a single 2D phase array."""
            return Stack_unwrap._branch_cut(phase_2d, correlation=corr_2d)

        def _conncomp_single(phase_2d):
            """Compute connected components for a single 2D array."""
            return Stack_unwrap._conncomp_2d(phase_2d).astype(np.float32)

        # use xr.apply_ufunc for parallel dask processing
        unwrap_da = xr.apply_ufunc(
            _unwrap_single,
            *([phase_da, weight_da] if weight_da is not None else [phase_da]),
            input_core_dims=([['y', 'x'], ['y', 'x']]) if weight_da is not None else [['y', 'x']],
            output_core_dims=[['y', 'x']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[np.float32],
        )

        # restore original chunks
        unwrap_da = unwrap_da.chunk(original_chunks if original_chunks is not None else phase_da.chunks)

        if conncomp_flag:
            comp_da = xr.apply_ufunc(
                _conncomp_single,
                unwrap_da,
                input_core_dims=[['y', 'x']],
                output_core_dims=[['y', 'x']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[np.float32],
            )
            # restore original chunks
            comp_da = comp_da.chunk(original_chunks if original_chunks is not None else phase_da.chunks)
            return unwrap_da, comp_da

        return unwrap_da, None

    def unwrap(self, phase, weight=None, conncomp=False):
        """
        Unwrap phase using branch-cut algorithm with max-flow optimization.

        This method processes BatchWrap phase data (and optional BatchUnit correlation)
        using a graph-based branch-cut algorithm. Unlike SNAPHU, it handles NaN values
        naturally without requiring interpolation.

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting. If provided, edges with
            lower correlation receive higher costs in the optimization.
        conncomp : bool, optional
            If True, also return connected components. Default is False.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase.
            If conncomp is True: tuple of (Batch unwrapped phase, BatchUnit conncomp).

        Examples
        --------
        Unwrap phase without correlation weighting:
        >>> unwrapped = stack.unwrap(intfs)

        Unwrap phase with correlation weighting:
        >>> unwrapped = stack.unwrap(intfs, corr)

        Unwrap with connected components:
        >>> unwrapped, conncomp = stack.unwrap(intfs, corr, conncomp=True)
        """
        import xarray as xr
        from .Batch import Batch, BatchWrap, BatchUnit

        assert isinstance(phase, BatchWrap), 'ERROR: phase should be a BatchWrap object'
        assert weight is None or isinstance(weight, BatchUnit), 'ERROR: weight should be a BatchUnit object'

        # process each burst in the batch
        unwrap_result = {}
        conncomp_result = {}

        for key in phase.keys():
            phase_ds = phase[key]
            weight_ds = weight[key] if weight is not None and key in weight else None

            # get data variables (typically polarization like 'VV')
            data_vars = list(phase_ds.data_vars)

            unwrap_vars = {}
            comp_vars = {}

            for var in data_vars:
                phase_da = phase_ds[var]
                weight_da = weight_ds[var] if weight_ds is not None and var in weight_ds else None

                unwrap_da, comp_da = self._unwrap_2d(phase_da, weight_da, conncomp)

                unwrap_vars[var] = unwrap_da
                if conncomp and comp_da is not None:
                    comp_vars[var] = comp_da

            unwrap_result[key] = xr.Dataset(unwrap_vars, attrs=phase_ds.attrs)
            if conncomp:
                conncomp_result[key] = xr.Dataset(comp_vars, attrs=phase_ds.attrs)

        # use Batch (not BatchWrap) to avoid re-wrapping the unwrapped phase
        if conncomp:
            return Batch(unwrap_result), BatchUnit(conncomp_result)
        return Batch(unwrap_result)

    @staticmethod
    def conncomp_main(data, start=0):
        import xarray as xr
        import numpy as np
        from scipy.ndimage import label

        if isinstance(data, xr.Dataset):
            conncomp = data.conncomp
        else:
            # 2D array expected for landmask, etc.
            labeled_array, num_features = label(data)
            conncomp = xr.DataArray(labeled_array, coords=data.coords).where(data)

        # Function to find the mode (most frequent value)
        def find_mode(array):
            values, counts = np.unique(array[~np.isnan(array)&(array>=start)], return_counts=True)
            max_count_index = np.argmax(counts)
            return values[max_count_index] if counts.size > 0 else np.nan

        # Apply the function along the 'pair' dimension
        maincomps =  xr.apply_ufunc(find_mode, conncomp.chunk(dict(y=-1, x=-1)), input_core_dims=[['y', 'x']],
                                    vectorize=True, dask='parallelized', output_dtypes=[int])
        return data.where(conncomp==maincomps)

    def plot_conncomps(self, data, caption='Connected Components', cols=4, size=4, nbins=5, aspect=1.2, y=1.05,
                       vmin=0, vmax=10, cmap='tab10_r'):
        import matplotlib.pyplot as plt

        # multi-plots ineffective for linked lazy data
        fg = data.plot.imshow(
            col='pair',
            col_wrap=cols, size=size, aspect=aspect,
            cmap=cmap, vmin=0, vmax=10
        )
        #fg.set_axis_labels('Range', 'Azimuth')
        fg.set_ticks(max_xticks=nbins, max_yticks=nbins)
        fg.fig.suptitle(caption, y=y)
