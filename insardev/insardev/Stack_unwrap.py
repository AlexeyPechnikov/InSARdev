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
    def _get_compact_edges(valid_mask_2d):
        """
        Construct edges for valid pixels only, with compact node indexing.

        Creates a mapping from 2D grid positions to compact indices (0, 1, 2, ...)
        and returns edges between adjacent valid pixels using these compact indices.

        Parameters
        ----------
        valid_mask_2d : np.ndarray
            2D boolean array where True indicates valid pixels.

        Returns
        -------
        edges : np.ndarray
            Array of shape (n_edges, 2) with compact node indices.
        compact_to_orig : np.ndarray
            Array mapping compact index -> original flat index.
        orig_to_compact : np.ndarray
            Array mapping original flat index -> compact index (-1 if invalid).
        """
        height, width = valid_mask_2d.shape
        n_pixels = height * width

        # Create mapping from original to compact indices
        orig_to_compact = np.full(n_pixels, -1, dtype=np.int32)
        valid_flat = valid_mask_2d.ravel()
        compact_to_orig = np.where(valid_flat)[0].astype(np.int32)
        n_valid = len(compact_to_orig)
        orig_to_compact[compact_to_orig] = np.arange(n_valid, dtype=np.int32)

        # Build edges between adjacent valid pixels
        edges_list = []

        # Horizontal edges (left-right neighbors)
        for r in range(height):
            for c in range(width - 1):
                if valid_mask_2d[r, c] and valid_mask_2d[r, c + 1]:
                    orig_left = r * width + c
                    orig_right = orig_left + 1
                    edges_list.append((orig_to_compact[orig_left], orig_to_compact[orig_right]))

        # Vertical edges (top-bottom neighbors)
        for r in range(height - 1):
            for c in range(width):
                if valid_mask_2d[r, c] and valid_mask_2d[r + 1, c]:
                    orig_top = r * width + c
                    orig_bot = orig_top + width
                    edges_list.append((orig_to_compact[orig_top], orig_to_compact[orig_bot]))

        if edges_list:
            edges = np.array(edges_list, dtype=np.int32)
        else:
            edges = np.zeros((0, 2), dtype=np.int32)

        return edges, compact_to_orig, orig_to_compact

    @staticmethod
    def _find_connected_components(valid_mask_2d):
        """
        Find connected components in a 2D valid mask using 4-connectivity.

        Parameters
        ----------
        valid_mask_2d : np.ndarray
            2D boolean array where True indicates valid pixels.

        Returns
        -------
        list of np.ndarray
            List of boolean masks, one per connected component.
        """
        from collections import deque

        height, width = valid_mask_2d.shape
        visited = np.zeros_like(valid_mask_2d, dtype=bool)
        components = []

        for start_r in range(height):
            for start_c in range(width):
                if valid_mask_2d[start_r, start_c] and not visited[start_r, start_c]:
                    # BFS to find all connected pixels
                    component_mask = np.zeros_like(valid_mask_2d, dtype=bool)
                    queue = deque([(start_r, start_c)])
                    visited[start_r, start_c] = True
                    component_mask[start_r, start_c] = True

                    while queue:
                        r, c = queue.popleft()
                        # 4-connectivity neighbors
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                if valid_mask_2d[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    component_mask[nr, nc] = True
                                    queue.append((nr, nc))

                    components.append(component_mask)

        return components

    @staticmethod
    def _branch_cut(phase, correlation=None, max_jump=1, norm=1, scale=2**16 - 1, max_iters=100, debug=False):
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
        debug : bool, optional
            If True, print diagnostic information. Default is False.

        Returns
        -------
        np.ndarray
            2D array of unwrapped phase values in radians.
        """
        import time
        from ortools.graph.python import max_flow

        shape = phase.shape

        # handle NaN values - create mask for valid pixels
        valid_mask_2d = ~np.isnan(phase)
        if not np.any(valid_mask_2d):
            if debug:
                print(f'Maxflow: no valid pixels in {shape} grid')
            return np.full(shape, np.nan, dtype=np.float32)

        # Find connected components and process each separately
        components = Stack_unwrap._find_connected_components(valid_mask_2d)

        if debug:
            total_valid = np.sum(valid_mask_2d)
            comp_sizes = [np.sum(c) for c in components]
            print(f'Maxflow: {shape} grid, {total_valid} valid pixels, {len(components)} components: {comp_sizes}')

        if len(components) > 1:
            # Multiple components - process each separately using bounding boxes
            result = np.full(shape, np.nan, dtype=np.float32)
            for i, comp_mask in enumerate(components):
                # Find bounding box of this component
                rows = np.any(comp_mask, axis=1)
                cols = np.any(comp_mask, axis=0)
                r_min, r_max = np.where(rows)[0][[0, -1]]
                c_min, c_max = np.where(cols)[0][[0, -1]]

                # Extract subgrid (bounding box)
                sub_phase = phase[r_min:r_max+1, c_min:c_max+1].copy()
                sub_mask = comp_mask[r_min:r_max+1, c_min:c_max+1]
                sub_phase[~sub_mask] = np.nan  # Mask out pixels not in this component
                sub_corr = correlation[r_min:r_max+1, c_min:c_max+1].copy() if correlation is not None else None
                if sub_corr is not None:
                    sub_corr[~sub_mask] = np.nan

                comp_size = np.sum(sub_mask)
                if debug:
                    print(f'  Component {i+1}/{len(components)}: {comp_size} pixels, bbox {sub_phase.shape}', end='', flush=True)
                    t0 = time.time()

                # Unwrap this component's subgrid
                sub_result = Stack_unwrap._branch_cut_single(
                    sub_phase, sub_corr, max_jump, norm, scale, max_iters, debug
                )

                # Merge back into result
                result[r_min:r_max+1, c_min:c_max+1][sub_mask] = sub_result[sub_mask]
                if debug:
                    elapsed = time.time() - t0
                    status = 'OK' if not np.all(np.isnan(sub_result[sub_mask])) else 'FAILED'
                    print(f' -> {status} ({elapsed:.2f}s)')
            return result

        # Single component - process directly
        if debug:
            print(f'  Single component: {np.sum(valid_mask_2d)} pixels', end='', flush=True)
            t0 = time.time()
        result = Stack_unwrap._branch_cut_single(phase, correlation, max_jump, norm, scale, max_iters, debug)
        if debug:
            elapsed = time.time() - t0
            status = 'OK' if not np.all(np.isnan(result[valid_mask_2d])) else 'FAILED'
            print(f' -> {status} ({elapsed:.2f}s)')
        return result

    @staticmethod
    def _branch_cut_single(phase, correlation=None, max_jump=1, norm=1, scale=2**16 - 1, max_iters=100, debug=False):
        """Core branch-cut implementation for a single connected component."""
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
    def _ilp_unwrap_2d(phase, correlation=None, max_time=300.0, search_workers=1, debug=False):
        """
        Phase unwrapping using Integer Linear Programming (ILP) with OR-Tools CP-SAT.

        This algorithm minimizes the L1 norm of phase jumps on edges subject to
        cycle consistency constraints (phase differences around any closed loop = 0).

        The formulation is: minimize w^T |k|
        subject to: Ak = -A(x/2π)
        where k are integer jumps on edges, w are weights, x are wrapped phase differences,
        and A encodes cycle constraints.

        ILP provides the mathematically optimal (minimum L1 norm) solution, which can
        be valuable for high-precision applications or validation datasets.

        Size limitations: ILP scales as O(N²) where N is the number of pixels.
        Practical limits are approximately:
        - 100×100: ~5-10 seconds
        - 150×150: ~10-20 seconds
        - 200×200: ~30-60 seconds
        - >200×200: may timeout

        If the solver doesn't reach OPTIMAL status within max_time, it returns
        NaN values. Use unwrap_maxflow() for larger grids.

        Parameters
        ----------
        phase : np.ndarray
            2D array of wrapped phase values in radians.
        correlation : np.ndarray, optional
            2D array of correlation values for weighting edges.
        max_time : float, optional
            Maximum solver time in seconds. Default is 300 (5 minutes).
            For complex cases, values up to 86400 (24 hours) may be useful.
        search_workers : int, optional
            Number of parallel workers for CP-SAT solver. Default is 1 for
            compatibility with Dask parallel processing. Set higher when
            processing few interferograms with many CPU cores available.
        debug : bool, optional
            If True, print diagnostic information. Default is False.

        Returns
        -------
        np.ndarray
            2D array of unwrapped phase values in radians.
        """
        import time
        shape = phase.shape

        # handle NaN values - create mask for valid pixels
        valid_mask_2d = ~np.isnan(phase)
        if not np.any(valid_mask_2d):
            if debug:
                print(f'ILP: no valid pixels in {shape} grid')
            return np.full(shape, np.nan, dtype=np.float32)

        # Find connected components and process each separately
        components = Stack_unwrap._find_connected_components(valid_mask_2d)

        if debug:
            total_valid = np.sum(valid_mask_2d)
            comp_sizes = [np.sum(c) for c in components]
            print(f'ILP: {shape} grid, {total_valid} valid pixels, {len(components)} components: {comp_sizes}')

        if len(components) > 1:
            # Multiple components - process each separately using bounding boxes
            result = np.full(shape, np.nan, dtype=np.float32)
            for i, comp_mask in enumerate(components):
                # Find bounding box of this component
                rows = np.any(comp_mask, axis=1)
                cols = np.any(comp_mask, axis=0)
                r_min, r_max = np.where(rows)[0][[0, -1]]
                c_min, c_max = np.where(cols)[0][[0, -1]]

                # Extract subgrid (bounding box)
                sub_phase = phase[r_min:r_max+1, c_min:c_max+1].copy()
                sub_mask = comp_mask[r_min:r_max+1, c_min:c_max+1]
                sub_phase[~sub_mask] = np.nan  # Mask out pixels not in this component
                sub_corr = correlation[r_min:r_max+1, c_min:c_max+1].copy() if correlation is not None else None
                if sub_corr is not None:
                    sub_corr[~sub_mask] = np.nan

                comp_size = np.sum(sub_mask)
                if debug:
                    print(f'  Component {i+1}/{len(components)}: {comp_size} pixels, bbox {sub_phase.shape}', end='', flush=True)
                    t0 = time.time()

                # Unwrap this component's subgrid
                sub_result = Stack_unwrap._ilp_unwrap_2d_single(sub_phase, sub_corr, max_time, search_workers, debug)

                # Merge back into result
                result[r_min:r_max+1, c_min:c_max+1][sub_mask] = sub_result[sub_mask]
                if debug:
                    elapsed = time.time() - t0
                    status = 'OK' if not np.all(np.isnan(sub_result[sub_mask])) else 'FAILED'
                    print(f' -> {status} ({elapsed:.2f}s)')
            return result

        # Single component - process directly
        if debug:
            print(f'  Single component: {np.sum(valid_mask_2d)} pixels', end='', flush=True)
            t0 = time.time()
        result = Stack_unwrap._ilp_unwrap_2d_single(phase, correlation, max_time, search_workers, debug)
        if debug:
            elapsed = time.time() - t0
            status = 'OK' if not np.all(np.isnan(result[valid_mask_2d])) else 'FAILED'
            print(f' -> {status} ({elapsed:.2f}s)')
        return result

    @staticmethod
    def _ilp_unwrap_2d_single(phase, correlation=None, max_time=300.0, search_workers=1, debug=False):
        """Core ILP implementation for a single connected component."""
        from collections import deque

        shape = phase.shape
        height, width = shape

        # flatten and normalize phase to cycles (divide by 2π)
        phase_flat = (phase.ravel().astype(np.float64)) / (2 * np.pi)

        # handle NaN values
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

        n_edges = len(edges)
        n_nodes = phase_flat.size

        # compute wrapped phase differences on edges (in cycles)
        phase_diff = phase_flat[edges[:, 1]] - phase_flat[edges[:, 0]]
        # wrap to [-0.5, 0.5] cycles
        phase_diff_wrapped = phase_diff - np.round(phase_diff)

        # prepare correlation weights if provided (scaled to integers for CP-SAT)
        if correlation is not None:
            corr_flat = correlation.ravel().astype(np.float64)
            edge_corr = (corr_flat[edges[:, 0]] + corr_flat[edges[:, 1]]) / 2
            weights_float = 1.0 / np.maximum(edge_corr, 0.01)
        else:
            weights_float = np.ones(n_edges)
        # scale weights to integers (CP-SAT requires integer coefficients)
        weight_scale = 1000
        weights = (weights_float * weight_scale).astype(np.int64)

        # Build edge index lookup using vectorized operations
        # Create a mapping from (node_i, node_j) -> (edge_idx, direction)
        # For edges stored as (i, j), direction i->j is +1, j->i is -1
        edge_from = edges[:, 0]
        edge_to = edges[:, 1]

        # vectorized cycle building for 2D grid
        # each 2x2 block forms a cycle: tl -> tr -> br -> bl -> tl
        yi, xi = np.mgrid[0:height-1, 0:width-1]
        tl = (yi * width + xi).ravel()
        tr = tl + 1
        bl = tl + width
        br = bl + 1

        # filter cycles where all 4 corners are valid
        valid_cycles_mask = (valid_mask[tl] & valid_mask[tr] &
                            valid_mask[bl] & valid_mask[br])
        tl = tl[valid_cycles_mask]
        tr = tr[valid_cycles_mask]
        bl = bl[valid_cycles_mask]
        br = br[valid_cycles_mask]
        n_cycles = len(tl)

        if n_cycles == 0:
            return phase.astype(np.float32)

        # Build edge lookup dict once for all cycle edge queries
        # Maps (node_i, node_j) -> (edge_idx, direction)
        edge_dict = {}
        for idx in range(n_edges):
            i, j = int(edges[idx, 0]), int(edges[idx, 1])
            edge_dict[(i, j)] = (idx, 1)
            edge_dict[(j, i)] = (idx, -1)

        # For cycles, we need all 4 edges to exist. Build arrays directly.
        # Cycle edges: tl->tr (horiz), tr->br (vert), br->bl (horiz, reverse), bl->tl (vert, reverse)
        n_c = len(tl)
        e_tl_tr = np.zeros(n_c, dtype=np.int32)
        d_tl_tr = np.zeros(n_c, dtype=np.int32)
        e_tr_br = np.zeros(n_c, dtype=np.int32)
        d_tr_br = np.zeros(n_c, dtype=np.int32)
        e_br_bl = np.zeros(n_c, dtype=np.int32)
        d_br_bl = np.zeros(n_c, dtype=np.int32)
        e_bl_tl = np.zeros(n_c, dtype=np.int32)
        d_bl_tl = np.zeros(n_c, dtype=np.int32)
        valid_cycle = np.ones(n_c, dtype=bool)

        for k in range(n_c):
            tl_k, tr_k, bl_k, br_k = int(tl[k]), int(tr[k]), int(bl[k]), int(br[k])
            # Check all 4 edges exist
            if (tl_k, tr_k) in edge_dict and (tr_k, br_k) in edge_dict and \
               (br_k, bl_k) in edge_dict and (bl_k, tl_k) in edge_dict:
                e_tl_tr[k], d_tl_tr[k] = edge_dict[(tl_k, tr_k)]
                e_tr_br[k], d_tr_br[k] = edge_dict[(tr_k, br_k)]
                e_br_bl[k], d_br_bl[k] = edge_dict[(br_k, bl_k)]
                e_bl_tl[k], d_bl_tl[k] = edge_dict[(bl_k, tl_k)]
            else:
                valid_cycle[k] = False

        # Filter to only valid cycles (all 4 edges exist)
        if not np.all(valid_cycle):
            tl = tl[valid_cycle]
            tr = tr[valid_cycle]
            bl = bl[valid_cycle]
            br = br[valid_cycle]
            e_tl_tr = e_tl_tr[valid_cycle]
            d_tl_tr = d_tl_tr[valid_cycle]
            e_tr_br = e_tr_br[valid_cycle]
            d_tr_br = d_tr_br[valid_cycle]
            e_br_bl = e_br_bl[valid_cycle]
            d_br_bl = d_br_bl[valid_cycle]
            e_bl_tl = e_bl_tl[valid_cycle]
            d_bl_tl = d_bl_tl[valid_cycle]
            n_cycles = len(tl)

        if n_cycles == 0:
            # No valid cycles - just integrate wrapped differences
            k = np.zeros(n_edges, dtype=np.float64)
        else:
            # Compute cycle sums (RHS values) - residue detection
            # In phase unwrapping theory, the discrete curl around a 2x2 cell equals:
            #   0 cycles: no residue
            #  +1 cycle:  positive residue (2π phase wrap)
            #  -1 cycle:  negative residue (-2π phase wrap)
            # Deviations from exact integers are due to noise in phase measurements.
            # Standard practice: round to nearest integer (implicit ±0.5 cycle threshold).
            cycle_sums = (d_tl_tr * phase_diff_wrapped[e_tl_tr] +
                          d_tr_br * phase_diff_wrapped[e_tr_br] +
                          d_br_bl * phase_diff_wrapped[e_br_bl] +
                          d_bl_tl * phase_diff_wrapped[e_bl_tl])

            cycle_rhs = np.round(cycle_sums).astype(np.int32)

            # OPTIMIZATION: check if solution is trivial (all cycle sums are 0)
            # This is common for smooth phase gradients
            if np.all(cycle_rhs == 0):
                # k=0 is optimal, just integrate wrapped differences
                k = np.zeros(n_edges, dtype=np.float64)
            else:
                # Use CP-SAT solver (much faster than MIP for this problem)
                from ortools.sat.python import cp_model

                # Stack cycle edge data
                cycle_edges = np.stack([e_tl_tr, e_tr_br, e_br_bl, e_bl_tl], axis=1)  # (n_cycles, 4)
                cycle_dirs = np.stack([d_tl_tr, d_tr_br, d_br_bl, d_bl_tl], axis=1)   # (n_cycles, 4)

                # Build edge -> cycles mapping for efficient constraint expansion
                edge_to_cycles = [[] for _ in range(n_edges)]
                for i in range(n_cycles):
                    for j in range(4):
                        edge_to_cycles[int(cycle_edges[i, j])].append(i)

                # Find active edges: start with edges in non-zero cycles
                # Then iteratively expand to include all cycles that share edges with active cycles
                # This ensures path independence while keeping problem size manageable
                nonzero_mask = cycle_rhs != 0
                active_cycles = set(np.where(nonzero_mask)[0])
                active_edges = set()
                for i in active_cycles:
                    for j in range(4):
                        active_edges.add(int(cycle_edges[i, j]))

                # Expand: for each active edge, include all cycles that use it
                # This ensures that if k[e] != 0, all cycles using edge e have constraints
                prev_n_cycles = 0
                while len(active_cycles) > prev_n_cycles:
                    prev_n_cycles = len(active_cycles)
                    new_edges = set()
                    for e in active_edges:
                        for cycle_idx in edge_to_cycles[e]:
                            if cycle_idx not in active_cycles:
                                active_cycles.add(cycle_idx)
                                for j in range(4):
                                    new_edges.add(int(cycle_edges[cycle_idx, j]))
                    active_edges.update(new_edges)

                active_edges = sorted(active_edges)
                active_cycles = sorted(active_cycles)
                n_active_edges = len(active_edges)
                n_active_cycles = len(active_cycles)

                # Create mapping from global edge index to active index
                edge_to_active = {e: idx for idx, e in enumerate(active_edges)}

                model = cp_model.CpModel()

                # Maximum expected jump (cycles)
                max_jump = 50

                # Create variables for active edges only
                k_vars = [model.NewIntVar(-max_jump, max_jump, f'k_{e}') for e in range(n_active_edges)]

                # For L1 norm, we need |k[e]|. Use AddAbsEquality for efficiency.
                k_abs = [model.NewIntVar(0, max_jump, f'abs_{e}') for e in range(n_active_edges)]
                for e in range(n_active_edges):
                    model.AddAbsEquality(k_abs[e], k_vars[e])

                # Objective: minimize weighted sum of |k|
                active_weights = [int(weights[active_edges[e]]) for e in range(n_active_edges)]
                model.Minimize(sum(active_weights[e] * k_abs[e] for e in range(n_active_edges)))

                # Add constraints for all active cycles (not just non-zero)
                # This ensures path independence within the active region
                for i in active_cycles:
                    rhs = -int(cycle_rhs[i])
                    terms = []
                    for j in range(4):
                        edge_idx = int(cycle_edges[i, j])
                        direction = int(cycle_dirs[i, j])
                        active_idx = edge_to_active[edge_idx]
                        terms.append(direction * k_vars[active_idx])
                    model.Add(sum(terms) == rhs)

                # Solve
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = max_time
                solver.parameters.num_search_workers = search_workers

                status = solver.Solve(model)

                if status != cp_model.OPTIMAL:
                    import warnings
                    warnings.warn(f'CP-SAT solver returned status {solver.StatusName(status)}, returning NaN', RuntimeWarning)
                    return np.full(shape, np.nan, dtype=np.float32)

                # Extract solution - k=0 for inactive edges
                k = np.zeros(n_edges, dtype=np.float64)
                for e in range(n_active_edges):
                    k[active_edges[e]] = solver.Value(k_vars[e])

        # integrate unwrapped phase differences from a reference node using BFS
        # Handle disconnected regions by starting new BFS from unvisited valid pixels
        unwrapped_cycles = np.full(n_nodes, np.nan, dtype=np.float64)
        visited = np.zeros(n_nodes, dtype=bool)

        # build adjacency list using vectorized approach
        # adj[node] = list of (neighbor, edge_idx, direction)
        # direction indicates how to use phase_diff_wrapped:
        #   +1 means we go from edges[e,0] to edges[e,1], so use +phase_diff_wrapped
        #   -1 means we go from edges[e,1] to edges[e,0], so use -phase_diff_wrapped
        adj_neighbors = [[] for _ in range(n_nodes)]
        for e in range(n_edges):
            i, j = edges[e, 0], edges[e, 1]
            adj_neighbors[i].append((j, e, 1))   # i -> j: use +phase_diff (which is phase[j] - phase[i])
            adj_neighbors[j].append((i, e, -1))  # j -> i: use -phase_diff (which is phase[i] - phase[j])

        # Process all connected components
        valid_indices = np.where(valid_mask)[0]
        for start_node in valid_indices:
            if visited[start_node]:
                continue

            # Start new BFS from this unvisited valid node
            queue = deque([start_node])
            visited[start_node] = True
            unwrapped_cycles[start_node] = phase_flat[start_node]

            while queue:
                node = queue.popleft()
                for neighbor, edge_idx, direction in adj_neighbors[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        # phase_diff_wrapped[e] = phase_flat[edges[e,1]] - phase_flat[edges[e,0]]
                        # k[e] corrects for 2π jumps on this edge
                        # direction tells us which way we're traversing
                        unwrapped_diff = phase_diff_wrapped[edge_idx] + k[edge_idx]
                        unwrapped_cycles[neighbor] = unwrapped_cycles[node] + direction * unwrapped_diff
                        queue.append(neighbor)

        unwrapped = (unwrapped_cycles * (2 * np.pi)).reshape(shape).astype(np.float32)
        unwrapped[np.isnan(phase)] = np.nan

        return unwrapped

    @staticmethod
    def _minflow_unwrap_2d(phase, correlation=None, conncomp_size=100, debug=False):
        """
        Phase unwrapping using Minimum Cost Flow (Costantini algorithm).

        This algorithm formulates phase unwrapping as a minimum cost flow problem
        on the dual grid. Each pixel edge becomes an arc in the flow network,
        with flow representing integer cycle corrections (k values). Residues at
        dual nodes (2x2 cell centers) act as sources (+1) or sinks (-1).

        The minflow solver finds minimum cost flow satisfying all residue constraints,
        giving the globally optimal L1-norm solution for branch cut placement.

        Note: Minflow optimizes total branch cut length (L1 norm of k values), not
        phase reconstruction error. For low-noise data (≤5% noise), results are
        equivalent to maxflow. For higher noise, maxflow typically produces lower
        phase error. Minflow is faster than maxflow on large grids.

        Parameters
        ----------
        phase : np.ndarray
            2D array of wrapped phase values in radians.
        correlation : np.ndarray, optional
            2D array of correlation values for weighting. Lower correlation
            means higher cost for branch cuts through that region.
        conncomp_size : int, optional
            Minimum number of pixels for a component to be processed.
            Components smaller than this are left as NaN. Default is 100.
        debug : bool, optional
            If True, print diagnostic information. Default is False.

        Returns
        -------
        np.ndarray
            2D array of unwrapped phase values in radians.

        References
        ----------
        M. Costantini, "A novel phase unwrapping method based on network
        programming," IEEE Trans. Geosci. Remote Sens., vol. 36, no. 3,
        pp. 813-821, 1998.
        """
        import time
        shape = phase.shape

        # handle NaN values - create mask for valid pixels
        valid_mask_2d = ~np.isnan(phase)
        if not np.any(valid_mask_2d):
            if debug:
                print(f'Minflow: no valid pixels in {shape} grid')
            return np.full(shape, np.nan, dtype=np.float32)

        # Find connected components and process each separately
        components = Stack_unwrap._find_connected_components(valid_mask_2d)

        if debug:
            total_valid = np.sum(valid_mask_2d)
            comp_sizes = [np.sum(c) for c in components]
            # Sort components by size (largest first) for display
            sorted_sizes = sorted(comp_sizes, reverse=True)
            n_tiny = sum(1 for s in comp_sizes if s < 10)
            print(f'Minflow: {shape} grid, {total_valid} valid pixels, {len(components)} components')
            if len(components) <= 10:
                print(f'  Component sizes: {sorted_sizes}')
            else:
                print(f'  Largest 5: {sorted_sizes[:5]}, smallest 5: {sorted_sizes[-5:]}, tiny(<10px): {n_tiny}')

        if len(components) > 1:
            # Multiple components - process each separately using bounding boxes
            # Sort components by size (largest first) to process main component first
            comp_sizes = [np.sum(c) for c in components]
            sorted_indices = np.argsort(comp_sizes)[::-1]  # Largest first
            min_size = max(conncomp_size, 4)
            n_skipped = sum(1 for s in comp_sizes if s < min_size)

            if debug and n_skipped > 0:
                print(f'  Skipping {n_skipped} components with < {min_size} pixels')

            result = np.full(shape, np.nan, dtype=np.float32)
            n_to_process = len(components) - n_skipped
            processed = 0
            for rank, i in enumerate(sorted_indices):
                comp_mask = components[i]
                comp_size = comp_sizes[i]

                # Skip small components (must be at least conncomp_size, and at least 4 for valid 2x2 cell)
                if comp_size < min_size:
                    continue

                processed += 1

                # Find bounding box of this component
                rows = np.any(comp_mask, axis=1)
                cols = np.any(comp_mask, axis=0)
                r_min, r_max = np.where(rows)[0][[0, -1]]
                c_min, c_max = np.where(cols)[0][[0, -1]]

                # Extract subgrid (bounding box)
                sub_phase = phase[r_min:r_max+1, c_min:c_max+1].copy()
                sub_mask = comp_mask[r_min:r_max+1, c_min:c_max+1]
                sub_phase[~sub_mask] = np.nan  # Mask out pixels not in this component
                sub_corr = correlation[r_min:r_max+1, c_min:c_max+1].copy() if correlation is not None else None
                if sub_corr is not None:
                    sub_corr[~sub_mask] = np.nan

                if debug:
                    # Check for narrow regions in the component
                    col_counts = np.sum(sub_mask, axis=0)
                    row_counts = np.sum(sub_mask, axis=1)
                    narrow_cols = np.sum((col_counts > 0) & (col_counts <= 2))
                    narrow_rows = np.sum((row_counts > 0) & (row_counts <= 2))
                    print(f'  Component {processed}/{n_to_process}: {comp_size} pixels, bbox {sub_phase.shape}', end='')
                    if narrow_cols > 0 or narrow_rows > 0:
                        print(f', NARROW: {narrow_cols} cols, {narrow_rows} rows', end='')
                    print('', flush=True)
                    t0 = time.time()

                # Unwrap this component's subgrid
                sub_result = Stack_unwrap._minflow_unwrap_2d_single(sub_phase, sub_corr, debug)

                # Merge back into result
                result[r_min:r_max+1, c_min:c_max+1][sub_mask] = sub_result[sub_mask]
                if debug:
                    elapsed = time.time() - t0
                    status = 'OK' if not np.all(np.isnan(sub_result[sub_mask])) else 'FAILED'
                    print(f' -> {status} ({elapsed:.2f}s)')
            return result

        # Single component - use bounding box like multi-component case for consistency
        comp_mask = components[0]
        rows = np.any(comp_mask, axis=1)
        cols = np.any(comp_mask, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        # Extract subgrid (bounding box)
        sub_phase = phase[r_min:r_max+1, c_min:c_max+1].copy()
        sub_mask = comp_mask[r_min:r_max+1, c_min:c_max+1]
        sub_phase[~sub_mask] = np.nan
        sub_corr = correlation[r_min:r_max+1, c_min:c_max+1].copy() if correlation is not None else None
        if sub_corr is not None:
            sub_corr[~sub_mask] = np.nan

        if debug:
            valid_phase = sub_phase[sub_mask]
            print(f'  Single component: {np.sum(sub_mask)} pixels, bbox {sub_phase.shape}, input phase range: [{valid_phase.min():.4f}, {valid_phase.max():.4f}]', end='', flush=True)
            t0 = time.time()

        sub_result = Stack_unwrap._minflow_unwrap_2d_single(sub_phase, sub_corr, debug)

        # Create full result and merge back
        result = np.full(shape, np.nan, dtype=np.float32)
        result[r_min:r_max+1, c_min:c_max+1][sub_mask] = sub_result[sub_mask]

        if debug:
            elapsed = time.time() - t0
            valid_result = result[valid_mask_2d]
            n_zeros = np.sum(valid_result == 0)
            status = 'OK' if not np.all(np.isnan(valid_result)) else 'FAILED'
            print(f' -> {status} ({elapsed:.2f}s), output range: [{valid_result.min():.4f}, {valid_result.max():.4f}], zeros={n_zeros}')
        return result

    @staticmethod
    def _minflow_unwrap_2d_single(phase, correlation=None, debug=False):
        """Core minflow implementation for a single connected component."""
        from ortools.graph.python import min_cost_flow
        from collections import deque
        from scipy import ndimage

        shape = phase.shape
        height, width = shape

        # Handle NaN values - first check for interior holes
        original_nan_mask = np.isnan(phase)  # Remember original NaN positions
        valid_mask_2d = ~original_nan_mask
        if not np.any(valid_mask_2d):
            return np.full(shape, np.nan, dtype=np.float32)

        # Fill small interior holes to make domain more simply-connected
        # This prevents topological issues with multiply-connected domains
        hole_labels, n_holes = ndimage.label(~valid_mask_2d)
        if n_holes > 1:
            # Find the exterior (largest NaN region)
            hole_sizes = ndimage.sum(~valid_mask_2d, hole_labels, range(1, n_holes + 1))
            exterior_label = np.argmax(hole_sizes) + 1
            # Fill small interior holes (< 100 pixels) with nearest neighbor interpolation
            phase_filled = phase.copy()
            n_filled = 0
            for label_id in range(1, n_holes + 1):
                if label_id == exterior_label:
                    continue
                hole_mask = hole_labels == label_id
                hole_size = hole_sizes[label_id - 1]
                if hole_size < 100:  # Fill small holes
                    # Find boundary pixels around the hole
                    dilated = ndimage.binary_dilation(hole_mask)
                    boundary = dilated & valid_mask_2d
                    if np.any(boundary):
                        # Use mean of boundary pixels
                        boundary_mean = np.nanmean(phase[boundary])
                        phase_filled[hole_mask] = boundary_mean
                        n_filled += int(hole_size)
            if n_filled > 0 and debug:
                print(f'    DEBUG: filled {n_filled} pixels in small interior holes')
            phase = phase_filled
            valid_mask_2d = ~np.isnan(phase)

        # Flatten and normalize phase to cycles
        phase_flat = (phase.ravel().astype(np.float64)) / (2 * np.pi)

        # Handle NaN values
        valid_mask = ~np.isnan(phase_flat)
        if not np.any(valid_mask):
            return np.full(shape, np.nan, dtype=np.float32)

        # Get edges for valid pixels only
        all_edges = Stack_unwrap._get_2d_edges(shape)
        valid_edges_mask = valid_mask[all_edges[:, 0]] & valid_mask[all_edges[:, 1]]
        edges = all_edges[valid_edges_mask]

        if len(edges) == 0:
            return np.full(shape, np.nan, dtype=np.float32)

        n_edges = len(edges)
        n_nodes = phase_flat.size

        # Compute wrapped phase differences (in cycles)
        phase_diff = phase_flat[edges[:, 1]] - phase_flat[edges[:, 0]]
        phase_diff_wrapped = phase_diff - np.round(phase_diff)

        # Prepare correlation weights for edge costs
        if correlation is not None:
            corr_flat = correlation.ravel().astype(np.float64)
            edge_corr = (corr_flat[edges[:, 0]] + corr_flat[edges[:, 1]]) / 2
            edge_costs = (1000.0 / np.maximum(edge_corr, 0.01)).astype(np.int64)
        else:
            edge_costs = np.ones(n_edges, dtype=np.int64) * 1000

        # Build edge lookup: (pixel_i, pixel_j) -> (edge_idx, direction)
        edge_dict = {}
        for idx in range(n_edges):
            i, j = int(edges[idx, 0]), int(edges[idx, 1])
            edge_dict[(i, j)] = (idx, 1)
            edge_dict[(j, i)] = (idx, -1)

        # Dual grid dimensions
        dual_height = height - 1
        dual_width = width - 1

        if dual_height <= 0 or dual_width <= 0:
            # No dual nodes - trivial unwrapping
            k = np.zeros(n_edges, dtype=np.float64)
        else:
            # First pass: identify valid dual nodes (2x2 cells with all 4 corners valid)
            # and compute their residues
            valid_dual_nodes = []  # list of (r, c) for valid dual nodes
            dual_to_idx = {}  # (r, c) -> index in valid_dual_nodes
            residue_list = []

            for r in range(dual_height):
                for c in range(dual_width):
                    tl = r * width + c
                    tr = tl + 1
                    bl = tl + width
                    br = bl + 1

                    if not (valid_mask[tl] and valid_mask[tr] and
                            valid_mask[bl] and valid_mask[br]):
                        continue

                    if ((tl, tr) not in edge_dict or (tr, br) not in edge_dict or
                        (br, bl) not in edge_dict or (bl, tl) not in edge_dict):
                        continue

                    e1, d1 = edge_dict[(tl, tr)]
                    e2, d2 = edge_dict[(tr, br)]
                    e3, d3 = edge_dict[(br, bl)]
                    e4, d4 = edge_dict[(bl, tl)]

                    cycle_sum = (d1 * phase_diff_wrapped[e1] + d2 * phase_diff_wrapped[e2] +
                                d3 * phase_diff_wrapped[e3] + d4 * phase_diff_wrapped[e4])
                    residue = int(np.round(cycle_sum))

                    dual_to_idx[(r, c)] = len(valid_dual_nodes)
                    valid_dual_nodes.append((r, c))
                    residue_list.append(residue)

            n_valid_dual = len(valid_dual_nodes)
            residues = np.array(residue_list, dtype=np.int32) if residue_list else np.array([], dtype=np.int32)
            total_residue = int(np.sum(residues))
            n_nonzero = np.count_nonzero(residues)

            if n_nonzero == 0 or n_valid_dual == 0:
                k = np.zeros(n_edges, dtype=np.float64)
            else:
                # Build MCF network on dual grid using only valid dual nodes
                # Each arc represents a pixel edge; flow = k value
                smcf = min_cost_flow.SimpleMinCostFlow()

                # Node indices: 0..n_valid_dual-1 are valid interior dual nodes
                # n_valid_dual is the boundary node
                boundary_node = n_valid_dual

                arc_start = []
                arc_end = []
                arc_capacity = []
                arc_cost = []
                arc_to_edge = []
                arc_direction = []

                max_cap = max(100, abs(total_residue) + 10)

                # Helper to add bidirectional arcs
                def add_arc_pair(node1, node2, e_idx, cost, dir_sign):
                    arc_start.append(node1)
                    arc_end.append(node2)
                    arc_capacity.append(max_cap)
                    arc_cost.append(cost)
                    arc_to_edge.append(e_idx)
                    arc_direction.append(dir_sign)

                    arc_start.append(node2)
                    arc_end.append(node1)
                    arc_capacity.append(max_cap)
                    arc_cost.append(cost)
                    arc_to_edge.append(e_idx)
                    arc_direction.append(-dir_sign)

                # Horizontal arcs between adjacent valid dual nodes (cross vertical pixel edges)
                for r in range(dual_height):
                    for c in range(dual_width - 1):
                        if (r, c) not in dual_to_idx or (r, c + 1) not in dual_to_idx:
                            continue
                        dual_left = dual_to_idx[(r, c)]
                        dual_right = dual_to_idx[(r, c + 1)]
                        pixel_top = r * width + (c + 1)
                        pixel_bot = (r + 1) * width + (c + 1)

                        if (pixel_top, pixel_bot) in edge_dict:
                            e_idx, e_dir = edge_dict[(pixel_top, pixel_bot)]
                            # Negate direction: flow from left to right should DECREASE k
                            # to properly cancel residue at left cell
                            add_arc_pair(dual_left, dual_right, e_idx, int(edge_costs[e_idx]), -e_dir)

                # Vertical arcs between adjacent valid dual nodes (cross horizontal pixel edges)
                for r in range(dual_height - 1):
                    for c in range(dual_width):
                        if (r, c) not in dual_to_idx or (r + 1, c) not in dual_to_idx:
                            continue
                        dual_top = dual_to_idx[(r, c)]
                        dual_bot = dual_to_idx[(r + 1, c)]
                        pixel_left = (r + 1) * width + c
                        pixel_right = (r + 1) * width + c + 1

                        if (pixel_left, pixel_right) in edge_dict:
                            e_idx, e_dir = edge_dict[(pixel_left, pixel_right)]
                            add_arc_pair(dual_top, dual_bot, e_idx, int(edge_costs[e_idx]), e_dir)  # Removed negation

                # Boundary connections: connect each valid dual node to boundary
                # A dual node at (r, c) is on the boundary of the valid region if any
                # of its 4 neighbors is not a valid dual node
                # Track which pixel edges have been connected to boundary to avoid duplicates
                boundary_edges_connected = set()

                for r, c in valid_dual_nodes:
                    dual_idx = dual_to_idx[(r, c)]

                    # Check top boundary (r == 0 or (r-1, c) not valid)
                    if r == 0 or (r - 1, c) not in dual_to_idx:
                        pixel_left = r * width + c
                        pixel_right = r * width + c + 1
                        if (pixel_left, pixel_right) in edge_dict:
                            e_idx, e_dir = edge_dict[(pixel_left, pixel_right)]
                            if e_idx not in boundary_edges_connected:
                                add_arc_pair(boundary_node, dual_idx, e_idx, int(edge_costs[e_idx]), e_dir)
                                boundary_edges_connected.add(e_idx)

                    # Check bottom boundary (r == dual_height-1 or (r+1, c) not valid)
                    # Boundary is BELOW the cell - use -e_dir (reverse of interior vertical arc)
                    if r == dual_height - 1 or (r + 1, c) not in dual_to_idx:
                        pixel_left = (r + 1) * width + c
                        pixel_right = (r + 1) * width + c + 1
                        if (pixel_left, pixel_right) in edge_dict:
                            e_idx, e_dir = edge_dict[(pixel_left, pixel_right)]
                            if e_idx not in boundary_edges_connected:
                                add_arc_pair(boundary_node, dual_idx, e_idx, int(edge_costs[e_idx]), -e_dir)
                                boundary_edges_connected.add(e_idx)

                    # Check left boundary (c == 0 or (r, c-1) not valid)
                    # This is a vertical edge - use -e_dir like interior horizontal arcs
                    if c == 0 or (r, c - 1) not in dual_to_idx:
                        pixel_top = r * width + c
                        pixel_bot = (r + 1) * width + c
                        if (pixel_top, pixel_bot) in edge_dict:
                            e_idx, e_dir = edge_dict[(pixel_top, pixel_bot)]
                            if e_idx not in boundary_edges_connected:
                                add_arc_pair(boundary_node, dual_idx, e_idx, int(edge_costs[e_idx]), -e_dir)
                                boundary_edges_connected.add(e_idx)

                    # Check right boundary (c == dual_width-1 or (r, c+1) not valid)
                    # Boundary is to the RIGHT of cell - use +e_dir (opposite of left boundary)
                    if c == dual_width - 1 or (r, c + 1) not in dual_to_idx:
                        pixel_top = r * width + (c + 1)
                        pixel_bot = (r + 1) * width + (c + 1)
                        if (pixel_top, pixel_bot) in edge_dict:
                            e_idx, e_dir = edge_dict[(pixel_top, pixel_bot)]
                            if e_idx not in boundary_edges_connected:
                                add_arc_pair(boundary_node, dual_idx, e_idx, int(edge_costs[e_idx]), e_dir)
                                boundary_edges_connected.add(e_idx)

                if debug:
                    print(f'    DEBUG: {len(boundary_edges_connected)} unique boundary edges connected')
                    # Check for duplicate arcs
                    arc_pairs = {}
                    for i in range(len(arc_start)):
                        pair = (arc_start[i], arc_end[i])
                        if pair not in arc_pairs:
                            arc_pairs[pair] = []
                        arc_pairs[pair].append(i)
                    n_dup = sum(1 for v in arc_pairs.values() if len(v) > 1)
                    total_dup_arcs = sum(len(v) - 1 for v in arc_pairs.values() if len(v) > 1)
                    print(f'    DEBUG: {len(arc_start)} arcs, {n_dup} pairs with duplicates, {total_dup_arcs} duplicate arcs')
                    # Show sample duplicates
                    if n_dup > 0:
                        boundary_dups = 0
                        interior_dups = 0
                        dup_samples = []
                        for pair, indices in arc_pairs.items():
                            if len(indices) > 1:
                                if boundary_node in pair:
                                    boundary_dups += len(indices) - 1
                                    if len(dup_samples) < 3:
                                        # Get details of this duplicate
                                        other_node = pair[0] if pair[1] == boundary_node else pair[1]
                                        e_idxs = [arc_to_edge[i] for i in indices]
                                        dup_samples.append((pair, e_idxs, indices))
                                else:
                                    interior_dups += len(indices) - 1
                        print(f'    DEBUG: boundary duplicates: {boundary_dups}, interior duplicates: {interior_dups}')
                        if dup_samples:
                            for samp in dup_samples[:3]:
                                print(f'    DEBUG: sample dup: pair={samp[0]}, e_idxs={samp[1]}, arc_indices={samp[2]}')

                if len(arc_start) == 0:
                    k = np.zeros(n_edges, dtype=np.float64)
                else:
                    smcf.add_arcs_with_capacity_and_unit_cost(
                        np.array(arc_start, dtype=np.int32),
                        np.array(arc_end, dtype=np.int32),
                        np.array(arc_capacity, dtype=np.int64),
                        np.array(arc_cost, dtype=np.int64)
                    )

                    n_total_nodes = n_valid_dual + 1
                    supplies = np.zeros(n_total_nodes, dtype=np.int64)
                    supplies[:n_valid_dual] = residues
                    supplies[boundary_node] = -total_residue

                    smcf.set_nodes_supplies(np.arange(n_total_nodes, dtype=np.int32), supplies)

                    status = smcf.solve()

                    if status != smcf.OPTIMAL:
                        import warnings
                        warnings.warn(f'Minflow solver failed with status {status}, returning NaN', RuntimeWarning)
                        return np.full(shape, np.nan, dtype=np.float32)

                    # Extract k values from flow
                    k = np.zeros(n_edges, dtype=np.float64)
                    n_arcs = len(arc_start)
                    edges_with_multiple_flows = {}  # Track edges receiving flow from multiple arcs
                    for arc in range(n_arcs):
                        flow = smcf.flow(arc)
                        if flow != 0:
                            e_idx = arc_to_edge[arc]
                            direction = arc_direction[arc]
                            if e_idx not in edges_with_multiple_flows:
                                edges_with_multiple_flows[e_idx] = []
                            edges_with_multiple_flows[e_idx].append((arc, flow, direction))
                            k[e_idx] += flow * direction

                    if debug:
                        multi_flow = [(e, flows) for e, flows in edges_with_multiple_flows.items() if len(flows) > 1]
                        if multi_flow:
                            print(f'    DEBUG: {len(multi_flow)} edges receiving flow from multiple arcs')
                            for e, flows in multi_flow[:3]:
                                total = sum(f * d for a, f, d in flows)
                                print(f'    DEBUG: edge {e}: flows={[(f,d) for a,f,d in flows]}, total k={total}')

        # Convert edge-based k values to node-based unwrapped phase
        # The unwrapped phase difference across edge (i,j) is:
        #   unwrapped_diff[e] = phase_diff_wrapped[e] + k[e] = unwrapped[j] - unwrapped[i]
        # We integrate these gradients using BFS on a spanning tree.

        # Build adjacency list for BFS integration
        adj_neighbors = [[] for _ in range(n_nodes)]
        for e in range(n_edges):
            i, j = edges[e, 0], edges[e, 1]
            adj_neighbors[i].append((j, e, 1))
            adj_neighbors[j].append((i, e, -1))

        if debug:
            has_neighbors = sum(1 for node in range(n_nodes) if len(adj_neighbors[node]) > 0)
            print(f'\n    DEBUG: n_nodes={n_nodes}, n_edges={n_edges}, nodes_with_neighbors={has_neighbors}')
            print(f'    DEBUG: n_valid={np.sum(valid_mask)}, k_nonzero={np.count_nonzero(k)}, k_range=[{k.min():.3f}, {k.max():.3f}]')
            print(f'    DEBUG: phase_diff range: [{phase_diff.min():.4f}, {phase_diff.max():.4f}]')
            print(f'    DEBUG: phase_diff_wrapped range: [{phase_diff_wrapped.min():.4f}, {phase_diff_wrapped.max():.4f}]')
            print(f'    DEBUG: round(phase_diff) nonzero: {np.count_nonzero(np.round(phase_diff))}')

        # The unwrapped phase difference across each edge is:
        #   unwrapped_diff[e] = phase_diff_wrapped[e] + k[e]
        # This is the "gradient" of the unwrapped phase.
        #
        # We integrate these gradients using BFS to get the unwrapped phase directly.
        # Starting from a reference node with its original phase value,
        # we propagate: unwrapped[j] = unwrapped[i] + unwrapped_diff[e]
        unwrapped_diff = phase_diff_wrapped + k

        # BFS to integrate unwrapped phase differences
        unwrapped_cycles = np.full(n_nodes, np.nan, dtype=np.float64)
        valid_indices = np.where(valid_mask)[0]

        for start_node in valid_indices:
            if not np.isnan(unwrapped_cycles[start_node]):
                continue

            queue = deque([start_node])
            # Initialize with the original phase value at the starting node
            unwrapped_cycles[start_node] = phase_flat[start_node]

            while queue:
                node = queue.popleft()
                for neighbor, edge_idx, direction in adj_neighbors[node]:
                    if np.isnan(unwrapped_cycles[neighbor]):
                        # unwrapped_diff[e] = unwrapped[j] - unwrapped[i] for edge from i to j
                        # direction=+1 means we go from node(=i) to neighbor(=j)
                        # direction=-1 means we go from node(=j) to neighbor(=i)
                        unwrapped_cycles[neighbor] = unwrapped_cycles[node] + direction * unwrapped_diff[edge_idx]
                        queue.append(neighbor)

        if debug:
            valid_unwrapped = unwrapped_cycles[valid_mask]
            print(f'    DEBUG: unwrapped_cycles range: [{valid_unwrapped.min():.4f}, {valid_unwrapped.max():.4f}]')
            # Check consistency: for each edge, does unwrapped[j] - unwrapped[i] == unwrapped_diff[e]?
            errors = []
            for e in range(n_edges):
                i, j = edges[e, 0], edges[e, 1]
                if not np.isnan(unwrapped_cycles[i]) and not np.isnan(unwrapped_cycles[j]):
                    actual_diff = unwrapped_cycles[j] - unwrapped_cycles[i]
                    expected_diff = unwrapped_diff[e]
                    err = abs(actual_diff - expected_diff)
                    if err > 0.001:  # significant error
                        errors.append((e, err, actual_diff, expected_diff, k[e]))
            if errors:
                print(f'    DEBUG: {len(errors)} edges with integration errors > 0.001')
                # Show distribution of errors
                err_vals = [e[1] for e in errors]
                print(f'    DEBUG: error range: [{min(err_vals):.4f}, {max(err_vals):.4f}], mean: {np.mean(err_vals):.4f}')
                # Show sample errors with pixel positions
                for e, err, actual, expected, k_val in errors[:5]:
                    i, j = edges[e, 0], edges[e, 1]
                    ri, ci = i // width, i % width
                    rj, cj = j // width, j % width
                    print(f'    DEBUG: edge ({ri},{ci})-({rj},{cj}): actual={actual:.4f}, expected={expected:.4f}, k={k_val:.4f}')
                # Analyze error locations
                error_edge_indices = set(e[0] for e in errors)
                k_nonzero_edges = set(np.where(k != 0)[0])
                overlap = error_edge_indices & k_nonzero_edges
                print(f'    DEBUG: {len(overlap)} error edges have non-zero k (of {len(k_nonzero_edges)} total)')
                # Check if errors form a connected region
                error_rows = [edges[e[0], 0] // width for e in errors]
                error_cols = [edges[e[0], 0] % width for e in errors]
                print(f'    DEBUG: error edges span rows [{min(error_rows)}, {max(error_rows)}], cols [{min(error_cols)}, {max(error_cols)}]')
                # Check for holes in the mask
                valid_mask_2d = valid_mask.reshape(height, width)
                from scipy import ndimage
                # Invert mask and find connected NaN regions (holes)
                hole_labels, n_holes = ndimage.label(~valid_mask_2d)
                # The largest NaN region is the exterior (not a hole)
                if n_holes > 0:
                    hole_sizes = ndimage.sum(~valid_mask_2d, hole_labels, range(1, n_holes + 1))
                    n_interior_holes = np.sum(np.array(hole_sizes) < np.max(hole_sizes))
                    if n_interior_holes > 0:
                        print(f'    DEBUG: {n_interior_holes} interior NaN holes (multiply-connected domain)')

            # Verify that k values make gradient conservative by checking residues
            corrected_diff = phase_diff_wrapped + k
            residue_errors = 0
            curls = []
            k_correction_needed = np.zeros_like(k)
            for r in range(height - 1):
                for c in range(width - 1):
                    tl = r * width + c
                    tr = tl + 1
                    bl = tl + width
                    br = bl + 1
                    if not (valid_mask[tl] and valid_mask[tr] and valid_mask[bl] and valid_mask[br]):
                        continue
                    # Get edges for this 2x2 cell
                    if (tl, tr) not in edge_dict or (tr, br) not in edge_dict:
                        continue
                    if (br, bl) not in edge_dict or (bl, tl) not in edge_dict:
                        continue
                    e1, d1 = edge_dict[(tl, tr)]
                    e2, d2 = edge_dict[(tr, br)]
                    e3, d3 = edge_dict[(br, bl)]
                    e4, d4 = edge_dict[(bl, tl)]
                    # Compute curl of corrected gradient
                    curl = (d1 * corrected_diff[e1] + d2 * corrected_diff[e2] +
                           d3 * corrected_diff[e3] + d4 * corrected_diff[e4])
                    if abs(curl) > 0.001:
                        residue_errors += 1
                        curls.append(curl)
            print(f'    DEBUG: {residue_errors} cells with non-zero curl after k correction')
            if curls:
                print(f'    DEBUG: curl values: [{min(curls):.4f}, {max(curls):.4f}], unique: {sorted(set([round(c) for c in curls]))}')
                # Show where the curl errors are located
                curl_positions = []
                for r in range(height - 1):
                    for c in range(width - 1):
                        tl = r * width + c
                        tr = tl + 1
                        bl = tl + width
                        br = bl + 1
                        if not (valid_mask[tl] and valid_mask[tr] and valid_mask[bl] and valid_mask[br]):
                            continue
                        if (tl, tr) not in edge_dict or (tr, br) not in edge_dict:
                            continue
                        if (br, bl) not in edge_dict or (bl, tl) not in edge_dict:
                            continue
                        e1, d1 = edge_dict[(tl, tr)]
                        e2, d2 = edge_dict[(tr, br)]
                        e3, d3 = edge_dict[(br, bl)]
                        e4, d4 = edge_dict[(bl, tl)]
                        curl = (d1 * corrected_diff[e1] + d2 * corrected_diff[e2] +
                               d3 * corrected_diff[e3] + d4 * corrected_diff[e4])
                        if abs(curl) > 0.001:
                            curl_positions.append((r, c, curl))
                if curl_positions:
                    cols_with_errors = sorted(set(p[1] for p in curl_positions))
                    rows_with_errors = sorted(set(p[0] for p in curl_positions))
                    print(f'    DEBUG: curl errors at columns: {cols_with_errors[:10]}{"..." if len(cols_with_errors) > 10 else ""}')
                    print(f'    DEBUG: curl errors at rows: {rows_with_errors[:10]}{"..." if len(rows_with_errors) > 10 else ""}')

                    # Detailed analysis of first few curl errors
                    for r, c, curl in curl_positions[:3]:
                        print(f'    DEBUG: Curl error at ({r},{c}), curl={curl:.4f}:')
                        tl = r * width + c
                        tr = tl + 1
                        bl = tl + width
                        br = bl + 1
                        e1, d1 = edge_dict[(tl, tr)]
                        e2, d2 = edge_dict[(tr, br)]
                        e3, d3 = edge_dict[(br, bl)]
                        e4, d4 = edge_dict[(bl, tl)]
                        # Check if this cell was in valid_dual_nodes
                        dual_idx = dual_to_idx.get((r, c), -1)
                        orig_res = residues[dual_idx] if dual_idx >= 0 else 'N/A'
                        print(f'      dual_idx={dual_idx}, original_residue={orig_res}')
                        # Show neighboring cell residues
                        neighbor_info = []
                        for dr, dc, name in [(-1,0,'top'), (1,0,'bot'), (0,-1,'left'), (0,1,'right')]:
                            nr, nc = r + dr, c + dc
                            n_idx = dual_to_idx.get((nr, nc), -1)
                            n_res = residues[n_idx] if n_idx >= 0 else 'N/A'
                            neighbor_info.append(f'{name}({nr},{nc})={n_res}')
                        print(f'      neighbors: {", ".join(neighbor_info)}')
                        print(f'      edges: e1={e1}(d={d1}), e2={e2}(d={d2}), e3={e3}(d={d3}), e4={e4}(d={d4})')
                        print(f'      k: k1={k[e1]:.0f}, k2={k[e2]:.0f}, k3={k[e3]:.0f}, k4={k[e4]:.0f}')
                        print(f'      phase_diff: p1={phase_diff_wrapped[e1]:.4f}, p2={phase_diff_wrapped[e2]:.4f}, p3={phase_diff_wrapped[e3]:.4f}, p4={phase_diff_wrapped[e4]:.4f}')
                        # Sum of k*d should equal -residue for zero curl
                        k_sum = d1*k[e1] + d2*k[e2] + d3*k[e3] + d4*k[e4]
                        print(f'      k_sum (d1*k1+d2*k2+d3*k3+d4*k4) = {k_sum:.0f}, should be -{orig_res} for zero curl')

            # Check: what fraction of k values are nonzero
            print(f'    DEBUG: k nonzero: {np.count_nonzero(k)} out of {len(k)} edges')

            # Verify residues: check original residues vs corrected (only if residues was computed)
            try:
                if residues is not None and len(residues) > 0:
                    print(f'    DEBUG: original residues sum={int(np.sum(residues))}, nonzero={n_nonzero}')
            except NameError:
                pass  # residues not defined (trivial case)

        # Convert back to radians
        unwrapped = (unwrapped_cycles * (2 * np.pi)).reshape(shape).astype(np.float32)
        # Restore NaN for originally invalid pixels (including filled holes)
        unwrapped[original_nan_mask] = np.nan

        if debug:
            valid_unwrapped = unwrapped[~np.isnan(unwrapped)]
            if len(valid_unwrapped) > 0:
                print(f'    DEBUG: final unwrapped range: [{valid_unwrapped.min():.4f}, {valid_unwrapped.max():.4f}]')

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

    def _unwrap_2d_maxflow(self, phase_da, weight_da=None, conncomp_flag=False,
                           max_jump=1, norm=1, scale=2**16-1, max_iters=100, debug=False):
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
        max_jump : int, optional
            Maximum phase jump step. Default is 1.
        norm : float, optional
            P-norm for energy calculation. Default is 1.
        scale : float, optional
            Scaling factor for integer conversion. Default is 2**16 - 1.
        max_iters : int, optional
            Maximum iterations per jump step. Default is 100.
        debug : bool
            If True, print diagnostic information.

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
            return Stack_unwrap._branch_cut(phase_2d, correlation=corr_2d,
                                            max_jump=max_jump, norm=norm,
                                            scale=scale, max_iters=max_iters, debug=debug)

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

    def unwrap_maxflow(self, phase, weight=None, conncomp=False,
                       max_jump=1, norm=1, scale=2**16-1, max_iters=100, debug=False):
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
        max_jump : int, optional
            Maximum phase jump step. Higher values allow larger discontinuities
            but may be slower. Default is 1.
        norm : float, optional
            P-norm for energy calculation. Use 1 for L1 norm (more robust to outliers)
            or 2 for L2 norm. Default is 1.
        scale : float, optional
            Scaling factor for integer conversion in the max-flow solver.
            Default is 2**16 - 1.
        max_iters : int, optional
            Maximum iterations per jump step. Default is 100.
        debug : bool, optional
            If True, print diagnostic information about connected components
            and processing times. Default is False.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase.
            If conncomp is True: tuple of (Batch unwrapped phase, BatchUnit conncomp).

        Examples
        --------
        Unwrap phase without correlation weighting:
        >>> unwrapped = stack.unwrap_maxflow(intfs)

        Unwrap phase with correlation weighting:
        >>> unwrapped = stack.unwrap_maxflow(intfs, corr)

        Unwrap with connected components:
        >>> unwrapped, conncomp = stack.unwrap_maxflow(intfs, corr, conncomp=True)

        Unwrap with custom parameters:
        >>> unwrapped = stack.unwrap_maxflow(intfs, max_jump=2, max_iters=200)
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

                unwrap_da, comp_da = self._unwrap_2d_maxflow(phase_da, weight_da, conncomp,
                                                           max_jump=max_jump, norm=norm,
                                                           scale=scale, max_iters=max_iters, debug=debug)

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

    def _unwrap_2d_ilp(self, phase_da, weight_da=None, conncomp_flag=False, max_time=300.0, search_workers=1, debug=False):
        """
        Process a single 3D DataArray (pair, y, x) for ILP unwrapping.

        Parameters
        ----------
        phase_da : xr.DataArray
            3D DataArray with dimensions (pair, y, x).
        weight_da : xr.DataArray, optional
            3D DataArray of correlation weights.
        conncomp_flag : bool
            Whether to compute connected components.
        max_time : float
            Maximum solver time in seconds.
        search_workers : int
            Number of parallel workers for CP-SAT solver.
        debug : bool
            If True, print diagnostic information.

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
            """Unwrap a single 2D phase array using ILP."""
            return Stack_unwrap._ilp_unwrap_2d(phase_2d, correlation=corr_2d, max_time=max_time, search_workers=search_workers, debug=debug)

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

    def unwrap_ilp(self, phase, weight=None, conncomp=False, max_time=300.0, search_workers=1, debug=False):
        """
        Unwrap phase using Integer Linear Programming (ILP) with OR-Tools CP-SAT solver.

        This method minimizes the L1 norm of phase jumps on edges subject to
        cycle consistency constraints. ILP provides the mathematically optimal
        solution, which can be valuable for high-precision applications.

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting. If provided, edges with
            lower correlation receive higher costs in the optimization.
        conncomp : bool, optional
            If True, also return connected components. Default is False.
        max_time : float, optional
            Maximum solver time in seconds per 2D slice. Default is 300 (5 minutes).
        search_workers : int, optional
            Number of parallel workers for CP-SAT solver. Default is 1 for
            compatibility with Dask parallel processing. Set higher when
            processing few interferograms with many CPU cores available.
        debug : bool, optional
            If True, print diagnostic information about connected components
            and processing times. Default is False.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase.
            If conncomp is True: tuple of (Batch unwrapped phase, BatchUnit conncomp).

        Notes
        -----
        ILP provides global L1-optimal solution but is computationally expensive.
        Best suited for grids up to ~200×200. Approximate timing:
        - 100×100: ~5-10 seconds
        - 150×150: ~10-20 seconds
        - 200×200: ~30-60 seconds

        For larger grids, if the solver doesn't reach OPTIMAL status within max_time,
        it returns NaN values. Use unwrap_maxflow() for larger grids.

        Examples
        --------
        Unwrap phase without correlation weighting:
        >>> unwrapped = stack.unwrap_ilp(intfs)

        Unwrap phase with correlation weighting:
        >>> unwrapped = stack.unwrap_ilp(intfs, corr)
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

                unwrap_da, comp_da = self._unwrap_2d_ilp(phase_da, weight_da, conncomp, max_time, search_workers, debug)

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

    def _unwrap_2d_minflow(self, phase_da, weight_da=None, conncomp_flag=False, conncomp_size=100, debug=False):
        """
        Process a single 3D DataArray (pair, y, x) for minflow unwrapping.

        Parameters
        ----------
        phase_da : xr.DataArray
            3D DataArray with dimensions (pair, y, x).
        weight_da : xr.DataArray, optional
            3D DataArray of correlation weights.
        conncomp_flag : bool
            Whether to compute connected components.
        conncomp_size : int
            Minimum component size to process.
        debug : bool
            If True, print diagnostic information.

        Returns
        -------
        tuple
            (unwrapped DataArray, conncomp DataArray or None)
        """
        import xarray as xr

        stackvar = phase_da.dims[0]  # 'pair'

        # save original chunks for restoring after processing
        original_chunks = phase_da.chunks

        # rechunk to single chunk per y,x for processing
        chunk_single = {stackvar: 1, 'y': -1, 'x': -1}
        phase_da = phase_da.chunk(chunk_single)
        if weight_da is not None:
            weight_da = weight_da.chunk(chunk_single)

        def _unwrap_single(phase_2d, corr_2d=None):
            """Unwrap a single 2D phase array using minflow."""
            return Stack_unwrap._minflow_unwrap_2d(phase_2d, correlation=corr_2d, conncomp_size=conncomp_size, debug=debug)

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
            comp_da = comp_da.chunk(original_chunks if original_chunks is not None else phase_da.chunks)
            return unwrap_da, comp_da

        return unwrap_da, None

    def unwrap_minflow(self, phase, weight=None, conncomp=False, conncomp_size=100, debug=False):
        """
        Unwrap phase using Minimum Cost Flow (Costantini algorithm).

        This method formulates phase unwrapping as a minimum cost flow problem
        on the dual grid. It minimizes total branch cut length (L1 norm).

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting. If provided, branch cuts
            prefer to go through low-correlation areas.
        conncomp : bool, optional
            If True, also return connected components. Default is False.
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Must be >= 2 since
            single pixels cannot be unwrapped. Default is 100.
        debug : bool, optional
            If True, print diagnostic information about connected components
            and processing times. Default is False.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase.
            If conncomp is True: tuple of (Batch unwrapped phase, BatchUnit conncomp).

        Notes
        -----
        Minflow finds globally optimal branch cut placement in L1 sense. It is faster
        than ILP and scales to large grids (O(E log V) complexity).

        For low-noise data (≤5%), minflow and maxflow produce equivalent results.
        For higher noise, maxflow typically produces lower phase error because
        it optimizes a different objective (greedy nearest-residue matching).

        Examples
        --------
        Unwrap phase without correlation weighting:
        >>> unwrapped = stack.unwrap_minflow(intfs)

        Unwrap phase with correlation weighting:
        >>> unwrapped = stack.unwrap_minflow(intfs, corr)

        Unwrap with connected components:
        >>> unwrapped, conncomp = stack.unwrap_minflow(intfs, corr, conncomp=True)

        Skip small components (less than 500 pixels):
        >>> unwrapped = stack.unwrap_minflow(intfs, corr, conncomp_size=500)
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

                unwrap_da, comp_da = self._unwrap_2d_minflow(phase_da, weight_da, conncomp, conncomp_size, debug)

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

    def unwrap(self, phase, weight=None, conncomp=False, method='maxflow', *args, **kwargs):
        """
        Unwrap phase using the specified method.

        This is a convenience wrapper that dispatches to unwrap_maxflow(),
        unwrap_minflow(), or unwrap_ilp() based on the method parameter.

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting.
        conncomp : bool, optional
            If True, also return connected components. Default is False.
        method : str, optional
            Unwrapping method to use. Options are:
            - 'maxflow': Branch-cut algorithm with max-flow optimization (default).
            - 'minflow': Minimum Cost Flow (Costantini algorithm) - scalable.
            - 'ilp': Integer Linear Programming - optimal but slow.
        *args, **kwargs
            Additional arguments passed to the underlying method.
            For 'maxflow': max_jump, norm, scale, max_iters.
            For 'ilp': max_time.
            For 'minflow': none.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase.
            If conncomp is True: tuple of (Batch unwrapped phase, BatchUnit conncomp).

        Notes
        -----
        Method comparison:

        ========  =======  ===========  =====================================
        Method    Speed    Quality      Best Use Case
        ========  =======  ===========  =====================================
        maxflow   Fast     Best         General use, default
        minflow   Fastest  Good (low    Large grids, clean data
                           noise)
        ilp       Slow     Optimal      Small grids (<200×200), when global
                                        optimum needed
        ========  =======  ===========  =====================================

        Examples
        --------
        Unwrap phase with default max-flow method:
        >>> unwrapped = stack.unwrap(intfs, corr)

        Unwrap phase with minflow method (Costantini):
        >>> unwrapped = stack.unwrap(intfs, corr, method='minflow')

        Unwrap phase with ILP method (optimal but slow):
        >>> unwrapped = stack.unwrap(intfs, corr, method='ilp', max_time=3600)
        """
        if method == 'maxflow':
            return self.unwrap_maxflow(phase, weight, conncomp, *args, **kwargs)
        elif method == 'minflow':
            return self.unwrap_minflow(phase, weight, conncomp, *args, **kwargs)
        elif method == 'ilp':
            return self.unwrap_ilp(phase, weight, conncomp, *args, **kwargs)
        else:
            raise ValueError(f"Unknown unwrapping method: '{method}'. Use 'maxflow', 'minflow', or 'ilp'.")

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
