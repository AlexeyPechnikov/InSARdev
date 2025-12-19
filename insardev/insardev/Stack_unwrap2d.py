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
from .Stack_unwrap1d import Stack_unwrap1d
import numpy as np

class Stack_unwrap2d(Stack_unwrap1d):
    """2D phase unwrapping using various algorithms (maxflow, minflow, ILP)."""

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

    # 4-connectivity structure for scipy.ndimage.label (no diagonals)
    _STRUCTURE_4CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    @staticmethod
    def _get_connected_components(valid_mask_2d, min_size=4):
        """
        Find connected components with bounding boxes using scipy.ndimage.

        Parameters
        ----------
        valid_mask_2d : np.ndarray
            2D boolean array where True indicates valid pixels.
        min_size : int, optional
            Minimum component size to include in results. Default is 4.

        Returns
        -------
        labeled_array : np.ndarray
            2D int32 array with component labels (0 = invalid, 1+ = component labels).
        components : list of dict
            List of component info dicts sorted by size (largest first), each with:
            - 'label': int, the component label in labeled_array
            - 'size': int, number of pixels in the component
            - 'slices': tuple of slices for bounding box
        n_total : int
            Total number of components found (before min_size filtering).
        sizes : np.ndarray
            Array of component sizes indexed by label (sizes[0] = 0, sizes[1] = size of label 1, etc.)
        """
        from scipy import ndimage

        labeled_array, n_total = ndimage.label(valid_mask_2d, structure=Stack_unwrap2d._STRUCTURE_4CONN)

        if n_total == 0:
            return labeled_array, [], 0, np.array([0])

        # Get sizes and bounding boxes efficiently
        sizes = np.bincount(labeled_array.ravel(), minlength=n_total + 1)
        slices = ndimage.find_objects(labeled_array)

        # Build component list sorted by size (largest first), filtering by min_size
        components = [
            {'label': i + 1, 'size': sizes[i + 1], 'slices': slices[i]}
            for i in np.argsort(sizes[1:])[::-1]
            if sizes[i + 1] >= min_size and slices[i] is not None
        ]

        return labeled_array, components, n_total, sizes

    @staticmethod
    def _print_component_stats_debug(method_name, shape, n_valid, n_components, sizes):
        """Print debug statistics about connected components."""
        comp_sizes = sizes[1:n_components + 1] if n_components > 0 else []
        sorted_sizes = np.sort(comp_sizes)[::-1]
        n_tiny = np.sum(comp_sizes < 10) if len(comp_sizes) > 0 else 0

        print(f'{method_name}: {shape} grid, {n_valid} valid pixels, {n_components} components')
        if n_components <= 10:
            print(f'  Component sizes: {list(sorted_sizes)}')
        else:
            print(f'  Largest 5: {list(sorted_sizes[:5])}, smallest 5: {list(sorted_sizes[-5:])}, tiny(<10px): {n_tiny}')

    @staticmethod
    def _find_connected_components(valid_mask_2d, min_size=None):
        """
        Find connected components in a 2D valid mask using 4-connectivity.

        Parameters
        ----------
        valid_mask_2d : np.ndarray
            2D boolean array where True indicates valid pixels.
        min_size : int, optional
            Minimum component size to include. If None, all components are returned.

        Returns
        -------
        list of np.ndarray
            List of boolean masks, one per connected component (sorted by size, largest first).
        """
        labeled_array, components, n_total, _ = Stack_unwrap2d._get_connected_components(
            valid_mask_2d, min_size=min_size or 1
        )
        return [(labeled_array == c['label']) for c in components]

    @staticmethod
    def _line_crosses_mask(p1, p2, mask):
        """
        Check if the line segment from p1 to p2 crosses any True pixels in mask.

        Uses Bresenham-like sampling along the line.

        Parameters
        ----------
        p1, p2 : tuple
            (row, col) endpoints of the line segment.
        mask : np.ndarray
            2D boolean array to check against.

        Returns
        -------
        bool
            True if the line crosses any True pixels in mask.
        """
        r1, c1 = p1
        r2, c2 = p2

        # Number of steps (at least the max of row/col difference)
        n_steps = max(abs(r2 - r1), abs(c2 - c1), 1)

        for step in range(1, n_steps):  # Skip endpoints
            t = step / n_steps
            r = int(round(r1 + t * (r2 - r1)))
            c = int(round(c1 + t * (c2 - c1)))

            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                if mask[r, c]:
                    return True

        return False

    @staticmethod
    def _find_component_connections(components, conncomp_gap=None, max_neighbors=30):
        """
        Find direct connections between components based on minimum distance.

        Uses a size-weighted approach: prioritizes connections to larger components
        to ensure small components connect to the main component network.

        Parameters
        ----------
        components : list of np.ndarray
            List of boolean masks, one per connected component.
        conncomp_gap : int or None, optional
            Maximum pixel distance to consider components as connectable.
            If None (default), no distance limit is applied.
        max_neighbors : int, optional
            Maximum number of nearest neighbors to check for each component.
            Default is 30.

        Returns
        -------
        list of tuple
            List of (comp_i, comp_j, closest_i, closest_j, distance) where:
            - comp_i, comp_j: component indices
            - closest_i: (row, col) of closest pixel in component i
            - closest_j: (row, col) of closest pixel in component j
            - distance: Euclidean distance between closest pixels
        """
        n_comps = len(components)
        if n_comps < 2:
            return []

        # Create combined mask of all components for intersection checking
        all_comps_mask = np.zeros_like(components[0], dtype=bool)
        for comp_mask in components:
            all_comps_mask |= comp_mask

        # Get pixel coordinates, centroids, and sizes for each component
        comp_coords = []
        centroids = []
        comp_sizes = []
        for comp_mask in components:
            rows, cols = np.where(comp_mask)
            comp_coords.append(np.column_stack([rows, cols]))
            centroids.append((np.mean(rows), np.mean(cols)))
            comp_sizes.append(len(rows))

        centroids = np.array(centroids)
        comp_sizes = np.array(comp_sizes)

        # Sort components by size (largest first) for prioritized connection
        size_order = np.argsort(comp_sizes)[::-1]

        # For each component, find candidate neighbors using size-weighted scoring
        # Score = size_weight / (distance + 1), prefer larger and closer components
        candidate_pairs = set()
        for i in range(n_comps):
            # Compute distances from this centroid to all others
            dists = np.sqrt(np.sum((centroids - centroids[i]) ** 2, axis=1))
            dists[i] = np.inf  # Exclude self

            # Size-weighted score: larger components get higher priority
            # Use log(size) to avoid extreme weighting
            size_weights = np.log1p(comp_sizes)
            scores = size_weights / (dists + 1)
            scores[i] = -np.inf  # Exclude self

            # Get indices of best candidates (highest scores)
            n_neighbors = min(max_neighbors, n_comps - 1)
            best_candidates = np.argpartition(scores, -n_neighbors)[-n_neighbors:]

            for j in best_candidates:
                if scores[j] > -np.inf:
                    # Add as sorted tuple to avoid duplicates
                    pair = (min(i, j), max(i, j))
                    candidate_pairs.add(pair)

        # Also ensure every component considers connecting to the largest components
        # This guarantees small isolated components can reach the main network
        n_largest = min(5, n_comps)
        largest_indices = size_order[:n_largest]
        for i in range(n_comps):
            if i not in largest_indices:
                for j in largest_indices:
                    pair = (min(i, j), max(i, j))
                    candidate_pairs.add(pair)

        connections = []

        # Check only candidate pairs
        for i, j in candidate_pairs:
            coords_i = comp_coords[i]
            coords_j = comp_coords[j]

            # For large components, subsample to speed up initial search
            max_sample = 500  # Reduced from 1000 to save memory
            if len(coords_i) > max_sample:
                idx_i = np.random.choice(len(coords_i), max_sample, replace=False)
                sample_i = coords_i[idx_i]
            else:
                sample_i = coords_i

            if len(coords_j) > max_sample:
                idx_j = np.random.choice(len(coords_j), max_sample, replace=False)
                sample_j = coords_j[idx_j]
            else:
                sample_j = coords_j

            # Compute pairwise distances efficiently using scipy
            from scipy.spatial.distance import cdist
            dists = cdist(sample_i, sample_j, metric='euclidean')

            min_dist = np.min(dists)

            # Check distance limit if specified
            if conncomp_gap is not None and min_dist > conncomp_gap:
                continue

            # Find the actual closest pair (in full set if we subsampled)
            search_radius = max(min_dist * 2, 100)  # Search radius for refinement

            if len(coords_i) > max_sample or len(coords_j) > max_sample:
                # Refine: find closest in full set near the approximate closest
                approx_idx = np.unravel_index(np.argmin(dists), dists.shape)
                approx_i = sample_i[approx_idx[0]]
                approx_j = sample_j[approx_idx[1]]

                # Search in neighborhood
                dist_to_approx_i = np.sqrt(np.sum((coords_i - approx_i) ** 2, axis=1))
                near_i = coords_i[dist_to_approx_i < search_radius]

                dist_to_approx_j = np.sqrt(np.sum((coords_j - approx_j) ** 2, axis=1))
                near_j = coords_j[dist_to_approx_j < search_radius]

                if len(near_i) == 0 or len(near_j) == 0:
                    continue

                # Limit to avoid memory explosion with large refinement sets
                max_refine = 1000
                if len(near_i) > max_refine:
                    sort_idx = np.argsort(dist_to_approx_i[dist_to_approx_i < search_radius])[:max_refine]
                    near_i = near_i[sort_idx]
                if len(near_j) > max_refine:
                    sort_idx = np.argsort(dist_to_approx_j[dist_to_approx_j < search_radius])[:max_refine]
                    near_j = near_j[sort_idx]

                dists = cdist(near_i, near_j, metric='euclidean')

                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                closest_i = tuple(near_i[min_idx[0]])
                closest_j = tuple(near_j[min_idx[1]])
                min_dist = dists[min_idx]
            else:
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                closest_i = tuple(coords_i[min_idx[0]])
                closest_j = tuple(coords_j[min_idx[1]])

            # Check if connection is direct (doesn't cross other components)
            # Instead of creating a full mask copy, check crossing inline
            def crosses_other_components(p1, p2, skip_i, skip_j):
                """Check if line crosses any component other than skip_i and skip_j."""
                r1, c1 = p1
                r2, c2 = p2
                n_steps = max(abs(r2 - r1), abs(c2 - c1), 1)
                for step in range(1, n_steps):
                    t = step / n_steps
                    r = int(round(r1 + t * (r2 - r1)))
                    c = int(round(c1 + t * (c2 - c1)))
                    if 0 <= r < all_comps_mask.shape[0] and 0 <= c < all_comps_mask.shape[1]:
                        if all_comps_mask[r, c] and not components[skip_i][r, c] and not components[skip_j][r, c]:
                            return True
                return False

            if crosses_other_components(closest_i, closest_j, i, j):
                # Connection crosses another component - skip it
                continue

            connections.append((i, j, closest_i, closest_j, min_dist))

        return connections

    @staticmethod
    def _estimate_component_offset(unwrapped, comp_mask_i, comp_mask_j, closest_i, closest_j, n_neighbors=5):
        """
        Estimate the integer 2π offset between two components.

        Uses N pixels closest to the connection point on each side,
        which includes interior pixels (less noisy than border-only).
        Uses median for robustness to outliers.

        Parameters
        ----------
        unwrapped : np.ndarray
            2D array of unwrapped phase values.
        comp_mask_i, comp_mask_j : np.ndarray
            Boolean masks for the two components.
        closest_i, closest_j : tuple
            (row, col) of the closest pixels defining the connection.
        n_neighbors : int, optional
            Number of pixels to use on each side. Default is 5.

        Returns
        -------
        tuple
            (k_offset, confidence) where:
            - k_offset: integer number of 2π cycles to add to component j
            - confidence: measure of how reliable the estimate is (0-1)
        """
        # Get coordinates of pixels in each component
        rows_i, cols_i = np.where(comp_mask_i)
        rows_j, cols_j = np.where(comp_mask_j)

        # Find N closest pixels to the connection point on each side
        dist_i = np.sqrt((rows_i - closest_i[0])**2 + (cols_i - closest_i[1])**2)
        dist_j = np.sqrt((rows_j - closest_j[0])**2 + (cols_j - closest_j[1])**2)

        # Get indices of N closest pixels
        n_i = min(n_neighbors, len(dist_i))
        n_j = min(n_neighbors, len(dist_j))

        if n_i < 3 or n_j < 3:
            return 0, 0.0

        idx_i = np.argpartition(dist_i, n_i - 1)[:n_i]
        idx_j = np.argpartition(dist_j, n_j - 1)[:n_j]

        # Get phase values at these pixels
        phase_i = unwrapped[rows_i[idx_i], cols_i[idx_i]]
        phase_j = unwrapped[rows_j[idx_j], cols_j[idx_j]]

        # Filter out NaN values
        valid_i = ~np.isnan(phase_i)
        valid_j = ~np.isnan(phase_j)

        if np.sum(valid_i) < 3 or np.sum(valid_j) < 3:
            return 0, 0.0

        # Use median for robustness to outliers
        median_phase_i = np.median(phase_i[valid_i])
        median_phase_j = np.median(phase_j[valid_j])

        # Phase difference and integer offset
        delta_phase = median_phase_i - median_phase_j
        k_offset = int(np.round(delta_phase / (2 * np.pi)))

        # Confidence based on:
        # 1. How close the fractional part is to an integer
        # 2. Standard deviation of the phase values (lower = more confident)
        fractional = (delta_phase / (2 * np.pi)) - k_offset
        frac_confidence = 1.0 - 2 * abs(fractional)  # 1.0 if exactly integer, 0.0 if halfway

        # Check consistency: std of phase values should be small relative to 2π
        std_i = np.std(phase_i[valid_i])
        std_j = np.std(phase_j[valid_j])
        std_confidence = max(0, 1.0 - (std_i + std_j) / (2 * np.pi))

        confidence = frac_confidence * std_confidence

        return k_offset, confidence

    @staticmethod
    def _connect_components_ilp(unwrapped, components, connections, n_neighbors=5, max_time=60.0, debug=False):
        """
        Connect separately-unwrapped components using ILP optimization.

        Finds optimal integer 2π offsets for each component to minimize
        phase discontinuities at connection points.

        Parameters
        ----------
        unwrapped : np.ndarray
            2D array with separately unwrapped components.
        components : list of np.ndarray
            List of boolean masks for each component.
        connections : list of tuple
            Output from _find_component_connections.
        n_neighbors : int, optional
            Number of pixels to use for offset estimation at each connection.
            Default is 50.
        max_time : float, optional
            Maximum solver time in seconds. Default is 60.
        debug : bool, optional
            If True, print diagnostic information.

        Returns
        -------
        np.ndarray
            2D array with connected unwrapped phase.
        """
        from ortools.sat.python import cp_model

        n_comps = len(components)
        if n_comps < 2 or len(connections) == 0:
            return unwrapped.copy()

        # Estimate offsets and weights for each connection
        edge_data = []
        for comp_i, comp_j, closest_i, closest_j, distance in connections:
            k_offset, confidence = Stack_unwrap2d._estimate_component_offset(
                unwrapped, components[comp_i], components[comp_j],
                closest_i, closest_j, n_neighbors=n_neighbors
            )
            # Weight by confidence and inverse distance
            weight = confidence / (distance + 1.0)
            edge_data.append((comp_i, comp_j, k_offset, weight))

            if debug:
                print(f'  Connection {comp_i}-{comp_j}: dist={distance:.1f}, '
                      f'k_offset={k_offset}, confidence={confidence:.3f}')

        # Build ILP model
        model = cp_model.CpModel()

        # Variables: k_i = integer offset for component i
        # Range: reasonable bounds (±100 cycles should be enough)
        k_vars = [model.NewIntVar(-100, 100, f'k_{i}') for i in range(n_comps)]

        # Fix component 0 as reference
        model.Add(k_vars[0] == 0)

        # Objective: minimize weighted sum of |measured_offset - (k_i - k_j)|
        # We use absolute value linearization: |x| = max(x, -x)
        scale = 1000  # Scale weights to integers for CP-SAT
        abs_vars = []

        for idx, (comp_i, comp_j, k_offset, weight) in enumerate(edge_data):
            # k_offset = round((phase_i - phase_j) / 2π)
            # To align: phase_j + k_offset*2π ≈ phase_i
            # After offsets: (phase_i + k_i*2π) ≈ (phase_j + k_j*2π)
            # So: phase_i - phase_j ≈ (k_j - k_i)*2π
            # Thus: k_offset ≈ k_j - k_i
            # Minimize: |k_offset - (k_j - k_i)| = |k_offset - k_j + k_i|

            # Create auxiliary variable for the difference
            diff_var = model.NewIntVar(-200, 200, f'diff_{idx}')
            model.Add(diff_var == k_offset - k_vars[comp_j] + k_vars[comp_i])

            # Absolute value
            abs_var = model.NewIntVar(0, 200, f'abs_{idx}')
            model.AddAbsEquality(abs_var, diff_var)

            abs_vars.append((abs_var, int(weight * scale)))

        # Objective: minimize weighted sum of absolute differences
        model.Minimize(sum(w * v for v, w in abs_vars))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if debug:
                print(f'  ILP solver failed with status {status}')
            return unwrapped.copy()

        # Extract solution
        k_offsets = [solver.Value(k_vars[i]) for i in range(n_comps)]

        if debug:
            print(f'  ILP solution: k_offsets = {k_offsets}')

        # Apply offsets to create connected result
        result = unwrapped.copy()
        for i, comp_mask in enumerate(components):
            if k_offsets[i] != 0:
                result[comp_mask] += k_offsets[i] * 2 * np.pi

        return result

    @staticmethod
    def _branch_cut(phase, correlation=None, conncomp_size=100, max_jump=1, norm=1, scale=2**16 - 1, max_iters=100, debug=False):
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
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 100.
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

        shape = phase.shape

        # handle NaN values - create mask for valid pixels
        valid_mask_2d = ~np.isnan(phase)
        if not np.any(valid_mask_2d):
            if debug:
                print(f'Maxflow: no valid pixels in {shape} grid')
            return np.full(shape, np.nan, dtype=np.float32)

        # Find connected components - use efficient scipy labeling
        min_size = max(conncomp_size, 4)
        labeled, components, n_total, sizes = Stack_unwrap2d._get_connected_components(valid_mask_2d, min_size)

        if debug:
            Stack_unwrap2d._print_component_stats_debug('Maxflow', shape, np.sum(valid_mask_2d), n_total, sizes)
            if n_total > len(components):
                print(f'  Skipping {n_total - len(components)} components with < {min_size} pixels')

        if len(components) == 0:
            return np.full(shape, np.nan, dtype=np.float32)

        result = np.full(shape, np.nan, dtype=np.float32)

        for i, comp in enumerate(components):
            slices, label, comp_size = comp['slices'], comp['label'], comp['size']
            sub_mask = (labeled[slices] == label)

            # Extract subgrid using slices from find_objects
            sub_phase = phase[slices].copy()
            sub_phase[~sub_mask] = np.nan
            sub_corr = correlation[slices].copy() if correlation is not None else None
            if sub_corr is not None:
                sub_corr[~sub_mask] = np.nan

            if debug:
                print(f'  Component {i+1}/{len(components)}: {comp_size} pixels, bbox {sub_phase.shape}', end='', flush=True)
                t0 = time.time()

            # Unwrap this component's subgrid
            sub_result = Stack_unwrap2d._branch_cut_single(
                sub_phase, sub_corr, max_jump, norm, scale, max_iters, debug
            )

            # Merge back into result
            result[slices][sub_mask] = sub_result[sub_mask]
            if debug:
                elapsed = time.time() - t0
                status = 'OK' if not np.all(np.isnan(sub_result[sub_mask])) else 'FAILED'
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
        all_edges = Stack_unwrap2d._get_2d_edges(shape)

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

        return unwrapped - np.nanmean(unwrapped)

    @staticmethod
    def _ilp_unwrap_2d(phase, correlation=None, conncomp_size=100, max_time=300.0, search_workers=1, debug=False):
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
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 100.
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

        # Find connected components - use efficient scipy labeling
        min_size = max(conncomp_size, 4)
        labeled, components, n_total, sizes = Stack_unwrap2d._get_connected_components(valid_mask_2d, min_size)

        if debug:
            Stack_unwrap2d._print_component_stats_debug('ILP', shape, np.sum(valid_mask_2d), n_total, sizes)
            if n_total > len(components):
                print(f'  Skipping {n_total - len(components)} components with < {min_size} pixels')

        if len(components) == 0:
            return np.full(shape, np.nan, dtype=np.float32)

        result = np.full(shape, np.nan, dtype=np.float32)

        for i, comp in enumerate(components):
            slices, label, comp_size = comp['slices'], comp['label'], comp['size']
            sub_mask = (labeled[slices] == label)

            # Extract subgrid using slices from find_objects
            sub_phase = phase[slices].copy()
            sub_phase[~sub_mask] = np.nan
            sub_corr = correlation[slices].copy() if correlation is not None else None
            if sub_corr is not None:
                sub_corr[~sub_mask] = np.nan

            if debug:
                print(f'  Component {i+1}/{len(components)}: {comp_size} pixels, bbox {sub_phase.shape}', end='', flush=True)
                t0 = time.time()

            # Unwrap this component's subgrid
            sub_result = Stack_unwrap2d._ilp_unwrap_2d_single(sub_phase, sub_corr, max_time, search_workers, debug)

            # Merge back into result
            result[slices][sub_mask] = sub_result[sub_mask]
            if debug:
                elapsed = time.time() - t0
                status = 'OK' if not np.all(np.isnan(sub_result[sub_mask])) else 'FAILED'
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
        all_edges = Stack_unwrap2d._get_2d_edges(shape)

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

        return unwrapped - np.nanmean(unwrapped)

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

        # Find connected components - use efficient scipy labeling
        min_size = max(conncomp_size, 4)
        labeled, components, n_total, sizes = Stack_unwrap2d._get_connected_components(valid_mask_2d, min_size)

        if debug:
            Stack_unwrap2d._print_component_stats_debug('Minflow', shape, np.sum(valid_mask_2d), n_total, sizes)
            if n_total > len(components):
                print(f'  Skipping {n_total - len(components)} components with < {min_size} pixels')

        if len(components) == 0:
            return np.full(shape, np.nan, dtype=np.float32)

        result = np.full(shape, np.nan, dtype=np.float32)

        for i, comp in enumerate(components):
            slices, label, comp_size = comp['slices'], comp['label'], comp['size']
            sub_mask = (labeled[slices] == label)

            # Extract subgrid using slices from find_objects
            sub_phase = phase[slices].copy()
            sub_phase[~sub_mask] = np.nan
            sub_corr = correlation[slices].copy() if correlation is not None else None
            if sub_corr is not None:
                sub_corr[~sub_mask] = np.nan

            if debug:
                # Check for narrow regions in the component
                col_counts = np.sum(sub_mask, axis=0)
                row_counts = np.sum(sub_mask, axis=1)
                narrow_cols = np.sum((col_counts > 0) & (col_counts <= 2))
                narrow_rows = np.sum((row_counts > 0) & (row_counts <= 2))
                print(f'  Component {i+1}/{len(components)}: {comp_size} pixels, bbox {sub_phase.shape}', end='')
                if narrow_cols > 0 or narrow_rows > 0:
                    print(f', NARROW: {narrow_cols} cols, {narrow_rows} rows', end='')
                print('', flush=True)
                t0 = time.time()

            # Unwrap this component's subgrid
            sub_result = Stack_unwrap2d._minflow_unwrap_2d_single(sub_phase, sub_corr, debug)

            # Merge back into result
            result[slices][sub_mask] = sub_result[sub_mask]
            if debug:
                elapsed = time.time() - t0
                status = 'OK' if not np.all(np.isnan(sub_result[sub_mask])) else 'FAILED'
                print(f' -> {status} ({elapsed:.2f}s)')

        return result

    @staticmethod
    def _minflow_unwrap_2d_single(phase, correlation=None, debug=False):
        """Core minflow implementation for a single connected component.

        Memory-optimized: uses numpy arrays instead of Python dicts for lookups.
        """
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

        # Fill interior holes to make domain simply-connected
        # This prevents topological issues with multiply-connected domains
        # The filled pixels are restored to NaN at the end (using original_nan_mask)
        hole_labels, n_holes = ndimage.label(~valid_mask_2d)
        if n_holes > 1:
            # Find the exterior (largest NaN region)
            hole_sizes = ndimage.sum(~valid_mask_2d, hole_labels, range(1, n_holes + 1))
            exterior_label = np.argmax(hole_sizes) + 1
            # Fill all interior holes with mean of boundary pixels
            phase_filled = phase.copy()
            n_filled = 0
            n_holes_filled = 0
            for label_id in range(1, n_holes + 1):
                if label_id == exterior_label:
                    continue
                hole_mask = hole_labels == label_id
                hole_size = hole_sizes[label_id - 1]
                # Find boundary pixels around the hole
                dilated = ndimage.binary_dilation(hole_mask)
                boundary = dilated & valid_mask_2d
                if np.any(boundary):
                    # Use mean of boundary pixels
                    boundary_mean = np.nanmean(phase[boundary])
                    phase_filled[hole_mask] = boundary_mean
                    n_filled += int(hole_size)
                    n_holes_filled += 1
            if n_filled > 0 and debug:
                print(f'    DEBUG: filled {n_holes_filled} interior holes ({n_filled} pixels)')
            phase = phase_filled
            valid_mask_2d = ~np.isnan(phase)

        # Flatten and normalize phase to cycles
        phase_flat = (phase.ravel().astype(np.float64)) / (2 * np.pi)

        # Handle NaN values
        valid_mask = ~np.isnan(phase_flat)
        if not np.any(valid_mask):
            return np.full(shape, np.nan, dtype=np.float32)

        # Vectorized edge construction using numpy operations
        valid_mask_2d_for_edges = valid_mask.reshape(height, width)

        # Horizontal edges: pixel (r,c) to (r,c+1)
        # Valid where both left and right pixels are valid
        h_valid = valid_mask_2d_for_edges[:, :-1] & valid_mask_2d_for_edges[:, 1:]
        h_edge_idx = np.full((height, width - 1), -1, dtype=np.int32)
        h_valid_flat = h_valid.ravel()
        n_h_edges = np.sum(h_valid_flat)
        h_edge_idx[h_valid] = np.arange(n_h_edges, dtype=np.int32)

        # Build horizontal edge pixel pairs vectorized
        h_rows, h_cols = np.where(h_valid)
        h_left = h_rows * width + h_cols
        h_right = h_left + 1

        # Vertical edges: pixel (r,c) to (r+1,c)
        # Valid where both top and bottom pixels are valid
        v_valid = valid_mask_2d_for_edges[:-1, :] & valid_mask_2d_for_edges[1:, :]
        v_edge_idx = np.full((height - 1, width), -1, dtype=np.int32)
        v_valid_flat = v_valid.ravel()
        n_v_edges = np.sum(v_valid_flat)
        v_edge_idx[v_valid] = np.arange(n_h_edges, n_h_edges + n_v_edges, dtype=np.int32)

        # Build vertical edge pixel pairs vectorized
        v_rows, v_cols = np.where(v_valid)
        v_top = v_rows * width + v_cols
        v_bot = v_top + width

        n_edges = n_h_edges + n_v_edges
        if n_edges == 0:
            return np.full(shape, np.nan, dtype=np.float32)

        # Build combined edge array
        edges = np.empty((n_edges, 2), dtype=np.int32)
        edges[:n_h_edges, 0] = h_left
        edges[:n_h_edges, 1] = h_right
        edges[n_h_edges:, 0] = v_top
        edges[n_h_edges:, 1] = v_bot
        n_nodes = phase_flat.size

        # Compute wrapped phase differences (in cycles)
        phase_diff = phase_flat[edges[:, 1]] - phase_flat[edges[:, 0]]
        phase_diff_wrapped = phase_diff - np.round(phase_diff)

        # Prepare correlation weights for edge costs
        if correlation is not None:
            corr_flat = correlation.ravel().astype(np.float64)
            edge_corr = (corr_flat[edges[:, 0]] + corr_flat[edges[:, 1]]) / 2
            # Suppress warning for NaN→int64 cast (NaN becomes large value = high cost, which is desired)
            with np.errstate(invalid='ignore'):
                edge_costs = (1000.0 / np.maximum(edge_corr, 0.01)).astype(np.int64)
        else:
            edge_costs = np.ones(n_edges, dtype=np.int64) * 1000

        # Dual grid dimensions
        dual_height = height - 1
        dual_width = width - 1

        if dual_height <= 0 or dual_width <= 0:
            # No dual nodes - trivial unwrapping
            k = np.zeros(n_edges, dtype=np.float64)
        else:
            # Vectorized dual node identification and residue computation
            # A valid dual node at (r,c) requires all 4 corners valid and all 4 edges exist

            # Check 4 corners validity
            tl_valid = valid_mask_2d[:-1, :-1]
            tr_valid = valid_mask_2d[:-1, 1:]
            bl_valid = valid_mask_2d[1:, :-1]
            br_valid = valid_mask_2d[1:, 1:]
            corners_valid = tl_valid & tr_valid & bl_valid & br_valid

            # Check 4 edges exist
            h_top_exists = h_edge_idx[:-1, :] >= 0  # top horizontal edge
            h_bot_exists = h_edge_idx[1:, :] >= 0   # bottom horizontal edge
            v_left_exists = v_edge_idx[:, :-1] >= 0  # left vertical edge
            v_right_exists = v_edge_idx[:, 1:] >= 0  # right vertical edge

            valid_dual_mask = corners_valid & h_top_exists & h_bot_exists & v_left_exists & v_right_exists

            # Get dual node coordinates
            dual_rows, dual_cols = np.where(valid_dual_mask)
            n_valid_dual = len(dual_rows)

            if n_valid_dual == 0:
                k = np.zeros(n_edges, dtype=np.float64)
            else:
                # Create dual node index lookup array (memory-efficient vs dict)
                dual_to_idx = np.full((dual_height, dual_width), -1, dtype=np.int32)
                dual_to_idx[dual_rows, dual_cols] = np.arange(n_valid_dual, dtype=np.int32)

                # Compute residues vectorized
                e1 = h_edge_idx[dual_rows, dual_cols]      # top horizontal
                e2 = v_edge_idx[dual_rows, dual_cols + 1]  # right vertical
                e3 = h_edge_idx[dual_rows + 1, dual_cols]  # bottom horizontal
                e4 = v_edge_idx[dual_rows, dual_cols]      # left vertical

                # Directions: top(+1), right(+1), bottom(-1), left(-1)
                cycle_sum = (phase_diff_wrapped[e1] + phase_diff_wrapped[e2]
                            - phase_diff_wrapped[e3] - phase_diff_wrapped[e4])
                residues = np.round(cycle_sum).astype(np.int32)
                total_residue = int(np.sum(residues))
                n_nonzero = np.count_nonzero(residues)

                if n_nonzero == 0:
                    k = np.zeros(n_edges, dtype=np.float64)
                else:
                    # Build MCF network using vectorized operations
                    smcf = min_cost_flow.SimpleMinCostFlow()
                    boundary_node = n_valid_dual
                    max_cap = max(100, abs(total_residue) + 10)

                    # Collect all arcs in lists, then convert to arrays at end
                    arc_lists = {'start': [], 'end': [], 'capacity': [], 'cost': [], 'edge': [], 'dir': []}

                    def add_arcs_vectorized(nodes1, nodes2, e_idxs, costs, dir_signs):
                        """Add bidirectional arcs from arrays."""
                        n = len(nodes1)
                        if n == 0:
                            return
                        arc_lists['start'].append(np.concatenate([nodes1, nodes2]))
                        arc_lists['end'].append(np.concatenate([nodes2, nodes1]))
                        arc_lists['capacity'].append(np.full(2 * n, max_cap, dtype=np.int64))
                        arc_lists['cost'].append(np.concatenate([costs, costs]))
                        arc_lists['edge'].append(np.concatenate([e_idxs, e_idxs]))
                        arc_lists['dir'].append(np.concatenate([dir_signs, -dir_signs]))

                    # Vectorized horizontal arcs (between horizontally adjacent dual nodes)
                    # These cross vertical pixel edges at column c+1
                    has_right = dual_cols + 1 < dual_width
                    right_idx = np.where(has_right)[0]
                    if len(right_idx) > 0:
                        r_r, c_r = dual_rows[right_idx], dual_cols[right_idx]
                        right_dual = dual_to_idx[r_r, c_r + 1]
                        valid_right = right_dual >= 0
                        if np.any(valid_right):
                            idx_valid = right_idx[valid_right]
                            r_v, c_v = dual_rows[idx_valid], dual_cols[idx_valid]
                            e_idx = v_edge_idx[r_v, c_v + 1]
                            edge_valid = e_idx >= 0
                            if np.any(edge_valid):
                                final_idx = idx_valid[edge_valid]
                                final_e = e_idx[edge_valid]
                                final_right = dual_to_idx[dual_rows[final_idx], dual_cols[final_idx] + 1]
                                add_arcs_vectorized(
                                    final_idx.astype(np.int32), final_right.astype(np.int32),
                                    final_e.astype(np.int32), edge_costs[final_e],
                                    np.full(len(final_idx), -1, dtype=np.int8)
                                )

                    # Vectorized vertical arcs (between vertically adjacent dual nodes)
                    # These cross horizontal pixel edges at row r+1
                    has_bot = dual_rows + 1 < dual_height
                    bot_idx = np.where(has_bot)[0]
                    if len(bot_idx) > 0:
                        r_b, c_b = dual_rows[bot_idx], dual_cols[bot_idx]
                        bot_dual = dual_to_idx[r_b + 1, c_b]
                        valid_bot = bot_dual >= 0
                        if np.any(valid_bot):
                            idx_valid = bot_idx[valid_bot]
                            r_v, c_v = dual_rows[idx_valid], dual_cols[idx_valid]
                            e_idx = h_edge_idx[r_v + 1, c_v]
                            edge_valid = e_idx >= 0
                            if np.any(edge_valid):
                                final_idx = idx_valid[edge_valid]
                                final_e = e_idx[edge_valid]
                                final_bot = dual_to_idx[dual_rows[final_idx] + 1, dual_cols[final_idx]]
                                add_arcs_vectorized(
                                    final_idx.astype(np.int32), final_bot.astype(np.int32),
                                    final_e.astype(np.int32), edge_costs[final_e],
                                    np.full(len(final_idx), 1, dtype=np.int8)
                                )

                    # Boundary connections - vectorized detection of boundary nodes
                    # A dual node is on boundary if any neighbor is invalid
                    top_neighbor = np.where(dual_rows > 0, dual_to_idx[np.maximum(dual_rows - 1, 0), dual_cols], -1)
                    bot_neighbor = np.where(dual_rows < dual_height - 1, dual_to_idx[np.minimum(dual_rows + 1, dual_height - 1), dual_cols], -1)
                    left_neighbor = np.where(dual_cols > 0, dual_to_idx[dual_rows, np.maximum(dual_cols - 1, 0)], -1)
                    right_neighbor = np.where(dual_cols < dual_width - 1, dual_to_idx[dual_rows, np.minimum(dual_cols + 1, dual_width - 1)], -1)

                    is_top_boundary = (dual_rows == 0) | (top_neighbor < 0)
                    is_bot_boundary = (dual_rows == dual_height - 1) | (bot_neighbor < 0)
                    is_left_boundary = (dual_cols == 0) | (left_neighbor < 0)
                    is_right_boundary = (dual_cols == dual_width - 1) | (right_neighbor < 0)

                    # Connect boundary nodes - use priority: top > bottom > left > right
                    # Track which nodes are already connected
                    connected = np.zeros(n_valid_dual, dtype=bool)
                    boundary_edges_used = set()

                    # Top boundary connections
                    top_candidates = np.where(is_top_boundary & ~connected)[0]
                    if len(top_candidates) > 0:
                        r_t, c_t = dual_rows[top_candidates], dual_cols[top_candidates]
                        e_idx = h_edge_idx[r_t, c_t]
                        valid_e = e_idx >= 0
                        for i, idx in enumerate(top_candidates):
                            if valid_e[i] and e_idx[i] not in boundary_edges_used:
                                arc_lists['start'].append(np.array([boundary_node, idx], dtype=np.int32))
                                arc_lists['end'].append(np.array([idx, boundary_node], dtype=np.int32))
                                arc_lists['capacity'].append(np.array([max_cap, max_cap], dtype=np.int64))
                                arc_lists['cost'].append(np.array([edge_costs[e_idx[i]], edge_costs[e_idx[i]]], dtype=np.int64))
                                arc_lists['edge'].append(np.array([e_idx[i], e_idx[i]], dtype=np.int32))
                                arc_lists['dir'].append(np.array([1, -1], dtype=np.int8))
                                boundary_edges_used.add(e_idx[i])
                                connected[idx] = True

                    # Bottom boundary connections
                    bot_candidates = np.where(is_bot_boundary & ~connected)[0]
                    if len(bot_candidates) > 0:
                        r_b, c_b = dual_rows[bot_candidates], dual_cols[bot_candidates]
                        e_idx = h_edge_idx[r_b + 1, c_b]
                        valid_e = e_idx >= 0
                        for i, idx in enumerate(bot_candidates):
                            if valid_e[i] and e_idx[i] not in boundary_edges_used:
                                arc_lists['start'].append(np.array([boundary_node, idx], dtype=np.int32))
                                arc_lists['end'].append(np.array([idx, boundary_node], dtype=np.int32))
                                arc_lists['capacity'].append(np.array([max_cap, max_cap], dtype=np.int64))
                                arc_lists['cost'].append(np.array([edge_costs[e_idx[i]], edge_costs[e_idx[i]]], dtype=np.int64))
                                arc_lists['edge'].append(np.array([e_idx[i], e_idx[i]], dtype=np.int32))
                                arc_lists['dir'].append(np.array([-1, 1], dtype=np.int8))
                                boundary_edges_used.add(e_idx[i])
                                connected[idx] = True

                    # Left boundary connections
                    left_candidates = np.where(is_left_boundary & ~connected)[0]
                    if len(left_candidates) > 0:
                        r_l, c_l = dual_rows[left_candidates], dual_cols[left_candidates]
                        e_idx = v_edge_idx[r_l, c_l]
                        valid_e = e_idx >= 0
                        for i, idx in enumerate(left_candidates):
                            if valid_e[i] and e_idx[i] not in boundary_edges_used:
                                arc_lists['start'].append(np.array([boundary_node, idx], dtype=np.int32))
                                arc_lists['end'].append(np.array([idx, boundary_node], dtype=np.int32))
                                arc_lists['capacity'].append(np.array([max_cap, max_cap], dtype=np.int64))
                                arc_lists['cost'].append(np.array([edge_costs[e_idx[i]], edge_costs[e_idx[i]]], dtype=np.int64))
                                arc_lists['edge'].append(np.array([e_idx[i], e_idx[i]], dtype=np.int32))
                                arc_lists['dir'].append(np.array([-1, 1], dtype=np.int8))
                                boundary_edges_used.add(e_idx[i])
                                connected[idx] = True

                    # Right boundary connections
                    right_candidates = np.where(is_right_boundary & ~connected)[0]
                    if len(right_candidates) > 0:
                        r_r, c_r = dual_rows[right_candidates], dual_cols[right_candidates]
                        e_idx = v_edge_idx[r_r, c_r + 1]
                        valid_e = e_idx >= 0
                        for i, idx in enumerate(right_candidates):
                            if valid_e[i] and e_idx[i] not in boundary_edges_used:
                                arc_lists['start'].append(np.array([boundary_node, idx], dtype=np.int32))
                                arc_lists['end'].append(np.array([idx, boundary_node], dtype=np.int32))
                                arc_lists['capacity'].append(np.array([max_cap, max_cap], dtype=np.int64))
                                arc_lists['cost'].append(np.array([edge_costs[e_idx[i]], edge_costs[e_idx[i]]], dtype=np.int64))
                                arc_lists['edge'].append(np.array([e_idx[i], e_idx[i]], dtype=np.int32))
                                arc_lists['dir'].append(np.array([1, -1], dtype=np.int8))
                                boundary_edges_used.add(e_idx[i])
                                connected[idx] = True

                    # Concatenate all arc arrays
                    if arc_lists['start']:
                        arc_start = np.concatenate(arc_lists['start'])
                        arc_end = np.concatenate(arc_lists['end'])
                        arc_capacity = np.concatenate(arc_lists['capacity'])
                        arc_cost = np.concatenate(arc_lists['cost'])
                        arc_to_edge = np.concatenate(arc_lists['edge'])
                        arc_direction = np.concatenate(arc_lists['dir'])
                    else:
                        arc_start = np.array([], dtype=np.int32)
                        arc_end = np.array([], dtype=np.int32)
                        arc_capacity = np.array([], dtype=np.int64)
                        arc_cost = np.array([], dtype=np.int64)
                        arc_to_edge = np.array([], dtype=np.int32)
                        arc_direction = np.array([], dtype=np.int8)

                    if debug:
                        print(f'    DEBUG: {len(boundary_edges_used)} unique boundary edges connected')
                        print(f'    DEBUG: {len(arc_start)} arcs total')

                    if len(arc_start) == 0:
                        k = np.zeros(n_edges, dtype=np.float64)
                    else:
                        smcf.add_arcs_with_capacity_and_unit_cost(
                            arc_start.astype(np.int32),
                            arc_end.astype(np.int32),
                            arc_capacity.astype(np.int64),
                            arc_cost.astype(np.int64)
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

                        # Extract k values from flow - vectorized where possible
                        k = np.zeros(n_edges, dtype=np.float64)
                        n_arcs = len(arc_start)
                        # Get all flows at once using list comprehension (still need individual calls)
                        flows = np.array([smcf.flow(arc) for arc in range(n_arcs)], dtype=np.int64)
                        # Use numpy scatter-add for efficiency
                        nonzero_mask = flows != 0
                        if np.any(nonzero_mask):
                            np.add.at(k, arc_to_edge[nonzero_mask],
                                     flows[nonzero_mask] * arc_direction[nonzero_mask])

                        if debug:
                            print(f'    DEBUG: {np.count_nonzero(k)} edges with non-zero k')

        # Convert edge-based k values to node-based unwrapped phase
        # The unwrapped phase difference across edge (i,j) is:
        #   unwrapped_diff[e] = phase_diff_wrapped[e] + k[e] = unwrapped[j] - unwrapped[i]
        # We integrate these gradients using BFS on a spanning tree.

        # Memory-optimized BFS using CSR-like sparse adjacency
        # Build arrays: adj_idx[node] gives start index in adj_data for that node's neighbors
        # adj_data stores (neighbor, edge_idx, direction) tuples flattened
        from scipy.sparse import csr_matrix

        # Build adjacency in CSR format for memory efficiency
        # Each edge contributes 2 entries (one for each direction)
        row_idx = np.concatenate([edges[:, 0], edges[:, 1]])
        col_idx = np.concatenate([edges[:, 1], edges[:, 0]])
        edge_indices = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
        directions = np.concatenate([np.ones(n_edges, dtype=np.int8), -np.ones(n_edges, dtype=np.int8)])

        # Sort by row to get CSR-like structure
        sort_idx = np.argsort(row_idx)
        row_sorted = row_idx[sort_idx]
        col_sorted = col_idx[sort_idx]
        edge_sorted = edge_indices[sort_idx]
        dir_sorted = directions[sort_idx]

        # Build indptr (row pointers)
        adj_indptr = np.zeros(n_nodes + 1, dtype=np.int32)
        np.add.at(adj_indptr[1:], row_sorted, 1)
        np.cumsum(adj_indptr, out=adj_indptr)

        if debug:
            has_neighbors = np.sum(adj_indptr[1:] > adj_indptr[:-1])
            print(f'\n    DEBUG: n_nodes={n_nodes}, n_edges={n_edges}, nodes_with_neighbors={has_neighbors}')
            print(f'    DEBUG: n_valid={np.sum(valid_mask)}, k_nonzero={np.count_nonzero(k)}, k_range=[{k.min():.3f}, {k.max():.3f}]')
            print(f'    DEBUG: phase_diff range: [{phase_diff.min():.4f}, {phase_diff.max():.4f}]')
            print(f'    DEBUG: phase_diff_wrapped range: [{phase_diff_wrapped.min():.4f}, {phase_diff_wrapped.max():.4f}]')
            print(f'    DEBUG: round(phase_diff) nonzero: {np.count_nonzero(np.round(phase_diff))}')

        # The unwrapped phase difference across each edge is:
        #   unwrapped_diff[e] = phase_diff_wrapped[e] + k[e]
        unwrapped_diff = phase_diff_wrapped + k

        # BFS to integrate unwrapped phase differences
        unwrapped_cycles = np.full(n_nodes, np.nan, dtype=np.float64)
        valid_indices = np.where(valid_mask)[0]

        for start_node in valid_indices:
            if not np.isnan(unwrapped_cycles[start_node]):
                continue

            queue = deque([start_node])
            unwrapped_cycles[start_node] = phase_flat[start_node]

            while queue:
                node = queue.popleft()
                # Get neighbors from CSR structure
                for idx in range(adj_indptr[node], adj_indptr[node + 1]):
                    neighbor = col_sorted[idx]
                    if np.isnan(unwrapped_cycles[neighbor]):
                        edge_idx = edge_sorted[idx]
                        direction = dir_sorted[idx]
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
                    if err > 0.001:
                        errors.append((e, err, actual_diff, expected_diff, k[e]))
            if errors:
                print(f'    DEBUG: {len(errors)} edges with integration errors > 0.001')
                err_vals = [e[1] for e in errors]
                print(f'    DEBUG: error range: [{min(err_vals):.4f}, {max(err_vals):.4f}], mean: {np.mean(err_vals):.4f}')

            # Verify residues using array-based lookup
            corrected_diff = phase_diff_wrapped + k
            residue_errors = 0
            curls = []
            for r in range(height - 1):
                for c in range(width - 1):
                    # Check all 4 edges exist
                    e1 = h_edge_idx[r, c] if c < width - 1 else -1
                    e2 = v_edge_idx[r, c + 1] if c + 1 < width else -1
                    e3 = h_edge_idx[r + 1, c] if r + 1 < height - 1 and c < width - 1 else -1
                    e4 = v_edge_idx[r, c] if c < width else -1
                    if e1 < 0 or e2 < 0 or e3 < 0 or e4 < 0:
                        continue
                    # Compute curl
                    curl = (corrected_diff[e1] + corrected_diff[e2]
                           - corrected_diff[e3] - corrected_diff[e4])
                    if abs(curl) > 0.001:
                        residue_errors += 1
                        curls.append(curl)
            print(f'    DEBUG: {residue_errors} cells with non-zero curl after k correction')
            if curls:
                print(f'    DEBUG: curl values: [{min(curls):.4f}, {max(curls):.4f}], unique: {sorted(set([round(c) for c in curls]))}')

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

        return unwrapped - np.nanmean(unwrapped)

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

    def _unwrap_2d_maxflow(self, phase_da, weight_da=None, conncomp_flag=False, conncomp_size=100,
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
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 100.
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

        def _unwrap_single(phase_2d, corr_2d=None):
            """Unwrap a single 2D phase array."""
            return Stack_unwrap2d._branch_cut(phase_2d, correlation=corr_2d,
                                             conncomp_size=conncomp_size, max_jump=max_jump,
                                             norm=norm, scale=scale, max_iters=max_iters, debug=debug)

        def _conncomp_single(phase_2d):
            """Compute connected components for a single 2D array."""
            return Stack_unwrap2d._conncomp_2d(phase_2d).astype(np.float32)

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
            return unwrap_da, comp_da

        return unwrap_da, None

    def unwrap2d_maxflow(self, phase, weight=None, conncomp=False, conncomp_size=100,
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
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 100.
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

        Skip small components (less than 500 pixels):
        >>> unwrapped = stack.unwrap_maxflow(intfs, corr, conncomp_size=500)
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
                                                           conncomp_size=conncomp_size,
                                                           max_jump=max_jump, norm=norm,
                                                           scale=scale, max_iters=max_iters, debug=debug)

                unwrap_vars[var] = unwrap_da
                if conncomp and comp_da is not None:
                    comp_vars[var] = comp_da

            # Compute coordinates to avoid lazy BPR in output
            computed_coords = {k: (v.compute() if hasattr(v, 'compute') else v) for k, v in phase_ds.coords.items()}
            unwrap_result[key] = xr.Dataset(unwrap_vars, coords=computed_coords, attrs=phase_ds.attrs)
            if conncomp:
                conncomp_result[key] = xr.Dataset(comp_vars, coords=computed_coords, attrs=phase_ds.attrs)

        # use Batch (not BatchWrap) to avoid re-wrapping the unwrapped phase
        if conncomp:
            return Batch(unwrap_result), BatchUnit(conncomp_result)
        return Batch(unwrap_result)

    def _unwrap_2d_ilp(self, phase_da, weight_da=None, conncomp_flag=False, conncomp_size=100,
                       max_time=300.0, search_workers=1, debug=False):
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
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 100.
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

        def _unwrap_single(phase_2d, corr_2d=None):
            """Unwrap a single 2D phase array using ILP."""
            return Stack_unwrap2d._ilp_unwrap_2d(phase_2d, correlation=corr_2d, conncomp_size=conncomp_size,
                                                max_time=max_time, search_workers=search_workers, debug=debug)

        def _conncomp_single(phase_2d):
            """Compute connected components for a single 2D array."""
            return Stack_unwrap2d._conncomp_2d(phase_2d).astype(np.float32)

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
            return unwrap_da, comp_da

        return unwrap_da, None

    def unwrap2d_ilp(self, phase, weight=None, conncomp=False, conncomp_size=100,
                   max_time=300.0, search_workers=1, debug=False):
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
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 100.
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

        Skip small components (less than 500 pixels):
        >>> unwrapped = stack.unwrap_ilp(intfs, corr, conncomp_size=500)
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

                unwrap_da, comp_da = self._unwrap_2d_ilp(phase_da, weight_da, conncomp,
                                                       conncomp_size=conncomp_size,
                                                       max_time=max_time, search_workers=search_workers, debug=debug)

                unwrap_vars[var] = unwrap_da
                if conncomp and comp_da is not None:
                    comp_vars[var] = comp_da

            # Compute coordinates to avoid lazy BPR in output
            computed_coords = {k: (v.compute() if hasattr(v, 'compute') else v) for k, v in phase_ds.coords.items()}
            unwrap_result[key] = xr.Dataset(unwrap_vars, coords=computed_coords, attrs=phase_ds.attrs)
            if conncomp:
                conncomp_result[key] = xr.Dataset(comp_vars, coords=computed_coords, attrs=phase_ds.attrs)

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
            Whether to compute and return connected components.
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

        def _unwrap_single(phase_2d, corr_2d=None):
            """Unwrap a single 2D phase array using minflow."""
            return Stack_unwrap2d._minflow_unwrap_2d(phase_2d, correlation=corr_2d,
                                                    conncomp_size=conncomp_size, debug=debug)

        def _conncomp_single(phase_2d):
            """Compute connected components for a single 2D array."""
            return Stack_unwrap2d._conncomp_2d(phase_2d).astype(np.float32)

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
            return unwrap_da, comp_da

        return unwrap_da, None

    def unwrap2d_minflow(self, phase, weight=None, conncomp=False, conncomp_size=100, debug=False):
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
            Components smaller than this are left as NaN. Default is 100.
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

        Note: This method unwraps each connected component separately. Components
        are NOT automatically linked - each has an independent phase reference.
        Use unwrap() with conncomp_link=True to automatically link components.

        Examples
        --------
        Unwrap phase:
        >>> unwrapped = stack.unwrap_minflow(intfs)

        Unwrap phase with correlation weighting:
        >>> unwrapped = stack.unwrap_minflow(intfs, corr)

        Return connected component labels:
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

                unwrap_da, comp_da = self._unwrap_2d_minflow(phase_da, weight_da, conncomp,
                                                           conncomp_size=conncomp_size,
                                                           debug=debug)

                unwrap_vars[var] = unwrap_da
                if conncomp and comp_da is not None:
                    comp_vars[var] = comp_da

            # Compute coordinates to avoid lazy BPR in output
            computed_coords = {k: (v.compute() if hasattr(v, 'compute') else v) for k, v in phase_ds.coords.items()}
            unwrap_result[key] = xr.Dataset(unwrap_vars, coords=computed_coords, attrs=phase_ds.attrs)
            if conncomp:
                conncomp_result[key] = xr.Dataset(comp_vars, coords=computed_coords, attrs=phase_ds.attrs)

        # use Batch (not BatchWrap) to avoid re-wrapping the unwrapped phase
        if conncomp:
            return Batch(unwrap_result), BatchUnit(conncomp_result)
        return Batch(unwrap_result)

    def _reorder_conncomp_by_size(self, conncomp_labels):
        """
        Reorder connected component labels by size (largest=1, smallest=max).

        Parameters
        ----------
        conncomp_labels : BatchUnit
            Batch of connected component labels.

        Returns
        -------
        BatchUnit
            Batch with reordered labels (1=largest, 2=second largest, etc.).
        """
        import xarray as xr
        from .Batch import BatchUnit

        def _reorder_2d(labels_2d):
            """Reorder labels in a single 2D array."""
            # Get unique labels (excluding 0 and NaN)
            valid_mask = ~np.isnan(labels_2d) & (labels_2d > 0)
            if not np.any(valid_mask):
                return labels_2d

            unique_labels = np.unique(labels_2d[valid_mask])
            if len(unique_labels) == 0:
                return labels_2d

            # Count pixels per label
            sizes = []
            for label in unique_labels:
                sizes.append(np.sum(labels_2d == label))

            # Sort by size (descending) and create mapping
            sorted_indices = np.argsort(sizes)[::-1]
            label_mapping = {}
            for new_label, idx in enumerate(sorted_indices, start=1):
                old_label = unique_labels[idx]
                label_mapping[old_label] = new_label

            # Apply mapping
            result = np.zeros_like(labels_2d)
            result[~valid_mask] = np.nan
            for old_label, new_label in label_mapping.items():
                result[labels_2d == old_label] = new_label

            return result.astype(np.float32)

        # Process each dataset in the batch
        result = {}
        for key in conncomp_labels.keys():
            ds = conncomp_labels[key]
            data_vars = list(ds.data_vars)

            reordered_vars = {}
            for var in data_vars:
                da = ds[var]

                # Apply reordering to each 2D slice
                reordered_da = xr.apply_ufunc(
                    _reorder_2d,
                    da,
                    input_core_dims=[['y', 'x']],
                    output_core_dims=[['y', 'x']],
                    vectorize=True,
                    dask='parallelized',
                    output_dtypes=[np.float32],
                )

                reordered_vars[var] = reordered_da

            result[key] = xr.Dataset(reordered_vars, coords=ds.coords, attrs=ds.attrs)

        return BatchUnit(result)

    def _link_components(self, unwrapped, conncomp_size=100, conncomp_gap=None,
                         conncomp_linksize=5, conncomp_linkcount=30, debug=False):
        """
        Link disconnected components in unwrapped phase by finding optimal 2π offsets.

        Parameters
        ----------
        unwrapped : Batch
            Batch of unwrapped phase datasets.
        conncomp_size : int
            Minimum component size to process.
        conncomp_gap : int or None
            Maximum pixel distance for connections.
        conncomp_linksize : int
            Number of pixels for offset estimation.
        conncomp_linkcount : int
            Maximum neighbor components to consider.
        debug : bool
            If True, print diagnostic information.

        Returns
        -------
        Batch
            Batch of unwrapped phase with linked components.
        """
        import xarray as xr
        from .Batch import Batch

        def _link_2d(phase_2d):
            """Link components in a single 2D array."""
            import time

            # Find connected components - use efficient labeling
            valid_mask = ~np.isnan(phase_2d)
            if not np.any(valid_mask):
                return phase_2d

            min_size = max(conncomp_size, 4)
            labeled, components, n_total, sizes = Stack_unwrap2d._get_connected_components(valid_mask, min_size)

            if len(components) < 2:
                return phase_2d  # Not enough components to link

            # Create masks for valid components (already sorted by size, largest first)
            processed_components = [(labeled == comp['label']) for comp in components]

            if debug:
                gap_str = 'unlimited' if conncomp_gap is None else str(conncomp_gap)
                print(f'  Linking {len(components)} components (conncomp_gap={gap_str})...')
                t0 = time.time()

            # Find connections
            connections = Stack_unwrap2d._find_component_connections(
                processed_components, conncomp_gap=conncomp_gap, max_neighbors=conncomp_linkcount
            )

            if debug:
                print(f'    Found {len(connections)} connections')

            if len(connections) == 0:
                return phase_2d  # No connections found

            # Apply ILP to find optimal offsets
            result = Stack_unwrap2d._connect_components_ilp(
                phase_2d, processed_components, connections,
                n_neighbors=conncomp_linksize, max_time=60.0, debug=debug
            )

            if debug:
                elapsed = time.time() - t0
                print(f'  Component linking done ({elapsed:.2f}s)')

            return result

        # Process each dataset in the batch
        result = {}
        for key in unwrapped.keys():
            ds = unwrapped[key]
            data_vars = list(ds.data_vars)

            linked_vars = {}
            for var in data_vars:
                da = ds[var]
                stackvar = da.dims[0]  # 'pair'

                # Apply linking to each 2D slice
                linked_da = xr.apply_ufunc(
                    _link_2d,
                    da,
                    input_core_dims=[['y', 'x']],
                    output_core_dims=[['y', 'x']],
                    vectorize=True,
                    dask='parallelized',
                    output_dtypes=[np.float32],
                )

                linked_vars[var] = linked_da

            result[key] = xr.Dataset(linked_vars, coords=ds.coords, attrs=ds.attrs)

        return Batch(result)

    def unwrap2d(self, phase, weight=None, conncomp=False, method='maxflow',
                conncomp_size=100, conncomp_gap=None,
                conncomp_linksize=5, conncomp_linkcount=30, debug=False, **kwargs):
        """
        Unwrap phase using the specified method.

        This is a convenience wrapper that dispatches to unwrap_maxflow(),
        unwrap_minflow(), or unwrap_ilp() based on the method parameter.

        When conncomp=False (default), disconnected components are automatically
        linked using ILP optimization to find optimal 2π offsets.

        When conncomp=True, components are kept separate and returned with
        size-ordered labels (1=largest, 2=second largest, etc.).

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting.
        conncomp : bool, optional
            If False (default), link disconnected components using ILP to find
            optimal 2π offsets, returning a single merged result.
            If True, keep components separate and return conncomp labels
            (1=largest component, 2=second largest, etc., 0=invalid).
        method : str, optional
            Unwrapping method to use. Options are:
            - 'maxflow': Branch-cut algorithm with max-flow optimization (default).
            - 'minflow': Minimum Cost Flow (Costantini algorithm) - scalable.
            - 'ilp': Integer Linear Programming - optimal but slow.
            - 'dct': DCT-based L² solver - very fast, GPU-accelerated, no weighting.
            - 'irls': IRLS L¹ solver - fast, GPU-accelerated, supports weighting.
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 100.
        conncomp_gap : int or None, optional
            Maximum pixel distance between components to consider them connectable.
            If None (default), no distance limit - all direct connections are used.
            Only used when conncomp=False.
        conncomp_linksize : int, optional
            Number of pixels to use on each side of a connection point for
            estimating the phase offset between components. Uses median for
            robustness - 5 pixels is sufficient to tolerate 2 outliers (40%).
            Default is 5. Only used when conncomp=False.
        conncomp_linkcount : int, optional
            Maximum number of nearest neighbor components to consider for
            connections from each component. Higher values find more potential
            connections but increase computation. Default is 30.
            Only used when conncomp=False.
        debug : bool, optional
            If True, print diagnostic information. Default is False.
        **kwargs
            Additional arguments passed to the underlying method.
            For 'maxflow': max_jump, norm, scale, max_iters.
            For 'ilp': max_time, search_workers.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase (components linked).
            If conncomp is True: tuple of (Batch unwrapped phase, BatchUnit conncomp)
            where conncomp labels are ordered by size (1=largest).

        Notes
        -----
        Method comparison:

        ========  =======  ===========  =====================================
        Method    Speed    Quality      Best Use Case
        ========  =======  ===========  =====================================
        maxflow   Fast     Best         General use, default
        minflow   Medium   Good         Large grids, CPU-only environments
        ilp       Slow     Optimal      Small grids (<200×200), when global
                                        optimum needed
        dct       Fastest  Fair (L²)    Quick estimation, no weighting needed
        irls      Fast     Good (L¹)    GPU-accelerated, with weighting
        ========  =======  ===========  =====================================

        GPU methods (dct, irls) automatically use:
        - MPS on Apple Silicon (M1/M2/M3/M4)
        - CUDA on NVIDIA GPUs
        - CPU fallback otherwise

        Component Linking (when conncomp=False):
        1. Unwraps each connected component separately
        2. Finds direct connections between components (not crossing others)
        3. Estimates phase offsets using conncomp_linksize pixels per connection
        4. Uses ILP to find globally optimal integer 2π offsets

        Examples
        --------
        Unwrap phase with component linking (default):
        >>> unwrapped = stack.unwrap(intfs, corr)

        Unwrap with minflow method:
        >>> unwrapped = stack.unwrap(intfs, corr, method='minflow')

        Keep components separate (no linking), get labels:
        >>> unwrapped, conncomp = stack.unwrap(intfs, corr, conncomp=True)
        >>> main_component = unwrapped.where(conncomp == 1)  # largest component

        Unwrap with ILP method (optimal but slow):
        >>> unwrapped = stack.unwrap(intfs, corr, method='ilp', max_time=3600)
        """
        # Validate parameters
        if not conncomp and conncomp_linksize > conncomp_size:
            raise ValueError(
                f'conncomp_linksize ({conncomp_linksize}) cannot be greater than conncomp_size ({conncomp_size}). '
                f'Components must have at least conncomp_linksize pixels for reliable offset estimation.'
            )

        # Call the appropriate unwrapping method (always get conncomp for internal use)
        if method == 'maxflow':
            unwrapped, conncomp_labels = self.unwrap2d_maxflow(phase, weight, conncomp=True,
                                                               conncomp_size=conncomp_size, debug=debug, **kwargs)
        elif method == 'minflow':
            unwrapped, conncomp_labels = self.unwrap2d_minflow(phase, weight, conncomp=True,
                                                               conncomp_size=conncomp_size, debug=debug, **kwargs)
        elif method == 'ilp':
            unwrapped, conncomp_labels = self.unwrap2d_ilp(phase, weight, conncomp=True,
                                                            conncomp_size=conncomp_size, debug=debug, **kwargs)
        elif method == 'dct':
            # DCT method doesn't support weighting
            if weight is not None:
                import warnings
                warnings.warn("DCT method doesn't support weighting. Use 'irls' for weighted unwrapping.", UserWarning)
            unwrapped, conncomp_labels = self.unwrap2d_dct(phase, conncomp=True,
                                                           conncomp_size=conncomp_size, debug=debug)
        elif method == 'irls':
            unwrapped, conncomp_labels = self.unwrap2d_irls(phase, weight, conncomp=True,
                                                            conncomp_size=conncomp_size, debug=debug, **kwargs)
        else:
            raise ValueError(f"Unknown unwrapping method: '{method}'. Use 'maxflow', 'minflow', 'ilp', 'dct', or 'irls'.")

        if conncomp:
            # Return separate components with size-ordered labels
            conncomp_labels = self._reorder_conncomp_by_size(conncomp_labels)
            return unwrapped, conncomp_labels
        else:
            # Link components
            unwrapped = self._link_components(
                unwrapped, conncomp_size=conncomp_size, conncomp_gap=conncomp_gap,
                conncomp_linksize=conncomp_linksize, conncomp_linkcount=conncomp_linkcount,
                debug=debug
            )
            return unwrapped

    # =========================================================================
    # GPU-Accelerated Phase Unwrapping Methods (PyTorch)
    # =========================================================================

    @staticmethod
    def _dct_unwrap_2d_single(phase, valid, device, dtype):
        """
        Single-scale DCT unwrapping (internal helper).

        Parameters
        ----------
        phase : torch.Tensor
            2D tensor of phase values (NaN regions filled with 0).
        valid : torch.Tensor
            2D boolean tensor indicating valid pixels.
        device : torch.device
            PyTorch device.
        dtype : torch.dtype
            Data type for computation.

        Returns
        -------
        torch.Tensor
            2D tensor of unwrapped phase.
        """
        import torch
        from torch_dct import dct_2d, idct_2d

        height, width = phase.shape

        # Compute wrapped phase differences (gradients)
        dx = torch.zeros_like(phase)
        dy = torch.zeros_like(phase)

        dx[:, :-1] = phase[:, 1:] - phase[:, :-1]
        dy[:-1, :] = phase[1:, :] - phase[:-1, :]

        # Wrap to [-π, π]
        dx = torch.atan2(torch.sin(dx), torch.cos(dx))
        dy = torch.atan2(torch.sin(dy), torch.cos(dy))

        # Zero out gradients at invalid edges
        dx[:, :-1] *= (valid[:, :-1] & valid[:, 1:]).to(dtype)
        dy[:-1, :] *= (valid[:-1, :] & valid[1:, :]).to(dtype)

        # Compute divergence (rho = div(gradient))
        rho = torch.zeros_like(phase)
        rho[:, 1:] += dx[:, :-1]
        rho[:, :-1] -= dx[:, :-1]
        rho[1:, :] += dy[:-1, :]
        rho[:-1, :] -= dy[:-1, :]

        # Solve Poisson equation using DCT
        rho_dct = dct_2d(rho)

        # Eigenvalues of Laplacian with Neumann boundary conditions
        i_idx = torch.arange(height, dtype=dtype, device=device)
        j_idx = torch.arange(width, dtype=dtype, device=device)
        cos_i = torch.cos(torch.pi * i_idx / height)
        cos_j = torch.cos(torch.pi * j_idx / width)
        eigenvalues = 2 * cos_i.unsqueeze(1) + 2 * cos_j.unsqueeze(0) - 4
        eigenvalues[0, 0] = 1.0  # Avoid division by zero

        # Divide in DCT domain
        phi_dct = rho_dct / (-eigenvalues + 1e-10)
        phi_dct[0, 0] = 0.0  # DC = 0

        result = idct_2d(phi_dct)

        # Re-center result to improve float32 precision (keeps values near zero)
        if valid.any():
            result = result - result[valid].mean()

        return result

    @staticmethod
    def _dct_unwrap_2d(phase, device=None, debug=False):
        """
        Unwrap 2D phase using GPU-accelerated DCT-based Poisson solver (unweighted L² norm).

        This is a fast, single-pass algorithm that solves:
            ∇²φ = ∇·(wrap(∇ψ))

        where ψ is the wrapped phase and φ is the unwrapped phase.
        Based on Ghiglia & Romero (1994) DCT method. GPU-accelerated using
        PyTorch (MPS on Apple Silicon, CUDA on NVIDIA, or CPU fallback).

        This method is also used internally as the initial solution for IRLS,
        providing a good starting point for weighted iterative refinement.

        Parameters
        ----------
        phase : np.ndarray
            2D array of wrapped phase values in radians.
        device : torch.device, optional
            PyTorch device to use. If None, auto-detects best device.
        debug : bool, optional
            If True, print diagnostic information.

        Returns
        -------
        np.ndarray
            2D array of unwrapped phase values.

        Notes
        -----
        **Important limitation**: Unweighted DCT does NOT consider phase
        residues (inconsistencies). Each residue acts as a dipole error source
        that propagates globally through the L² solution. Different residue
        configurations in different images lead to different error patterns,
        causing inconsistency between bursts.

        For data with residues (typical InSAR), use IRLS which uses this DCT
        solution as initialization and then refines it with weighted
        least-squares iterations using correlation-based weighting.

        References
        ----------
        Ghiglia, D.C. & Romero, L.A. (1994). "Robust two-dimensional weighted
        and unweighted phase unwrapping that uses fast transforms and iterative
        methods." J. Opt. Soc. Am. A, 11(1), 107-117.
        """
        import torch

        # Validate and set device
        if device is None or device == 'auto':
            # Auto-select best available device
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        # Check GPU availability for explicit device requests
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but not available. "
                f"Use device='cpu' or device='auto' instead."
            )
        if device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError(
                f"MPS device requested but not available. "
                f"Use device='cpu' or device='auto' instead."
            )

        height, width = phase.shape

        # Use float32 for all devices - sufficient for phase unwrapping
        dtype = torch.float32
        np_dtype = np.float32

        # Handle NaN values
        nan_mask = np.isnan(phase)
        if np.all(nan_mask):
            return np.full_like(phase, np.nan)

        # Fill NaN with 0 for computation
        phase_filled = np.where(nan_mask, 0.0, phase)

        # Convert to torch
        phi = torch.from_numpy(phase_filled.astype(np_dtype)).to(device)
        valid = torch.from_numpy(~nan_mask).to(device)

        # Single-scale DCT unwrap
        unwrapped = Stack_unwrap2d._dct_unwrap_2d_single(phi, valid, device, dtype)

        # Convert back to numpy
        unwrapped = unwrapped.cpu().numpy().astype(np.float32)

        if debug:
            print(f'    DCT raw output range: [{np.nanmin(unwrapped):.2f}, {np.nanmax(unwrapped):.2f}]')

        # Restore NaN values
        unwrapped[nan_mask] = np.nan

        # Apply k_median correction to bring unwrapped phase close to wrapped phase
        valid_mask = ~nan_mask
        if np.any(valid_mask):
            diff = unwrapped[valid_mask] - phase[valid_mask]
            k_values = np.round(diff / (2 * np.pi))
            k_median = np.median(k_values)
            unwrapped[valid_mask] = unwrapped[valid_mask] - k_median * 2 * np.pi

        if debug and np.any(valid_mask):
            # Check wrapping consistency
            rewrapped = np.arctan2(np.sin(unwrapped[valid_mask]), np.cos(unwrapped[valid_mask]))
            wrap_error = np.abs(rewrapped - phase[valid_mask])
            wrap_error = np.minimum(wrap_error, 2*np.pi - wrap_error)
            print(f'    Wrap consistency: mean_err={np.mean(wrap_error):.4f}, max_err={np.max(wrap_error):.4f}')
            print(f'    DCT final range: [{np.nanmin(unwrapped):.2f}, {np.nanmax(unwrapped):.2f}]')

        return unwrapped

    @staticmethod
    def _irls_unwrap_2d(phase, weight=None, device=None, max_iter=10, tol=1e-2,
                        cg_max_iter=10, cg_tol=1e-3, epsilon=1e-2, debug=False):
        """
        Unwrap 2D phase using GPU-accelerated Iteratively Reweighted Least Squares (L¹ norm).

        This algorithm solves the L¹ phase unwrapping problem:
            min Σ w_ij |∇φ_ij - wrap(∇ψ_ij)|

        by iteratively solving weighted L² problems using preconditioned
        conjugate gradient. GPU-accelerated using PyTorch (MPS on Apple Silicon,
        CUDA on NVIDIA, or CPU fallback).

        Parameters
        ----------
        phase : np.ndarray
            2D array of wrapped phase values in radians.
        weight : np.ndarray, optional
            2D array of quality weights (e.g., correlation). Higher values
            indicate more reliable phase. If None, uniform weights are used.
        device : torch.device, optional
            PyTorch device to use. If None, auto-detects best device.
        max_iter : int, optional
            Maximum IRLS iterations. Default is 10.
        tol : float, optional
            Convergence tolerance for relative change in solution. Default is 1e-2.
        cg_max_iter : int, optional
            Maximum conjugate gradient iterations per IRLS step. Default is 10.
        cg_tol : float, optional
            Conjugate gradient convergence tolerance. Default is 1e-3.
        epsilon : float, optional
            Smoothing parameter for L¹ approximation. Default is 1e-2.
        debug : bool, optional
            If True, print convergence information. Default is False.

        Returns
        -------
        np.ndarray
            2D array of unwrapped phase values.

        Notes
        -----
        The algorithm uses GPU-accelerated DCT as initial solution (same as
        standalone DCT method), then refines it through weighted iterations.
        The DCT initialization provides a good starting point, and the IRLS
        iterations correct for residue-induced errors using correlation weights.

        Based on: Dubois-Taine et al., "Iteratively Reweighted Least Squares
        for Phase Unwrapping", arXiv:2401.09961 (2024).

        Achieves 10-20x speedup over SNAPHU on GPU.
        """
        import torch
        from torch_dct import dct_2d, idct_2d
        import time

        # Validate and set device
        if device is None or device == 'auto':
            # Auto-select best available device
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        # Check GPU availability for explicit device requests
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but not available. "
                f"Use device='cpu' or device='auto' instead."
            )
        if device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError(
                f"MPS device requested but not available. "
                f"Use device='cpu' or device='auto' instead."
            )

        device_name = str(device)
        height, width = phase.shape

        # Use float32 for all devices - sufficient for phase unwrapping
        dtype = torch.float32
        np_dtype = np.float32

        _t_start = time.time()

        # Handle NaN values
        nan_mask = np.isnan(phase)
        if np.all(nan_mask):
            return np.full_like(phase, np.nan)

        # Create valid mask for computation
        valid_mask = ~nan_mask

        # Fill NaN with 0 for computation
        phase_filled = np.where(nan_mask, 0.0, phase)

        # Prepare weights
        if weight is not None:
            # Handle NaN in weights (both from phase nan_mask AND from weight itself)
            weight_nan_mask = np.isnan(weight)
            weight_filled = np.where(nan_mask | weight_nan_mask, 0.0, weight)
            weight_filled = np.clip(weight_filled, 0.01, 1.0)  # Avoid zero weights
        else:
            weight_filled = np.where(nan_mask, 0.0, 1.0)

        # Convert to torch tensors
        phi = torch.from_numpy(phase_filled.astype(np_dtype)).to(device)
        w = torch.from_numpy(weight_filled.astype(np_dtype)).to(device)
        valid = torch.from_numpy(valid_mask).to(device)

        # Compute wrapped phase differences (target gradients)
        dx_target = torch.zeros_like(phi)
        dy_target = torch.zeros_like(phi)
        dx_target[:, :-1] = phi[:, 1:] - phi[:, :-1]
        dy_target[:-1, :] = phi[1:, :] - phi[:-1, :]

        # Wrap to [-π, π]
        dx_target = torch.atan2(torch.sin(dx_target), torch.cos(dx_target))
        dy_target = torch.atan2(torch.sin(dy_target), torch.cos(dy_target))

        # Edge weights (average of adjacent pixel weights)
        wx = torch.zeros_like(w)
        wy = torch.zeros_like(w)
        wx[:, :-1] = (w[:, :-1] + w[:, 1:]) / 2
        wy[:-1, :] = (w[:-1, :] + w[1:, :]) / 2

        # Zero out weights at invalid edges
        wx[:, :-1] *= (valid[:, :-1] & valid[:, 1:]).to(dtype)
        wy[:-1, :] *= (valid[:-1, :] & valid[1:, :]).to(dtype)

        # Precompute DCT eigenvalues for preconditioner
        # IMPORTANT: Trigonometric functions require float64 precision!
        # Small errors in cos() accumulate through thousands of preconditioner
        # applications in the CG solver. Using float32 on GPU for eigenvalues
        # causes residuals to degrade from ~0.4 rad to ~1.4 rad at 6400x6400.
        # Compute on CPU with float64, then convert to float32 and transfer.
        i_idx = torch.arange(height, dtype=torch.float64)
        j_idx = torch.arange(width, dtype=torch.float64)
        cos_i = torch.cos(torch.pi * i_idx / height)
        cos_j = torch.cos(torch.pi * j_idx / width)
        eigenvalues = (2 * cos_i.unsqueeze(1) + 2 * cos_j.unsqueeze(0) - 4).to(dtype).to(device)
        eigenvalues[0, 0] = 1.0  # Avoid division by zero

        # Pre-allocate buffers for gradient computation (avoid repeated allocation)
        _dx_buf = torch.zeros_like(phi)
        _dy_buf = torch.zeros_like(phi)
        _result_buf = torch.zeros_like(phi)

        def apply_laplacian(x, wx_irls, wy_irls):
            """Apply weighted Laplacian operator: -∇·(w·∇x)"""
            # Forward differences (reuse buffers)
            _dx_buf.zero_()
            _dy_buf.zero_()
            _dx_buf[:, :-1] = x[:, 1:] - x[:, :-1]
            _dy_buf[:-1, :] = x[1:, :] - x[:-1, :]

            # Weight the gradients in-place
            _dx_buf.mul_(wx_irls)
            _dy_buf.mul_(wy_irls)

            # Backward differences (divergence)
            _result_buf.zero_()
            _result_buf[:, 1:].sub_(_dx_buf[:, :-1])
            _result_buf[:, :-1].add_(_dx_buf[:, :-1])
            _result_buf[1:, :].sub_(_dy_buf[:-1, :])
            _result_buf[:-1, :].add_(_dy_buf[:-1, :])

            return _result_buf.neg()

        def apply_preconditioner(r):
            """Apply DCT-based preconditioner (approximate inverse Laplacian)"""
            r_dct = dct_2d(r)
            r_dct.div_(-eigenvalues + 1e-10)
            r_dct[0, 0] = 0.0
            return idct_2d(r_dct)

        def conjugate_gradient(b, wx_irls, wy_irls, x0, max_iter_cg, tol_cg):
            """Preconditioned conjugate gradient solver for IRLS."""
            x = x0  # No clone - modify in place, caller provides buffer
            r = b - apply_laplacian(x, wx_irls, wy_irls)

            # Check for NaN in initial residual
            if not torch.isfinite(r).all():
                return x0

            z = apply_preconditioner(r)
            p = z.clone()  # Need clone here - p gets modified
            rz = torch.sum(r * z)

            for i in range(max_iter_cg):
                Ap = apply_laplacian(p, wx_irls, wy_irls).clone()  # Need clone - buffer reused
                pAp = torch.sum(p * Ap)
                if pAp.abs() < 1e-15 or not torch.isfinite(pAp):
                    break
                alpha = rz / pAp

                # Clamp alpha to prevent explosion
                alpha = torch.clamp(alpha, -1e6, 1e6)
                alpha_val = alpha.item()

                # Update x
                x.add_(p, alpha=alpha_val)

                r.sub_(Ap, alpha=alpha_val)

                # Check for numerical issues mid-iteration
                if not torch.isfinite(x).all():
                    break

                r_norm = torch.sqrt(torch.sum(r * r))
                if r_norm < tol_cg or not torch.isfinite(r_norm):
                    break

                z = apply_preconditioner(r)
                rz_new = torch.sum(r * z)
                if rz.abs() < 1e-15:
                    break
                beta = rz_new / rz
                beta = torch.clamp(beta, -1e6, 1e6)
                p.mul_(beta.item()).add_(z)
                rz = rz_new

            return x

        # Initialize with DCT solution (L² result)
        rho = torch.zeros_like(phi)
        rho[:, 1:] += wx[:, :-1] * dx_target[:, :-1]
        rho[:, :-1] -= wx[:, :-1] * dx_target[:, :-1]
        rho[1:, :] += wy[:-1, :] * dy_target[:-1, :]
        rho[:-1, :] -= wy[:-1, :] * dy_target[:-1, :]

        rho_dct = dct_2d(rho)
        u_dct = rho_dct / (-eigenvalues + 1e-10)
        u_dct[0, 0] = 0.0
        u = idct_2d(u_dct)

        # Check DCT initialization for NaN/inf
        if not torch.isfinite(u).all():
            if debug:
                nan_count = (~torch.isfinite(u)).sum().item()
                print(f'  DCT init produced {nan_count} NaN/inf values, filling with zeros')
            u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))

        # Re-center DCT result to improve float32 precision
        # This keeps values near zero where float32 has best precision
        if valid.any():
            u_mean = u[valid].mean()
            u.sub_(u_mean)

        _t_init = time.time()

        if debug:
            # Handle case where weights might still have issues
            w_valid = w[valid]
            if w_valid.numel() > 0:
                w_min, w_max = w_valid.min().item(), w_valid.max().item()
            else:
                w_min, w_max = 0.0, 0.0
            print(f'  Input: {height}x{width}, valid pixels: {valid_mask.sum().item()}, '
                  f'weight range: [{w_min:.3f}, {w_max:.3f}]')
            print(f'  DCT init range: [{u.min().item():.2f}, {u.max().item():.2f}]')

        # IRLS iterations - keep track of last good solution
        u_best = u.clone()
        best_residual = float('inf')

        # Pre-allocate buffers for IRLS loop
        dx_u = torch.zeros_like(u)
        dy_u = torch.zeros_like(u)
        rx = torch.zeros_like(u)
        ry = torch.zeros_like(u)
        wx_irls = torch.zeros_like(u)
        wy_irls = torch.zeros_like(u)
        b = torch.zeros_like(u)
        u_prev = torch.zeros_like(u)
        eps_sq = epsilon * epsilon

        for iteration in range(max_iter):
            u_prev.copy_(u)

            # Re-center u to prevent numerical drift (important for float32)
            # Phase unwrapping only cares about gradients, so mean is arbitrary
            if valid.any():
                u_mean = u[valid].mean()
                u.sub_(u_mean)

            # Compute current gradients (reuse buffers)
            dx_u.zero_()
            dy_u.zero_()
            dx_u[:, :-1] = u[:, 1:] - u[:, :-1]
            dy_u[:-1, :] = u[1:, :] - u[:-1, :]

            # Compute residuals in-place
            torch.sub(dx_u, dx_target, out=rx)
            torch.sub(dy_u, dy_target, out=ry)

            # Track best solution by residual magnitude
            current_residual = (torch.sum(rx * rx) + torch.sum(ry * ry)).item()
            if current_residual < best_residual and torch.isfinite(u).all():
                best_residual = current_residual
                u_best.copy_(u)

            # Update IRLS weights: w_irls = w / sqrt(r² + ε²)
            # In-place operations
            torch.addcmul(torch.full_like(rx, eps_sq), rx, rx, out=wx_irls)
            wx_irls.sqrt_()
            torch.div(wx, wx_irls, out=wx_irls)

            torch.addcmul(torch.full_like(ry, eps_sq), ry, ry, out=wy_irls)
            wy_irls.sqrt_()
            torch.div(wy, wy_irls, out=wy_irls)

            # Clamp weights in-place
            wx_irls.clamp_(min=1e-6, max=1e6)
            wy_irls.clamp_(min=1e-6, max=1e6)

            # Compute right-hand side: -∇·(w_irls · ∇target)
            b.zero_()
            b[:, 1:].addcmul_(wx_irls[:, :-1], dx_target[:, :-1])
            b[:, :-1].addcmul_(wx_irls[:, :-1], dx_target[:, :-1], value=-1)
            b[1:, :].addcmul_(wy_irls[:-1, :], dy_target[:-1, :])
            b[:-1, :].addcmul_(wy_irls[:-1, :], dy_target[:-1, :], value=-1)

            # Solve weighted Laplacian system using CG
            u = conjugate_gradient(b, wx_irls, wy_irls, u, cg_max_iter, cg_tol)

            # Check for numerical issues during iteration
            if not torch.isfinite(u).all():
                if debug:
                    print(f'  IRLS iter {iteration}: NaN/inf detected, reverting to best solution')
                u = u_best.clone()
                break

            # Check convergence (in-place computation)
            diff = torch.norm(u - u_prev)
            norm_u = torch.norm(u) + 1e-10
            rel_change = (diff / norm_u).item()

            if debug and iteration % 5 == 0:
                print(f'  IRLS iter {iteration}: rel_change = {rel_change:.2e}, residual = {current_residual:.2e}')

            if rel_change < tol:
                if debug:
                    print(f'  IRLS converged at iteration {iteration}')
                break

        # Use best solution found during iterations if current is invalid
        if not torch.isfinite(u).all():
            if debug:
                print(f'  Final solution has NaN/inf, reverting to best solution')
            u = u_best

        _t_end = time.time()

        # Convert back to numpy
        unwrapped = u.cpu().numpy().astype(np.float32)

        # Check for NaN/inf from numerical issues - return NaN array (no hidden fallback)
        if not np.isfinite(unwrapped[~nan_mask]).all():
            import warnings
            nan_count = np.sum(~np.isfinite(unwrapped[~nan_mask]))
            total_valid = np.sum(~nan_mask)
            warnings.warn(f'IRLS produced {nan_count}/{total_valid} NaN/inf values - returning NaN for this component', RuntimeWarning)
            return np.full_like(phase, np.nan, dtype=np.float32)

        # Restore NaN values
        unwrapped[nan_mask] = np.nan

        # Validate and correct: rewrapped phase should match input
        valid_mask = ~nan_mask
        if np.any(valid_mask):
            diff = unwrapped[valid_mask] - phase[valid_mask]
            k_values = np.round(diff / (2 * np.pi))
            k_median = np.median(k_values)
            unwrapped[valid_mask] = unwrapped[valid_mask] - k_median * 2 * np.pi

        if debug:
            print(f'TIMING irls_unwrap_2d ({height}x{width}) on {device_name}:')
            print(f'  init (DCT):   {(_t_init - _t_start)*1000:.1f} ms')
            print(f'  IRLS iters:   {(_t_end - _t_init)*1000:.1f} ms ({iteration+1} iterations)')
            print(f'  TOTAL:        {(_t_end - _t_start)*1000:.1f} ms')

        return unwrapped

    def unwrap2d_irls(self, phase, weight=None, conncomp=False, conncomp_size=100, device='auto',
                      max_iter=10, tol=1e-2, cg_max_iter=10, cg_tol=1e-3, epsilon=1e-2, debug=False):
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L¹ norm).

        This algorithm provides high-quality unwrapping with L¹ norm that
        preserves discontinuities, and supports quality weighting from
        correlation data. GPU-accelerated using PyTorch (MPS on Apple Silicon,
        CUDA on NVIDIA, or CPU fallback).

        Uses GPU-accelerated DCT as initial solution, then refines it through
        weighted IRLS iterations. This handles phase residues properly by
        down-weighting inconsistent regions based on correlation.

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting. Higher values indicate
            more reliable phase measurements.
        conncomp : bool, optional
            If True, also return connected components. Default is False.
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 100.
        device : str, optional
            Device for computation: 'auto' (default), 'cpu', 'cuda', or 'mps'.
            'auto' selects the best available GPU, falling back to CPU.
        max_iter : int, optional
            Maximum IRLS iterations. Default is 10.
        tol : float, optional
            Convergence tolerance for relative change. Default is 1e-2.
        cg_max_iter : int, optional
            Maximum conjugate gradient iterations per IRLS step. Default is 10.
        cg_tol : float, optional
            Conjugate gradient convergence tolerance. Default is 1e-3.
        epsilon : float, optional
            Smoothing parameter for L¹ approximation. Larger values improve
            numerical stability but reduce L¹ approximation quality. Default is 1e-2.
        debug : bool, optional
            If True, print diagnostic information. Default is False.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase.
            If conncomp is True: tuple of (Batch unwrapped phase, BatchUnit conncomp).

        Notes
        -----
        - GPU-accelerated using PyTorch (MPS on Apple Silicon, CUDA, or CPU)
        - Uses GPU-accelerated DCT for fast initialization
        - L¹ norm preserves discontinuities better than L² (DCT)
        - Correlation weighting handles phase residues properly
        - Provides consistent results across multi-burst data
        - Based on arXiv:2401.09961

        See Also
        --------
        unwrap2d_minflow : CPU-based L¹ unwrapping using min-cost flow
        """
        import torch
        import xarray as xr
        from .Batch import Batch, BatchWrap, BatchUnit

        assert isinstance(phase, BatchWrap), 'ERROR: phase should be a BatchWrap object'
        assert weight is None or isinstance(weight, BatchUnit), 'ERROR: weight should be a BatchUnit object'

        # Resolve device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif torch.backends.mps.is_available():
                device = 'mps'
                device_name = "MPS (Apple Silicon GPU)"
            else:
                device = 'cpu'
                device_name = "CPU"
        else:
            device_name = device.upper()

        if debug:
            print(f'Using device: {device_name}')

        # Process each burst in the batch
        unwrap_result = {}
        conncomp_result = {}

        burst_idx = 0
        for key in phase.keys():
            phase_ds = phase[key]
            weight_ds = weight[key] if weight is not None and key in weight else None

            if debug:
                print(f'\nProcessing burst {burst_idx}: {key}')
            burst_idx += 1

            # Get data variables (typically polarization like 'VV')
            data_vars = list(phase_ds.data_vars)

            unwrap_vars = {}
            comp_vars = {}

            for var in data_vars:
                phase_da = phase_ds[var]
                weight_da = weight_ds[var] if weight_ds is not None and var in weight_ds else None

                # Process each pair in the stack
                if 'pair' in phase_da.dims:
                    results = []
                    comp_results = []
                    for i in range(phase_da.sizes['pair']):
                        phase_slice = phase_da.isel(pair=i).values
                        weight_slice = weight_da.isel(pair=i).values if weight_da is not None else None
                        unwrapped, labels = self._process_irls_slice(
                            phase_slice, weight_slice, device, conncomp_size,
                            max_iter, tol, cg_max_iter, cg_tol, epsilon, debug
                        )
                        results.append(unwrapped)
                        comp_results.append(labels)

                    result_data = np.stack(results, axis=0)
                    result_da = xr.DataArray(
                        result_data,
                        dims=phase_da.dims,
                        coords=phase_da.coords,
                        attrs={'units': 'radians'}
                    )
                    comp_data = np.stack(comp_results, axis=0)
                    comp_da = xr.DataArray(
                        comp_data,
                        dims=phase_da.dims,
                        coords=phase_da.coords
                    )
                else:
                    phase_np = phase_da.values
                    weight_np = weight_da.values if weight_da is not None else None
                    unwrapped, labels = self._process_irls_slice(
                        phase_np, weight_np, device, conncomp_size,
                        max_iter, tol, cg_max_iter, cg_tol, epsilon, debug
                    )
                    result_da = xr.DataArray(
                        unwrapped,
                        dims=phase_da.dims,
                        coords=phase_da.coords,
                        attrs={'units': 'radians'}
                    )
                    comp_da = xr.DataArray(
                        labels,
                        dims=phase_da.dims,
                        coords=phase_da.coords
                    )

                unwrap_vars[var] = result_da
                comp_vars[var] = comp_da

            # Preserve dataset attributes (subswath, pathNumber, etc.)
            unwrap_result[key] = xr.Dataset(unwrap_vars, attrs=phase_ds.attrs)
            conncomp_result[key] = xr.Dataset(comp_vars, attrs=phase_ds.attrs)

        output = Batch(unwrap_result)

        if conncomp:
            return output, BatchUnit(conncomp_result)
        return output

    def _process_irls_slice(self, phase_np, weight_np, device, conncomp_size,
                            max_iter, tol, cg_max_iter, cg_tol, epsilon, debug):
        """Process a single 2D phase slice with IRLS unwrapping."""
        if debug:
            print(f'  Slice shape: {phase_np.shape}, '
                  f'valid: {np.sum(~np.isnan(phase_np))}, '
                  f'phase range: [{np.nanmin(phase_np):.3f}, {np.nanmax(phase_np):.3f}]')
            if weight_np is not None:
                print(f'  Weight range: [{np.nanmin(weight_np):.3f}, {np.nanmax(weight_np):.3f}]')

        if np.all(np.isnan(phase_np)):
            if debug:
                print('  All NaN, skipping')
            return phase_np.astype(np.float32), np.zeros_like(phase_np, dtype=np.int32)

        # Get connected components
        labels = self._conncomp_2d(phase_np)
        unique_labels = np.unique(labels[labels > 0])

        if debug:
            print(f'  Connected components: {len(unique_labels)}')

        # Process each component
        result = np.full_like(phase_np, np.nan, dtype=np.float32)

        for label in unique_labels:
            mask = labels == label
            comp_size = np.sum(mask)
            if comp_size < conncomp_size:
                continue

            # Extract component bounding box for efficiency
            rows, cols = np.where(mask)
            r0, r1 = rows.min(), rows.max() + 1
            c0, c1 = cols.min(), cols.max() + 1

            if debug:
                print(f'  Component {label}: size={comp_size}, '
                      f'bbox=[{r0}:{r1}, {c0}:{c1}] ({r1-r0}x{c1-c0})')

            phase_crop = phase_np[r0:r1, c0:c1].copy()
            mask_crop = mask[r0:r1, c0:c1]
            phase_crop[~mask_crop] = np.nan

            weight_crop = None
            if weight_np is not None:
                weight_crop = weight_np[r0:r1, c0:c1].copy()
                weight_crop[~mask_crop] = np.nan

            # Unwrap using IRLS
            unwrapped_crop = self._irls_unwrap_2d(
                phase_crop, weight=weight_crop, device=device,
                max_iter=max_iter, tol=tol, cg_max_iter=cg_max_iter,
                cg_tol=cg_tol, epsilon=epsilon, debug=debug
            )

            # Check result
            valid_in_crop = ~np.isnan(phase_crop)
            nan_in_result = np.sum(np.isnan(unwrapped_crop[valid_in_crop]))
            if debug and nan_in_result > 0:
                print(f'    WARNING: {nan_in_result}/{np.sum(valid_in_crop)} NaN in unwrapped result')

            # Place back
            result[r0:r1, c0:c1] = np.where(mask_crop, unwrapped_crop, result[r0:r1, c0:c1])

        # Final check
        if debug:
            valid_original = ~np.isnan(phase_np)
            nan_in_final = np.sum(np.isnan(result[valid_original]))
            print(f'  Final result: {nan_in_final}/{np.sum(valid_original)} NaN in valid region')

        return result, labels
