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
    """2D phase unwrapping using GPU-accelerated IRLS algorithm with DCT initialization."""

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

    def unwrap2d(self, phase, weight=None, conncomp=False,
                conncomp_size=1000, conncomp_gap=None,
                conncomp_linksize=5, conncomp_linkcount=30, device='cpu', debug=False, **kwargs):
        """
        Unwrap phase using GPU-accelerated IRLS algorithm (L¹ norm).

        Uses Iteratively Reweighted Least Squares with DCT-based preconditioner.
        GPU-accelerated using PyTorch (MPS on Apple Silicon, CUDA on NVIDIA,
        or CPU fallback).

        When conncomp=False (default), disconnected components are automatically
        linked using ILP optimization to find optimal 2π offsets.

        When conncomp=True, components are kept separate and returned with
        size-ordered labels (1=largest, 2=second largest, etc.).

        Parameters
        ----------
        phase : BatchWrap
            Batch of wrapped phase datasets with 'pair' dimension.
        weight : BatchUnit, optional
            Batch of correlation values for weighting. Higher values indicate
            more reliable phase measurements.
        conncomp : bool, optional
            If False (default), link disconnected components using ILP to find
            optimal 2π offsets, returning a single merged result.
            If True, keep components separate and return conncomp labels
            (1=largest component, 2=second largest, etc., 0=invalid).
        conncomp_size : int, optional
            Minimum number of pixels for a connected component to be processed.
            Components smaller than this are left as NaN. Default is 1000.
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
        device : str, optional
            PyTorch device to use: 'cpu' (default), 'cuda', or 'mps'.
            None selects the best available GPU or falls back to CPU.
        debug : bool, optional
            If True, print diagnostic information. Default is False.
        **kwargs
            Additional arguments passed to unwrap2d_irls:
            max_iter, tol, cg_max_iter, cg_tol, epsilon.

        Returns
        -------
        Batch or tuple
            If conncomp is False: Batch of unwrapped phase (components linked).
            If conncomp is True: tuple of (Batch unwrapped phase, BatchUnit conncomp)
            where conncomp labels are ordered by size (1=largest).

        Notes
        -----
        GPU acceleration:
        - mps on Apple Silicon (M1/M2/M3/M4)
        - cuda on NVIDIA GPUs
        - cpu fallback otherwise

        Component Linking (when conncomp=False):
        1. Unwraps each connected component separately
        2. Finds direct connections between components (not crossing others)
        3. Estimates phase offsets using conncomp_linksize pixels per connection
        4. Uses ILP to find globally optimal integer 2π offsets

        Examples
        --------
        Unwrap phase with component linking (default):
        >>> unwrapped = stack.unwrap2d(intfs, corr)

        Unwrap without weighting:
        >>> unwrapped = stack.unwrap2d(intfs)

        Keep components separate (no linking), get labels:
        >>> unwrapped, conncomp = stack.unwrap2d(intfs, corr, conncomp=True)
        >>> main_component = unwrapped.where(conncomp == 1)  # largest component

        Force CPU processing:
        >>> unwrapped = stack.unwrap2d(intfs, corr, device='cpu')
        """
        # Validate parameters
        if not conncomp and conncomp_linksize > conncomp_size:
            raise ValueError(
                f'conncomp_linksize ({conncomp_linksize}) cannot be greater than conncomp_size ({conncomp_size}). '
                f'Components must have at least conncomp_linksize pixels for reliable offset estimation.'
            )

        # Use IRLS unwrapping (always get conncomp for internal use)
        unwrapped, conncomp_labels = self.unwrap2d_irls(phase, weight, conncomp=True,
                                                        conncomp_size=conncomp_size, device=device,
                                                        debug=debug, **kwargs)

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

    def unwrap2d_dataset(self, phase, weight=None, conncomp=False,
                         conncomp_size=1000, conncomp_gap=None,
                         conncomp_linksize=5, conncomp_linkcount=30, device='cpu', debug=False, **kwargs):
        """
        Unwrap a single phase Dataset using GPU-accelerated IRLS algorithm.

        Convenience wrapper around unwrap2d() for working with merged datasets
        instead of per-burst batches. Useful when you have already dissolved
        and merged your data.

        Parameters
        ----------
        phase : xr.Dataset
            Wrapped phase dataset (from intf.align().dissolve().to_dataset()).
        weight : xr.Dataset, optional
            Correlation values for weighting (from corr.dissolve().to_dataset()).
        conncomp : bool, optional
            If False (default), link disconnected components.
            If True, keep components separate and return conncomp labels.
        conncomp_size : int, optional
            Minimum pixels for a connected component. Default is 1000.
        conncomp_gap : int or None, optional
            Maximum pixel distance between components. Default is None.
        conncomp_linksize : int, optional
            Pixels for offset estimation. Default is 5.
        conncomp_linkcount : int, optional
            Maximum neighbor components to consider. Default is 30.
        device : str, optional
            PyTorch device: 'cpu', 'cuda', or 'mps'. Default is 'cpu'.
        debug : bool, optional
            Print diagnostic information. Default is False.
        **kwargs
            Additional arguments passed to unwrap2d_irls.

        Returns
        -------
        xr.Dataset or tuple
            If conncomp is False: Unwrapped phase Dataset.
            If conncomp is True: tuple of (unwrapped Dataset, conncomp Dataset).

        Examples
        --------
        Basic usage with merged datasets:
        >>> intf_ds = intf.align().dissolve().compute().to_dataset()
        >>> corr_ds = corr.dissolve().compute().to_dataset()
        >>> unwrapped = stack.unwrap2d_dataset(intf_ds, corr_ds)

        Get connected components:
        >>> unwrapped, conncomp = stack.unwrap2d_dataset(intf_ds, corr_ds, conncomp=True)

        Convert back to per-burst Batch:
        >>> phase_batch = intf.from_dataset(unwrapped)
        """
        import xarray as xr
        from .Batch import BatchWrap, BatchUnit

        # Validate input types
        if not isinstance(phase, xr.Dataset):
            raise TypeError(f"phase must be xr.Dataset, got {type(phase).__name__}")
        if weight is not None and not isinstance(weight, xr.Dataset):
            raise TypeError(f"weight must be xr.Dataset, got {type(weight).__name__}")

        # Rechunk to single chunk for y/x dimensions (required by unwrap2d_irls)
        phase = phase.chunk({'y': -1, 'x': -1})
        if weight is not None:
            weight = weight.chunk({'y': -1, 'x': -1})

        # Wrap datasets in temporary batches with empty key
        intf_batch = BatchWrap({'': phase})
        corr_batch = BatchUnit({'': weight}) if weight is not None else None

        # Call unwrap2d
        result = self.unwrap2d(
            intf_batch, corr_batch, conncomp=conncomp,
            conncomp_size=conncomp_size, conncomp_gap=conncomp_gap,
            conncomp_linksize=conncomp_linksize, conncomp_linkcount=conncomp_linkcount,
            device=device, debug=debug, **kwargs
        )

        # Extract and return the dataset(s)
        if conncomp:
            unwrapped, conncomp_labels = result
            return unwrapped[''], conncomp_labels['']
        else:
            return result['']

    # =========================================================================
    # GPU-Accelerated Phase Unwrapping Methods (PyTorch)
    # =========================================================================

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

    def unwrap2d_irls(self, phase, weight=None, conncomp=False, conncomp_size=100, device=None,
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
        """
        import dask
        import dask.array
        import torch
        import xarray as xr
        from .Batch import Batch, BatchWrap, BatchUnit

        assert isinstance(phase, BatchWrap), 'ERROR: phase should be a BatchWrap object'
        assert weight is None or isinstance(weight, BatchUnit), 'ERROR: weight should be a BatchUnit object'

        # Resolve device
        if device is None:
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

            # Get data variables (typically polarization like 'VV'), excluding spatial_ref
            data_vars = [v for v in phase_ds.data_vars if v != 'spatial_ref']

            unwrap_vars = {}
            comp_vars = {}

            for var in data_vars:
                phase_da = phase_ds[var]
                weight_da = weight_ds[var] if weight_ds is not None else None

                # Save original coordinates before chunking (they may become lazy)
                # For dask arrays, compute() to get actual values
                # Preserve dimension info for non-dimension coordinates (like BPR along 'pair')
                original_coords = {}
                for k, v in phase_da.coords.items():
                    if hasattr(v, 'data') and hasattr(v.data, 'compute'):
                        vals = v.compute().values
                    elif hasattr(v, 'values'):
                        vals = v.values
                    else:
                        vals = v
                    # Preserve dimension tuple for coordinates that have dimensions
                    # different from their name (e.g., BPR with dims=('pair',))
                    if hasattr(v, 'dims') and len(v.dims) > 0 and v.dims != (k,):
                        original_coords[k] = (v.dims, vals)
                    else:
                        original_coords[k] = vals

                # Ensure data is chunked for lazy processing (1 chunk per pair)
                if 'pair' in phase_da.dims:
                    if not isinstance(phase_da.data, dask.array.Array):
                        phase_da = phase_da.chunk({'pair': 1})
                    if weight_da is not None and not isinstance(weight_da.data, dask.array.Array):
                        weight_da = weight_da.chunk({'pair': 1})

                # Use xr.apply_ufunc with dask='parallelized' for proper lazy execution
                # With chunk={'pair': 1}, dask splits (n_pairs, y, x) into n_pairs chunks of (1, y, x)
                # input_core_dims=[['y', 'x']] means pair is non-core and gets chunked

                def process_wrapper(phase_chunk, weight_chunk=None):
                    """Wrapper for IRLS processing that returns stacked results.

                    Input: phase_chunk shape (1, y, x) - single pair chunk
                    Output: shape (1, 2, y, x) where dim 1 is [unwrapped, labels]
                    """
                    # Squeeze to 2D for processing
                    phase_2d = phase_chunk[0]
                    weight_2d = weight_chunk[0] if weight_chunk is not None else None

                    unwrapped, labels = self._process_irls_slice(
                        phase_2d, weight_2d, device, conncomp_size,
                        max_iter, tol, cg_max_iter, cg_tol, epsilon, debug
                    )
                    # Stack and add pair dim back: (2, y, x) -> (1, 2, y, x)
                    result = np.stack([unwrapped, labels.astype(np.float32)], axis=0)
                    result = result[np.newaxis, ...]
                    return result

                # dask='parallelized' processes each chunk independently
                # input_core_dims=[['y', 'x']] so pair dim is chunked, function receives (1, y, x)
                with dask.annotate(resources={'gpu': 1} if device != 'cpu' else {}):
                    result = xr.apply_ufunc(
                        process_wrapper,
                        *([phase_da] if weight_da is None else [phase_da, weight_da]),
                        input_core_dims=[['y', 'x']] if weight_da is None else [['y', 'x'], ['y', 'x']],
                        output_core_dims=[['output', 'y', 'x']],
                        dask='parallelized',
                        output_dtypes=[np.float32],
                        dask_gufunc_kwargs={'output_sizes': {'output': 2}}
                    )

                # Extract unwrapped phase and connected components
                result_da = result.isel(output=0)
                result_da.attrs['units'] = 'radians'
                comp_da = result.isel(output=1).astype(np.int32)

                # Restore coordinates from original (non-lazy) values
                result_da = result_da.assign_coords(original_coords)
                comp_da = comp_da.assign_coords(original_coords)

                unwrap_vars[var] = result_da
                comp_vars[var] = comp_da

            # Preserve dataset attributes (subswath, pathNumber, etc.)
            unwrap_result[key] = xr.Dataset(unwrap_vars, attrs=phase_ds.attrs)
            conncomp_result[key] = xr.Dataset(comp_vars, attrs=phase_ds.attrs)
            # Preserve CRS from input dataset
            if phase_ds.rio.crs is not None:
                unwrap_result[key].rio.write_crs(phase_ds.rio.crs, inplace=True)
                conncomp_result[key].rio.write_crs(phase_ds.rio.crs, inplace=True)

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
