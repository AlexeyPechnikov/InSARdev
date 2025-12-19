#!/usr/bin/env python3
"""
Benchmark test for phase unwrapping algorithms.

Compares processing time and residuals for all unwrappers vs SNAPHU
on different size realistic synthetic data with NaNs.

Synthetic data generation based on:
- Surface deformation from elastic source models (Mogi, Okada)
- Atmospheric phase delays (stratified + turbulent)
- Decorrelation noise patterns

References:
- UnwrapDiff: Conditional Diffusion for InSAR Phase Unwrapping (2024)
- Phase unwrapping of SAR interferogram from modified U-net (2024)

Usage:
    python test_unwrap_benchmark.py
"""
import sys
import os
import time
import tempfile
import subprocess
import multiprocessing as mp
from pathlib import Path

import numpy as np

# Use spawn context for multiprocessing on macOS
# This is required for MPS (Metal) to work correctly in subprocesses
# Fork doesn't work with MPS - it produces silent garbage results
if sys.platform == 'darwin':
    mp.set_start_method('spawn', force=True)


def generate_mogi_deformation(height, width, depth=5000, volume_change=1e6):
    """
    Generate surface deformation from a Mogi point source (magma chamber).

    Parameters
    ----------
    height, width : int
        Grid dimensions in pixels.
    depth : float
        Source depth in meters.
    volume_change : float
        Volume change in cubic meters.

    Returns
    -------
    los_phase : np.ndarray
        Line-of-sight phase in radians.
    """
    # Assume 100m pixel spacing
    pixel_size = 100
    y, x = np.mgrid[0:height, 0:width]

    # Center the source
    x0, y0 = width // 2, height // 2

    # Convert to meters from source
    dx = (x - x0) * pixel_size
    dy = (y - y0) * pixel_size
    r = np.sqrt(dx**2 + dy**2 + depth**2)

    # Mogi solution for vertical displacement
    # uz = (1-nu) * dV / (pi * r^3) * d
    nu = 0.25  # Poisson's ratio
    uz = (1 - nu) * volume_change * depth / (np.pi * r**3)

    # Convert to LOS (assume ~23 deg incidence, ~12 deg azimuth)
    inc_angle = np.deg2rad(23)
    az_angle = np.deg2rad(-12)

    # Horizontal displacement (radial)
    ur = (1 - nu) * volume_change / (np.pi * r**3) * np.sqrt(dx**2 + dy**2)

    # LOS = -uz*cos(inc) + ur*sin(inc)*cos(az)
    los = -uz * np.cos(inc_angle) + ur * np.sin(inc_angle)

    # Convert to phase (C-band, ~5.6cm wavelength)
    wavelength = 0.056
    phase = 4 * np.pi * los / wavelength

    return phase.astype(np.float32)


def generate_atmospheric_phase(height, width, stratified_strength=1.0, turbulent_strength=1.0, seed=None):
    """
    Generate atmospheric phase screen with stratified and turbulent components.

    Parameters
    ----------
    height, width : int
        Grid dimensions.
    stratified_strength : float
        Strength of stratified (elevation-correlated) delay.
    turbulent_strength : float
        Strength of turbulent (random) delay.
    seed : int, optional
        Random seed.

    Returns
    -------
    atm_phase : np.ndarray
        Atmospheric phase in radians.
    """
    if seed is not None:
        np.random.seed(seed)

    y, x = np.mgrid[0:height, 0:width]

    # Stratified component - smooth gradient (elevation-like)
    # Simulates delay proportional to elevation
    stratified = stratified_strength * (0.01 * y + 0.005 * x + 0.00005 * x * y)

    # Turbulent component - fractal/power-law spectrum
    # Use 2D FFT with power-law decay
    freq_y = np.fft.fftfreq(height)
    freq_x = np.fft.fftfreq(width)
    fy, fx = np.meshgrid(freq_y, freq_x, indexing='ij')
    freq_mag = np.sqrt(fx**2 + fy**2)
    freq_mag[0, 0] = 1  # Avoid division by zero

    # Power-law spectrum with exponent -8/3 (Kolmogorov turbulence)
    # Add random phase
    spectrum = turbulent_strength * freq_mag**(-4/3) * np.exp(2j * np.pi * np.random.rand(height, width))
    spectrum[0, 0] = 0  # Zero mean

    turbulent = np.real(np.fft.ifft2(spectrum))

    return (stratified + turbulent).astype(np.float32)


def generate_decorrelation_noise(height, width, base_correlation=0.7, seed=None):
    """
    Generate spatially varying decorrelation noise.

    Parameters
    ----------
    height, width : int
        Grid dimensions.
    base_correlation : float
        Mean correlation value.
    seed : int, optional
        Random seed.

    Returns
    -------
    noise : np.ndarray
        Phase noise in radians.
    correlation : np.ndarray
        Correlation values [0, 1].
    """
    if seed is not None:
        np.random.seed(seed)

    # Create spatially varying correlation
    # Use smooth random field for spatial structure
    freq_y = np.fft.fftfreq(height)
    freq_x = np.fft.fftfreq(width)
    fy, fx = np.meshgrid(freq_y, freq_x, indexing='ij')
    freq_mag = np.sqrt(fx**2 + fy**2)
    freq_mag[0, 0] = 1

    # Low-pass filter for smooth variation
    spectrum = freq_mag**(-2) * np.exp(2j * np.pi * np.random.rand(height, width))
    spectrum[0, 0] = 0

    corr_variation = np.real(np.fft.ifft2(spectrum))
    corr_variation = (corr_variation - corr_variation.min()) / (corr_variation.max() - corr_variation.min())

    # Correlation: base +/- 0.2 variation
    correlation = np.clip(base_correlation + 0.3 * (corr_variation - 0.5), 0.1, 0.99)

    # Phase noise standard deviation from correlation
    # sigma_phi = sqrt(1 - gamma^2) / gamma  (simplified Cramer-Rao bound)
    noise_std = np.sqrt(1 - correlation**2) / np.clip(correlation, 0.1, 1.0)

    # Generate noise
    noise = noise_std * np.random.randn(height, width)

    return noise.astype(np.float32), correlation.astype(np.float32)


def generate_nan_gaps(height, width, nan_fraction=0.1, seed=None):
    """
    Generate realistic NaN mask (water bodies, shadows, layover).

    Parameters
    ----------
    height, width : int
        Grid dimensions.
    nan_fraction : float
        Approximate fraction of NaN pixels.
    seed : int, optional
        Random seed.

    Returns
    -------
    mask : np.ndarray
        Boolean mask (True = valid, False = NaN).
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.ones((height, width), dtype=bool)

    # Random rectangular holes (water bodies, shadows)
    n_holes = max(1, int(height * width * nan_fraction / 500))
    for _ in range(n_holes):
        cy = np.random.randint(20, height - 20)
        cx = np.random.randint(20, width - 20)
        ry = np.random.randint(5, min(40, height // 8))
        rx = np.random.randint(5, min(40, width // 8))
        mask[max(0, cy-ry):min(height, cy+ry), max(0, cx-rx):min(width, cx+rx)] = False

    # Add some edge gaps (layover/shadow in mountains)
    if height > 100 and width > 100:
        # Diagonal strips
        for i in range(3):
            start_y = np.random.randint(0, height - 50)
            start_x = np.random.randint(0, width - 50)
            length = np.random.randint(30, min(100, height//3, width//3))
            thickness = np.random.randint(3, 8)
            for j in range(length):
                y_idx = min(start_y + j, height - 1)
                x_idx = min(start_x + j, width - 1)
                mask[max(0, y_idx-thickness):min(height, y_idx+thickness),
                     max(0, x_idx-thickness//2):min(width, x_idx+thickness//2)] = False

    return mask


def generate_test_phase(height, width, noise_level=0.3, nan_fraction=0.1, seed=42, realistic=True):
    """
    Generate synthetic wrapped phase with noise and NaN gaps.

    Parameters
    ----------
    height, width : int
        Grid dimensions.
    noise_level : float
        Noise level multiplier.
    nan_fraction : float
        Fraction of pixels to set as NaN (gaps).
    seed : int
        Random seed for reproducibility.
    realistic : bool
        If True, use realistic deformation + atmosphere model.
        If False, use simple ramp (backward compatibility).

    Returns
    -------
    phase : np.ndarray
        Wrapped phase in radians [-π, π].
    correlation : np.ndarray
        Simulated correlation (higher = less noise).
    true_unwrapped : np.ndarray
        True unwrapped phase for residual calculation.
    """
    np.random.seed(seed)

    if realistic:
        # Realistic synthetic data
        # 1. Deformation signal (Mogi source)
        defo_phase = generate_mogi_deformation(height, width,
                                                depth=3000 + np.random.randint(-1000, 1000),
                                                volume_change=5e5 + np.random.rand() * 1e6)

        # 2. Atmospheric phase
        atm_phase = generate_atmospheric_phase(height, width,
                                                stratified_strength=0.5 + 0.5 * np.random.rand(),
                                                turbulent_strength=0.3 + 0.3 * np.random.rand(),
                                                seed=seed + 1)

        # 3. True phase (before noise)
        true_phase = defo_phase + atm_phase

        # 4. Decorrelation noise
        noise, correlation = generate_decorrelation_noise(height, width,
                                                           base_correlation=0.7 - 0.2 * noise_level,
                                                           seed=seed + 2)
        noisy_phase = true_phase + noise * noise_level

    else:
        # Simple ramp (original implementation)
        y, x = np.mgrid[0:height, 0:width]
        true_phase = (0.05 * x + 0.03 * y + 0.0001 * x * y).astype(np.float32)
        noisy_phase = true_phase + noise_level * np.random.randn(height, width)
        correlation = np.clip(1.0 - 0.5 * np.abs(np.random.randn(height, width)) * noise_level,
                             0.1, 1.0).astype(np.float32)

    # Wrap to [-π, π]
    wrapped = np.arctan2(np.sin(noisy_phase), np.cos(noisy_phase)).astype(np.float32)

    # Apply NaN mask
    nan_mask = generate_nan_gaps(height, width, nan_fraction, seed=seed + 3)
    wrapped[~nan_mask] = np.nan
    correlation[~nan_mask] = np.nan

    return wrapped, correlation, true_phase.astype(np.float32)


def compute_residual(unwrapped, true_phase):
    """
    Compute RMS residual between unwrapped and true phase.

    Accounts for constant offset (unwrapping ambiguity).
    """
    valid = np.isfinite(unwrapped) & np.isfinite(true_phase)
    if not np.any(valid):
        return np.nan

    diff = unwrapped[valid] - true_phase[valid]
    # Remove mean offset
    diff = diff - np.mean(diff)

    return np.sqrt(np.mean(diff ** 2))


def run_snaphu(phase, correlation, timeout=900):
    """
    Run SNAPHU unwrapper using pygmtsar-style configuration.

    Returns (unwrapped, elapsed_time, peak_memory_mb) or (None, None, None) on failure/timeout.
    """
    height, width = phase.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write input files
        phase_file = os.path.join(tmpdir, 'phase.bin')
        corr_file = os.path.join(tmpdir, 'corr.bin')
        mask_file = os.path.join(tmpdir, 'mask.bin')
        out_file = os.path.join(tmpdir, 'unwrapped.bin')

        # SNAPHU expects float32 for phase, float32 for correlation
        phase_clean = np.where(np.isnan(phase), 0.0, phase).astype(np.float32)
        corr_clean = np.where(np.isnan(correlation), 0.0, correlation).astype(np.float32)

        # Create byte mask (1=valid, 0=masked)
        mask = np.where(np.isnan(phase), 0, 1).astype(np.uint8)

        phase_clean.tofile(phase_file)
        corr_clean.tofile(corr_file)
        mask.tofile(mask_file)

        # Create config file (pygmtsar-style)
        config = f"""
        INFILEFORMAT   FLOAT_DATA
        OUTFILEFORMAT  FLOAT_DATA
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
        DEFOMAX_CYCLE  0
        TILEDIR        {tmpdir}/snaphu_tiles
        """

        # Run SNAPHU with DEFO mode and mask
        cmd = [
            'snaphu',
            phase_file,
            str(width),
            '-M', mask_file,
            '-f', '/dev/stdin',
            '-o', out_file,
            '-d',  # DEFO mode
            '-c', corr_file,
        ]

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                input=config,
                capture_output=True,
                timeout=timeout,
                text=True
            )
            elapsed = time.time() - t0

            if result.returncode != 0:
                return None, None, None

            # Read output
            if os.path.exists(out_file):
                unwrapped = np.fromfile(out_file, dtype=np.float32).reshape(height, width)
                # Restore NaN mask
                unwrapped[np.isnan(phase)] = np.nan
                # SNAPHU memory is external process, estimate from file sizes
                mem_estimate = (height * width * 4 * 6) / (1024 * 1024)  # rough estimate
                return unwrapped, elapsed, mem_estimate
            else:
                return None, None, None

        except subprocess.TimeoutExpired:
            return None, None, None
        except Exception:
            return None, None, None


def run_unwrapper(method, phase, correlation, timeout=3600):
    """
    Run internal unwrapper method.

    Returns (unwrapped, elapsed_time, peak_memory_mb) or (None, None, None) on failure/timeout.
    """
    import tracemalloc
    import torch
    from insardev.Stack_unwrap2d import Stack_unwrap2d

    # Start memory tracking
    tracemalloc.start()

    t0 = time.time()
    try:
        if method == 'irls_gpu':
            # IRLS on GPU (MPS on Apple Silicon, CUDA on NVIDIA)
            result = Stack_unwrap2d._irls_unwrap_2d(phase, weight=correlation, device='auto')
        elif method == 'irls_cpu':
            # IRLS on CPU only
            result = Stack_unwrap2d._irls_unwrap_2d(phase, weight=correlation, device='cpu')
        else:
            tracemalloc.stop()
            return None, None, None

        elapsed = time.time() - t0

        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024 * 1024)  # bytes to MB

        if elapsed > timeout:
            return None, None, None

        return result, elapsed, peak_mb

    except Exception as e:
        tracemalloc.stop()
        return None, None, None


def benchmark_single(args):
    """
    Benchmark a single method at a single size.
    Called in separate process for isolation.

    Returns dict with results including unwrapped data and memory usage.
    """
    method, size, data_file, timeout = args

    # Load pre-generated data from file (ensures all methods use identical input)
    data = np.load(data_file)
    phase = data['phase']
    correlation = data['correlation']
    true_phase = data['true_phase']

    if method == 'snaphu':
        unwrapped, elapsed, memory_mb = run_snaphu(phase, correlation, timeout=timeout)
    else:
        unwrapped, elapsed, memory_mb = run_unwrapper(method, phase, correlation, timeout=timeout)

    if unwrapped is None:
        return {
            'method': method,
            'size': size,
            'time': None,
            'residual': None,
            'memory_mb': None,
            'status': 'timeout/failed',
            'unwrapped': None
        }

    residual = compute_residual(unwrapped, true_phase)

    return {
        'method': method,
        'size': size,
        'time': elapsed,
        'residual': residual,
        'memory_mb': memory_mb,
        'status': 'ok',
        'unwrapped': unwrapped
    }


def save_phase_image(phase, output_path, cmap='gist_rainbow_r', title=None, is_wrapped=True):
    """
    Save phase as RGB image with colormap.

    Parameters
    ----------
    phase : np.ndarray
        Phase data.
    output_path : str
        Output file path.
    cmap : str
        Colormap name ('gist_rainbow_r' for wrapped, 'turbo' for unwrapped).
    title : str, optional
        Title for the image.
    is_wrapped : bool
        If True, use [-π, π] normalization.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(10, 10))

    if is_wrapped:
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
    else:
        valid = np.isfinite(phase)
        if np.any(valid):
            vmin, vmax = np.nanpercentile(phase, [2, 98])
        else:
            vmin, vmax = -10, 10
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(phase, cmap=cmap, norm=norm, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Phase (radians)')

    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel('Range (pixels)')
    ax.set_ylabel('Azimuth (pixels)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_benchmarks(max_time_per_test=3600, output_dir=None, save_images=True):
    """
    Run all benchmarks and return results.

    Continues doubling sizes until all algorithms timeout.
    All methods receive identical input data (pre-generated and saved to temp files).
    """
    # Methods to test
    methods = ['snaphu', 'irls_gpu', 'irls_cpu']

    # Start at 100, double until all timeout
    initial_size = 100
    max_reasonable_size = 12800  # Support large data processing

    # Track which methods have timed out (stop testing larger sizes)
    method_max_size = {m: max_reasonable_size for m in methods}

    # Track which methods are still active
    active_methods = set(methods)

    results = []
    tested_sizes = []

    if output_dir is None:
        output_dir = Path(__file__).parent

    size = initial_size

    while active_methods and size <= max_reasonable_size:
        tested_sizes.append(size)
        print(f"\n=== Testing size {size}x{size} ===")

        # Generate data at this size (once, in main process)
        print(f"  Generating test data...")
        phase, correlation, true_phase = generate_test_phase(size, size)

        # Save data to temp file so all workers load identical data
        data_file = output_dir / f'benchmark_data_{size}x{size}.npz'
        np.savez_compressed(str(data_file), phase=phase, correlation=correlation, true_phase=true_phase)
        print(f"  Data saved to: {data_file}")

        # Save input phase image
        if save_images:
            input_path = output_dir / f'benchmark_{size}x{size}.png'
            save_phase_image(phase, str(input_path), cmap='gist_rainbow_r',
                           title=f'Wrapped Phase {size}x{size}', is_wrapped=True)
            print(f"  Saved input: {input_path}")

        # Prepare tasks for methods that haven't timed out
        # Pass file path instead of data arrays - ensures all methods use identical input
        tasks = []
        for method in methods:
            if method in active_methods:
                tasks.append((method, size, str(data_file), max_time_per_test))

        if not tasks:
            print("  All methods have timed out, stopping.")
            # Clean up data file
            if data_file.exists():
                data_file.unlink()
            break

        # Run in parallel (one process per method)
        print(f"  Testing {len(tasks)} methods in parallel...")

        with mp.Pool(processes=min(len(tasks), mp.cpu_count())) as pool:
            task_results = pool.map(benchmark_single, tasks)

        # Clean up data file after tests complete
        if data_file.exists():
            data_file.unlink()

        # Process results
        for r in task_results:
            # Store result (without the large unwrapped array for memory)
            result_copy = {k: v for k, v in r.items() if k != 'unwrapped'}
            results.append(result_copy)

            if r['status'] != 'ok':
                # Mark this method as timed out - don't test larger sizes
                active_methods.discard(r['method'])
                method_max_size[r['method']] = size - 1
                print(f"  {r['method']}: TIMEOUT/FAILED at {size}x{size}")
            else:
                time_str = f"{r['time']:.2f}s" if r['time'] else "N/A"
                res_str = f"{r['residual']:.4f}" if r['residual'] and np.isfinite(r['residual']) else "N/A"
                mem_str = f"{r['memory_mb']:.0f}MB" if r['memory_mb'] else "N/A"
                print(f"  {r['method']}: {time_str}, residual={res_str}, mem={mem_str}")

                # Save unwrapped phase image
                if save_images and r['unwrapped'] is not None:
                    method_path = output_dir / f'benchmark_{size}x{size}.{r["method"].upper()}.png'
                    save_phase_image(r['unwrapped'], str(method_path), cmap='turbo',
                                   title=f'{r["method"].upper()} Unwrapped {size}x{size} (t={r["time"]:.2f}s)',
                                   is_wrapped=False)
                    print(f"  Saved output: {method_path}")

        # Double size for next iteration
        size *= 2

    return results, tested_sizes


def format_results_table(results, tested_sizes):
    """
    Format results as a text table.
    """
    # Organize by method and size
    methods = sorted(set(r['method'] for r in results))
    sizes = tested_sizes

    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r['method'], r['size'])] = r

    # Header
    lines = []
    lines.append("=" * 100)
    lines.append("PHASE UNWRAPPING BENCHMARK RESULTS")
    lines.append("Synthetic data: Mogi deformation + atmospheric delays + decorrelation noise")
    lines.append("=" * 100)
    lines.append("")

    # Time table
    lines.append("PROCESSING TIME (seconds)")
    lines.append("-" * 100)

    header = f"{'Method':<12}"
    for size in sizes:
        header += f" {size:>10}"
    lines.append(header)
    lines.append("-" * 100)

    for method in methods:
        row = f"{method:<12}"
        for size in sizes:
            key = (method, size)
            if key in lookup and lookup[key]['time'] is not None:
                row += f" {lookup[key]['time']:>10.2f}"
            else:
                row += f" {'---':>10}"
        lines.append(row)

    lines.append("")

    # Residual table
    lines.append("RMS RESIDUAL (radians)")
    lines.append("-" * 100)

    header = f"{'Method':<12}"
    for size in sizes:
        header += f" {size:>10}"
    lines.append(header)
    lines.append("-" * 100)

    for method in methods:
        row = f"{method:<12}"
        for size in sizes:
            key = (method, size)
            if key in lookup and lookup[key]['residual'] is not None and np.isfinite(lookup[key]['residual']):
                row += f" {lookup[key]['residual']:>10.4f}"
            else:
                row += f" {'---':>10}"
        lines.append(row)

    lines.append("")

    # Memory table
    lines.append("PEAK MEMORY (MB)")
    lines.append("-" * 100)

    header = f"{'Method':<12}"
    for size in sizes:
        header += f" {size:>10}"
    lines.append(header)
    lines.append("-" * 100)

    for method in methods:
        row = f"{method:<12}"
        for size in sizes:
            key = (method, size)
            if key in lookup and lookup[key].get('memory_mb') is not None:
                row += f" {lookup[key]['memory_mb']:>10.0f}"
            else:
                row += f" {'---':>10}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 100)
    lines.append("--- = timeout or failed")
    lines.append("")

    return "\n".join(lines)


def plot_results(results, tested_sizes, output_path=None):
    """
    Plot results as figures (3 subplots: time, residual, memory).
    """
    import matplotlib.pyplot as plt

    # Organize data
    methods = sorted(set(r['method'] for r in results))
    sizes = tested_sizes

    lookup = {}
    for r in results:
        lookup[(r['method'], r['size'])] = r

    # Color map for methods
    colors = {
        'snaphu': 'black',
        'maxflow': 'blue',
        'minflow': 'green',
        'ilp': 'red',
        'irls_gpu': 'purple',
        'irls_cpu': 'magenta'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Time plot (log-log)
    ax1 = axes[0]
    for method in methods:
        x = []
        y = []
        for size in sizes:
            key = (method, size)
            if key in lookup and lookup[key]['time'] is not None:
                x.append(size)
                y.append(lookup[key]['time'])
        if x:
            ax1.loglog(x, y, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=8)

    ax1.set_xlabel('Grid Size (pixels)', fontsize=12)
    ax1.set_ylabel('Processing Time (seconds)', fontsize=12)
    ax1.set_title('Processing Time vs Grid Size', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=300, color='red', linestyle='--', alpha=0.5)
    ax1.text(sizes[0], 320, 'timeout (5 min)', color='red', fontsize=10)

    # Residual plot (log-log)
    ax2 = axes[1]
    for method in methods:
        x = []
        y = []
        for size in sizes:
            key = (method, size)
            if key in lookup and lookup[key]['residual'] is not None and np.isfinite(lookup[key]['residual']):
                x.append(size)
                y.append(lookup[key]['residual'])
        if x:
            ax2.loglog(x, y, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=8)

    ax2.set_xlabel('Grid Size (pixels)', fontsize=12)
    ax2.set_ylabel('RMS Residual (radians)', fontsize=12)
    ax2.set_title('Unwrapping Accuracy vs Grid Size', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Memory plot (log-log)
    ax3 = axes[2]
    for method in methods:
        x = []
        y = []
        for size in sizes:
            key = (method, size)
            if key in lookup and lookup[key].get('memory_mb') is not None:
                x.append(size)
                y.append(lookup[key]['memory_mb'])
        if x:
            ax3.loglog(x, y, 'o-', label=method, color=colors.get(method, 'gray'), linewidth=2, markersize=8)

    ax3.set_xlabel('Grid Size (pixels)', fontsize=12)
    ax3.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax3.set_title('Memory Usage vs Grid Size', fontsize=14)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Check SNAPHU is available
    try:
        result = subprocess.run(['snaphu'], capture_output=True, timeout=5)
    except FileNotFoundError:
        print("ERROR: SNAPHU not found in PATH. Please install SNAPHU first.")
        print("  brew install snaphu  (macOS)")
        print("  apt install snaphu   (Ubuntu)")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        pass  # OK, just checking it exists

    print("Phase Unwrapping Benchmark")
    print("=" * 60)
    print("Methods: SNAPHU, DCT, IRLS (GPU), IRLS (CPU)")
    print("Synthetic data: Mogi deformation + atmosphere + noise")
    print("Sizes: 100 → 200 → 400 → ... (doubling until all timeout)")
    print("Timeout: 3600 seconds (1 hour) per test")
    print("Images: gist_rainbow_r (wrapped), turbo (unwrapped)")
    print("")

    # Output directory
    output_dir = Path(__file__).parent

    # Run benchmarks
    results, tested_sizes = run_benchmarks(max_time_per_test=3600, output_dir=output_dir, save_images=True)

    # Print table
    table = format_results_table(results, tested_sizes)
    print(table)

    # Save table to file
    table_path = output_dir / 'benchmark_results.txt'
    with open(table_path, 'w') as f:
        f.write(table)
    print(f"Table saved to: {table_path}")

    # Plot results
    try:
        plot_path = output_dir / 'benchmark_results.png'
        plot_results(results, tested_sizes, output_path=str(plot_path))
    except ImportError:
        print("matplotlib not available, skipping plot")
    except Exception as e:
        print(f"Plot failed: {e}")
