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

import threading
import functools
import inspect as _inspect

# Metal/CUDA graph and allocator state is process-global, but dask drives this
# library from several worker threads at once (the notebooks use
# threads_per_worker=2). Concurrent non-CPU work corrupts that shared state and
# kills the interpreter outright -- an MPSGraph "over-released" warning, then a
# segfault or an NSException from nan_to_num_out_mps / arange_mps_out.
#
# ONE lock for the whole process. Per-module locks do not compose: a thread in
# the gaussian path and a thread in the IRLS path would hold different locks and
# still collide. Re-entrant, because these entry points call each other.
GPU_LOCK = threading.RLock()


def serialize_gpu(fn):
    """Serialize a GPU entry point. CPU calls run unlocked and stay parallel."""
    sig = _inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            dev = bound.arguments.get('device', 'cpu')
        except TypeError:
            dev = kwargs.get('device', 'cpu')
        dev = str(getattr(dev, 'type', dev))
        if dev == 'auto':
            dev = get_torch_device('auto')
            dev = str(getattr(dev, 'type', dev))
        if dev == 'cpu':
            return fn(*args, **kwargs)
        with GPU_LOCK:
            return fn(*args, **kwargs)
    return wrapper


def get_torch_device(device='auto', debug=False):
    """
    Get PyTorch device for GPU-accelerated operations.

    Checks Dask cluster resources:
    - If workers have resources={'gpu': N} where N >= 1 → use GPU
    - Otherwise (default) → CPU for parallel processing

    Parameters
    ----------
    device : str
        Device specification: 'auto', 'cuda', 'mps', or 'cpu'.
        'auto' uses CPU by default, GPU only if Dask has resources={'gpu': 1}.
    debug : bool
        Print debug information.

    Returns
    -------
    torch.device
        PyTorch device object.
    """
    import torch

    if device == 'auto':
        gpu_enabled = False

        try:
            from dask.distributed import get_client
            client = get_client()
            workers = client.scheduler_info().get('workers', {})
            if workers:
                # Only enable GPU if explicitly set to gpu >= 1
                gpu_enabled = any(w.get('resources', {}).get('gpu', 0) >= 1 for w in workers.values())
        except ValueError:
            # No Dask client active - still default to CPU
            pass

        if gpu_enabled:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        else:
            device = 'cpu'

    if debug:
        print(f"DEBUG: using device={device}")

    return torch.device(device)
