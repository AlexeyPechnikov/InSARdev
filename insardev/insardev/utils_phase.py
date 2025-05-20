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
import numpy as np
import xarray as xr

def uncertainty(phase, weight=None):
    """
    Median absolute deviation of wrapped residuals |φ_i - delta| over all pixels.
    Returns a rounded value in [0, π] with 3 decimal places. Too small differences are not significant.
    """
    import numpy as np
    x = phase.values.ravel() if hasattr(phase, 'values') else np.asarray(phase).ravel()
    resid = np.abs((x + np.pi) % (2*np.pi) - np.pi)
    if weight is not None:
        w = weight.values.ravel() if hasattr(weight, 'values') else np.asarray(weight).ravel()
        validmask = np.isfinite(resid) & np.isfinite(w)
        if not np.any(validmask):
            return np.nan
        resid = resid[validmask]
        w = w[validmask]
        if w.sum() == 0:
            return np.nan
        order = np.argsort(resid)
        c = np.cumsum(w[order])
        return np.round(resid[order[np.searchsorted(c, 0.5 * c[-1])]], 3)
    return np.round(np.nanmedian(resid), 3)

def overlap_offset(phase : xr.DataArray|np.ndarray,
                            weight : xr.DataArray|np.ndarray|None=None,
                            func_aggregate : callable = None,
                            nbins : int = 360,
                            window : float = np.pi/3,
                            func_uncertainty : callable = uncertainty,
                            median_window_size : int = 15,
                            threshold : float = 0.02) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the phase offset in burst interferogram overlaps,
    using optional per-pixel weights (correlation).
    
    Parameters
    ----------
    phase : xarray.DataArray
        Wrapped or unwrapped phase differences (radians).
    weights : xarray.DataArray or None
        Same shape as `phase`, values in [0,1], higher = more reliable.
    nbins : int
        Number of histogram bins over [-π,π).
    window : float
        Half-width (radians) of the arc to refine around the coarse peak.
    median_window_size : int
        Size of the median filter to apply to the histogram. For window π/24 and nbins=360 corresponds to 15.
    func_uncertainty : callable
        Function to compute the uncertainty of the phase offset.
    threshold : float
        Threshold for the uncertainty to compare the main peak with the second peak.
        Main histogram peak should be preferred unless its uncertainty is significantly higher than the second peak (for multimodal distributions).
    
    Returns
    -------
    midpoint : float
        Estimated bulk phase jump (radians in [-π,π)).
    bins : ndarray[nbins+1]
        The histogram bin edges.
    densities : ndarray[nbins]
        The windowed (local) sum of weights in each bin ("density" curve).

    Examples:
    --------
    >>> sbas.phase_overlap_offset(data, weight=weight, window=np.pi/24, median_window_size=15)
    >>> sbas.phase_overlap_offset(data.reduce(utils.circular_mean, dim='x'), weight=weight.reduce(utils.circular_mean, dim='x'), window=np.pi/24)
    >>> sbas.phase_overlap_offset(data.reduce(utils.circular_mean, dim='y'), weight=weight.reduce(utils.circular_mean, dim='y'), window=np.pi/24)
    """
    import xarray as xr
    import numpy as np
    from scipy.ndimage import median_filter, uniform_filter1d

    a = np.asarray((func_aggregate(phase) if func_aggregate is not None else phase).data.ravel())
    if weight is not None:
        w = np.asarray((func_aggregate(weight) if func_aggregate is not None else weight).data.ravel())
    else:
        w = np.ones_like(a)
    mask = np.isfinite(a) & np.isfinite(w)
    a = a[mask]
    w = w[mask]
    if a.size == 0:
        return np.nan, None, None

    # wrap to [-π,π)
    a = (a + np.pi) % (2*np.pi) - np.pi

    bins = np.linspace(-np.pi, np.pi, nbins+1)
    hist, _ = np.histogram(a, bins=bins, weights=w, density=True)
    #centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    half = int(np.round(window / bin_width))

    # remove outliers
    hist = median_filter(hist, size=median_window_size, mode='wrap')
    # densities[i] represents the average count in the window centered on centers[i]
    densities = uniform_filter1d(hist.astype(float),
                                size=2*half + 1,
                                mode='wrap')

    def find_peak(densities):
        # find the most populous arc on the circle
        idx = np.argmax(densities)
        centers = 0.5 * (bins[:-1] + bins[1:])
        midpoint = centers[idx]
        # try to improve the estimation by taking the circular mean of the cluster
        delta = (a - midpoint + np.pi) % (2*np.pi) - np.pi
        cluster = np.abs(delta) <= window
        if np.count_nonzero(cluster) > 1:
            a_cl = a[cluster]
            w_cl = w[cluster]
            sinm = np.sum(w_cl * np.sin(a_cl))
            cosm = np.sum(w_cl * np.cos(a_cl))
            midpoint = np.arctan2(sinm, cosm)
        # wrap to [-π,π)
        midpoint = (midpoint + np.pi) % (2*np.pi) - np.pi
        return midpoint, idx

    def get_mask(peak_idx, width):
        return np.arange(peak_idx-width, peak_idx+width+1, dtype=int) % densities.size

    # find the main peak
    main_midpoint, main_idx = find_peak(densities)
    main_uncertainty = func_uncertainty(phase - main_midpoint, weight)
    # create mask for non-peak regions and set them to 0
    #densities[~np.isin(np.arange(densities.size), get_mask(main_idx, half))] = np.nan
    #return main_midpoint, bins, hist, densities
    print ('main_uncertainty',   np.round(main_uncertainty, 2),   'for', np.round(main_midpoint, 2))
    
    # remove the main peak and continue to find the second peak
    densities2 = densities.copy()
    # do not allow the peaks windows to overlap
    main_mask = get_mask(main_idx, 2*half)
    densities2[main_mask] = 0.0

    # find the second peak
    second_midpoint, second_idx = find_peak(densities2)
    second_uncertainty = func_uncertainty(phase - second_midpoint, weight)
    print ('second_uncertainty', np.round(second_uncertainty, 2), 'for', np.round(second_midpoint, 2))

    # select the peak with the lower uncertainty
    # midpoint = main_midpoint if main_uncertainty - threshold <= second_uncertainty else second_midpoint
    # idx = main_idx if main_uncertainty - threshold <= second_uncertainty else second_idx
    if main_uncertainty - threshold <= second_uncertainty:
        midpoint = main_midpoint
        idx = main_idx
        uncertainty  = main_uncertainty
    else:
        midpoint = second_midpoint
        idx = second_idx
        uncertainty = second_uncertainty

    # create mask for the peak region
    densities[~np.isin(np.arange(densities.size), get_mask(idx, half))] = np.nan
    #main_mask = get_mask(main_idx, half)
    #second_mask = get_mask(second_idx, half)
    # scale the second peak to the main peak to estimate the probability of the main peak being the correct one
    #scale = (densities[main_mask].sum()/densities[second_mask].sum()).round(2)
    #print ('main_midpoint', np.round(main_midpoint, 2), 'second_midpoint', np.round(second_midpoint, 2), '/', scale)
    return midpoint, uncertainty, bins, hist, densities

def wrap(phase):
    import xarray as xr
    import numpy as np
    import dask

    if isinstance(phase, xr.DataArray):
        return xr.DataArray(dask.array.mod(phase.data + np.pi, 2 * np.pi) - np.pi, phase.coords).rename(phase.name)
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi

def positive_range(phase):
    """
    Convert phase from the range [-pi, pi] to [0, 2pi].

    Parameters
    ----------
    phase : array_like
        Input phase values in the range [-pi, pi].

    Returns
    -------
    ndarray
        Phase values converted to the range [0, 2pi].
    
    Examples
    --------
    >>> phase_to_positive_range(np.array([-np.pi, -np.pi/2, np.pi, 2*-np.pi-1e-6, 2*-np.pi]))
    array([3.14159265, 4.71238898, 3.14159265, 6.28318431, 0.        ])
    """
    import numpy as np
    return (phase + 2 * np.pi) % (2 * np.pi)

def symmetric_range(phase):
    """
    Convert phase from the range [0, 2pi] to [-pi, pi].

    Parameters
    ----------
    phase : array_like
        Input phase values in the range [0, 2pi].

    Returns
    -------
    ndarray
        Phase values converted to the range [-pi, pi].
    
    Examples
    --------
    >>> phase_to_symmetric_range(np.array([0, np.pi, 3*np.pi/2, 2*np.pi]))
    array([ 0.        ,  3.14159265, -1.57079633,  0.        ])
    """
    import numpy as np
    return (phase + np.pi) % (2 * np.pi) - np.pi
