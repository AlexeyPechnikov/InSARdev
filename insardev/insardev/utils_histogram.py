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

def histogram(data, bins, range):
    """
    hist, bins = utils.histogram(corr60m.mean('pair'), 10, (0,1))
    print ('bins', bins)
    hist = dask.compute(hist)[0]
    print (hist)
    """
    import dask
    return dask.array.histogram(data, bins=bins, range=range)
