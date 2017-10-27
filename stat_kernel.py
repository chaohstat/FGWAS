"""
Construct kernel functions in MVCM.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-08-14
"""

import numpy as np

"""
installed all the libraries above
"""


def ep_kernel(x, h):
    """
    Construct Epanechnikov kernel function in MVCM.

    Args:
        x (matrix): vector or matrix in coordinate matrix
        h (scalar): bandwidth
    """

    x[np.absolute(x) > 1] = 1
    ep_k = 0.75 * (1 - x**2) / h

    return ep_k


def gau_kernel(x, h):
    """
    Construct Gaussian kernel function in MVCM.

    Args:
        x (matrix): vector or matrix in coordinate matrix
        h (scalar): bandwidth
    """

    gau_k = 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2)/h

    return gau_k
