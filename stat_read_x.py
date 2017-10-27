"""
Read and construct design matrix.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-08-14
"""

import numpy as np

"""
installed all the libraries above
"""


def read_x(var_matrix, var_type):
    """
    Read and construct design matrix.

    Args:
        var_matrix (matrix): un-normalized design matrix (n*(p-1))
        var_type (vector): covariate type in var_matrix (0-discrete; 1-continuous)
    """

    n, p = var_matrix.shape
    if n < p:
        mat = var_matrix.T
    else:
        mat = var_matrix
    n, p = mat.shape

    mat_new = np.zeros((n, p))

    for kk in range(p):
        if var_type[kk] == 1:
            mat_new[:, kk] = (mat[:, kk] - np.mean(mat[:, kk]))/np.std(mat[:, kk])
            #  mat_new[:, kk] = mat[:, kk]/np.sqrt(np.sum(mat[:, kk]**2))
        else:
            mat_new[:, kk] = mat[:, kk]
    const = np.ones((n, 1))
    x_design = np.hstack((const, mat_new))

    return x_design
