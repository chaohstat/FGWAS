"""
optimal bandwidth selection (Scott's Rule) in multivariate varying coefficient model (MVCM).

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-09-18
"""

import numpy as np
from numpy.linalg import inv
from statsmodels import robust

"""
installed all the libraries above
"""


def bw_rt(coord_mat, x_design, y_design):
    """
        optimal bandwidth (Scott's Rule) selection in MVCM.

        Args:
            coord_mat (matrix): common coordinate matrix (l*d)
            x_design (matrix): non-genetic design matrix (n*p)
            y_design (matrix): imaging response data (response matrix, n*l*m)
    """

    # Set up
    n, p = x_design.shape
    l, d = coord_mat.shape
    m = y_design.shape[2]
    h_rt = np.zeros((d, m))

    # calculate the hat matrix
    hat_mat = np.dot(np.dot(x_design, inv(np.dot(x_design.T, x_design)+np.eye(p)*0.000001)), x_design.T)

    # calculate the median absolute deviation (mad) on both coordinate data and response data

    for mii in range(m):
        res_y = np.dot(np.eye(n)-hat_mat, y_design[:, :, mii])
        h_y = robust.mad(res_y, axis=1) / 0.6745    # type: np.ndarray
        h_y_max = np.median(h_y)
        for dii in range(d):
            h_coord = robust.mad(coord_mat[:, dii]) / 0.6745
            h_rt[dii, mii] = 1.06*(1/n)**(1/(d+4))*np.sqrt(h_coord*h_y_max)
            # h_rt[dii, mii] = 1.06*(1/n)**(1/(d+4))*np.sqrt(h_coord*h_y_max)
            # the constant 1.06 is considered for Gaussian kernel

    return h_rt, hat_mat
