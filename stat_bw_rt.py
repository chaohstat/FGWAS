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

        :param
            coord_mat (matrix): common coordinate matrix (n_v*d)
            x_design (matrix): non-genetic design matrix (n*p)
            y_design (matrix): imaging response data (response matrix, m*n*n_v)
        :return
            h_rt (matrix): optimal bandwidth matrix (n_v*d)
            hat_mat (matrix): hat matrix (n*n)
    """

    # Set up
    n, p = x_design.shape
    n_v, d = coord_mat.shape
    m = y_design.shape[0]
    h_rt = np.zeros((d, m))

    # calculate the hat matrix
    hat_mat = np.dot(np.dot(x_design, inv(np.dot(x_design.T, x_design)+np.eye(p)*0.000001)), x_design.T)

    # calculate the median absolute deviation (mad) on both coordinate data and response data

    for mii in range(m):
        z = y_design[mii, :, :]
        h_y = robust.mad(z, axis=1) / 0.6745    # type: np.ndarray
        h_y_max = np.max(h_y)
        for dii in range(d):
            h_coord = robust.mad(coord_mat[:, dii]) / 0.6745
            h_rt[dii, mii] = 1.06*(1/n)**(1/(d+4))*np.sqrt(h_coord*h_y_max)

    return h_rt, hat_mat
