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


def bw_opt(coord_mat, x_design, y_design):
    """
        optimal bandwidth selection in MVCM.

        Args:
            coord_mat (matrix): common coordinate matrix (l*d)
            x_design (matrix): non-genetic design matrix (n*p)
            y_design (matrix): imaging response data (response matrix, n*l*m)
    """

    # Set up
    n, p = x_design.shape
    l, d = coord_mat.shape
    m = y_design.shape[2]
    h_coord = np.zeros((1, d))
    h_y = np.zeros((1, m))
    h_opt = np.zeros((d, m))

    # calculate the hat matrix
    hat_mat = np.dot(np.dot(x_design, inv(np.dot(x_design.T, x_design)+np.eye(p)*0.0001)), x_design.T)
    qhat_mat = np.eye(n)-hat_mat

    # calculate the median absolute deviation (mad) on both coordinate data and response data
    for mii in range(m):
        res_y = np.mean(np.dot(qhat_mat, y_design[:, :, mii]), axis=0)
        h_y[0, mii] = robust.mad(res_y)/0.6745
    for dii in range(d):
        h_coord[0, dii] = robust.mad(coord_mat[:, dii])/0.6745

    for mii in range(m):
        for dii in range(d):
            h_opt[dii, mii] = (1/n)**(1/(d+4))*np.sqrt(h_coord[0, dii]*h_y[0, mii])

    return h_opt, hat_mat
