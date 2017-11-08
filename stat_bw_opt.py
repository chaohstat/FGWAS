"""
optimal bandwidth selection in multivariate varying coefficient model (MVCM).

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-09-18
"""

import numpy as np
from numpy.linalg import inv
from multiprocessing import Pool
from stat_bw import bw

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
        Outputs:
            h_opt (matrix): optimal choices of bandwidth (d*m)
            hat_mat (matrix): hat matrix (n*n)
    """

    # Set up
    n, p = x_design.shape
    d = coord_mat.shape[1]
    m = y_design.shape[2]
    resy_design = y_design * 0

    nh = 40
    vh = np.zeros((nh, d))
    for dii in range(d):
        coord_range = np.ptp(coord_mat[:, dii])
        h_min = 0.01  # minimum bandwidth
        h_max = 0.5 * coord_range  # maximum bandwidth
        vh[:, dii] = np.logspace(np.log10(h_min), np.log10(h_max), nh)  # candidate bandwidth
    gcv = np.zeros((nh, m))
    h_opt = np.zeros((d, m))

    # calculate the hat matrix
    hat_mat = np.dot(np.dot(x_design, inv(np.dot(x_design.T, x_design)+np.eye(p)*0.0001)), x_design.T)

    for mii in range(m):
        resy_design[:, :, mii] = np.dot(np.eye(n)-hat_mat, y_design[:, :, mii])

    processes = 10  # number of processes to to used
    pool = Pool(processes=processes)
    result = [pool.apply(bw, args=(resy_design, coord_mat, vh, h_ii)) for h_ii in np.arange(nh)]
    for ii in range(nh):
        gcv[ii, :] = result[ii]
    h_opt_idx = np.argmin(gcv, axis=0)

    for dii in range(d):
        for mii in range(m):
            h_opt[dii, mii] = vh[h_opt_idx[mii], dii]

    return h_opt, hat_mat
