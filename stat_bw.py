"""
optimal bandwidth selection (Scott's Rule) in multivariate varying coefficient model (MVCM).

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-09-18
"""

import numpy as np
from numpy.linalg import inv
from stat_kernel import ep_kernel

"""
installed all the libraries above
"""


def bw(resy_design, coord_mat, vh, h_ii):
    """
        optimal bandwidth selection in MVCM.

        Args:
            resy_design (matrix): residuals of imaging response data (n*l*m)
            coord_mat (matrix): common coordinate matrix (l*d)
            vh (matrix): candidate bandwidth matrix (nh*d)
            h_ii (scalar): candidate bandwidth index
        Outputs:
            gcv (vector): generalized cross validation index (m)
    """

    # Set up
    l, d = coord_mat.shape
    m = resy_design.shape[2]
    efit_resy = resy_design * 0
    gcv = np.zeros(m)

    w = np.zeros((1, d + 1))
    w[0] = 1

    t_mat0 = np.zeros((l, l, d + 1))  # L x L x d + 1 matrix
    t_mat0[:, :, 0] = np.ones((l, l))

    for dii in range(d):
        t_mat0[:, :, dii + 1] = np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, l))) \
                                - np.dot(np.ones((l, 1)), np.atleast_2d(coord_mat[:, dii]))

    t_mat = np.transpose(t_mat0, [0, 2, 1])  # L x d+1 x L matrix

    k_mat = np.ones((l, l))

    for dii in range(d):
        h = vh[h_ii, dii]
        k_mat = k_mat * ep_kernel(t_mat0[:, :, dii + 1] / h, h)  # Epanechnikov kernel smoothing function

        for mii in range(m):
            for lii in range(l):
                kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[:, :, lii]  # L0 x d+1 matrix
                sm_weight = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[:, :, lii])+np.eye(d+1)*0.0001)), kx.T)
                efit_resy[:, lii, mii] = np.squeeze(np.dot(resy_design[:, :, mii], sm_weight.T))

            gcv[mii] = np.sum((efit_resy[:, :, mii]-resy_design[:, :, mii])**2)

    return gcv
