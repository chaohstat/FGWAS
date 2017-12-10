"""
Local linear kernel smoothing on MVCM in FGWAS.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-09-18
"""

import numpy as np
from numpy.linalg import inv
from stat_kernel import gau_kernel

"""
installed all the libraries above
"""


def mvcm(coord_mat, y_design, h_opt, hat_mat):
    """
        Local linear kernel smoothing on MVCM in FGWAS.

        :param
            coord_mat (matrix): common coordinate matrix (n_v*d)
            y_design (matrix): imaging response data (response matrix, m*n*n_v)
            h_opt (matrix): optimal bandwidth (d*m)
            hat_mat (matrix): hat matrix (n*n)
        :return
            qr_smy_mat (matrix): constant matrix inside the test statistic (n*n)
            esig_eta (matrix): estimated covariance matrix of eta (n_v*m*m)
            smy_design (matrix): smoothed image response data (m*n*n_v)
            resy_design (matrix): estimated residual matrix (m*n*n_v)
            efit_eta (matrix): estimated eta matrix (m*n*n_v)
    """

    # Set up
    d = coord_mat.shape[1]
    m, n, n_v = y_design.shape
    resy_design = y_design * 0
    smy_design = y_design * 0
    efit_eta = y_design * 0

    w = np.zeros((1, d + 1))
    w[0] = 1
    t_mat0 = np.zeros((d + 1, n_v, n_v))  # L x L x d + 1 matrix
    t_mat0[0, :, :] = np.ones((n_v, n_v))

    for dii in range(d):
        t_mat0[dii + 1, :, :] = np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, n_v))) \
                                - np.dot(np.ones((n_v, 1)), np.atleast_2d(coord_mat[:, dii]))

    for mii in range(m):

        k_mat = np.ones((n_v, n_v))

        for dii in range(d):
            h = h_opt[dii, mii]
            k_mat = k_mat * gau_kernel(t_mat0[dii + 1, :, :] / h, h)  # Gaussian kernel smoothing function

        t_mat = np.transpose(t_mat0, [2, 1, 0])  # L x d+1 x L matrix

        smy_design[mii, :, :] = np.dot(hat_mat, y_design[mii, :, :])
        resy_design[mii, :, :] = y_design[mii, :, :]-smy_design[mii, :, :]

        for lii in range(n_v):
            kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[lii, :, :]  # L0 x d+1 matrix
            sm_weight = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[lii, :, :])+np.eye(d+1)*0.000001)), kx.T)
            efit_eta[mii, :, lii] = np.squeeze(np.dot(resy_design[mii, :, :], sm_weight.T))

    esig_eta = np.zeros((n_v, m, m))
    qr_smy_mat = np.zeros((n, n))
    for lii in range(n_v):
        esig_eta[lii, :, :] = np.dot(np.squeeze(efit_eta[:, :, lii]), np.squeeze(efit_eta[:, :, lii]).T)/n
        qr_smy_mat = qr_smy_mat+np.dot(np.dot(np.squeeze(smy_design[:, :, lii]).T,
                                              inv(esig_eta[lii, :, :])),
                                       np.squeeze(smy_design[:, :, lii]))/n_v

    return qr_smy_mat, esig_eta, smy_design, resy_design, efit_eta
