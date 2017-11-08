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
            coord_mat (matrix): common coordinate matrix (l*d)
            y_design (matrix): imaging response data (response matrix, n*l*m)
            h_opt (matrix): optimal bandwidth (d*m)
            hat_mat (matrix): hat matrix (n*n)
        :return
            qr_smy_mat (matrix): constant matrix inside the test statistic (n*n)
            esig_eta (matrix): estimated covariance matrix of eta (m*m*l)
            smy_design (matrix): smoothed image response data (n*l*m)
            resy_design (matrix): estimated residual matrix (*n*l*m)
            efit_eta (matrix): estimated eta matrix (n*l*m)
    """

    # Set up
    d = coord_mat.shape[1]
    n, l, m = y_design.shape
    resy_design = y_design * 0
    smy_design = y_design * 0
    efit_eta = y_design * 0

    w = np.zeros((1, d + 1))
    w[0] = 1
    t_mat0 = np.zeros((d + 1, l, l))  # L x L x d + 1 matrix
    t_mat0[0, :, :] = np.ones((l, l))

    for dii in range(d):
        t_mat0[dii + 1, :, :] = np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, l))) \
                                - np.dot(np.ones((l, 1)), np.atleast_2d(coord_mat[:, dii]))

    for mii in range(m):

        k_mat = np.ones((l, l))

        for dii in range(d):
            h = h_opt[dii, mii]
            k_mat = k_mat * gau_kernel(t_mat0[dii + 1, :, :] / h, h)  # Gaussian kernel smoothing function

        t_mat = np.transpose(t_mat0, [2, 1, 0])  # L x d+1 x L matrix

        smy_design[:, :, mii] = np.dot(hat_mat, y_design[:, :, mii])
        resy_design[:, :, mii] = y_design[:, :, mii]-smy_design[:, :, mii]

        for lii in range(l):
            kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[lii, :, :]  # L0 x d+1 matrix
            sm_weight = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[lii, :, :])+np.eye(d+1)*0.000001)), kx.T)
            efit_eta[:, lii, mii] = np.squeeze(np.dot(resy_design[:, :, mii], sm_weight.T))

    esig_eta = np.zeros((m, m, l))
    qr_smy_mat = np.zeros((n, n))
    for lii in range(l):
        esig_eta[:, :, lii] = np.dot(np.squeeze(efit_eta[:, lii, :]).T, np.squeeze(efit_eta[:, lii, :]))/n
        qr_smy_mat = qr_smy_mat+np.dot(np.dot(np.squeeze(smy_design[:, lii, :]),
                                              inv(esig_eta[:, :, lii])),
                                       np.squeeze(smy_design[:, lii, :]).T)/l

    return qr_smy_mat, esig_eta, smy_design, resy_design, efit_eta
