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


def mvcm(coord_mat, x_design, y_design, h_opt, hat_mat):
    """
        Local linear kernel smoothing on MVCM in FGWAS.

        Args:
            coord_mat (matrix): common coordinate matrix (l*d)
            x_design (matrix): design matrix (n*p)
            y_design (matrix): imaging response data (response matrix, n*l*m)
            h_opt (matrix): optimal bandwidth (d*m)
            hat_mat (matrix): hat matrix (n*n)
    """

    # Set up
    n, p = x_design.shape
    l, d = coord_mat.shape
    m = y_design.shape[2]
    resy_design = y_design * 0
    smy_design = y_design * 0
    efit_eta = y_design * 0

    w = np.zeros((1, d + 1))
    w[0] = 1
    t_mat0 = np.zeros((l, l, d + 1))  # L x L x d + 1 matrix
    sm_weight = np.zeros((l, l, m))   # L x L x m matrix
    t_mat0[:, :, 1] = np.ones((l, l))

    for dii in range(d):
        t_mat0[:, :, dii + 1] = np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, l))) \
                                - np.dot(np.ones((l, 1)), np.atleast_2d(coord_mat[:, dii]))

    for mii in range(m):

        k_mat = np.ones((l, l))

        for dii in range(d):
            h = h_opt[dii, mii]
            k_mat = k_mat * gau_kernel(t_mat0[:, :, dii + 1] / h, h)  # Epanechnikov kernel smoothing function

        t_mat = np.transpose(t_mat0, [0, 2, 1])  # L x d+1 x L matrix

        for lii in range(l):
            kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[:, :, lii]  # L0 x d+1 matrix
            hat_y_design = np.dot(np.eye(n)-hat_mat, y_design[:, :, mii])
            sm_weight_lii = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[:, :, lii])+np.eye(d+1)*0.0001)), kx.T)
            smy_design[:, lii, mii] = np.squeeze(np.dot(y_design[:, :, mii], sm_weight_lii.T))
            resy_design[:, lii, mii] = np.squeeze(np.dot(hat_y_design, sm_weight_lii.T))
            sm_weight[lii, :, mii] = sm_weight_lii

        for lkk in range(l):
            sm_weight_lkk = sm_weight[lkk, :, mii]
            efit_eta[:, lkk, mii] = np.squeeze(np.dot(resy_design[:, :, mii], np.atleast_2d(sm_weight_lkk).T))

    esig_eta = np.zeros((m, m, l))
    qr_smy_mat = np.zeros((n, n))
    for lii in range(l):
        esig_eta[:, :, lii] = np.dot(np.squeeze(efit_eta[:, lii, :]).T, np.squeeze(efit_eta[:, lii, :]))/n
        qr_smy_mat = qr_smy_mat+np.dot(np.dot(np.squeeze(smy_design[:, lii, :]), inv(esig_eta[:, :, lii])),
                             np.squeeze(smy_design[:, lii, :]).T)/l

    return qr_smy_mat, esig_eta, smy_design, resy_design, efit_eta, sm_weight
