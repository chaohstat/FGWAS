"""
Wild Bootstrap procedure in FGWAS.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-09-28
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig
from scipy.stats import chi2
from S2_GSIS import gsis
from stat_label_region import label_region

"""
installed all the libraries above
"""


def wild_bstp(snp_mat, y_design, resy_design, efit_eta, esig_eta, hat_mat,
              img_size, img_idx, c_alpha, g_num, b_num):
    """
        Significant locus-voxel testing procedure

        :param
            snp_mat (matrix): snp data (n*t)
            y_design (matrix): imaging response data (response matrix, n*l*m)
            resy_design (matrix): estimated difference between y_design and X*B
            efit_eta (matrix): the estimated of eta (n*l*m)
            esig_eta (matrix): the estimated covariance matrix of eta (m*m*l)
            hat_mat (matrix): hat matrix (n*n)
            img_size (vector): image dimension (1*d, d=2, 3)
            img_idx (vector): image index in non-background region (1*l)
            c_alpha (scalar): thresholding for labeling the significant subregions
            g_num (scalar): number of candidate snps
            b_num (scalar): number of Bootstrap sampling
    """

    # Set up
    n, l, m = y_design.shape

    res_y_bstp = np.zeros((n, l, m))
    smy_design = np.zeros((n, l, m))
    max_stat_bstp = np.zeros(shape=(b_num, 1))
    max_area_bstp = np.zeros(shape=(b_num, 1))

    y_mean = y_design-resy_design
    residual = resy_design-efit_eta

    all_zx_mat = np.dot(np.eye(n) - hat_mat, snp_mat).T
    inv_q_all_zx = np.sum(all_zx_mat * all_zx_mat, axis=1) ** (-1)

    for bii in range(b_num):

        l_stat_top = np.zeros((g_num, l))
        area_top = np.zeros((g_num, 1))

        for nii in range(n):
            rand_sub = np.random.normal(0, 1, 1)
            rand_vex = np.dot(np.atleast_2d(np.random.normal(0, 1, l)).T, np.ones((1, m)))
            res_y_bstp[nii, :, :] = rand_sub*efit_eta[nii, :, :] + rand_vex*residual[nii, :, :]

        y_bstp = y_mean + res_y_bstp

        for mii in range(m):
            smy_design[:, :, mii] = np.dot(hat_mat, y_bstp[:, :, mii])

        const = np.zeros((n, n, l))
        for lii in range(l):
            const[:, :, lii] = np.dot(np.dot(np.squeeze(smy_design[:, lii, :]), inv(esig_eta[:, :, lii])),
                                      np.squeeze(smy_design[:, lii, :]).T)
        qr_smy_mat = np.mean(const, axis=2)
        w, v = eig(qr_smy_mat)
        w = np.real(w)
        w[w < 0] = 0
        w_diag = np.diag(w ** (1 / 2))
        sq_qr_smy_mat = np.dot(np.dot(v, w_diag), v.T)
        sq_qr_smy_mat = np.real(sq_qr_smy_mat)

        g_stat = np.sum(np.dot(all_zx_mat, sq_qr_smy_mat) ** 2, axis=1)*inv_q_all_zx
        indx = np.argsort(-g_stat)
        top_snp_mat = snp_mat[:, indx[0:g_num]]
        zx_mat = np.dot(np.eye(n) - hat_mat, top_snp_mat).T
        inv_q_zx = np.sum(zx_mat * zx_mat, axis=1) ** (-1)

        const_all = no.zeros((n, n*l))
        for lii in range(l):
            const_all[:, (lii*n):((lii+1)*n)] = const[:, :, lii]
        for gii in range(g_num):
            temp_1 = np.dot(np.atleast_2d(zx_mat[:, gii]), const_all)
            temp = temp_1.reshape(l, n)
            l_stat_top[gii, :] = np.squeeze(np.dot(temp, np.atleast_2d(zx_mat[:, gii]).T))*inv_q_zx

        max_stat_bstp[bii, 0] = np.max(l_stat_top)

        for gii in range(g_num):
            k1 = np.mean(l_stat_top[gii, :])
            k2 = np.var(l_stat_top[gii, :])
            k3 = np.mean((l_stat_top[gii, :] - k1) ** 3)
            a = k3 / (4 * k2)
            b = k1 - 2 * k2 ** 2 / k3
            d = 8 * k2 ** 3 / k3 ** 2
            pv = 1 - chi2.cdf((l_stat_top[gii, :] - b) / a, d)
            pv_log10 = -np.log10(pv)
            area_top[gii, 0] = label_region(img_size, img_idx, pv_log10, c_alpha)
        max_area_bstp[bii, 0] = np.max(area_top)

    return max_stat_bstp, max_area_bstp
