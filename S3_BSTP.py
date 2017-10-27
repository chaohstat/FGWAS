"""
Wild Bootstrap procedure in FGWAS.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-09-28
"""

import numpy as np
from numpy.linalg import inv
from S2_GSIS import gsis
from stat_label_region import label_region

"""
installed all the libraries above
"""


def wild_bstp(snp_mat, y_design, resy_design, efit_eta, esig_eta, sm_weight, hat_mat,
              img_size, img_idx, c_alpha, g_num, b_num):
    """
        Significant locus-voxel testing procedure

        Args:
            snp_mat (matrix): snp data (n*t)
            y_design (matrix): imaging response data (response matrix, n*l*m)
            resy_design (matrix): estimated difference between y_design and X*B
            efit_eta (matrix): the estimated of eta (n*l*m)
            esig_eta (matrix): the estimated covariance matrix of eta (m*m*l)
            sm_weight (matrix): the smoothing weights matrix (l*l*m)
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

    for bii in range(b_num):

        for nii in range(n):
            rand_sub = np.random.normal(0, 1, 1)
            rand_vex = np.dot(np.atleast_2d(np.random.normal(0, 1, l)).T, np.ones((1, m)))
            res_y_bstp[nii, :, :] = rand_sub*efit_eta[nii, :, :] + rand_vex*residual[nii, :, :]

        y_bstp = y_mean + res_y_bstp

        for mii in range(m):
            for lii in range(l):
                sm_weight_lii = np.atleast_2d(sm_weight[lii, :, mii]).T
                smy_design[:, lii, mii] = np.squeeze(np.dot(y_bstp[:, :, mii], sm_weight_lii))

        const = np.zeros((n, n, l))
        for lii in range(l):
            const[:, :, lii] = np.dot(np.dot(np.squeeze(smy_design[:, lii, :]), inv(esig_eta[:, :, lii])),
                                      np.squeeze(smy_design[:, lii, :]).T) / l
        qr_smy_mat = np.sum(const, axis=2)

        g_pv_log10 = gsis(snp_mat, qr_smy_mat, hat_mat)[0]
        indx = np.argsort(-g_pv_log10)
        top_snp_mat = snp_mat[:, indx[1:g_num]]

        l_stat_top = np.zeros((g_num, l))
        area_top = np.zeros((g_num, 1))
        for lii in range(l):
            l_stat_top[:, lii] = gsis(top_snp_mat, const[:, :, lii], hat_mat)[1]
        max_stat_bstp[bii, 0] = np.max(l_stat_top)
        for gii in range(g_num):
            area_top[gii, 0] = label_region(img_size, img_idx, l_stat_top[gii, :], c_alpha)
        max_area_bstp[bii, 0] = np.max(area_top)

    return max_stat_bstp, max_area_bstp
