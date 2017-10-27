"""
Significant locus-voxel and locus-subregion testing procedure in FGWAS.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-09-28
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig
from stat_label_region import label_region
from scipy.stats import chi2

"""
installed all the libraries above
"""


def local_test(top_snp_mat, esig_eta, smy_design, hat_mat, img_size, img_idx, c_alpha, max_stat_bstp, max_area_bstp):
    """
        Significant locus-voxel testing procedure

        Args:
            top_snp_mat (matrix): snp data (n*t)
            esig_eta (matrix): the estimated variance of eta (m*m*l)
            smy_design (matrix): smoothed response (n*l*m)
            hat_mat (matrix): hat matrix (n*n)
            img_size (vector): image dimension (1*d, d=2, 3)
            img_idx (vector): image index in non-background region (1*l)
            c_alpha (scalar): thresholding for labeling the significant subregions
            max_stat_bstp (vector): Bootstrap sampling for construction null distribution of local test statistic (1*l)
            max_area_bstp (vector): Bootstrap sampling for construction null distribution of significant subregion (1*l)
    """

    # Set up
    g_num = top_snp_mat.shape[1]
    n, l, m = smy_design.shape
    l_stat = np.zeros((g_num, l))
    cluster_pv = np.zeros(g_num)

    zx_mat = np.dot(np.eye(n) - hat_mat, top_snp_mat).T
    inv_q_zx = np.sum(zx_mat * zx_mat, axis=1) ** (-1)

    const_d = np.zeros((n, n, l))
    for lii in range(l):
        const_d[:, :, lii] = np.dot(np.dot(np.squeeze(smy_design[:, lii, :]), inv(esig_eta[:, :, lii])),
                                    np.squeeze(smy_design[:, lii, :]).T)
        w, v = eig(const_d[:, :, lii])
        w[w < 0] = 0
        w_diag = np.diag(w ** (1 / 2))
        sq_qr_smy_mat = np.dot(np.dot(v, w_diag), v.T)
        l_stat[:, lii] = np.sum(np.dot(zx_mat, sq_qr_smy_mat) ** 2, axis=1) * inv_q_zx

    # approximate of chi2 distribution
    k1 = np.mean(max_stat_bstp)
    k2 = np.var(max_stat_bstp)
    k3 = np.mean((max_stat_bstp - k1) ** 3)
    a = k3 / (4 * k2)
    b = k1 - 2 * k2 ** 2 / k3
    d = 8 * k2 ** 3 / k3 ** 2
    l_pv_adj = 1 - chi2.cdf((l_stat - b) / a, d)

    for gii in range(g_num):
        max_area = label_region(img_size, img_idx, l_stat[gii, :], c_alpha)
        cluster_pv[gii] = len(np.where(max_area_bstp >= max_area))/len(max_area_bstp)

    return l_pv_adj, l_stat, cluster_pv
