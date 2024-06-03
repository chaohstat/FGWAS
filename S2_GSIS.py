import numpy as np
from numpy.linalg import eig
from scipy.stats import chi2


def gsis(snp_mat, qr_smy_mat, proj_mat):
    """
        Global sure independence screening (GSIS) procedure.

        :param
            snp_mat (matrix): snp data (n*g)
            qr_smy_mat (matrix): common part in global test statistic (n*n)
            proj_mat (matrix): projection matrix (n*n)
        :return
         g_pv_log10 (vector): -log10 p-values of across all SNPs
         g_stat (vector): the global wald test statistics across all SNPs
    """

    # Set up
    n, g = snp_mat.shape

    # calculate the hat matrix
    zx_mat = np.dot(proj_mat, snp_mat).T
    inv_q_zx = np.sum(zx_mat*zx_mat, axis=1)**(-1)
    w, v = eig(qr_smy_mat)
    w = np.real(w)
    w[w < 0] = 0
    w_diag = np.diag(w**(1/2))
    sq_qr_smy_mat = np.dot(np.dot(v, w_diag), v.T)
    sq_qr_smy_mat = np.real(sq_qr_smy_mat)
    g_stat = np.sum(np.dot(zx_mat, sq_qr_smy_mat)**2, axis=1)*inv_q_zx

    # approximate of chi2 distribution
    k1 = np.mean(g_stat)
    k2 = np.var(g_stat)
    k3 = np.mean((g_stat-k1)**3)
    a = k3/(4*k2)
    b = k1-2*k2**2/k3
    d = 8*k2**3/k3**2
    g_pv = 1-chi2.cdf((g_stat-b)/a, d)
    g_pv_log10 = -np.log10(g_pv)

    return g_pv_log10, g_stat
