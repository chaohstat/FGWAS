import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2
from stat_kernel import gau_kernel
from S2_GSIS import gsis
from stat_label_region import label_region

def wild_bstp(snp_mat, proj_y_design, efit_eta, proj_mat, coord_mat, h_opt,
            img_size, img_idx, alpha_log10, g_num, b_num):
    """
        Significant locus-voxel testing procedure

        :param
            snp_mat (matrix): projected snp data (n*t)
            proj_y_design (matrix): projected imaging response data (response matrix, m*n*l)
            efit_eta (matrix): estimated eta (m*n*l)
            proj_mat (matrix): projection matrix (n*n)
            coord_mat (matrix): coordinate matrix (l*d)
            h_opt (vector): optimal bandwidth (len=d)
            img_size (vector): image dimension (1*d, d=2, 3)
            img_idx (vector): image index in non-background region (1*l)
            alpha_log10 (scalar): thresholding for labeling the significant subregions
            g_num (scalar): number of candidate snps
            b_num (scalar): number of Bootstrap sampling
    """
    
    # Set up
    m, n, l = proj_y_design.shape
    
    proj_y_bstp = 0*proj_y_design
    
    max_lstat_bstp = np.zeros(shape=(b_num, 1))
    max_gstat_bstp = np.zeros(shape=(b_num, 1))
    max_area_bstp = np.zeros(shape=(b_num, 1))
    
    residual = proj_y_design-efit_eta
    
    for bii in range(b_num):
        
        l_stat_top = np.zeros((g_num, l))
        area_top = np.zeros((g_num, 1))
        
        for nii in range(n):
            rand_sub = np.random.normal(0, 1, 1)
            rand_vex = np.dot(np.ones((m,1)), np.atleast_2d(np.random.normal(0, 1, l)))
            proj_y_bstp[:, nii, :] = rand_sub*efit_eta[:, nii, :] + rand_vex*residual[:, nii, :]

        ################### smooth proj_y_bstp #########################
        d = coord_mat.shape[1]
        efit_eta_bstp = proj_y_bstp * 0

        w = np.zeros((1, d + 1))
        w[0] = 1
        t_mat0 = np.zeros((d + 1, l, l))  # d + 1 x L x L matrix
        t_mat0[0, :, :] = np.ones((l, l))

        for dii in range(d):
            t_mat0[dii + 1, :, :] = (np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, l))) \
                                - np.dot(np.ones((l, 1)), np.atleast_2d(coord_mat[:, dii])))/h_opt[dii]

        for mii in range(m):

            k_mat = np.ones((l, l))

            for dii in range(d):
                h = h_opt[dii]
                k_mat = k_mat * gau_kernel(t_mat0[dii + 1, :, :], h)  # Gaussian kernel smoothing function

            t_mat = np.transpose(t_mat0, [2, 1, 0])  # L x d+1 x L matrix

            for lii in range(l):
                kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[lii, :, :]  # L0 x d+1 matrix
                sm_weight = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[lii, :, :])+np.eye(d+1)*0.000001)), kx.T)
                efit_eta_bstp[mii, :, lii] = np.squeeze(np.dot(proj_y_bstp[mii, :, :], sm_weight.T))
        
        const = np.zeros((n, n))
        for lii in range(l):
            if m==1:
                esig_eta_bstp_lii_inv = n/np.dot(efit_eta_bstp[:, :, lii], efit_eta_bstp[:, :, lii].T) # a number
                const = const + esig_eta_bstp_lii_inv * np.dot(efit_eta_bstp[:, :, lii].T, efit_eta_bstp[:, :, lii]) # n x n
            else:
                esig_eta_bstp_lii_inv = n * inv(np.dot(efit_eta_bstp[:, :, lii], efit_eta_bstp[:, :, lii].T)) # m x m
                const = const + np.dot(np.dot(efit_eta_bstp[:, :, lii].T, esig_eta_bstp_lii_inv), efit_eta_bstp[:, :, lii]) # n x n
        qr_smy_mat = const / l
        
        ################### test statistic & p-value ###################
        wg_pv_log10, g_stat = gsis(snp_mat, qr_smy_mat, proj_mat)
        max_gstat_bstp[bii, 0] = np.max(g_stat)   #  np.max(np.mean(l_stat_top, axis=1))
        
        indx = np.argsort(-g_stat)
        top_snp_mat = snp_mat[:, indx[0:g_num]]
        zx_mat = np.dot(proj_mat, top_snp_mat).T
        inv_q_zx = np.sum(zx_mat * zx_mat, axis=1) ** (-1)
        
        const = np.zeros((n, n))
        for lii in range(l):
            if m==1:
                esig_eta_bstp_lii_inv = n/np.dot(efit_eta_bstp[:, :, lii], efit_eta_bstp[:, :, lii].T) # a number
                const = esig_eta_bstp_lii_inv * np.dot(efit_eta_bstp[:, :, lii].T, efit_eta_bstp[:, :, lii]) # n x n
            else:
                esig_eta_bstp_lii_inv = n * inv(np.dot(efit_eta_bstp[:, :, lii], efit_eta_bstp[:, :, lii].T)) # m x m
                const = np.dot(np.dot(efit_eta_bstp[:, :, lii].T, esig_eta_bstp_lii_inv), efit_eta_bstp[:, :, lii])

        for gii in range(g_num):
            for lii in range(l):
                if m == 1:
                    esig_eta_bstp_lii_inv = n / np.dot(efit_eta_bstp[:, :, lii], efit_eta_bstp[:, :, lii].T)  # a number
                    const = esig_eta_bstp_lii_inv * np.dot(efit_eta_bstp[:, :, lii].T, efit_eta_bstp[:, :, lii]) # n x n
                else:
                    esig_eta_bstp_lii_inv = n * inv(np.dot(efit_eta_bstp[:, :, lii], efit_eta_bstp[:, :, lii].T))  # m x m
                    const = np.dot(np.dot(efit_eta_bstp[:, :, lii].T, esig_eta_bstp_lii_inv), efit_eta_bstp[:, :, lii])
                temp = np.dot(np.atleast_2d(zx_mat[gii, :]), const)
                l_stat_top[gii, lii] = np.squeeze(np.dot(temp, np.atleast_2d(zx_mat[gii, :]).T))*inv_q_zx[gii]
        
        max_lstat_bstp[bii, 0] = np.max(l_stat_top)
        
        for gii in range(g_num):
            k1 = np.mean(l_stat_top[gii, :])
            k2 = np.var(l_stat_top[gii, :])
            k3 = np.mean((l_stat_top[gii, :] - k1) ** 3)
            a = k3 / (4 * k2)
            b = k1 - 2 * k2 ** 2 / k3
            d = 8 * k2 ** 3 / k3 ** 2
            pv = 1 - chi2.cdf((l_stat_top[gii, :] - b) / a, d)
            pv_log10 = -np.log10(pv)
            area_top[gii, 0] = label_region(img_size, img_idx, pv_log10, alpha_log10)
        
        max_area_bstp[bii, 0] = np.max(area_top)
    
    return max_gstat_bstp, max_lstat_bstp, max_area_bstp
