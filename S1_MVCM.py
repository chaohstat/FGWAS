"""
Local linear kernel smoothing on MVCM in FGWAS.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2022-01-08
"""

import numpy as np
from numpy.linalg import inv
from stat_kernel import gau_kernel

"""
installed all the libraries above
"""


def mvcm(coord_mat, proj_y_design, h_opt):
    """
        Local linear kernel smoothing on MVCM in FGWAS.

        :param
            coord_mat (matrix): common coordinate matrix (n_v*d)
            proj_y_design (matrix): projected imaging response data (response matrix, m*n*n_v)
            h_opt (vector): optimal bandwidth (m)
        :return
            qr_smy_mat (matrix): constant matrix inside the test statistic (n*n)
            esig_eta (matrix): estimated covariance matrix of eta (n_v*m*m)
            smy_design (matrix): smoothed image response data (m*n*n_v)
            resy_design (matrix): estimated residual matrix (m*n*n_v)
            efit_eta (matrix): estimated eta matrix (m*n*n_v)
    """
    
    # Set up
    d = coord_mat.shape[1]
    m, n, n_v = proj_y_design.shape
    efit_eta = proj_y_design * 0
    
    w = np.zeros((1, d + 1))
    w[0,0] = 1
    t_mat0 = np.zeros((d + 1, n_v, n_v))  # L x L x d + 1 matrix
    t_mat0[0, :, :] = np.ones((n_v, n_v))
    
    for dii in range(d):
        t_mat0[dii + 1, :, :] = np.dot(np.atleast_2d(coord_mat[:, dii]).T, np.ones((1, n_v))) \
                                - np.dot(np.ones((n_v, 1)), np.atleast_2d(coord_mat[:, dii]))
    
    for mii in range(m):
        k_mat = np.ones((n_v, n_v))
        # t_mat = np.zeros((n_v, d+1, n_v))
        for dii in range(d):
            h = h_opt[dii]
            # t_mat[:,dii+1,:] = t_mat0[dii + 1, :, :] / h
            k_mat = k_mat * gau_kernel(t_mat0[dii+1,:,:]/h, h)  # Gaussian kernel smoothing function

        t_mat = np.transpose(t_mat0, [2,1,0])   
        
        for lii in range(n_v):
            kx = np.dot(np.atleast_2d(k_mat[:, lii]).T, np.ones((1, d + 1)))*t_mat[lii, :, :]  # n_v x d+1 matrix
            sm_weight = np.dot(np.dot(w, inv(np.dot(kx.T, t_mat[lii, :, :])+np.eye(d+1)*0.000001)), kx.T)
            efit_eta[mii, :, lii] = np.squeeze(np.dot(proj_y_design[mii, :, :], sm_weight.T))
    
    esig_eta = np.zeros((n_v, m, m))
    qr_smy_mat = np.zeros((n, n))
    for lii in range(n_v):
        esig_eta[lii, :, :] = np.dot(np.squeeze(efit_eta[:, :, lii]), np.squeeze(efit_eta[:, :, lii]).T)/n
        qr_smy_mat = qr_smy_mat+np.dot(np.dot(efit_eta[:, :, lii].T,
                                              inv(esig_eta[lii, :, :]+np.eye(m)*0.000001)),
                                       efit_eta[:, :, lii])/n_v
    
    
    return qr_smy_mat, efit_eta, esig_eta
