"""
Run main script: functional genome wide association analysis (FGWAS) pipeline (step 1 & step 2)
Usage: python ./test.py ./data/ ./result/

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-10-18
"""

import sys
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
from stat_read_x import read_x
from stat_bw_rt import bw_rt
from S1_MVCM import mvcm
from S2_GSIS import gsis

"""
installed all the libraries above
"""


def run_script(input_dir, output_dir):

    """
    Run the commandline script for FGWAS.

    :param
        input_dir (str): full path to the data folder
        output_dir (str): full path to the output folder
    """

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 0. load dataset """)
    print("+++++++Read the imaging data+++++++")
    img_file_name = input_dir + "img_data.mat"
    mat = loadmat(img_file_name)
    img_data = mat['img_data']
    if len(img_data.shape) == 2:
        img_data = img_data.reshape(1, img_data.shape[0], img_data.shape[1])
    m, n, n_v = img_data.shape
    y_design = np.log10(img_data)  # log transformation on response
    print("The matrix dimension of image data is " + str(img_data.shape))
    print("+++++++Read the imaging coordinate data+++++++")
    coord_file_name = input_dir + "coord_data.txt"
    coord_data = np.loadtxt(coord_file_name)
    print("The matrix dimension of coordinate data is " + str(coord_data.shape))
    print("+++++++Read the SNP data+++++++")
    snp_file_name = input_dir + "snp_data.txt"
    snp_data = np.loadtxt(snp_file_name)
    print("The matrix dimension of original snp data is " + str(snp_data.shape))
    print("+++++++Read the covariate data+++++++")
    design_data_file_name = input_dir + "design_data.txt"
    design_data = np.loadtxt(design_data_file_name)
    print("The matrix dimension of covariate data is " + str(design_data.shape))

    # read the covariate type
    var_type_file_name = input_dir + "var_type.txt"
    var_type = np.loadtxt(var_type_file_name)
    var_type = np.array([int(i) for i in var_type])

    print("+++++++++Matrix preparing and Data preprocessing++++++++")
    print("+++++++Construct the imaging response, design, coordinate matrix: normalization+++++++")
    x_design, coord_data = read_x(coord_data, design_data, var_type)
    p = x_design.shape[1]
    print("The dimension of normalized design matrix is " + str(x_design.shape))
    print("+++++++Preprocess SNP: filtering+++++++")
    max_num = np.zeros(shape=(3, snp_data.shape[1]))
    for i in range(3):
        bw = np.zeros(snp_data.shape)
        bw[snp_data == i] = 1
        max_num[i, :] = np.sum(bw, axis=0)
    max_num_idx = np.argmax(max_num, axis=0)
    indx = np.where(snp_data < 0)
    for i in range(len(indx[1])):
        snp_data[indx[0][i], indx[1][i]] = max_num_idx[indx[1][i]]

    min_maf = 0.05  # threshold for MAF
    maf = np.sum(snp_data, axis=0) / (2 * n)
    temp_idx = np.where(maf > 0.5)
    maf[temp_idx] = 1 - maf[temp_idx]
    rm_snp_index = np.where(maf <= min_maf)
    snp = np.delete(snp_data, rm_snp_index, axis=1)
    g = snp.shape[1]
    print("There are " + str(snp.shape[1]) + " snps with MAF>0.05.")

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 1. Fit the multivariate varying coefficient model (MVCM) under H0 """)
    start_1 = time.time()
    # find the optimal bandwidth
    h_opt, hat_mat = bw_rt(coord_data, x_design, y_design)
    print("the optimal bandwidth by Scott's Rule is ", h_opt)
    qr_smy_mat, esig_eta, smy_design, resy_design, efit_eta = mvcm(coord_data, y_design, h_opt, hat_mat)
    end_1 = time.time()
    print("Elapsed time in Step 1 is ", end_1 - start_1)
    for mii in range(m):
        res_mii = resy_design[mii, :, :]-efit_eta[mii, :, :]
        print("The bound of the residual is [" + str(np.min(res_mii)) + ", " + str(np.max(res_mii)) + "]")

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 2. Global sure independence screening (GSIS) """)
    start_2 = time.time()
    g_num = 1000  # number of top candidate snps
    g_pv_log10, g_stat = gsis(snp, qr_smy_mat, hat_mat)
    snp_pv = 10 ** (-g_pv_log10)
    top_snp_idx = np.argsort(-g_pv_log10)
    top_snp = snp[:, top_snp_idx[0:g_num]]
    snp_info_file = input_dir + "snp_info.map"
    fd = open(snp_info_file, 'r')
    snp_info = np.loadtxt(fd, delimiter='\t', dtype=bytes).astype(str)
    fd.close()
    snp_chr_tp = np.delete(snp_info[:, 0], rm_snp_index)
    snp_chr = np.array([int(i) for i in snp_chr_tp])
    snp_name = np.delete(snp_info[:, 1], rm_snp_index)
    snp_bp_tp = np.delete(snp_info[:, 3], rm_snp_index)
    snp_bp = np.array([int(i) for i in snp_bp_tp])
    gsis_all = np.vstack((snp_chr, snp_bp, snp_pv)).T  # input for plotting Manhattan plot
    top_snp_chr = snp_chr[top_snp_idx[0:g_num]]
    top_snp_name = snp_name[top_snp_idx[0:g_num]]
    top_snp_bp = snp_bp[top_snp_idx[0:g_num]]
    top_snp_pv_log10 = g_pv_log10[top_snp_idx[0:g_num]]
    gsis_top = np.vstack((top_snp_name, top_snp_chr, top_snp_bp, top_snp_pv_log10)).T  # top SNP GSIS results
    gsis_all_file_name = output_dir + "GSIS_all.txt"
    np.savetxt(gsis_all_file_name, gsis_all, delimiter="\t", fmt="%d %d %f")
    gsis_top_file_name = output_dir + "GSIS_top.txt"
    np.savetxt(gsis_top_file_name, gsis_top, delimiter="\t", fmt="%s", comments='',
               header="SNP\tCHR\tBP\tP")
    end_2 = time.time()
    print("Elapsed time in Step 2 is ", end_2 - start_2)

    # save results in temp folder for next step
    start_3 = time.time()
    temp_dir = output_dir + "/temp/"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    data_dim = np.array([n, n_v, m, p, g, g_num])
    data_dim_file_name = temp_dir + "data_dim.mat"
    savemat(data_dim_file_name, mdict={'data_dim': data_dim})
    all_snp_file_name = temp_dir + "snp.mat"
    savemat(all_snp_file_name, mdict={'snp': snp})
    top_snp_file_name = temp_dir + "top_snp.mat"
    savemat(top_snp_file_name, mdict={'top_snp': top_snp})
    y_design_file_name = temp_dir + "y_design.mat"
    savemat(y_design_file_name, mdict={'y_design': y_design})
    resy_design_file_name = temp_dir + "resy_design.mat"
    savemat(resy_design_file_name, mdict={'resy_design': resy_design})
    efit_eta_file_name = temp_dir + "efit_eta.mat"
    savemat(efit_eta_file_name, mdict={'efit_eta': efit_eta})
    esig_eta_file_name = temp_dir + "esig_eta.mat"
    savemat(esig_eta_file_name, mdict={'esig_eta': esig_eta})
    hat_mat_file_name = temp_dir + "hat_mat.mat"
    savemat(hat_mat_file_name, mdict={'hat_mat': hat_mat})
    end_3 = time.time()
    print("Elapsed time in saving temp results is ", end_3 - start_3)


if __name__ == '__main__':
    input_dir0 = sys.argv[1]
    output_dir0 = sys.argv[2]
    run_script(input_dir0, output_dir0)
