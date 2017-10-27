"""
Run main script: functional genome wide association analysis (FGWAS) pipeline
Usage: python ./FGWAS_run_script.py ./data/ ./result/

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-10-18
"""

# import os
import time
from scipy.io import loadmat
import numpy as np
from scipy.stats import f
from multiprocessing import Pool
from stat_read_x import read_x
from S0_bw_opt import bw_opt
from S1_MVCM import mvcm
from S2_GSIS import gsis
from S3_BSTP import wild_bstp
from S3_TEST import local_test

"""
installed all the libraries above
"""


def run_script(input_dir, output_dir):

    """
    Run the commandline script for FGWAS.

    Args:
        input_dir (str): full path to the data folder
        output_dir (str): full path to the output folder
    """

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 0. load dataset """)
    print("+++++++Read the imaging data+++++++")
    img_file_name = input_dir + "img_data.mat"
    mat = loadmat(img_file_name)
    y_design = mat['img_data']
    n, l, m = y_design.shape
    print("The matrix dimension of image data is " + str(y_design.shape))
    print("+++++++Read the imaging coordinate data+++++++")
    coord_file_name = input_dir + "coord_data.txt"
    coord_data = np.loadtxt(coord_file_name)
    # d = coord_data.shape[1]
    print("The matrix dimension of coordinate data is " + str(coord_data.shape))
    print("+++++++Read the SNP data+++++++")
    snp_file_name = input_dir + "snp_data.txt"
    snp_data = np.loadtxt(snp_file_name)
    # g = snp_data.shape[1]
    print("The matrix dimension of original snp data is " + str(snp_data.shape))
    print("+++++++Read the covariate data+++++++")
    design_data_file_name = input_dir + "design_data.txt"
    design_data = np.loadtxt(design_data_file_name)
    print("The matrix dimension of covariate data is " + str(design_data.shape))

    # read the covariate type
    var_type_file_name = input_dir + "var_type.txt"
    var_type = np.loadtxt(var_type_file_name)
    # read the image size
    img_size_file_name = input_dir + "img_size.txt"
    img_size = np.loadtxt(img_size_file_name)
    # read the image index of non-background region
    img_idx_file_name = input_dir + "img_idx.txt"
    img_idx = np.loadtxt(img_idx_file_name)

    print("+++++++++Matrix preparing and Data preprocessing++++++++")
    print("+++++++Construct the design matrix: normalization+++++++")
    x_design = read_x(design_data, var_type)
    p = x_design.shape[1]
    print("The dimension of normalized design matrix is " + str(x_design.shape))
    print("+++++++Preprocess SNP: filtering+++++++")
    max_num = np.zeros(3, snp_data.shape[1])
    for i in range(3):
        bw = np.zeros(snp_data.shape)
        bw[snp_data == i] = 1
        max_num[i, :] = np.sum(bw, axis=0)
    max_num_idx = np.argmax(max_num, axis=0)
    indx = np.where(snp_data < 0)
    for i in range(len(indx[1])):
        snp_data[indx[0][i], indx[1][i]] = max_num_idx[indx[1][i]]

    min_maf = 0.05   # threshold for MAF
    maf = np.sum(snp_data, axis=0)/(2*n)
    temp_idx = np.where(maf > 0.5)
    maf[temp_idx] = 1 - maf[temp_idx]
    rm_snp_index = np.where(maf <= min_maf)
    snp = np.delete(snp_data, rm_snp_index, axis=1)
    print("There are" + str(snp.shape[1]) + "snps with MAF>0.05.")

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 1. Fit the multivariate varying coefficient model (MVCM) """)
    start_1 = time.time()
    # find the optimal bandwidth
    h_opt, hat_mat = bw_opt(coord_data, x_design, y_design)
    qr_smy_mat, esig_eta, smy_design, resy_design, efit_eta, sm_weight = \
        mvcm(coord_data, x_design, y_design, h_opt, hat_mat)
    end_1 = time.time()
    print("Elapsed time in Step 1 is ", end_1 - start_1)

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 2. Global sure independence screening (GSIS) """)
    start_2 = time.time()
    g_num = 1000   # number of top candidate snps
    g_pv_log10 = gsis(snp, qr_smy_mat, hat_mat)[0]
    snp_pv = 10 ** (-g_pv_log10)
    top_snp_idx = np.argsort(-g_pv_log10)
    top_snp = snp[:, top_snp_idx[0:g_num]]
    snp_info_file = output_dir + "snp_info.map"
    fd = open(snp_info_file, 'r')
    snp_chr, snp_name, snp_bp = np.loadtxt(fd, delimiter='\t',
                                           usecols=(0, 1, 3), dtype={'names': ('snp_chr', 'snp_name', 'snp_bp'),
                                                                     'formats': ('i4', 'S16', 'i4')})
    fd.close()
    snp_chr = np.delete(snp_chr, rm_snp_index)
    snp_name = np.delete(snp_name, rm_snp_index)
    snp_bp = np.delete(snp_bp, rm_snp_index)
    gsis_all = np.vstack((snp_chr, snp_bp, snp_pv)).T   # input for plotting Manhattan plot
    top_snp_chr = snp_chr[top_snp_idx[0:g_num]]
    top_snp_name = snp_name[top_snp_idx[0:g_num]]
    top_snp_bp = snp_bp[top_snp_idx[0:g_num]]
    top_snp_pv_log10 = g_pv_log10[top_snp_idx[0:g_num]]
    gsis_top = np.vstack((top_snp_name, top_snp_chr, top_snp_bp, top_snp_pv_log10)).T   # top SNP GSIS results
    end_2 = time.time()
    print("Elapsed time in Step 2 is ", end_2 - start_2)

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 3. Significant locus-voxel and locus-subregion detection """)
    start_3 = time.time()
    alpha = 1e-5
    c_alpha = f.ppf(1-alpha, dfn=1, dfd=n-p)
    processes = 5  # number of processes to to used
    pool = Pool(processes=processes)
    b_num = 100 * np.ones(processes)  # number of Bootstrap sampling = 500
    result_bstp = [pool.apply_async(wild_bstp(snp, y_design, resy_design, efit_eta, esig_eta, sm_weight, hat_mat,
                                              img_size, img_idx, c_alpha, g_num, b_num0)) for b_num0 in b_num]
    result_bstp = [p.get() for p in result_bstp]
    max_stat_bstp = np.zeros(shape=(100, processes))
    max_area_bstp = np.zeros(shape=(100, processes))
    for r in range(processes):
        max_stat_bstp[:, ] = result_bstp[r][0]
        max_area_bstp[:, ] = result_bstp[r][0]
    max_stat_bstp = np.reshape(max_stat_bstp, (100*processes, ))
    max_area_bstp = np.reshape(max_area_bstp, (100*processes, ))
    l_pv_adj, l_stat, cluster_pv = local_test(top_snp, esig_eta, smy_design, hat_mat, img_size,
                                              img_idx, c_alpha, max_stat_bstp, max_area_bstp)
    l_pv_raw = f.cdf(l_stat, dfn=1, dfd=n-p)
    end_3 = time.time()
    print("Elapsed time in Step 3 is ", end_3 - start_3)

    """+++++++++++++++++++++++++++++++++++"""
    """ Step4. Save all the results """
    gsis_all_file_name = output_dir + "GSIS_all.txt"
    np.savetxt(gsis_all_file_name, gsis_all)
    gsis_top_file_name = output_dir + "GSIS_top.txt"
    np.savetxt(gsis_top_file_name, gsis_top)
    l_pv_raw_file_name = output_dir + "local_pv_raw.txt"
    np.savetxt(l_pv_raw_file_name, l_pv_raw)
    l_pv_adj_file_name = output_dir + "local_pv_adj.txt"
    np.savetxt(l_pv_adj_file_name, l_pv_adj)
    cluster_pv_file_name = output_dir + "cluster_pv.txt"
    np.savetxt(cluster_pv_file_name, cluster_pv)
