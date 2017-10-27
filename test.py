"""
Run main script: functional genome wide association analysis (FGWAS) pipeline
Usage: python ./test.py ./data/ ./result/

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-10-18
"""

import sys
import time
from scipy.io import loadmat
import numpy as np
# from scipy.stats import f
# from multiprocessing import Pool
from stat_read_x import read_x
from S0_bw_opt import bw_opt
from S1_MVCM import mvcm
# from S2_GSIS import gsis
# from S3_BSTP import wild_bstp
# from S3_TEST import local_test

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
    print("There are " + str(snp.shape[1]) + " snps with MAF>0.05.")

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 1. Fit the multivariate varying coefficient model (MVCM) """)
    start_1 = time.time()
    # find the optimal bandwidth
    h_opt, hat_mat = bw_opt(coord_data, x_design, y_design)
    qr_smy_mat, esig_eta, smy_design, resy_design, efit_eta, sm_weight = \
        mvcm(coord_data, x_design, y_design, h_opt, hat_mat)
    end_1 = time.time()
    print("Elapsed time in Step 1 is ", end_1 - start_1)


if __name__ == '__main__':
    input_dir0 = sys.argv[1]
    output_dir0 = sys.argv[2]
    run_script(input_dir0, output_dir0)
