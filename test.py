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
from stat_bw_rt import bw_rt
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
    n, l, m = img_data.shape
    img_data = np.log(img_data)
    print("The matrix dimension of image data is " + str(img_data.shape))
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
    # design_data = design_data[:, np.arange(5)]
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
    print("+++++++Construct the imaging response, design, coordinate matrix: normalization+++++++")
    x_design, y_design, coord_data = read_x(img_data, coord_data, design_data, var_type)
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
    h_opt, hat_mat = bw_rt(coord_data, x_design, y_design)
    print("the optimal bandwidth by Scott's Rule is ", h_opt)
    qr_smy_mat, esig_eta, smy_design, resy_design, efit_eta = mvcm(coord_data, y_design, h_opt, hat_mat)
    end_1 = time.time()
    print("Elapsed time in Step 1 is ", end_1 - start_1)
    print(resy_design)
    print(efit_eta)
    for mii in range(m):
        res_mii = resy_design[:, :, mii]-efit_eta[:, :, mii]
        print("The bound of the residual is [" + str(np.min(res_mii)) + ", " + str(np.max(res_mii)))
        res_img = np.reshape(np.mean(res_mii, axis=0), (int(img_size[0]), int(img_size[1])))
        res_img_file_name = output_dir + "residual_%d.txt" % mii
        np.savetxt(res_img_file_name, res_img)


if __name__ == '__main__':
    input_dir0 = sys.argv[1]
    output_dir0 = sys.argv[2]
    run_script(input_dir0, output_dir0)
