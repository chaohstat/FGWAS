"""
Run main script: functional genome wide association analysis (FGWAS) pipeline (step 3-1: Bootstrap resampling)
Usage: python ./test.py ./data/ ./result/ bstp_1.txt 50

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-10-18
"""

import sys
import time
import numpy as np
from scipy.io import loadmat
# from numpy.linalg import eig
# from scipy.stats import f
from S3_BSTP import wild_bstp

"""
installed all the libraries above
"""


def run_script(input_dir, output_dir, output_file, bstp_num):

    """
    Run the commandline script for FGWAS.

    :param
        input_dir (str): full path to the data folder
        output_dir (str): full path to the output folder
        output_file (str): output file name
        bstp (scalar): number of Bootstrap sampling
    """

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Load results from step 1 & 2 """)
    start_0 = time.time()
    data_dim_file_name = output_dir + "/temp/data_dim.mat"
    mat = loadmat(data_dim_file_name)
    data_dim = mat['data_dim']
    data_dim = np.array([int(i) for i in data_dim[0, :]])
    n, l, m, p, g, g_num = data_dim
    y_design_file_name = output_dir + "/temp/y_design.mat"
    mat = loadmat(y_design_file_name)
    y_design = mat['y_design']
    resy_design_file_name = output_dir + "/temp/resy_design.mat"
    mat = loadmat(resy_design_file_name)
    resy_design = mat['resy_design']
    efit_eta_file_name = output_dir + "/temp/efit_eta.mat"
    mat = loadmat(efit_eta_file_name)
    efit_eta = mat['efit_eta']
    esig_eta_file_name = output_dir + "/temp/esig_eta.mat"
    mat = loadmat(esig_eta_file_name)
    esig_eta = mat['esig_eta']
    hat_mat_file_name = output_dir + "/temp/hat_mat.mat"
    mat = loadmat(hat_mat_file_name)
    hat_mat = mat['hat_mat']
    snp_file_name = output_dir + "/temp/snp.mat"
    mat = loadmat(snp_file_name)
    snp = mat['snp']
    # read the image size
    img_size_file_name = input_dir + "img_size.txt"
    img_size = np.loadtxt(img_size_file_name)
    img_size = np.array([int(i) for i in img_size])
    # read the image index of non-background region
    img_idx_file_name = input_dir + "img_idx.txt"
    img_idx = np.loadtxt(img_idx_file_name)
    img_idx = np.array([int(i) for i in img_idx])
    end_0 = time.time()
    print("Elapsed time in Step 3 is ", end_0 - start_0)

    """+++++++++++++++++++++++++++++++++++"""
    print(""" Step 3. Significant locus-voxel and locus-subregion detection """)
    start_3 = time.time()
    alpha = 1e-5
    c_alpha = -10**alpha
    bstp_num = int(bstp_num)
    max_stat_bstp, max_area_bstp = wild_bstp(snp, y_design, resy_design, efit_eta, esig_eta, hat_mat,
                                             img_size, img_idx, c_alpha, g_num, bstp_num)
    print(max_stat_bstp)
    print(max_area_bstp)
    bstp_out = np.hstack((max_stat_bstp, max_area_bstp))
    bstp_out_file_name = output_dir + output_file
    np.savetxt(bstp_out_file_name, bstp_out)
    end_3 = time.time()
    print("Elapsed time in Step 3 is ", end_3 - start_3)


if __name__ == '__main__':
    input_dir0 = sys.argv[1]
    output_dir0 = sys.argv[2]
    output_file0 = sys.argv[3]
    bstp_num0 = sys.argv[4]
    run_script(input_dir0, output_dir0, output_file0, bstp_num0)
