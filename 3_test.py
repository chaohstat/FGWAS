import sys
import pickle
import time
import numpy as np
from scipy.stats import f
from S3_TEST import local_test

LorR = 'left'
input_dir  = 'PATH/data/'
interm_dir = 'vars/' + LorR + '/'
bstp_dir   = 'bstp/' + LorR + '/'
output_dir = 'res/' + LorR + '/'

max_lstat_bstp = np.loadtxt(bstp_dir + "max_lstat_bstp.txt")
max_area_bstp = np.loadtxt(bstp_dir + "max_area_bstp.txt")
img_size = np.loadtxt(input_dir + "img_size.txt")
img_size = img_size.astype(int)
img_idx = np.loadtxt(input_dir + "img_idx.txt")
img_idx = (img_idx-1).astype(int)
top_snp = pickle.load(open(interm_dir+"top_snp.dat","rb"))
esig_eta = pickle.load(open(interm_dir+"esig_eta.dat","rb"))
efit_eta = pickle.load(open(interm_dir+'efit_eta.dat','rb')) # m x n x l
proj_mat = pickle.load(open(interm_dir+'proj_mat.dat','rb'))


start_3 = time.time()
p = 1
n = efit_eta.shape[1]
alpha = 0.005
alpha_log10 = -np.log10(alpha)
l_pv_adj, l_stat, cluster_pv = local_test(top_snp, esig_eta, efit_eta, proj_mat, img_size, img_idx, alpha_log10, max_lstat_bstp, max_area_bstp)
l_pv_raw = f.cdf(l_stat, dfn=1, dfd=n-p)
end_3 = time.time()
print("Elapsed time in Step 3 is ", end_3 - start_3)

np.savetxt(output_dir + "local_pv_raw.txt", l_pv_raw)
np.savetxt(output_dir + "local_stat.txt", l_stat)
np.savetxt(output_dir + "local_pv_adj.txt", l_pv_adj)
np.savetxt(output_dir + "cluster_pv.txt", cluster_pv)


