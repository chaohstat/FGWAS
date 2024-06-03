import sys
import time
import pickle
import numpy as np
from scipy.io import loadmat
from numpy.linalg import inv
from stat_read_x import read_x
from stat_bw_ys import bw_ys
from S1_MVCM import mvcm
from S2_GSIS import gsis

LorR = 'right' # 'left' or 'right'

input_dir  = 'PATH/data/'
output_dir = 'res/' + LorR + '/'
interm_dir = 'vars/' + LorR + '/'


y_design = loadmat(input_dir + "img_data_"+LorR+".mat")['img_data'] # m*n*n_v or n*n_v
if y_design.ndim==2:
  y_design = y_design[np.newaxis,:, :]

coord_data = np.loadtxt(input_dir + 'coord_data_'+LorR+'.txt')

design_data = np.loadtxt(input_dir + 'design_data.txt')
var_type = np.loadtxt(input_dir + 'var_type.txt')

x_design,coord_data = read_x(coord_data,design_data, var_type)


##########
## MVCM ##
##########

print(""" Step 1. Fit the multivariate varying coefficient model (MVCM) """)
start_1 = time.time()
n, p = x_design.shape
proj_mat = np.eye(n) - np.dot(np.dot(x_design, inv(np.dot(x_design.T, x_design)+np.eye(p)*0.000001)), x_design.T)
m = y_design.shape[0]
proj_y_design = 0*y_design
for mii in range(m):
      proj_y_design[mii, :, :] = np.dot(proj_mat, np.squeeze(y_design[mii, :, :]))

h_opt = bw_ys(coord_data) # 8.55681693e-05, 8.55681693e-05, 6.84545354e-05
qr_smy_mat, efit_eta, esig_eta = mvcm(coord_data, proj_y_design, h_opt)
end_1 = time.time()
print("Elapsed time in Step 1 is ", end_1 - start_1) # 7.6 minutes

pickle.dump(proj_mat, open(interm_dir+'proj_mat.dat','wb'))       # n x n
pickle.dump(qr_smy_mat, open(interm_dir+'qr_smy_mat.dat','wb'))   # n x n
pickle.dump(esig_eta, open(interm_dir+'esig_eta.dat','wb'))       # l x m x m
pickle.dump(efit_eta, open(interm_dir+'efit_eta.dat','wb'))       # m x n x l
pickle.dump(x_design, open(interm_dir+'x_design.dat','wb'))
pickle.dump(coord_data, open(interm_dir+'coord_data.dat','wb'))
pickle.dump(h_opt, open(interm_dir+'h_opt.dat','wb'))



##########
## GSIS ##
##########

print(""" Step 2. GSIS """)
start_2 = time.time()

snp = np.loadtxt(input_dir + 'snp_data.txt') # takes a few minutes
fd = open(input_dir + 'snp_info.map', 'r')
snp_chr, snp_name, snp_bp = np.loadtxt(fd, delimiter='\t', unpack=True, \
      dtype={'names': ('snp_chr', 'snp_name', 'snp_bp'), 'formats': ('i4', 'S16', 'i4')})
fd.close()
print('The matrix dimension of original snp data is ' + str(snp.shape))

g_num = 2000   # number of top candidate snps
g_pv_log10, g_stat = gsis(snp, qr_smy_mat, proj_mat)
snp_pv = 10 ** (-g_pv_log10)
top_snp_idx = np.argsort(-g_pv_log10)
top_snp = snp[:, top_snp_idx[0:g_num]]
pickle.dump(top_snp, open(interm_dir+'top_snp.dat','wb'))   # n x n
gsis_all = np.vstack((snp_chr, snp_bp, g_stat, snp_pv)).T   # input for plotting Manhattan plot
np.savetxt(output_dir + 'GSIS_all.txt', gsis_all)

top_snp_chr = snp_chr[top_snp_idx[0:g_num]]
top_snp_name = snp_name[top_snp_idx[0:g_num]]
top_snp_bp = snp_bp[top_snp_idx[0:g_num]]
top_snp_pv_log10 = g_pv_log10[top_snp_idx[0:g_num]]
gsis_top = np.vstack((top_snp_name, top_snp_chr, top_snp_bp, top_snp_pv_log10)).T   # top SNP GSIS results
np.savetxt(output_dir + 'GSIS_top.txt', gsis_top, fmt='%s %s %s %s')
end_2 = time.time()
print('Elapsed time in Step 2 is ', end_2 - start_2) # 5.5 minutes





