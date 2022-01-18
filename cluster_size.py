# cd /nas/longleaf/home/yshan/adni/fgwas/fgwas_v1_left
# cd /nas/longleaf/home/yshan/adni/fgwas/fgwas_v1_right
import numpy as np
from stat_label_region import label_region

input_dir  = '/nas/longleaf/home/yshan/adni/fgwas/data/'
img_size = np.loadtxt(input_dir + "img_size.txt")
img_size = img_size.astype(int)
img_idx = np.loadtxt(input_dir + "img_idx.txt")
img_idx = (img_idx-1).astype(int)

max_area_bstp = np.loadtxt('bstp/max_area_bstp.txt') # left:3~7 right:3~6
p_raw = np.loadtxt('res/local_pv_raw.txt')
p_log10 = -np.log10(1-p_raw)

alpha = 0.005
c_alpha = -np.log10(alpha)

g_num = p_raw.shape[0]
cluster_pv = np.zeros(g_num)
max_area = np.zeros(g_num)

for gii in range(g_num):
    max_area[gii] = label_region(img_size, img_idx, p_log10[gii, :], c_alpha)
    cluster_pv[gii] = np.sum(max_area_bstp >= max_area[gii])/len(max_area_bstp)

np.savetxt('res/cluster_pv.txt', cluster_pv)
