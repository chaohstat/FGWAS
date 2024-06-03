A.Pipeline: run the following scripts in order in python3:
0.preparation
  0.1 modify PATH in the following 3 scripts to your path that contains all python scripts in this package
  0.2 create the file directory structure (as in section B)
  0.3 prepare input data (as under section B:data)
1_gsis.py [GSIS]
  output:
  - GSIS_all.txt
    [each SNP per row for all SNPs, with columns: 1.SNP chromosome, 2.SNP position (bp), 3.global test statistic, and 4.p-value]
  - GSIS_top.txt
    [each SNP per row for top SNPs (sorted from most significant to less significant), 
    with columns: 1. SNP name, 2.SNP chromosome, 3.SNP position (bp), 4.-log10(global p-value)]
2_bstp.py [Bootstrap: can submit multiple jobs, each running this script for b bootstrap samples]
  output: 
    max_lstat_bstp_i [bootstrap local test statistics]
    max_gstat_bstp_i [bootstrap global test statistics]
    max_area_bstp_i  [bootstrap cluster-size statistics]
  combine output:
    cat max_lstat_bstp_* > max_lstat_bstp.txt
    cat max_gstat_bstp_* > max_gstat_bstp.txt
    cat max_area_bstp_*  > max_area_bstp.txt
3_test.py [Local Test & Cluster-Size Analysis]
  output:
    local_pv_raw.txt [raw p-values at each voxel as local tests]
    local_stat.txt   [local statistics at each voxel]
    local_pv_adj.txt [p-values at each voxel adjusted for # SNPs and # voxels]
    cluster_pv.txt   [p-values for cluster-size analysis for each SNP]
4.[optional] global test: calculate p-values adjusted for the number of SNPs (Ng)
  based on column 3 of GSIS_all.txt and its distribution max_gstat_bstp.txt


B.Directory structure under PATH:
- data [input data; all w.o. row names nor column names]
  1.img_data_left.mat, img_data_right.mat 
    [m x n x l matrix of image response; m: # responses, n: # sumjects, l: # voxels]
  2.coord_data_left.txt, coord_data_right.txt 
    [l x d matrix of image coordinates; d: dimension of image (e.g. 2 for 2D image, 3 for 3D image)]
  3.design_data.txt
    [n x p matrix of design matrix (w.o. 1st column of 1's; the 1st column of 1's will be added in code)]
  4.var_type.txt
    [vector of length p; the ith element is 1 if the ith column (covariate) in design_data.txt is numeric, and 0 otherwise]
  5.snp_data.txt
    [n x Ng matrix of SNP dosage levels]
  6.snp_info.map
    [Ng x 3 matrix of SNP info, w. each row per SNP;
    with columns: 1. SNP chromosome, 2.SNP name, 3.SNP position (bp)]
  7.img_size.txt
    [length-d vector, indicating the size of the image (e.g. '40 50 30' for a 40x50x30 3D image)]
  8.img_idx.txt
    [length-l vector, containing the order of the voxels (e.g. 1:l)]
- res [contains final results from 1_gsis.py and 3_test.py]
  - left [assume there are left side and right side such as for hippocampus. Otherwise, modify the file structure accordingly.]
  - right [similar to 'left']
    ...
- vars/ [contains intermediate outputs]
  - left/
  - right/
- bstp [contains boostrap statistics (output from 2_bstp.py)]
  - left/
  - right/
- work [optional directory to hold your job-running log files]