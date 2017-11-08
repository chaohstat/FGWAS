"""
Read and construct design matrix.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-08-14
"""

import numpy as np

"""
installed all the libraries above
"""


def read_x(img_data, coord_data, var_matrix, var_type):
    """
        Read and construct design matrix.

        :param
            img_data (matrix): un-normalized imaging response matrix (n*l*m)
            coord_data (matrix): un-normalized coordinate matrix (l*d)
            var_matrix (matrix): un-normalized design matrix (n*(p-1))
            var_type (vector): covariate type in var_matrix (0-discrete; 1-continuous)
        :return
            x_design (matrix): normalized design matrix (n*p)
            y_design (matrix): normalized imaging response matrix (n*l*m)
            n_coord_data (matrix): normalized coordinate matrix (l*d)
    """

    n, p = var_matrix.shape
    if n < p:
        mat = var_matrix.T
    else:
        mat = var_matrix
    n, p = mat.shape

    mat_new = np.zeros((n, p))

    for kk in range(p):
        if var_type[kk] == 1:
            mat_new[:, kk] = (mat[:, kk] - np.mean(mat[:, kk]))/np.std(mat[:, kk])
        else:
            mat_new[:, kk] = mat[:, kk]
    const = np.ones((n, 1))
    x_design = np.hstack((const, mat_new))

    y_design = img_data
    # y_design = np.zeros(img_data.shape)
    # for mii in range(img_data.shape[2]):
    #     img_data_mii = img_data[:, :, mii]
    #     range_mii = np.max(img_data_mii)-np.min(img_data_mii)
    #     y_design[:, :, mii] = (img_data_mii-np.min(img_data_mii))/range_mii

    # n_coord_data = coord_data
    c_coord = np.mean(coord_data, axis=0)  # find the center of all coordinates
    coord_data = coord_data-c_coord
    coord_norm = np.sqrt(np.sum(coord_data**2, axis=1))
    coord_scale = np.max(coord_norm)
    n_coord_data = coord_data/coord_scale

    return x_design, y_design, n_coord_data
