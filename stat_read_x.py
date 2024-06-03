import numpy as np


def read_x(coord_data, var_matrix, var_type):
    """
        Read and construct design matrix.

        :param
            coord_data (matrix): un-normalized coordinate matrix (n_v*d)
            var_matrix (matrix): un-normalized design matrix (n*(p-1))
            var_type (vector): covariate type in var_matrix (0-discrete; 1-continuous)
        :return
            x_design (matrix): normalized design matrix (n*p)
            n_coord_data (matrix): normalized coordinate matrix (n_v*d)
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

    c_coord = np.mean(coord_data, axis=0)  # find the center of all coordinates
    coord_data = coord_data-c_coord
    coord_norm = np.sqrt(np.mean(coord_data**2, axis=0))
    coord_scale = np.max(coord_norm)
    n_coord_data = coord_data/coord_scale

    return x_design, n_coord_data
