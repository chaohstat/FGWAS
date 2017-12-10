"""
Label connected regions and return the corresponding areas.

Author: Chao Huang (chaohuang.stat@gmail.com)
Last update: 2017-10-14
"""

import numpy as np
from skimage.measure import regionprops

"""
installed all the libraries above
"""


def label_region(img_size, img_idx, l_stat, c_alpha):
    """
        Label connected regions and return the corresponding areas.

        :param
            img_size (vector): image dimension (1*d, d=2, 3)
            img_idx (vector): image index in non-background region (1*l)
            l_stat (vector): local statistics on all pixels from non-background region (1*l)
            c_alpha (scalar): the threshold for binaries the local statistics
    """

    def label_region_nd(label_img):
        """
        Label connected regions and return the corresponding maximum area.

        Args:
            label_img (matrix): binary matrix after thresholding
        """

        group = regionprops(label_img)
        cluster_area = np.zeros(len(group))
        for k in range(len(group)):
            cluster_area[k] = group[k].area
        max_cluster_area = np.max(cluster_area)

        return max_cluster_area

    img_tp = np.zeros(shape=img_size)
    for lii in range(len(img_idx)):
        img_sub = np.unravel_index(img_idx[lii], img_size)
        img_tp[img_sub] = l_stat[lii]
    img_tp[img_tp <= c_alpha] = 0
    img_tp[img_tp > c_alpha] = 1
    max_area = label_region_nd(img_tp)

    return max_area
