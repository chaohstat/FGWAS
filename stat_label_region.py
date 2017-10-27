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

        Args:
            img_size (vector): image dimension (1*d, d=2, 3)
            img_idx (vector): image index in non-background region (1*l)
            l_stat (vector): local statistics on all pixels from non-background region (1*l)
            c_alpha (scalar): the threshold for binaries the local statistics
    """

    img_sub = np.unravel_index(img_idx, img_size)
    img_tp = np.zeros(shape=img_size)
    img_tp[img_sub] = l_stat
    img_tp[img_tp <= c_alpha] = 0
    img_tp[img_tp > c_alpha] = 1

    def label_region_1d(label_img):
        """
        Label connected regions and return the corresponding maximum area.

        Args:
            label_img (vector): vector or matrix after thresholding
        """

        idx_roi = np.where(label_img == 1)
        run = []
        group = [run]
        expect = None
        for v in idx_roi:
            if (v == expect) or (expect is None):
                run.append(v)
            else:
                run = [v]
                group.append(run)
            expect = v + 1

        cluster_len = np.zeros(len(group))
        for k in range(len(group)):
            cluster_len[k] = len(group[k])
        max_cluster_len = np.max(cluster_len)

        return max_cluster_len

    def label_region_nd(label_img):
        """
        Label connected regions and return the corresponding maximum area.

        Args:
            label_img (matrix): vector or matrix after thresholding
        """

        group = regionprops(label_img)
        cluster_area = np.zeros(len(group))
        for k in range(len(group)):
            cluster_area[k] = group[k].area
        max_cluster_area = np.max(cluster_area)

        return max_cluster_area

    if img_size[0] == 1:
        max_area = label_region_1d(img_tp)
    else:
        max_area = label_region_nd(img_tp)

    return max_area
