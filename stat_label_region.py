import numpy as np
from scipy.ndimage import label



def label_region(img_size, img_idx, pv_log10, alpha_log10):
  """
      Label connected regions and return the corresponding areas.

      :param
          img_size (vector): image dimension (1*d, d=2, 3)
          img_idx (vector): image index in non-background region (1*l)
          pv_log10 (vector): -log10(local p-values) on all pixels from non-background region (1*l)
          alpha_log10 (scalar): the threshold for binaries the -log10(local p-values)
  """
  
  # wrap image to original shape
  img_tp = np.zeros(shape=img_size)
  for lii in range(len(img_idx)):
    img_sub = np.unravel_index(img_idx[lii], img_size)
    img_tp[img_sub] = pv_log10[lii]
  
  # create binary image based on pval
  img_tp[img_tp <= alpha_log10] = 0
  img_tp[img_tp > alpha_log10] = 1
  img_tp = img_tp.astype(int)
  
  max_cluster_area = 0
  if np.sum(img_tp) > 0:
    labeled_image, num_features = label(img_tp) # label each connected region as a cluster
    feature_areas = np.bincount(labeled_image.ravel())[1:] # get size of each region: count # voxels in each region
    max_cluster_area = np.max(feature_areas) # get the max area
  
  return max_cluster_area
