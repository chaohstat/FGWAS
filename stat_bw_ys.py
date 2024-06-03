import numpy as np


def bw_ys(coord_mat):
  """
      :param
          coord_mat (matrix): common coordinate matrix (n_v*d)
      :return
          h_opt (vector): optimal bandwidth vector (len=d)
  """
  
  # Set up
  n_v, d = coord_mat.shape
  h_opt = np.zeros(d)
  J = np.ones((n_v,1))
  
  for dii in range(d):
    dm = np.zeros(n_v)
    for vii in range(n_v):
      d = np.abs(coord_mat[vii,dii] - coord_mat[:,dii])
      d_0 = d[d!=0]
      dm[vii] = min(d_0)
    
    h_opt[dii] = np.median(dm)
  
  return h_opt
