import random
import numpy as np

def make_invmat(matrix):
  invMat = np.identity(4)
  invMat[:3, :3] = matrix[:3, :3].T
  invMat[:3, 3] = -np.dot(matrix[:3, :3].T, matrix[:3,3])
  return invMat

def set_global_seeds(seed):
  try:
    import tf.compat.v1 as tf
  except ImportError:
    pass
  else:
    tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  return
