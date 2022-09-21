import numpy as np
from time import sleep
from scipy.signal import convolve2d

class pdf(object):
  def __init__(self, x, y, theta, kappa, initial_grid):
    self.x = x
    self.y = y
    self.theta = theta
    self.kappa = kappa
    self.lattice = np.zeros([self.x+2, self.y+2], dtype='float')
    self.lattice = initial_grid

  def diffusion_para_dir(self):
    kernel=np.array([[0.5  , 0,   0.5]])
    self.lattice = convolve2d(self.lattice,kernel,boundary='wrap')

  def diffusion_perp_dir(self):
    kernel=np.array([[0.5],
                    [0],
                    [0.5]])
    self.lattice = convolve2d(self.lattice,kernel,boundary='wrap')

  def drift_para_dir(self):
    self.lattice = np.roll(self.lattice, -1, axis=0)

  def drift_perp_dir(self):
    self.lattice = np.roll(self.lattice, 1, axis=1)
    
#   def theta_diffusion(self,):
    
    
#   def kappa_diffusion(self,):
   
