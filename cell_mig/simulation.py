import torch
import numpy as np
import scipy as sp
import numba as nb

from torch import nn
from numba import jit
from time import sleep




class pdf(object):
    def __init__(self, x, y, theta, kappa, initial_grid):
        self.x = x
        self.y = y
        self.theta = theta
        self.kappa = kappa
        # self.temp = np.zeros([self.x+2, self.y+2], dtype='float')
        self.lattice = np.zeros([self.x+2, self.y+2], dtype='float')
        self.lattice = initial_grid

    def comp_conv2d(X):
        # (1, 1) indicates that batch size and the number of channels are both 1
        X = X.reshape((1, 1) + X.shape)
        Y = self.conv2d(X)
        # Strip the first two dimensions: examples and channels
        return Y.reshape(Y.shape[2:])

    def diffusion_para_dir(self):
        kernel=np.array([[0.,0.,0.],
                        [1./3.,1./3.,1./3.],
                        [0.,0.,0.]])
        self.lattice = sp.signal.convolve2d(self.lattice,kernel,mode="same",boundary="symm")


    def diffusion_perp_dir(self):
        kernel=np.array([[0.,1./3.,0.],
                    [0.,1./3.,0.],
                    [0.,1./3.,0.]])
        self.lattice = sp.signal.convolve2d(self.lattice,kernel,mode="same",boundary="symm")

    def drift_para_dir(self):
        self.lattice = np.roll(self.lattice, -1, axis=0)

    def drift_perp_dir(self):
        self.lattice = np.roll(self.lattice, 1, axis=1)
        
#   def theta_diffusion(self,):
    
    
#   def kappa_diffusion(self,):
   
