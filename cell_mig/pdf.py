import torch
import math
import numpy as np
import scipy as sp
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch import nn
from numba import jit
from time import sleep

from scipy.ndimage import rotate

def create_grid(n_x, n_y, n_pol, n_theta):
    return np.zeros([n_x, n_y, n_pol, n_theta])

class cell(object):
    def __init__(self, ini_grid):
        self.lattice = ini_grid
        self.n_theta = len(self.lattice[0,0,0,:])
        self.n_pol = len(self.lattice[0,0,:,0])
        self.n_x = len(self.lattice[:,0,0,0])
        self.n_y = len(self.lattice[0,:,0,0])

        self.real_lattice = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])

        self.D_theta = 0.05
        self.D_para_dir = 0.01
        self.D_perp_dir = 0.01

    def diffusion_para_dir(self):
        D = self.D_para_dir
        kernel= np.array([[0.,0.,0.],
                        [D,1.0-2.0*D,D],
                        [0.,0.,0.]])

        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                self.lattice[:,:,i,j] = sp.signal.convolve2d(self.lattice[:,:,i,j], kernel, mode="same", boundary="wrap")

    def diffusion_perp_dir(self):
        D = self.D_perp_dir
        kernel= np.array([[0.,D,0.],
                    [0.,1.0-2.0*D,0.],
                    [0.,D,0.]])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                self.lattice[:,:,i,j] = sp.signal.convolve2d(self.lattice[:,:,i,j], kernel, mode="same", boundary="wrap")

    def drift_para_dir(self):
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                self.lattice[:,:,i,j] = np.roll(self.lattice[:,:,i,j], -1, axis=0)

    def drift_perp_dir(self):
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                self.lattice[:,:,i,j] = np.roll(self.lattice[:,:,i,j], 1, axis=1)

    def diffusion_theta(self):
        aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])
        for j in range(0, self.n_pol):
            for i in range(0, self.n_theta):
                i_min = i-1
                if (i_min<0): i_min = self.n_theta-1
                i_plus = i+1
                if (i_plus>self.n_theta-1): i_plus = 0

                diff = 0.5*(self.real_lattice[:,:,i_min,j] + self.real_lattice[:,:,i_plus,j] - 2.*self.real_lattice[:,:,i,j])
                aux[:,:,i,j] = aux[:,:,i,j] + diff

        self.real_lattice = self.real_lattice + self.D_theta*aux



    def diffusion_kappa(self):
        kernel=np.array([[1./3.,1./3.,1./3.]])
        for i in range(0, self.n_pol):
            self.lattice[:,:,i,:] = sp.signal.convolve2d(self.lattice[:,:,i,:], kernel, mode="same", boundary="symm")
        return None

    def to_real_lattice(self):
        aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                d_theta = i*(360./self.n_theta)
                aux[:,:,i,j] = aux[:,:,i,j] + rotate(self.lattice[:,:,i,j], angle=d_theta, reshape=False, mode='wrap')
        self.real_lattice = aux

    def from_real_lattice(self):
        aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                d_theta = -i*(360./self.n_theta)
                aux[:,:,i,j] = aux[:,:,i,j] + rotate(self.real_lattice[:,:,i,j], angle=d_theta, reshape=False, mode='wrap')
        self.lattice = aux

    def print_state(self):
        aux = np.zeros([self.n_x,self.n_y])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                aux[:,:] = aux[:,:] + self.real_lattice[:,:,i,j]

        fig = plt.imshow(aux, cmap='hot')
        block=False
        plt.pause(0.001)

    def total(self):
        prob = np.sum(np.concatenate(self.real_lattice))
        return prob
