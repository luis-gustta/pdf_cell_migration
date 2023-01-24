import sys
import math
import torch
import numpy as np
import scipy as sp
import numba as nb
import skimage as sk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch import nn
from numba import jit
from time import sleep
from skimage.transform import rotate as rot

from scipy.ndimage import rotate

np.set_printoptions(threshold=np.inf)

def create_grid(n_x, n_y, n_pol, n_theta):
    return np.zeros([n_x, n_y, n_pol, n_theta])

def ressample(arr, N):
    A = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        A.extend([*np.hsplit(v, arr.shape[0] // N)])
    return np.array(A)

class cell(object):
    def __init__(self, ini_grid):
        self.lattice = ini_grid
        self.n_theta = len(self.lattice[0,0,0,:])
        self.n_pol = len(self.lattice[0,0,:,0])
        self.n_x = len(self.lattice[:,0,0,0])
        self.n_y = len(self.lattice[0,:,0,0])
        self.real_lattice = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])

        self.kappa = 0.0
        self.gamma = 0.0
        self.D_theta = 0.0
        self.D_para_dir = 0.0
        self.D_perp_dir = 0.0

        self.max_x = (1./(math.sqrt(2)))*self.n_x
        self.max_y = (1./(math.sqrt(2)))*self.n_y
        # self.max_x = 0.7*self.n_x
        # self.max_y = 0.7*self.n_y

    def diffusion_para_dir(self):
        D = self.D_para_dir
        aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])

        kernel= np.array([[0.,0.,0.],
                        [0.5,-1.,0.5],
                        [0.,0.,0.]])

        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                aux[:,:,i,j] = self.lattice[:,:,i,j] + D*sp.signal.convolve2d(self.lattice[:,:,i,j], kernel, mode="same", boundary="wrap")
                # aux[:,:,i,j] = self.lattice[:,:,i,j] + D*sp.signal.convolve2d(self.lattice[:,:,i,j], kernel, mode="same", boundary="fill")
        self.lattice = np.copy(aux)

    def diffusion_perp_dir(self):
        D = self.D_perp_dir
        aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])

        kernel= np.array([[0.,0.5,0.],
                    [0.,-1.,0.],
                    [0.,0.5,0.]])

        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                aux[:,:,i,j] = self.lattice[:,:,i,j] + D*sp.signal.convolve2d(self.lattice[:,:,i,j], kernel, mode="same", boundary="wrap")
                # aux[:,:,i,j] = self.lattice[:,:,i,j] + D*sp.signal.convolve2d(self.lattice[:,:,i,j], kernel, mode="same", boundary="fill")
        self.lattice = np.copy(aux)

    def drift_para_dir(self):
        # ROLL METHOD (Uses numpy function roll to convect the pulse in a forwards direction)
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                 self.lattice[:,:,i,j] = np.roll(self.lattice[:,:,i,j], j+1, axis=1) #Introduce velocity dynamics (dissipation, diffusion and from 0 to 1)
        # STANDARD METHOD (Same results for both methods)
        # aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])
        # for i in range(0, self.n_theta):
        #     for j in range(0, self.n_pol):
        #         for l in range(0, self.n_x):
        #             for m in range(0, self.n_y):
        #                 v_drift = int((j+1)*self.kappa) #fazer com que o j=0 passe para j=1
        #                 aux[l,m,i,j] = self.lattice[l,m-v_drift,i,j]
        # self.lattice = np.copy(aux)

    def diffusion_theta(self):
        D = self.D_theta
        aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])
        for j in range(0, self.n_pol):
            for i in range(0, self.n_theta):
                i_min = i-1
                if (i_min<0): i_min = self.n_theta-1
                i_plus = i+1
                if (i_plus>self.n_theta-1): i_plus = 0

                diff = 0.5*(self.real_lattice[:,:,i_min,j] + self.real_lattice[:,:,i_plus,j] - 2.*self.real_lattice[:,:,i,j])
                aux[:,:,i,j] = self.real_lattice[:,:,i,j] + D*diff

        self.real_lattice = np.copy(aux)

    def polarization_dynamics_dissipation(self):
        aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for l in range(0, self.n_x):
                    for m in range(0, self.n_y):
                        j_diss = int((j+self.gamma)) #force the pdf from j=0 to j=1
                        if j_diss < self.n_pol:
                            aux[l,m,i,j] = self.lattice[l,m,i,j_diss]
                        else:
                            aux[l,m,i,j] = self.lattice[l,m,i,j]
        self.lattice = np.copy(aux)

    def to_real_lattice(self):
        aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                d_theta = i*(360./self.n_theta) # NEVER USE GRID WRAP, USE MODE 'CONSTANT' OR ELSE IT WILL MESS UP WITH THE BOUNDARY CONDITIONS
                aux[:,:,i,j] = aux[:,:,i,j] + rot(self.lattice[:,:,i,j],d_theta, preserve_range=True)
                # aux[:,:,i,j] = aux[:,:,i,j] + rotate(self.lattice[:,:,i,j], angle=d_theta, reshape=False, mode='constant', order=5)
        self.real_lattice = np.copy(aux)

    def from_real_lattice(self):
        aux = np.zeros([self.n_x,self.n_y,self.n_theta,self.n_pol])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                d_theta = -i*(360./self.n_theta) # NEVER USE GRID WRAP, USE MODE 'CONSTANT' OR ELSE IT WILL MESS UP WITH THE BOUNDARY CONDITIONS
                aux[:,:,i,j] = aux[:,:,i,j] + rot(self.lattice[:,:,i,j],d_theta, preserve_range=True)
                # aux[:,:,i,j] = aux[:,:,i,j] + rotate(self.real_lattice[:,:,i,j], angle=d_theta, reshape=False, mode='constant', order=5)
        self.lattice = np.copy(aux)

    #Try to apply the numba function to speed up the process
    def real_periodic_boundary(self):
        self.to_real_lattice()
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for l in range(0, self.n_x):
                    for m in range(0, self.n_y):
                        if l > self.n_x/2. + self.max_x/2.:
                            aux = self.real_lattice[l,m,i,j]
                            self.real_lattice[l,m,i,j] = 0.
                            self.real_lattice[l-int(self.max_x),m,i,j] = self.real_lattice[l-int(self.max_x),m,i,j] + aux
                        if m > self.n_x/2. + self.max_y/2.:
                            aux = self.real_lattice[l,m,i,j]
                            self.real_lattice[l,m,i,j] = 0.
                            self.real_lattice[l,m-int(self.max_y),i,j] = self.real_lattice[l,m-int(self.max_y),i,j] + aux
                        if l < self.n_x/2. - self.max_x/2.:
                            aux = self.real_lattice[l,m,i,j]
                            self.real_lattice[l,m,i,j] = 0.
                            self.real_lattice[l+int(self.max_x),m,i,j] = self.real_lattice[l+int(self.max_x),m,i,j] + aux
                        if m < self.n_x/2. - self.max_y/2.:
                            aux = self.real_lattice[l,m,i,j]
                            self.real_lattice[l,m,i,j] = 0.
                            self.real_lattice[l,m+int(self.max_y),i,j] = self.real_lattice[l,m+int(self.max_y),i,j] + aux
        self.from_real_lattice()

    def print_state(self):
        aux = np.zeros([self.n_x,self.n_y])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                # aux[:,:] = aux[:,:] + self.lattice[:,:,i,j]
                aux[:,:] = aux[:,:] + self.real_lattice[:,:,i,j]

        fig = plt.imshow(aux, cmap='hot')
        block=False
        plt.pause(0.001)

    def total(self):
        prob = np.sum(np.concatenate(self.real_lattice))
        # prob = np.sum(np.concatenate(self.lattice))
        return prob
