import sys
import math
import cupy as cp
#import torch
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#from torch import nn
from time import sleep
from IPython import display
from matplotlib.animation import FuncAnimation

def create_grid(n_y, n_x, n_pol, n_theta):
    return np.zeros([n_y, n_x, n_pol, n_theta])

def ressample(arr, N):
    A = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        A.extend([*np.hsplit(v, arr.shape[0] // N)])
    return np.array(A)

class tissue(object):
    def __init__(self, ini_grid):
        self.real_lattice = cp.asarray(ini_grid)
        self.n_theta = len(self.real_lattice[0,0,0,:])
        self.n_pol = len(self.real_lattice[0,0,:,0])
        self.n_x = len(self.real_lattice[0,:,0,0])
        self.n_y = len(self.real_lattice[:,0,0,0])
        self.lattice = np.zeros([self.n_y,self.n_x,self.n_theta,self.n_pol])

        self.kappa = 0.0
        self.gamma = 0.0
        self.D_theta = 0.0
        self.D_para_dir = 0.0
        self.D_perp_dir = 0.0

        self.max_x = (1./(np.sqrt(2)))*self.n_x
        self.max_y = (1./(np.sqrt(2)))*self.n_y
        
    def diffusion_para_dir(self):
        D = self.D_para_dir
        aux = np.zeros((self.n_y,self.n_x,self.n_theta,self.n_pol))
        kernel= np.array([[0.,0.,0.],
                        [0.5,-1.,0.5],
                        [0.,0.,0.]])

        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                aux[:,:,i,j] = self.lattice[:,:,i,j] + D*sp.signal.convolve2d(self.lattice[:,:,i,j], kernel, mode="same", boundary="wrap")
        self.lattice = np.copy(aux)

    def diffusion_perp_dir(self):
        D = self.D_perp_dir
        aux = np.zeros((self.n_y,self.n_x,self.n_theta,self.n_pol))

        kernel= np.array([[0.,0.5,0.],
                    [0.,-1.,0.],
                    [0.,0.5,0.]])

        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                aux[:,:,i,j] = self.lattice[:,:,i,j] + D*sp.signal.convolve2d(self.lattice[:,:,i,j], kernel, mode="same", boundary="wrap")
        self.lattice = np.copy(aux)

    def drift_para_dir(self):
        # ROLL METHOD (Uses numpy function roll to convect the pulse in a forwards direction)
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                self.lattice[:,:,i,j] = np.roll(self.lattice[:,:,i,j], j, axis=1)
                 #self.lattice[:,:,i,j] = np.roll(self.lattice[:,:,i,j], j+1, axis=1)
                 #Introduce velocity dynamics (dissipation, diffusion and from 0 to 1)
                 # The "j+1" term is a quick fix, the correct method should be gathering all particles with j=0 and redistributing then in all orientations
                 #equally distributed and with j=1 in all of these orientations
                 
        # STANDARD METHOD (Same results for both methods)
        # aux = np.zeros((self.n_y,self.n_x,self.n_theta,self.n_pol))
        # for i in range(0, self.n_theta):
        #     for j in range(0, self.n_pol):
        #         for y in range(0, self.n_y):
        #             for x in range(0, self.n_x):
        #                 v_drift = int((j)*self.kappa)
        #                 aux[y,x,i,j] = self.lattice[y-v_drift,x,i,j]
        # self.lattice = np.copy(aux)

    def diffusion_theta(self):
        D = self.D_theta
        aux = cp.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        lattice = cp.asarray(self.real_lattice)  # Convert real_lattice to CuPy array
        
        for j in range(0, self.n_pol):
            for i in range(0, self.n_theta):
                i_min = i - 1
                if (i_min < 0):
                    i_min = self.n_theta - 1
                i_plus = i + 1
                if (i_plus > self.n_theta - 1):
                    i_plus = 0
    
                diff = 0.5 * (lattice[:, :, i_min, j] + lattice[:, :, i_plus, j] - 2. * lattice[:, :, i, j])
                aux[:, :, i, j] = lattice[:, :, i, j] + D * diff
    
        # Convert aux back to a NumPy array
        self.real_lattice = cp.copy(aux)

    def polarization_dynamics_dissipation(self):
        aux = np.zeros((self.n_y,self.n_x,self.n_theta,self.n_pol))
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for x in range(0, self.n_x):
                    for y in range(0, self.n_y):
                        j_diss = int((j+self.gamma))
                        if j_diss < self.n_pol:
                            aux[y,x,i,j] = self.lattice[y,x,i,j_diss]
                        else:
                            aux[y,x,i,j] = self.lattice[y,x,i,j]
        self.lattice = np.copy(aux)
    
    def to_real_lattice(self):
        aux = cp.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        lattice = cp.asarray(self.lattice)
        
        for i in range(0, self.n_theta):
            mask_i = cp.any(lattice[:, :, i, :], axis=(0, 1)).any() #.astype(bool)
            if cp.all(mask_i != 0.0):
                for j in range(0, self.n_pol):
                    mask_ij = cp.any(lattice[:, :, i, j], axis=(0,)).any()  #.astype(bool)
                    if mask_ij:
                        x_vals = cp.arange(self.n_x)
                        y_vals = cp.arange(self.n_y)
                        x_grid, y_grid = cp.meshgrid(x_vals, y_vals)
                        
                        d_theta = i * (2.0 * cp.pi / self.n_theta) - cp.pi
                        
                        y_real = y_grid - self.n_y / 2.0
                        x_real = x_grid - self.n_x / 2.0
                        
                        x_new = x_real * cp.cos(d_theta) - y_real * cp.sin(d_theta) + self.n_x / 2.0
                        y_new = x_real * cp.sin(d_theta) + y_real * cp.cos(d_theta) + self.n_y / 2.0
                        
                        x_new = self.cu_wrap(x_new, self.n_x - 1)
                        y_new = self.cu_wrap(y_new, self.n_y - 1)
                        
                        y_new_int = cp.asnumpy(y_new).astype(int)
                        x_new_int = cp.asnumpy(x_new).astype(int)
                    
                        aux[y_new_int, x_new_int, i, j] += lattice[y_grid, x_grid, i, j]
                        
                        # aux[y_new, x_new, i, j] += lattice[y_grid, x_grid, i, j]
            
        self.real_lattice = cp.copy(aux)

    def from_real_lattice(self):
        aux = cp.zeros((self.n_y, self.n_x, self.n_theta, self.n_pol))
        lattice = cp.asarray(self.real_lattice)  # Convert real_lattice to CuPy array
        
        for i in range(0, self.n_theta):
            mask_i = cp.any(lattice[:, :, i, :], axis=(0, 1)).any()
            if mask_i:
                for j in range(0, self.n_pol):
                    mask_ij = cp.any(lattice[:, :, i, j], axis=(0,)).any()
                    if mask_ij:
                        x_vals = cp.arange(self.n_x)
                        y_vals = cp.arange(self.n_y)
                        x_grid, y_grid = cp.meshgrid(x_vals, y_vals)
                        
                        d_theta = i * (2.0 * cp.pi / self.n_theta) - cp.pi
                        
                        y_real = y_grid - self.n_y / 2.0
                        x_real = x_grid - self.n_x / 2.0
                        
                        x_new = x_real * cp.cos(d_theta) + y_real * cp.sin(d_theta) + self.n_x / 2.0
                        y_new = -x_real * cp.sin(d_theta) + y_real * cp.cos(d_theta) + self.n_y / 2.0
                        
                        # WHY DID I HAVE TO ADD 0.15 TO THESE VARIABLES????? IF I DO NOT ADD THESE, SOME PULSES STOPPING
                        # I DO NOT KNOW THE REASON OF WHY THE PULSES STOP
                        x_new = self.cu_wrap(x_new + 0.15, self.n_x - 1)
                        y_new = self.cu_wrap(y_new + 0.15, self.n_y - 1)
                        
                        y_new_int = cp.asnumpy(y_new).astype(int)
                        x_new_int = cp.asnumpy(x_new).astype(int)
                        
                        aux[y_new_int, x_new_int, i, j] += lattice[y_grid, x_grid, i, j]
    
        # Copy aux back to the lattice
        self.lattice = cp.copy(aux)

    #Try to apply the numba function to speed up the process
    def real_periodic_boundary(self):
        self.to_real_lattice()
    
        lattice = cp.asarray(self.real_lattice)  # Convert real_lattice to CuPy array
        
        for i in range(0, self.n_theta):
            mask_i = cp.any(lattice[:, :, i, :], axis=(0, 1)).any()
            if mask_i:
                for j in range(0, self.n_pol):
                    mask_ij = cp.any(lattice[:, :, i, j], axis=(0,)).any()
                    if mask_ij:
                        for x in range(0, self.n_x):
                            mask_x = lattice[:, x, i, j].any()
                            if mask_x:
                                for y in range(0, self.n_y):
                                    mask_y = lattice[y, :, i, j].any()
                                    if mask_y:
                                        if x > self.n_x/2. + self.max_x/2.:
                                            aux = lattice[y, x, i, j]
                                            lattice[y, x, i, j] = 0.
                                            lattice[y, x-int(self.max_x), i, j] += aux
                                        if y > self.n_x/2. + self.max_y/2.:
                                            aux = lattice[y, x, i, j]
                                            lattice[y, x, i, j] = 0.
                                            lattice[y-int(self.max_y), x, i, j] += aux
                                        if x < self.n_x/2. - self.max_x/2.:
                                            aux = lattice[y, x, i, j]
                                            lattice[y, x, i, j] = 0.
                                            lattice[y, x+int(self.max_x), i, j] += aux
                                        if y < self.n_x/2. - self.max_y/2.:
                                            aux = lattice[y, x, i, j]
                                            lattice[y, x, i, j] = 0.
                                            lattice[y+int(self.max_y), x, i, j] += aux
        
        # Convert real_lattice_cp back to a NumPy array
        self.real_lattice = cp.copy(lattice)
        self.from_real_lattice()

    def collision_real_lattice(self):
        aux = np.copy(self.real_lattice)
        
        for y in range(0,self.n_y):
            if (self.real_lattice[y,:,:,:].any() == True):
                for x in range(0,self.n_x):
                    if (self.real_lattice[y,x,:,:].any() == True):
                        for j in range(0, self.n_theta):
                            if (self.real_lattice[y,x,j,:].any() == True):
                                for l in range(0, self.n_pol):
                                    trans_factor = 0.
                                    norm_factor = 0.
                                    
                                    mean_pn_x = 0.
                                    mean_pn_y = 0.
                        
                                    for m in range(0, self.n_theta):
                                        if (self.real_lattice[y,x,m,:].any() == True):
                                            for n in range(0, self.n_pol):
                                                if m!=j or n!=l:
                                                    trans_factor += self.real_lattice[y,x,j,l] * self.real_lattice[y,x,m,n]
                                                    norm_factor += self.real_lattice[y,x,m,n]
                                                    
                                                    p_n = (n) # * self.kappa 
                                                    theta_m = m*(2.*np.pi/self.n_theta) 
                                                    mean_pn_x += self.real_lattice[y,x,m,n] * p_n * np.cos(theta_m)
                                                    mean_pn_y += self.real_lattice[y,x,m,n] * p_n * np.sin(theta_m)
                                    
                                    p_l = l # * self.kappa
                                    
                                    theta_l = l*(2.*np.pi/self.n_theta) 
                                    theta_k = np.arctan2(mean_pn_y,mean_pn_x) #+ np.pi
        
                                    px_new = 0.5 * ( p_l * np.cos(theta_l) + mean_pn_x)
                                    py_new = 0.5 * ( p_l * np.sin(theta_l) + mean_pn_y)
                                    
                                    theta_new = np.arctan2(py_new,px_new) #+ np.pi
                                    
                                    p_new = np.sqrt(px_new**2 + py_new**2)
                                    
                                    j_new = self.wrap((int(theta_new*(self.n_theta/(2.*math.pi)))),self.n_theta-1)
                                    l_new = round(p_new)
                                    l_new = self.n_pol-1 if l_new >= self.n_pol else l_new
                                    
                                    # if trans_factor!=0.:
                                        # print(theta_l, theta_k, theta_new, (theta_l+theta_k)/2., trans_factor)
                                    
                                    aux[y,x,j_new,l_new] += trans_factor
                                    aux[y,x,j,l] -= trans_factor
                            
        self.real_lattice = np.copy(aux)
                                
                                
    def percolated_diffusion(self):
        aux = np.zeros((self.n_y,self.n_x,self.n_theta,self.n_pol))
        D = 0.01
        
        for y in range(1,self.n_y-1):
            # if (self.real_lattice[y,:,:,:].any() == True):
                for x in range(1,self.n_x-1):
                    # if (self.real_lattice[y,x,:,:].any() == True):
                        for i in range(0, self.n_theta):
                            # if (self.real_lattice[y,x,i,:].any() == True):
                                for j in range(0, self.n_pol):
                                    P_x_out = D*(self.real_lattice[y,x,i,j] * ( (1 - self.real_lattice[y,x+1,i,j]) + (1 - self.real_lattice[y,x-1,i,j]) ))
                                    P_x_in = D*((1 - self.real_lattice[y,x,i,j]) * ( self.real_lattice[y,x+1,i,j] + self.real_lattice[y,x-1,i,j] ))
                                    
                                    P_y_out = D*(self.real_lattice[y,x,i,j] * ( (1 - self.real_lattice[y+1,x,i,j]) + (1 - self.real_lattice[y-1,x,i,j]) ))
                                    P_y_in = D*((1 - self.real_lattice[y,x,i,j]) * ( self.real_lattice[y+1,x,i,j] + self.real_lattice[y-1,x,i,j] ))
                                    
                                    
                                    ###### Usual Diffusion
#                                     P_x_out = 2.*D*self.real_lattice[y,x,i,j]
#                                     P_x_in = D*(self.real_lattice[y,x+1,i,j] + self.real_lattice[y,x-1,i,j])
#                                     
#                                     P_y_out = 2.*D*self.real_lattice[y,x,i,j]
#                                     P_y_in = D*(self.real_lattice[y+1,x,i,j] + self.real_lattice[y-1,x,i,j])
                                    
                                    aux[y,x,i,j] = self.real_lattice[y,x,i,j] + P_x_in - P_x_out + P_y_in - P_y_out 
                                    
                                    # kernel= np.array([[0.,0.5*P_y_in,0.],
                                        # [0.5*P_x_in,-(P_x_out + P_y_out),0.5*P_x_in],
                                        # [0.,0.5*P_y_in,0.]])
                                    # aux[y,x,i,j] = self.real_lattice[y,x,i,j] + sp.signal.convolve2d(self.real_lattice[:,:,i,j], kernel, mode="same", boundary="wrap")
        self.real_lattice = np.copy(aux)
        # aux = np.zeros((self.n_y,self.n_x,self.n_theta,self.n_pol))
        
        # for y in range(1,self.n_y-1):
        #     # if (self.real_lattice[y,:,:,:].any() == True):
        #         for x in range(1,self.n_x-1):
        #             # if (self.real_lattice[y,x,:,:].any() == True):
        #                 for i in range(0, self.n_theta):
        #                     # if (self.real_lattice[y,x,i,:].any() == True):
        #                         for j in range(0, self.n_pol):
                                    # P_x_out = D*(self.real_lattice[y,x,i,j] * ( (1 - self.real_lattice[y,x+1,i,j]) + (1 - self.real_lattice[y,x-1,i,j]) ))
                                    # P_x_in = D*((1 - self.real_lattice[y,x,i,j]) * ( self.real_lattice[y,x+1,i,j] + self.real_lattice[y,x-1,i,j] ))
                                    
                                    # P_y_out = D*(self.real_lattice[y,x,i,j] * ( (1 - self.real_lattice[y+1,x,i,j]) + (1 - self.real_lattice[y-1,x,i,j]) ))
                                    # P_y_in = D*((1 - self.real_lattice[y,x,i,j]) * ( self.real_lattice[y+1,x,i,j] + self.real_lattice[y-1,x,i,j] ))
                                    
                                    # P_x_out = 2.*D*self.real_lattice[y,x,i,j]
                                    # P_x_in = D*(self.real_lattice[y,x+1,i,j] + self.real_lattice[y,x-1,i,j])
                                    
                                    # P_y_out = 2.*D*self.real_lattice[y,x,i,j]
                                    # P_y_in = D*(self.real_lattice[y+1,x,i,j] + self.real_lattice[y-1,x,i,j])
                                    
                                    # aux[y,x,i,j] = self.real_lattice[y,x,i,j] + P_y_in - P_y_out 
        # self.real_lattice = np.copy(aux)

        
        #The interactions must be on the real lattice right?
        #But then how do I evolve the convection in the self lattice?
        #Do I create a Metropolis like algorithm? Where there is a probability of acceptance after the 
        #The convection ocurre in the self lattice but it had low probability of doing so in the real lattice due to interactions?
                

    def print_state(self):
        # aux = np.zeros([self.n_y,self.n_x])
        aux = cp.zeros([self.n_y,self.n_x])
        
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                # aux[:,:] = aux[:,:] + self.lattice[:,:,i,j]
                aux[:,:] += self.real_lattice[:,:,i,j]
    
        heatmap = ax.pcolormesh(aux.get(), cmap='hot')
        return heatmap


        # aux = np.zeros([self.n_x,self.n_y])
        # for i in range(0, self.n_theta):
        #     for j in range(0, self.n_pol):
        #         # aux[:,:] = aux[:,:] + self.lattice[:,:,i,j]
        #         aux[:,:] = aux[:,:] + self.real_lattice[:,:,i,j]
        # 
        # img = plt.imshow(aux, cmap='hot')
        # block=False
        # plt.pause(0.001)

        
    def wrap(self,val,max_val):
        return round(val - max_val * math.floor(val/max_val))
    
    def cu_wrap(self, val, max_val):
        if isinstance(val, cp.ndarray) and val.size > 1:
            # If val is a Cupy array with size greater than 1, apply the operation element-wise
            return cp.round(val - max_val * cp.floor(val / max_val))
        else:
            # If val is a scalar or an array of size 1, convert to scalar and apply the operation
            val = val.item() if isinstance(val, cp.ndarray) else val
            max_val = max_val.item()
            return round(val - max_val * math.floor(val / max_val))


    def total(self):
        prob = np.sum(np.concatenate(self.real_lattice))
        # prob = np.sum(np.concatenate(self.lattice))
        return prob
    
    def test(self,time):
        for x in range(0,self.n_x):
            for y in range(0,self.n_y):
                for j in range(0, self.n_theta):
                    for l in range(0, self.n_pol):
                        if self.lattice[y,x,j,l]!=0:
                            d_theta = j*(2.*np.pi/self.n_theta) - np.pi
                                    
                            y_real = y - self.n_y/2.
                            x_real = x - self.n_x/2.
                            
                            x_new = x_real*np.cos(d_theta) - y_real*np.sin(d_theta) + self.n_x/2.
                            y_new = x_real*np.sin(d_theta) + y_real*np.cos(d_theta) + self.n_y/2.
                            
                            x_new = self.wrap(x_new,self.n_x-1)
                            y_new = self.wrap(y_new,self.n_y-1)
                            print(self.lattice[y,x,j,l], self.real_lattice[y_new,x_new,j,l], time)

#############################################################################
np.set_printoptions(threshold=np.inf)
file = open('prob_sum.dat', 'w')
video_name = "simu"

total_time = 100
anim_fps = 5

N = 20
ini_grid = create_grid(N,N,12,12)

ini_grid[int(N/2.)+1,int(N/2.),0,1] = 1./2.
ini_grid[int(N/2.),int(N/2.),-1,1] = 1./2.

#[y_axis, x_axis, theta, polarization]
# ini_grid[int(N/2.),int(N/2.),0,0] = 1.

#WHY ARE THE CELLS STOPPING AT RANDOM SPOTS????????????
#YOU MUST FIX THE CHANGE OF POLARIZATION DYNAMICS, THAT PASS CELLS WITH ZERO TO ONE IN ALL ORIENTATIONS

fig, ax = plt.subplots()

tissue = tissue(ini_grid)
tissue.D_theta = 0.1
tissue.D_para_dir = 0.1
tissue.D_perp_dir = 0.1
tissue.kappa = 1.
tissue.gamma = 1.

tissue.from_real_lattice()

def update(frame):
        tissue.drift_para_dir()
        # tissue.real_periodic_boundary()
        # tissue.diffusion_para_dir()
        # tissue.real_periodic_boundary()
        # tissue.diffusion_perp_dir()
        # tissue.polarization_dynamics_dissipation()
        tissue.real_periodic_boundary()
    
        tissue.to_real_lattice()
        tissue.diffusion_theta()
        # tissue.percolated_diffusion() #For testing purposes, this function is producing the usual diffusion
        #tissue.collision_real_lattice()
        tissue.from_real_lattice()
        
        # tissue.test(time)
    
        if (frame%10==0):
            ax.clear()
    
        # if (frame>=total_time):
        #    anim.event_source.stop()
    
        stats = str(frame)+" "+str(tissue.total())
        # print(stats,file=file)
        print(stats)
        
        heatmap = tissue.print_state()

update(100)

anim = FuncAnimation(fig, update, frames=total_time, interval=total_time/anim_fps)
writervideo = animation.FFMpegWriter(fps=anim_fps)
anim.save(video_name+".mp4", writer=writervideo)
