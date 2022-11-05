import numpy as np
from time import sleep
from cell_mig import pdf

total_time = 1000
N_x = 50
N_y = 50
N_theta = 8
N_vel = 8

ini_grid = create_grid(N_x, N_y, N_theta, N_vel)
ini_grid[N_x/2., N_y/2.,:,:] = 1.

cell1 = cell(ini_grid)
cell1.D_theta = 0.05
cell1.D_para_dir = 0.05
cell1.D_perp_dir = 0.05

for time in range(0,total_time):
    cell1.drift_para_dir()
    cell1.drift_perp_dir()
    cell1.diffusion_para_dir()
    cell1.diffusion_perp_dir()

    cell1.to_real_lattice()
    cell1.diffusion_theta()
    cell1.from_real_lattice()
    cell1.print_state()

    if time%15==0 :
        plt.clf()
        plt.cla()
        plt.close('all')
        print(cell1.total())

plt.clf()
plt.cla()
plt.close('all')
