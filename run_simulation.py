import numpy as np
from time import sleep
from cell_mig import pdf

file = open('prob_sum.dat', 'w')

total_time = 1000

N = 50
ini_grid = create_grid(N,N,12,12)
ini_grid[int(N/2.),int(N/2.),0,0] = 1.

cell1 = cell(ini_grid)
cell1.D_theta = 0.9
cell1.D_para_dir = 0.9
cell1.D_perp_dir = 0.9
cell1.kappa = 1.
cell1.gamma = 1.

for time in range(0,total_time):

    cell1.drift_para_dir()
    # cell1.real_periodic_boundary()
    cell1.diffusion_para_dir()
    # cell1.real_periodic_boundary()
    cell1.diffusion_perp_dir()
    # cell1.polarization_dynamics_dissipation()
    cell1.real_periodic_boundary()

    cell1.to_real_lattice() #-----> look https://gautamnagrawal.medium.com/rotating-image-by-any-angle-shear-transformation-using-only-numpy-d28d16eb5076
    cell1.diffusion_theta()
    cell1.from_real_lattice()
    cell1.print_state()

    temp = str(time)+" "+str(cell1.total())
    print(temp,file=file)
    print(temp)

    if time%10==0 :
        # plt.clf()
        # plt.cla()
        plt.close('all')

plt.clf()
plt.cla()
plt.close('all')
file.close()
