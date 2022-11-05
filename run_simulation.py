import numpy as np
from time import sleep
from cell_mig import pdf

ini_grid = create_grid(50,50,8,8)
ini_grid[25,25,:,:] = 1.

cell1 = cell(ini_grid)

for time in range(0,1000):
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
