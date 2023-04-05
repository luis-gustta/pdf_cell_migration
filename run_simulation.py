from cell_mig.pdf import *
from time import time as tm

file = open('prob_sum.dat', 'w')

total_time = 1000

n = 25
ini_grid = create_grid(n, n, 8, 8)
ini_grid[int(n / 2.), int(n / 2.), 5, 0] = 1.

cell1 = Cell(ini_grid)
cell1.diff_theta = 0.001
cell1.diff_para_dir = 0.001
cell1.diff_perp_dir = 0.001
cell1.kappa = 1.
cell1.gamma = 1.

for time in range(0, total_time):
    t0 = tm()
    cell1.drift_para_dir()
    # cell1.real_periodic_boundary()
    cell1.diffusion_para_dir()
    # cell1.real_periodic_boundary()
    cell1.diffusion_perp_dir()
    # cell1.polarization_dynamics_dissipation()
    cell1.real_periodic_boundary()

    cell1.to_real_lattice()  # -----> look https://gautamnagrawal.medium.com/rotating-image-by-any-angle-shear
    # -transformation-using-only-numpy-d28d16eb5076
    cell1.diffusion_theta()
    cell1.from_real_lattice()
    cell1.print_state()  # mpl.print...

    temp = f"{time} {cell1.total()}"
    print(temp, file=file)
    print(temp)
    print(tm()-t0)

    if time % 10 == 0:
        # plt.clf()
        # plt.cla()
        plt.close('all')
        #print(cell1)

plt.clf()
plt.cla()
plt.close('all')
file.close()
