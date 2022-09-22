import numpy as np
from time import sleep
from cell_mig import pdf


ini_grid = np.array([
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1.,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0]])

cell1 = pdf.cell(10,10,0,0,ini_grid)

for i in range(0,10):
    #cell1.drift_para_dir()
    #cell1.drift_perp_dir()
    cell1.diffusion_para_dir()
    cell1.diffusion_perp_dir()
    print(cell1.lattice)
    print("\n")
    sleep(1)
