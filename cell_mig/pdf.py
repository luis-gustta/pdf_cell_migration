import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# import matplotlib.animation as animation

np.set_printoptions(threshold=np.inf)


def create_grid(n_x, n_y, n_pol, n_theta):
    return np.zeros([n_x, n_y, n_pol, n_theta])


def resample(arr, n):
    aux_arr = []
    for v in np.vsplit(arr, arr.shape[0] // n):
        aux_arr.extend([*np.hsplit(v, arr.shape[0] // n)])
    return np.array(aux_arr)


def wrap(val, max_val):
    return round(val - max_val * math.floor(val / max_val))


# noinspection DuplicatedCode
class Cell(object):
    def __init__(self, ini_grid):
        self.lattice = ini_grid
        self.n_theta = len(self.lattice[0, 0, 0, :])
        self.n_pol = len(self.lattice[0, 0, :, 0])
        self.n_x = len(self.lattice[:, 0, 0, 0])
        self.n_y = len(self.lattice[0, :, 0, 0])
        self.real_lattice = np.zeros([self.n_x, self.n_y, self.n_theta, self.n_pol])

        self.kappa = 0.0
        self.gamma = 0.0
        self.diff_theta = 0.0
        self.diff_para_dir = 0.0
        self.diff_perp_dir = 0.0

        self.max_x = (1. / (math.sqrt(2))) * self.n_x
        self.max_y = (1. / (math.sqrt(2))) * self.n_y

    def diffusion_para_dir(self):
        diff = self.diff_para_dir
        aux = np.zeros([self.n_x, self.n_y, self.n_theta, self.n_pol])

        kernel = np.array([[0., 0., 0.],
                           [0.5, -1., 0.5],
                           [0., 0., 0.]])

        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                aux[:, :, i, j] = self.lattice[:, :, i, j] + diff * signal.convolve2d(self.lattice[:, :, i, j], kernel,
                                                                                      mode="same", boundary="wrap")
                # aux[:,:,i,j] = self.lattice[:,:,i,j] + D*signal.convolve2d(self.lattice[:,:,i,j], kernel,
                # mode="same", boundary="fill")
        self.lattice = np.copy(aux)

    def diffusion_perp_dir(self):
        diff = self.diff_perp_dir
        aux = np.zeros([self.n_x, self.n_y, self.n_theta, self.n_pol])

        kernel = np.array([[0., 0.5, 0.],
                           [0., -1., 0.],
                           [0., 0.5, 0.]])

        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                aux[:, :, i, j] = self.lattice[:, :, i, j] + diff * signal.convolve2d(self.lattice[:, :, i, j], kernel,
                                                                                      mode="same", boundary="wrap")
                # aux[:,:,i,j] = self.lattice[:,:,i,j] + diff*signal.convolve2d(self.lattice[:,:,i,j], kernel,
                # mode="same", boundary="fill")
        self.lattice = np.copy(aux)

    def drift_para_dir(self):
        # ROLL METHOD (Uses numpy function roll to convect the pulse in a forwards direction)
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                self.lattice[:, :, i, j] = np.roll(self.lattice[:, :, i, j], j + 1, axis=1)  # Introduce velocity
                # dynamics (dissipation, diffusion and from 0 to 1)

    def diffusion_theta(self):
        diff = self.diff_theta
        aux = np.zeros([self.n_x, self.n_y, self.n_theta, self.n_pol])
        for j in range(0, self.n_pol):
            for i in range(0, self.n_theta):
                i_min = i - 1
                if i_min < 0:
                    i_min = self.n_theta - 1
                i_plus = i + 1
                if i_plus > self.n_theta - 1:
                    i_plus = 0

                aux_diff = 0.5 * (self.real_lattice[:, :, i_min, j] +
                                  self.real_lattice[:, :, i_plus, j] - 2. * self.real_lattice[:, :, i, j])
                aux[:, :, i, j] = self.real_lattice[:, :, i, j] + diff * aux_diff

        self.real_lattice = np.copy(aux)

    def polarization_dynamics_dissipation(self):
        aux = np.zeros([self.n_x, self.n_y, self.n_theta, self.n_pol])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for k in range(0, self.n_x):
                    for m in range(0, self.n_y):
                        j_diss = int((j + self.gamma))  # force the pdf from j=0 to j=1
                        if j_diss < self.n_pol:
                            aux[k, m, i, j] = self.lattice[k, m, i, j_diss]
                        else:
                            aux[k, m, i, j] = self.lattice[k, m, i, j]
        self.lattice = np.copy(aux)

    def to_real_lattice(self):
        aux = np.zeros([self.n_x, self.n_y, self.n_theta, self.n_pol])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for x in range(0, math.floor(self.n_x)):
                    for y in range(0, math.floor(self.n_y)):
                        d_theta = i * (2. * math.pi / self.n_theta)

                        y_real = y - self.n_y / 2.
                        x_real = x - self.n_x / 2.

                        x_new = x_real * math.cos(d_theta) - y_real * math.sin(d_theta) + self.n_x / 2.
                        y_new = x_real * math.sin(d_theta) + y_real * math.cos(d_theta) + self.n_y / 2.

                        x_new = wrap(x_new, self.n_x - 1)
                        y_new = wrap(y_new, self.n_y - 1)

                        aux[x_new, y_new, i, j] = aux[x_new, y_new, i, j] + self.lattice[x, y, i, j]
        self.real_lattice = np.copy(aux)

    def from_real_lattice(self):
        aux = np.zeros([self.n_x, self.n_y, self.n_theta, self.n_pol])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for x in range(0, math.floor(self.n_x)):
                    for y in range(0, math.floor(self.n_y)):
                        d_theta = i * (2. * math.pi / self.n_theta)
                        y_real = y - self.n_y / 2.
                        x_real = x - self.n_x / 2.

                        x_new = x_real * math.cos(d_theta) + y_real * math.sin(d_theta) + self.n_x / 2.
                        y_new = -x_real * math.sin(d_theta) + y_real * math.cos(d_theta) + self.n_y / 2.

                        x_new = wrap(x_new, self.n_x - 1)
                        y_new = wrap(y_new, self.n_y - 1)

                        aux[x_new, y_new, i, j] = aux[x_new, y_new, i, j] + self.real_lattice[x, y, i, j]
        self.lattice = np.copy(aux)

    # Try to apply the numba function to speed up the process
    def real_periodic_boundary(self):
        self.to_real_lattice()
        # for i, j, k, m in zip(range(0, self.n_theta), range):
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                for k in range(0, self.n_x):
                    for m in range(0, self.n_y):
                        if k > self.n_x / 2. + self.max_x / 2.:
                            aux = self.real_lattice[k, m, i, j]
                            self.real_lattice[k, m, i, j] = 0.
                            self.real_lattice[k - int(self.max_x), m, i, j] = self.real_lattice[
                                                                                  k - int(self.max_x), m, i, j] + aux
                        if m > self.n_x / 2. + self.max_y / 2.:
                            aux = self.real_lattice[k, m, i, j]
                            self.real_lattice[k, m, i, j] = 0.
                            self.real_lattice[k, m - int(self.max_y), i, j] = self.real_lattice[
                                                                                  k, m - int(self.max_y), i, j] + aux
                        if k < self.n_x / 2. - self.max_x / 2.:
                            aux = self.real_lattice[k, m, i, j]
                            self.real_lattice[k, m, i, j] = 0.
                            self.real_lattice[k + int(self.max_x), m, i, j] = self.real_lattice[
                                                                                  k + int(self.max_x), m, i, j] + aux
                        if m < self.n_x / 2. - self.max_y / 2.:
                            aux = self.real_lattice[k, m, i, j]
                            self.real_lattice[k, m, i, j] = 0.
                            self.real_lattice[k, m + int(self.max_y), i, j] = self.real_lattice[
                                                                                  k, m + int(self.max_y), i, j] + aux
        self.from_real_lattice()

    def print_state(self):
        aux = np.zeros([self.n_x, self.n_y])
        for i in range(0, self.n_theta):
            for j in range(0, self.n_pol):
                # aux[:,:] = aux[:,:] + self.lattice[:,:,i,j]
                aux[:, :] = aux[:, :] + self.real_lattice[:, :, i, j]

        plt.imshow(aux, cmap='hot')
        # block = False
        plt.pause(0.001)

    def total(self):
        prob = np.sum(np.concatenate(self.real_lattice))
        # prob = np.sum(np.concatenate(self.lattice))
        return prob
