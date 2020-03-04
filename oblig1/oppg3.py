import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve, broyden1


class Minimise:
    """
    Class that solves a set of equations obtained from the Lagrangian multiplier
    method.
    N is the number of rods
    V is the dimensionless volume
    alpha, and gamma are parameters
    """
    def __init__(self, N, V, alpha=1, gamma=1):
        self.N = N
        self.V = V
        self.alpha = alpha
        self.gamma = gamma

    def helmholtz(self, Nx, Ny, Nz, V):
        N = self.N
        alpha, gamma = self.alpha, self.gamma
        #FT = Nx*np.log(alpha*Nx/V) + Ny*np.log(alpha*Ny/V) + Nz*np.log(alpha*Nz/V) + gamma*(Nx*Ny + Ny*Nz + Nz*Nx)/V
        FT = Nx*np.log(Nx) + Ny*np.log(Ny) + Nz*np.log(Nz) + gamma*(Nx*Ny + Ny*Nz + Nz*Nx)/V + N*np.log(alpha) - N*np.log(V)
        return FT

    def equation(self, Ni, Nj, Nk, lamb, V):
        return np.log(Ni) + self.gamma*(Nj + Nk)/V + np.log(self.alpha/V) + lamb + 1

    def equations(self, p, V):
        Nx, Ny, Nz, lamb = p
        return [
                self.equation(Nx, Ny, Nz, lamb, V),
                self.equation(Ny, Nx, Nz, lamb, V),
                self.equation(Nz, Ny, Nx, lamb, V),
                Nx + Ny + Nz - self.N
                ]

    def solve(self, guesses):
        helmholtz = []
        for V in self.V:
            #guesses[-1] -= 1
            Nx, Ny, Nz, lamb = fsolve(self.equations, guesses, args=(V), factor=0.1)
            helmholtz.append(self.helmholtz(Nx, Ny, Nz, V))
        return self.N/self.V, helmholtz





N = 100
V = np.linspace(1e-2, 1, 2e4)

minimise = Minimise(N, V, alpha=1, gamma=0.001)
u, v = minimise.solve([1/3*N, 1/3*N, 1/3*N, -100])


sns.set()
sns.set_style("white")

fig, ax = plt.subplots(2, 1)
ax[0].semilogx(u, v, linewidth=1.5, label="Minimised")
ax[0].semilogx(u, minimise.helmholtz(1/6*N, 2/6*N, 3/6*N, V), label="Test case")
ax[0].set_ylabel("F/T", fontsize=14)
ax[1].semilogx(u, np.gradient(v))
ax[1].set_xlabel("N/V", fontsize=14)
ax[1].set_ylabel(r"$\nabla$ F/T", fontsize=14)
ax[0].legend(frameon=False, fontsize=14, loc="best")
fig.tight_layout()
plt.show()
