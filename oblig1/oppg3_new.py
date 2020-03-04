import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Helmholtz:
    def __init__(self, N, V, alpha, gamma):
        self.N, self.V = N, V
        self.alpha, self.gamma = alpha, gamma
        return

    def f(self, Nx, Ny, Nz, V):
        """
        Calculates temperature scaled helmholtz free energy for constant temperature
        """
        N = self.N
        alpha, gamma = self.alpha, self.gamma
        F = Nx*np.log(Nx) + Ny*np.log(Ny) + Nz*np.log(Nz) + gamma*(Nx*Ny + Ny*Nz + Nz*Nx)/V + N*np.log(alpha) - N*np.log(V)
        return F

    def construct_grid(self):
        """
        Calculates possible combinations of Nx, Ny, and Nz with constraint
            Nx + Ny + Nz = N.
        Calculates diagonal of a (N-2)x(N-2)x(N-2) meshgrid.
        """
        N = np.linspace(1, self.N-2, self.N-2, dtype=int)
        Nmatrix = []
        for i in N:
            for j in N:
                for k in N:
                    if i + j + k == self.N:
                        Nmatrix.append([i, j, k])
        Nmatrix = np.array(Nmatrix)
        return Nmatrix[:, 0], Nmatrix[:, 1], Nmatrix[:, 2]

    def minimise(self):
        """
        Locates minima of helmholtz(self, ...) over all combinations of Nx, Ny, Nz, i.e.
        construct_grid(self).
        """
        Nx, Ny, Nz = self.construct_grid()
        minimised_F = np.zeros_like(self.V)
        for i in range(len(self.V)):
            F = self.f(Nx, Ny, Nz, self.V[i])
            minimum_index = np.argmin(F)
            minimised_F[i] = F[minimum_index]
        return minimised_F

class Gibbs(Helmholtz):
    def f(self, Nx, Ny, Nz, V):
        P = self.gamma*(Nx*Ny + Ny*Nz + Nx*Nz)/V**2 + N/V
        G = super().f(Nx, Ny, Nz, V) + P*V
        return G


N = 100
V = np.linspace(1e-2, 1e-1, 1e3)
gamma = 0.001

helmholtz = Helmholtz(N, V, alpha=1, gamma=gamma)
minimised_helmholtz = helmholtz.minimise()

gibbs = Gibbs(N, V, alpha=1, gamma=gamma)
minimised_gibbs = gibbs.minimise()

P = -np.diff(minimised_helmholtz)/np.diff(V)

mini_gibbs = minimised_helmholtz[:-1] + P*V[:-1]




sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")

fig, ax = plt.subplots(3, 1)
ax[0].semilogx(N/V, minimised_helmholtz, color="k", label="Helmholtz")
ax[0].semilogx(N/V, minimised_gibbs, color="k", linestyle="dashdot", label="Gibbs")
ax[0].set_ylabel("F/T", fontsize=14)
ax[1].semilogx(N/V, np.gradient(minimised_helmholtz), color="k")
ax[1].semilogx(N/V, np.gradient(minimised_gibbs), color="k", linestyle="dashdot")
ax[1].set_ylabel(r"$\nabla$ F/T", fontsize=14)
ax[2].semilogx(N/V, np.gradient(np.gradient(minimised_helmholtz)), color="k")
ax[2].semilogx(N/V, np.gradient(np.gradient(minimised_gibbs)), color="k", linestyle="dashdot")
ax[2].set_ylabel(r"$\nabla^2$ F/T", fontsize=14)
ax[2].set_xlabel("N/V", fontsize=14)
#ax[0].legend(frameon=False, fontsize=14, loc="upper left")
fig.tight_layout()
#plt.savefig("oppg3c.pdf")

fig, ax = plt.subplots(1, 1)
ax.semilogx(P, mini_gibbs, color="k")
ax.set_xlabel("P", fontsize=14)
ax.set_ylabel("G/T", fontsize=14)
fig.tight_layout()
#plt.savefig("oppg3c.pdf")
plt.show()
