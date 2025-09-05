import numpy as np
import matplotlib.pyplot as plt

# Params for UWT PDE
nx, ny = 50, 50
nt = 100
dx = dy = 10 / (nx - 1)
dt = 0.01  # Stability
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)
X, Y = np.meshgrid(x, y)
r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)  # Offset for bottom-left rise
rho = 15 + 5 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 1.0)  # Sunrise IC at (0.5, 0.5)

# UWT PDE params
D, k, beta = 1, 1, 4.07  # Positive growth

# Sim loop (explicit FD)
max_rho = 1000  # Cap
for n in range(nt):
    rho_new = rho.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            lap = (rho[i+1,j] - 2*rho[i,j] + rho[i-1,j]) / dx**2 + \
                  (rho[i,j+1] - 2*rho[i,j] + rho[i,j-1]) / dy**2
            rho_new[i,j] += dt * (D * lap + k * np.where(rho[i,j] > 0, min(rho[i,j], max_rho), 0.01)**beta)
    rho = np.where(np.isnan(rho_new) | np.isinf(rho_new), rho, rho_new)  # Handle nan/inf
    rho[0, :] = rho[1, :]  # No-flux
    rho[-1, :] = rho[-2, :]  # No-flux
    rho[:, 0] = rho[:, 1]  # No-flux
    rho[:, -1] = rho[:, -2]  # No-flux

# Plot
plt.contourf(X, Y, rho, levels=50, cmap='YlOrRd')  # Sunrise colors
plt.colorbar(label='ρ')
plt.title('UWT 2D Wave Propagation at t=50 (Sunrise)')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('uwt_2d_sunrise.png')
plt.show()
