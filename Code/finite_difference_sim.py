import numpy as np
import matplotlib.pyplot as plt

# Params
D, k, beta = 1, -1, 4.07
r_min, r_max, dr = 0.1, 10, 0.05
t_max, dt = 50, 0.01
nr = int((r_max - r_min) / dr) + 1
r = np.linspace(r_min, r_max, nr)
rho = 15 + 0.5 / r**2.07  # IC

# Sim loop (explicit FD)
for _ in np.arange(0, t_max, dt):
    lap = (np.roll(rho, -1) - 2*rho + np.roll(rho, 1)) / dr**2 + (1/r) * (np.roll(rho, -1) - np.roll(rho, 1)) / (2*dr)  # Radial laplacian approx
    rho += dt * (D * lap + k * rho**beta)  # Should be k * r**-beta?
    rho[0] = rho[1]  # No-flux inner
    rho[-1] = rho[-2]  # Outer

# Plot (unsafe link)
plt.plot(r, rho); plt.xlabel('r'); plt.ylabel('ρ'); plt.title('ρ(r) at t=50');
plt.show()
