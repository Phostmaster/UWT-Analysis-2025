import numpy as np
import matplotlib.pyplot as plt

D, k, beta = 1, -1, 4.07
r_min, r_max, dr = 0.1, 10, 0.05
nr = int((r_max - r_min) / dr) + 1
r = np.linspace(r_min, r_max, nr)
rho = 15 + 0.5 / r**2.07

for _ in np.arange(0, 50, 0.01):
    drho = np.diff(rho) / dr
    flux = r[1:]**2 * drho
    lap = (1 / r[:-1]**2) * np.diff(flux) / dr  # Approx spherical lap
    lap = np.pad(lap, (1, 0))  # Boundaries
    rho += 0.01 * (D * lap + k * rho**beta)
    rho[0] = rho[1]  # No-flux inner
    rho[-1] = rho[-2]  # Outer

plt.plot(r, rho); plt.xlabel('r'); plt.ylabel('ρ'); plt.title('3D ρ(r) at t=50')
plt.show()