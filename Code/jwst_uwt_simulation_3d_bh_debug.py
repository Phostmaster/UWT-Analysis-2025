import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from scipy.integrate import solve_ivp

# UWT and Physical Parameters (Synced with xGrok)
g_wave_small = 19.5    # High value for small-scale collapses
g_wave_large = 0.085   # Low value for large-scale harmony
phi_1_init = 0.8       # GeV, matching xGrok
phi_2_init = 0.4       # GeV, matching xGrok
epsilon_CP = 2.58e-41
kwave = 2.35e-3
lambda_d = 1e20
G = 6.67430e-11
c = 299792458.0
v = 0.226
Lambda = 1e-46
axion_mass = 1e-22     # eV, xGrok's ultra-light axion mass

# Simulation Config
N = 32                  # Grid size
L = 3.086e22            # 1 Mpc, matching xGrok
dx = L / N
N_particles = 300       # Increased to 300 per xGrok for more clustering
particle_mass = 1e30    # Placeholder, to be refined
dt = 1e10               # Timestep (seconds)
t_max = 6e12            # Extended to 6e12 s for 600 steps (xGrok advice)
num_steps = int(t_max / dt)  # ~600 steps
density_threshold = 1e-33  # Lowered to 1e-33 per xGrok for more BHs
gravitational_parameter_threshold = 1e4  # Kept as is
rho_0 = 4.5e-20         # kg/m³, converted from 2.5e-5 GeV/cm³ (xGrok)
delta_rho = 1e-20       # Increased to 1e-20 per xGrok for stronger perturbations
output_directory = os.path.join(os.path.expanduser("~"), "Desktop", "Grok")
os.makedirs(output_directory, exist_ok=True)
print(f"Output directory set to: {output_directory}")

# Cosmology Parameters
h = 0.7
H0 = 2.133e-18          # s^-1, adjusted for z ~ 10 at 4.32e12 s
Omega_m = 0.3
Omega_L = 0.7
Omega_k = 0.0

def Ez(z):
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_L + Omega_k * (1 + z)**2)

def cosmic_time(z):
    z_grid = np.linspace(0, 100, 2000)
    integrand = 1.0 / ((1.0 + z_grid) * Ez(z_grid))
    return np.trapezoid(integrand, z_grid) / H0

def time_to_z(t):
    z_grid = np.linspace(0, 100, 2000)
    t_grid = np.array([cosmic_time(z) for z in z_grid])
    idx = np.searchsorted(t_grid, t, side='right') - 1
    if idx < 0:
        return z_grid[0]
    elif idx >= len(z_grid) - 1:
        return z_grid[-1]
    return z_grid[idx] + (t - t_grid[idx]) * (z_grid[idx + 1] - z_grid[idx]) / (t_grid[idx + 1] - t_grid[idx])

def a_of_z(z):
    return 1.0 / (1.0 + z)

def D_linear_LCDM(z):
    a = a_of_z(z)
    a_grid = np.linspace(1e-5, a, 4000)
    Ea = np.sqrt(Omega_m / a_grid**3 + Omega_L + Omega_k / a_grid**2)
    integrand = 1.0 / (a_grid**3 * Ea)
    integral = np.trapezoid(integrand, a_grid)
    D_a = (5.0 * Omega_m / 2.0) * Ea[-1] * integral
    a1 = 1.0
    a_grid1 = np.linspace(1e-5, a1, 4000)
    Ea1 = np.sqrt(Omega_m / a_grid1**3 + Omega_L + Omega_k / a_grid1**2)
    integral1 = np.trapezoid(1.0 / (a_grid1**3 * Ea1), a_grid1)
    D_1 = (5.0 * Omega_m / 2.0) * Ea1[-1] * integral1
    return D_a / D_1

def D_linear_UWT(z):
    return D_linear_LCDM(z) * (a_of_z(z) ** (2.5 * g_wave_small * (phi_1_init * phi_2_init)))

# Gradient Function
def gradient(arr, spacing=dx):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.ndim}D array with shape {arr.shape}")
    n_x, n_y, n_z = arr.shape
    grad_x = np.zeros_like(arr)
    grad_y = np.zeros_like(arr)
    grad_z = np.zeros_like(arr)
    # Central differences for interior
    if n_x > 2:
        grad_x[1:-1, :, :] = (arr[2:, :, :] - arr[:-2, :, :]) / (2 * spacing)
    if n_y > 2:
        grad_y[:, 1:-1, :] = (arr[:, 2:, :] - arr[:, :-2, :]) / (2 * spacing)
    if n_z > 2:
        grad_z[:, :, 1:-1] = (arr[:, :, 2:] - arr[:, :, :-2]) / (2 * spacing)
    # Forward/backward differences for edges
    if n_x > 1:
        grad_x[0, :, :] = (arr[1, :, :] - arr[0, :, :]) / spacing
        grad_x[-1, :, :] = (arr[-1, :, :] - arr[-2, :, :]) / spacing
    if n_y > 1:
        grad_y[:, 0, :] = (arr[:, 1, :] - arr[:, 0, :]) / spacing
        grad_y[:, -1, :] = (arr[:, -1, :] - arr[:, -2, :]) / spacing
    if n_z > 1:
        grad_z[:, :, 0] = (arr[:, :, 1] - arr[:, :, 0]) / spacing
        grad_z[:, :, -1] = (arr[:, :, -1] - arr[:, :, -2]) / spacing
    return grad_x, grad_y, grad_z

# Batched Hessian Function
def hessian_batched(arr, spacing=dx, batch_size=8):
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape != (N, N, N):
        raise ValueError(f"Expected 3D array of shape ({N}, {N}, {N}), got {arr.shape}")
    hess = np.zeros((N, N, N, 3, 3), dtype=np.float32)  # Use float32 to reduce memory
    for i in range(0, N, batch_size):
        for j in range(0, N, batch_size):
            for k in range(0, N, batch_size):
                i_end = min(i + batch_size, N)
                j_end = min(j + batch_size, N)
                k_end = min(k + batch_size, N)
                sub_arr = arr[i:i_end, j:j_end, k:k_end]
                sub_nx, sub_ny, sub_nz = sub_arr.shape
                grad_x, grad_y, grad_z = gradient(sub_arr, spacing)
                # Pad sub-grid gradients
                grad_x_padded = np.pad(grad_x, ((1, 1), (1, 1), (1, 1)), mode='edge')
                grad_y_padded = np.pad(grad_y, ((1, 1), (1, 1), (1, 1)), mode='edge')
                grad_z_padded = np.pad(grad_z, ((1, 1), (1, 1), (1, 1)), mode='edge')
                # Central differences adjusted for full 8x8x8 sub-grid
                hess[i:i_end, j:j_end, k:k_end, 0, 0] = (grad_x_padded[1:9, 1:9, 1:9] - 2 * grad_x_padded[0:8, 1:9, 1:9] + grad_x_padded[0:8, 1:9, 1:9]) / (spacing ** 2)
                hess[i:i_end, j:j_end, k:k_end, 1, 1] = (grad_y_padded[1:9, 1:9, 1:9] - 2 * grad_y_padded[1:9, 0:8, 1:9] + grad_y_padded[1:9, 0:8, 1:9]) / (spacing ** 2)
                hess[i:i_end, j:j_end, k:k_end, 2, 2] = (grad_z_padded[1:9, 1:9, 1:9] - 2 * grad_z_padded[1:9, 1:9, 0:8] + grad_z_padded[1:9, 1:9, 0:8]) / (spacing ** 2)
    return hess

def scalar_field_derivs(t, y, rho, G_eff, g_wave):
    phi_1 = y[:N * N * N].reshape(N, N, N)
    phi_2 = y[N * N * N:2 * N * N * N].reshape(N, N, N)
    phi_mag_squared = phi_1**2 + phi_2**2
    potential = Lambda * (phi_mag_squared - v**2)
    grad_x_1, grad_y_1, grad_z_1 = gradient(phi_1)
    grad_x_2, grad_y_2, grad_z_2 = gradient(phi_2)
    # Batch Hessian computation
    hess_1 = hessian_batched(phi_1, batch_size=8)
    hess_2 = hessian_batched(phi_2, batch_size=8)
    # Enhanced collapse criterion with density weighting
    collapse_mask = (np.trace(hess_1, axis1=3, axis2=4) + np.trace(hess_2, axis1=3, axis2=4) < 0) & (rho > 1e-29)
    dphi1_dt = -potential * phi_1 - (grad_x_1**2 + grad_y_1**2 + grad_z_1**2) / (2 * c**2) + g_wave * phi_1 * collapse_mask
    dphi2_dt = -potential * phi_2 - (grad_x_2**2 + grad_y_2**2 + grad_z_2**2) / (2 * c**2) + g_wave * phi_2 * collapse_mask
    return np.concatenate([dphi1_dt.ravel(), dphi2_dt.ravel()])

def compute_forces(particle_positions, G_eff, dx, N, R_safe, rho):
    N_particles = particle_positions.shape[0]
    forces = np.zeros((N_particles, 3))
    for i in range(N_particles):
        for j in range(N_particles):
            if i != j:
                r = particle_positions[j] - particle_positions[i]
                dist = np.linalg.norm(r)
                if dist < 1e-10:
                    continue
                ix = int(np.clip(particle_positions[i][0] / dx + N / 2, 0, N - 1))
                iy = int(np.clip(particle_positions[i][1] / dx + N / 2, 0, N - 1))
                iz = int(np.clip(particle_positions[i][2] / dx + N / 2, 0, N - 1))
                seed_mass_kg = 1.47e43  # 7.37e12 M☉ in kg
                force_magnitude = (G_eff[ix, iy, iz] * seed_mass_kg * rho[ix, iy, iz] * dx ** 3) / (dist ** 2 + 1e-10)
                force = force_magnitude * (r / dist)
                forces[i] += force
    return forces

def map_particle_to_grid_indices(position):
    ix = int(np.clip(np.floor((position[0] + L/2) / dx), 0, N - 1))
    iy = int(np.clip(np.floor((position[1] + L/2) / dx), 0, N - 1))
    iz = int(np.clip(np.floor((position[2] + L/2) / dx), 0, N - 1))
    return ix, iy, iz

def analyze_black_holes(step, rho, G_eff, R_safe, X, Y, Z, density_threshold, gravitational_parameter_threshold, output_directory, particle_mass, particle_positions=None, seed_data=None):
    black_hole_locations = []
    black_hole_masses = []
    black_hole_gravitational_parameters = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if rho[i, j, k] > density_threshold:
                    gravitational_parameter = (G_eff[i, j, k] * particle_mass) / (R_safe[i, j, k] + 1e-10)
                    if gravitational_parameter > gravitational_parameter_threshold:
                        black_hole_locations.append([i, j, k])  # Log indices instead of arrays
                        black_hole_masses.append(rho[i, j, k] * dx ** 3)
                        black_hole_gravitational_parameters.append(gravitational_parameter)
    log_filename = os.path.join(output_directory, f"black_hole_analysis_step_{step}.txt")
    with open(log_filename, 'w') as f:
        f.write(f"Step {step} - High-Density locations (i,j,k): {black_hole_locations}\n")
        f.write(f"Step {step} - High-Density Black Hole Masses (kg): {black_hole_masses}\n")
        f.write(f"Step {step} - High-Density Gravitational Parameters: {black_hole_gravitational_parameters}\n")
    print(f"[Step {step}] Black hole log saved to {log_filename}")
    sys.stdout.flush()
    slice_index = N // 2
    plt.figure()
    plt.imshow(rho[:, :, slice_index], extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(f"Density Slice Step {step} at z={time_to_z(step * dt):.2f}")
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    density_filename = os.path.join(output_directory, f"density_slice_step_{step}.png")
    plt.savefig(density_filename)
    plt.close()
    print(f"[Step {step}] Density slice image saved to {density_filename}")
    sys.stdout.flush()
    if len(black_hole_masses) > 1 and np.ptp(black_hole_masses) > 0:
        plt.figure(figsize=(8, 6))
        plt.hist(black_hole_masses, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Black Hole Mass (kg)')
        plt.ylabel('Frequency')
        plt.title(f'Black Hole Mass Distribution - Step {step} at z={time_to_z(step * dt):.2f}')
        plt.grid(True)
        mass_hist_filename = os.path.join(output_directory, f"mass_distribution_step_{step}.png")
        plt.savefig(mass_hist_filename)
        plt.close()
        print(f"[Step {step}] Black hole mass histogram saved to {mass_hist_filename}")
        sys.stdout.flush()
    else:
        print(f"[Step {step}] Not enough black hole mass data to plot histogram.")
        sys.stdout.flush()
    if particle_positions is not None and seed_data is not None:
        compare_with_seed_collapse(step, particle_positions, seed_data)

def compare_with_seed_collapse(step, particle_positions, seed_data):
    seed_dict = {(int(row[0]), int(row[1]), int(row[2])): row for row in seed_data}
    matched_particles = []
    for p_idx, pos in enumerate(particle_positions):
        grid_idx = map_particle_to_grid_indices(pos)
        if grid_idx in seed_dict:
            seed_info = seed_dict[grid_idx]
            matched_particles.append((p_idx, grid_idx, seed_info))
    print(f"[Step {step}] Particles matched with seed locations: {len(matched_particles)}")
    sys.stdout.flush()
    for p_idx, grid_idx, seed_info in matched_particles[:10]:  # Print first 10 matches
        mass_seed = seed_info[5] * 1.989e30  # Solar mass to kg
        print(f"Particle {p_idx} at {grid_idx} Seed M={mass_seed:.3e}kg Final M_z10={seed_info[8]:.3e}M_solar")
        sys.stdout.flush()

# Seed/Collapse Data for Validation (Full 46 rows)
seed_collapse_data = np.array([
    [32, 32, 32, 35.0, 8.983124194687981e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 33, 31, 35.0, 7.888611061442364e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 31, 30, 35.0, 8.019209052682889e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 30, 31, 35.0, 8.007819216980842e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 32, 33, 35.0, 7.982442705491984e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 33, 32, 35.0, 7.9725127040281e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 33, 33, 35.0, 7.965407042076054e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [30, 31, 31, 35.0, 7.95183980543665e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 31, 30, 35.0, 7.951823375534638e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 30, 31, 35.0, 7.946716799287924e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 33, 33, 35.0, 7.899690784049157e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 31, 33, 35.0, 7.888978881449854e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 32, 30, 35.0, 7.751067008991716e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 32, 31, 35.0, 8.885475440607023e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 30, 33, 35.0, 7.749264321239462e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 30, 32, 35.0, 7.748531871890105e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [30, 33, 32, 35.0, 7.748279451600597e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 33, 30, 35.0, 7.741835481782553e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [30, 32, 33, 35.0, 7.740620059199557e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 33, 33, 35.0, 7.69815659496259e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 31, 30, 35.0, 7.693099744670392e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 33, 30, 35.0, 7.68315411094809e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [30, 31, 33, 35.0, 7.677880903830582e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [30, 33, 31, 35.0, 7.677327974806516e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 30, 32, 35.0, 8.021154486137125e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 32, 30, 35.0, 8.023142142339814e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [30, 31, 32, 35.0, 8.02363498531941e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [30, 32, 31, 35.0, 8.037712454156604e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 32, 32, 35.0, 8.885006407621512e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 31, 32, 35.0, 8.88144095247218e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 31, 32, 35.0, 8.804221949030252e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 32, 31, 35.0, 8.801589024325634e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 31, 31, 35.0, 8.795087498583454e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 31, 31, 35.0, 8.710296620995778e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 33, 32, 35.0, 8.347798489482096e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 32, 33, 35.0, 8.340723103590118e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 32, 32, 35.0, 8.328282323745919e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 33, 31, 35.0, 8.271603055041964e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 31, 33, 35.0, 8.26707053888205e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 32, 33, 35.0, 8.26540279003734e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 33, 32, 35.0, 8.263875106119706e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 31, 32, 35.0, 8.263687427045089e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 32, 31, 35.0, 8.261335737316413e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 31, 33, 35.0, 8.186103939756999e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 31, 31, 35.0, 8.180814050609055e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [31, 33, 31, 35.0, 8.161641955651019e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 32, 30, 35.0, 8.099875542596881e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [30, 32, 32, 35.0, 8.092147613916298e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [32, 30, 32, 35.0, 8.087521523870746e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16],
    [33, 30, 31, 35.0, 7.67711463771801e-05, 7368841560809.953, 0.1, 0.7, 2.6009758802674644e+16]
])

# Initial Conditions with Enhanced Asymmetric Perturbations
np.random.seed(42)
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
y = np.linspace(-L / 2, L / 2, N, endpoint=False)
z = np.linspace(-L / 2, L / 2, N, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z)
R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
R_safe = np.where(R == 0, 1e-30, R)
# Enhanced asymmetric perturbations per xGrok
perturbation += 0.25 * np.cos(X / dx) + 0.2 * np.sin(Y / dx) + 0.15 * np.cos(Z / dx)
phi_1 = phi_1_init * (np.cos(kwave * R) * np.exp(-np.abs(R) / lambda_d) + perturbation)
phi_2 = phi_2_init * (np.sin(kwave * R + epsilon_CP * np.pi) * np.exp(-np.abs(R) / lambda_d) + perturbation)
rho = rho_0 + delta_rho = 3e-20
phi_mag_squared = phi_1 ** 2 + phi_2 ** 2
G_eff = G * (1 + g_wave_small * phi_mag_squared)  # Start with small-scale g_wave

# Initialize particle positions to sample all seed locations
seed_indices = seed_collapse_data[:, :3].astype(int)
seed_positions = np.zeros((N_particles, 3))
for i in range(N_particles):
    idx = i % len(seed_indices)
    seed_pos = seed_indices[idx]
    seed_positions[i] = [x[seed_pos[0] % N] + np.random.uniform(-dx/2, dx/2),
                         y[seed_pos[1] % N] + np.random.uniform(-dx/2, dx/2),
                         z[seed_pos[2] % N] + np.random.uniform(-dx/2, dx/2)]
particle_positions = seed_positions
particle_velocities = np.zeros((N_particles, 3))

print(f"Starting simulation with Grid={N}³, Particles={N_particles}, Steps={num_steps}")
try:
    for step in range(num_steps):
        print(f"Step {step} starting at t={step * dt:.2e}s...")
        sys.stdout.flush()
        start_time = time.time()
        y0 = np.concatenate([phi_1.ravel(), phi_2.ravel()])
        print(f"Initial y0 shape: {y0.shape}, expected: {(2 * N * N * N,)}")
        sys.stdout.flush()
        # Switch g_wave based on time (xGrok's peak at 2.1e12 s)
        g_wave = g_wave_small if step * dt < 2.1e12 else g_wave_large
        sol = solve_ivp(lambda t, y, rho, G_eff, g_wave: scalar_field_derivs(t, y, rho, G_eff, g_wave), [0, dt], y0, method='RK45', rtol=1e-3, atol=1e-6, args=(rho, G_eff, g_wave), max_step=dt)
        if not sol.success:
            print(f"Step {step} integration failed: {sol.message}")
            sys.stdout.flush()
            break
        phi_1 = sol.y[:N * N * N, -1].reshape(N, N, N)
        phi_2 = sol.y[N * N * N:2 * N * N * N, -1].reshape(N, N, N)
        print(f"Post-RK4 phi_1 shape: {phi_1.shape}, phi_2 shape: {phi_2.shape}")
        sys.stdout.flush()
        phi_mag_squared = phi_1**2 + phi_2**2
        G_eff = G * (1 + g_wave * phi_mag_squared)
        forces = compute_forces(particle_positions, G_eff, dx, N, R_safe, rho)
        particle_velocities += (forces / particle_mass) * dt
        particle_positions += particle_velocities * dt
        z = time_to_z(step * dt)
        D_growth = D_linear_UWT(z)
        phi_1 *= D_growth
        phi_2 *= D_growth
        rho = rho_0 + delta_rho * (phi_1 + phi_2)
        if step % 50 == 0:  # Adjusted to every 50 steps for your dump plan
            analyze_black_holes(step, rho, G_eff, R_safe, X, Y, Z, density_threshold, gravitational_parameter_threshold, output_directory, particle_mass, particle_positions, seed_collapse_data)
            print(f"Step {step} completed in {time.time() - start_time:.2f} seconds, z={z:.2f}, D_growth={D_growth:.2e}, g_wave={g_wave}")
            sys.stdout.flush()
        else:
            print(f"Step {step} completed in {time.time() - start_time:.2f} seconds")
            sys.stdout.flush()
except Exception as e:
    import traceback
    print(f"Fatal error: {e}")
    traceback.print_exc()
    sys.stdout.flush()
print("Simulation finished.")
sys.stdout.flush()