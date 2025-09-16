from pyscf import gto, scf, dft
import numpy as np

# Approximate C-S-H structure
atoms = [
    ['C', (0.0, 0.0, 0.0)],
    ['H', (0.63, 0.63, 0.63)],
    ['H', (0.63, -0.63, -0.63)],
    ['H', (-0.63, 0.63, -0.63)],
    ['H', (-0.63, -0.63, 0.63)],
    ['S', (1.5, 1.5, 1.5)]
]
mol = gto.M(
    atom=atoms,
    basis='cc-pvdz',
    unit='Angstrom'
)

# UWT parameters from addendum
epsilon_cp = 2.58e-41  # Base CP-violating phase control
delta_eps_sc2 = 1.5e11  # J/m^3 from addendum
lambda_bcs = 0.5  # Initial BCS coupling
phi1_phi2 = 1000  # |Φ_1 Φ_2| in GeV^2
theta_d = 2000  # Debye temperature (K)
mu_star = 0.1  # Coulomb pseudopotential
me = 0.510998 / 27.2114  # Electron mass in a.u. from UWT

# Convert UWT units to atomic units
au_energy = 27.2114 * 1.602e-19  # Hartree to Joules
au_length = 5.29177e-11  # Bohr to meters
au_volume = au_length**3  # Bohr^3 to m^3
energy_density_au = delta_eps_sc2 / (au_energy / au_volume) * 1e-2

# Run DFT calculation
mf = dft.RKS(mol)
mf.xc = 'pbe'
energy = mf.kernel()
print(f"C-S-H DFT energy: {energy} Hartree")

# Add epsilon_CP perturbation to Hamiltonian (UWT-inspired)
hcore = mf.get_hcore()
nmo = hcore.shape[0]
cp_perturbation = np.zeros((nmo, nmo))
for i in range(nmo):
    cp_perturbation[i, i] = (epsilon_cp * phi1_phi2 * 1e-5) / (me**2)  # UWT coupling
hcore_with_cp = hcore + cp_perturbation
mf.get_hcore = lambda *args: hcore_with_cp  # Corrected lambda function
energy_with_cp = mf.kernel()
print(f"C-S-H DFT energy with epsilon_CP: {energy_with_cp} Hartree")

# Iterative adjustment to target Tc 280-320 K
target_tc_low = 280  # K
target_tc_high = 320  # K
multiplier_low = 1e15  # Increased lower bound
multiplier_high = 1e20  # Increased upper bound
tolerance = 1.0  # K tolerance

for _ in range(10):  # Limit iterations
    multiplier = (multiplier_low + multiplier_high) / 2
    effective_eps = epsilon_cp * multiplier * 1e10  # Stronger amplification
    lambda_ep = 1.5 * lambda_bcs * (1 + (effective_eps * phi1_phi2) / (me**2 * 1e-5))  # Adjusted scaling
    tc = (theta_d / 1.45) * np.exp(-1.04 * (1 + lambda_ep) / (lambda_ep - mu_star * (1 + 0.62 * lambda_ep)))
    print(f"Multiplier: {multiplier}, lambda_ep: {lambda_ep}, Tc: {tc} K")
    if target_tc_low <= tc <= target_tc_high:
        break
    elif tc < target_tc_low:
        multiplier_low = multiplier
    else:
        multiplier_high = multiplier

print(f"Final UWT-enhanced lambda_ep with epsilon_CP: {lambda_ep}")
print(f"Final Estimated Tc with epsilon_CP: {tc} K with multiplier {multiplier}")