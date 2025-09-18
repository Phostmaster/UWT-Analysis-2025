import numpy as np
from pyscf import gto, scf, dft
from pyscf.dft import Grids
from pyscf.dft import numint
import scipy
import os
from scipy.constants import physical_constants

# Check SciPy version and define mu_B
print(f"SciPy version: {scipy.__version__}")
try:
    from scipy.constants import mu_B
except ImportError:
    print("mu_B not found in scipy.constants. Defining manually.")
    mu_B = 9.2740100783e-24 / 4.3597447222071e-18  # J/T -> Ha/T
    from scipy.constants import e, epsilon_0, c, hbar

# Define physical constants
m_e, a_0 = physical_constants['electron mass'], physical_constants['Bohr radius']
m_e = m_e[0]  # Value in kg
a_0 = a_0[0]  # Value in m
hbar_ha = hbar / 4.3597447222071e-18  # hbar in Ha·s
a_0_bohr = a_0 * 1.8897261245650618e10  # m to Bohr
energy_density_factor = (hbar_ha**2) / (m_e * a_0_bohr**2)  # Ha/Bohr² per kg/m²
volume_conversion = (1e-10)**3  # m³ to Bohr³

# Define molecule (C-S-H structure)
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
    basis='cc-pvdz',  # Improved basis
    unit='Angstrom'
)
mol.build()

# UWT Parameters (adjusted to Hartree units)
m = 0.02 / 27.2114  # Mass scale (eV -> Ha)
lmbda = 1e-3        # Self-interaction strength (dimensionless)
gwave = 0.1         # Coupling strength (dimensionless)
theta = (2.58e-41 / (3e-6)**3) * energy_density_factor * volume_conversion  # kg/m³ to Ha/Bohr³
kappa = (9.109e-41 / (3e-6)**3) * energy_density_factor * volume_conversion  # kg/m³ to Ha/Bohr³
Phi1Phi2_abs = 1.0 / 27.2114**2  # GeV^2 -> Ha^2
eps_CP = 1e-3      # CP violation tuning (dimensionless)
g_SO_strength = 6.5e-3  # Adjusted spin-orbit coupling strength
Delta_a = 1.165e-3  # Anomalous magnetic moment (muon g-2, dimensionless)

# Scalar field and UWT potential
def scalar_field(coords):
    """Defines the scalar field phi(r) at a given coordinate."""
    r = np.linalg.norm(coords)
    return np.exp(-r**2 / 1e-10)  # Gaussian decay, adjusted sigma

def uwt_potential(coords, phi_global=None):
    """Calculates the UWT potential at a given coordinate."""
    if phi_global is None:
        phi = scalar_field(coords)
    else:
        phi = phi_global
    pot = m**2 * phi**2 + lmbda * phi**4 + eps_CP * gwave * phi**4 / Phi1Phi2_abs
    return pot

# Custom Hamiltonian with UWT terms
def get_veff_uwt(mol, dm):
    grids = Grids(mol)
    grids.level = 2
    grids.build()
    coords = grids.coords
    weights = grids.weights
    phi_global = np.mean([scalar_field(coord) for coord in mol.atom_coords()])
    uwt_pot = np.array([uwt_potential(coord, phi_global) for coord in coords])
    ni = numint.NumInt()
    ao_grid = ni.eval_ao(mol, coords, deriv=0)  # AO values on grid
    uwt_mat = np.einsum('pi,p,p,pj->ij', ao_grid, weights, uwt_pot, ao_grid)
    return uwt_mat

def uwt_hamiltonian(mol, dm=None):
    if dm is None:
        dm = np.zeros((mol.nao_nr(), mol.nao_nr()))
    hcore = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    uwt_mat = get_veff_uwt(mol, dm)
    phi_global = np.mean([scalar_field(coord) for coord in mol.atom_coords()])
    em_term = -gwave**2 * phi_global**2 * np.eye(mol.nao_nr())  # Simplified
    cp_term = theta * phi_global**2 * np.eye(mol.nao_nr())
    hcore += uwt_mat + em_term + cp_term
    return hcore

# SCF Calculation
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: uwt_hamiltonian(mol)
mf.max_cycle = 500
mf.conv_tol = 1e-10
mf.conv_tol_grad = 1e-8
mf.level_shift = 0.4
mf0 = scf.RHF(mol)
mf0.kernel()
dm_init = mf0.make_rdm1()
mf.kernel(dm0=dm_init)
print("UWT SCF energy:", mf.e_tot)

# g-Factor Calculation
def zeeman_term(mol, B_field):
    """Calculates an approximated Zeeman splitting term."""
    r_ao = mol.intor('int1e_r', comp=3)  # Position integrals
    phi = np.mean([scalar_field(coord) for coord in mol.atom_coords()])
    LSO = (mu_B * g_SO_strength * phi * 1e1) * np.einsum('xij,x->ij', r_ao, B_field)  # Moderate scaling
    return LSO

def calculate_anomalous_magnetic_moment(mol):
    """Calculates the anomalous magnetic moment contribution."""
    phi = np.mean([scalar_field(coord) for coord in mol.atom_coords()])
    Lanom = -(mu_B / 2) * Delta_a * phi**2 * np.eye(mol.nao_nr())
    return Lanom

def get_zeeman(B, mol, dm):
    """Computes Zeeman energy with approximated term."""
    r_ao = mol.intor('int1e_r', comp=3)  # Use position integrals
    spin_ao = np.einsum('xij,x->ij', r_ao, B)
    E = np.einsum('ij,ji', spin_ao, dm)
    return E

B_field = np.array([0, 0, 1e-1])  # Kept at 1e-1 T
num_steps = 2

def evolve_g_factor(mol, B_field, num_steps):
    dm = mf.make_rdm1()
    mf.get_hcore = lambda *args: uwt_hamiltonian(mol, dm) + zeeman_term(mol, B_field) + calculate_anomalous_magnetic_moment(mol)
    for _ in range(num_steps):
        dm_new = mf.make_rdm1()
        mf.kernel(dm0=dm_new)
    spin_up_E = mf.e_tot
    B_field = np.array([0, 0, -1e-1])
    mf.get_hcore = lambda *args: uwt_hamiltonian(mol, dm) + zeeman_term(mol, B_field) + calculate_anomalous_magnetic_moment(mol)
    for _ in range(num_steps):
        dm_new = mf.make_rdm1()
        mf.kernel(dm0=dm_new)
    spin_down_E = mf.e_tot
    delta_E = abs(spin_up_E - spin_down_E)  # Ensure positive difference
    g_factor = 2 * delta_E / (mu_B * np.linalg.norm(B_field))
    return g_factor

g_factor = evolve_g_factor(mol, B_field, num_steps)
g_factor_experimental = 2.002319304361
percentage_difference = abs((g_factor - g_factor_experimental) / g_factor_experimental) * 100

print(f"Calculated Electron g-factor: {g_factor}")
print(f"Experimental Electron g-factor: {g_factor_experimental}")
print(f"Percentage Difference: {percentage_difference:.2f}%")
if percentage_difference <= 7.0:
    print(f"Model validated. Simulation produces g-factor with {percentage_difference:.2f}% deviation...")
elif percentage_difference > 7.0 and percentage_difference < 15.0:
    print(f"Model deviation found... {percentage_difference:.2f}% deviation...")
else:
    print(f"Significant deviation found... {percentage_difference:.2f}% deviation...")