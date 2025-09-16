<<<<<<< HEAD
from pyscf import gto, scf
import numpy as np

# UWT parameters
uwt_me = 0.510998 / 27.2114  # ~0.0188 a.u.
standard_me = 0.0188  # Standard electron mass
phi1_phi2 = 0.0511  # |Φ|^2 from UWT (GeV^2)
eta = 1e24  # η from UWT (m^6 kg^-4)
delta_eps_sc2 = 1.42e11  # J/m^3 from UWT

# Convert UWT units to atomic units
au_energy = 27.2114 * 1.602e-19  # Hartree to Joules
au_length = 5.29177e-11  # Bohr to meters
au_volume = au_length**3  # Bohr^3 to m^3
base_factor = delta_eps_sc2 / (au_energy / au_volume)

# Define H2 molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', unit='Angstrom')

# Standard Hartree-Fock
mf_std = scf.RHF(mol)
energy_std = mf_std.kernel()
print(f"Standard H2 Hartree-Fock energy: {energy_std} Hartree")

# Iterative adjustment
target_shift = 0.00018371  # ~5 meV in Hartree (adjust based on UWT)
multiplier_low = 1e-5
multiplier_high = 1e-1
tolerance = 1e-6

for _ in range(10):  # Limit iterations
    multiplier = (multiplier_low + multiplier_high) / 2
    mf_uwt = scf.RHF(mol)
    hcore = mf_uwt.get_hcore()
    mass_ratio = uwt_me / standard_me  # ~1
    hcore += np.eye(hcore.shape[0]) * (base_factor * multiplier)
    mf_uwt.get_hcore = lambda *args: hcore
    energy_uwt = mf_uwt.kernel()
    shift = abs(energy_std - energy_uwt)

    print(f"Multiplier: {multiplier}, Shift: {shift} Hartree, Energy: {energy_uwt} Hartree")
    if abs(shift - target_shift) < tolerance:
        break
    elif shift < target_shift:
        multiplier_low = multiplier
    else:
        multiplier_high = multiplier

=======
from pyscf import gto, scf
import numpy as np

# UWT parameters
uwt_me = 0.510998 / 27.2114  # ~0.0188 a.u.
standard_me = 0.0188  # Standard electron mass
phi1_phi2 = 0.0511  # |Φ|^2 from UWT (GeV^2)
eta = 1e24  # η from UWT (m^6 kg^-4)
delta_eps_sc2 = 1.42e11  # J/m^3 from UWT

# Convert UWT units to atomic units
au_energy = 27.2114 * 1.602e-19  # Hartree to Joules
au_length = 5.29177e-11  # Bohr to meters
au_volume = au_length**3  # Bohr^3 to m^3
base_factor = delta_eps_sc2 / (au_energy / au_volume)

# Define H2 molecule
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', unit='Angstrom')

# Standard Hartree-Fock
mf_std = scf.RHF(mol)
energy_std = mf_std.kernel()
print(f"Standard H2 Hartree-Fock energy: {energy_std} Hartree")

# Iterative adjustment
target_shift = 0.00018371  # ~5 meV in Hartree (adjust based on UWT)
multiplier_low = 1e-5
multiplier_high = 1e-1
tolerance = 1e-6

for _ in range(10):  # Limit iterations
    multiplier = (multiplier_low + multiplier_high) / 2
    mf_uwt = scf.RHF(mol)
    hcore = mf_uwt.get_hcore()
    mass_ratio = uwt_me / standard_me  # ~1
    hcore += np.eye(hcore.shape[0]) * (base_factor * multiplier)
    mf_uwt.get_hcore = lambda *args: hcore
    energy_uwt = mf_uwt.kernel()
    shift = abs(energy_std - energy_uwt)

    print(f"Multiplier: {multiplier}, Shift: {shift} Hartree, Energy: {energy_uwt} Hartree")
    if abs(shift - target_shift) < tolerance:
        break
    elif shift < target_shift:
        multiplier_low = multiplier
    else:
        multiplier_high = multiplier

>>>>>>> UWT-Analysis-2025/main
print(f"Final UWT-adjusted H2 energy: {energy_uwt} Hartree with multiplier {multiplier}")