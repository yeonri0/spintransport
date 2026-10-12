"""Real-time MPS/TEBD study of a minimal chiral Hubbard chain.

This is an effective time-dependent model inspired by the twisted-p-orbital
Hubbard chain in Savi et al., J. Phys. Chem. Lett. 16, 9107 (2025).
It is NOT a direct reproduction of their current-constrained steady-state method.

Each physical site has basis |0>, |up>, |down>, |up down> (local dimension 4).
The effective chiral SOC is spin-conserving and changes sign under handedness
reversal. A real next-nearest-neighbor hopping t_nnn creates loops with the
nearest-neighbor SOC bonds. The NNN fermionic hopping is implemented with its
Jordan-Wigner parity string as a three-site gate.

For this first version, the small-system MPO is assembled from a dense
Hamiltonian (practical for roughly L <= 6). TEBD itself applies local MPS gates
using a symmetric second-order Suzuki-Trotter decomposition, including 3-site
NNN gates. This is a convenient learning prototype, not yet an efficient
large-L implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import quimb.tensor as qtn


# ============================================================
# Model and numerical parameters
# ============================================================
L = 6
hopping_nn = 1.0
t_nnn = 0.15                # next-nearest-neighbor hopping; creates loops
U = 2.0
mu = U / 2.0
chi_soc = 0.20               # effective SOC scale (same energy unit as t and U)

# Chirality kappa = +1 or -1 will be compared.  The SOC term changes sign
# under kappa -> -kappa.  It is an effective low-energy representation of
# the handedness dependence, not a full orbital-integral reconstruction.
handedness_values = [+1, -1]

dt = 0.05
t_max = 20.0
max_bond = 64
cutoff = 1e-10

# Initial extra-electron wavepacket, centered toward the left and moving right.
packet_center = 1.0
packet_width = 0.9
packet_momentum = np.pi / 2.0


# ============================================================
# Local physical-site basis: |0>, |up>, |down>, |up down>
# ============================================================
c_up = np.zeros((4, 4), dtype=complex)
c_up[0, 1] = 1.0
c_up[2, 3] = 1.0

c_down = np.zeros((4, 4), dtype=complex)
c_down[0, 2] = 1.0
c_down[1, 3] = -1.0

c_ops = [c_up, c_down]
c_dag_ops = [c_up.conj().T, c_down.conj().T]

n_up = c_up.conj().T @ c_up
n_down = c_down.conj().T @ c_down
n_total = n_up + n_down
Sz = 0.5 * (n_up - n_down)

# Fermion parity P = (-1)^(n_up+n_down) on one physical site.
P = np.diag([1.0, -1.0, -1.0, 1.0]).astype(complex)
I4 = np.eye(4, dtype=complex)


# ============================================================
# Small helpers for operators and dense Hamiltonian construction
# ============================================================
def embed_one_site(op, site, nsites):
    factors = []
    for i in range(nsites):
        if i == site:
            factors.append(op)
        else:
            factors.append(I4)
    result = factors[0]
    for i in range(1, len(factors)):
        result = np.kron(result, factors[i])
    return result


def embed_two_site(op, start, nsites):
    factors = []
    i = 0
    while i < nsites:
        if i == start:
            factors.append(op)
            i += 2
        else:
            factors.append(I4)
            i += 1
    result = factors[0]
    for i in range(1, len(factors)):
        result = np.kron(result, factors[i])
    return result


def embed_three_site(op, start, nsites):
    factors = []
    i = 0
    while i < nsites:
        if i == start:
            factors.append(op)
            i += 3
        else:
            factors.append(I4)
            i += 1
    result = factors[0]
    for i in range(1, len(factors)):
        result = np.kron(result, factors[i])
    return result


def apply_one_site_tensor(state_tensor, op, site):
    """Apply a local operator to a rank-L state tensor at one site."""
    result = np.tensordot(op, state_tensor, axes=([1], [site]))
    return np.moveaxis(result, 0, site)


def create_fermionic_wavepacket(psi0, spin_index, nsites):
    """Create a right-moving extra-electron wavepacket on top of the ground state.

    The physical fermion creation operator on site i includes a Jordan-Wigner
    parity string over all sites to its left. spin_index=0 is up; spin_index=1
    is down. The packet is normalized after applying the creation operator.
    """
    ground_tensor = np.asarray(psi0.to_dense()).reshape([4] * nsites)
    x = np.arange(nsites, dtype=float)
    amplitudes = np.exp(-0.5 * ((x - packet_center) / packet_width) ** 2)
    amplitudes = amplitudes * np.exp(1.0j * packet_momentum * x)
    amplitudes = amplitudes / np.sqrt(np.sum(np.abs(amplitudes) ** 2))

    packet_tensor = np.zeros_like(ground_tensor, dtype=complex)
    for site in range(nsites):
        # C_i^dagger = P_0 ... P_(i-1) c_i^dagger in the JW representation.
        term = ground_tensor.copy()
        for j in range(site):
            term = apply_one_site_tensor(term, P, j)
        term = apply_one_site_tensor(term, c_dag_ops[spin_index], site)
        packet_tensor = packet_tensor + amplitudes[site] * term

    packet_dense = packet_tensor.reshape(-1)
    norm = np.linalg.norm(packet_dense)
    if norm < 1e-12:
        raise ValueError("The injected wavepacket has almost zero norm.")
    packet_dense = packet_dense / norm
    psi_packet = qtn.MatrixProductState.from_dense(
        packet_dense, dims=[4] * nsites, cutoff=1e-12
    )
    return psi_packet


def build_local_terms(handedness, nsites):
    """Return onsite, nearest-neighbor, and 3-site NNN terms.

    H_nn = sum_{i,sigma} [(-t + i*kappa*chi*s_sigma) c^dag_{i,sigma} c_{i+1,sigma} + h.c.]

    H_nnn = -t_nnn sum_{i,sigma} [c^dag_{i,sigma} c_{i+2,sigma} + h.c.]

    For a physical-site basis, c^dag_i c_(i+2) contains the middle-site parity
    P_(i+1), hence the NNN term is a three-site operator.
    """
    h1_terms = []
    for i in range(nsites):
        h1 = U * (n_up @ n_down) - mu * n_total
        h1_terms.append(h1.astype(complex))

    h2 = np.zeros((16, 16), dtype=complex)
    for spin_index in range(2):
        spin_sign = +1.0 if spin_index == 0 else -1.0
        hop = np.kron(c_dag_ops[spin_index] @ P, c_ops[spin_index])
        coefficient = -hopping_nn + 1.0j * handedness * chi_soc * spin_sign
        h2 = h2 + coefficient * hop + np.conj(coefficient) * hop.conj().T

    h3 = np.zeros((64, 64), dtype=complex)
    for spin_index in range(2):
        hop = np.kron(
            c_dag_ops[spin_index] @ P,
            np.kron(P, c_ops[spin_index]),
        )
        h3 = h3 - t_nnn * hop - t_nnn * hop.conj().T

    if not np.allclose(h2, h2.conj().T):
        raise ValueError("Nearest-neighbor term is not Hermitian.")
    if not np.allclose(h3, h3.conj().T):
        raise ValueError("Next-nearest-neighbor term is not Hermitian.")
    return h1_terms, h2, h3


def build_mpo(handedness, nsites):
    """Build an MPO from the small-system dense Hamiltonian."""
    h1_terms, h2, h3 = build_local_terms(handedness, nsites)
    dim = 4**nsites
    h_dense = np.zeros((dim, dim), dtype=complex)

    for i in range(nsites):
        h_dense = h_dense + embed_one_site(h1_terms[i], i, nsites)
    for i in range(nsites - 1):
        h_dense = h_dense + embed_two_site(h2, i, nsites)
    for i in range(nsites - 2):
        h_dense = h_dense + embed_three_site(h3, i, nsites)

    if not np.allclose(h_dense, h_dense.conj().T, atol=1e-10):
        raise ValueError("Full Hamiltonian is not Hermitian.")

    H_mpo = qtn.MatrixProductOperator.from_dense(
        h_dense, dims=[4] * nsites, cutoff=1e-12
    )
    del h_dense
    return H_mpo, h1_terms, h2, h3


def expect_local(psi, op, site):
    return float(np.real(psi.local_expectation_canonical(op, site, normalized=True)))


def measure_state(psi, nsites):
    nup_values = []
    ndown_values = []
    for site in range(nsites):
        nup_values.append(expect_local(psi, n_up, site))
        ndown_values.append(expect_local(psi, n_down, site))
    nup_values = np.asarray(nup_values)
    ndown_values = np.asarray(ndown_values)
    return nup_values + ndown_values, 0.5 * (nup_values - ndown_values), nup_values, ndown_values


# ============================================================
# Symmetric second-order TEBD using 1-, 2-, and 3-site gates
# ============================================================
def apply_gate(psi, gate, where, one_site=False):
    if one_site:
        psi.gate_(gate, where=where, contract=True)
    elif len(where) == 2:
        psi.gate_(
            gate,
            where=where,
            contract="split",
            max_bond=max_bond,
            cutoff=cutoff,
        )
    else:
        # Quimb's nonlocal-gate path decomposes a multi-site gate as an MPO
        # and compresses it back to an MPS, keeping the physical site order.
        psi.gate_nonlocal_(
            gate,
            where=where,
            dims=[4] * len(where),
            method="direct",
            max_bond=max_bond,
            cutoff=cutoff,
        )


def make_trotter_gates(h1_terms, h2, h3):
    gate1_half = [expm(-1.0j * (dt / 2.0) * h) for h in h1_terms]
    gate2_half = expm(-1.0j * (dt / 2.0) * h2)
    gate3_half = expm(-1.0j * (dt / 2.0) * h3)
    gate3_full = expm(-1.0j * dt * h3)
    return gate1_half, gate2_half, gate3_half, gate3_full


def tebd_step(psi, nsites, gates):
    gate1_half, gate2_half, gate3_half, gate3_full = gates

    # Symmetric composition: onsite, NN-even, NN-odd, NNN mod 0, mod 1,
    # NNN mod 2 (full step), then the reverse sequence.
    for site in range(nsites):
        apply_gate(psi, gate1_half[site], site, one_site=True)

    for start in range(0, nsites - 1, 2):
        apply_gate(psi, gate2_half, (start, start + 1))
    for start in range(1, nsites - 1, 2):
        apply_gate(psi, gate2_half, (start, start + 1))

    for residue in [0, 1]:
        for start in range(residue, nsites - 2, 3):
            apply_gate(psi, gate3_half, (start, start + 1, start + 2))
    for start in range(2, nsites - 2, 3):
        apply_gate(psi, gate3_full, (start, start + 1, start + 2))

    for residue in [1, 0]:
        for start in range(residue, nsites - 2, 3):
            apply_gate(psi, gate3_half, (start, start + 1, start + 2))

    for start in range(1, nsites - 1, 2):
        apply_gate(psi, gate2_half, (start, start + 1))
    for start in range(0, nsites - 1, 2):
        apply_gate(psi, gate2_half, (start, start + 1))

    for site in range(nsites):
        apply_gate(psi, gate1_half[site], site, one_site=True)

    # Truncation can slightly change the norm; restore a normalized state.
    psi.normalize()


def time_evolve(psi_initial, psi_ground, nsites, h1_terms, h2, h3, label):
    psi = psi_initial.copy()
    gates = make_trotter_gates(h1_terms, h2, h3)
    n_steps = int(round(t_max / dt))
    times = np.arange(n_steps + 1, dtype=float) * dt

    n_history = []
    sz_history = []
    nup_history = []
    ndown_history = []

    for step in range(n_steps + 1):
        n_values, sz_values, nup_values, ndown_values = measure_state(psi, nsites)
        n_history.append(n_values)
        sz_history.append(sz_values)
        nup_history.append(nup_values)
        ndown_history.append(ndown_values)
        if step < n_steps:
            tebd_step(psi, nsites, gates)
        if step % 40 == 0:
            print(f"  {label}: t = {times[step]:.2f} / {t_max:.2f}")

    return (
        times,
        np.asarray(n_history),
        np.asarray(sz_history),
        np.asarray(nup_history),
        np.asarray(ndown_history),
    )


def run_handedness(handedness):
    print("\n" + "=" * 60)
    print(f"Running handedness kappa = {handedness:+d}")
    H_mpo, h1_terms, h2, h3 = build_mpo(handedness, L)
    dmrg = qtn.DMRG2(H_mpo, bond_dims=[16, 32, 64], cutoffs=1e-10)
    dmrg.solve(tol=1e-8, verbosity=1)
    psi_ground = dmrg.state.copy()
    print("Ground-state energy =", dmrg.energy)

    n_ground, sz_ground, nup_ground, ndown_ground = measure_state(psi_ground, L)
    print("Ground-state electron number =", np.sum(n_ground))

    # Keep the injected spin fixed: compare an up-spin wavepacket for both
    # handedness values, kappa = +1 and -1.
    psi_up = create_fermionic_wavepacket(psi_ground, 0, L)
    up_data = time_evolve(
        psi_up, psi_ground, L, h1_terms, h2, h3,
        f"kappa={handedness:+d}, spin-up input",
    )

    times = up_data[0]
    # Both spin-resolved densities below come from the SAME up-spin-injection run.
    delta_n = up_data[1] - n_ground[None, :]
    delta_sz = up_data[2] - sz_ground[None, :]

    out_site = L - 1
    delta_up_at_out = up_data[3][:, out_site] - nup_ground[out_site]
    delta_down_at_out = up_data[4][:, out_site] - ndown_ground[out_site]
    denominator = delta_up_at_out + delta_down_at_out
    p_edge = np.full_like(denominator, np.nan, dtype=float)
    threshold = max(1e-3, 0.05 * float(np.max(np.abs(denominator))))
    # Only display the ratio when the excess total density at the edge is
    # positive and large enough to avoid a near-zero denominator.
    valid = denominator > threshold
    p_edge[valid] = (
        delta_up_at_out[valid] - delta_down_at_out[valid]
    ) / denominator[valid]

    return {
        "handedness": handedness,
        "times": times,
        "delta_n": delta_n,
        "delta_sz": delta_sz,
        "delta_up_out": delta_up_at_out,
        "delta_down_out": delta_down_at_out,
        "p_edge": p_edge,
        "ground_energy": dmrg.energy,
    }


# ============================================================
# Main calculation and plots
# ============================================================
results = []
for handedness in handedness_values:
    results.append(run_handedness(handedness))

extent = [0.0, t_max, -0.5, L - 0.5]

global_charge_limit = max(1e-12, max(float(np.max(np.abs(result["delta_n"]))) for result in results))
global_spin_limit = max(1e-12, max(float(np.max(np.abs(result["delta_sz"]))) for result in results))
fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for row, result in enumerate(results):
    charge_limit = global_charge_limit
    spin_limit = global_spin_limit
    im0 = ax[row, 0].imshow(
        result["delta_n"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-charge_limit,
        vmax=charge_limit,
        cmap="coolwarm",
        interpolation="nearest",
    )
    ax[row, 0].set_title(f"Excess charge, kappa={result['handedness']:+d}")
    ax[row, 0].set_xlabel("Time")
    ax[row, 0].set_ylabel("Site")
    fig.colorbar(im0, ax=ax[row, 0], label=r"$\Delta n_i(t)$")

    spin_limit = global_spin_limit
    im1 = ax[row, 1].imshow(
        result["delta_sz"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-spin_limit,
        vmax=spin_limit,
        cmap="coolwarm",
        interpolation="nearest",
    )
    ax[row, 1].set_title(f"Spin response, kappa={result['handedness']:+d}")
    ax[row, 1].set_xlabel("Time")
    ax[row, 1].set_ylabel("Site")
    fig.colorbar(im1, ax=ax[row, 1], label=r"$\Delta S_i^z(t)$")

plt.savefig("hubbard_chiral_L6_t20.png", dpi=300, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
for result in results:
    ax.plot(
        result["times"],
        result["p_edge"],
        label=f"kappa = {result['handedness']:+d}",
    )
ax.axhline(0.0, linewidth=0.8)
ax.set_xlabel("Time")
ax.set_ylabel(r"$P_{\rm R}(t)$ (excess-density polarization)")
ax.set_title("Right-edge spin polarization for an up-spin injected wavepacket")
ax.legend()
plt.savefig("hubbard_chiral_polarization_L6_t20.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nInterpretation:")
print("- The first two panels show charge and spin-density response to an up-spin injected packet.")
print("- P_R uses excess spin-up and spin-down densities from the SAME up-spin-injection run.")
print("- P_R is a transient right-edge excess-density polarization, not a steady-state transmission polarization.")
print("- Reverse handedness should reverse the chiral SOC contribution; finite-size and truncation effects remain.")
print("- This is an effective TEBD extension, not a direct reproduction of the paper's current-constrained method.")
