"""Real-time MPS/TEBD study of an up-spin wavepacket in a Hubbard chain.

This is a small-system learning prototype, not a direct reproduction of a
current-constrained steady-state CISS calculation.

Model:
    H = H_Hubbard + H_NN-SOC

The next-nearest-neighbor hopping has been removed. The effective SOC is
spin-conserving and changes sign with the handedness parameter kappa.
Each physical site has basis |0>, |up>, |down>, |up down> (local dimension 4).

The code injects an up-spin wavepacket only, evolves it with TEBD, measures
spin-resolved particle current across a chosen bond, and integrates that
current in time to obtain the net excess particle number transported across
the bond. Because the chain is finite and closed, this is a transient net
transport measure, not a steady-state transmission probability.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import quimb.tensor as qtn


# ============================================================
# Model and numerical parameters
# ============================================================
L = 8
hopping_nn = 1.0
U = 2.0
mu = U / 2.0
chi_soc = 0.20

# Compare opposite effective handedness values while keeping the input spin up.
handedness_values = [+1, -1]

# TEBD settings
dt = 0.05
t_max = 12.0
max_bond = 64
cutoff = 1e-10

# Initial extra-electron wavepacket: centered toward the left and aimed right.
packet_center = 1.0
packet_width = 0.9
packet_momentum = np.pi / 2.0

# Measure the current across this bond (left site -> right site).
# For L=6, current_bond=3 means the bond between site 3 and site 4 (0-based).
current_bond = L - 3

# Report the integrated current at this early-time cutoff as a first-passage
# estimate. Adjust it after inspecting the current trace so it ends before the
# packet reflected by the right boundary returns across the measurement bond.
first_passage_end_time = 2.5

if L < 4:
    raise ValueError("Use L >= 4 so the measurement bond is away from the left edge.")
if not (0 <= current_bond < L - 1):
    raise ValueError("current_bond must label a bond between sites i and i+1.")


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
# Dense Hamiltonian helpers (intended only for small L)
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


def apply_one_site_tensor(state_tensor, op, site):
    """Apply a local operator to a rank-L state tensor at one site."""
    result = np.tensordot(op, state_tensor, axes=([1], [site]))
    return np.moveaxis(result, 0, site)


def create_fermionic_wavepacket(psi0, nsites):
    """Add an up-spin wavepacket to psi0 and normalize the resulting state.

    The fermionic creation operator at site i includes the Jordan-Wigner parity
    string over sites to its left. Only up-spin injection is used in this file.
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
        term = apply_one_site_tensor(term, c_dag_ops[0], site)  # up spin only
        packet_tensor = packet_tensor + amplitudes[site] * term

    packet_dense = packet_tensor.reshape(-1)
    norm = np.linalg.norm(packet_dense)
    if norm < 1e-12:
        raise ValueError("The injected up-spin wavepacket has almost zero norm.")
    packet_dense = packet_dense / norm
    return qtn.MatrixProductState.from_dense(
        packet_dense, dims=[4] * nsites, cutoff=1e-12
    )


def build_local_terms(handedness, nsites):
    """Return onsite terms and the nearest-neighbor Hubbard-SOC term.

    H_onsite = sum_i [U n_(i,up) n_(i,down) - mu n_i]

    H_NN = sum_(i,sigma) [a_sigma c^dag_(i,sigma)c_(i+1,sigma) + h.c.]
    with a_sigma = -t + i*kappa*chi_soc*sigma_sign.

    The SOC is diagonal in spin: it does not flip up and down spins.
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

    if not np.allclose(h2, h2.conj().T):
        raise ValueError("Nearest-neighbor term is not Hermitian.")
    return h1_terms, h2


def build_mpo(handedness, nsites):
    """Build an MPO from the dense small-system Hamiltonian."""
    h1_terms, h2 = build_local_terms(handedness, nsites)
    dim = 4**nsites
    h_dense = np.zeros((dim, dim), dtype=complex)

    for i in range(nsites):
        h_dense = h_dense + embed_one_site(h1_terms[i], i, nsites)
    for i in range(nsites - 1):
        h_dense = h_dense + embed_two_site(h2, i, nsites)

    if not np.allclose(h_dense, h_dense.conj().T, atol=1e-10):
        raise ValueError("Full Hamiltonian is not Hermitian.")

    H_mpo = qtn.MatrixProductOperator.from_dense(
        h_dense, dims=[4] * nsites, cutoff=1e-12
    )
    del h_dense
    return H_mpo, h1_terms, h2


def build_current_operators(handedness):
    """Build the NN particle-current operators, positive from i to i+1.

    For H_ij = a c_i^dag c_(i+1) + a* c_(i+1)^dag c_i, the current is
        J_(i->i+1) = i [a c_i^dag c_(i+1) - a* c_(i+1)^dag c_i].

    The local two-site operators include the physical-site Jordan-Wigner parity.
    Return [J_up, J_down].
    """
    current_ops = []
    for spin_index in range(2):
        spin_sign = +1.0 if spin_index == 0 else -1.0
        hop = np.kron(c_dag_ops[spin_index] @ P, c_ops[spin_index])
        coefficient = -hopping_nn + 1.0j * handedness * chi_soc * spin_sign
        current = 1.0j * (
            coefficient * hop - np.conj(coefficient) * hop.conj().T
        )
        if not np.allclose(current, current.conj().T, atol=1e-10):
            raise ValueError("Particle-current operator is not Hermitian.")
        current_ops.append(current)
    return current_ops


def expect_local(psi, op, site):
    return float(np.real(psi.local_expectation_canonical(op, site, normalized=True)))


def expect_bond(psi, op, bond):
    """Expectation value of a two-site operator on sites bond and bond+1."""
    return float(
        np.real(
            psi.local_expectation_canonical(
                op, (bond, bond + 1), normalized=True
            )
        )
    )


def measure_state(psi, nsites):
    nup_values = []
    ndown_values = []
    for site in range(nsites):
        nup_values.append(expect_local(psi, n_up, site))
        ndown_values.append(expect_local(psi, n_down, site))
    nup_values = np.asarray(nup_values)
    ndown_values = np.asarray(ndown_values)
    return (
        nup_values + ndown_values,
        0.5 * (nup_values - ndown_values),
        nup_values,
        ndown_values,
    )


def cumulative_trapezoid_explicit(values, times):
    """Cumulative trapezoidal time integral, with integral zero at t=0."""
    integral = np.zeros(len(times), dtype=float)
    for i in range(1, len(times)):
        integral[i] = integral[i - 1] + 0.5 * (
            values[i] + values[i - 1]
        ) * (times[i] - times[i - 1])
    return integral


# ============================================================
# Symmetric second-order TEBD with on-site and NN gates only
# ============================================================
def apply_gate(psi, gate, where, one_site=False):
    if one_site:
        psi.gate_(gate, where=where, contract=True)
    else:
        psi.gate_(
            gate,
            where=where,
            contract="split",
            max_bond=max_bond,
            cutoff=cutoff,
        )


def make_trotter_gates(h1_terms, h2):
    gate1_half = [expm(-1.0j * (dt / 2.0) * h) for h in h1_terms]
    gate2_half = expm(-1.0j * (dt / 2.0) * h2)
    gate2_full = expm(-1.0j * dt * h2)
    return gate1_half, gate2_half, gate2_full


def tebd_step(psi, nsites, gates):
    gate1_half, gate2_half, gate2_full = gates

    # Symmetric splitting: onsite/2, even bonds/2, odd bonds, even bonds/2, onsite/2.
    for site in range(nsites):
        apply_gate(psi, gate1_half[site], site, one_site=True)

    for start in range(0, nsites - 1, 2):
        apply_gate(psi, gate2_half, (start, start + 1))

    for start in range(1, nsites - 1, 2):
        apply_gate(psi, gate2_full, (start, start + 1))

    for start in range(0, nsites - 1, 2):
        apply_gate(psi, gate2_half, (start, start + 1))

    for site in range(nsites):
        apply_gate(psi, gate1_half[site], site, one_site=True)

    # Truncation can slightly change the norm; restore a normalized state.
    psi.normalize()


def time_evolve(
    psi_initial,
    nsites,
    h1_terms,
    h2,
    current_ops,
    current_ground,
    bond,
    label,
):
    psi = psi_initial.copy()
    gates = make_trotter_gates(h1_terms, h2)
    n_steps = int(round(t_max / dt))
    times = np.arange(n_steps + 1, dtype=float) * dt

    n_history = []
    sz_history = []
    nup_history = []
    ndown_history = []
    current_history = []

    for step in range(n_steps + 1):
        n_values, sz_values, nup_values, ndown_values = measure_state(psi, nsites)
        j_up = expect_bond(psi, current_ops[0], bond)
        j_down = expect_bond(psi, current_ops[1], bond)

        n_history.append(n_values)
        sz_history.append(sz_values)
        nup_history.append(nup_values)
        ndown_history.append(ndown_values)
        current_history.append([j_up, j_down])

        if step < n_steps:
            tebd_step(psi, nsites, gates)
        if step % 40 == 0:
            print(f"  {label}: t = {times[step]:.2f} / {t_max:.2f}")

    current_history = np.asarray(current_history)
    # Subtract the stationary ground-state current to isolate the response
    # associated with adding the up-spin wavepacket.
    excess_current = current_history - current_ground[None, :]

    return {
        "times": times,
        "n": np.asarray(n_history),
        "sz": np.asarray(sz_history),
        "nup": np.asarray(nup_history),
        "ndown": np.asarray(ndown_history),
        "current": excess_current,
    }


def run_handedness(handedness):
    print("\n" + "=" * 60)
    print(f"Running handedness kappa = {handedness:+d}; input spin = up")
    H_mpo, h1_terms, h2 = build_mpo(handedness, L)
    dmrg = qtn.DMRG2(H_mpo, bond_dims=[16, 32, 64], cutoffs=1e-10)
    dmrg.solve(tol=1e-8, verbosity=1)
    psi_ground = dmrg.state.copy()
    print("Ground-state energy =", dmrg.energy)

    n_ground, sz_ground, nup_ground, ndown_ground = measure_state(psi_ground, L)
    print("Ground-state electron number =", np.sum(n_ground))

    current_ops = build_current_operators(handedness)
    current_ground = np.zeros(2, dtype=float)
    for spin_index in range(2):
        current_ground[spin_index] = expect_bond(
            psi_ground, current_ops[spin_index], current_bond
        )

    # Hold the injected spin fixed: up-spin wavepacket for both kappa values.
    psi_up = create_fermionic_wavepacket(psi_ground, L)
    data = time_evolve(
        psi_up,
        L,
        h1_terms,
        h2,
        current_ops,
        current_ground,
        current_bond,
        f"kappa={handedness:+d}, up-spin input",
    )

    delta_n = data["n"] - n_ground[None, :]
    delta_sz = data["sz"] - sz_ground[None, :]
    delta_nup = data["nup"] - nup_ground[None, :]
    delta_ndown = data["ndown"] - ndown_ground[None, :]
    excess_current = data["current"]
    transported = np.zeros_like(excess_current)
    for spin_index in range(2):
        transported[:, spin_index] = cumulative_trapezoid_explicit(
            excess_current[:, spin_index], data["times"]
        )

    # Continuity check: with NN hopping only, the integrated current across
    # the cut should equal the change in excess particle number to its right.
    right_up = np.sum(delta_nup[:, current_bond + 1 :], axis=1)
    right_down = np.sum(delta_ndown[:, current_bond + 1 :], axis=1)
    right_region_change = np.column_stack((
        right_up - right_up[0],
        right_down - right_down[0],
    ))
    continuity_error = np.max(np.abs(transported - right_region_change), axis=0)
    print(
        "Max continuity-check deviation [up, down] = "
        f"[{continuity_error[0]:.3e}, {continuity_error[1]:.3e}]"
    )

    cutoff_index = int(np.argmin(np.abs(data["times"] - first_passage_end_time)))
    print(
        f"Net excess particle flow across bond {current_bond}->{current_bond + 1} "
        f"by t={data['times'][cutoff_index]:.2f}: "
        f"up={transported[cutoff_index, 0]:+.6f}, "
        f"down={transported[cutoff_index, 1]:+.6f}, "
        f"total={np.sum(transported[cutoff_index, :]):+.6f}"
    )

    return {
        "handedness": handedness,
        "times": data["times"],
        "delta_n": delta_n,
        "delta_sz": delta_sz,
        "current": excess_current,
        "transported": transported,
        "continuity_error": continuity_error,
        "cutoff_index": cutoff_index,
        "ground_energy": dmrg.energy,
    }


# ============================================================
# Run the two handedness cases and plot results
# ============================================================
results = []
for handedness in handedness_values:
    results.append(run_handedness(handedness))

extent = [0.0, t_max, -0.5, L - 0.5]

global_charge_limit = max(
    1e-12,
    max(float(np.max(np.abs(result["delta_n"]))) for result in results),
)
global_spin_limit = max(
    1e-12,
    max(float(np.max(np.abs(result["delta_sz"]))) for result in results),
)
fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for row, result in enumerate(results):
    im0 = ax[row, 0].imshow(
        result["delta_n"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-global_charge_limit,
        vmax=global_charge_limit,
        cmap="coolwarm",
        interpolation="nearest",
    )
    ax[row, 0].set_title(f"Excess charge, kappa={result['handedness']:+d}")
    ax[row, 0].set_xlabel("Time")
    ax[row, 0].set_ylabel("Site")
    fig.colorbar(im0, ax=ax[row, 0], label=r"$\Delta n_i(t)$")

    im1 = ax[row, 1].imshow(
        result["delta_sz"].T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-global_spin_limit,
        vmax=global_spin_limit,
        cmap="coolwarm",
        interpolation="nearest",
    )
    ax[row, 1].set_title(f"Spin response, kappa={result['handedness']:+d}")
    ax[row, 1].set_xlabel("Time")
    ax[row, 1].set_ylabel("Site")
    fig.colorbar(im1, ax=ax[row, 1], label=r"$\Delta S_i^z(t)$")

plt.show()

# Plot instantaneous excess current. Positive values mean net motion rightward.
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
for result in results:
    ax.plot(
        result["times"],
        result["current"][:, 0],
        label=f"kappa={result['handedness']:+d}, up",
    )
    ax.plot(
        result["times"],
        result["current"][:, 1],
        linestyle="--",
        label=f"kappa={result['handedness']:+d}, down response",
    )
ax.axhline(0.0, linewidth=0.8)
ax.axvline(first_passage_end_time, linestyle=":", linewidth=1.0, label="first-pass cutoff")
ax.set_xlabel("Time")
ax.set_ylabel(r"Excess particle current $\Delta J_\sigma(t)$")
ax.set_title(
    f"Spin-resolved excess current across bond {current_bond} to {current_bond + 1}"
)
ax.legend()
plt.show()

# Plot the time-integrated, signed current (net transported excess particles).
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
for result in results:
    ax.plot(
        result["times"],
        result["transported"][:, 0],
        label=f"kappa={result['handedness']:+d}, up",
    )
    ax.plot(
        result["times"],
        result["transported"][:, 1],
        linestyle="--",
        label=f"kappa={result['handedness']:+d}, down response",
    )
ax.axhline(0.0, linewidth=0.8)
ax.axvline(first_passage_end_time, linestyle=":", linewidth=1.0, label="first-pass cutoff")
ax.set_xlabel("Time")
ax.set_ylabel(r"$Q_\sigma(t)=\int_0^t \Delta J_\sigma(t')\,dt'$ ")
ax.set_title("Cumulative net excess particle flow across the measurement bond")
ax.legend()
plt.show()

print("\nInterpretation and caveats:")
print("- Only an up-spin wavepacket is injected for both handedness values.")
print("- Current is measured across one bond; positive means left-to-right flow.")
print("- The integrated current is signed net transport: backflow decreases it.")
print("- The value at the first-pass cutoff is an early-time transport estimate, not a steady-state transmission probability.")
print("- The dashed down-spin current is a response of the down-spin background, not a down-spin injected-electron transmission channel.")
print("- With only uniform spin-conserving NN SOC on an open 1D chain, the hopping phase can be gauged away; a kappa difference alone is not proof of CISS.")
print("- This small-L dense-MPO prototype should be validated against ED before physical conclusions are drawn.")
