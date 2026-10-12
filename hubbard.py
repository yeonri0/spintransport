# Hubbard模型、フェルミオン系、スピン輸送

import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn


# ============================================================
# 1. Parameters
# ============================================================

L = 6
t = 1.0
U = 2.0
mu = U / 2

dt = 0.05
t_max = 30.0


# ============================================================
# 2. Local operators
#
# Basis:
# |0>, |up>, |down>, |up down>
# ============================================================

c_up = np.zeros((4, 4), dtype=complex)
c_up[0, 1] = 1.0
c_up[2, 3] = 1.0

c_down = np.zeros((4, 4), dtype=complex)
c_down[0, 2] = 1.0
c_down[1, 3] = -1.0

n_up = c_up.conj().T @ c_up
n_down = c_down.conj().T @ c_down

n_total = n_up + n_down
Sz = 0.5 * (n_up - n_down)

# Fermion parity: (-1)^(n_up + n_down)
P = np.diag([1.0, -1.0, -1.0, 1.0])

I4 = np.eye(4, dtype=complex)


# ============================================================
# 3. Local Hubbard Hamiltonian terms
# ============================================================

# On-site interaction and chemical-potential terms
h_u_site = U * (n_up @ n_down) - mu * n_total

# Nearest-neighbor hopping term
# The parity operator P accounts for fermionic signs.
h_t_bond = np.zeros((16, 16), dtype=complex)

for c in [c_up, c_down]:
    hop = np.kron(c.conj().T @ P, c)
    h_t_bond = h_t_bond - t * (hop + hop.conj().T)


# ============================================================
# 4. Construct the full-system H_t and H_u
#
# This dense-matrix construction is for small L only.
# ============================================================

def embed_one_site(op, site, L):
    factors = []

    for i in range(L):
        if i == site:
            factors.append(op)
        else:
            factors.append(I4)

    result = factors[0]

    for i in range(1, len(factors)):
        result = np.kron(result, factors[i])

    return result


def embed_two_site(op, left_site, L):
    factors = []
    i = 0

    while i < L:
        if i == left_site:
            factors.append(op)
            i = i + 2
        else:
            factors.append(I4)
            i = i + 1

    result = factors[0]

    for i in range(1, len(factors)):
        result = np.kron(result, factors[i])

    return result


dim = 4**L

# Full hopping Hamiltonian
H_t_terms = []

for i in range(L - 1):
    H_t_terms.append(embed_two_site(h_t_bond, i, L))

H_t = sum(H_t_terms)

# Full on-site Hamiltonian
H_u_terms = []

for i in range(L):
    H_u_terms.append(embed_one_site(h_u_site, i, L))

H_u = sum(H_u_terms)


# ============================================================
# 5. Hubbard Hamiltonian H as an MPO
# ============================================================

H_dense = H_t + H_u

H = qtn.MatrixProductOperator.from_dense(
    H_dense,
    dims=[4] * L,
    cutoff=1e-12,
)


# ============================================================
# 6. TEBD Hamiltonian
# ============================================================

H_tebd = qtn.LocalHam1D(
    L=L,
    H2=h_t_bond,
    H1=h_u_site,
    cyclic=False,
)


# ============================================================
# 7. DMRG ground state
# ============================================================

dmrg = qtn.DMRG2(
    H,
    bond_dims=[16, 32, 64],
    cutoffs=1e-10,
)

dmrg.solve(tol=1e-8, verbosity=1)
psi0 = dmrg.state.copy()

print("Ground-state energy =", dmrg.energy)


# Measure ground-state density and spin density
def expect_local(psi, op, site):
    return psi.local_expectation_canonical(
        op,
        site,
        normalized=True,
    )

n_ground = []
sz_ground = []

for i in range(L):
    n_ground.append(
        float(np.real(expect_local(psi0, n_total, i)))
    )

    sz_ground.append(
        float(np.real(expect_local(psi0, Sz, i)))
    )

N_ground = sum(n_ground)
Sz_ground = sum(sz_ground)

print("Ground-state electron number =", N_ground)
print("Ground-state total Sz =", Sz_ground)


# ============================================================
# 8. Add one up-spin electron at the left end
# ============================================================

psi_exc = psi0.copy()

# At site 0, no Jordan-Wigner string is needed before the site.
psi_exc.gate_(
    c_up.conj().T,
    where=0,
    contract=True,
)

exc_norm = psi_exc.norm()

if exc_norm < 1e-12:
    raise ValueError(
        "Electron creation gave a zero state. Check the initial state."
    )

# Normalize the excited state
psi_exc.normalize()

N_exc = sum(
    float(np.real(expect_local(psi_exc, n_total, i)))
    for i in range(L)
)

print("Electron number after injection =", N_exc)


# ============================================================
# 9. TEBD time evolution
# ============================================================

times = np.arange(0.0, t_max + 0.5 * dt, dt)

tebd = qtn.TEBD(
    psi_exc,
    H_tebd,
    dt=dt,
    split_opts={
        "max_bond": 64,
        "cutoff": 1e-10,
    },
    progbar=True,
)


# ============================================================
# 10. Measure local electron density and spin density
# ============================================================

n_t = []
sz_t = []

for psi_t in tebd.at_times(times, order=2):

    n_site = []
    sz_site = []

    for i in range(L):
        n_site.append(
            float(np.real(expect_local(psi_t, n_total, i)))
        )
        sz_site.append(
            float(np.real(expect_local(psi_t, Sz, i)))
        )

    n_t.append(n_site)
    sz_t.append(sz_site)


n_t = np.array(n_t)
sz_t = np.array(sz_t)

# Subtract the ground-state background
delta_n_t = n_t - np.array(n_ground)[None, :]
delta_sz_t = sz_t - np.array(sz_ground)[None, :]


# ============================================================
# 11. Check particle-number and spin conservation
# ============================================================

N_t = np.sum(n_t, axis=1)
Sz_t = np.sum(sz_t, axis=1)

print("Total N at initial time =", N_t[0])
print("Total N at final time   =", N_t[-1])
print("Maximum variation in N  =", np.max(N_t) - np.min(N_t))

print("Total Sz at initial time =", Sz_t[0])
print("Total Sz at final time   =", Sz_t[-1])


# ============================================================
# 12. Plot the excess charge and spin propagation
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

extent = [times[0], times[-1], -0.5, L - 0.5]

charge_limit = max(np.max(np.abs(delta_n_t)), 1e-12)

im0 = ax[0].imshow(
    delta_n_t.T,
    origin="lower",
    aspect="auto",
    extent=extent,
    vmin=-charge_limit,
    vmax=charge_limit,
    cmap="coolwarm",
)

ax[0].set_xlabel("Time")
ax[0].set_ylabel("Site")
ax[0].set_title("Excess electron density")
fig.colorbar(im0, ax=ax[0], label=r"$\Delta n_i(t)$")

spin_limit = max(np.max(np.abs(delta_sz_t)), 1e-12)

im1 = ax[1].imshow(
    delta_sz_t.T,
    origin="lower",
    aspect="auto",
    extent=extent,
    vmin=-spin_limit,
    vmax=spin_limit,
    cmap="coolwarm",
)

ax[1].set_xlabel("Time")
ax[1].set_ylabel("Site")
ax[1].set_title("Change in spin density")
fig.colorbar(im1, ax=ax[1], label=r"$\Delta S_i^z(t)$")

plt.savefig("hubbard_L6_t30.png", dpi=300, bbox_inches="tight")
plt.show()