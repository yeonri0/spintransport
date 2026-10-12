import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn

# 図	表している量
# 左上	電子密度の変化 \(\Delta n_i(t)\)
# 右上	スピン密度の変化 \(\Delta S_i^z(t)\)
# 左下	各サイトの局所スピン偏極率
# 右下	電子密度とスピン密度の変化の比

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
L = 6
t = 1.0
U = 2.0
mu = U / 2
lam_soc = 0.2       # SOC strength; use the same energy units as t and U

dt = 0.05
t_max = 20.0

# ------------------------------------------------------------
# Local basis: |0>, |up>, |down>, |up down>
# ------------------------------------------------------------
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

# Fermion parity for a single physical site.
P = np.diag([1.0, -1.0, -1.0, 1.0]).astype(complex)
I4 = np.eye(4, dtype=complex)
c_ops = [c_up, c_down]

# On-site Hubbard interaction and chemical-potential term.
h_u_site = U * (n_up @ n_down) - mu * n_total

# ------------------------------------------------------------
# Nearest-neighbour hopping including a 1D SOC term
#
# H_hop = -sum_i [ c_i^dagger (t I + i lambda_SOC sigma_y) c_(i+1)
#                  + h.c. ]
#
# sigma_y = [[0, -i], [i, 0]], so i*lambda_SOC*sigma_y is a
# real spin-flip hopping matrix with opposite signs for up/down.
# ------------------------------------------------------------
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0j]], dtype=complex)
T_hop = t * np.eye(2, dtype=complex) + 1.0j * lam_soc * sigma_y

h_t_bond = np.zeros((16, 16), dtype=complex)
for a in range(2):
    for b in range(2):
        # c_(i,a)^dagger c_(i+1,b), including the site-parity factor.
        hop = np.kron(c_ops[a].conj().T @ P, c_ops[b])
        h_t_bond = h_t_bond - T_hop[a, b] * hop
        h_t_bond = h_t_bond - np.conj(T_hop[a, b]) * hop.conj().T

if not np.allclose(h_t_bond, h_t_bond.conj().T):
    raise ValueError("The two-site hopping Hamiltonian is not Hermitian.")

# ------------------------------------------------------------
# Embed local operators into the full 4^L-dimensional space.
# This dense construction is practical only for small L.
# ------------------------------------------------------------
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


# Build H as a dense matrix, adding one local term at a time to avoid
# keeping many full 4^L by 4^L matrices in memory simultaneously.
dim = 4**L
H_dense = np.zeros((dim, dim), dtype=complex)
for i in range(L - 1):
    H_dense += embed_two_site(h_t_bond, i, L)
for i in range(L):
    H_dense += embed_one_site(h_u_site, i, L)

H = qtn.MatrixProductOperator.from_dense(H_dense, dims=[4] * L, cutoff=1e-12)
del H_dense

# TEBD needs a local Hamiltonian, not the full MPO.
H_tebd = qtn.LocalHam1D(L=L, H2=h_t_bond, H1=h_u_site, cyclic=False)

# ------------------------------------------------------------
# Ground state
# ------------------------------------------------------------
dmrg = qtn.DMRG2(H, bond_dims=[16, 32, 64], cutoffs=1e-10)
dmrg.solve(tol=1e-8, verbosity=1)
psi0 = dmrg.state.copy()
print("Ground-state energy =", dmrg.energy)


def expect_local(psi, op, site):
    return psi.local_expectation_canonical(op, site, normalized=True)


n_ground = []
sz_ground = []
for i in range(L):
    n_ground.append(float(np.real(expect_local(psi0, n_total, i))))
    sz_ground.append(float(np.real(expect_local(psi0, Sz, i))))

N_ground = sum(n_ground)
Sz_ground = sum(sz_ground)
print("Ground-state electron number =", N_ground)
print("Ground-state total Sz =", Sz_ground)

# Add one spin-up electron at site 0 and normalize the resulting state.
psi_exc = psi0.copy()
psi_exc.gate_(c_up.conj().T, where=0, contract=True)
exc_norm = psi_exc.norm()
if exc_norm < 1e-12:
    raise ValueError("Creating an up-spin electron at site 0 gave a zero state.")
psi_exc.normalize()

N_exc = sum(
    float(np.real(expect_local(psi_exc, n_total, i)))
    for i in range(L)
)
print("Electron number after injection =", N_exc)

# ------------------------------------------------------------
# Time evolution and local observables
# ------------------------------------------------------------
times = np.arange(0.0, t_max + 0.5 * dt, dt)
tebd = qtn.TEBD(
    psi_exc,
    H_tebd,
    dt=dt,
    split_opts={"max_bond": 64, "cutoff": 1e-10},
    progbar=True,
)

n_t = []
sz_t = []
for psi_t in tebd.at_times(times, order=2):
    n_site = []
    sz_site = []
    for i in range(L):
        n_site.append(float(np.real(expect_local(psi_t, n_total, i))))
        sz_site.append(float(np.real(expect_local(psi_t, Sz, i))))
    n_t.append(n_site)
    sz_t.append(sz_site)

n_t = np.array(n_t)
sz_t = np.array(sz_t)

delta_n_t = n_t - np.array(n_ground)[None, :]
delta_sz_t = sz_t - np.array(sz_ground)[None, :]
N_t = np.sum(n_t, axis=1)
Sz_t = np.sum(sz_t, axis=1)

print("Total electron number during evolution: min =", N_t.min(), "max =", N_t.max())
print("Total Sz during evolution: min =", Sz_t.min(), "max =", Sz_t.max())
print("With SOC, total Sz is generally not conserved; total electron number should be conserved.")

# Two related quantities:
# 1) Local polarization of the actual state: 2 <Sz_i> / <n_i>.
#    This is a conventional local z-spin polarization and lies in [-1, 1].
local_pol_t = np.full_like(sz_t, np.nan, dtype=float)
valid_n = n_t > 1e-10
local_pol_t[valid_n] = 2.0 * sz_t[valid_n] / n_t[valid_n]

# 2) Ratio of changes from the ground state: 2 Delta<Sz_i> / Delta<n_i>.
#    This is only a diagnostic of the induced changes, not generally a
#    physical polarization fraction. Mask sites/times where Delta n is tiny.
density_scale = float(np.max(np.abs(delta_n_t)))
density_threshold = max(1e-3, 0.05 * density_scale)
change_ratio_t = np.full_like(delta_n_t, np.nan, dtype=float)
valid_delta_n = np.abs(delta_n_t) > density_threshold
change_ratio_t[valid_delta_n] = (
    2.0 * delta_sz_t[valid_delta_n] / delta_n_t[valid_delta_n]
)
print("Masked 2*Delta<Sz>/Delta<n> wherever |Delta n| <=", density_threshold)

# ------------------------------------------------------------
# Plot charge and spin-density changes, plus two polarization measures
# ------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
extent = [times[0], times[-1], -0.5, L - 0.5]

charge_limit = max(np.max(np.abs(delta_n_t)), 1e-12)
im0 = ax[0, 0].imshow(
    delta_n_t.T,
    origin="lower",
    aspect="auto",
    extent=extent,
    vmin=-charge_limit,
    vmax=charge_limit,
    cmap="coolwarm",
    interpolation="nearest",
)
ax[0, 0].set_xlabel("Time")
ax[0, 0].set_ylabel("Site")
ax[0, 0].set_title("Excess electron density")
fig.colorbar(im0, ax=ax[0, 0], label=r"$\Delta n_i(t)$")

spin_limit = max(np.max(np.abs(delta_sz_t)), 1e-12)
im1 = ax[0, 1].imshow(
    delta_sz_t.T,
    origin="lower",
    aspect="auto",
    extent=extent,
    vmin=-spin_limit,
    vmax=spin_limit,
    cmap="coolwarm",
    interpolation="nearest",
)
ax[0, 1].set_xlabel("Time")
ax[0, 1].set_ylabel("Site")
ax[0, 1].set_title("Change in spin density (with SOC)")
fig.colorbar(im1, ax=ax[0, 1], label=r"$\Delta S_i^z(t)$")

im2 = ax[1, 0].imshow(
    local_pol_t.T,
    origin="lower",
    aspect="auto",
    extent=extent,
    vmin=-1.0,
    vmax=1.0,
    cmap="coolwarm",
    interpolation="nearest",
)
ax[1, 0].set_xlabel("Time")
ax[1, 0].set_ylabel("Site")
ax[1, 0].set_title(r"Local polarization: $2\langle S_i^z\rangle/\langle n_i\rangle$")
fig.colorbar(im2, ax=ax[1, 0], label=r"$p_i^z(t)$")

masked_change_ratio = np.ma.masked_invalid(change_ratio_t.T)
finite_ratio = np.abs(change_ratio_t[np.isfinite(change_ratio_t)])
if finite_ratio.size > 0:
    ratio_limit = max(1.0, float(np.percentile(finite_ratio, 95)))
else:
    ratio_limit = 1.0
im3 = ax[1, 1].imshow(
    masked_change_ratio,
    origin="lower",
    aspect="auto",
    extent=extent,
    vmin=-ratio_limit,
    vmax=ratio_limit,
    cmap="coolwarm",
    interpolation="nearest",
)
ax[1, 1].set_xlabel("Time")
ax[1, 1].set_ylabel("Site")
ax[1, 1].set_title(r"Change ratio (not a standard polarization): $2\Delta S_i^z/\Delta n_i$")
fig.colorbar(im3, ax=ax[1, 1], label=r"$2\Delta S_i^z/\Delta n_i$")

plt.savefig("hubbard_soc_L6_t30.png", dpi=300, bbox_inches="tight")
plt.show()
