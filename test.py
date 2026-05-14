# ──────────────────────────────────────────────
# 3.  2-D Differentiable Topology Optimization
#     KAN parameterises the density field ρ(x,y)
# ──────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ──────────────────────────────────────────────
# 1.  Efficient KAN Linear Layer (B-spline edges)
# ──────────────────────────────────────────────
class KANLinear(nn.Module):
    """One KAN layer: in_features → out_features, each edge is a learnable B-spline."""
    def __init__(self, in_features, out_features,
                 grid_size=5, spline_order=3,
                 grid_range=(-1, 1),
                 base_activation=nn.SiLU):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        # --- B-spline knot vector (uniform, extended) ---
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1,
                            dtype=torch.float32) * h + grid_range[0]
        self.register_buffer('grid', grid.unsqueeze(0))  # (1, n_knots)

        # Learnable spline coefficients: (out, in, grid_size + spline_order)
        n_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, n_bases) * 0.1)

        # Optional residual (SiLU) connection  ── like in efficient-kan
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / np.sqrt(in_features)))
        self.base_activation = base_activation()

    def b_splines(self, x):
        """Evaluate B-spline bases of order `spline_order` at points x.
        x : (batch, in_features)
        returns : (batch, in_features, n_bases)
        """
        x = x.unsqueeze(-1)          # (B, in, 1)
        grid = self.grid              # (1, n_knots)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()  # order-0

        for k in range(1, self.spline_order + 1):
            left_num  = x - grid[:, :-(k + 1)]
            left_den  = grid[:, k:-1] - grid[:, :-(k + 1)]
            right_num = grid[:, k + 1:] - x
            right_den = grid[:, k + 1:] - grid[:, 1:-k]

            left  = left_num  / left_den.clamp(min=1e-7)  * bases[:, :, :-1]
            right = right_num / right_den.clamp(min=1e-7) * bases[:, :, 1:]
            bases = left + right
        return bases

    def forward(self, x):
        # x : (B, in_features)
        # B-spline part
        spline_out = torch.einsum(
            'bin,oin->bo',
            self.b_splines(x),      # (B, in, n_bases)
            self.spline_weight       # (out, in, n_bases)
        )
        # Residual / base part
        base_out = F.linear(self.base_activation(x), self.base_weight)
        return base_out + spline_out


class KAN(nn.Module):
    """Stack of KAN layers."""
    def __init__(self, layer_dims, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(layer_dims[i], layer_dims[i+1], **kwargs)
            for i in range(len(layer_dims) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ─── Domain & mesh ───
nelx, nely = 40, 20          # number of elements in x, y
volfrac    = 0.5              # target volume fraction
penal      = 3.0              # SIMP penalty
E0, Emin   = 1.0, 1e-9       # solid / void stiffness
nu         = 0.3              # Poisson's ratio

# Element stiffness matrix (bilinear quad, plane stress)
def element_stiffness(E0=1.0, nu=0.3):
    """8×8 stiffness matrix for a unit-size bilinear quad."""
    k = np.array([
        1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8])
    KE = E0 / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE

KE0 = torch.tensor(element_stiffness(1.0, nu), dtype=torch.float64)

# ─── DOF connectivity ───
ndof = 2 * (nelx + 1) * (nely + 1)
edofMat = np.zeros((nelx * nely, 8), dtype=int)
for elx in range(nelx):
    for ely in range(nely):
        el = elx * nely + ely
        n1 = 2 * (elx * (nely + 1) + ely)
        edofMat[el] = [n1, n1+1, n1+2*(nely+1), n1+2*(nely+1)+1,
                       n1+2*(nely+1)+2, n1+2*(nely+1)+3, n1+2, n1+3]

# ─── Load & BC (cantilever: left fixed, midpoint right load) ───
F_vec = torch.zeros(ndof, 1, dtype=torch.float64)
F_vec[2 * (nelx + 1) * (nely + 1) - nely - 1, 0] = -1.0   # downward at mid-right

fixed_dofs = np.arange(0, 2 * (nely + 1))          # left edge fixed
all_dofs   = np.arange(ndof)
free_dofs  = np.setdiff1d(all_dofs, fixed_dofs)
free_idx   = torch.tensor(free_dofs, dtype=torch.long)

# ─── Element centroid coordinates (normalised to [-1,1]) ───
cx = (np.arange(nelx) + 0.5) / nelx * 2 - 1          # [-1, 1]
cy = (np.arange(nely) + 0.5) / nely * 2 - 1
CX, CY = np.meshgrid(cx, cy, indexing='ij')
coords = torch.tensor(
    np.stack([CX.ravel(), CY.ravel()], axis=1),
    dtype=torch.float64)    # (nel, 2)

# ─── KAN density network ───
class KAN64(nn.Module):
    """Lightweight KAN: (x,y) → ρ ∈ (0,1)"""
    def __init__(self):
        super().__init__()
        self.kan = KAN([2, 16, 16, 1],
                       grid_size=8, spline_order=3,
                       grid_range=(-1.0, 1.0))

    def forward(self, xy):
        return torch.sigmoid(self.kan(xy.float())).squeeze(-1).double()

density_net = KAN64()

# ─── Differentiable FEA solver ───
def solve_fem(rho):
    """
    Given element densities rho (nel,), assemble & solve Ku=f.
    Returns compliance  C = f^T u.
    """
    nel = rho.shape[0]
    # SIMP interpolation
    Ee = Emin + rho.pow(penal) * (E0 - Emin)   # (nel,)

    # Assemble global stiffness (dense, fine for small problems)
    K = torch.zeros(ndof, ndof, dtype=torch.float64)
    for e in range(nel):
        idx = edofMat[e]
        K[np.ix_(idx, idx)] += Ee[e] * KE0

    # Solve on free DOFs
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    f_f  = F_vec[free_idx]
    u_f  = torch.linalg.solve(K_ff, f_f)

    u = torch.zeros(ndof, 1, dtype=torch.float64)
    u[free_idx] = u_f

    compliance = (F_vec * u).sum()
    return compliance, u

# ─── Optimisation loop ───
optimizer = torch.optim.Adam(density_net.parameters(), lr=5e-3)
print("Optimising topology...")
n_iters   = 200
history   = []

for it in range(n_iters):
    optimizer.zero_grad()

    rho = density_net(coords)                     # (nel,)
    rho_clamped = rho.clamp(1e-3, 1.0)

    compliance, u = solve_fem(rho_clamped)
    vol = rho_clamped.mean()

    print(f"Iter {it+1:4d}  C={compliance.item():.4f}  "
          f"vol={vol.item():.3f}")
    # Volume constraint as penalty
    vol_penalty = 1000.0 * ((vol/volfrac) - 1.0).pow(2)

    loss = compliance + vol_penalty
    loss.backward()
    optimizer.step()

    history.append((compliance.item(), vol.item()))
    if (it + 1) % 20 == 0:
        print(f"Iter {it+1:4d}  C={compliance.item():.4f}  "
              f"vol={vol.item():.3f}")

# ─── Visualise final density ───
rho_final = density_net(coords).detach().numpy().reshape(nelx, nely).T

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(1 - rho_final, cmap='gray', origin='upper',
           extent=[0, nelx, 0, nely], aspect='equal')
plt.colorbar(label='1 − ρ (white=void)')
plt.title(f'KAN TopOpt  –  Cantilever {nelx}×{nely}')
plt.xlabel('x'); plt.ylabel('y')

plt.subplot(1, 2, 2)
c_hist = [h[0] for h in history]
plt.plot(c_hist)
plt.xlabel('Iteration'); plt.ylabel('Compliance')
plt.title('Convergence'); plt.grid(True)
plt.tight_layout(); plt.show()