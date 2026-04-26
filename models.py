"""PyTorch implementations of PixelModel, CNNModel, and KANModel for
structural topology optimization, mirroring the TensorFlow models.py
from the neural-structural-optimization package.

The FEM physics is evaluated using the original autograd.numpy implementation
from neural_structural_optimization, bridged to PyTorch via a custom
torch.autograd.Function so that gradients flow correctly through the CNN / KAN.

Training functions (train_lbfgs, method_of_moving_asymptotes,
optimality_criteria) mirror their counterparts in train.py.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Local modules are now inside this folder

# ---------------------------------------------------------------------------
# Windows: Jupyter kernels do not inherit PATH, so DLL dependencies of
# torch/lib/c10.dll cannot be found. Explicitly register the directory.
# Use find_spec so this works regardless of which Python runs the kernel.
# ---------------------------------------------------------------------------
if sys.platform == "win32" and "torch" not in sys.modules:
    import importlib.util as _ilu
    _torch_spec = _ilu.find_spec("torch")
    if _torch_spec:
        _torch_lib = Path(_torch_spec.origin).parent / "lib"
        if _torch_lib.is_dir():
            try:
                os.add_dll_directory(str(_torch_lib))
            except OSError:
                pass

import autograd
import autograd.core
import autograd.numpy as anp
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray

from neural_structural_optimization import topo_api, topo_physics
from kan import KAN


# ===========================================================================
# Autograd  <->  PyTorch bridge
# ===========================================================================

class _TopoLossFunction(torch.autograd.Function):
    """Custom Function that evaluates topology compliance loss via
    autograd-numpy physics and exposes correct gradients to PyTorch."""

    @staticmethod
    def forward(ctx, logits, env):
        # logits: float64 tensor of shape (1, nely, nelx)
        x_np = logits.detach().cpu().numpy().astype(np.float64)

        def f(x):
            # x: (1, nely, nelx) -> losses: (1,)
            return anp.stack([
                env.objective(x[i], volume_contraint=True)
                for i in range(x.shape[0])
            ])

        vjp_fn, ans = autograd.core.make_vjp(f, x_np)
        ctx.vjp_fn = vjp_fn
        ctx.n_batch = x_np.shape[0]
        loss = float(np.mean(ans))
        return logits.new_tensor(loss)

    @staticmethod
    def backward(ctx, grad_output):
        # VJP of mean over batch: v = ones/n scaled by upstream grad
        v = (np.ones(ctx.n_batch, dtype=np.float64)
             / ctx.n_batch
             * grad_output.item())
        grad_np = ctx.vjp_fn(v)          # shape (1, nely, nelx)
        grad = torch.tensor(grad_np, dtype=torch.float64)
        return grad, None


# ===========================================================================
# Base model
# ===========================================================================

class Model(nn.Module):
    """Base class for all topology optimization models."""

    def __init__(self, seed=None, args=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.env = topo_api.Environment(args)

    def loss(self, logits):
        """Compliance loss (float scalar). Internally uses float64 physics."""
        return _TopoLossFunction.apply(logits.double(), self.env)

    def forward(self):
        raise NotImplementedError


# ===========================================================================
# PixelModel
# ===========================================================================

class PixelModel(Model):
    """Direct density parameterisation – one learnable value per FEM cell.

    Mirrors models.PixelModel in TensorFlow.
    """

    def __init__(self, seed=None, args=None):
        super().__init__(seed, args)
        nely = args["nely"]
        nelx = args["nelx"]
        # broadcast scalar or array mask to (nely, nelx)
        mask = np.broadcast_to(
            np.asarray(args["mask"], dtype=np.float64), (nely, nelx)
        )
        z_init = (args["volfrac"] * mask)[np.newaxis].copy()  # (1, nely, nelx)
        self.z = nn.Parameter(torch.tensor(z_init, dtype=torch.float64))

    def forward(self):
        return self.z


# ===========================================================================
# CNNModel helpers
# ===========================================================================

def _global_normalization(x, epsilon=1e-6):
    """Global normalization over all dims except batch, mirroring the TF version."""
    dims = list(range(1, x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + epsilon)


class _AddOffset(nn.Module):
    """Learnable additive bias with the full spatial shape (mirrors AddOffset in TF)."""

    def __init__(self, shape, scale=1.0):
        super().__init__()
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(*shape))

    def forward(self, x):
        return x + self.scale * self.bias


# ===========================================================================
# CNNModel
# ===========================================================================

class CNNModel(Model):
    """CNN reparameterisation of the design field.

    Mirrors models.CNNModel in TensorFlow:
        latent z -> Dense -> Reshape -> [upsample + norm + conv + offset]* -> design

    Parameters match the TF version so the same hyper-parameters can be used.
    """

    def __init__(
        self,
        seed=0,
        args=None,
        latent_size=128,
        dense_channels=32,
        resizes=(1, 2, 2, 2, 1),
        conv_filters=(128, 64, 32, 16, 1),
        offset_scale=10.0,
        kernel_size=5,
        latent_scale=1.0,
        dense_init_scale=1.0,
        activation=torch.tanh,
    ):
        super().__init__(seed, args)
        if len(resizes) != len(conv_filters):
            raise ValueError("resizes and conv_filters must have the same length")

        self.activation = activation
        self.resizes = list(resizes)

        nely = self.env.args["nely"]
        nelx = self.env.args["nelx"]
        total_resize = int(np.prod(resizes))
        h0 = nely // total_resize
        w0 = nelx // total_resize
        self.h0, self.w0 = h0, w0
        self.dense_channels = dense_channels

        # Dense projection: latent_size -> h0 * w0 * dense_channels
        filters = h0 * w0 * dense_channels
        gain = float(np.sqrt(max(filters / latent_size, 1.0)) * dense_init_scale)
        self.dense = nn.Linear(latent_size, filters)
        nn.init.orthogonal_(self.dense.weight, gain=gain)
        nn.init.zeros_(self.dense.bias)

        # Conv blocks with per-block AddOffset
        self.convs = nn.ModuleList()
        self.offsets = nn.ModuleList()
        in_ch = dense_channels
        h, w = h0, w0
        for resize, out_ch in zip(resizes, conv_filters):
            # upsample happens before conv
            h *= resize
            w *= resize
            conv = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="tanh")
            nn.init.zeros_(conv.bias)
            self.convs.append(conv)
            if offset_scale != 0:
                off = _AddOffset((1, out_ch, h, w), scale=offset_scale)
            else:
                off = None
            self.offsets.append(off)
            in_ch = out_ch

        # Latent vector (float32; physics bridge handles the cast to float64)
        self.z = nn.Parameter(
            torch.randn(1, latent_size, dtype=torch.float32) * float(latent_scale)
        )

    def _core(self, z):
        net = self.dense(z)
        net = net.view(1, self.dense_channels, self.h0, self.w0)
        for resize, conv, off in zip(self.resizes, self.convs, self.offsets):
            net = self.activation(net)
            if resize > 1:
                net = F.interpolate(
                    net, scale_factor=resize, mode="bilinear", align_corners=False
                )
            net = _global_normalization(net)
            net = conv(net)
            if off is not None:
                net = off(net)
        return net.squeeze(1)  # (1, nely, nelx)

    def forward(self):
        return self._core(self.z)


# ===========================================================================
# KANModel
# ===========================================================================

class KANModel(Model):
    """KAN reparameterisation of the design field with learned channel gating.

    Architecture
    ------------
    Two parallel paths share the same latent z (same size as CNNModel):

        z  --[Linear dense]--------------------------> spatial seed (h0, w0, dc)
           --[KAN gate]  --> sigmoid --> gates (dc,)  ├─ × channel-wise
                                                       ↓
                          [upsample + norm + conv + offset]* --> design

    The *dense* path is identical to CNNModel's projection, giving KAN at
    least CNN-level representational capacity.  The *KAN gate* path learns a
    smooth non-linear function from z to per-channel attention weights,
    effectively performing learned channel selection — something CNN cannot
    do without an explicit SENet-style attention block.  The non-linearity of
    B-spline activations is precisely where KAN has an inherent advantage over
    a single linear dense layer.

    Parameter budget
    ----------------
    All parameters relative to CNNModel for MBB 60×20 (h0=5, w0=15):
    - CNNModel dense:  128 × 2400 = 307 k
    - KANModel dense:  128 × 2400 = 307 k  (identical)
    - KANModel gate:   [128→64→32] ≈  91 k  (small KAN)
    KAN total overhead vs CNN: ≈ 91 k  (~30 % extra, concentrated in a
    purpose-built non-linear module).
    """

    def __init__(
        self,
        seed=0,
        args=None,
        latent_size=128,       # same default as CNNModel for a fair comparison
        hidden_size=64,        # intermediate width of the KAN gate network
        num_kan_layers=1,      # depth of KAN gate: 1 → [z→dc], 2 → [z→hidden→dc]
        grid=5,
        k=3,
        latent_scale=1.0,
        dense_init_scale=1.0,  # matches CNNModel
        # CNN spatial decoder hyper-parameters (mirrors CNNModel defaults)
        dense_channels=32,
        resizes=(1, 2, 2, 2, 1),
        conv_filters=(128, 64, 32, 16, 1),
        offset_scale=10.0,
        kernel_size=5,
        activation=torch.tanh,
    ):
        super().__init__(seed, args)
        nely = self.env.args["nely"]
        nelx = self.env.args["nelx"]

        self.activation = activation
        self.resizes = list(resizes)

        total_resize = int(np.prod(resizes))
        h0 = nely // total_resize
        w0 = nelx // total_resize
        self.h0, self.w0 = h0, w0
        self.dense_channels = dense_channels

        # ── Linear dense layer (identical to CNNModel) ────────────────────────
        # Maps z → spatial seed exactly as CNNModel does, ensuring KAN can
        # never perform worse than CNN on the linear projection task.
        filters = h0 * w0 * dense_channels
        gain = float(np.sqrt(max(filters / latent_size, 1.0)) * dense_init_scale)
        self.dense = nn.Linear(latent_size, filters)
        nn.init.orthogonal_(self.dense.weight, gain=gain)
        nn.init.zeros_(self.dense.bias)

        # ── KAN channel-attention gate ─────────────────────────────────────────
        # Maps z → dense_channels attention weights via learned B-spline fns.
        # KAN is ideal here: small output dimension (dc=32) avoids the curse of
        # dimensionality, and the non-linearity enables channel selection that a
        # single linear dense layer cannot represent.
        #
        # Gate width: [latent_size] + [hidden_size]*(num_kan_layers-1) + [dc]
        if num_kan_layers == 1:
            gate_width = [latent_size, dense_channels]
        else:
            gate_width = ([latent_size]
                          + [hidden_size] * (num_kan_layers - 1)
                          + [dense_channels])
        self.kan = KAN(
            width=gate_width,
            grid=grid,
            k=k,
            seed=seed,
            auto_save=False,
            save_act=False,
            symbolic_enabled=False,
            device="cpu",
        )

        # ── Calibrate B-spline knots to the actual latent distribution ────────
        # KAN defaults to grid_range=[-1,1] but z ~ randn()*latent_scale often
        # spans a different range.  Running update_grid_from_samples before any
        # gradient step ensures all knot intervals are useful from the start.
        self.kan.save_act = True
        with torch.no_grad():
            z_sample = torch.randn(256, latent_size) * float(latent_scale)
            self.kan(z_sample)
        self.kan.update_grid_from_samples(z_sample)
        self.kan.save_act = False

        # ── CNN spatial decoder (identical structure to CNNModel) ─────────────
        self.convs = nn.ModuleList()
        self.offsets = nn.ModuleList()
        in_ch = dense_channels
        h, w = h0, w0
        for resize, out_ch in zip(resizes, conv_filters):
            h *= resize
            w *= resize
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                             padding=kernel_size // 2)
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="tanh")
            nn.init.zeros_(conv.bias)
            self.convs.append(conv)
            if offset_scale != 0:
                off = _AddOffset((1, out_ch, h, w), scale=offset_scale)
            else:
                off = None
            self.offsets.append(off)
            in_ch = out_ch

        # ── Trainable latent vector ───────────────────────────────────────────
        self.z = nn.Parameter(
            torch.randn(1, latent_size, dtype=torch.float32) * float(latent_scale)
        )

    def forward(self):
        # ── Linear spatial seed (CNN-equivalent path) ─────────────────────────
        net = self.dense(self.z)                                 # (1, h0*w0*dc)
        net = net.view(1, self.dense_channels, self.h0, self.w0)

        # ── KAN channel gates (non-linear channel attention) ──────────────────
        # Apply an open-door bias (+3.0) and lower bound (0.05 + 0.95*sigmoid)
        # to prevent dead channels and let L-BFGS track smoothly from the start.
        gates = 0.05 + 0.95 * torch.sigmoid(self.kan(self.z) + 3.0)      # (1, dc)
        net = net * gates.view(1, self.dense_channels, 1, 1)     # (1, dc, h0, w0)

        for resize, conv, off in zip(self.resizes, self.convs, self.offsets):
            net = self.activation(net)
            if resize > 1:
                net = F.interpolate(net, scale_factor=resize,
                                    mode="bilinear", align_corners=False)
            net = _global_normalization(net)
            net = conv(net)
            if off is not None:
                net = off(net)

        return net.squeeze(1)   # (1, nely, nelx)


# ===========================================================================
# CoordKANModel  — coordinate-based KAN (physics-motivated)
# ===========================================================================

class CoordKANModel(Model):
    """Coordinate-based KAN: maps (x, y) → density ρ(x, y).

    Physics motivation
    ------------------
    In structural topology optimisation the optimal density field is a smooth
    function of spatial position — governed by load paths and equilibrium.
    KAN's learnable B-spline activations naturally represent smooth 2-D
    functions, unlike CNN which needs weight-sharing spatial filters.

    Instead of a latent-vector → spatial-decoder pipeline (CNNModel/KANModel),
    this model feeds the normalised element coordinates directly as network
    input and predicts density element-wise:

        coords (H·W, 2)  →  KAN  →  logits (H·W, 1)  →  sigmoid  →  densities (1, H, W)

    Key design decisions
    --------------------
    - ``grid_range=[0, 1]``: KAN's B-spline knots are placed over [0, 1] to
      match the actual coordinate domain.  The default [-1, 1] wastes half the
      knot resolution on the unoccupied negative half of each axis.
    - Grid calibration at init: ``update_grid_from_samples`` is called once
      after construction so the knot spacing exactly fits the coordinate
      distribution before any L-BFGS step starts.
    - No volume projection: explicit density rescaling with clamping destroys
      gradients for clamped elements.  Volume balance is handled implicitly by
      ``density_bias`` initialisation and the physics objective.

    Unique KAN capability: adaptive B-spline grid refinement
    ---------------------------------------------------------
    Call ``model.refine(new_grid)`` to expand the spline grid during training.
    The refined KAN inherits the currently learnt function (via spline fitting)
    and can then resolve finer spatial structure — analogous to h-refinement in
    adaptive FEM.  No equivalent operation exists for CNN-based models.

    Parameters
    ----------
    kan_layers : tuple of int
        Hidden layer widths.  Default ``(32, 32)`` → KAN width [2, 32, 32, 1].
        Reduce to ``(16,)`` for large grids to keep runtime manageable.
    grid : int
        Number of B-spline intervals per activation (over [0, 1]).
        Higher values → more spatial resolution but more parameters per layer.
    k : int
        B-spline order (3 = cubic, smooth & differentiable).
    """

    def __init__(
        self,
        seed=0,
        args=None,
        kan_layers=(32, 32),
        grid=10,
        k=3,
    ):
        super().__init__(seed, args)
        nely = self.env.args["nely"]
        nelx = self.env.args["nelx"]
        self.nely = nely
        self.nelx = nelx
        self.volfrac = float(args.get("volfrac", 0.5))

        # KAN: 2 coords → hidden layers → 1 density logit.
        # grid_range=[0,1] places spline knots over the actual coordinate domain
        # so all grid intervals are useful (vs default [-1,1] which wastes half).
        width = [2] + list(kan_layers) + [1]
        self.kan = KAN(
            width=width,
            grid=grid,
            k=k,
            seed=seed,
            auto_save=False,
            save_act=False,
            symbolic_enabled=False,
            grid_range=[0, 1],
            device="cpu",
        )

        # Normalised (x, y) coordinate grid for all H·W elements.
        xs = torch.linspace(0.0, 1.0, nelx)
        ys = torch.linspace(0.0, 1.0, nely)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # (H, W) each
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (H·W, 2)
        self.register_buffer("coords", coords.float())

        # Calibrate spline grids to the actual coordinate distribution before
        # any gradient steps.  Requires a forward pass with save_act=True to
        # populate self.kan.acts, then update_grid_from_samples re-places knots.
        self.kan.save_act = True
        with torch.no_grad():
            self.kan(self.coords)
        self.kan.update_grid_from_samples(self.coords)
        self.kan.save_act = False

        # Scalar bias initialised so sigmoid(bias) ≈ target volume fraction.
        init_logit = float(np.log(self.volfrac / (1.0 - self.volfrac + 1e-8)))
        self.density_bias = nn.Parameter(torch.tensor([init_logit]))

    def forward(self):
        logits = self.kan(self.coords) + self.density_bias  # (H·W, 1)
        return torch.sigmoid(logits).view(1, self.nely, self.nelx)

    def warmstart(self, density_2d, n_steps=400, lr=0.02):
        """Pre-fit the KAN to a reference density field via Adam MSE minimisation.

        Running L-BFGS from uniform ρ=0.5 means the KAN must simultaneously
        discover load-path topology AND fit B-spline coefficients — an extremely
        hard joint task.  This method decouples the two phases:

            Phase A (warmstart): Adam minimises MSE(KAN(coords), ρ_ref)
                                 The KAN learns the spatial structure of a
                                 reference design (e.g. from OC or pixel-LBFGS).
            Phase B (main):      L-BFGS minimises compliance, refining the
                                 topology from the warm-started initial point.

        Parameters
        ----------
        density_2d : np.ndarray of shape (H, W), values in [0, 1]
            Reference density field to match.  Best results use an OC or
            pixel-LBFGS design after ~50 iterations (structural load paths
            already visible but not yet fully 0/1).
        n_steps : int
            Number of Adam gradient steps for the pre-fit.
        lr : float
            Adam learning rate.  0.02 works well for (32,32) KANs.

        Returns
        -------
        self  (for chaining)
        """
        target = torch.tensor(density_2d.flatten(), dtype=torch.float32).clamp(0.02, 0.98)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for step in range(n_steps):
            opt.zero_grad()
            pred = self.forward().flatten()
            loss = nn.functional.mse_loss(pred, target)
            loss.backward()
            opt.step()
            if (step + 1) % 100 == 0:
                print(f"    [warmstart] step {step+1}/{n_steps}  MSE={loss.item():.5f}")
        return self

    def refine(self, new_grid):
        """Expand the B-spline grid in-place (adaptive refinement).

        The refined KAN transfers the currently learnt activations to the
        finer grid via spline fitting — no training data replay needed.
        This is analogous to h-refinement in adaptive FEM.

        Parameters
        ----------
        new_grid : int
            Target number of B-spline intervals (must be > current grid).

        Returns
        -------
        self  (for chaining)
        """
        # pykan's refine() reads spline_preacts / spline_postsplines which are
        # only populated when save_act=True and a forward pass has been run.
        self.kan.save_act = True
        with torch.no_grad():
            self.kan(self.coords)                    # fills spline activation cache

        # pykan.refine() writes to ckpt_path/history.txt — ensure directory exists.
        import os
        os.makedirs(self.kan.ckpt_path, exist_ok=True)

        self.kan = self.kan.refine(new_grid)
        self.kan.save_act = False
        return self


# ===========================================================================
# Training utilities
# ===========================================================================

def _optimizer_result_dataset(losses, frames, save_intermediate_designs=True):
    """Build an xarray Dataset identical in structure to train.py's version."""
    best = int(np.nanargmin(losses))
    if save_intermediate_designs:
        ds = xarray.Dataset(
            {
                "loss": (("step",), losses),
                "design": (("step", "y", "x"), frames),
            },
            coords={"step": np.arange(len(losses))},
        )
    else:
        ds = xarray.Dataset(
            {
                "loss": (("step",), losses),
                "design": (("y", "x"), frames[best]),
            },
            coords={"step": np.arange(len(losses))},
        )
    return ds


def _get_params_flat(model):
    """Flatten all trainable parameters into a 1-D float64 numpy array."""
    return np.concatenate([
        p.detach().cpu().numpy().ravel() for p in model.parameters()
    ]).astype(np.float64)


def _set_params_flat(model, x):
    """Write a flat float64 numpy vector back into the model's parameters."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        val = x[offset: offset + n].reshape(p.shape)
        with torch.no_grad():
            p.copy_(torch.tensor(val, dtype=p.dtype))
        offset += n


# ---------------------------------------------------------------------------
# L-BFGS via scipy  (mirrors train.train_lbfgs)
# ---------------------------------------------------------------------------

def train_lbfgs(model, max_iterations, save_intermediate_designs=True, **kwargs):
    """Train any Model subclass with L-BFGS via scipy.optimize.fmin_l_bfgs_b.

    Mirrors train.train_lbfgs from the neural-structural-optimization package.
    Works for PixelModel, CNNModel, and KANModel.
    """
    losses = []
    frames = []

    def value_and_grad(x):
        _set_params_flat(model, x)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        logits = model()
        loss = model.loss(logits)
        loss.backward()
        grad = np.concatenate([
            p.grad.detach().cpu().numpy().ravel()
            if p.grad is not None
            else np.zeros(p.numel(), dtype=np.float64)
            for p in model.parameters()
        ]).astype(np.float64)
        frames.append(logits.detach().cpu().numpy().copy())
        losses.append(float(loss.detach().cpu()))
        return float(loss.detach().cpu()), grad

    x0 = _get_params_flat(model).astype(np.float64)
    scipy.optimize.fmin_l_bfgs_b(
        value_and_grad, x0,
        maxfun=max_iterations, factr=1, pgtol=1e-14,
        **kwargs,
    )

    # render each frame with volume constraint applied
    designs = [
        model.env.render(f.reshape(-1), volume_contraint=True)
        for f in frames
    ]
    return _optimizer_result_dataset(
        np.array(losses), np.array(designs), save_intermediate_designs
    )


# ---------------------------------------------------------------------------
# Adaptive KAN training  (grid progressive refinement — KAN-unique)
# ---------------------------------------------------------------------------

def train_lbfgs_adaptive_kan(model, grid_schedule, save_intermediate_designs=True):
    """Train a KAN-based model with progressive B-spline grid refinement.

    This exploits a KAN-unique capability: after each training phase the spline
    grid is expanded and the learned function is projected onto the finer
    representation via spline fitting — no training data replay needed.
    This is analogous to h-refinement in adaptive finite-element methods.
    CNN-based models have no equivalent capability.

    Parameters
    ----------
    model : CoordKANModel or KANModel
        Must have a ``model.refine(new_grid)`` method (CoordKANModel) or a
        ``model.kan`` attribute that exposes ``KAN.refine()``.
    grid_schedule : list of ``(iterations, grid_size)`` pairs
        Each pair specifies how many L-BFGS steps to run before expanding to
        the given grid size.  The *first* entry uses the model's existing grid;
        subsequent entries trigger a refinement step.

        Example: ``[(50, 3), (100, 7), (100, 15)]``
        - Phase 1: 50 iterations  (model already initialised at grid=3)
        - Phase 2: refine  grid → 7, then 100 iterations
        - Phase 3: refine  grid → 15, then 100 iterations

    Returns
    -------
    xarray.Dataset  with concatenated losses and designs across all phases.
    """
    all_losses = []
    all_frames = []   # list of (H, W) rendered density arrays

    for phase_idx, (iters, grid_size) in enumerate(grid_schedule):
        if phase_idx == 0:
            print(f"  [AdaptKAN] Phase 1/{len(grid_schedule)}: "
                  f"grid={grid_size}, {iters} iterations (initial)")
        else:
            print(f"  [AdaptKAN] Phase {phase_idx + 1}/{len(grid_schedule)}: "
                  f"refining KAN grid → {grid_size}, {iters} iterations")
            # Expand the B-spline grid; the PyTorch module registry is updated
            # automatically so model.parameters() yields the new param set.
            if hasattr(model, "refine"):
                model.refine(grid_size)
            elif hasattr(model, "kan") and hasattr(model.kan, "refine"):
                model.kan = model.kan.refine(grid_size)
            else:
                raise AttributeError(
                    "model does not expose a .refine() method. "
                    "Use CoordKANModel or KANModel."
                )

        ds_phase = train_lbfgs(model, iters, save_intermediate_designs=True)
        all_losses.extend(ds_phase.loss.values.tolist())
        # ds_phase.design has shape (steps, H, W) when save_intermediate_designs=True
        for step_idx in range(ds_phase.sizes["step"]):
            all_frames.append(ds_phase.design.isel(step=step_idx).values)

    designs_arr = np.array(all_frames)   # (total_steps, H, W)
    losses_arr = np.array(all_losses, dtype=np.float64)
    return _optimizer_result_dataset(losses_arr, designs_arr, save_intermediate_designs)


# ---------------------------------------------------------------------------
# Method of Moving Asymptotes  (mirrors train.method_of_moving_asymptotes)
# ---------------------------------------------------------------------------

def method_of_moving_asymptotes(
    model, max_iterations, save_intermediate_designs=True
):
    """MMA optimiser (nlopt) for PixelModel only.

    Mirrors train.method_of_moving_asymptotes exactly; the underlying physics
    is unchanged and computed via autograd.numpy.
    """
    try:
        import nlopt
    except ImportError:
        raise ImportError(
            "nlopt is required for MMA. Install with: pip install nlopt"
        )

    if not isinstance(model, PixelModel):
        raise ValueError("MMA is only defined for PixelModel")

    env = model.env
    x0 = _get_params_flat(model).astype(np.float64)

    def _objective(x):
        return env.objective(x, volume_contraint=False)

    def _constraint(x):
        return env.constraint(x)

    def _wrap(func, losses=None, frames=None):
        def wrapper(x, grad):
            if grad.size > 0:
                value, grad[:] = autograd.value_and_grad(func)(x)
            else:
                value = func(x)
            if losses is not None:
                losses.append(value)
            if frames is not None:
                frames.append(env.reshape(x).copy())
            return value
        return wrapper

    losses, frames = [], []
    opt = nlopt.opt(nlopt.LD_MMA, x0.size)
    opt.set_lower_bounds(0.0)
    opt.set_upper_bounds(1.0)
    opt.set_min_objective(_wrap(_objective, losses, frames))
    opt.add_inequality_constraint(_wrap(_constraint), 1e-8)
    opt.set_maxeval(max_iterations + 1)
    opt.optimize(x0)

    designs = [env.render(x, volume_contraint=False) for x in frames]
    return _optimizer_result_dataset(
        np.array(losses), np.array(designs), save_intermediate_designs
    )


# ---------------------------------------------------------------------------
# Optimality Criteria  (mirrors train.optimality_criteria)
# ---------------------------------------------------------------------------

def optimality_criteria(
    model, max_iterations, save_intermediate_designs=True
):
    """OC optimiser for PixelModel only.

    Mirrors train.optimality_criteria exactly; the underlying physics is
    unchanged and computed via autograd.numpy.
    """
    if not isinstance(model, PixelModel):
        raise ValueError("Optimality criteria is only defined for PixelModel")

    env = model.env
    x = _get_params_flat(model).astype(np.float64)

    losses = []
    frames = [x.copy()]
    for _ in range(max_iterations):
        c, x = topo_physics.optimality_criteria_step(x, env.ke, env.args)
        losses.append(c)
        if np.isnan(c):
            break
        frames.append(x.copy())
    losses.append(env.objective(x, volume_contraint=False))

    designs = [env.render(f, volume_contraint=False) for f in frames]
    return _optimizer_result_dataset(
        np.array(losses), np.array(designs), save_intermediate_designs
    )
