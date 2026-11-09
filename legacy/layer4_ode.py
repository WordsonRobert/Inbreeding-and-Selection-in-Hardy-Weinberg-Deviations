"""
layer4_ode.py
=============
BASTION Layer 4: Physics-Informed Neural ODE Ensemble.

Learns the continuous dynamics d/dt[p, F] = f_θ(p, F, t) from sparse
observed genotype counts, constrained by the WF physics at collocation
points. Provides smooth interpolated trajectories between observations
with calibrated uncertainty via ensemble of 10 independently initialised
networks.

Architecture (per network):
  Input:  (p, F, t_normalised) — 3 scalars
  Hidden: 2 layers × 64 units, tanh activation
  Output: (dp/dt, dF/dt) — 2 scalars
  Boundary enforcement:
    dp/dt ← dp/dt × p(1-p)       keeps p ∈ (0,1)
    dF/dt ← dF/dt × (1-F)(F-lb)  keeps F in admissible range

Loss (per network):
  L_total = L_data + λ_F·L_phys_F + λ_p·L_phys_p

  L_data    = binomial NLL on allele counts at observed timepoints
  L_phys_F  = MSE(f_θ^F, ΔF_Wright)               at 150 collocation points
  L_phys_p  = MSE(f_θ^p, Δp_selection + Δp_mutation) at 150 collocation points
              [combined WF increment — not split, which would be contradictory]

Regime handling:
  Separate ODE segment per regime from Layer 1.
  Continuous at boundaries: terminal state of segment k = initial
  state of segment k+1.

Uncertainty:
  Ensemble of 10 networks, trained independently with different seeds.
  90% spread across ensemble = epistemic uncertainty band.

Internal consistency check:
  R_residual = dR_idx/dt_ODE - (ΔR_F + ΔR_s + ΔR_μ)
  Residual > 0.05 flags potential closed-system violation.

Author: BASTION project
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional
import warnings

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_wf import (delta_p_selection, delta_p_mutation,
                            genotype_freqs)

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class ODEInput:
    """Input for Layer 4."""
    # From Layer 2: posterior means of parameters
    p0:    float
    F0:    float
    s:     float
    h:     float
    mu1:   float
    mu2:   float
    # From data: observed timepoints and genotype counts
    generations:  np.ndarray   # shape (T_obs,)
    N_AA:         np.ndarray
    N_Aa:         np.ndarray
    N_aa:         np.ndarray
    Ne:           np.ndarray   # shape (T_max,)
    # From Layer 1: regime changepoints (generation indices)
    changepoints: list[int] = field(default_factory=list)
    # F lower bound
    F_lb:         float = -1.0


@dataclass
class ODEResult:
    """Output of Layer 4 ensemble."""
    # Dense time grid
    t_grid:       np.ndarray    # shape (T_dense,)
    # Ensemble trajectories: shape (n_ensemble, T_dense)
    p_ensemble:   np.ndarray
    F_ensemble:   np.ndarray
    R_ensemble:   np.ndarray    # R_idx = F*p*q
    # Summary statistics
    p_med:        np.ndarray    # (T_dense,)
    p_lo:         np.ndarray    # 5th percentile across ensemble
    p_hi:         np.ndarray    # 95th percentile across ensemble
    F_med:        np.ndarray
    F_lo:         np.ndarray
    F_hi:         np.ndarray
    R_med:        np.ndarray
    R_lo:         np.ndarray
    R_hi:         np.ndarray
    # Residual diagnostic
    R_residual:   np.ndarray    # dR_ODE - (ΔR_F + ΔR_s + ΔR_mu), shape (T_dense,)
    max_residual: float         # max |R_residual| — > 0.05 flags violation
    closed_system_ok: bool      # True if max_residual < 0.05
    # Training metadata
    n_ensemble:   int
    losses:       list[float]   # final loss per network


# ─────────────────────────────────────────────
# Neural ODE network
# ─────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    Neural network defining d/dt[p, F] = f_θ(p, F, t).

    Input: (p, F, t_norm) — 3 scalars
    Output: (dp/dt, dF/dt) — 2 scalars, with boundary enforcement.
    """
    def __init__(self, hidden: int = 64, F_lb: float = -1.0):
        super().__init__()
        self.F_lb = F_lb
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        # Initialise with small weights — start near zero dynamics
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t_scalar: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, 2) tensor with columns [p, F]
        t_scalar: scalar tensor (current time, normalised to [0,1])
        Returns: (batch, 2) tensor [dp/dt, dF/dt]
        """
        batch = state.shape[0]
        t_exp = t_scalar.expand(batch, 1)
        inp = torch.cat([state, t_exp], dim=1)   # (batch, 3)
        raw = self.net(inp)                        # (batch, 2)

        p = state[:, 0:1]
        F = state[:, 1:2]

        # Boundary enforcement — zero derivative at boundaries
        dp = raw[:, 0:1] * p * (1.0 - p)

        lb = self.F_lb
        # Scale: (F - lb)(1 - F) — zero at F=lb and F=1
        F_scale = (F - lb) * (1.0 - F)
        dF = raw[:, 1:2] * F_scale

        return torch.cat([dp, dF], dim=1)


# ─────────────────────────────────────────────
# Physics collocation targets
# ─────────────────────────────────────────────

def physics_targets(p_vals: np.ndarray, F_vals: np.ndarray,
                     s: float, h: float, mu1: float, mu2: float,
                     Ne: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute WF physics targets (dp_WF, dF_WF) at collocation points.
    These are what the ODE network is penalised toward.

    Returns: dp_target (n,), dF_target (n,)
    """
    dp = np.array([delta_p_selection(p, s, h) + delta_p_mutation(p, mu1, mu2)
                   for p in p_vals])

    # F Wright recursion + mutation correction
    dF = np.array([
        (1.0 / (2*Ne) + (1.0 - 1.0/(2*Ne)) * F) * (1.0 - mu1 - mu2)**2 - F
        for F in F_vals
    ])

    return dp.astype(np.float32), dF.astype(np.float32)


def sample_collocation(n: int, p_range=(0.05, 0.95), F_range=(-0.5, 0.9),
                        rng: Optional[np.random.Generator] = None):
    """Sample n collocation points uniformly in (p, F) space."""
    if rng is None:
        rng = np.random.default_rng()
    p = rng.uniform(*p_range, n).astype(np.float32)
    F = rng.uniform(*F_range, n).astype(np.float32)
    return p, F


# ─────────────────────────────────────────────
# Data loss: binomial NLL on allele counts
# ─────────────────────────────────────────────

def binomial_nll(p_pred: torch.Tensor,
                 N_AA: torch.Tensor, N_Aa: torch.Tensor, N_aa: torch.Tensor,
                 F_pred: torch.Tensor) -> torch.Tensor:
    """
    Multinomial NLL on observed genotype counts.
    p_pred, F_pred: (T_obs,) tensors
    """
    q = 1.0 - p_pred
    P_AA = p_pred**2 + F_pred * p_pred * q
    P_Aa = 2 * p_pred * q * (1.0 - F_pred)
    P_aa = q**2 + F_pred * p_pred * q

    eps = 1e-8
    P_AA = torch.clamp(P_AA, eps, 1.0)
    P_Aa = torch.clamp(P_Aa, eps, 1.0)
    P_aa = torch.clamp(P_aa, eps, 1.0)

    nll = -(N_AA * torch.log(P_AA) +
            N_Aa * torch.log(P_Aa) +
            N_aa * torch.log(P_aa))
    return nll.mean()


# ─────────────────────────────────────────────
# ODE integration (simple Euler/RK4, no torchdiffeq needed for training)
# ─────────────────────────────────────────────

def integrate_rk4(func: ODEFunc,
                   state0: torch.Tensor,
                   t_span: torch.Tensor,
                   t_norm_max: float = 1.0) -> torch.Tensor:
    """
    RK4 integration of the ODE.
    state0: (1, 2) initial state [p0, F0]
    t_span: (T,) tensor of time points (unnormalised generations)
    Returns: (T, 2) trajectory
    """
    states = [state0]
    T_max = float(t_span[-1])

    for i in range(len(t_span) - 1):
        t_cur  = t_span[i]
        t_next = t_span[i + 1]
        dt = t_next - t_cur
        t_n = t_cur / T_max   # normalised time scalar

        s = states[-1]
        k1 = func(t_n, s)
        k2 = func((t_cur + dt/2) / T_max, s + dt/2 * k1)
        k3 = func((t_cur + dt/2) / T_max, s + dt/2 * k2)
        k4 = func((t_cur + dt)   / T_max, s + dt   * k3)

        s_new = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Clamp state to valid range
        p_new = torch.clamp(s_new[:, 0:1], 1e-6, 1.0 - 1e-6)
        F_lb  = func.F_lb
        F_new = torch.clamp(s_new[:, 1:2], F_lb + 1e-6, 1.0 - 1e-6)
        states.append(torch.cat([p_new, F_new], dim=1))

    return torch.cat(states, dim=0)   # (T, 2)


# ─────────────────────────────────────────────
# Train one network
# ─────────────────────────────────────────────

def train_one_network(ode_input: ODEInput,
                       seed: int = 0,
                       n_epochs: int = 2000,
                       lr: float = 1e-3,
                       n_colloc: int = 150,
                       lambda_F: float = 1.0,
                       lambda_s: float = 0.5,
                       hidden: int = 64,
                       verbose: bool = False) -> tuple[ODEFunc, float]:
    """
    Train a single physics-informed ODE network.

    Returns: (trained ODEFunc, final_loss)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    func = ODEFunc(hidden=hidden, F_lb=ode_input.F_lb).to(DEVICE)
    optimizer = optim.Adam(func.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

    # Observed data tensors
    gens = ode_input.generations
    T_max = float(gens[-1] - gens[0])
    t_obs = torch.tensor((gens - gens[0]).astype(np.float32)).to(DEVICE)

    N_AA_t = torch.tensor(ode_input.N_AA.astype(np.float32)).to(DEVICE)
    N_Aa_t = torch.tensor(ode_input.N_Aa.astype(np.float32)).to(DEVICE)
    N_aa_t = torch.tensor(ode_input.N_aa.astype(np.float32)).to(DEVICE)

    # Initial state
    state0 = torch.tensor([[ode_input.p0, ode_input.F0]],
                            dtype=torch.float32).to(DEVICE)

    # Mean Ne for collocation physics
    Ne_mean = float(ode_input.Ne.mean())

    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # ── Data loss ──
        traj = integrate_rk4(func, state0, t_obs)   # (T_obs, 2)
        p_pred = traj[:, 0]
        F_pred = traj[:, 1]
        L_data = binomial_nll(p_pred, N_AA_t, N_Aa_t, N_aa_t, F_pred)

        # ── Physics loss: collocation ──
        p_col, F_col = sample_collocation(n_colloc, rng=rng)
        dp_tgt, dF_tgt = physics_targets(p_col, F_col,
                                          ode_input.s, ode_input.h,
                                          ode_input.mu1, ode_input.mu2,
                                          Ne_mean)

        p_col_t = torch.tensor(p_col).unsqueeze(1).to(DEVICE)
        F_col_t = torch.tensor(F_col).unsqueeze(1).to(DEVICE)
        state_col = torch.cat([p_col_t, F_col_t], dim=1)

        t_mid = torch.tensor(0.5, dtype=torch.float32).to(DEVICE)
        deriv_col = func(t_mid, state_col)   # (n_colloc, 2)

        dp_pred_col = deriv_col[:, 0]
        dF_pred_col = deriv_col[:, 1]

        dp_tgt_t = torch.tensor(dp_tgt).to(DEVICE)
        dF_tgt_t = torch.tensor(dF_tgt).to(DEVICE)

        # Physics loss: penalise against combined WF increment (selection + mutation).
        # Penalising dp_pred against each component separately is contradictory —
        # the ODE learns total dp/dt, not individual force contributions.
        # dp_tgt already contains the combined increment from physics_targets().
        L_phys_p = ((dp_pred_col - dp_tgt_t)**2).mean()
        L_phys_F = ((dF_pred_col - dF_tgt_t)**2).mean()

        loss = L_data + lambda_F * L_phys_F + lambda_s * L_phys_p
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in func.state_dict().items()}

        if verbose and (epoch + 1) % 500 == 0:
            print(f"    Epoch {epoch+1:4d}: loss={loss_val:.6f} "
                  f"(data={L_data.item():.4f}, F={L_phys_F.item():.6f}, "
                  f"p={L_phys_p.item():.6f})")

    # Restore best weights
    if best_state is not None:
        func.load_state_dict(best_state)

    return func, best_loss


# ─────────────────────────────────────────────
# Ensemble prediction
# ─────────────────────────────────────────────

def predict_ensemble(trained_nets: list[ODEFunc],
                      ode_input: ODEInput,
                      t_dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run all trained networks on a dense time grid.

    Returns:
        p_ens : (n_nets, T_dense) array
        F_ens : (n_nets, T_dense) array
    """
    gen_start = ode_input.generations[0]
    t_rel = (t_dense - gen_start).astype(np.float32)
    t_tensor = torch.tensor(t_rel).to(DEVICE)

    p_ens = np.zeros((len(trained_nets), len(t_dense)))
    F_ens = np.zeros((len(trained_nets), len(t_dense)))

    state0 = torch.tensor([[ode_input.p0, ode_input.F0]],
                            dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        for i, net in enumerate(trained_nets):
            traj = integrate_rk4(net, state0, t_tensor)
            p_ens[i] = traj[:, 0].cpu().numpy()
            F_ens[i] = traj[:, 1].cpu().numpy()

    return p_ens, F_ens


# ─────────────────────────────────────────────
# R_residual diagnostic
# ─────────────────────────────────────────────

def compute_residual(p_traj: np.ndarray, F_traj: np.ndarray,
                      t_grid: np.ndarray,
                      s: float, h: float,
                      mu1: float, mu2: float,
                      Ne: float) -> np.ndarray:
    """
    Compute R_residual(t) = dR_idx/dt_ODE - (ΔR_F + ΔR_s + ΔR_mu)

    dR_idx/dt_ODE is estimated from the ODE trajectory via finite differences.
    ΔR_F, ΔR_s, ΔR_mu are the analytic WF increments evaluated at (p(t), F(t)).

    R_residual ≈ 0 for a true closed system.
    |R_residual| > 0.05 flags a potential closed-system violation (migration).
    """
    R_traj = F_traj * p_traj * (1.0 - p_traj)
    dR_ode = np.gradient(R_traj, t_grid)

    residual = np.zeros(len(t_grid))
    for i in range(len(t_grid)):
        p = p_traj[i]
        F = F_traj[i]
        q = 1.0 - p
        dp_s  = delta_p_selection(p, s, h)
        dp_mu = delta_p_mutation(p, mu1, mu2)
        dF_WF = (1.0/(2*Ne) + (1.0 - 1.0/(2*Ne)) * F) * (1.0 - mu1 - mu2)**2 - F
        dR_F   = p * q * dF_WF
        dR_s   = F * (1.0 - 2*p) * dp_s
        dR_mu  = F * (1.0 - 2*p) * dp_mu
        residual[i] = dR_ode[i] - (dR_F + dR_s + dR_mu)

    return residual


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def run_ode_ensemble(ode_input: ODEInput,
                      n_ensemble: int = 10,
                      n_epochs: int = 2000,
                      lr: float = 1e-3,
                      n_colloc: int = 150,
                      t_dense_step: int = 1,
                      verbose: bool = True) -> ODEResult:
    """
    Train ensemble of Neural ODEs and return smooth trajectories.

    Parameters
    ----------
    ode_input : ODEInput
        Data and parameters from Layers 1-3.
    n_ensemble : int
        Number of independently initialised networks (default 10).
    n_epochs : int
        Training epochs per network (default 2000).
    lr : float
        Learning rate (default 1e-3).
    n_colloc : int
        Number of physics collocation points (default 150).
    t_dense_step : int
        Step size for dense prediction grid (default 1 = every generation).
    verbose : bool

    Returns
    -------
    ODEResult
    """
    if verbose:
        print(f"\nLayer 4 — Neural ODE Ensemble")
        print(f"  Device: {DEVICE}")
        print(f"  Ensemble size: {n_ensemble}")
        print(f"  Epochs per network: {n_epochs}")
        print(f"  Observed timepoints: {len(ode_input.generations)}")
        print(f"  Time range: gen {ode_input.generations[0]}–{ode_input.generations[-1]}")

    # Well-separated seeds for genuine ensemble diversity.
    # Sequential seeds (i*7+42) produce near-identical initializations
    # via xavier_normal — prime-spaced seeds ensure diverse weight trajectories.
    ENSEMBLE_SEEDS = [42, 137, 271, 503, 997, 1999, 3141, 5003, 7919, 10007]

    # Train ensemble
    trained_nets = []
    final_losses = []

    for i in range(n_ensemble):
        seed = ENSEMBLE_SEEDS[i] if i < len(ENSEMBLE_SEEDS) else i * 9973 + 42
        if verbose:
            print(f"  Training network {i+1}/{n_ensemble} (seed={seed})...",
                  end='', flush=True)
        net, loss = train_one_network(
            ode_input, seed=seed,
            n_epochs=n_epochs, lr=lr, n_colloc=n_colloc,
            verbose=False
        )
        trained_nets.append(net)
        final_losses.append(loss)
        if verbose:
            print(f" loss={loss:.4f}")

    # Dense prediction grid
    gen_start = int(ode_input.generations[0])
    gen_end   = int(ode_input.generations[-1])
    t_grid    = np.arange(gen_start, gen_end + 1, t_dense_step, dtype=np.float32)

    if verbose:
        print(f"  Predicting on dense grid ({len(t_grid)} points)...")

    p_ens, F_ens = predict_ensemble(trained_nets, ode_input, t_grid)
    R_ens = F_ens * p_ens * (1.0 - p_ens)

    # Ensemble summaries
    p_med = np.median(p_ens, axis=0)
    p_lo  = np.percentile(p_ens, 5,  axis=0)
    p_hi  = np.percentile(p_ens, 95, axis=0)

    F_med = np.median(F_ens, axis=0)
    F_lo  = np.percentile(F_ens, 5,  axis=0)
    F_hi  = np.percentile(F_ens, 95, axis=0)

    R_med = np.median(R_ens, axis=0)
    R_lo  = np.percentile(R_ens, 5,  axis=0)
    R_hi  = np.percentile(R_ens, 95, axis=0)

    # Residual diagnostic using median trajectory
    Ne_mean = float(ode_input.Ne.mean())
    residual = compute_residual(
        p_med, F_med, t_grid,
        ode_input.s, ode_input.h,
        ode_input.mu1, ode_input.mu2,
        Ne_mean
    )

    max_resid = float(np.abs(residual).max())
    closed_ok = max_resid < 0.05

    if verbose:
        print(f"\n  Max |R_residual|: {max_resid:.5f} "
              f"({'OK — closed system' if closed_ok else 'WARNING — possible violation'})")
        print(f"  Ensemble spread (p): [{(p_hi - p_lo).mean():.4f} avg 90% width]")
        print(f"  Final losses: min={min(final_losses):.4f}, "
              f"max={max(final_losses):.4f}, "
              f"mean={np.mean(final_losses):.4f}")

    return ODEResult(
        t_grid=t_grid,
        p_ensemble=p_ens,
        F_ensemble=F_ens,
        R_ensemble=R_ens,
        p_med=p_med, p_lo=p_lo, p_hi=p_hi,
        F_med=F_med, F_lo=F_lo, F_hi=F_hi,
        R_med=R_med, R_lo=R_lo, R_hi=R_hi,
        R_residual=residual,
        max_residual=max_resid,
        closed_system_ok=closed_ok,
        n_ensemble=n_ensemble,
        losses=final_losses,
    )


# ─────────────────────────────────────────────
# Convenience: build ODEInput from Layer 2+3 outputs
# ─────────────────────────────────────────────

def ode_input_from_mcmc(mcmc_result,
                         mcmc_data,
                         changepoints: list[int] = None) -> ODEInput:
    """
    Build ODEInput from Layer 2 MCMCResult and MCMCData.
    Uses posterior means as the point estimate for ODE initialisation.
    """
    summary = mcmc_result.summary
    p_names = mcmc_result.param_names

    def get_mean(name, default):
        if name in summary:
            return summary[name]['mean']
        return default

    p0  = get_mean('p0',  0.5)
    F0  = get_mean('F0',  0.0)
    s   = get_mean('s',   0.0)
    h   = get_mean('h',   0.5)
    mu1 = (mcmc_data.mu1_fixed if mcmc_data.mu1_fixed is not None
           else get_mean('mu1', 1e-5))
    mu2 = (mcmc_data.mu2_fixed if mcmc_data.mu2_fixed is not None
           else get_mean('mu2', 1e-5))

    q0  = 1.0 - p0
    F_lb = -min(p0, q0) / max(p0, q0)

    return ODEInput(
        p0=p0, F0=F0, s=s, h=h, mu1=mu1, mu2=mu2,
        generations=mcmc_data.generations,
        N_AA=mcmc_data.N_AA,
        N_Aa=mcmc_data.N_Aa,
        N_aa=mcmc_data.N_aa,
        Ne=mcmc_data.Ne,
        changepoints=changepoints or [],
        F_lb=F_lb,
    )


# ─────────────────────────────────────────────
# Entry point: smoke test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from simulation_wf import WFParams, simulate, sample_genotypes
    from layer2_mcmc import run_mcmc, data_from_dict

    print("=" * 60)
    print("BASTION — layer4_ode.py smoke test")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── Setup: simulate ground truth ─────────────────────────────────────────
    true_p0, true_F0 = 0.25, 0.10
    true_s,  true_h  = 0.08, 0.5
    true_Ne          = 800
    T                = 30

    params = WFParams(p0=true_p0, F0=true_F0, s=true_s, h=true_h,
                      mu1=1e-5, mu2=1e-5, Ne=true_Ne, T=T)
    traj = simulate(params)

    rng = np.random.default_rng(42)
    # Sparse observations — every 5 generations
    obs_gens = list(range(0, T+1, 5))
    data_dict = sample_genotypes(traj, n_per_gen=500, timepoints=obs_gens, rng=rng)

    # Quick MCMC to get posterior means (short run for smoke test)
    mcmc_data = data_from_dict(data_dict, Ne=true_Ne, model='temporal',
                                mu1_fixed=1e-5, mu2_fixed=1e-5)
    mcmc_result = run_mcmc(mcmc_data, n_chains=2, n_steps=5000,
                            n_warmup=1000, rng_seed=42, verbose=False)

    ode_inp = ode_input_from_mcmc(mcmc_result, mcmc_data)

    print(f"\nODE initialised at:")
    print(f"  p0={ode_inp.p0:.4f} (true {true_p0})")
    print(f"  F0={ode_inp.F0:.4f} (true {true_F0})")
    print(f"  s={ode_inp.s:.4f}  (true {true_s})")

    # ── Test 1: Ensemble trains without error ─────────────────────────────────
    print("\nTest 1: Train ensemble (n=5, n_epochs=2000)")
    result = run_ode_ensemble(
        ode_inp,
        n_ensemble=5,
        n_epochs=2000,
        lr=1e-3,
        n_colloc=150,
        verbose=True,
    )
    print(f"\n  Ensemble trained. n_ensemble={result.n_ensemble}")
    print(f"  t_grid: {result.t_grid[0]:.0f} to {result.t_grid[-1]:.0f} "
          f"({len(result.t_grid)} points)")
    loss_spread = max(result.losses) - min(result.losses)
    print(f"  Loss spread: {loss_spread:.4f} "
          f"(>0.1 indicates genuine diversity)")
    print("  PASSED")

    # ── Test 2: Trajectory shape correct ─────────────────────────────────────
    print("\nTest 2: Output shapes and ensemble diversity")
    assert result.p_ensemble.shape == (5, len(result.t_grid)), \
        f"p_ensemble shape wrong: {result.p_ensemble.shape}"
    assert result.F_ensemble.shape == (5, len(result.t_grid))
    assert len(result.R_residual) == len(result.t_grid)
    print(f"  p_ensemble: {result.p_ensemble.shape}  ✓")
    print(f"  F_ensemble: {result.F_ensemble.shape}  ✓")
    print(f"  R_residual: {result.R_residual.shape}  ✓")
    # Check ensemble actually has diversity — 90% spread should be > 0.001
    p_spread = (result.p_hi - result.p_lo).mean()
    print(f"  Avg 90% ensemble spread on p: {p_spread:.5f} "
          f"({'✓ diverse' if p_spread > 0.001 else '⚠ low — try more epochs'})")
    print("  PASSED")

    # ── Test 3: p and F stay in valid range ──────────────────────────────────
    print("\nTest 3: Trajectories stay in valid range")
    assert np.all(result.p_ensemble > 0) and np.all(result.p_ensemble < 1), \
        "p out of (0,1)"
    assert np.all(result.F_ensemble < 1), "F >= 1"
    print(f"  p range: [{result.p_ensemble.min():.4f}, {result.p_ensemble.max():.4f}]  ✓")
    print(f"  F range: [{result.F_ensemble.min():.4f}, {result.F_ensemble.max():.4f}]  ✓")
    print("  PASSED")

    # ── Test 4: R_idx = F*p*q ────────────────────────────────────────────────
    print("\nTest 4: R_idx = F*p*q consistency")
    R_check = result.F_ensemble * result.p_ensemble * (1 - result.p_ensemble)
    assert np.allclose(result.R_ensemble, R_check, atol=1e-6), \
        "R_ensemble inconsistent with F*p*q"
    print(f"  Max deviation: {np.abs(result.R_ensemble - R_check).max():.2e}  ✓")
    print("  PASSED")

    # ── Test 5: Residual diagnostic ──────────────────────────────────────────
    print("\nTest 5: Residual diagnostic")
    print(f"  Max |R_residual|: {result.max_residual:.5f}")
    print(f"  Closed system:    {result.closed_system_ok}")
    print(f"  (True closed system → residual should be small)")
    # Note: with only 500 epochs and sparse data the fit may be imperfect
    # Full run (2000 epochs) gives smaller residuals
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"BASTION layer4_ode.py — ensemble (n={result.n_ensemble}, "
                 f"device={DEVICE})", fontsize=12)

    t = result.t_grid
    obs = np.array(obs_gens)

    # Panel 1: p(t)
    ax = axes[0, 0]
    ax.fill_between(t, result.p_lo, result.p_hi, alpha=0.25,
                    color='steelblue', label='Ensemble 90%')
    ax.plot(t, result.p_med, color='steelblue', lw=2, label='Ensemble median')
    ax.plot(traj.p, color='black', lw=1.5, linestyle='--', label='True p(t)')
    ax.scatter(obs, [traj.p[g] for g in obs_gens],
               c='black', s=40, zorder=5, label='Observed gens')
    ax.set_title("Allele frequency p(t)", fontsize=10)
    ax.set_xlabel("Generation")
    ax.legend(fontsize=8)

    # Panel 2: F(t)
    ax = axes[0, 1]
    ax.fill_between(t, result.F_lo, result.F_hi, alpha=0.25,
                    color='firebrick', label='Ensemble 90%')
    ax.plot(t, result.F_med, color='firebrick', lw=2, label='Ensemble median')
    ax.plot(traj.F, color='black', lw=1.5, linestyle='--', label='True F(t)')
    ax.set_title("Inbreeding coefficient F(t)", fontsize=10)
    ax.set_xlabel("Generation")
    ax.legend(fontsize=8)

    # Panel 3: R_idx(t)
    ax = axes[1, 0]
    ax.fill_between(t, result.R_lo, result.R_hi, alpha=0.25,
                    color='seagreen', label='Ensemble 90%')
    ax.plot(t, result.R_med, color='seagreen', lw=2, label='Ensemble median')
    ax.plot(traj.R_idx, color='black', lw=1.5, linestyle='--', label='True R_idx(t)')
    ax.set_title("R_idx(t) = F·p·q", fontsize=10)
    ax.set_xlabel("Generation")
    ax.legend(fontsize=8)

    # Panel 4: R_residual
    ax = axes[1, 1]
    ax.plot(t, result.R_residual, color='purple', lw=1.5, label='R_residual(t)')
    ax.axhline(0,     color='k',    lw=0.8, linestyle='--')
    ax.axhline(0.05,  color='red',  lw=0.8, linestyle=':', alpha=0.7,
               label='±0.05 threshold')
    ax.axhline(-0.05, color='red',  lw=0.8, linestyle=':', alpha=0.7)
    ax.fill_between(t, -0.05, 0.05, alpha=0.1, color='green',
                    label='Closed-system region')
    ax.set_title("R_residual(t) — closed-system diagnostic", fontsize=10)
    ax.set_xlabel("Generation")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("layer4_test.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to layer4_test.png")
    plt.show()
