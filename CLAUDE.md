# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Gradient-based inversion of rate-and-state friction parameters for spring-slider models of afterslip. Two configurations are supported:

- **Single-block**: invert for `a` (and optionally `k`). One spring couples the block to the loading plate. Still uses the continuous adjoint (legacy).
- **Two-block (symmetric, Abe & Kato 2013)**: invert for any subset of `{a1, a2, k0, k12}`. Topology: `Plate ←(k0)→ Block 1 ←(k12)→ Block 2 ←(k0)→ Plate`. **Two-block now uses forward sensitivity** (Cao, Li, Petzold 2003), not the adjoint.

The forward model integrates a DAE (force balance + two ODEs per block). For the two-block case, forward sensitivity equations `ds_x/dp = ∂x/∂p` are integrated alongside the nominal state on the same adaptive grid, and `dJ/dp` is computed from a single combined pass.

## Why forward sensitivity (two-block)

The continuous adjoint with an adaptively-stepped forward solver became **dual-inconsistent** during fast slip events: the adjoint integrates against forward-grid-interpolated Jacobians whose denominator `tau_V + eta` pinches near rupture, producing spurious blowup (the explicit RK3 adjoint and a Radau adjoint agreed with each other but disagreed with FD by many orders of magnitude on long horizons). See Alexe & Sandu, *J. Comput. Appl. Math.* 233 (2009) for the general phenomenon.

Forward sensitivity sidesteps this entirely:
- integrated on the same adaptive grid as the nominal state (no separate interpolation),
- no reverse-time integration through near-singular regions,
- produces the *exact* gradient of the discretised `J` by construction.

With 4 parameters and 1 scalar output `J`, forward sensitivity costs ~5× a forward solve — well-justified.

## Running the code

The primary workflow is through Jupyter notebooks:
- `slip_adjoint_double_springslider.ipynb` — two-block: forward+sensitivity solve, gradient validation (FS vs FD), J landscape, inversion for `a1`/`a2`/`k0`/`k12` via forward sensitivity
- `slip_adjoint_springslider_adapttime.ipynb` — single-block (legacy, still uses adjoint)
- `adjoint_springslider.ipynb` — earlier single-block dev notebook
- `visualize_objective.ipynb` — objective function visualization

The Python modules use `%autoreload 2` in the notebooks.

No test runner. Gradient correctness is validated inline via FD checks in the notebooks. The key check: `rel_err = |fs_grad - fd_grad| / |fd_grad| < 5%`.

## Module dependency order

```
friction_derivs.py   ← physics primitives (no imports from project)
      ↓
adapt_fwd_solve.py   ← adaptive RK forward solver (nominal + sensitivity)
      ↓
adjoint_solve.py     ← single-block adjoint (legacy)
      ↓
compute_obj.py       ← J and dJ/dp (forward-sensitivity gradient for two-block)
```

`landscape_worker.py` is a process-pool worker for the J-landscape scan; it uses the sensitivity solver when `COMPUTE_GRADIENT=True`.

## Physics

**Friction law (both blocks):** regularised RS, `tau = N*a*arcsinh(V/(2V0) * exp(psi/a))`
**State evolution:** aging law (Dieterich), `dpsi/dt = (b*V0/dc)*exp(-(psi-f0)/b) - b*V/dc`
where `psi = f0 + b*ln(theta*V0/dc)` and theta is the Dieterich state variable.

### Single-block (legacy)
**Algebraic constraint:** `tau(V,psi) + eta*V + k*u = tau_L(t)`, solved for `V` via `brentq`
**Loading:** `tau_L(t) = tau0 + k*V_bg*t`
**`M` keys:** `f0, V0, a, b, dc, N, eta, k, V_bg, tau0`

### Two-block (symmetric, Abe & Kato 2013)
**Force balances** (each solved independently for V_i via `brentq`):
```
Block 1: tau1 + eta*V1 + (k0+k12)*u1 - k12*u2 = tau0_1 + k0*V_bg*t
Block 2: tau2 + eta*V2 + (k0+k12)*u2 - k12*u1 = tau0_2 + k0*V_bg*t
```
Both blocks are independently loaded by the plate via `k0`; `k12` is the coupling spring.
**`M` keys:** `f0, V0, a1, a2, b, dc, N, eta, k0, k12, V_bg, tau0_1, tau0_2`
`tau0_1` and `tau0_2` are computed by `setup_initial_conditions_2block(M)` in `friction_derivs.py`.

## Numerical scheme

The forward solver uses a 3-stage embedded RK method (2nd/3rd-order error-control pair).

**`forward_solve_adaptive_2block(M, T, u1_0, psi1_0, u2_0, psi2_0, ...)`** returns:
`t, u1, psi1, V1, u2, psi2, V2, tau_L1, tau_L2` plus per-block Jacobians.

**`forward_solve_adaptive_2block_sens(M, T, u1_0, psi1_0, u2_0, psi2_0, params=('a1','a2','k0','k12'), ...)`** returns the same `t/u/psi/V/tau_L` plus `sens[p] = {'s_u1','s_psi1','s_u2','s_psi2'}` for each parameter `p`. The error controller monitors only the nominal state, so the adaptive grid is identical to the nominal-only solver at the same tolerance.

## Forward sensitivity equations (two-block, aging law)

For each parameter `p`, with `s_x = ∂x/∂p`:

**Algebraic sensitivity** (from differentiating the force balance):
```
sigma_V_i = dV_i/dp = -[dF_i/dp + (k0+k12)*s_u_i - k12*s_u_j + tau_psi_i*s_psi_i] / (tau_V_i + eta)
```

**Sensitivity ODEs** (aging law: explicit ∂G/∂p = 0 for all four parameters):
```
ds_u_i/dt   = sigma_V_i
ds_psi_i/dt = G_V_i * sigma_V_i + G_psi_i * s_psi_i
```

**Explicit ∂F/∂p** (with `tau0_i` treated as independent — frozen-IC convention):
```
p='a1':  dF1/da1 = dtau_da1,        dF2/da1 = 0
p='a2':  dF1/da2 = 0,                dF2/da2 = dtau_da2
p='k0':  dF1/dk0 = u1 - V_bg*t,      dF2/dk0 = u2 - V_bg*t
p='k12': dF1/dk12 = u1 - u2,         dF2/dk12 = u2 - u1
```

## Gradient formula

```
dJ/dp = sum_{i=1,2} int_{t_ref} (S*u_i - S*u_{i,obs}) * (S * (du_i/dp)_ref) dt_ref
```
where `(du_i/dp)_ref = np.interp(t_ref, t_fwd, s_u_i)`. Implemented in `compute_grad_forward_sens_2block`.

## Aging law Jacobians

```
G_V   = -b/dc                              (constant)
G_psi = -(V0/dc) * exp(-(psi-f0)/b)
dG/da = dG/dk0 = dG/dk12 = 0               (G does not depend on a, k0, k12)
```

## Smoothing matrix

`make_smoothing_matrix(t, sigma)` in `friction_derivs.py` builds a row-normalised Gaussian `S` with **trapezoidal integration weights** on each column. This corrects for non-uniform node spacing on adaptive grids.

For uniform grids the weights are constant and cancel, recovering the standard un-weighted Gaussian.

## Gradient consistency requirement

Both `J` and `dJ/dp` are evaluated on a **fixed reference grid `t_ref`** (built once at the initial guess, uniform). Forward solutions and their sensitivities are interpolated from the native adaptive grid onto `t_ref` before assembling `J` and the gradient. This makes the computed gradient self-consistent with the `J` the optimiser sees, and FD vs forward-sensitivity gradients agree to within numerical noise.

## Inversion setup

**Two-block (`slip_adjoint_double_springslider.ipynb`):**
- `INVERT_PARAMS` controls which subset of `['a1','a2','k0','k12']` is optimised.
- `fun_and_grad(x_norm)` runs one `forward_solve_adaptive_2block_sens` per evaluation (sensitivities for exactly `INVERT_PARAMS`), then `compute_grad_forward_sens_2block` extracts the gradient.
- Parameters are normalised by their initial values (`scales`) so all components are O(1) inside the optimiser.
- Optimiser: `scipy.optimize.minimize` with `method='trust-constr'` (or `'L-BFGS-B'`), `jac=True`, physical bounds.
- IC (`u*_0_inv`, `psi*_0_inv`) is built once from the initial guess `_M0` and frozen; `Mc = dict(M_true)` carries `tau0_*` through unchanged. This matches the frozen-IC convention the sensitivity equations are derived under. Recomputing IC per iterate would re-introduce an implicit `a`-dependence in `psi_ss(a)` and `tau0(a)` that the sensitivities (as derived) do not track, biasing the gradient.

## Known issues / pending work

- **Single-block notebook** still uses the continuous adjoint and has not been migrated to forward sensitivity. The same migration would apply if needed.

## Github

Make sure to commit and push changes to github when completed.
