# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Adjoint-based inversion of rate-and-state friction parameters for spring-slider models of afterslip. Two configurations are supported:

- **Single-block**: invert for `a` (and optionally `k`). One spring couples the block to the loading plate.
- **Two-block (symmetric, Abe & Kato 2013)**: invert for any subset of `{a1, a2, k0, k12}`. Topology: `Plate ←(k0)→ Block 1 ←(k12)→ Block 2 ←(k0)→ Plate`.

The forward model integrates a DAE (force balance + two ODEs per block), and the adjoint solver computes exact gradients `dJ/d(params)`.

## Running the code

The primary workflow is through Jupyter notebooks:
- `slip_adjoint_springslider_adapttime.ipynb` — single-block: forward solve, adjoint, gradient validation, J(a) landscape, inversion
- `slip_adjoint_double_springslider.ipynb` — two-block: forward solve, adjoint, gradient validation, inversion for a1/a2/k0/k12 (in progress)
- `adjoint_springslider.ipynb` — earlier development notebook
- `visualize_objective.ipynb` — objective function visualization

The Python modules use `%autoreload 2` in the notebooks, so edits to `.py` files are picked up without restarting the kernel.

No test runner is configured. Gradient correctness is validated inline via finite-difference checks in the notebooks. The key check: `rel_err = |adj_grad - fd_grad| / |fd_grad| < 5%`.

## Module dependency order

```
friction_derivs.py   ← physics primitives (no imports from project)
      ↓
adapt_fwd_solve.py   ← adaptive RK forward solver
      ↓
adjoint_solve.py     ← adjoint solver (imports all three above)
      ↓
compute_obj.py       ← J and dJ/da (imports all four above)
```

## Physics

**Friction law (both blocks):** regularised RS, `tau = N*a*arcsinh(V/(2V0) * exp(psi/a))`  
**State evolution:** aging law (Dieterich), `dpsi/dt = (b*V0/dc)*exp(-(psi-f0)/b) - b*V/dc`  
where `psi = f0 + b*ln(theta*V0/dc)` and theta is the Dieterich state variable.

### Single-block
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

Both forward and adjoint solvers use the **same 3-stage embedded RK method** (2nd/3rd-order error-control pair). The adjoint integrates forwards in reversed time `τ = T - t` through the stored forward grid, linearly interpolating Jacobian coefficients at `α ∈ {0, ½, 1}`.

**Single-block forward** returns: `t, u, psi, V, tau_L, tau_V, tau_psi, G_V, G_psi, dtau_da, dG_da`  
**Single-block adjoint** returns: `t, p, r, lam` where `lam = (p + G_V*r)/(tau_V+eta)`

**Two-block forward** returns: `t, u1, psi1, V1, u2, psi2, V2, tau_L1, tau_L2` plus per-block Jacobians `tau_V1/2, tau_psi1/2, G_V1/2, G_psi1/2, dtau_da1/2, dG_da1/2`  
**Two-block adjoint** returns: `t, pu1, r1, lam1, pu2, r2, lam2`

## Two-block adjoint equations (reversed time τ = T - t)

```
lam1 = (pu1 + G_V1*r1) / (tau_V1 + eta)
lam2 = (pu2 + G_V2*r2) / (tau_V2 + eta)

dpu1/dτ = -(k0+k12)*lam1 + k12*lam2 - sm1
dr1/dτ  = -tau_psi1*lam1 + G_psi1*r1
dpu2/dτ = +k12*lam1 - (k0+k12)*lam2 - sm2
dr2/dτ  = -tau_psi2*lam2 + G_psi2*r2
```
IC: all zero at τ=0 (t=T). Note the **minus sign** on the smoothed-misfit source terms (`-sm`), matching the single-block convention.

## Gradient formulas

```
dJ/da1 = ∫ [-lam1*dtau_da1] dt          (dG/da1 = 0 for aging law)
dJ/da2 = ∫ [-lam2*dtau_da2] dt          (dG/da2 = 0 for aging law)
dJ/dk0 = ∫ [-lam1*(u1 - V_bg*t) - lam2*(u2 - V_bg*t)] dt
dJ/dk12 = ∫ [(lam2 - lam1)*(u1 - u2)] dt
```

## Aging law Jacobians

```
G_V   = -b/dc                              (constant, no V or psi dependence)
G_psi = -(V0/dc) * exp(-(psi-f0)/b)
dG/da = 0                                  (G does not depend on a)
```

## Smoothing matrix

`make_smoothing_matrix(t, sigma)` in `friction_derivs.py` builds a row-normalised Gaussian `S` with **trapezoidal integration weights** on each column. This corrects for non-uniform node spacing on adaptive grids — without it, dense intervals are over-weighted and `S` no longer represents a fixed-time-window average.

For uniform (Forward Euler) grids, a plain row-normalised Gaussian (no weights) is correct. These two conventions must not be mixed when comparing FE and adaptive results.

## Gradient consistency requirement

Because `S` depends on the forward time grid (which changes with `a`), finite-difference gradients computed with per-solve `S` matrices contain a **spurious grid-variation contribution** that the adjoint does not compute. The correct approach — used in the inversion cell — is:

1. Fix a **reference time grid** `t_ref` (built once at `a_init`, or a uniform linspace)
2. Fix `S_fixed = make_smoothing_matrix(t_ref, sigma)` for the entire inversion
3. Interpolate every forward solve onto `t_ref` before evaluating `J` and the adjoint

This makes `J` and `dJ/da` consistent, and FD vs adjoint gradients agree to < 1%.

## Inversion setup

**Single-block:** `scipy.optimize.minimize` with `method='trust-constr'` and adjoint gradient. Bounds on `a` are physical (`[0.006, 0.03]`). The `fun_and_grad` function returns `(J, [dJ/da])` together (jac=True). A time-shift inner optimisation is available (`USE_TIME_SHIFT` flag) but currently disabled.

**Two-block (planned):** `INVERT_PARAMS` list controls which subset of `['a1', 'a2', 'k0', 'k12']` are optimised. `fun_and_grad` recomputes `tau0_1`, `tau0_2` via `setup_initial_conditions_2block` whenever `a1` or `a2` change. Uses fixed reference grid + `S_fixed` strategy (same as single-block).

## Known issues / pending work

- **Two-block gradient check**: The adjoint gradients for `a1`/`a2`/`k0`/`k12` have not yet been re-verified against finite differences after the topology change (series → symmetric) and state law change (slip → aging). The a-gradient check requires the fixed reference grid + `S_fixed` approach (see "Gradient consistency requirement"), and `tau0_1`/`tau0_2` must be recomputed consistently in both the adjoint and the FD perturbation.
- **Single-block gradient check**: The aging law changes `G_fn`, `G_V_fn`, `G_psi_fn`, and `dG_da_fn`. The single-block adjoint and gradient validation cells in `slip_adjoint_springslider_adapttime.ipynb` should be re-run to confirm < 5% relative error.
- **`slip_adjoint_double_springslider.ipynb`**: Two-block setup cells updated; full gradient validation and inversion cells still need to be implemented for the two-block case.

## Github

Make sure to commit and push changes to github when completed.