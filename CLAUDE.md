# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Adjoint-based inversion of the rate-and-state friction parameter `a` for a 1-D spring-slider model of afterslip. The forward model integrates a DAE (force balance + two ODEs for slip and state), and the adjoint solver computes `dJ/da` exactly, enabling gradient-based recovery of `a` from observed cumulative slip.

## Running the code

The primary workflow is through Jupyter notebooks:
- `slip_adjoint_springslider_adapttime.ipynb` — main notebook: forward solve, adjoint, gradient validation, J(a) landscape, inversion
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

**State variables:** cumulative slip `u`, state `psi`  
**Algebraic constraint:** `tau(V,psi) + eta*V + k*u = tau_L(t)` — solved for velocity `V` via `brentq` at every time step  
**Friction law:** regularised RS, `tau = N*a*arcsinh(V/(2V0) * exp(psi/a))`  
**State evolution:** slip law, `dpsi/dt = -(V/dc)*(tau/N - fss(V))`  
**Loading:** `tau_L(t) = tau0 + k*V_bg*t`

**Model parameters dict `M`:** `f0, V0, a, b, dc, N, eta, k, V_bg, tau0`

## Numerical scheme

Both the forward and adjoint solvers use the **same 3-stage embedded RK method** (2nd/3rd-order error-control pair). The adjoint integrates forwards in reversed time `τ = T - t` through the stored forward grid, linearly interpolating Jacobian coefficients `(tau_V, tau_psi, G_V, G_psi)` within each step at `α ∈ {0, ½, 1}`.

The forward solver returns a dict with keys: `t, u, psi, V, tau_L, tau_V, tau_psi, G_V, G_psi, dtau_da, dG_da`.  
The adjoint solver returns: `t, p, r, lam` where `lam = (p + G_V*r)/(tau_V+eta)`.

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

`scipy.optimize.minimize` with `method='trust-constr'` and adjoint gradient. Bounds on `a` are physical (`[0.006, 0.03]`). The `fun_and_grad` function returns `(J, [dJ/da])` together (jac=True). A time-shift inner optimisation is available (`USE_TIME_SHIFT` flag) but currently disabled.