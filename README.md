# spring-slider-adjoint

Adjoint-based inversion for the rate-and-state friction parameter `a` using a 1-D spring-slider model of afterslip.

## Overview

The forward model integrates a DAE system (force balance + ODEs for slip `u` and state `psi`) using an adaptive 3-stage embedded RK solver. The adjoint solver computes `dJ/da` exactly, enabling gradient-based recovery of `a` from observed cumulative slip.

**Objective:** `J = 0.5 * ∫(Su - Su_obs)² dt` where `S` is a Gaussian smoothing operator.

**Inversion:** `scipy.optimize.minimize` with `method='trust-constr'` and adjoint gradient, bounds `a ∈ [0.006, 0.03]`.

## Physics

- **Friction law:** regularised rate-and-state, `τ = N·a·arcsinh(V/(2V₀) · exp(ψ/a))`
- **State evolution:** slip law, `dψ/dt = -(V/dc)·(τ/N - fss(V))`
- **Constraint:** `τ(V,ψ) + η·V + k·u = τ_L(t)` — solved for `V` via Brent's method
- **Loading:** `τ_L(t) = τ₀ + k·V_bg·t`

## Module structure

```
friction_derivs.py   ← physics primitives
adapt_fwd_solve.py   ← adaptive RK forward solver
adjoint_solve.py     ← adjoint solver
compute_obj.py       ← J and dJ/da
```

## Usage

Open `slip_adjoint_springslider_adapttime.ipynb` in Jupyter. The notebook runs the forward solve, adjoint, gradient validation (finite-difference check, target `< 1%` relative error), and inversion.

Edits to `.py` modules are picked up automatically via `%autoreload 2`.

## Key implementation notes

- The smoothing matrix `S` uses trapezoidal integration weights to handle non-uniform adaptive grids.
- For gradient consistency, `S` is fixed to a reference time grid `t_ref` for the entire inversion — this prevents spurious grid-variation contributions from appearing in finite-difference checks.

## Dependencies

Python 3, NumPy, SciPy, Matplotlib, Jupyter.
