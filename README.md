# spring-slider-sensitivity

Gradient-based inversion for rate-and-state friction parameters in 1-D and 2-block spring-slider models of afterslip. Gradients are computed via **forward sensitivity equations** (Cao, Li, Petzold 2003).

## Overview

The forward model integrates a DAE (force balance + ODEs for slip `u` and state `psi`) using an adaptive 3-stage embedded RK solver. For the two-block case, **forward sensitivity equations** `ds_x/dp = ∂x/∂p` are integrated alongside the nominal state on the same adaptive grid, giving exact gradients `dJ/dp` for parameters `p ∈ {a1, a2, k0, k12}`.

**Objective:** `J = 0.5 * ∫(Su - Su_obs)² dt` where `S` is a Gaussian smoothing operator on a fixed reference grid.

**Inversion:** `scipy.optimize.minimize` with `method='trust-constr'` (or L-BFGS-B), forward-sensitivity gradient, physical bounds.

## Why forward sensitivity?

An earlier version used the continuous adjoint. With adaptive time stepping and fast slip events (ruptures), the continuous adjoint becomes dual-inconsistent (Alexe & Sandu 2009): it integrates against forward-grid-interpolated Jacobians whose denominator `tau_V + eta` pinches near rupture, producing spurious blowup. Two independent adjoint solvers (explicit RK3 and implicit Radau) agreed with each other but disagreed with FD by many orders of magnitude on long horizons.

Forward sensitivity:
- runs on the same adaptive grid as the nominal state,
- has no reverse-time integration through near-singular regions,
- produces the *exact* gradient of the discretised `J`.

With only 4 parameters the cost is modest (~5× a forward solve).

## Physics

- **Friction law:** regularised rate-and-state, `τ = N·a·arcsinh(V/(2V₀) · exp(ψ/a))`
- **State evolution:** aging law (Dieterich), `dψ/dt = (b·V₀/dc)·exp(-(ψ-f0)/b) - b·V/dc`
- **Two-block force balances** (Abe & Kato 2013 topology — Plate ↔ k0 ↔ Block1 ↔ k12 ↔ Block2 ↔ k0 ↔ Plate):
  ```
  τ₁ + η·V₁ + (k0+k12)·u₁ - k12·u₂ = τ₀,₁ + k0·V_bg·t
  τ₂ + η·V₂ + (k0+k12)·u₂ - k12·u₁ = τ₀,₂ + k0·V_bg·t
  ```

## Module structure

```
friction_derivs.py   ← physics primitives, smoothing matrix, IC setup
adapt_fwd_solve.py   ← adaptive RK forward solver
                       • forward_solve_adaptive_2block (nominal only)
                       • forward_solve_adaptive_2block_sens (nominal + sensitivity)
adjoint_solve.py     ← single-block adjoint (legacy)
compute_obj.py       ← J and dJ/dp via forward sensitivity
landscape_worker.py  ← process-pool worker for J landscape scan
```

## Usage

Open `slip_adjoint_double_springslider.ipynb` in Jupyter. The notebook runs:
1. Forward + sensitivity solve at the production T (with sensitivity time-series plots)
2. Gradient validation: forward-sensitivity vs centred finite differences, all 4 parameters
3. J landscape scan (optional gradient arrows)
4. Inversion via `scipy.optimize.minimize` with the forward-sensitivity gradient
5. Convergence plots and animated trajectory viewer

Edits to `.py` modules are picked up automatically via `%autoreload 2`.

## Key implementation notes

- The smoothing matrix `S` uses trapezoidal integration weights on each column for non-uniform grids; for the inversion it is built once on a fixed uniform reference grid `t_ref` and reused for every iterate.
- Forward solutions and sensitivities are interpolated from the native adaptive grid onto `t_ref` before assembling `J` and the gradient, so the gradient is exactly self-consistent with the `J` the optimiser sees.
- Initial conditions (`u_0`, `psi_0`, `tau0_1`, `tau0_2`) are **frozen** across iterates — recomputing them per iterate introduces implicit `a`-dependence in `psi_ss(a)`, `tau0(a)` that the sensitivities (as derived) do not track.

## References

- Cao, Y., Li, S., Petzold, L., & Serban, R. (2003). Adjoint sensitivity analysis for differential-algebraic equations: The adjoint DAE system and its numerical solution. *SIAM J. Sci. Comput.* 24, 1076–1089.
- Alexe, M. & Sandu, A. (2009). On the discrete adjoints of adaptive time stepping algorithms. *J. Comput. Appl. Math.* 233, 1005–1020.
- Abe, Y. & Kato, N. (2013). Complex earthquake cycle simulations using a two-degree-of-freedom spring-block model with a rate- and state-friction law. *Pure Appl. Geophys.* 170, 745–765.

## Dependencies

Python 3, NumPy, SciPy, Matplotlib, Jupyter (with ipywidgets for interactive plots), optional ffmpeg for saving animations.
