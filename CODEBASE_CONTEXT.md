# Codebase Context

## What This Is

Adjoint-based inversion of a 1-D spring-slider afterslip model. The goal is to recover physical parameters (`a`, `k`, etc.) from observed cumulative slip `u(t)` using gradient-based optimization, where the gradient `dJ/d(param)` is computed exactly and cheaply via the continuous adjoint method.

---

## Module Dependency Graph

```
friction_derivs.py        ← physics primitives: tau, G, all partials, smoothing matrix
        ↓
adapt_fwd_solve.py        ← adaptive RK forward solver (3-stage embedded, error-controlled)
        ↓
adjoint_solve.py          ← adjoint solver (imports all three above)
        ↓
compute_obj.py            ← J evaluator + dJ/da, dJ/dk gradient functions
```

No circular imports. `%autoreload 2` in the notebooks so `.py` edits are picked up live.

---

## Key Files

| File | Role |
|------|------|
| `friction_derivs.py` | Rate-and-state friction law (`tau_fn`), its partials (`tau_V_fn`, `tau_psi_fn`, etc.), state evolution `G_fn` and its partials, force-balance root-find (`solve_V_algebraic`), smoothing matrix (`make_smoothing_matrix`) |
| `adapt_fwd_solve.py` | Adaptive-timestep 3-stage RK forward solver. Returns dict: `t, u, psi, V, tau_L, tau_V, tau_psi, G_V, G_psi, dtau_da, dG_da` |
| `adjoint_solve.py` | Same RK scheme, integrated backwards in reversed time `τ = T − t`. Returns `t, p, r, lam` |
| `compute_obj.py` | `compute_J`, `compute_grad_a`, `compute_grad_k` |
| `slip_adjoint_springslider_adapttime.ipynb` | Main notebook: forward solve → adjoint → FD validation → J(param) landscape → inversion |
| `adjoint_springslider.ipynb` | Earlier development notebook |
| `visualize_objective.ipynb` | Objective function visualization |

---

## Physics (DAE System)

**State variables:** cumulative slip `u`, RS state `θ` (called `psi` in code)

**Algebraic constraint** (solved for `V` at every time step via `brentq`):
```
τ(V, ψ) + η·V + k·u = τ_L(t)
τ_L(t) = τ₀ + k·V_bg·t
```

**Friction law** (regularised rate-and-state):
```
τ = N·a·arcsinh(V/(2V₀) · exp(ψ/a))
```

**State evolution** (slip law):
```
dψ/dt = −(V/dc)·(τ/N − f_ss(V))
f_ss(V) = f₀ + (a−b)·ln(V/V₀)
```

**Loading:** `τ_L(t) = τ₀ + k·V_bg·t`

**Parameter dict `M`:** `f0, V0, a, b, dc, N, eta, k, V_bg, tau0`

**Nominal values:**
- `N = 50 MPa`, `a = 0.010`, `b = 0.015` (velocity-strengthening: `a > b`)
- `k = 0.9 · k_crit` where `k_crit = N(b−a)/dc`
- `V_bg = 1e-9 m/s`, `V_init = 1e-12 m/s` (post-seismic near-zero)

---

## Numerical Scheme

**Forward solver** (`adapt_fwd_solve.py`): Bogacki–Shampine 3-stage embedded RK (2nd/3rd-order error-control pair). Adaptive step-size via local truncation error estimate.

**Adjoint solver** (`adjoint_solve.py`): Same 3-stage scheme, reversed time. Forward-state Jacobians (`tau_V, tau_psi, G_V, G_psi`) linearly interpolated at `α ∈ {0, ½, 1}` within each step.

**Adjoint equations** (co-states `p` for `u`, `r` for `ψ`):
```
dp/dt = −k·λ − sm
dr/dt = −τ_ψ·λ + G_ψ·r
λ = (p + G_V·r) / (τ_V + η)       ← Lagrange multiplier for force-balance
```
where `sm = S^T(Su − Su_obs)` is the smoothed misfit source.

**Gradient formulas:**
```
dJ/da = ∫ [λ·∂τ/∂a  −  r·∂G/∂a] dt        (compute_grad_a)
dJ/dk = ∫ λ·(u − V_bg·t) dt                 (compute_grad_k)
```

---

## Objective Function & Smoothing

```
J = 0.5 · ∫₀ᵀ (Su − Su_obs)² dt
```

`S` = row-normalised Gaussian with trapezoidal integration weights (for non-uniform adaptive grids). `sigma` controls smoothing width.

**Critical rule:** `sigma = 0` (identity S) makes the `J(k)` landscape non-convex — the adjoint gradient for `k` has the wrong sign in regions of parameter space, causing gradient descent to diverge. Always use `sigma_smooth = 0.01 * T` (or larger) for k-inversion.

**Fixed reference grid pattern** (used in inversion cell):
1. Build `t_ref` once (uniform, at initial parameter values)
2. Fix `S_fixed = make_smoothing_matrix(t_ref, sigma_smooth)` for the entire inversion
3. Interpolate each forward `u` onto `t_ref` before evaluating `J` and the adjoint source
4. Adjoint itself runs on the native adaptive grid (full time resolution)

This eliminates the spurious grid-variation term that FD gradients would otherwise pick up.

---

## Inversion Setup (notebook cell 20)

- `INVERT_PARAMS`: list of params to invert; supports `'a'`, `'k'`, or both
- `sigma_smooth = 0.01 * T`: smoothing width (must be non-zero for k-inversion)
- Optimizer: `scipy.optimize.minimize` with `method='trust-constr'`
- Gradient: `jac=True` (function returns `(J, grad_vec)`)
- Bounds: physical (`a ∈ [0.006, 0.03]`, `k ∈ [0.5·k_crit, 1.3·k_crit]`)

**Printed per eval:** `J`, parameter value + % error, `dJ/d{param}` for each active param, `Δt` (time-shift if enabled).

---

## Gradient Validation

FD check in cell 15 (`CHECK_PARAM = 'k'` or `'a'`):
- Uses Forward Euler + adaptive RK forward solves
- Fixed reference grid `t_ref_fd` + `S_fixed_fd` for both FD and adjoint (eliminates grid-variation artefact)
- Passes criterion: `|adj_grad − fd_grad| / |fd_grad| < 5%`
- Uses `sigma_test = 0.1 * T_test` (non-zero — required for agreement)

---

## Known Issues / Design Decisions

- **sigma=0 and k-inversion**: Non-convex landscape. Always use `sigma_smooth ≥ 0.01·T` when inverting for `k`.
- **Smoothing matrix conventions**: `make_smoothing_matrix` uses trapezoidal weights (correct for non-uniform adaptive grids). `fe_gaussian_S` in the FD validation cell uses a plain row-normalised Gaussian (correct for the uniform FE grid). Do not mix.
- **tau0 derivation**: `tau0` is DERIVED from the force balance at `t=0` given `V_init`, not set independently. `setup_initial_conditions(M)` returns `(u0, psi0, V_init)` and sets `M['tau0']` as a side effect.
- **USE_TIME_SHIFT flag**: Inner optimisation over a time shift `Δt` — currently disabled (`False`).
