# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Gradient-based inversion of rate-and-state friction parameters for spring-slider models of afterslip. Two configurations are supported:

- **Single-block**: invert for `a` (and optionally `k`). One spring couples the block to the loading plate. Still uses the continuous adjoint (legacy).
- **Two-block (symmetric, Abe & Kato 2013)**: invert for any subset of `{a1, a2, k0, k12}`. Topology: `Plate ←(k0)→ Block 1 ←(k12)→ Block 2 ←(k0)→ Plate`. Two gradient methods are now supported:
  1. **Forward sensitivity** (Cao, Li, Petzold 2003) — original numpy-based implementation.
  2. **Discrete adjoint via AD** (JAX + Diffrax + Optimistix) — current preferred method. Backpropagates through the actual adaptive ODE stepper using `RecursiveCheckpointAdjoint`; scales independently of parameter count.

The forward model integrates a DAE (force balance + two ODEs per block). The two-block forward-sensitivity path integrates `ds_x/dp = ∂x/∂p` alongside the nominal state on the same adaptive grid. The discrete-adjoint path eliminates `V` via a differentiable root-find (Optimistix Newton in `log V`) inside the ODE RHS and reverse-mode-differentiates through Diffrax's `Tsit5`/`Dopri8` stepper.

## Why not the continuous adjoint (two-block)

The continuous adjoint with an adaptively-stepped forward solver became **dual-inconsistent** during fast slip events: the adjoint integrates against forward-grid-interpolated Jacobians whose denominator `tau_V + eta` pinches near rupture, producing spurious blowup (the explicit RK3 adjoint and a Radau adjoint agreed with each other but disagreed with FD by many orders of magnitude on long horizons). See Alexe & Sandu, *J. Comput. Appl. Math.* 233 (2009) for the general phenomenon.

Both replacement methods sidestep this. They differ in scaling:

- **Forward sensitivity** integrates `s_x = ∂x/∂p` on the same adaptive grid as the nominal state. No reverse-time integration. Produces the exact gradient of the discretised `J`. Cost ≈ `(1 + n_params) ×` forward solve — fine for the 4-parameter problem.
- **Discrete adjoint via AD** (JAX/Diffrax `RecursiveCheckpointAdjoint`) reverse-mode-differentiates through the actual time-stepping algorithm with logarithmic checkpointing. Cost ≈ `O(1) ×` forward solve regardless of parameter count, which matters once we want to invert for more than 4 parameters. Diffrax explicitly recommends it over `BacksolveAdjoint` (the continuous adjoint) for stiff/near-stiff systems — exactly the failure mode we hit.

## Running the code

The primary workflow is through Jupyter notebooks:
- `slip_discrete_adjoint_double_springslider.ipynb` — **two-block, discrete adjoint via JAX/Diffrax AD** (current preferred path): JAX rewrite of the forward model, differentiable root-find for `V`, `RecursiveCheckpointAdjoint` backprop, gradient validation (AD vs FD step-size sweep), inversion via `scipy.optimize.minimize` / `basinhopping` with `jax.value_and_grad`.
- `slip_sensitivity_double_springslider.ipynb` / `slip_adjoint_double_springslider.ipynb` — two-block, numpy forward-sensitivity path: gradient validation (FS vs FD), J landscape, inversion for `a1`/`a2`/`k0`/`k12`.
- `slip_adjoint_springslider_adapttime.ipynb` — single-block (legacy, still uses continuous adjoint)
- `visualize_objective.ipynb` — objective function visualization

The Python modules use `%autoreload 2` in the notebooks. The discrete-adjoint notebook additionally requires `jax`, `diffrax`, and `optimistix` (with `jax_enable_x64=True`).

No test runner. Gradient correctness is validated inline via FD checks in the notebooks. Targets: `rel_err < 5%` for the forward-sensitivity path (FD-limited), and `rel_err ≲ 1e-4` for the discrete-adjoint path with a centred-FD step-size sweep used to locate the FD noise/bias plateau.

## Module dependency order

```
friction_derivs.py   ← physics primitives, IC setup, smoothing matrix (no imports from project)
      ↓
adapt_fwd_solve.py   ← adaptive RK forward solver (nominal + forward sensitivity)
      ↓
adjoint_solve.py     ← single-block continuous adjoint (legacy)
      ↓
compute_obj.py       ← J and dJ/dp (forward-sensitivity gradient for two-block)
```

`landscape_worker.py` is a process-pool worker for the J-landscape scan; it uses the sensitivity solver when `COMPUTE_GRADIENT=True`.

The **discrete-adjoint notebook is self-contained**: it re-implements the two-block forward model in JAX (rather than importing from `adapt_fwd_solve.py`) so AD can trace it end-to-end. It still imports `setup_initial_conditions_2block` and `make_smoothing_matrix` from `friction_derivs.py`, and uses `forward_solve_adaptive_2block` from `adapt_fwd_solve.py` only as a numpy reference for sanity-checking the JAX forward.

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

**Per-block friction parameters.** Each block carries its own `a, N, b, dc, f0` — keys `a1/a2`, `N1/N2`, `b1/b2`, `dc1/dc2`, `f0_1/f0_2`. The helper `block_M(M, i)` (defined in `friction_derivs.py`, re-exported by `adapt_fwd_solve.py`) builds a per-block scalar dict by picking the suffixed key when present and falling back to the shared name (`a`, `N`, `b`, `dc`, `f0`) otherwise — so legacy callers that set only shared values keep working unchanged.

**`M` keys:** shared — `V0, eta, k0, k12, V_bg, tau0_1, tau0_2`; per-block — `a1, a2, N1, N2, b1, b2, dc1, dc2, f0_1, f0_2` (or shared-name fallbacks). `tau0_1` and `tau0_2` are computed by `setup_initial_conditions_2block(M)` in `friction_derivs.py` from each block's own friction parameters.

**Sensitivity/AD scope.** The forward-sensitivity equations are derived only for `{a1, a2, k0, k12}`; `N_i, b_i, dc_i, f0_i` are accommodated as configurable per-block constants but are not currently invertable through the numpy sensitivity path. The JAX/Diffrax discrete-adjoint path captures the per-block constants by closure — they are inputs to the differentiable forward but are not part of `p_vec`, so AD treats them as fixed. Adding them to `p_vec` would just require widening the closure-captured tuple.

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

## Discrete adjoint via JAX/Diffrax (two-block)

`slip_discrete_adjoint_double_springslider.ipynb` is the JAX/Diffrax rewrite. Key implementation choices:

1. **Algebraic constraint** `tau(V,psi) + eta*V = rhs` is solved via `optimistix.Newton` in `log V` (log-space keeps Newton stable across the 12+ orders of magnitude `V` traverses through rupture). Optimistix is a differentiable implicit-layer library — AD propagates through the root-find via the implicit function theorem, so backprop "just works" and is taken at a genuine root.
   - Initial guess from the friction-dominated approximation (ignoring `eta*V`): `logV0 = log(V0) - psi/a + rhs/(N*a)`. Lands within O(1) in log space.
   - Tolerances `rtol=1e-12, atol=1e-13`; `throw=True` so silent non-convergence raises.
2. **Vector field** is the 4-D ODE in `(u1, psi1, u2, psi2)` with `V_i` eliminated inside the RHS by `solve_V`. Diffrax sees only the ODE.
3. **Time stepping**: `diffrax.Dopri8` (8th-order explicit RK with PI step-size control), `rtol=1e-11`, `atol=1e-13`, `max_steps=500_000`. The high-order solver shrinks the discretisation noise of `J(p)` (making FD a clean reference) and reduces the step-grid wobble backprop must propagate through. `Tsit5` also works; `Dopri8` is the current default.
4. **Saving**: `SaveAt(ts=t_ref)` returns the state directly on the fixed reference grid `J` uses — no post-hoc interpolation needed.
5. **Adjoint**: `diffrax.RecursiveCheckpointAdjoint()` — reverse-mode AD through the actual stepper with logarithmic checkpointing. Recommended over `BacksolveAdjoint` for stiff/near-stiff systems.
6. **Objective**: `J_fn(p)` builds residuals `S @ u_i - Su_i_obs` and integrates `0.5 * (r1² + r2²)` via `jnp.trapezoid` on `t_ref`. `J_and_grad = jax.jit(jax.value_and_grad(J_fn))`.

**Gradient validation cell** does a centred-FD sweep across `eps_rel ∈ logspace(-2, -10)` to locate the noise/bias plateau, then picks per-parameter best `eps` and reports `|AD - FD| / |FD|`. The plot of FD vs `eps` exposes the U-shape (truncation bias at large `eps`, round-off noise at small `eps`) directly.

**`jax_enable_x64=True` is required** — double precision throughout; `arcsinh` arguments reach ~1e35 during rupture and float32 would be useless.

## Inversion setup

**Two-block forward-sensitivity (`slip_adjoint_double_springslider.ipynb` / `slip_sensitivity_double_springslider.ipynb`):**
- `INVERT_PARAMS` controls which subset of `['a1','a2','k0','k12']` is optimised.
- `fun_and_grad(x_norm)` runs one `forward_solve_adaptive_2block_sens` per evaluation (sensitivities for exactly `INVERT_PARAMS`), then `compute_grad_forward_sens_2block` extracts the gradient.

**Two-block discrete-adjoint (`slip_discrete_adjoint_double_springslider.ipynb`):**
- Same `INVERT_PARAMS` convention. `fun_and_grad(x_norm)` calls the cached `J_and_grad(p_vec)` once per evaluation; the active-parameter gradient is sliced out of the full 4-component `jax.grad` result.
- Optimiser: `scipy.optimize.minimize(method='trust-constr', jac=True, ...)` or `scipy.optimize.basinhopping` wrapping the same `trust-constr` local step for global search.

**Shared conventions:**
- Parameters are normalised by their initial values (`scales`) so all components are O(1) inside the optimiser.
- IC (`u*_0_inv`, `psi*_0_inv`, `tau0_*`) is built once from the initial guess and frozen across iterates. This matches the frozen-IC convention the sensitivities are derived under, and is required for the JAX path too (closure captures `tau0_1`, `tau0_2`, `u1_0`, etc. as constants). Recomputing IC per iterate would re-introduce implicit `a`-dependence in `psi_ss(a)` and `tau0(a)` that neither the analytic sensitivities nor the JAX trace tracks, biasing the gradient.

## Known issues / pending work

- **Single-block notebook** still uses the continuous adjoint and has not been migrated to forward sensitivity. The same migration would apply if needed.

## Github

Make sure to commit and push changes to github when completed.
