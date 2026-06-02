# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Gradient-based inversion of rate-and-state friction parameters for spring-slider models of afterslip. Two configurations are supported:

- **Single-block** (legacy): invert for `a` (and optionally `k`) using the continuous adjoint.
- **Two-block (symmetric, Abe & Kato 2013)**: invert for any subset of `{a1, a2, k0, k12}`. Topology: `Plate ←(k0)→ Block 1 ←(k12)→ Block 2 ←(k0)→ Plate`. The gradient is obtained via the **discrete adjoint via AD** (JAX + Diffrax + Optimistix) — the only active gradient path. It backpropagates through the actual adaptive ODE stepper using `RecursiveCheckpointAdjoint`, and scales independently of parameter count.

The forward model is a DAE (force balance + two ODEs per block). The JAX/Diffrax path eliminates `V` via a differentiable root-find (Optimistix Newton in `log V`) inside the ODE RHS, then reverse-mode-differentiates through Diffrax's `Dopri8` stepper.

## Why not the continuous adjoint (two-block)

The continuous adjoint with an adaptively-stepped forward solver became **dual-inconsistent** during fast slip events: the adjoint integrates against forward-grid-interpolated Jacobians whose denominator `tau_V + eta` pinches near rupture, producing spurious blowup (the explicit RK3 adjoint and a Radau adjoint agreed with each other but disagreed with FD by many orders of magnitude on long horizons). See Alexe & Sandu, *J. Comput. Appl. Math.* 233 (2009) for the general phenomenon.

A previous numpy **forward sensitivity** path (`ds_x/dp = ∂x/∂p` integrated alongside the nominal state) also sidestepped the issue but scaled linearly in the number of parameters and is no longer used.

The **discrete adjoint via AD** (JAX/Diffrax `RecursiveCheckpointAdjoint`) reverse-mode-differentiates through the actual time-stepping algorithm with logarithmic checkpointing. Cost ≈ `O(1) ×` forward solve regardless of parameter count, which matters once we want to invert for more than 4 parameters. Diffrax explicitly recommends it over `BacksolveAdjoint` (the continuous adjoint) for stiff/near-stiff systems — exactly the failure mode we hit.

## Objective function landscape

The two-block objective is **highly non-convex and rough**, even when restricted to two parameters at a time. The figure below shows `log10 J(a1, k12)` from the 2D landscape sweep in `slip_discrete_adjoint_double_springslider.ipynb`:

![2D objective landscape — J(a1, k12)](Figures/2D_landscape_example.png)

Notable features visible in the surface and contour:

- **Long, narrow ridges and valleys** in the `(a1, k12)` plane — the misfit varies by orders of magnitude across a thin ridge but is nearly flat along it, so steepest descent stalls or zig-zags.
- **Many shallow local minima** clustered near the true parameters (red star at `(0.001, 40)`). The grid minimum (cyan circle) and the recovered inversion result (magenta X) sit in different basins from the true minimum, despite all three being within a few percent in `a1`.
- **Discretisation-scale wobble** — even at `Dopri8` with `rtol=1e-11, atol=1e-13` the surface shows small-amplitude high-frequency texture from the adaptive stepper boundary changing across the parameter grid. Loosening tolerances makes this dominate the basin structure.

Practical consequences:
- Local optimisers (`trust-constr`, `L-BFGS-B`) reliably descend into *a* nearby minimum but rarely the global one. Use `basinhopping` (or multi-start) for any meaningful inversion.
- The gradient is correct (validated to `rel_err ≲ 1e-4` against FD) — local-minimum trapping is a property of the objective, not a gradient bug.
- The smoothing scale `sigma` in `S` is the main lever for regularising the landscape: larger `sigma` washes out high-frequency structure at the cost of resolving fewer features.

## Running the code

The primary workflow is `slip_discrete_adjoint_double_springslider.ipynb` — the JAX/Diffrax discrete-adjoint two-block notebook. It re-implements the two-block forward model in JAX, uses a differentiable root-find for `V`, runs `RecursiveCheckpointAdjoint` backprop, validates the gradient via an AD-vs-FD step-size sweep, runs 1D and 2D landscape scans (`run_J_landscape_jax`, `run_J_landscape_2d_jax` from `adjoint_tests.py`), and inverts via `scipy.optimize.minimize(trust-constr)` or `scipy.optimize.basinhopping` with `jax.value_and_grad`.

Other notebooks (legacy):
- `slip_adjoint_double_springslider.ipynb` — two-block continuous adjoint (legacy; superseded by the discrete-adjoint notebook).
- `slip_adjoint_springslider_adapttime.ipynb` — single-block continuous adjoint.
- `visualize_objective.ipynb` — objective function visualization.

The Python modules use `%autoreload 2` in the notebooks. The discrete-adjoint notebook requires `jax`, `diffrax`, and `optimistix` (with `jax_enable_x64=True`).

No test runner. Gradient correctness is validated inline via FD checks in the notebook: a centred-FD step-size sweep locates the noise/bias plateau and reports `|AD - FD| / |FD|`. Target: `rel_err ≲ 1e-4`.

## Module dependency order

```
friction_derivs.py   ← physics primitives, IC setup, smoothing matrix (no imports from project)
      ↓
adapt_fwd_solve.py   ← adaptive RK forward solver (numpy reference; sanity-check only)
      ↓
adjoint_solve.py     ← single-block continuous adjoint (legacy)
      ↓
compute_obj.py       ← J and dJ/dp via continuous adjoint / forward sensitivity (legacy)
```

`adjoint_tests.py` holds the JAX landscape drivers used by `slip_discrete_adjoint_double_springslider.ipynb`: `run_J_landscape_jax(...)` (1D per-parameter scan with optional AD-gradient arrows) and `run_J_landscape_2d_jax(...)` (2D parameter sweep via `jax.vmap` chunks, produces the surface + contour figure shown above). The legacy `validate_gradient_vs_fd(...)` and `run_J_landscape(...)` numpy drivers are kept for the older notebooks.

`landscape_worker.py` is the process-pool worker for the legacy numpy J-landscape scan.

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
**Force balances** (each solved independently for V_i):
```
Block 1: tau1 + eta*V1 + (k0+k12)*u1 - k12*u2 = tau0_1 + k0*V_bg*t
Block 2: tau2 + eta*V2 + (k0+k12)*u2 - k12*u1 = tau0_2 + k0*V_bg*t
```
Both blocks are independently loaded by the plate via `k0`; `k12` is the coupling spring. In the JAX path, both are solved via Optimistix Newton in `log V`.

**Per-block friction parameters.** Each block carries its own `a, N, b, dc, f0` — keys `a1/a2`, `N1/N2`, `b1/b2`, `dc1/dc2`, `f0_1/f0_2`. The helper `block_M(M, i)` (defined in `friction_derivs.py`, re-exported by `adapt_fwd_solve.py`) builds a per-block scalar dict by picking the suffixed key when present and falling back to the shared name (`a`, `N`, `b`, `dc`, `f0`) otherwise — so legacy callers that set only shared values keep working unchanged.

**`M` keys:** shared — `V0, eta, k0, k12, V_bg, tau0_1, tau0_2`; per-block — `a1, a2, N1, N2, b1, b2, dc1, dc2, f0_1, f0_2` (or shared-name fallbacks). `tau0_1` and `tau0_2` are computed by `setup_initial_conditions_2block(M)` in `friction_derivs.py` from each block's own friction parameters.

**AD scope.** The JAX/Diffrax discrete-adjoint path captures per-block constants `N_i, b_i, dc_i, f0_i` by closure — they are inputs to the differentiable forward but are not part of `p_vec`, so AD treats them as fixed. The active parameter vector is `p_vec = [a1, a2, k0, k12]`. Adding parameters to `p_vec` only requires widening the closure-captured tuple and the `args` passed to `diffeqsolve`.

## Discrete adjoint via JAX/Diffrax (two-block)

`slip_discrete_adjoint_double_springslider.ipynb` is self-contained. Key implementation choices:

1. **Algebraic constraint** `tau(V,psi) + eta*V = rhs` is solved via `optimistix.Newton` in `log V` (log-space keeps Newton stable across the 12+ orders of magnitude `V` traverses through rupture). Optimistix is a differentiable implicit-layer library — AD propagates through the root-find via the implicit function theorem, so backprop "just works" and is taken at a genuine root.
   - Initial guess from the friction-dominated approximation (ignoring `eta*V`): `logV0 = log(V0) - psi/a + rhs/(N*a)`. Lands within O(1) in log space.
   - Tolerances `rtol=1e-12, atol=1e-13`; `throw=True` so silent non-convergence raises.
2. **`tau_fn` is computed in log-space** (`log_xi = log(V) - log(2V0) + psi/a`, then `arcsinh(exp(log_xi)) = log_xi + log1p(sqrt(1 + exp(-2 log_xi)))`). Forming `V/(2V0)*exp(psi/a)` directly overflows at steady state where `psi/a ~ 600` and the reverse pass would silently zero `d(tau)/d(logV)`, sending Newton to NaN.
3. **Vector field** is the 4-D ODE in `(u1, psi1, u2, psi2)` with `V_i` eliminated inside the RHS by `solve_V`. Diffrax sees only the ODE.
4. **Time stepping**: `diffrax.Dopri8` (8th-order explicit RK with PI step-size control), `rtol=1e-11`, `atol=1e-13`, `max_steps=500_000`. The high-order solver shrinks the discretisation noise of `J(p)` (making FD a clean reference) and reduces the step-grid wobble backprop must propagate through.
5. **Saving**: `SaveAt(ts=t_ref)` returns the state directly on the fixed reference grid `J` uses — no post-hoc interpolation needed.
6. **Adjoint**: `diffrax.RecursiveCheckpointAdjoint()` — reverse-mode AD through the actual stepper with logarithmic checkpointing. Recommended over `BacksolveAdjoint` for stiff/near-stiff systems.
7. **Objective**: `J_fn(p)` builds residuals `S @ u_i - Su_i_obs` and integrates `0.5 * (r1² + r2²)` via `jnp.trapezoid` on `t_ref`. `J_and_grad = jax.jit(jax.value_and_grad(J_fn))`.

**Gradient validation cell** does a centred-FD sweep across `eps_rel ∈ logspace(-2, -10)` to locate the noise/bias plateau, then picks per-parameter best `eps` and reports `|AD - FD| / |FD|`. The plot of FD vs `eps` exposes the U-shape (truncation bias at large `eps`, round-off noise at small `eps`) directly.

**`jax_enable_x64=True` is required** — double precision throughout; `arcsinh` arguments reach ~1e35 during rupture and float32 would be useless.

## Smoothing matrix

`make_smoothing_matrix(t, sigma)` in `friction_derivs.py` builds a row-normalised Gaussian `S` with **trapezoidal integration weights** on each column. This corrects for non-uniform node spacing on adaptive grids.

For uniform grids the weights are constant and cancel, recovering the standard un-weighted Gaussian. The inversion builds `S` once on a fixed uniform `t_ref` and reuses it for every iterate.

## Gradient consistency requirement

Both `J` and `dJ/dp` are evaluated on a **fixed reference grid `t_ref`** (built once at the initial guess, uniform). In the JAX path, `SaveAt(ts=t_ref)` returns the forward solution directly on `t_ref`, so the gradient AD computes matches the `J` the optimiser sees by construction.

## Inversion setup

**Two-block discrete-adjoint (`slip_discrete_adjoint_double_springslider.ipynb`):**
- `INVERT_PARAMS` controls which subset of `['a1','a2','k0','k12']` is optimised.
- `fun_and_grad(x_norm)` calls the cached `J_and_grad(p_vec)` once per evaluation; the active-parameter gradient is sliced out of the full 4-component `jax.grad` result.
- Optimiser: `scipy.optimize.minimize(method='trust-constr', jac=True, ...)` or `scipy.optimize.basinhopping` wrapping the same `trust-constr` local step for global search. **Basin-hopping is strongly recommended** given the landscape complexity documented above.
- Parameters are normalised by their initial values (`scales`) so all components are O(1) inside the optimiser.
- IC (`u*_0_inv`, `psi*_0_inv`, `tau0_*`) is built once from the initial guess and frozen across iterates. The JAX closure captures `tau0_1`, `tau0_2`, `u1_0`, etc. as constants. Recomputing IC per iterate would re-introduce implicit `a`-dependence in `psi_ss(a)` and `tau0(a)` that AD does not track through the closure, biasing the gradient.

## Known issues / pending work

- **Single-block notebook** still uses the continuous adjoint and has not been migrated to the JAX discrete-adjoint approach. The same migration would apply if needed.
- **Non-convex landscape** — even with a correct gradient and basin-hopping, the inversion can settle in a local minimum offset from the truth (see figure above). Multistart from physically-motivated initial guesses, or a stronger smoothing `sigma`, are the practical levers.

## Github

Make sure to commit and push changes to github when completed.
