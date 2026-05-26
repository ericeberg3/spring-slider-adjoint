from adapt_fwd_solve import *
from friction_derivs import *
from compute_obj import *
import numpy as np
from scipy.integrate import solve_ivp

def adjoint_solve(fwd, t_obs, u_obs, M, sigma, fwd_interp=None, S=None, smooth_misfit=None):
    """
    Adjoint solver integrated **forwards in reversed time** τ = T - t using
    the same 3-stage embedded RK scheme as the forward solver.

    Within each step the forward-state Jacobian coefficients
    (τ_V, τ_ψ, G_V, G_ψ) and the smoothed misfit source (sm) are linearly
    interpolated at α ∈ {0, ½, 1} so the adjoint RHS sees a consistent
    picture of the forward trajectory inside each interval.

    Adjoint RHS at reversed-time step j → j+1
    (Δτ = t_r[j] − t_r[j+1] > 0):

        f(p, r; α) = [ −k·λ − sm(α),
                       −τ_ψ(α)·λ + G_ψ(α)·r ]
        λ = (p + G_V(α)·r) / (τ_V(α) + η)

    3-stage (Bogacki–Shampine style) update:
        k1 = f(y_j;       α=0  )
        k2 = f(y_j + Δτ/2·k1;  α=½)
        k3 = f(y_j + Δτ(−k1+2k2); α=1)
        y_{j+1} = y_j + Δτ/6·(k1 + 4·k2 + k3)   [3rd-order]

    Smoothed misfit source: sm = S^T(Su − Su_obs),  S = Gaussian(t, σ).
    IC: p=r=0 at τ=0 (t=T).
    At the end, arrays are flipped back to original time order.

    Optional fwd_interp: if provided, τ_V, τ_ψ, G_V, G_ψ, and u are
    linearly interpolated from fwd_interp onto fwd['t'].  The time stepping
    still uses the grid in fwd['t'].  This lets you advance the adjoint on
    (e.g.) an adaptive time grid while drawing Jacobians from a different
    forward solve (e.g. a Forward-Euler solution).

    Optional S: pre-built smoothing matrix.  If provided, sigma is ignored for
    the misfit source term.  Pass a plain row-normalised Gaussian (no integration
    weights) for Forward Euler; omit (or pass None) to use make_smoothing_matrix,
    which applies trapezoidal integration weights for non-uniform adaptive grids.
    """
    k   = M['k']
    eta = M['eta']
    n   = len(fwd['t'])

    # --- choose source arrays for forward-state quantities ---
    if fwd_interp is not None:
        t_src       = fwd_interp['t']
        u_src       = np.interp(fwd['t'], t_src, fwd_interp['u'])
        tau_V_src   = np.interp(fwd['t'], t_src, fwd_interp['tau_V'])
        tau_psi_src = np.interp(fwd['t'], t_src, fwd_interp['tau_psi'])
        G_V_src     = np.interp(fwd['t'], t_src, fwd_interp['G_V'])
        G_psi_src   = np.interp(fwd['t'], t_src, fwd_interp['G_psi'])
    else:
        u_src       = fwd['u']
        tau_V_src   = fwd['tau_V']
        tau_psi_src = fwd['tau_psi']
        G_V_src     = fwd['G_V']
        G_psi_src   = fwd['G_psi']

    # --- smoothed misfit: S^T (S u − S u_obs) ---
    if smooth_misfit is None:
        u_obs_at_fwd = np.interp(fwd['t'], t_obs, u_obs)
        if sigma is None and S is None:
            # S = I  →  S^T(Su − Su_obs) = u − u_obs  (no matrix needed)
            smooth_misfit = u_src - u_obs_at_fwd
        else:
            if S is None:
                S = make_smoothing_matrix(fwd['t'], sigma)
            smooth_misfit = S.T @ (S @ u_src - S @ u_obs_at_fwd)   # shape (n,)

    # --- reverse arrays so index 0 = t=T, index n-1 = t=0 ---
    rev  = slice(None, None, -1)
    tV_r = tau_V_src[rev]
    tP_r = tau_psi_src[rev]
    GV_r = G_V_src[rev]
    GP_r = G_psi_src[rev]
    sm_r = smooth_misfit[rev]
    t_r  = fwd['t'][rev]

    p_r = np.zeros(n)   # IC at τ=0 (t=T)
    r_r = np.zeros(n)

    u_r   = fwd['u'][::-1]
    psi_r = fwd['psi'][::-1]

    def _rhs(p, r, tV, tP, GV, GP, sm):
        """Adjoint RHS at a single point with given forward-state coefficients."""
        D   = tV + eta
        lam = (p + GV * r) / D
        return -k * lam - sm, -tP * lam + GP * r

    for j in range(n - 1):
        dt_tau = t_r[j] - t_r[j + 1]   # > 0  (t_r is decreasing)

        pj, rj = p_r[j], r_r[j]

        # Forward-state coefficients at the three interpolation points
        #   α=0  (stage 1 — current node j)
        tV1, tP1, GV1, GP1, sm1 = tV_r[j],     tP_r[j],     GV_r[j],     GP_r[j],     sm_r[j]
        #   α=½  (stage 2 — midpoint, linearly interpolated)
        tV2, tP2, GV2, GP2, sm2 = (0.5*(tV_r[j]+tV_r[j+1]), 0.5*(tP_r[j]+tP_r[j+1]),
                                    0.5*(GV_r[j]+GV_r[j+1]), 0.5*(GP_r[j]+GP_r[j+1]),
                                    0.5*(sm_r[j]+sm_r[j+1]))
        #   α=1  (stage 3 — next node j+1)
        tV3, tP3, GV3, GP3, sm3 = tV_r[j+1],   tP_r[j+1],   GV_r[j+1],   GP_r[j+1],   sm_r[j+1]

        # Stage 1
        dp1, dr1 = _rhs(pj, rj, tV1, tP1, GV1, GP1, sm1)

        # Stage 2 (midpoint)
        p2 = pj + 0.5 * dt_tau * dp1
        r2 = rj + 0.5 * dt_tau * dr1
        dp2, dr2 = _rhs(p2, r2, tV2, tP2, GV2, GP2, sm2)

        # Stage 3 (endpoint, same formula as forward solver)
        p3 = pj + dt_tau * (-dp1 + 2.0 * dp2)
        r3 = rj + dt_tau * (-dr1 + 2.0 * dr2)
        dp3, dr3 = _rhs(p3, r3, tV3, tP3, GV3, GP3, sm3)

        # 3rd-order update (same weights as forward solver)
        p_r[j + 1] = pj + dt_tau / 6.0 * (dp1 + 4.0 * dp2 + dp3)
        r_r[j + 1] = rj + dt_tau / 6.0 * (dr1 + 4.0 * dr2 + dr3)

    # --- re-invert time: flip back to original order ---
    p   = p_r[rev]
    r   = r_r[rev]

    # λ = Lagrange multiplier of the force-balance constraint
    lam = (p + G_V_src * r) / (tau_V_src + eta)

    return dict(t=fwd['t'], p=p, r=r, lam=lam)


def adjoint_solve_2block(fwd, t_obs, u1_obs, u2_obs, M, sigma,
                         S=None, smooth_misfit1=None, smooth_misfit2=None,
                         use_implicit=False, implicit_rtol=1e-8, implicit_atol=1e-10):
    """
    Adjoint solver for the two-block symmetrically loaded spring-slider system.

    Adjoint variables: pu1, r1 (=pps1) for Block 1; pu2, r2 (=pps2) for Block 2.
    Lagrange multipliers:
        lam1 = (pu1 + G_V1*r1) / (tau_V1 + eta)
        lam2 = (pu2 + G_V2*r2) / (tau_V2 + eta)

    Reversed-time RHS (dpu_i/dtau, dr_i/dtau):
        dpu1 = -(k0+k12)*lam1 + k12*lam2 - sm1
        dr1  = -tau_psi1*lam1 + G_psi1*r1
        dpu2 = +k12*lam1 - (k0+k12)*lam2 - sm2
        dr2  = -tau_psi2*lam2 + G_psi2*r2

    Pass u1_obs=None or u2_obs=None to exclude that block from the misfit.
    IC: all adjoint variables zero at t=T.

    use_implicit: if True, dispatch to adjoint_solve_2block_implicit (Radau IIA)
        instead of the explicit RK3 below. Use this when the adjoint blows up
        on long horizons (stiff dissipative forward => anti-stable adjoint).
    """
    if use_implicit:
        return adjoint_solve_2block_implicit(
            fwd, t_obs, u1_obs, u2_obs, M, sigma,
            S=S, smooth_misfit1=smooth_misfit1, smooth_misfit2=smooth_misfit2,
            rtol=implicit_rtol, atol=implicit_atol,
        )
    k0, k12, eta = M['k0'], M['k12'], M['eta']
    n = len(fwd['t'])

    # --- smoothed misfit sources ---
    def _build_sm(u_src, u_obs_arr):
        if u_obs_arr is None:
            return np.zeros(n)
        u_obs_at_fwd = np.interp(fwd['t'], t_obs, u_obs_arr)
        if sigma is None and S is None:
            return u_src - u_obs_at_fwd
        _S = S if S is not None else make_smoothing_matrix(fwd['t'], sigma)
        return _S.T @ (_S @ u_src - _S @ u_obs_at_fwd)

    if smooth_misfit1 is None:
        smooth_misfit1 = _build_sm(fwd['u1'], u1_obs)
    if smooth_misfit2 is None:
        smooth_misfit2 = _build_sm(fwd['u2'], u2_obs)

    # --- reverse all forward arrays ---
    rev = slice(None, None, -1)
    tV1_r = fwd['tau_V1'][rev];   tP1_r = fwd['tau_psi1'][rev]
    GV1_r = fwd['G_V1'][rev];     GP1_r = fwd['G_psi1'][rev]
    tV2_r = fwd['tau_V2'][rev];   tP2_r = fwd['tau_psi2'][rev]
    GV2_r = fwd['G_V2'][rev];     GP2_r = fwd['G_psi2'][rev]
    sm1_r = smooth_misfit1[rev];  sm2_r = smooth_misfit2[rev]
    t_r   = fwd['t'][rev]

    pu1_r = np.zeros(n); r1_r = np.zeros(n)
    pu2_r = np.zeros(n); r2_r = np.zeros(n)

    def _rhs(pu1, r1, pu2, r2, tV1, tP1, GV1, GP1, sm1,
                               tV2, tP2, GV2, GP2, sm2):
        lam1 = (pu1 + GV1 * r1) / (tV1 + eta)
        lam2 = (pu2 + GV2 * r2) / (tV2 + eta)
        dpu1 = -(k0 + k12) * lam1 + k12 * lam2 + sm1
        dr1  = -tP1 * lam1 + GP1 * r1
        dpu2 =  k12 * lam1 - (k0 + k12) * lam2 + sm2
        dr2  = -tP2 * lam2 + GP2 * r2
        return dpu1, dr1, dpu2, dr2

    for j in range(n - 1):
        dt_tau = t_r[j] - t_r[j + 1]   # > 0

        pu1j, r1j = pu1_r[j], r1_r[j]
        pu2j, r2j = pu2_r[j], r2_r[j]

        # stage-1 (α=0)
        c1 = (tV1_r[j], tP1_r[j], GV1_r[j], GP1_r[j], sm1_r[j],
              tV2_r[j], tP2_r[j], GV2_r[j], GP2_r[j], sm2_r[j])
        # stage-2 (α=½, linear interpolation)
        c2 = (0.5*(tV1_r[j]+tV1_r[j+1]), 0.5*(tP1_r[j]+tP1_r[j+1]),
              0.5*(GV1_r[j]+GV1_r[j+1]), 0.5*(GP1_r[j]+GP1_r[j+1]),
              0.5*(sm1_r[j]+sm1_r[j+1]),
              0.5*(tV2_r[j]+tV2_r[j+1]), 0.5*(tP2_r[j]+tP2_r[j+1]),
              0.5*(GV2_r[j]+GV2_r[j+1]), 0.5*(GP2_r[j]+GP2_r[j+1]),
              0.5*(sm2_r[j]+sm2_r[j+1]))
        # stage-3 (α=1)
        c3 = (tV1_r[j+1], tP1_r[j+1], GV1_r[j+1], GP1_r[j+1], sm1_r[j+1],
              tV2_r[j+1], tP2_r[j+1], GV2_r[j+1], GP2_r[j+1], sm2_r[j+1])

        dpu1_1, dr1_1, dpu2_1, dr2_1 = _rhs(pu1j, r1j, pu2j, r2j, *c1)

        pu1_2 = pu1j + 0.5*dt_tau*dpu1_1; r1_2 = r1j + 0.5*dt_tau*dr1_1
        pu2_2 = pu2j + 0.5*dt_tau*dpu2_1; r2_2 = r2j + 0.5*dt_tau*dr2_1
        dpu1_2, dr1_2, dpu2_2, dr2_2 = _rhs(pu1_2, r1_2, pu2_2, r2_2, *c2)

        pu1_3 = pu1j + dt_tau*(-dpu1_1+2.0*dpu1_2); r1_3 = r1j + dt_tau*(-dr1_1+2.0*dr1_2)
        pu2_3 = pu2j + dt_tau*(-dpu2_1+2.0*dpu2_2); r2_3 = r2j + dt_tau*(-dr2_1+2.0*dr2_2)
        dpu1_3, dr1_3, dpu2_3, dr2_3 = _rhs(pu1_3, r1_3, pu2_3, r2_3, *c3)

        pu1_r[j+1] = pu1j + dt_tau/6.0*(dpu1_1 + 4.0*dpu1_2 + dpu1_3)
        r1_r[j+1]  = r1j  + dt_tau/6.0*(dr1_1  + 4.0*dr1_2  + dr1_3)
        pu2_r[j+1] = pu2j + dt_tau/6.0*(dpu2_1 + 4.0*dpu2_2 + dpu2_3)
        r2_r[j+1]  = r2j  + dt_tau/6.0*(dr2_1  + 4.0*dr2_2  + dr2_3)

    pu1 = pu1_r[rev]; r1 = r1_r[rev]
    pu2 = pu2_r[rev]; r2 = r2_r[rev]

    lam1 = (pu1 + fwd['G_V1'] * r1) / (fwd['tau_V1'] + eta)
    lam2 = (pu2 + fwd['G_V2'] * r2) / (fwd['tau_V2'] + eta)

    return dict(t=fwd['t'], pu1=pu1, r1=r1, lam1=lam1, pu2=pu2, r2=r2, lam2=lam2)


def adjoint_solve_2block_implicit(fwd, t_obs, u1_obs, u2_obs, M, sigma,
                                   S=None, smooth_misfit1=None, smooth_misfit2=None,
                                   rtol=1e-8, atol=1e-10):
    """
    Stiff implicit two-block adjoint solver using scipy.integrate.solve_ivp
    with method='Radau' (L-stable, 5th-order implicit Runge-Kutta).

    Use this when the explicit RK3 adjoint blows up on long horizons. The
    forward DAE is dissipative, so the adjoint is anti-stable backward in
    time and grows like exp(T / t_relax). Radau handles those stiff modes
    implicitly instead of trying to resolve them.

    Integration is done in forward time t from t=T down to t=0
    (solve_ivp accepts t_span with t_span[0] > t_span[1]). The forward-state
    coefficients (tau_V*, tau_psi*, G_V*, G_psi*) and the misfit sources
    are linearly interpolated from fwd['t'] inside the RHS. Output is
    re-flipped to original (increasing) time order so it matches the
    explicit solver's return contract.

    Forward-time adjoint RHS (negation of the reversed-time RHS):
        dpu1/dt = +(k0+k12)*lam1 - k12*lam2 - sm1
        dr1 /dt = +tau_psi1*lam1 - G_psi1*r1
        dpu2/dt = -k12*lam1 + (k0+k12)*lam2 - sm2
        dr2 /dt = +tau_psi2*lam2 - G_psi2*r2
    Terminal condition: pu1=r1=pu2=r2=0 at t=T.

    Args mirror adjoint_solve_2block; rtol/atol set Radau tolerances.
    """
    k0, k12, eta = M['k0'], M['k12'], M['eta']
    t_fwd = fwd['t']

    # --- build misfit sources on forward grid (same convention as explicit) ---
    n = len(t_fwd)
    def _build_sm(u_src, u_obs_arr):
        if u_obs_arr is None:
            return np.zeros(n)
        u_obs_at_fwd = np.interp(t_fwd, t_obs, u_obs_arr)
        if sigma is None and S is None:
            return u_src - u_obs_at_fwd
        _S = S if S is not None else make_smoothing_matrix(t_fwd, sigma)
        return _S.T @ (_S @ u_src - _S @ u_obs_at_fwd)

    if smooth_misfit1 is None:
        smooth_misfit1 = _build_sm(fwd['u1'], u1_obs)
    if smooth_misfit2 is None:
        smooth_misfit2 = _build_sm(fwd['u2'], u2_obs)

    tV1, tP1, GV1, GP1 = fwd['tau_V1'], fwd['tau_psi1'], fwd['G_V1'], fwd['G_psi1']
    tV2, tP2, GV2, GP2 = fwd['tau_V2'], fwd['tau_psi2'], fwd['G_V2'], fwd['G_psi2']

    def _coeffs_at(t):
        return (np.interp(t, t_fwd, tV1), np.interp(t, t_fwd, tP1),
                np.interp(t, t_fwd, GV1), np.interp(t, t_fwd, GP1),
                np.interp(t, t_fwd, smooth_misfit1),
                np.interp(t, t_fwd, tV2), np.interp(t, t_fwd, tP2),
                np.interp(t, t_fwd, GV2), np.interp(t, t_fwd, GP2),
                np.interp(t, t_fwd, smooth_misfit2))

    def rhs(t, y):
        pu1_, r1_, pu2_, r2_ = y
        tV1_, tP1_, GV1_, GP1_, sm1_, tV2_, tP2_, GV2_, GP2_, sm2_ = _coeffs_at(t)
        D1, D2 = tV1_ + eta, tV2_ + eta
        lam1 = (pu1_ + GV1_ * r1_) / D1
        lam2 = (pu2_ + GV2_ * r2_) / D2
        dpu1 =  (k0 + k12) * lam1 - k12 * lam2 - sm1_
        dr1  =  tP1_ * lam1 - GP1_ * r1_
        dpu2 = -k12 * lam1 + (k0 + k12) * lam2 - sm2_
        dr2  =  tP2_ * lam2 - GP2_ * r2_
        return [dpu1, dr1, dpu2, dr2]

    def jac(t, _y):
        # RHS is linear in y, so Jacobian depends only on the time-dependent
        # forward-state coefficients (not on y itself). sm terms drop out too.
        tV1_, tP1_, GV1_, GP1_, _, tV2_, tP2_, GV2_, GP2_, _ = _coeffs_at(t)
        D1, D2 = tV1_ + eta, tV2_ + eta
        # d(lam1)/d(pu1, r1) = (1/D1, GV1/D1);  d(lam2)/d(pu2, r2) = (1/D2, GV2/D2)
        J = np.zeros((4, 4))
        # row 0: dpu1/dt
        J[0, 0] =  (k0 + k12) / D1
        J[0, 1] =  (k0 + k12) * GV1_ / D1
        J[0, 2] = -k12 / D2
        J[0, 3] = -k12 * GV2_ / D2
        # row 1: dr1/dt
        J[1, 0] =  tP1_ / D1
        J[1, 1] =  tP1_ * GV1_ / D1 - GP1_
        # row 2: dpu2/dt
        J[2, 0] = -k12 / D1
        J[2, 1] = -k12 * GV1_ / D1
        J[2, 2] =  (k0 + k12) / D2
        J[2, 3] =  (k0 + k12) * GV2_ / D2
        # row 3: dr2/dt
        J[3, 2] =  tP2_ / D2
        J[3, 3] =  tP2_ * GV2_ / D2 - GP2_
        return J

    y_T = np.zeros(4)
    # Integrate backward in t: span (T, 0); t_eval must be sorted same direction.
    t_eval_rev = t_fwd[::-1]
    sol = solve_ivp(rhs, (t_fwd[-1], t_fwd[0]), y_T,
                    method='Radau', jac=jac, t_eval=t_eval_rev,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Radau adjoint failed: {sol.message}")

    # sol.y is (4, n) at t_eval_rev (decreasing). Flip to increasing-t order.
    pu1 = sol.y[0][::-1]
    r1  = sol.y[1][::-1]
    pu2 = sol.y[2][::-1]
    r2  = sol.y[3][::-1]

    lam1 = (pu1 + fwd['G_V1'] * r1) / (fwd['tau_V1'] + eta)
    lam2 = (pu2 + fwd['G_V2'] * r2) / (fwd['tau_V2'] + eta)

    return dict(t=t_fwd, pu1=pu1, r1=r1, lam1=lam1, pu2=pu2, r2=r2, lam2=lam2)

