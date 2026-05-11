from adapt_fwd_solve import *
from friction_derivs import *
from compute_obj import *
import numpy as np

def adjoint_solve(fwd, t_obs, u_obs, M, sigma, fwd_interp=None, S=None,
                  smooth_misfit=None):
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

    Optional smooth_misfit: pre-computed source array on fwd['t'].  When
    provided, the internal misfit computation (using S / sigma / u_obs) is
    skipped entirely.  t_obs and u_obs may be None in this case.  Use this
    when J is defined on a different grid (e.g. a dense uniform reference
    grid) and the source has already been interpolated onto the adjoint grid.
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
