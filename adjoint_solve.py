from adapt_fwd_solve import *
from friction_derivs import *
from compute_obj import *
import numpy as np

def adjoint_solve(fwd, t_obs, u_obs, M, sigma):
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
    """
    k   = M['k']
    eta = M['eta']
    n   = len(fwd['t'])

    # --- smoothed misfit: S^T (S u − S u_obs) ---
    S             = make_smoothing_matrix(fwd['t'], sigma)
    u_obs_at_fwd  = np.interp(fwd['t'], t_obs, u_obs)
    smooth_misfit = S.T @ (S @ fwd['u'] - S @ u_obs_at_fwd)   # shape (n,)

    # --- reverse arrays so index 0 = t=T, index n-1 = t=0 ---
    rev  = slice(None, None, -1)
    tV_r = fwd['tau_V'][rev]
    tP_r = fwd['tau_psi'][rev]
    GV_r = fwd['G_V'][rev]
    GP_r = fwd['G_psi'][rev]
    sm_r = smooth_misfit[rev]
    t_r  = fwd['t'][rev]

    p_r = np.zeros(n)   # IC at τ=0 (t=T)
    r_r = np.zeros(n)

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
    lam = (p + fwd['G_V'] * r) / (fwd['tau_V'] + eta)

    return dict(t=fwd['t'], p=p, r=r, lam=lam)

def adjoint_solve_feat(fwd, events_obs, M, w_T=1.0, w_du=1.0, V_thresh=1e-6):
    """
    Adjoint solver for the feature-based objective J_feat.

      J_feat = w_du * Σ_i (Δu_i_mod - Δu_i_obs)²
             + w_T  * Σ_i (T_i_mod  - T_i_obs )²

    Uses the same 3-stage embedded RK as adjoint_solve, but replaces the
    Gaussian-smoothed waveform misfit with feature-based adjoint forcing
    sm(t) = dJ_feat/du(t) built by build_feat_adjoint_forcing.

    The gradient formula compute_grad_a is identical to the waveform case and
    works unchanged with the adjoint variables returned here.

    Parameters
    ----------
    fwd        : forward solution dict (from forward_solve_adaptive).
    events_obs : list of observed event dicts (from detect_events on the true fwd).
    M          : model-parameter dict.
    w_T, w_du  : weights for the T and Δu terms in J_feat.
    V_thresh   : velocity threshold for event detection (m/s).

    Returns
    -------
    adj : dict with keys t, p, r, lam  (same structure as adjoint_solve).
    """
    k   = M['k']
    eta = M['eta']
    n   = len(fwd['t'])

    # Detect events in current forward solution and build feature-based forcing
    events_mod = detect_events(fwd, V_thresh)
    sm_feat    = build_feat_adjoint_forcing(fwd, events_mod, events_obs, M, w_T, w_du)

    # --- reverse arrays so index 0 = t=T, index n-1 = t=0 ---
    rev  = slice(None, None, -1)
    tV_r = fwd['tau_V'][rev]
    tP_r = fwd['tau_psi'][rev]
    GV_r = fwd['G_V'][rev]
    GP_r = fwd['G_psi'][rev]
    sm_r = sm_feat[rev]
    t_r  = fwd['t'][rev]

    p_r = np.zeros(n)
    r_r = np.zeros(n)

    def _rhs(p, r, tV, tP, GV, GP, sm):
        D   = tV + eta
        lam = (p + GV * r) / D
        return -k * lam - sm, -tP * lam + GP * r

    for j in range(n - 1):
        dt_tau = t_r[j] - t_r[j + 1]

        pj, rj = p_r[j], r_r[j]

        tV1, tP1, GV1, GP1, sm1 = tV_r[j],     tP_r[j],     GV_r[j],     GP_r[j],     sm_r[j]
        tV2, tP2, GV2, GP2, sm2 = (0.5*(tV_r[j]+tV_r[j+1]), 0.5*(tP_r[j]+tP_r[j+1]),
                                    0.5*(GV_r[j]+GV_r[j+1]), 0.5*(GP_r[j]+GP_r[j+1]),
                                    0.5*(sm_r[j]+sm_r[j+1]))
        tV3, tP3, GV3, GP3, sm3 = tV_r[j+1],   tP_r[j+1],   GV_r[j+1],   GP_r[j+1],   sm_r[j+1]

        dp1, dr1 = _rhs(pj, rj, tV1, tP1, GV1, GP1, sm1)

        p2 = pj + 0.5 * dt_tau * dp1
        r2 = rj + 0.5 * dt_tau * dr1
        dp2, dr2 = _rhs(p2, r2, tV2, tP2, GV2, GP2, sm2)

        p3 = pj + dt_tau * (-dp1 + 2.0 * dp2)
        r3 = rj + dt_tau * (-dr1 + 2.0 * dr2)
        dp3, dr3 = _rhs(p3, r3, tV3, tP3, GV3, GP3, sm3)

        p_r[j + 1] = pj + dt_tau / 6.0 * (dp1 + 4.0 * dp2 + dp3)
        r_r[j + 1] = rj + dt_tau / 6.0 * (dr1 + 4.0 * dr2 + dr3)

    p   = p_r[rev]
    r   = r_r[rev]
    lam = (p + fwd['G_V'] * r) / (fwd['tau_V'] + eta)

    return dict(t=fwd['t'], p=p, r=r, lam=lam)


def make_smoothing_matrix(t, sigma):
    """
    Row-normalised Gaussian smoothing matrix on an arbitrary time grid.

    S[i, j] = exp(-(t[i] - t[j])^2 / (2*sigma^2))  then row-normalised.

    Forms a dense (n x n) array — keep sigma modest relative to n*dt_max
    if memory is a concern.
    """
    diff2 = (t[:, None] - t[None, :]) ** 2 / (2.0 * sigma ** 2)
    S = np.exp(-diff2)
    S /= S.sum(axis=1, keepdims=True)
    return S