import numpy as np
from friction_derivs import *
from friction_derivs import block_M
from adjoint_solve import *

def forward_solve_adaptive(M, T, u0, psi0, V_init=None,
                           tol=1e-8, dt0=1.0, dtmax=1e5, safety=0.9,
                           freeze=None):
    """
    Adaptive-step forward solve using a 3-stage embedded RK method
    (2nd/3rd-order error-control pair, matching the MATLAB reference).

    Loading: tau_L(t) = tau0 + k*V_bg*t
    ODE:     du/dt = V,   dpsi/dt = G(V,psi)
    Algebraic: tau(V,psi) + eta*V + k*u = tau_L(t)  =>  V

    freeze : iterable of str, optional
        Variables whose time derivatives are forced to zero.
        Supported values: 'u' (sets du/dt=0), 'psi' (sets dpsi/dt=0).
        Example: freeze={'psi'} disables state evolution.
    """
    freeze = set(freeze) if freeze is not None else set()
    tau_L_fn = lambda t: M['tau0'] + M['k'] * M['V_bg'] * t

    V0 = solve_V_algebraic(u0, psi0, M, tau_L_fn(0.0))
    if V_init is not None:
        rel_err = abs(V0 - V_init) / V_init
        assert rel_err < 1e-6, (
            f"Init mismatch: V(0)={V0:.6e}, V_init={V_init:.6e} (rel={rel_err:.2e})"
        )

    def _rhs(u_v, psi_v, t_v):
        V = solve_V_algebraic(u_v, psi_v, M, tau_L_fn(t_v))
        du  = 0.0 if 'u'   in freeze else V
        dps = 0.0 if 'psi' in freeze else G_fn(V, psi_v, M)
        return du, dps

    def _jac(V, psi):
        return (tau_V_fn(V,psi,M), tau_psi_fn(V,psi,M),
                G_V_fn(V,psi,M),   G_psi_fn(V,psi,M),
                dtau_da_fn(V,psi,M), dG_da_fn(V,psi,M))

    # Initialise storage
    t_arr = [0.0]; u_arr = [u0]; psi_arr = [psi0]
    V_arr = [V0];  tauL_arr = [tau_L_fn(0.0)]
    j = _jac(V0, psi0)
    tV_arr=[j[0]]; tP_arr=[j[1]]; GV_arr=[j[2]]
    GP_arr=[j[3]]; da_arr=[j[4]]; dGa_arr=[j[5]]

    t = 0.0; u = u0; psi = psi0; dt = dt0
    V1, G1 = _rhs(u0, psi0, 0.0)   # stage-1 values at t=0

    while t < T:
        if t + dt > T:
            dt = T - t

        # --- three-stage embedded RK ---
        V2, G2 = _rhs(u + 0.5*dt*V1,         psi + 0.5*dt*G1,         t + 0.5*dt)
        V3, G3 = _rhs(u + dt*(-V1 + 2.0*V2), psi + dt*(-G1 + 2.0*G2), t +     dt)

        u2   = u   + dt/2.0*(V1 + V3)            # 2nd-order update
        psi2 = psi + dt/2.0*(G1 + G3)
        u3   = u   + dt/6.0*(V1 + 4.0*V2 + V3)  # 3rd-order update
        psi3 = psi + dt/6.0*(G1 + 4.0*G2 + G3)

        er = np.sqrt((u2 - u3)**2 + (psi2 - psi3)**2)

        if er < tol:
            t += dt;  u = u3;  psi = psi3
            tL    = tau_L_fn(t)
            V_new = solve_V_algebraic(u, psi, M, tL)

            t_arr.append(t);     u_arr.append(u);    psi_arr.append(psi)
            V_arr.append(V_new); tauL_arr.append(tL)
            j = _jac(V_new, psi)
            tV_arr.append(j[0]); tP_arr.append(j[1]); GV_arr.append(j[2])
            GP_arr.append(j[3]); da_arr.append(j[4]); dGa_arr.append(j[5])

            V1, G1 = _rhs(u, psi, t)  # stage-1 for next step

        # Step-size control  (q=2 → exponent 1/3)
        dt = safety * dt * (tol / er)**(1.0/3.0) if er > 0.0 else dtmax
        dt = min(dt, dtmax)

    return dict(
        t       = np.array(t_arr),
        u       = np.array(u_arr),
        psi     = np.array(psi_arr),
        V       = np.array(V_arr),
        tau_L   = np.array(tauL_arr),
        tau_V   = np.array(tV_arr),
        tau_psi = np.array(tP_arr),
        G_V     = np.array(GV_arr),
        G_psi   = np.array(GP_arr),
        dtau_da = np.array(da_arr),
        dG_da   = np.array(dGa_arr),)


def forward_solve_adaptive_2block(M, T, u1_0, psi1_0, u2_0, psi2_0,
                                  V1_init=None, V2_init=None,
                                  tol=1e-6, dt0=1.0, dtmax=1e5, safety=0.9):
    """
    Adaptive-step forward solve for two symmetrically loaded spring-sliders.

    Topology:   Plate --(k0)--> Block 1 --(k12)--> Block 2 <--(k0)-- Plate

    Force balances (each solved independently for V_i via Brent):
      Block 1: tau1(V1,psi1) + eta*V1 + (k0+k12)*u1 - k12*u2 = tau0_1 + k0*V_bg*t
      Block 2: tau2(V2,psi2) + eta*V2 + (k0+k12)*u2 - k12*u1 = tau0_2 + k0*V_bg*t

    M must contain:  a1, a2, k0, k12, tau0_1, tau0_2, eta, V_bg, V0.
    Per-block friction parameters (a_i, N_i, b_i, dc_i, f0_i) are honoured when
    present; otherwise the shared keys (a, N, b, dc, f0) are used as fallback.
    """
    M1 = block_M(M, 1)
    M2 = block_M(M, 2)
    k0, k12, eta = M['k0'], M['k12'], M['eta']
    tau_L1_fn = lambda t: M['tau0_1'] + k0 * M['V_bg'] * t
    tau_L2_fn = lambda t: M['tau0_2'] + k0 * M['V_bg'] * t

    def _solve_V1(u1, psi1, u2, t):
        rhs = tau_L1_fn(t) - (k0 + k12) * u1 + k12 * u2
        if rhs <= 0.0:
            raise ValueError(f"Block-1 force-balance RHS={rhs:.4g}<=0 at t={t:.3e}")
        def res(V): return tau_fn(V, psi1, M1) + eta * V - rhs
        return brentq(res, 1e-30, rhs / eta, xtol=1e-20, rtol=1e-10)

    def _solve_V2(u1, psi2, u2, t):
        rhs = tau_L2_fn(t) - (k0 + k12) * u2 + k12 * u1
        if rhs <= 0.0:
            raise ValueError(f"Block-2 force-balance RHS={rhs:.4g}<=0 at t={t:.3e}")
        def res(V): return tau_fn(V, psi2, M2) + eta * V - rhs
        return brentq(res, 1e-30, rhs / eta, xtol=1e-20, rtol=1e-10)

    def _rhs(u1, psi1, u2, psi2, t):
        V1 = _solve_V1(u1, psi1, u2, t)
        V2 = _solve_V2(u1, psi2, u2, t)
        return V1, G_fn(V1, psi1, M1), V2, G_fn(V2, psi2, M2)

    def _jac(V1, psi1, V2, psi2):
        return (tau_V_fn(V1, psi1, M1), tau_psi_fn(V1, psi1, M1),
                G_V_fn(V1, psi1, M1),   G_psi_fn(V1, psi1, M1),
                dtau_da_fn(V1, psi1, M1), dG_da_fn(V1, psi1, M1),
                tau_V_fn(V2, psi2, M2), tau_psi_fn(V2, psi2, M2),
                G_V_fn(V2, psi2, M2),   G_psi_fn(V2, psi2, M2),
                dtau_da_fn(V2, psi2, M2), dG_da_fn(V2, psi2, M2))

    # Verify / infer initial velocities
    V1_0 = _solve_V1(u1_0, psi1_0, u2_0, 0.0)
    V2_0 = _solve_V2(u1_0, psi2_0, u2_0, 0.0)
    if V1_init is not None:
        assert abs(V1_0 - V1_init) / V1_init < 1e-6, \
            f"Block-1 init mismatch: V(0)={V1_0:.6e}, V1_init={V1_init:.6e}"
    if V2_init is not None:
        assert abs(V2_0 - V2_init) / V2_init < 1e-6, \
            f"Block-2 init mismatch: V(0)={V2_0:.6e}, V2_init={V2_init:.6e}"

    # Storage
    t_arr = [0.0]
    u1_arr=[u1_0]; psi1_arr=[psi1_0]; V1_arr=[V1_0]
    u2_arr=[u2_0]; psi2_arr=[psi2_0]; V2_arr=[V2_0]
    tL1_arr=[tau_L1_fn(0.0)]; tL2_arr=[tau_L2_fn(0.0)]

    j = _jac(V1_0, psi1_0, V2_0, psi2_0)
    tV1_arr=[j[0]]; tP1_arr=[j[1]]; GV1_arr=[j[2]]; GP1_arr=[j[3]]
    da1_arr=[j[4]]; dGa1_arr=[j[5]]
    tV2_arr=[j[6]]; tP2_arr=[j[7]]; GV2_arr=[j[8]]; GP2_arr=[j[9]]
    da2_arr=[j[10]]; dGa2_arr=[j[11]]

    t  = 0.0
    u1 = u1_0; psi1 = psi1_0
    u2 = u2_0; psi2 = psi2_0
    dt = dt0
    k1_rhs, G1_rhs, k2_rhs, G2_rhs = _rhs(u1, psi1, u2, psi2, 0.0)

    while t < T:
        if t + dt > T:
            dt = T - t

        # --- three-stage embedded RK (same scheme as single-block solver) ---
        k1_2, G1_2, k2_2, G2_2 = _rhs(
            u1 + 0.5*dt*k1_rhs,           psi1 + 0.5*dt*G1_rhs,
            u2 + 0.5*dt*k2_rhs,           psi2 + 0.5*dt*G2_rhs,
            t + 0.5*dt)
        k1_3, G1_3, k2_3, G2_3 = _rhs(
            u1 + dt*(-k1_rhs + 2.0*k1_2), psi1 + dt*(-G1_rhs + 2.0*G1_2),
            u2 + dt*(-k2_rhs + 2.0*k2_2), psi2 + dt*(-G2_rhs + 2.0*G2_2),
            t + dt)

        # 2nd-order updates
        u1_2   = u1   + dt/2.0*(k1_rhs + k1_3)
        psi1_2 = psi1 + dt/2.0*(G1_rhs + G1_3)
        u2_2   = u2   + dt/2.0*(k2_rhs + k2_3)
        psi2_2 = psi2 + dt/2.0*(G2_rhs + G2_3)
        # 3rd-order updates
        u1_3   = u1   + dt/6.0*(k1_rhs + 4.0*k1_2 + k1_3)
        psi1_3 = psi1 + dt/6.0*(G1_rhs + 4.0*G1_2 + G1_3)
        u2_3   = u2   + dt/6.0*(k2_rhs + 4.0*k2_2 + k2_3)
        psi2_3 = psi2 + dt/6.0*(G2_rhs + 4.0*G2_2 + G2_3)

        er = np.sqrt((u1_2-u1_3)**2 + (psi1_2-psi1_3)**2 +
                     (u2_2-u2_3)**2 + (psi2_2-psi2_3)**2)

        if er < tol:
            t += dt
            u1 = u1_3; psi1 = psi1_3
            u2 = u2_3; psi2 = psi2_3

            V1_new = _solve_V1(u1, psi1, u2, t)
            V2_new = _solve_V2(u1, psi2, u2, t)

            t_arr.append(t)
            u1_arr.append(u1);   psi1_arr.append(psi1); V1_arr.append(V1_new)
            u2_arr.append(u2);   psi2_arr.append(psi2); V2_arr.append(V2_new)
            tL1_arr.append(tau_L1_fn(t)); tL2_arr.append(tau_L2_fn(t))

            j = _jac(V1_new, psi1, V2_new, psi2)
            tV1_arr.append(j[0]);  tP1_arr.append(j[1])
            GV1_arr.append(j[2]);  GP1_arr.append(j[3])
            da1_arr.append(j[4]);  dGa1_arr.append(j[5])
            tV2_arr.append(j[6]);  tP2_arr.append(j[7])
            GV2_arr.append(j[8]);  GP2_arr.append(j[9])
            da2_arr.append(j[10]); dGa2_arr.append(j[11])

            k1_rhs, G1_rhs, k2_rhs, G2_rhs = _rhs(u1, psi1, u2, psi2, t)

        dt = safety * dt * (tol / er)**(1.0/3.0) if er > 0.0 else dtmax
        dt = min(dt, dtmax)

    return dict(
        t        = np.array(t_arr),
        u1       = np.array(u1_arr),  psi1 = np.array(psi1_arr), V1 = np.array(V1_arr),
        u2       = np.array(u2_arr),  psi2 = np.array(psi2_arr), V2 = np.array(V2_arr),
        tau_L1   = np.array(tL1_arr), tau_L2 = np.array(tL2_arr),
        tau_V1   = np.array(tV1_arr), tau_psi1 = np.array(tP1_arr),
        G_V1     = np.array(GV1_arr), G_psi1   = np.array(GP1_arr),
        dtau_da1 = np.array(da1_arr), dG_da1   = np.array(dGa1_arr),
        tau_V2   = np.array(tV2_arr), tau_psi2 = np.array(tP2_arr),
        G_V2     = np.array(GV2_arr), G_psi2   = np.array(GP2_arr),
        dtau_da2 = np.array(da2_arr), dG_da2   = np.array(dGa2_arr),
    )


def forward_solve_adaptive_2block_sens(M, T, u1_0, psi1_0, u2_0, psi2_0,
                                        params=('a1', 'a2', 'k0', 'k12'),
                                        V1_init=None, V2_init=None,
                                        tol=1e-9, dt0=1.0, dtmax=1e5, safety=0.9):
    """
    Adaptive-step forward solve augmented with forward sensitivity equations
    for parameters p in `params`.  Frozen-IC convention (s_x(0)=0) to match the
    inversion (CLAUDE.md).

    Sensitivities s_x_p = dx/dp are integrated alongside the nominal state with
    the same 3-stage embedded RK scheme.  The error controller monitors only the
    nominal state, so the adaptive grid is identical to forward_solve_adaptive_2block
    at the same tolerance — sensitivities are evaluated on the same grid the
    adjoint sees, making the two methods directly comparable.

    Algebraic constraint (each block):
        F_i = tau_i + eta*V_i + (k0+k12)*u_i - k12*u_{j} - tau0_i - k0*V_bg*t = 0
        => sigma_V_i = dV_i/dp = -[dF_i/dp + (k0+k12)*s_u_i - k12*s_u_j
                                   + tau_psi_i*s_psi_i] / (tau_V_i + eta)

    ODEs:
        ds_u_i/dt   = sigma_V_i
        ds_psi_i/dt = G_V_i * sigma_V_i + G_psi_i * s_psi_i
        (explicit dG/dp = 0 for aging law in all four parameters)

    Explicit dF/dp (frozen tau0_i):
        p='a1': dF1/da1 = dtau_da1, dF2/da1 = 0
        p='a2': dF1/da2 = 0,        dF2/da2 = dtau_da2
        p='k0': dF1/dk0 = u1 - V_bg*t, dF2/dk0 = u2 - V_bg*t
        p='k12':dF1/dk12 = u1 - u2,    dF2/dk12 = u2 - u1

    Returns the same dict as forward_solve_adaptive_2block, plus
        sens: dict[p, dict] with keys 's_u1','s_psi1','s_u2','s_psi2' (each array of length len(t)).

    Per-block friction parameters (a_i, N_i, b_i, dc_i, f0_i) are honoured when
    present in M; otherwise the shared keys are used as fallback via block_M.
    """
    M1 = block_M(M, 1)
    M2 = block_M(M, 2)
    k0, k12, eta = M['k0'], M['k12'], M['eta']
    V_bg = M['V_bg']
    tau_L1_fn = lambda t: M['tau0_1'] + k0 * V_bg * t
    tau_L2_fn = lambda t: M['tau0_2'] + k0 * V_bg * t

    def _solve_V1(u1, psi1, u2, t):
        rhs = tau_L1_fn(t) - (k0 + k12) * u1 + k12 * u2
        if rhs <= 0.0:
            raise ValueError(f"Block-1 force-balance RHS={rhs:.4g}<=0 at t={t:.3e}")
        def res(V): return tau_fn(V, psi1, M1) + eta * V - rhs
        return brentq(res, 1e-30, rhs / eta, xtol=1e-20, rtol=1e-10)

    def _solve_V2(u1, psi2, u2, t):
        rhs = tau_L2_fn(t) - (k0 + k12) * u2 + k12 * u1
        if rhs <= 0.0:
            raise ValueError(f"Block-2 force-balance RHS={rhs:.4g}<=0 at t={t:.3e}")
        def res(V): return tau_fn(V, psi2, M2) + eta * V - rhs
        return brentq(res, 1e-30, rhs / eta, xtol=1e-20, rtol=1e-10)

    params = tuple(params)

    def _explicit_dF_dp(p, u1, u2, t, dta1, dta2):
        if p == 'a1':  return dta1, 0.0
        if p == 'a2':  return 0.0, dta2
        if p == 'k0':  return u1 - V_bg * t, u2 - V_bg * t
        if p == 'k12': return u1 - u2,       u2 - u1
        raise KeyError(f"Unsupported parameter '{p}'")

    def _rhs_aug(u1, psi1, u2, psi2, s_state, t):
        """Returns (V1, G1, V2, G2, sens_rhs) at the given state."""
        V1 = _solve_V1(u1, psi1, u2, t)
        V2 = _solve_V2(u1, psi2, u2, t)
        G1 = G_fn(V1, psi1, M1)
        G2 = G_fn(V2, psi2, M2)

        tV1 = tau_V_fn(V1, psi1, M1); tP1 = tau_psi_fn(V1, psi1, M1)
        GV1 = G_V_fn(V1, psi1, M1);   GP1 = G_psi_fn(V1, psi1, M1)
        dta1 = dtau_da_fn(V1, psi1, M1)
        tV2 = tau_V_fn(V2, psi2, M2); tP2 = tau_psi_fn(V2, psi2, M2)
        GV2 = G_V_fn(V2, psi2, M2);   GP2 = G_psi_fn(V2, psi2, M2)
        dta2 = dtau_da_fn(V2, psi2, M2)

        D1 = tV1 + eta
        D2 = tV2 + eta

        sens_rhs = {}
        for p in params:
            su1, sp1, su2, sp2 = s_state[p]
            dF1_dp, dF2_dp = _explicit_dF_dp(p, u1, u2, t, dta1, dta2)
            sigV1 = -(dF1_dp + (k0 + k12) * su1 - k12 * su2 + tP1 * sp1) / D1
            sigV2 = -(dF2_dp + (k0 + k12) * su2 - k12 * su1 + tP2 * sp2) / D2
            sens_rhs[p] = (sigV1,
                           GV1 * sigV1 + GP1 * sp1,
                           sigV2,
                           GV2 * sigV2 + GP2 * sp2)
        return V1, G1, V2, G2, sens_rhs

    # Verify ICs
    V1_0 = _solve_V1(u1_0, psi1_0, u2_0, 0.0)
    V2_0 = _solve_V2(u1_0, psi2_0, u2_0, 0.0)
    if V1_init is not None:
        assert abs(V1_0 - V1_init) / V1_init < 1e-6
    if V2_init is not None:
        assert abs(V2_0 - V2_init) / V2_init < 1e-6

    # Frozen-IC: sensitivities start at zero
    s_state = {p: (0.0, 0.0, 0.0, 0.0) for p in params}

    # Storage
    t_arr   = [0.0]
    u1_arr  = [u1_0]; psi1_arr = [psi1_0]; V1_arr = [V1_0]
    u2_arr  = [u2_0]; psi2_arr = [psi2_0]; V2_arr = [V2_0]
    tL1_arr = [tau_L1_fn(0.0)]; tL2_arr = [tau_L2_fn(0.0)]
    sens_arr = {p: {'s_u1':  [0.0], 's_psi1': [0.0],
                    's_u2':  [0.0], 's_psi2': [0.0]} for p in params}

    t = 0.0
    u1, psi1 = u1_0, psi1_0
    u2, psi2 = u2_0, psi2_0
    dt = dt0
    V1_k1, G1_k1, V2_k1, G2_k1, s_k1 = _rhs_aug(u1, psi1, u2, psi2, s_state, 0.0)

    while t < T:
        if t + dt > T:
            dt = T - t

        # Stage-2 inputs
        u1_s2   = u1   + 0.5 * dt * V1_k1
        psi1_s2 = psi1 + 0.5 * dt * G1_k1
        u2_s2   = u2   + 0.5 * dt * V2_k1
        psi2_s2 = psi2 + 0.5 * dt * G2_k1
        s_state_s2 = {p: tuple(s_state[p][i] + 0.5 * dt * s_k1[p][i] for i in range(4))
                      for p in params}
        V1_k2, G1_k2, V2_k2, G2_k2, s_k2 = _rhs_aug(
            u1_s2, psi1_s2, u2_s2, psi2_s2, s_state_s2, t + 0.5 * dt)

        # Stage-3 inputs
        u1_s3   = u1   + dt * (-V1_k1 + 2.0 * V1_k2)
        psi1_s3 = psi1 + dt * (-G1_k1 + 2.0 * G1_k2)
        u2_s3   = u2   + dt * (-V2_k1 + 2.0 * V2_k2)
        psi2_s3 = psi2 + dt * (-G2_k1 + 2.0 * G2_k2)
        s_state_s3 = {p: tuple(s_state[p][i] + dt * (-s_k1[p][i] + 2.0 * s_k2[p][i])
                               for i in range(4))
                      for p in params}
        V1_k3, G1_k3, V2_k3, G2_k3, s_k3 = _rhs_aug(
            u1_s3, psi1_s3, u2_s3, psi2_s3, s_state_s3, t + dt)

        # 2nd / 3rd-order nominal updates (error controller only sees these)
        u1_2nd   = u1   + dt / 2.0 * (V1_k1 + V1_k3)
        psi1_2nd = psi1 + dt / 2.0 * (G1_k1 + G1_k3)
        u2_2nd   = u2   + dt / 2.0 * (V2_k1 + V2_k3)
        psi2_2nd = psi2 + dt / 2.0 * (G2_k1 + G2_k3)
        u1_3rd   = u1   + dt / 6.0 * (V1_k1 + 4.0 * V1_k2 + V1_k3)
        psi1_3rd = psi1 + dt / 6.0 * (G1_k1 + 4.0 * G1_k2 + G1_k3)
        u2_3rd   = u2   + dt / 6.0 * (V2_k1 + 4.0 * V2_k2 + V2_k3)
        psi2_3rd = psi2 + dt / 6.0 * (G2_k1 + 4.0 * G2_k2 + G2_k3)

        er = np.sqrt((u1_2nd-u1_3rd)**2 + (psi1_2nd-psi1_3rd)**2 +
                     (u2_2nd-u2_3rd)**2 + (psi2_2nd-psi2_3rd)**2)

        if er < tol:
            t += dt
            u1, psi1 = u1_3rd, psi1_3rd
            u2, psi2 = u2_3rd, psi2_3rd
            # 3rd-order sensitivity update (same weights)
            s_state = {p: tuple(s_state[p][i]
                                + dt / 6.0 * (s_k1[p][i] + 4.0 * s_k2[p][i] + s_k3[p][i])
                                for i in range(4))
                       for p in params}

            V1_new = _solve_V1(u1, psi1, u2, t)
            V2_new = _solve_V2(u1, psi2, u2, t)

            t_arr.append(t)
            u1_arr.append(u1); psi1_arr.append(psi1); V1_arr.append(V1_new)
            u2_arr.append(u2); psi2_arr.append(psi2); V2_arr.append(V2_new)
            tL1_arr.append(tau_L1_fn(t)); tL2_arr.append(tau_L2_fn(t))
            for p in params:
                sens_arr[p]['s_u1'].append(s_state[p][0])
                sens_arr[p]['s_psi1'].append(s_state[p][1])
                sens_arr[p]['s_u2'].append(s_state[p][2])
                sens_arr[p]['s_psi2'].append(s_state[p][3])

            V1_k1, G1_k1, V2_k1, G2_k1, s_k1 = _rhs_aug(
                u1, psi1, u2, psi2, s_state, t)

        dt = safety * dt * (tol / er)**(1.0/3.0) if er > 0.0 else dtmax
        dt = min(dt, dtmax)

    return dict(
        t      = np.array(t_arr),
        u1     = np.array(u1_arr),  psi1 = np.array(psi1_arr), V1 = np.array(V1_arr),
        u2     = np.array(u2_arr),  psi2 = np.array(psi2_arr), V2 = np.array(V2_arr),
        tau_L1 = np.array(tL1_arr), tau_L2 = np.array(tL2_arr),
        sens   = {p: {k: np.array(v) for k, v in d.items()} for p, d in sens_arr.items()},
    )