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
                                  tol=1e-8, dt0=1.0, dtmax=1e5, safety=0.9):
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

    # Per-step RK stage storage (filled once per accepted step). Indexing
    # convention: stage_*[j, s] is stage s of the step that maps accepted
    # state j -> j+1. Stage 0 == accepted state j; stage 2 lives at
    # accepted-state j+1's TIME but uses the RK-Heun extrapolated state
    # (not the 3rd-order accepted state).
    dt_steps   = []
    st_t       = []   # (n_steps, 3) once stacked
    st_u1=[];  st_psi1=[];  st_V1=[]
    st_u2=[];  st_psi2=[];  st_V2=[]
    st_tV1=[]; st_tP1=[];   st_GV1=[]; st_GP1=[]
    st_da1=[]; st_dGa1=[]
    st_tV2=[]; st_tP2=[];   st_GV2=[]; st_GP2=[]
    st_da2=[]; st_dGa2=[]

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
            # ---- capture per-step stage data BEFORE advancing state ----
            # Stage 1: at the accepted state (t, u1, psi1, u2, psi2);
            # V comes from cached k1_rhs/k2_rhs.
            s1_t = t
            s1_u1, s1_psi1, s1_V1 = u1, psi1, k1_rhs
            s1_u2, s1_psi2, s1_V2 = u2, psi2, k2_rhs
            j_s1 = _jac(s1_V1, s1_psi1, s1_V2, s1_psi2)
            # Stage 2: at the RK mid-stage state, t + 0.5*dt.
            s2_t   = t + 0.5*dt
            s2_u1  = u1   + 0.5*dt*k1_rhs
            s2_psi1= psi1 + 0.5*dt*G1_rhs
            s2_u2  = u2   + 0.5*dt*k2_rhs
            s2_psi2= psi2 + 0.5*dt*G2_rhs
            s2_V1, s2_V2 = k1_2, k2_2      # _rhs returns V at that state
            j_s2 = _jac(s2_V1, s2_psi1, s2_V2, s2_psi2)
            # Stage 3: at the RK-Heun extrapolated end state, t + dt.
            # (NOT the 3rd-order accepted state used below.)
            s3_t   = t + dt
            s3_u1  = u1   + dt*(-k1_rhs + 2.0*k1_2)
            s3_psi1= psi1 + dt*(-G1_rhs + 2.0*G1_2)
            s3_u2  = u2   + dt*(-k2_rhs + 2.0*k2_2)
            s3_psi2= psi2 + dt*(-G2_rhs + 2.0*G2_2)
            s3_V1, s3_V2 = k1_3, k2_3
            j_s3 = _jac(s3_V1, s3_psi1, s3_V2, s3_psi2)

            dt_steps.append(dt)
            st_t.append((s1_t, s2_t, s3_t))
            st_u1.append((s1_u1, s2_u1, s3_u1))
            st_psi1.append((s1_psi1, s2_psi1, s3_psi1))
            st_V1.append((s1_V1, s2_V1, s3_V1))
            st_u2.append((s1_u2, s2_u2, s3_u2))
            st_psi2.append((s1_psi2, s2_psi2, s3_psi2))
            st_V2.append((s1_V2, s2_V2, s3_V2))
            st_tV1.append((j_s1[0], j_s2[0], j_s3[0]))
            st_tP1.append((j_s1[1], j_s2[1], j_s3[1]))
            st_GV1.append((j_s1[2], j_s2[2], j_s3[2]))
            st_GP1.append((j_s1[3], j_s2[3], j_s3[3]))
            st_da1.append((j_s1[4], j_s2[4], j_s3[4]))
            st_dGa1.append((j_s1[5], j_s2[5], j_s3[5]))
            st_tV2.append((j_s1[6], j_s2[6], j_s3[6]))
            st_tP2.append((j_s1[7], j_s2[7], j_s3[7]))
            st_GV2.append((j_s1[8], j_s2[8], j_s3[8]))
            st_GP2.append((j_s1[9], j_s2[9], j_s3[9]))
            st_da2.append((j_s1[10], j_s2[10], j_s3[10]))
            st_dGa2.append((j_s1[11], j_s2[11], j_s3[11]))

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
        # Per-step Bogacki-Shampine RK3 stage data, shape (n_steps, 3).
        # Stage indices: 0=alpha=0 (start), 1=alpha=1/2 (mid), 2=alpha=1 (Heun-end).
        # Lets the adjoint use the actual stage Jacobians instead of linearly
        # interpolating accepted-endpoint Jacobians at alpha=1/2.
        dt_arr        = np.array(dt_steps),
        stage_t       = np.array(st_t),
        stage_u1      = np.array(st_u1),     stage_psi1   = np.array(st_psi1),
        stage_V1      = np.array(st_V1),
        stage_u2      = np.array(st_u2),     stage_psi2   = np.array(st_psi2),
        stage_V2      = np.array(st_V2),
        stage_tau_V1  = np.array(st_tV1),    stage_tau_psi1 = np.array(st_tP1),
        stage_G_V1    = np.array(st_GV1),    stage_G_psi1   = np.array(st_GP1),
        stage_dtau_da1= np.array(st_da1),    stage_dG_da1   = np.array(st_dGa1),
        stage_tau_V2  = np.array(st_tV2),    stage_tau_psi2 = np.array(st_tP2),
        stage_G_V2    = np.array(st_GV2),    stage_G_psi2   = np.array(st_GP2),
        stage_dtau_da2= np.array(st_da2),    stage_dG_da2   = np.array(st_dGa2),
    )