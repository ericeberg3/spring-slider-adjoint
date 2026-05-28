"""
Tests and diagnostics for the two-block adjoint solver.

Extracted from slip_adjoint_double_springslider.ipynb so the notebook can focus
on the inversion workflow. Two entry points:

- validate_gradient_vs_fd(...): adjoint gradient vs centred-difference FD,
  with optional Forward-Euler and implicit (Radau) adjoint comparisons.
- run_J_landscape(...): J(p) scan over one or more parameters under several
  smoothing levels, with optional adjoint-gradient arrows.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

from friction_derivs import (
    tau_V_fn, tau_psi_fn, G_V_fn, G_psi_fn, G_fn,
    dtau_da_fn, dG_da_fn, solve_V_algebraic,
    setup_initial_conditions_2block, make_smoothing_matrix,
)
from adapt_fwd_solve import forward_solve_adaptive_2block
from adjoint_solve import adjoint_solve_2block
from compute_obj import (
    compute_grad_a1, compute_grad_a2, compute_grad_k0, compute_grad_k12,
)
from landscape_worker import evaluate_landscape_point, init_worker


# ======================================================================
# Forward-Euler two-block forward solver (used only by the FD validation)
# ======================================================================
def _forward_solve_euler_2block(M, T, u1_0, psi1_0, u2_0, psi2_0, dt,
                                 V1_init=None, V2_init=None):
    """Plain Forward-Euler two-block solver, used as a discretisation
    sanity check against the adaptive RK forward solver."""
    k0, k12 = M['k0'], M['k12']
    M1 = {**M, 'a': M['a1'], 'k': k0 + k12}
    M2 = {**M, 'a': M['a2'], 'k': k0 + k12}

    def _tL1(u1, u2, t): return M['tau0_1'] + k0 * M['V_bg'] * t + k12 * u2
    def _tL2(u1, u2, t): return M['tau0_2'] + k0 * M['V_bg'] * t + k12 * u1
    def _jac1(V, psi):
        return (tau_V_fn(V, psi, M1), tau_psi_fn(V, psi, M1),
                G_V_fn(V, psi, M1),   G_psi_fn(V, psi, M1),
                dtau_da_fn(V, psi, M1), dG_da_fn(V, psi, M1))
    def _jac2(V, psi):
        return (tau_V_fn(V, psi, M2), tau_psi_fn(V, psi, M2),
                G_V_fn(V, psi, M2),   G_psi_fn(V, psi, M2),
                dtau_da_fn(V, psi, M2), dG_da_fn(V, psi, M2))

    tL1_0 = _tL1(u1_0, u2_0, 0.0); tL2_0 = _tL2(u1_0, u2_0, 0.0)
    V1c = (V1_init if V1_init is not None
           else solve_V_algebraic(u1_0, psi1_0, M1, tL1_0))
    V2c = (V2_init if V2_init is not None
           else solve_V_algebraic(u2_0, psi2_0, M2, tL2_0))
    j1_0 = _jac1(V1c, psi1_0); j2_0 = _jac2(V2c, psi2_0)

    t_arr = [0.0]
    u1_arr=[u1_0]; psi1_arr=[psi1_0]; V1_arr=[V1c]
    u2_arr=[u2_0]; psi2_arr=[psi2_0]; V2_arr=[V2c]
    tL1_arr=[tL1_0]; tL2_arr=[tL2_0]
    tV1=[j1_0[0]]; tP1=[j1_0[1]]; GV1=[j1_0[2]]; GP1=[j1_0[3]]
    da1=[j1_0[4]]; dGa1=[j1_0[5]]
    tV2=[j2_0[0]]; tP2=[j2_0[1]]; GV2=[j2_0[2]]; GP2=[j2_0[3]]
    da2=[j2_0[4]]; dGa2=[j2_0[5]]

    u1c, psi1c, u2c, psi2c, tc = u1_0, psi1_0, u2_0, psi2_0, 0.0
    while tc < T - 1e-12 * T:
        dts = min(dt, T - tc)
        G1 = G_fn(V1_arr[-1], psi1c, M1)
        G2 = G_fn(V2_arr[-1], psi2c, M2)
        u1n  = u1c  + dts * V1_arr[-1]; psi1n = psi1c + dts * G1
        u2n  = u2c  + dts * V2_arr[-1]; psi2n = psi2c + dts * G2
        tn   = tc   + dts
        tL1n = _tL1(u1n, u2n, tn); tL2n = _tL2(u1n, u2n, tn)
        V1n  = solve_V_algebraic(u1n, psi1n, M1, tL1n)
        V2n  = solve_V_algebraic(u2n, psi2n, M2, tL2n)
        j1n  = _jac1(V1n, psi1n); j2n = _jac2(V2n, psi2n)

        t_arr.append(tn)
        u1_arr.append(u1n);   psi1_arr.append(psi1n); V1_arr.append(V1n)
        u2_arr.append(u2n);   psi2_arr.append(psi2n); V2_arr.append(V2n)
        tL1_arr.append(tL1n); tL2_arr.append(tL2n)
        tV1.append(j1n[0]); tP1.append(j1n[1]); GV1.append(j1n[2]); GP1.append(j1n[3])
        da1.append(j1n[4]); dGa1.append(j1n[5])
        tV2.append(j2n[0]); tP2.append(j2n[1]); GV2.append(j2n[2]); GP2.append(j2n[3])
        da2.append(j2n[4]); dGa2.append(j2n[5])
        u1c, psi1c, u2c, psi2c, tc = u1n, psi1n, u2n, psi2n, tn

    return dict(
        t=np.array(t_arr),
        u1=np.array(u1_arr),  psi1=np.array(psi1_arr), V1=np.array(V1_arr),
        u2=np.array(u2_arr),  psi2=np.array(psi2_arr), V2=np.array(V2_arr),
        tau_L1=np.array(tL1_arr), tau_L2=np.array(tL2_arr),
        tau_V1=np.array(tV1),  tau_psi1=np.array(tP1),
        G_V1=np.array(GV1),    G_psi1=np.array(GP1),
        dtau_da1=np.array(da1),  dG_da1=np.array(dGa1),
        tau_V2=np.array(tV2),  tau_psi2=np.array(tP2),
        G_V2=np.array(GV2),    G_psi2=np.array(GP2),
        dtau_da2=np.array(da2),  dG_da2=np.array(dGa2),
    )


def _trapz_weights(t):
    w = np.zeros(len(t)); w[:-1] += np.diff(t); w[1:] += np.diff(t)
    return 0.5 * w


def _interp_adjoint_scatter(v, t_src, t_dst):
    """P^T v: scatter ref-grid values back to native grid via add-at."""
    idx = np.searchsorted(t_src, t_dst, side='right') - 1
    idx = np.clip(idx, 0, len(t_src) - 2)
    alpha = np.clip((t_dst - t_src[idx]) / (t_src[idx + 1] - t_src[idx]), 0.0, 1.0)
    result = np.zeros(len(t_src))
    np.add.at(result, idx,     (1.0 - alpha) * v)
    np.add.at(result, idx + 1, alpha          * v)
    return result


# ======================================================================
# Adjoint vs FD gradient validation
# ======================================================================
def validate_gradient_vs_fd(M_true, T, V1_init, V2_init,
                            t_obs_arr, u1_obs, u2_obs,
                            check_param='a2',
                            run_fe=False,
                            run_implicit_adj=False):
    """Compare adjoint-derived gradient to centred-FD reference.

    Parameters
    ----------
    M_true : dict
        Reference material model. Used to build the perturbed nominal model
        (true + 10% offset in check_param) and the FD perturbations.
    T : float
        Total simulation horizon (s).
    V1_init, V2_init : float
        Initial-guess velocities for solve_V_algebraic.
    t_obs_arr, u1_obs, u2_obs : ndarray
        Synthetic observations on a fine grid.
    check_param : {'a1','a2','k0','k12'}
        Which parameter to validate.
    run_fe : bool
        If True, also run a Forward-Euler forward + adjoint pass and use
        FE-derived FD as the reference gradient. Much slower.
    run_implicit_adj : bool
        If True, additionally run the stiff implicit (Radau) two-block
        adjoint and include it in the comparison.

    Returns
    -------
    dict with the gradient values and relative errors.
    """
    _grad_fns_fd = {
        'a1': compute_grad_a1, 'a2': compute_grad_a2,
        'k0': compute_grad_k0, 'k12': compute_grad_k12,
    }

    T_test   = T * 1
    dt_euler = T_test / 70000.0
    n_euler  = int(round(T_test / dt_euler))
    sigma_test = 0 * T_test

    print(f"Test window : T_test  = {T_test:.3e} s  ({T_test/86400:.1f} days)")
    if run_fe:
        print(f"FE timestep : dt      = {dt_euler:.3e} s  =>  {n_euler} steps")
    else:
        print("FE timestep : skipped  (run_fe=False)")
    print(f"Checking gradient for parameter: '{check_param}'")

    # Nominal model: true parameters with 10% offset in the checked param
    M_fd = dict(M_true)
    M_fd[check_param] = M_true[check_param] * 1.1
    if check_param in ('a1', 'a2'):
        setup_initial_conditions_2block(M_fd, V1_init=V1_init, V2_init=V2_init)

    eps_fd = abs(M_fd[check_param]) * 1e-8
    M_p = dict(M_fd); M_p[check_param] = M_fd[check_param] + eps_fd
    M_m = dict(M_fd); M_m[check_param] = M_fd[check_param] - eps_fd
    if check_param in ('a1', 'a2'):
        setup_initial_conditions_2block(M_p, V1_init=V1_init, V2_init=V2_init)
        setup_initial_conditions_2block(M_m, V1_init=V1_init, V2_init=V2_init)

    print(f"  Nominal {check_param} = {M_fd[check_param]:.6g}  "
          f"(1.1 x true = {M_true[check_param]:.6g})")
    print(f"  FD perturbation: eps = {eps_fd:.2e}")

    # Observations clipped to [0, T_test]
    mask_obs    = t_obs_arr <= T_test + 1.0
    t_obs_test  = t_obs_arr[mask_obs]
    u1_obs_test = u1_obs[mask_obs]
    u2_obs_test = u2_obs[mask_obs]

    # Fixed reference grid (uniform -> plain Gaussian, no trapz weights)
    n_ref_fd      = min(int(T_test / ((sigma_test + 0.01) / 20)) + 1, 5000)
    t_ref_fd      = np.linspace(0.0, T_test, n_ref_fd)
    S_fixed_fd    = make_smoothing_matrix(t_ref_fd, sigma_test)
    u1_obs_ref_fd = np.interp(t_ref_fd, t_obs_test, u1_obs_test)
    u2_obs_ref_fd = np.interp(t_ref_fd, t_obs_test, u2_obs_test)

    def _J(fwd):
        u1r = np.interp(t_ref_fd, fwd['t'], fwd['u1'])
        u2r = np.interp(t_ref_fd, fwd['t'], fwd['u2'])
        r1  = S_fixed_fd @ u1r - S_fixed_fd @ u1_obs_ref_fd
        r2  = S_fixed_fd @ u2r - S_fixed_fd @ u2_obs_ref_fd
        return 0.5 * np.trapz(r1**2 + r2**2, t_ref_fd)

    w_ref_fd = _trapz_weights(t_ref_fd)

    def _sm_native(fwd):
        u1r  = np.interp(t_ref_fd, fwd['t'], fwd['u1'])
        u2r  = np.interp(t_ref_fd, fwd['t'], fwd['u2'])
        res1 = S_fixed_fd @ u1r - S_fixed_fd @ u1_obs_ref_fd
        res2 = S_fixed_fd @ u2r - S_fixed_fd @ u2_obs_ref_fd
        sm1_ref = S_fixed_fd.T @ (w_ref_fd * res1)
        sm2_ref = S_fixed_fd.T @ (w_ref_fd * res2)
        w_nat = _trapz_weights(fwd['t'])
        sm1 = _interp_adjoint_scatter(sm1_ref, fwd['t'], t_ref_fd) / w_nat
        sm2 = _interp_adjoint_scatter(sm2_ref, fwd['t'], t_ref_fd) / w_nat
        return sm1, sm2

    # Initial conditions for validation (from M_true, frozen)
    u1_0_fd, psi1_0_fd, _, u2_0_fd, psi2_0_fd, _ = setup_initial_conditions_2block(
        dict(M_true), V1_init=V1_init, V2_init=V2_init)

    # FE forward
    if run_fe:
        print(f"\n[1/6] FE forward solve  ({check_param} = {M_fd[check_param]:.5g}) ...")
        fwd_fe = _forward_solve_euler_2block(M_fd, T_test, u1_0_fd, psi1_0_fd,
                                              u2_0_fd, psi2_0_fd, dt_euler,
                                              V1_init=V1_init, V2_init=V2_init)
        print(f"      {len(fwd_fe['t'])-1} steps  |  V1(T)={fwd_fe['V1'][-1]:.4e}  V2(T)={fwd_fe['V2'][-1]:.4e} m/s")
    else:
        print("\n[1/6] FE forward solve  -- skipped (run_fe=False)")

    print(f"[2/6] Adaptive RK forward solve ...")
    fwd_ad = forward_solve_adaptive_2block(M_fd, T_test, u1_0_fd, psi1_0_fd,
                                            u2_0_fd, psi2_0_fd)
    print(f"      {len(fwd_ad['t'])-1} steps  |  V1(T)={fwd_ad['V1'][-1]:.4e}  V2(T)={fwd_ad['V2'][-1]:.4e} m/s")

    if run_fe:
        J_fe = _J(fwd_fe)
    J_ad = _J(fwd_ad)

    if run_fe:
        print(f"[3/6] FE adjoint solve ...")
        sm1_fe, sm2_fe = _sm_native(fwd_fe)
        adj_fe = adjoint_solve_2block(fwd_fe, None, None, None, M_fd, sigma_test,
                                       smooth_misfit1=sm1_fe, smooth_misfit2=sm2_fe)
        grad_adj_fe = _grad_fns_fd[check_param](fwd_fe, adj_fe, M_fd)
    else:
        print("[3/6] FE adjoint solve  -- skipped (run_fe=False)")

    print(f"[4/6] Adaptive adjoint solve (explicit RK3) ...")
    sm1_ad, sm2_ad = _sm_native(fwd_ad)
    adj_ad = adjoint_solve_2block(fwd_ad, None, None, None, M_fd, sigma_test,
                                   smooth_misfit1=sm1_ad, smooth_misfit2=sm2_ad)
    grad_adj_ad = _grad_fns_fd[check_param](fwd_ad, adj_ad, M_fd)

    if run_implicit_adj:
        print(f"[4b/6] Adaptive adjoint solve (implicit Radau) ...")
        adj_ad_imp = adjoint_solve_2block(fwd_ad, None, None, None, M_fd, sigma_test,
                                           smooth_misfit1=sm1_ad, smooth_misfit2=sm2_ad,
                                           use_implicit=True)
        grad_adj_ad_imp = _grad_fns_fd[check_param](fwd_ad, adj_ad_imp, M_fd)
    else:
        print("[4b/6] Implicit Radau adjoint solve -- skipped (run_implicit_adj=False)")

    M_p['tau0_1'] = M_fd['tau0_1']; M_p['tau0_2'] = M_fd['tau0_2']
    M_m['tau0_1'] = M_fd['tau0_1']; M_m['tau0_2'] = M_fd['tau0_2']

    if run_fe:
        print(f"[5/6] FD gradient via FE  (eps = {eps_fd:.2e}) ...")
        fwd_fe_p = _forward_solve_euler_2block(M_p, T_test, u1_0_fd, psi1_0_fd,
                                                u2_0_fd, psi2_0_fd, dt_euler)
        fwd_fe_m = _forward_solve_euler_2block(M_m, T_test, u1_0_fd, psi1_0_fd,
                                                u2_0_fd, psi2_0_fd, dt_euler)
        grad_fd_fe = (_J(fwd_fe_p) - _J(fwd_fe_m)) / (2.0 * eps_fd)
    else:
        print(f"[5/6] FD gradient via FE  -- skipped (run_fe=False)")

    print(f"[6/6] FD gradient via adaptive RK  (eps = {eps_fd:.2e}) ...")
    fwd_p = forward_solve_adaptive_2block(M_p, T_test, u1_0_fd, psi1_0_fd,
                                           u2_0_fd, psi2_0_fd)
    fwd_m = forward_solve_adaptive_2block(M_m, T_test, u1_0_fd, psi1_0_fd,
                                           u2_0_fd, psi2_0_fd)
    grad_fd_ad = (_J(fwd_p) - _J(fwd_m)) / (2.0 * eps_fd)

    # Summary table
    p = check_param
    grad_ref = grad_fd_fe if run_fe else grad_fd_ad

    rel_ad    = abs(grad_adj_ad - grad_ref) / (abs(grad_ref) + 1e-30)
    rel_fd_ad = abs(grad_fd_ad  - grad_ref) / (abs(grad_ref) + 1e-30)
    if run_implicit_adj:
        rel_ad_imp = abs(grad_adj_ad_imp - grad_ref) / (abs(grad_ref) + 1e-30)

    print(f"\n{'':->76}")
    if run_fe:
        rel_fe = abs(grad_adj_fe - grad_fd_fe) / (abs(grad_fd_fe) + 1e-30)
        print(f"{'Quantity':<38}  {'FE forward':>14}  {'Adaptive fwd':>14}")
        print(f"{'':->76}")
        print(f"{'J':<38}  {J_fe:14.6e}  {J_ad:14.6e}")
        print(f"{f'dJ/d{p}  (adjoint)':<38}  {grad_adj_fe:14.6e}  {grad_adj_ad:14.6e}")
        print(f"{f'dJ/d{p}  (FD)':<38}  {grad_fd_fe:14.6e}  {grad_fd_ad:14.6e}")
        print(f"{'relative error vs FD (FE)':<38}  {rel_fe:14.2e}  {rel_ad:14.2e}")
        print(f"{'PASS / FAIL  (< 5 %)':<38}  {'PASS' if rel_fe<0.05 else 'FAIL':>14}  "
              f"{'PASS' if rel_ad<0.05 else 'FAIL':>14}")
        print(f"{'FD (adaptive) vs FD (FE) rel err':<38}  {'---':>14}  {rel_fd_ad:14.2e}")
    else:
        print(f"{'Quantity':<38}  {'Adaptive fwd':>14}")
        print(f"{'':->76}")
        print(f"{'J':<38}  {J_ad:14.6e}")
        print(f"{f'dJ/d{p}  (adjoint)':<38}  {grad_adj_ad:14.6e}")
        print(f"{f'dJ/d{p}  (FD)':<38}  {grad_fd_ad:14.6e}")
        print(f"{'relative error vs FD (adaptive)':<38}  {rel_ad:14.2e}")
        print(f"{'PASS / FAIL  (< 5 %)':<38}  {'PASS' if rel_ad<0.05 else 'FAIL':>14}")
    if run_implicit_adj:
        print(f"{f'dJ/d{p}  (adjoint, Radau)':<38}  {grad_adj_ad_imp:14.6e}")
        print(f"{'relative error vs FD (Radau)':<38}  {rel_ad_imp:14.2e}")
        print(f"{'PASS / FAIL  (< 5 %)  [Radau]':<38}  {'PASS' if rel_ad_imp<0.05 else 'FAIL':>14}")
    print(f"{'':->76}")

    # Plot A: Forward solution overlay (V, u, psi for both blocks)
    kw_ad1 = dict(color='C3', marker='.', ms=2, lw=0, label=f'Adaptive Block 1  ({len(fwd_ad["t"])-1} steps)')
    kw_ad2 = dict(color='C2', marker='.', ms=2, lw=0, label=f'Adaptive Block 2')

    fig_a, axs_a = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    if run_fe:
        kw_fe1 = dict(color='C0', lw=1.8, label=f'FE Block 1  ({n_euler} steps)')
        kw_fe2 = dict(color='C1', lw=1.8, label=f'FE Block 2')
        axs_a[0].semilogy(fwd_fe['t'], fwd_fe['V1'], **kw_fe1)
        axs_a[0].semilogy(fwd_fe['t'], fwd_fe['V2'], **kw_fe2)
        axs_a[1].plot(fwd_fe['t'], fwd_fe['u1'], **kw_fe1)
        axs_a[1].plot(fwd_fe['t'], fwd_fe['u2'], **kw_fe2)
        axs_a[2].plot(fwd_fe['t'], fwd_fe['psi1'], **kw_fe1)
        axs_a[2].plot(fwd_fe['t'], fwd_fe['psi2'], **kw_fe2)
    axs_a[0].semilogy(fwd_ad['t'], fwd_ad['V1'], **kw_ad1)
    axs_a[0].semilogy(fwd_ad['t'], fwd_ad['V2'], **kw_ad2)
    axs_a[0].set_ylabel('V (m/s)'); axs_a[0].legend(fontsize=8); axs_a[0].grid(True,ls=':',lw=0.5)
    axs_a[0].set_title(f'Forward overlay -- {p}={M_fd[p]:.5g},  T={T_test:.1e} s')
    axs_a[1].plot(fwd_ad['t'], fwd_ad['u1'], **kw_ad1)
    axs_a[1].plot(fwd_ad['t'], fwd_ad['u2'], **kw_ad2)
    axs_a[1].set_ylabel('u (m)'); axs_a[1].legend(fontsize=8); axs_a[1].grid(True,ls=':',lw=0.5)
    axs_a[2].plot(fwd_ad['t'], fwd_ad['psi1'], **kw_ad1)
    axs_a[2].plot(fwd_ad['t'], fwd_ad['psi2'], **kw_ad2)
    axs_a[2].set_ylabel('psi'); axs_a[2].set_xlabel('Time (s)')
    axs_a[2].legend(fontsize=8); axs_a[2].grid(True,ls=':',lw=0.5)
    plt.tight_layout(); plt.show()

    # Plot B: Adjoint solution overlay (pu, r, lambda) + diagnostics
    _adj_panels = []
    if run_fe:
        _adj_panels.append((adj_fe, fwd_fe, 'FE', 'C0', 'C1'))
    _adj_panels.append((adj_ad, fwd_ad, 'Adaptive (RK3)', 'C3', 'C2'))
    if run_implicit_adj:
        _adj_panels.append((adj_ad_imp, fwd_ad, 'Adaptive (Radau)', 'C4', 'C5'))

    fig_b, axs_b = plt.subplots(5, len(_adj_panels), figsize=(6*len(_adj_panels), 13), sharex=True,
                                 squeeze=False)
    _eta_b = M_fd['eta']
    for ax_col, (adj, fwd, lbl, c1, c2) in enumerate(_adj_panels):
        axs_b[0, ax_col].plot(adj['t'], adj['pu1'], color=c1, lw=1.5, label='Block 1')
        axs_b[0, ax_col].plot(adj['t'], adj['pu2'], color=c2, lw=1.5, label='Block 2')
        axs_b[0, ax_col].set_title(f'{lbl} adjoint'); axs_b[0, ax_col].set_ylabel('pu (u+)')
        axs_b[0, ax_col].legend(fontsize=8); axs_b[0, ax_col].grid(True,ls=':',lw=0.5)

        axs_b[1, ax_col].plot(adj['t'], adj['r1'], color=c1, lw=1.5, label='Block 1')
        axs_b[1, ax_col].plot(adj['t'], adj['r2'], color=c2, lw=1.5, label='Block 2')
        axs_b[1, ax_col].set_ylabel('r (psi+)')
        axs_b[1, ax_col].legend(fontsize=8); axs_b[1, ax_col].grid(True,ls=':',lw=0.5)

        axs_b[2, ax_col].plot(adj['t'], adj['lam1'], color=c1, lw=1.5, label='lam1')
        axs_b[2, ax_col].plot(adj['t'], adj['lam2'], color=c2, lw=1.5, label='lam2')
        axs_b[2, ax_col].set_ylabel('lambda')
        axs_b[2, ax_col].legend(fontsize=8); axs_b[2, ax_col].grid(True,ls=':',lw=0.5)

        _D1 = fwd['tau_V1'] + _eta_b
        _D2 = fwd['tau_V2'] + _eta_b
        axs_b[3, ax_col].semilogy(fwd['t'], np.abs(_D1), color=c1, lw=1.5, label='Block 1')
        axs_b[3, ax_col].semilogy(fwd['t'], np.abs(_D2), color=c2, lw=1.5, label='Block 2')
        axs_b[3, ax_col].axhline(_eta_b, color='k', ls='--', lw=0.8, label=f'eta={_eta_b:.2g}')
        axs_b[3, ax_col].set_ylabel(r'$|\tau_V+\eta|$')
        axs_b[3, ax_col].legend(fontsize=8); axs_b[3, ax_col].grid(True,ls=':',lw=0.5, which='both')

        axs_b[4, ax_col].plot(fwd['t'], fwd['dtau_da1'], color=c1, lw=1.5, label='Block 1')
        axs_b[4, ax_col].plot(fwd['t'], fwd['dtau_da2'], color=c2, lw=1.5, label='Block 2')
        axs_b[4, ax_col].set_ylabel(r'$\partial\tau/\partial a$'); axs_b[4, ax_col].set_xlabel('Time (s)')
        axs_b[4, ax_col].legend(fontsize=8); axs_b[4, ax_col].grid(True,ls=':',lw=0.5)
    plt.tight_layout(); plt.show()

    # Plot C: Gradient bar chart
    fig_c, ax_c = plt.subplots(figsize=(9, 4))
    if run_fe:
        labels_c = ['FD\n(FE fwd)', 'FD\n(Adaptive fwd)',
                    'Adjoint\n(FE fwd)', 'Adjoint RK3\n(Adaptive fwd)']
        vals_c   = [grad_fd_fe, grad_fd_ad, grad_adj_fe, grad_adj_ad]
        colors_c = ['gray', 'C0', 'C3', 'C2']
        ref_label = f'FD (FE) reference = {grad_fd_fe:.3e}'
        rel_title = (f'FE adj rel err={rel_fe:.2e}  |  Adaptive adj rel err={rel_ad:.2e}  |  '
                     f'Adaptive FD rel err={rel_fd_ad:.2e}')
    else:
        labels_c = ['FD\n(Adaptive fwd)', 'Adjoint RK3\n(Adaptive fwd)']
        vals_c   = [grad_fd_ad, grad_adj_ad]
        colors_c = ['C0', 'C2']
        ref_label = f'FD (Adaptive) reference = {grad_fd_ad:.3e}'
        rel_title = f'Adaptive adj rel err={rel_ad:.2e}'
    if run_implicit_adj:
        labels_c.append('Adjoint Radau\n(Adaptive fwd)')
        vals_c.append(grad_adj_ad_imp)
        colors_c.append('C4')
        rel_title += f'  |  Radau adj rel err={rel_ad_imp:.2e}'

    bars_c = ax_c.bar(labels_c, vals_c, color=colors_c, alpha=0.75, edgecolor='k', lw=0.8)
    ax_c.axhline(grad_ref, color='k', ls='--', lw=1.0, label=ref_label)
    for bar, v in zip(bars_c, vals_c):
        offset = 0.03 * max(abs(grad_ref), 1e-30)
        va = 'bottom' if v >= 0 else 'top'
        ax_c.text(bar.get_x()+bar.get_width()/2, v+(offset if v>=0 else -offset),
                  f'{v:.2e}', ha='center', va=va, fontsize=9)
    ax_c.set_ylabel(f'dJ/d{p}')
    ax_c.set_title(f'Gradient comparison   {p}={M_fd[p]:.5g},  T={T_test:.1e} s\n{rel_title}')
    ax_c.legend(fontsize=9); ax_c.grid(True, ls=':', lw=0.5, axis='y')
    plt.tight_layout(); plt.show()

    out = {
        'grad_adj_ad': grad_adj_ad,
        'grad_fd_ad':  grad_fd_ad,
        'rel_ad':      rel_ad,
        'J_ad':        J_ad,
    }
    if run_fe:
        out.update(grad_adj_fe=grad_adj_fe, grad_fd_fe=grad_fd_fe, rel_fe=rel_fe, J_fe=J_fe)
    if run_implicit_adj:
        out.update(grad_adj_ad_imp=grad_adj_ad_imp, rel_ad_imp=rel_ad_imp)
    return out


# ======================================================================
# Objective-function landscape scan (parallel)
# ======================================================================
def run_J_landscape(M_true, T, V1_init, V2_init,
                    t_obs_arr, u1_obs, u2_obs,
                    landscape_params=('a2',),
                    compute_gradient=False,
                    use_parallel=True,
                    n_workers=None,
                    save_dir='Figures'):
    """Scan J(p) over each parameter in landscape_params under several
    smoothing levels. Optionally overlays adjoint-gradient arrows.

    Returns a dict keyed by parameter name containing the scan arrays
    (p_scan, J per smoothing label, optional gradient per smoothing label).
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    sigma_medium    = 0.01 * T
    sigma_heavy     = 0.3  * T
    sigma_inversion = 0.1  * T

    t_ref_land   = np.linspace(0.0, T, 1000)
    S_fixed_land = make_smoothing_matrix(t_ref_land, sigma_inversion)

    smoothing_cases = [
        ('No smoothing (identity)',                            None,            'C0', 'native'),
        (f'Medium  ($\\sigma$ = {sigma_medium:.1e} s)',        sigma_medium,    'C1', 'native'),
        (f'Heavy   ($\\sigma$ = {sigma_heavy:.1e} s)',         sigma_heavy,     'C2', 'native'),
        (f'Inversion ($\\sigma$ = {sigma_inversion:.1e} s, fixed t_ref)',
                                                               sigma_inversion, 'C3', 'inversion'),
    ]

    _sigmas_for_worker = []
    for lbl, s, _, kind in smoothing_cases:
        if kind == 'inversion':
            _sigmas_for_worker.append((lbl, s, t_ref_land, S_fixed_land))
        else:
            _sigmas_for_worker.append((lbl, s))

    _scan_config = {
        'a1': {
            'vals':     np.linspace(M_true['a1'] * 0.85, M_true['a1'] * 1.15, 36),
            'true_val': M_true['a1'],
            'xlabel':   '$a_1$',
            'title_J':  '$J(a_1)$',
        },
        'a2': {
            'vals':     np.linspace(M_true['a2'] * 0.85, M_true['a2'] * 1.15, 30),
            'true_val': M_true['a2'],
            'xlabel':   '$a_2$',
            'title_J':  '$J(a_2)$',
        },
        'k0': {
            'vals':     np.linspace(0.7 * M_true['k0'], 1.3 * M_true['k0'], 30),
            'true_val': M_true['k0'],
            'xlabel':   '$k_0$  (MPa/m)',
            'title_J':  '$J(k_0)$',
        },
        'k12': {
            'vals':     np.linspace(0.7 * M_true['k12'], 1.3 * M_true['k12'], 20),
            'true_val': M_true['k12'],
            'xlabel':   '$k_{12}$  (MPa/m)',
            'title_J':  '$J(k_{12})$',
        },
    }

    results = {}

    for scan_param in landscape_params:
        cfg     = _scan_config[scan_param]
        p_scan  = cfg['vals']
        p_true  = cfg['true_val']

        print(f"\n{'='*60}")
        print(f"Landscape: {scan_param}  (true = {p_true:.5g})")
        _mode = f"on {n_workers} worker(s)" if use_parallel else "serially (use_parallel=False)"
        print(f"Evaluating {len(p_scan)} parameter values {_mode}"
              f"{'  (with adjoint gradients)' if compute_gradient else ''} ...")

        # Fixed-IC `u_const` for the inversion-style landscape curve.
        # Built from the same initial-guess a (1.2*true) that fun_and_grad uses,
        # so the "Inversion" curve here matches the optimizer's J exactly.
        _M0_land = dict(M_true)
        if scan_param in ('a1', 'a2'):
            _M0_land[scan_param] = M_true[scan_param] * 1.2
        _u1_0_land, _psi1_0_land, _, _u2_0_land, _psi2_0_land, _ = \
            setup_initial_conditions_2block(_M0_land, V1_init=V1_init, V2_init=V2_init)
        u_const = (_u1_0_land, _psi1_0_land, _u2_0_land, _psi2_0_land)

        point_results = [None] * len(p_scan)

        t_start = time.time()
        if not use_parallel or n_workers == 1:
            for i, p_val in enumerate(p_scan):
                point_results[i] = evaluate_landscape_point(
                    p_val, scan_param, M_true, T, V1_init, V2_init, u_const,
                    t_obs_arr, u1_obs, u2_obs,
                    _sigmas_for_worker, compute_gradient)
                if (i + 1) % 8 == 0 or i == len(p_scan) - 1:
                    print(f"  [{i+1}/{len(p_scan)}]  {scan_param}={p_val:.5g}")
        else:
            project_dir = os.getcwd()
            with ProcessPoolExecutor(max_workers=n_workers,
                                      initializer=init_worker,
                                      initargs=(project_dir,)) as pool:
                futures = {
                    pool.submit(evaluate_landscape_point,
                                p_val, scan_param, M_true, T,
                                V1_init, V2_init, u_const,
                                t_obs_arr, u1_obs, u2_obs,
                                _sigmas_for_worker, compute_gradient): i
                    for i, p_val in enumerate(p_scan)
                }
                done = 0
                for fut in as_completed(futures):
                    i = futures[fut]
                    point_results[i] = fut.result()
                    done += 1
                    if done % 8 == 0 or done == len(p_scan):
                        print(f"  [{done}/{len(p_scan)}]  completed")

        elapsed = time.time() - t_start
        print(f"  Done in {elapsed:.1f} s")

        # Per-task timing diagnostic
        times   = np.array([r['t_total'] for r in point_results])
        fwd_t   = np.array([r['t_fwd']   for r in point_results])
        n_steps = np.array([r['n_steps'] for r in point_results])
        cpu_sum = times.sum()
        speedup = cpu_sum / elapsed if elapsed > 0 else float('nan')
        print(f"  Total CPU-time across tasks: {cpu_sum:.1f} s  "
              f"(parallel speedup {speedup:.2f}x on {n_workers} workers; ideal={n_workers}x)")
        print(f"  Per-task total: min={times.min():.2f}s  mean={times.mean():.2f}s  max={times.max():.2f}s")
        print(f"  Per-task fwd:   min={fwd_t.min():.2f}s  mean={fwd_t.mean():.2f}s  max={fwd_t.max():.2f}s")
        print(f"  Per-task steps: min={n_steps.min()}  mean={n_steps.mean():.0f}  max={n_steps.max()}")
        slowest = np.argsort(times)[::-1][:5]
        print('  Slowest 5 tasks (load-imbalance suspects):')
        for k in slowest:
            print(f"    {scan_param}={p_scan[k]:.5g}  t_total={times[k]:.2f}s  "
                  f"t_fwd={fwd_t[k]:.2f}s  n_steps={n_steps[k]}")

        for i, res in enumerate(point_results):
            if res['error'] is not None:
                print(f"  {scan_param}={p_scan[i]:.5g} failed: {res['error']}")

        J_results, grad_results = {}, {}
        for j, (label, _, _, _) in enumerate(smoothing_cases):
            J_results[label]    = np.array([res['J'][j]    for res in point_results])
            grad_results[label] = np.array([res['grad'][j] for res in point_results])

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for label, sigma, color, kind in smoothing_cases:
            J_arr = J_results[label]
            J_pos = np.where(J_arr > 0, J_arr, np.nan)
            ls = '--' if kind == 'inversion' else '-'
            axes[0].plot(p_scan, J_arr,  marker='o', ms=3, lw=1.5, ls=ls, color=color, label=label)
            axes[1].semilogy(p_scan, J_pos, marker='o', ms=3, lw=1.5, ls=ls, color=color, label=label)

        if compute_gradient:
            p_range  = p_scan[-1] - p_scan[0]
            all_J    = np.concatenate([J_results[lbl] for lbl, _, _, _ in smoothing_cases])
            J_range  = np.nanmax(all_J) - np.nanmin(all_J)
            arrow_frac = 0.04
            g_global_max = max(
                np.nanmax(np.abs(grad_results[lbl])) for lbl, _, _, _ in smoothing_cases
            )

            for label, sigma, color, kind in smoothing_cases:
                J_arr = J_results[label]
                g_arr = grad_results[label]
                valid = ~np.isnan(g_arr) & ~np.isnan(J_arr)
                p_v, J_v, g_v = p_scan[valid], J_arr[valid], g_arr[valid]
                hat_x  = 1.0 / p_range
                hat_y  = g_v / J_range
                mag_hat = np.sqrt(hat_x**2 + hat_y**2)
                rel_mag = np.abs(g_v) / g_global_max
                U      = arrow_frac * (hat_x / mag_hat) * rel_mag * p_range
                V_arr  = arrow_frac * (hat_y / mag_hat) * rel_mag * J_range
                axes[0].quiver(p_v, J_v, U, V_arr,
                               color=color, scale=1, scale_units='xy', angles='xy',
                               width=0.003, headwidth=5, headlength=5, zorder=5, alpha=0.9)

            proxy = plt.Line2D([0], [0], marker=r'$\rightarrow$', color='gray',
                               linestyle='none', markersize=10,
                               label=f'Adjoint $dJ/d{{{scan_param}}}$ (arrows)')

        for ax in axes:
            ax.axvline(p_true, color='red', ls='--', lw=1.2,
                       label=f'${scan_param}_{{\\rm true}}$ = {p_true:.5g}')
            handles, lbls = ax.get_legend_handles_labels()
            if ax is axes[0] and compute_gradient:
                ax.legend(handles + [proxy], lbls + [proxy.get_label()],
                          fontsize=8, ncol=2)
            else:
                ax.legend(fontsize=8)
            ax.grid(True, ls=':', lw=0.5, which='both')

        axes[-1].set_xlabel(cfg['xlabel'])
        axes[0].set_ylabel(f'Objective {cfg["title_J"]}')
        grad_tag = (f'Arrows: adjoint gradient $dJ/d{{{scan_param}}}$'
                    if compute_gradient else 'Gradient: OFF')
        axes[0].set_title(
            f'Objective function landscape {cfg["title_J"]} — effect of smoothing\n'
            f'{grad_tag}'
        )
        axes[1].set_ylabel(f'{cfg["title_J"]}  (log scale)')

        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f'J_landscape_{scan_param}_smoothing.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show()

        print()
        for label, sigma, _, _ in smoothing_cases:
            J_arr = J_results[label]
            imin  = np.nanargmin(J_arr)
            print(f"{label:60s}  min J = {J_arr[imin]:.3e}  at {scan_param} = {p_scan[imin]:.5g}")

        results[scan_param] = {
            'p_scan': p_scan,
            'J':      J_results,
            'grad':   grad_results,
            'smoothing_cases': smoothing_cases,
        }

    return results


# ======================================================================
# Objective-function landscape scan (JAX / discrete adjoint via Diffrax)
# ======================================================================
def run_J_landscape_jax(M_true, T,
                        forward_solve_jax,
                        u1_obs, u2_obs,
                        t_ref=None,
                        landscape_params=('a2',),
                        compute_gradient=False,
                        sigma_inversion=None,
                        n_save=1000,
                        save_dir='Figures'):
    """JAX/Diffrax counterpart of `run_J_landscape` for the discrete-adjoint
    notebook. Scans J(p) under several Gaussian-smoothing levels for each
    parameter in `landscape_params`, using the JIT'd JAX forward solver
    provided by the notebook (which closes over the frozen IC and tau0).
    Optionally overlays gradient arrows obtained by `jax.value_and_grad`
    backpropagating through Diffrax's `RecursiveCheckpointAdjoint`.

    Unlike the numpy version, every smoothing case here saves on the same
    fixed `t_ref` grid and runs against the same frozen IC (baked into
    `forward_solve_jax` by closure) — there is no per-iterate IC step.

    Parameters
    ----------
    M_true : dict
        Reference model. Provides a1/a2/k0/k12 baselines and scan ranges.
    T : float
        Simulation horizon (s); sets sigma scales.
    forward_solve_jax : callable
        `forward_solve_jax(p_vec, t_save) -> ys` of shape `(len(t_save), 4)`
        with state ordering `(u1, psi1, u2, psi2)`.
    u1_obs, u2_obs : array-like
        Synthetic observations sampled on `t_ref` (if `t_ref` is None they
        must be sampled on `linspace(0, T, n_save)`).
    t_ref : array-like, optional
        Reference grid for `J`. Defaults to `linspace(0, T, n_save)`.
    landscape_params : tuple of str
        Subset of `{'a1','a2','k0','k12'}` to scan.
    compute_gradient : bool
        If True, also compute `dJ/dp` via `jax.value_and_grad` and overlay
        gradient arrows on the linear-scale plot.
    sigma_inversion : float, optional
        Smoothing length used by the "Inversion" curve. Defaults to
        `0.1 * T` to match the numpy landscape's labelling. Pass the
        notebook's own `sigma_smooth` here to make the "Inversion" curve
        match the optimizer's J exactly.
    n_save : int
        Number of save points if `t_ref` is None.
    save_dir : str or None
        Directory for output PNGs. None disables saving.

    Returns
    -------
    results : dict keyed by scan param, with `p_scan`, `J`, `grad`,
    `smoothing_cases` arrays (same shape convention as `run_J_landscape`).
    """
    import jax
    import jax.numpy as jnp

    if t_ref is None:
        t_ref_np = np.linspace(0.0, T, n_save)
    else:
        t_ref_np = np.asarray(t_ref)
    t_ref_j = jnp.asarray(t_ref_np)

    u1_obs_np = np.asarray(u1_obs)
    u2_obs_np = np.asarray(u2_obs)

    sigma_medium = 0.01 * T
    sigma_heavy  = 0.2  * T
    if sigma_inversion is None:
        sigma_inversion = 0.1 * T

    smoothing_cases = [
        ('No smoothing (identity)',                            None,            'C0', 'native'),
        (f'Medium  ($\\sigma$ = {sigma_medium:.1e} s)',        sigma_medium,    'C1', 'native'),
        (f'Heavy   ($\\sigma$ = {sigma_heavy:.1e} s)',         sigma_heavy,     'C2', 'native'),
        (f'Inversion ($\\sigma$ = {sigma_inversion:.1e} s, fixed t_ref)',
                                                               sigma_inversion, 'C3', 'inversion'),
    ]

    # Per-smoothing-case jitted value-and-grad. Each closes over its own
    # smoothing matrix and pre-smoothed observations so JAX can fold them
    # into the compiled graph.
    def _make_J_and_grad(sigma):
        if sigma is None:
            Su1_obs = jnp.asarray(u1_obs_np)
            Su2_obs = jnp.asarray(u2_obs_np)
            def J_fn(p_vec):
                ys = forward_solve_jax(p_vec, t_ref_j)
                r1 = ys[:, 0] - Su1_obs
                r2 = ys[:, 2] - Su2_obs
                return 0.5 * jnp.trapezoid(r1**2 + r2**2, t_ref_j)
        else:
            S_np = make_smoothing_matrix(t_ref_np, sigma)
            S_j  = jnp.asarray(S_np)
            Su1_obs = jnp.asarray(S_np @ u1_obs_np)
            Su2_obs = jnp.asarray(S_np @ u2_obs_np)
            def J_fn(p_vec):
                ys = forward_solve_jax(p_vec, t_ref_j)
                r1 = S_j @ ys[:, 0] - Su1_obs
                r2 = S_j @ ys[:, 2] - Su2_obs
                return 0.5 * jnp.trapezoid(r1**2 + r2**2, t_ref_j)
        return jax.jit(jax.value_and_grad(J_fn))

    case_J_and_grad = [_make_J_and_grad(sigma) for _, sigma, _, _ in smoothing_cases]

    PARAM_IDX = {'a1': 0, 'a2': 1, 'k0': 2, 'k12': 3}
    p_base = jnp.array([M_true['a1'], M_true['a2'], M_true['k0'], M_true['k12']],
                       dtype=jnp.float64)

    _scan_config = {
        'a1': {
            'vals':     np.linspace(M_true['a1'] * 0.7, M_true['a1'] * 1.3, 60),
            'true_val': M_true['a1'],
            'xlabel':   '$a_1$',
            'title_J':  '$J(a_1)$',
        },
        'a2': {
            'vals':     np.linspace(M_true['a2'] * 0.7, M_true['a2'] * 1.3, 60),
            'true_val': M_true['a2'],
            'xlabel':   '$a_2$',
            'title_J':  '$J(a_2)$',
        },
        'k0': {
            'vals':     np.linspace(0.7 * M_true['k0'], 1.3 * M_true['k0'], 60),
            'true_val': M_true['k0'],
            'xlabel':   '$k_0$  (MPa/m)',
            'title_J':  '$J(k_0)$',
        },
        'k12': {
            'vals':     np.linspace(0.7 * M_true['k12'], 1.3 * M_true['k12'], 60),
            'true_val': M_true['k12'],
            'xlabel':   '$k_{12}$  (MPa/m)',
            'title_J':  '$J(k_{12})$',
        },
    }

    results = {}

    for scan_param in landscape_params:
        cfg     = _scan_config[scan_param]
        p_scan  = cfg['vals']
        p_true  = cfg['true_val']
        idx_p   = PARAM_IDX[scan_param]

        print(f"\n{'='*60}")
        print(f"Landscape: {scan_param}  (true = {p_true:.5g})")
        print(f"Evaluating {len(p_scan)} parameter values "
              f"x {len(smoothing_cases)} smoothing cases"
              f"{'  (with AD gradients)' if compute_gradient else ''} ...")

        J_results    = {lbl: np.full(len(p_scan), np.nan) for lbl, _, _, _ in smoothing_cases}
        grad_results = {lbl: np.full(len(p_scan), np.nan) for lbl, _, _, _ in smoothing_cases}

        t_start = time.time()
        for i, p_val in enumerate(p_scan):
            p_vec = p_base.at[idx_p].set(float(p_val))
            for j, (label, _, _, _) in enumerate(smoothing_cases):
                Jv, gv = case_J_and_grad[j](p_vec)
                Jv.block_until_ready()
                J_results[label][i] = float(Jv)
                if compute_gradient:
                    grad_results[label][i] = float(np.asarray(gv)[idx_p])
            if (i + 1) % 8 == 0 or i == len(p_scan) - 1:
                print(f"  [{i+1}/{len(p_scan)}]  {scan_param}={p_val:.5g}")
        elapsed = time.time() - t_start
        print(f"  Done in {elapsed:.1f} s "
              f"({elapsed/(len(p_scan)*len(smoothing_cases)):.2f} s/eval)")

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for label, sigma, color, kind in smoothing_cases:
            J_arr = J_results[label]
            J_pos = np.where(J_arr > 0, J_arr, np.nan)
            ls = '--' if kind == 'inversion' else '-'
            axes[0].plot(p_scan, J_arr,  marker='o', ms=3, lw=1.5, ls=ls, color=color, label=label)
            axes[1].semilogy(p_scan, J_pos, marker='o', ms=3, lw=1.5, ls=ls, color=color, label=label)

        if compute_gradient:
            p_range  = p_scan[-1] - p_scan[0]
            all_J    = np.concatenate([J_results[lbl] for lbl, _, _, _ in smoothing_cases])
            J_range  = np.nanmax(all_J) - np.nanmin(all_J)
            arrow_frac = 0.04
            g_global_max = max(
                np.nanmax(np.abs(grad_results[lbl])) for lbl, _, _, _ in smoothing_cases
            )

            for label, sigma, color, kind in smoothing_cases:
                J_arr = J_results[label]
                g_arr = grad_results[label]
                valid = ~np.isnan(g_arr) & ~np.isnan(J_arr)
                p_v, J_v, g_v = p_scan[valid], J_arr[valid], g_arr[valid]
                hat_x  = 1.0 / p_range
                hat_y  = g_v / J_range
                mag_hat = np.sqrt(hat_x**2 + hat_y**2)
                rel_mag = np.abs(g_v) / (g_global_max + 1e-30)
                U      = arrow_frac * (hat_x / mag_hat) * rel_mag * p_range
                V_arr  = arrow_frac * (hat_y / mag_hat) * rel_mag * J_range
                axes[0].quiver(p_v, J_v, U, V_arr,
                               color=color, scale=1, scale_units='xy', angles='xy',
                               width=0.003, headwidth=5, headlength=5, zorder=5, alpha=0.9)

            proxy = plt.Line2D([0], [0], marker=r'$\rightarrow$', color='gray',
                               linestyle='none', markersize=10,
                               label=f'AD $dJ/d{{{scan_param}}}$ (arrows)')

        for ax in axes:
            ax.axvline(p_true, color='red', ls='--', lw=1.2,
                       label=f'${scan_param}_{{\\rm true}}$ = {p_true:.5g}')
            handles, lbls = ax.get_legend_handles_labels()
            if ax is axes[0] and compute_gradient:
                ax.legend(handles + [proxy], lbls + [proxy.get_label()],
                          fontsize=8, ncol=2)
            else:
                ax.legend(fontsize=8)
            ax.grid(True, ls=':', lw=0.5, which='both')

        axes[-1].set_xlabel(cfg['xlabel'])
        axes[0].set_ylabel(f'Objective {cfg["title_J"]}')
        grad_tag = (f'Arrows: discrete-adjoint AD gradient $dJ/d{{{scan_param}}}$'
                    if compute_gradient else 'Gradient: OFF')
        axes[0].set_title(
            f'Objective function landscape {cfg["title_J"]} — effect of smoothing  '
            f'(JAX / Diffrax discrete adjoint)\n{grad_tag}'
        )
        axes[1].set_ylabel(f'{cfg["title_J"]}  (log scale)')

        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f'J_landscape_{scan_param}_smoothing_jax.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show()

        print()
        for label, sigma, _, _ in smoothing_cases:
            J_arr = J_results[label]
            imin  = np.nanargmin(J_arr)
            print(f"{label:60s}  min J = {J_arr[imin]:.3e}  at {scan_param} = {p_scan[imin]:.5g}")

        results[scan_param] = {
            'p_scan': p_scan,
            'J':      J_results,
            'grad':   grad_results,
            'smoothing_cases': smoothing_cases,
        }

    return results


# ======================================================================
# 2D objective-function landscape (JAX vmap-parallelised)
# ======================================================================
def run_J_landscape_2d_jax(M, T,
                           forward_solve_jax,
                           u1_obs, u2_obs,
                           t_ref=None,
                           params=('a1', 'a2'),
                           n1=30, n2=30,
                           sigma=None,
                           p1_range=None, p2_range=None,
                           chunk_size=None,
                           n_save=1000,
                           view_init=(30, -60),
                           save_dir='Figures'):
    """3D surface scan of log10(J) over two parameters, batched-parallel via
    `jax.vmap` through the Diffrax discrete-adjoint stack.

    Each chunk of the parameter grid is dispatched as one vmap'd XLA
    computation. XLA parallelises the batch internally across CPU threads (or
    SIMD lanes / GPU streams when available). `chunk_size` is the parallelism
    knob and the peak-memory knob — larger means more concurrent solves but
    higher RAM, since every batch member carries its own adaptive-solver state.

    Parameters
    ----------
    M : dict
        Reference model. Provides parameter defaults for non-scanned slots and
        the true-value markers on the plot.
    T : float
        Simulation horizon (s). Defaults `sigma` to 0.01*T if not given.
    forward_solve_jax : callable
        `forward_solve_jax(p_vec, t_save) -> ys` from the notebook (closes over
        the frozen IC and tau0_1/tau0_2).
    u1_obs, u2_obs : array-like
        Observations sampled on `t_ref`.
    t_ref : array-like, optional
        Reference grid for J. Defaults to `linspace(0, T, n_save)`.
    params : (str, str)
        The two parameters to scan, from `{'a1','a2','k0','k12'}`. Plot x is
        params[0], plot y is params[1].
    n1, n2 : int
        Sampling density along each axis. Total evaluations = `n1 * n2`.
    sigma : float, optional
        Gaussian smoothing scale on `t_ref`. Defaults to `0.01 * T` (matches
        the notebook's `sigma_smooth`).
    p1_range, p2_range : (lo, hi) tuples, optional
        Override the default ranges (±15% on a, ±30% on k).
    chunk_size : int, optional
        vmap batch size. Defaults to `min(64, n1*n2)`.
    n_save : int
        Number of t_ref points if not supplied.
    view_init : (elev, azim)
        Matplotlib 3D viewing angle for the surface panel.
    save_dir : str or None
        Output directory for the PNG. None disables saving.

    Returns
    -------
    dict with `p1_name`, `p2_name`, `p1_vals`, `p2_vals`, `J_grid`,
    `logJ_grid`, `sigma`.
    """
    import jax
    import jax.numpy as jnp
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    PARAM_IDX = {'a1': 0, 'a2': 1, 'k0': 2, 'k12': 3}
    p1_name, p2_name = params
    if p1_name not in PARAM_IDX or p2_name not in PARAM_IDX:
        raise ValueError(f"params must be from {list(PARAM_IDX)}, got {params}")
    if p1_name == p2_name:
        raise ValueError("params must be two distinct names")
    idx_p1 = PARAM_IDX[p1_name]
    idx_p2 = PARAM_IDX[p2_name]

    def _default_range(name):
        if name in ('a1', 'a2'):
            return (M[name] * 0.85, M[name] * 1.15)
        return (0.85 * M[name], 1.15 * M[name])

    p1_lo, p1_hi = p1_range if p1_range is not None else _default_range(p1_name)
    p2_lo, p2_hi = p2_range if p2_range is not None else _default_range(p2_name)

    p1_vals = np.linspace(p1_lo, p1_hi, n1)
    p2_vals = np.linspace(p2_lo, p2_hi, n2)

    if t_ref is None:
        t_ref_np = np.linspace(0.0, T, n_save)
    else:
        t_ref_np = np.asarray(t_ref)
    t_ref_j = jnp.asarray(t_ref_np)

    if sigma is None:
        sigma = 0.01 * T

    u1_obs_np = np.asarray(u1_obs)
    u2_obs_np = np.asarray(u2_obs)

    if sigma == 0:
        Su1 = jnp.asarray(u1_obs_np)
        Su2 = jnp.asarray(u2_obs_np)
        def J_fn(p_vec):
            ys = forward_solve_jax(p_vec, t_ref_j)
            r1 = ys[:, 0] - Su1
            r2 = ys[:, 2] - Su2
            return 0.5 * jnp.trapezoid(r1**2 + r2**2, t_ref_j)
    else:
        S_np = make_smoothing_matrix(t_ref_np, sigma)
        S_j  = jnp.asarray(S_np)
        Su1 = jnp.asarray(S_np @ u1_obs_np)
        Su2 = jnp.asarray(S_np @ u2_obs_np)
        def J_fn(p_vec):
            ys = forward_solve_jax(p_vec, t_ref_j)
            r1 = S_j @ ys[:, 0] - Su1
            r2 = S_j @ ys[:, 2] - Su2
            return 0.5 * jnp.trapezoid(r1**2 + r2**2, t_ref_j)

    # vmap-parallelised batched evaluator. No grad is taken — surface scans
    # do not need backprop, and skipping it halves the per-point cost.
    J_batched = jax.jit(jax.vmap(J_fn))

    p_base = np.array([M['a1'], M['a2'], M['k0'], M['k12']], dtype=np.float64)

    # (n1*n2, 4) parameter grid, row-major over (i, j) = (p1_idx, p2_idx).
    P1, P2 = np.meshgrid(p1_vals, p2_vals, indexing='ij')
    p_grid = np.tile(p_base, (n1 * n2, 1))
    p_grid[:, idx_p1] = P1.ravel()
    p_grid[:, idx_p2] = P2.ravel()

    if chunk_size is None:
        chunk_size = min(64, n1 * n2)

    print(f"\n{'='*60}")
    print(f"2D landscape: ({p1_name}, {p2_name})  grid = {n1} x {n2}  "
          f"({n1*n2} evaluations)")
    print(f"  {p1_name}: [{p1_lo:.5g}, {p1_hi:.5g}]")
    print(f"  {p2_name}: [{p2_lo:.5g}, {p2_hi:.5g}]")
    print(f"  sigma = {sigma:.3e} s   vmap chunk = {chunk_size}")

    n_chunks = (n1 * n2 + chunk_size - 1) // chunk_size
    J_flat = np.full(n1 * n2, np.nan)

    t_start = time.time()
    for ci in range(n_chunks):
        i0 = ci * chunk_size
        i1 = min(i0 + chunk_size, n1 * n2)
        # Pad the final chunk to keep the JIT cache hot (same trace shape).
        chunk_arr = p_grid[i0:i1]
        pad = chunk_size - chunk_arr.shape[0]
        if pad > 0:
            chunk_arr = np.vstack([chunk_arr, np.tile(p_base, (pad, 1))])
        Js = J_batched(jnp.asarray(chunk_arr))
        Js.block_until_ready()
        J_flat[i0:i1] = np.asarray(Js)[:i1 - i0]
        if (ci + 1) % 4 == 0 or ci == n_chunks - 1:
            print(f"  chunk [{ci+1}/{n_chunks}]  ({i1}/{n1*n2} pts)  "
                  f"elapsed {time.time()-t_start:.1f}s")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f} s  ({elapsed/(n1*n2)*1000:.1f} ms/point)")

    J_grid    = J_flat.reshape(n1, n2)
    logJ_grid = np.log10(np.where(J_grid > 0, J_grid, np.nan))

    # ----------------------------------------------------------------------
    # 3D surface + companion contour view
    # ----------------------------------------------------------------------
    _xlabel = {'a1': r'$a_1$', 'a2': r'$a_2$',
               'k0': r'$k_0$ (MPa/m)', 'k12': r'$k_{12}$ (MPa/m)'}
    p1_true, p2_true = M[p1_name], M[p2_name]

    fig = plt.figure(figsize=(13, 6))
    ax_s = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax_s.plot_surface(P1, P2, logJ_grid,
                              cmap='viridis', edgecolor='none', alpha=0.92,
                              rstride=1, cstride=1)
    i1_t = int(np.argmin(np.abs(p1_vals - p1_true)))
    i2_t = int(np.argmin(np.abs(p2_vals - p2_true)))
    if np.isfinite(logJ_grid[i1_t, i2_t]):
        ax_s.scatter([p1_true], [p2_true], [float(logJ_grid[i1_t, i2_t])],
                     color='red', s=70, marker='*',
                     label=f'true ({p1_name}, {p2_name})', zorder=10)
    ax_s.set_xlabel(_xlabel[p1_name])
    ax_s.set_ylabel(_xlabel[p2_name])
    ax_s.set_zlabel(r'$\log_{10} J$')
    ax_s.set_title(f'$J({p1_name}, {p2_name})$  surface  (log scale)')
    ax_s.view_init(elev=view_init[0], azim=view_init[1])
    ax_s.legend(fontsize=8, loc='upper left')
    fig.colorbar(surf, ax=ax_s, shrink=0.6, pad=0.1, label=r'$\log_{10} J$')

    ax_c = fig.add_subplot(1, 2, 2)
    cf = ax_c.contourf(P1, P2, logJ_grid, levels=30, cmap='viridis')
    ax_c.contour(P1, P2, logJ_grid, levels=15,
                 colors='k', linewidths=0.4, alpha=0.5)
    ax_c.scatter([p1_true], [p2_true],
                 color='red', s=110, marker='*', edgecolor='white', linewidth=1.0,
                 label=f'true ({p1_true:.4g}, {p2_true:.4g})', zorder=10)
    if np.isfinite(J_grid).any():
        imin = np.unravel_index(np.nanargmin(J_grid), J_grid.shape)
        ax_c.scatter([p1_vals[imin[0]]], [p2_vals[imin[1]]],
                     color='cyan', s=90, marker='o',
                     edgecolor='black', linewidth=0.8,
                     label=f'grid min @ ({p1_vals[imin[0]]:.4g}, '
                           f'{p2_vals[imin[1]]:.4g})',
                     zorder=11)
    ax_c.set_xlabel(_xlabel[p1_name])
    ax_c.set_ylabel(_xlabel[p2_name])
    ax_c.set_title(f'$\\log_{{10}} J({p1_name}, {p2_name})$  contour')
    ax_c.legend(fontsize=8, loc='best')
    fig.colorbar(cf, ax=ax_c, shrink=0.85, label=r'$\log_{10} J$')

    fig.suptitle(
        f'2D objective landscape — discrete adjoint via JAX/Diffrax  '
        f'($\\sigma$ = {sigma:.2e} s, grid {n1}x{n2})'
    )
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir,
                              f'J_landscape_2d_{p1_name}_{p2_name}_jax.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()

    print()
    print(f"True location:  ({p1_name}={p1_true:.5g}, {p2_name}={p2_true:.5g})")
    if np.isfinite(J_grid).any():
        imin = np.unravel_index(np.nanargmin(J_grid), J_grid.shape)
        print(f"Grid minimum:   ({p1_name}={p1_vals[imin[0]]:.5g}, "
              f"{p2_name}={p2_vals[imin[1]]:.5g})")
        print(f"  J at true grid pt = {J_grid[i1_t, i2_t]:.3e}")
        print(f"  J at grid min     = {J_grid[imin]:.3e}")

    return {
        'p1_name':   p1_name,
        'p2_name':   p2_name,
        'p1_vals':   p1_vals,
        'p2_vals':   p2_vals,
        'J_grid':    J_grid,
        'logJ_grid': logJ_grid,
        'sigma':     sigma,
    }
