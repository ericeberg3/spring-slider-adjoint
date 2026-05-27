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
