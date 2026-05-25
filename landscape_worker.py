"""Per-point worker for the two-block objective-landscape scan.

Lives in a real module (rather than a notebook cell) so it can be imported
by `concurrent.futures.ProcessPoolExecutor` workers via pickle reference.

Gradients are computed via forward sensitivity (Cao, Li, Petzold 2003),
which avoids the dual-inconsistency issues that the adaptively-stepped
continuous adjoint hits when forward trajectories contain fast slip events.
"""
import sys
import time
import numpy as np


def init_worker(project_dir):
    """Pool initializer: put the project root on sys.path so workers can
    import the project modules (friction_derivs, adapt_fwd_solve, ...)."""
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

from friction_derivs import setup_initial_conditions_2block
from adapt_fwd_solve import (
    forward_solve_adaptive_2block,
    forward_solve_adaptive_2block_sens,
)
from compute_obj import compute_grad_forward_sens_2block


def _smooth_apply(t, sigma, u, chunk=512):
    """Equivalent to (make_smoothing_matrix(t, sigma) @ u) but streamed in
    row chunks. Peak extra memory is O(chunk * N) instead of O(N**2),
    which matters when N is ~30k and the dense matrix would be ~7 GB."""
    N = len(t)
    dt = np.diff(t)
    w = np.empty(N)
    w[0]    = dt[0] / 2.0
    w[1:-1] = (dt[:-1] + dt[1:]) / 2.0
    w[-1]   = dt[-1] / 2.0
    inv2s2 = 1.0 / (2.0 * sigma ** 2)
    out = np.empty(N)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        diff2 = (t[i0:i1, None] - t[None, :]) ** 2 * inv2s2
        K = np.exp(-diff2) * w[None, :]
        K /= K.sum(axis=1, keepdims=True)
        out[i0:i1] = K @ u
    return out


def _eval_J(fwd, u1_on, u2_on, sigma):
    if sigma is None or sigma == 0:
        return 0.5 * np.trapz(
            (fwd['u1'] - u1_on)**2 + (fwd['u2'] - u2_on)**2, fwd['t'])
    r1 = _smooth_apply(fwd['t'], sigma, fwd['u1'] - u1_on)
    r2 = _smooth_apply(fwd['t'], sigma, fwd['u2'] - u2_on)
    return 0.5 * np.trapz(r1**2 + r2**2, fwd['t'])


def _eval_J_on_tref(fwd, u1_obs_ref, u2_obs_ref, t_ref, S_fixed):
    """Inversion-style J: interpolate u1, u2 to a fixed uniform t_ref and
    apply a precomputed S_fixed.  Mirrors fun_and_grad in the notebook so the
    landscape and the optimizer see the same objective."""
    u1r = np.interp(t_ref, fwd['t'], fwd['u1'])
    u2r = np.interp(t_ref, fwd['t'], fwd['u2'])
    r1  = S_fixed @ u1r - S_fixed @ u1_obs_ref
    r2  = S_fixed @ u2r - S_fixed @ u2_obs_ref
    return 0.5 * np.trapz(r1**2 + r2**2, t_ref)


def evaluate_landscape_point(p_val, scan_param, M_true, T,
                              V1_init, V2_init, u_const,
                              t_obs, u1_obs, u2_obs,
                              sigmas, compute_gradient):
    """Forward-solve at one parameter value; return J (and grad) per sigma.

    Two forward solves are run when scan_param is a1/a2:
      - fwd_true: IC recomputed for the scanned a-value (per-point IC).  Used
        for the native-grid J curves.
      - fwd_fix:  IC frozen at `u_const` (the initial-guess IC used by the
        inversion's fun_and_grad). Used for the inversion-style entries.

    When `compute_gradient` is True the run is upgraded to the sensitivity
    solver, which adds ~5x cost per parameter.  Gradients are evaluated only
    for the inversion-style sigma entries (the ones with a fixed t_ref) so
    they're directly comparable to the optimizer's dJ/dp.

    `sigmas` is a list of entries:
        (label, sigma)                          — native-grid Gaussian smoothing
        (label, sigma, t_ref, S_fixed)          — inversion-style J on fixed grid
    """
    t_start = time.perf_counter()
    Mc = dict(M_true)
    Mc[scan_param] = p_val
    if scan_param in ('a1', 'a2'):
        u1_0_true, psi1_0_true, _, u2_0_true, psi2_0_true, _ = setup_initial_conditions_2block(
            Mc, V1_init=V1_init, V2_init=V2_init)
        tau0_1_true, tau0_2_true = Mc['tau0_1'], Mc['tau0_2']
    else:
        u1_0_true, psi1_0_true, u2_0_true, psi2_0_true = u_const

    t_fwd0 = time.perf_counter()
    try:
        fwd = forward_solve_adaptive_2block(
            Mc, T, u1_0_true, psi1_0_true, u2_0_true, psi2_0_true,
            V1_init=V1_init, V2_init=V2_init)
    except Exception as e:
        return {'p_val':   p_val,
                'error':   str(e),
                'J':       [np.nan] * len(sigmas),
                'grad':    [np.nan] * len(sigmas),
                'n_steps': 0,
                't_fwd':   time.perf_counter() - t_fwd0,
                't_adj':   0.0,
                't_total': time.perf_counter() - t_start}
    t_fwd = time.perf_counter() - t_fwd0

    u1_on = np.interp(fwd['t'], t_obs, u1_obs)
    u2_on = np.interp(fwd['t'], t_obs, u2_obs)

    # Lazily build the fixed-IC forward solve (with sensitivities when needed),
    # only if an inversion-style sigma asks for it.
    fwd_fix = None
    fwd_fix_sens = None  # populated when compute_gradient is True

    def _ensure_fwd_fix():
        nonlocal fwd_fix, fwd_fix_sens
        if fwd_fix is not None:
            return
        if scan_param in ('a1', 'a2'):
            Mc['tau0_1'] = M_true['tau0_1']
            Mc['tau0_2'] = M_true['tau0_2']
            u1_0_fix, psi1_0_fix, u2_0_fix, psi2_0_fix = u_const
        else:
            u1_0_fix, psi1_0_fix, u2_0_fix, psi2_0_fix = u_const

        if compute_gradient:
            fwd_fix_sens = forward_solve_adaptive_2block_sens(
                Mc, T, u1_0_fix, psi1_0_fix, u2_0_fix, psi2_0_fix,
                params=(scan_param,),
                V1_init=None, V2_init=None)
            fwd_fix = fwd_fix_sens  # has u1/u2/psi1/psi2/t as well
        else:
            fwd_fix = forward_solve_adaptive_2block(
                Mc, T, u1_0_fix, psi1_0_fix, u2_0_fix, psi2_0_fix,
                V1_init=None, V2_init=None)

        if scan_param in ('a1', 'a2'):
            Mc['tau0_1'], Mc['tau0_2'] = tau0_1_true, tau0_2_true

    Js, grads = [], []
    t_grad0 = time.perf_counter()
    for entry in sigmas:
        if len(entry) == 2:
            _, sigma = entry
            t_ref_e, S_fixed_e = None, None
        elif len(entry) == 4:
            _, sigma, t_ref_e, S_fixed_e = entry
        else:
            raise ValueError(f"sigmas entry has unexpected length {len(entry)}")

        if t_ref_e is not None:
            _ensure_fwd_fix()
            u1_obs_ref_e = np.interp(t_ref_e, t_obs, u1_obs)
            u2_obs_ref_e = np.interp(t_ref_e, t_obs, u2_obs)
            Js.append(_eval_J_on_tref(fwd_fix, u1_obs_ref_e, u2_obs_ref_e,
                                      t_ref_e, S_fixed_e))
            if compute_gradient and fwd_fix_sens is not None:
                grad_dict = compute_grad_forward_sens_2block(
                    fwd_fix_sens, t_obs, u1_obs, u2_obs,
                    sigma=sigma, t_ref=t_ref_e, S=S_fixed_e)
                grads.append(grad_dict[scan_param])
            else:
                grads.append(np.nan)
        else:
            Js.append(_eval_J(fwd, u1_on, u2_on, sigma))
            grads.append(np.nan)
    t_grad = time.perf_counter() - t_grad0

    return {'p_val':   p_val,
            'error':   None,
            'J':       Js,
            'grad':    grads,
            'n_steps': len(fwd['t']),
            't_fwd':   t_fwd,
            't_adj':   t_grad,   # kept name for compat with existing diagnostic
            't_total': time.perf_counter() - t_start}
