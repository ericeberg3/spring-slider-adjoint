"""Per-point worker for the two-block objective-landscape scan.

Lives in a real module (rather than a notebook cell) so it can be imported
by `concurrent.futures.ProcessPoolExecutor` workers via pickle reference.
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
from adapt_fwd_solve import forward_solve_adaptive_2block
from adjoint_solve import adjoint_solve_2block
from compute_obj import (
    compute_grad_a1,
    compute_grad_a2,
    compute_grad_k0,
    compute_grad_k12,
)

_GRAD_FNS = {
    'a1':  compute_grad_a1,
    'a2':  compute_grad_a2,
    'k0':  compute_grad_k0,
    'k12': compute_grad_k12,
}


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
    # S @ a - S @ b == S @ (a - b); one smoothing per residual instead of two.
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
        for the native-grid J curves -- these show the "true" J(a) landscape
        whose minimum is at a_true.
      - fwd_fix:  IC frozen at `u_const` (the initial-guess IC used by the
        inversion's fun_and_grad). Used for the inversion-style entries so the
        landscape curve matches exactly the J the optimizer sees.

    `sigmas` is a list of entries describing each J evaluation:
        (label, sigma)                          — native-grid Gaussian smoothing
                                                  (uses per-point IC for a1/a2)
        (label, sigma, t_ref, S_fixed)          — inversion-style J: interpolate
                                                  to t_ref then apply precomputed
                                                  S_fixed (uses fixed IC u_const).
    `u_const` is the tuple (u1_0, psi1_0, u2_0, psi2_0); for inversion-style
    landscapes pass the inversion's u1_0_inv/psi1_0_inv/u2_0_inv/psi2_0_inv.
    """
    t_start = time.perf_counter()
    Mc = dict(M_true)
    Mc[scan_param] = p_val
    if scan_param in ('a1', 'a2'):
        u1_0_true, psi1_0_true, _, u2_0_true, psi2_0_true, _ = setup_initial_conditions_2block(
            Mc, V1_init=V1_init, V2_init=V2_init)
        # Mc['tau0_1'/'tau0_2'] were just mutated for the per-point IC solve.
        # Save them for the per-point fwd; restore M_true's tau0 for the fixed-IC fwd.
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

    # Lazily build the fixed-IC forward solve, only if an inversion-style entry
    # asks for it. For a1/a2 scans we restore Mc's tau0 to M_true's values so
    # this fwd mirrors fun_and_grad (Mc = dict(M_true), then Mc[p] = p_val).
    fwd_fix = None
    u1_on_fix = u2_on_fix = None
    def _ensure_fwd_fix():
        nonlocal fwd_fix, u1_on_fix, u2_on_fix
        if fwd_fix is not None:
            return
        if scan_param in ('a1', 'a2'):
            Mc['tau0_1'] = M_true['tau0_1']
            Mc['tau0_2'] = M_true['tau0_2']
            u1_0_fix, psi1_0_fix, u2_0_fix, psi2_0_fix = u_const
            fwd_fix_local = forward_solve_adaptive_2block(
                Mc, T, u1_0_fix, psi1_0_fix, u2_0_fix, psi2_0_fix,
                V1_init=None, V2_init=None)
            # restore the per-point tau0 so the adjoint/native-grid path is unaffected
            Mc['tau0_1'], Mc['tau0_2'] = tau0_1_true, tau0_2_true
        else:
            fwd_fix_local = fwd
        fwd_fix = fwd_fix_local
        u1_on_fix = np.interp(fwd_fix['t'], t_obs, u1_obs)
        u2_on_fix = np.interp(fwd_fix['t'], t_obs, u2_obs)

    grad_fn = _GRAD_FNS[scan_param] if compute_gradient else None
    Js, grads = [], []
    t_adj0 = time.perf_counter()
    for entry in sigmas:
        # Back-compat: (label, sigma) -> native-grid J on the per-point fwd.
        # Inversion-style: (label, sigma, t_ref, S_fixed) -> fixed-grid J on the
        # fixed-IC fwd (matches the inversion's fun_and_grad).
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
        else:
            Js.append(_eval_J(fwd, u1_on, u2_on, sigma))

        if compute_gradient:
            try:
                adj = adjoint_solve_2block(fwd, fwd['t'], u1_on, u2_on, Mc, sigma)
                grads.append(grad_fn(fwd, adj, Mc))
            except Exception:
                grads.append(np.nan)
        else:
            grads.append(np.nan)
    t_adj = time.perf_counter() - t_adj0

    return {'p_val':   p_val,
            'error':   None,
            'J':       Js,
            'grad':    grads,
            'n_steps': len(fwd['t']),
            't_fwd':   t_fwd,
            't_adj':   t_adj,
            't_total': time.perf_counter() - t_start}
