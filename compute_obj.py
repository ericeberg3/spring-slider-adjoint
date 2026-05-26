from adjoint_solve import *
from adapt_fwd_solve import *
from friction_derivs import *

def compute_J(fwd, t_obs, u_obs, sigma, S=None):
    """J = 0.5 * int_0^T (Su - Su_obs)^2 dt  (trapezoidal, Gaussian-smoothed misfit).

    If S is provided it is used directly; otherwise one is built from sigma via
    make_smoothing_matrix (integration-weight Gaussian for non-uniform grids).
    """
    if S is None:
        S = make_smoothing_matrix(fwd['t'], sigma)
    u_obs_at_fwd = np.interp(fwd['t'], t_obs, u_obs)
    Su           = S @ fwd['u']
    Su_obs       = S @ u_obs_at_fwd
    return 0.5 * np.trapz((Su - Su_obs) ** 2, fwd['t'])

def compute_grad_a(fwd, adj, M):
    """
    dJ/da = int_0^T [ λ·∂τ/∂a  −  r·∂G/∂a ] dt

    λ = (p + G_V·r)/(τ_V+η) is the Lagrange multiplier of the force-balance constraint,
    already stored in adj['lam'] by adjoint_solve.
    """
    integrand = adj['lam'] * fwd['dtau_da'] - adj['r'] * fwd['dG_da']
    return np.trapz(-integrand, fwd['t'])

def compute_grad_k(fwd, adj, M):
    """
    dJ/dk = int_0^T lambda * (u - V_bg*t) dt

    k enters only via the force balance constraint (k*u - k*V_bg*t);
    tau and G have no explicit k dependence, so dtau/dk = dG/dk = 0.
    """
    integrand = -adj['lam'] * (fwd['u'] - M['V_bg'] * fwd['t'])
    return np.trapz(integrand, fwd['t'])


# ------------------------------------------------------------------
# Two-block objective and forward-sensitivity gradient
# ------------------------------------------------------------------

def compute_J_2block(fwd, t_obs, u1_obs, u2_obs, sigma, t_ref, S=None):
    """J = J1 + J2 where Ji = 0.5*int(S*ui - S*ui_obs)^2 dt on a fixed t_ref."""
    if S is None:
        S = make_smoothing_matrix(t_ref, sigma)

    J = 0.0
    for u_native, u_obs_arr in ((fwd['u1'], u1_obs), (fwd['u2'], u2_obs)):
        if u_obs_arr is None:
            continue
        u_ref     = np.interp(t_ref, fwd['t'], u_native)
        u_obs_ref = np.interp(t_ref, t_obs, u_obs_arr)
        Su        = S @ u_ref
        Su_obs    = S @ u_obs_ref
        J        += 0.5 * np.trapz((Su - Su_obs)**2, t_ref)
    return J


def compute_grad_a1(fwd, adj, M):
    """dJ/da1 = int [ -lam1*(dtau1/da1) + r1*(dG1/da1) ] dt  (two-block adjoint)."""
    integrand = adj['lam1'] * fwd['dtau_da1'] - adj['r1'] * fwd['dG_da1']
    return np.trapz(-integrand, fwd['t'])


def compute_grad_a2(fwd, adj, M):
    """dJ/da2 = int [ -lam2*(dtau2/da2) + r2*(dG2/da2) ] dt  (two-block adjoint)."""
    integrand = adj['lam2'] * fwd['dtau_da2'] - adj['r2'] * fwd['dG_da2']
    return np.trapz(-integrand, fwd['t'])


def compute_grad_k0(fwd, adj, M):
    """dJ/dk0 = int [ -lam1*(u1 - V_bg*t) - lam2*(u2 - V_bg*t) ] dt  (two-block adjoint)."""
    t = fwd['t']
    return np.trapz(
        -adj['lam1'] * (fwd['u1'] - M['V_bg'] * t)
        - adj['lam2'] * (fwd['u2'] - M['V_bg'] * t),
        t
    )


def compute_grad_k12(fwd, adj, M):
    """dJ/dk12 = int [ (lam2 - lam1)*(u1 - u2) ] dt  (two-block adjoint)."""
    del M  # k12 gradient has no explicit M dependence; M kept for API consistency
    return np.trapz((adj['lam2'] - adj['lam1']) * (fwd['u1'] - fwd['u2']), fwd['t'])


def compute_grad_forward_sens_2block(fwd_sens, t_obs, u1_obs, u2_obs, sigma, t_ref, S=None):
    """
    Gradients dJ/dp via forward sensitivity, using the same fixed-reference-grid
    convention as compute_J_2block (frozen S on t_ref; trapz integration on t_ref).

    For a parameter p,
        dJ/dp = sum_{i=1,2} int_{t_ref} (S u_i - S u_{i,obs})*(S * ds_i/dp_ref) dt_ref
    with s_i(t_fwd) interpolated onto t_ref the same way as u_i.

    Returns dict[p -> dJ/dp].
    """
    if S is None:
        S = make_smoothing_matrix(t_ref, sigma)

    t_fwd = fwd_sens['t']

    # Precompute residuals (block-wise) on the reference grid
    residuals = {}
    for idx, (u_native, u_obs_arr) in enumerate(
            ((fwd_sens['u1'], u1_obs), (fwd_sens['u2'], u2_obs)), start=1):
        if u_obs_arr is None:
            residuals[idx] = None
            continue
        u_ref     = np.interp(t_ref, t_fwd, u_native)
        u_obs_ref = np.interp(t_ref, t_obs, u_obs_arr)
        residuals[idx] = S @ u_ref - S @ u_obs_ref

    grads = {}
    for p, sens in fwd_sens['sens'].items():
        g = 0.0
        for idx in (1, 2):
            if residuals[idx] is None:
                continue
            s_u_native = sens[f's_u{idx}']
            s_u_ref    = np.interp(t_ref, t_fwd, s_u_native)
            g += np.trapz(residuals[idx] * (S @ s_u_ref), t_ref)
        grads[p] = g

    return grads


def trapz_weights(t):
    w = np.zeros(len(t)); w[:-1] += np.diff(t); w[1:] += np.diff(t)
    return 0.5 * w

def interp_adjoint_scatter(v, t_src, t_dst):
    """True adjoint of linear interp P: t_src -> t_dst.  P^T v via scatter-add."""
    idx = np.searchsorted(t_src, t_dst, side='right') - 1
    idx = np.clip(idx, 0, len(t_src) - 2)
    alpha = np.clip((t_dst - t_src[idx]) / (t_src[idx + 1] - t_src[idx]), 0.0, 1.0)
    result = np.zeros(len(t_src))
    np.add.at(result, idx,     (1.0 - alpha) * v)
    np.add.at(result, idx + 1, alpha          * v)
    return result