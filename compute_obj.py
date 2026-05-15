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
# Two-block objective and gradients
# ------------------------------------------------------------------

def compute_J_2block(fwd, t_obs, u1_obs, u2_obs, sigma, S=None):
    """J = J1 + J2 where Ji = 0.5*int(S*ui - S*ui_obs)^2 dt.
    Pass u1_obs=None or u2_obs=None to exclude that block."""
    t = fwd['t']
    if S is None:
        S = make_smoothing_matrix(t, sigma)
    J = 0.0
    for u, u_obs_arr in ((fwd['u1'], u1_obs), (fwd['u2'], u2_obs)):
        if u_obs_arr is None:
            continue
        u_obs_at_fwd = np.interp(t, t_obs, u_obs_arr)
        Su = S @ u; Su_obs = S @ u_obs_at_fwd
        J += 0.5 * np.trapz((Su - Su_obs)**2, t)
    return J


def compute_grad_a1(fwd, adj, M):
    """dJ/da1 = int [ -lam1*(dtau1/da1) + r1*(dG1/da1) ] dt"""
    integrand = adj['lam1'] * fwd['dtau_da1'] - adj['r1'] * fwd['dG_da1']
    return np.trapz(-integrand, fwd['t'])


def compute_grad_a2(fwd, adj, M):
    """dJ/da2 = int [ -lam2*(dtau2/da2) + r2*(dG2/da2) ] dt"""
    integrand = adj['lam2'] * fwd['dtau_da2'] - adj['r2'] * fwd['dG_da2']
    return np.trapz(-integrand, fwd['t'])


def compute_grad_k0(fwd, adj, M):
    """dJ/dk0 = int [ -lam1*(u1 - V_bg*t) - lam2*(u2 - V_bg*t) ] dt"""
    t = fwd['t']
    return np.trapz(
        +adj['lam1'] * (fwd['u1'] - M['V_bg'] * t)
        + adj['lam2'] * (fwd['u2'] - M['V_bg'] * t),
        t
    )


def compute_grad_k12(fwd, adj, M):
    """dJ/dk12 = int [ (lam2 - lam1)*(u1 - u2) ] dt"""
    del M  # k12 gradient has no explicit M dependence; M kept for API consistency
    return np.trapz(-(adj['lam2'] - adj['lam1']) * (fwd['u1'] - fwd['u2']), fwd['t'])