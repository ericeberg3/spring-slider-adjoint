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
    return np.trapz(integrand, fwd['t'])

def compute_grad_k(fwd, adj, M):
    """
    dJ/dk = int_0^T lambda * (u - V_bg*t) dt

    k enters only via the force balance constraint (k*u - k*V_bg*t);
    tau and G have no explicit k dependence, so dtau/dk = dG/dk = 0.
    """
    integrand = adj['lam'] * (fwd['u'] - M['V_bg'] * fwd['t'])
    return np.trapz(integrand, fwd['t'])