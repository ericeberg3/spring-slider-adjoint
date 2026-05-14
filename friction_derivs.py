import numpy as np
from scipy.optimize import brentq

# ------------------------------------------------------------------
# Friction law
# ------------------------------------------------------------------

def xi_fn(V, psi, M):
    """Argument of arcsinh: xi = V/(2*V0) * exp(psi/a)."""
    return V / (2.0 * M['V0']) * np.exp(psi / M['a'])

def tau_fn(V, psi, M):
    """Frictional strength (MPa)."""
    return M['N'] * M['a'] * np.arcsinh(xi_fn(V, psi, M))

def fss_fn(V, M):
    """Steady-state friction coefficient."""
    return M['f0'] + (M['a'] - M['b']) * np.log(V / M['V0'])

def G_fn(V, psi, M):
    """State evolution dpsi/dt (Dieterich aging law, in terms of psi = f0 + b*ln(theta*V0/dc))."""
    return M['b'] * M['V0'] / M['dc'] * np.exp(-(psi - M['f0']) / M['b']) - M['b'] * V / M['dc']

# ------------------------------------------------------------------
# Partial derivatives
# ------------------------------------------------------------------

def tau_V_fn(V, psi, M):
    """dtau/dV = N*a / sqrt(1+xi^2) * xi/V."""
    xi = xi_fn(V, psi, M)
    return M['N'] * M['a'] / np.sqrt(1.0 + xi**2) * xi / V

def tau_psi_fn(V, psi, M):
    """dtau/dpsi = N * xi / sqrt(1+xi^2)."""
    xi = xi_fn(V, psi, M)
    return M['N'] * xi / np.sqrt(1.0 + xi**2)

def G_V_fn(V, psi, M):
    """dG/dV for aging law: G = b*V0/dc * exp(-(psi-f0)/b) - b*V/dc, so dG/dV = -b/dc."""
    return -M['b'] / M['dc']

def G_psi_fn(V, psi, M):
    """dG/dpsi for aging law: dG/dpsi = -(V0/dc) * exp(-(psi-f0)/b)."""
    return -M['V0'] / M['dc'] * np.exp(-(psi - M['f0']) / M['b'])

def dtau_da_fn(V, psi, M):
    """
    Explicit partial dtau/da  (holding V, psi fixed).
    tau = N*a*arcsinh(xi),  xi = V/(2*V0)*exp(psi/a)
    d(tau)/da = N*arcsinh(xi) - N*psi*xi / (a*sqrt(1+xi^2))
    """
    xi = xi_fn(V, psi, M)
    return M['N'] * np.arcsinh(xi) - M['N'] * psi * xi / (M['a'] * np.sqrt(1.0 + xi**2))

def dG_da_fn(V, psi, M):
    """dG/da for aging law: G does not depend explicitly on a, so dG/da = 0."""
    return 0.0

print("Physics functions defined.")

# ------------------------------------------------------------------
# Force-balance solver:  tau(V,psi) + eta*V + k*u = tau_L  =>  V
# ------------------------------------------------------------------

def solve_V_algebraic(u, psi, M, tau_L):
    """
    Root-find V from  tau(V,psi) + eta*V = tau_L - k*u.
    tau_L is the current loading stress (may vary with time).
    """
    rhs = tau_L - M['k'] * u
    if rhs <= 0.0:
        raise ValueError(f"Force-balance RHS = {rhs:.4g} <= 0; check tau_L and k*u.")
    def res(V):
        return tau_fn(V, psi, M) + M['eta'] * V - rhs
    Vmin = 1e-30
    Vmax = rhs / M['eta']
    return brentq(res, Vmin, Vmax, xtol=1e-20, rtol=1e-10)

def make_smoothing_matrix(t, sigma):
    """
    Row-normalised Gaussian smoothing matrix for a non-uniform time grid.

    Each column j is weighted by the trapezoidal integration weight w[j] so
    that each row computes a proper time-domain Gaussian average regardless of
    step-size variation.  This makes the kernel consistent with a fixed
    time-window Gaussian for adaptive (non-uniform) time grids.

        S[i,j] ∝ exp(-(t[i]-t[j])^2 / (2*sigma^2)) * w[j],  then row-normalised.

    For a uniform grid the weights are all equal and cancel, recovering the
    standard un-weighted Gaussian to within negligible endpoint corrections.
    """
    if sigma == 0:
        return np.identity(len(t))

    dt = np.diff(t)
    w = np.empty(len(t))
    w[0]    = dt[0] / 2.0
    w[1:-1] = (dt[:-1] + dt[1:]) / 2.0
    w[-1]   = dt[-1] / 2.0

    diff2 = (t[:, None] - t[None, :]) ** 2 / (2.0 * sigma ** 2)
    S = np.exp(-diff2) * w[None, :]   # weight columns by node spacing
    S /= S.sum(axis=1, keepdims=True)
    return S

def setup_initial_conditions(M):
    fss_bg = fss_fn(M['V_bg'], M)
    psi_ss = M['a'] * np.log(2.0 * M['V0'] / M['V_bg'] * np.sinh(fss_bg / M['a']))
    V_init = 1.0e-12
    M['tau0'] = tau_fn(V_init, psi_ss, M) + M['eta'] * V_init
    return 0.0, psi_ss, V_init

def setup_initial_conditions_2block(M, V1_init=1.0e-12, V2_init=1.0e-12):
    """
    Set initial conditions for the two-block coupled system.

    Both blocks start at steady-state psi_ss (evaluated with their respective a_i),
    with user-supplied post-seismic velocities V1_init, V2_init.  tau0_1 and tau0_2
    are derived from the force-balance at t=0 (u1=u2=0) and stored in M.

    Returns (u1_0, psi1_ss, V1_init, u2_0, psi2_ss, V2_init).
    """
    M1 = {**M, 'a': M['a1']}
    M2 = {**M, 'a': M['a2']}

    fss_bg1 = fss_fn(M['V_bg'], M1)
    psi1_ss = M['a1'] * np.log(2.0 * M['V0'] / M['V_bg'] * np.sinh(fss_bg1 / M['a1']))

    fss_bg2 = fss_fn(M['V_bg'], M2)
    psi2_ss = M['a2'] * np.log(2.0 * M['V0'] / M['V_bg'] * np.sinh(fss_bg2 / M['a2']))

    M['tau0_1'] = tau_fn(V1_init, psi1_ss, M1) + M['eta'] * V1_init
    M['tau0_2'] = tau_fn(V2_init, psi2_ss, M2) + M['eta'] * V2_init

    return 0.0, psi1_ss, V1_init, 0.0, psi2_ss, V2_init