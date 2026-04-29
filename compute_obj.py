from adjoint_solve import *
from adapt_fwd_solve import *

def compute_J(fwd, t_obs, u_obs, sigma):
    """J = 0.5 * int_0^T (Su - Su_obs)^2 dt  (trapezoidal, Gaussian-smoothed misfit)."""
    S            = make_smoothing_matrix(fwd['t'], sigma)
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

# -----------------------------------------------------------------------
# Feature-based objective  J = Σ_i (T_i_mod - T_i_obs)² + (Δu_i_mod - Δu_i_obs)²
# -----------------------------------------------------------------------

def detect_events(fwd, V_thresh=1e-6):
    """
    Detect earthquake events as contiguous intervals where V >= V_thresh.

    Parameters
    ----------
    fwd      : forward solution dict (output of forward_solve_adaptive).
    V_thresh : velocity threshold that defines a seismic event (m/s).

    Returns
    -------
    events : list of dict, one per detected event, with keys:
        idx_start, idx_end, idx_peak — array indices into fwd arrays
        t_start, t_end, t_peak       — times (s)
        delta_u                       — coseismic slip u(t_end) - u(t_start)  (m)
        T                             — recurrence interval to previous event (s);
                                        None for the first event
    """
    V = fwd['V']
    t = fwd['t']
    u = fwd['u']
    n = len(t)

    in_event = V >= V_thresh
    events   = []
    i = 0
    while i < n:
        if in_event[i]:
            start = i
            while i < n and in_event[i]:
                i += 1
            end  = i - 1
            peak = start + int(np.argmax(V[start:end + 1]))
            events.append({
                'idx_start': start,
                'idx_end'  : end,
                'idx_peak' : peak,
                't_start'  : t[start],
                't_end'    : t[end],
                't_peak'   : t[peak],
                'delta_u'  : u[end] - u[start],
                'T'        : None,
            })
        else:
            i += 1

    for k in range(1, len(events)):
        events[k]['T'] = events[k]['t_peak'] - events[k - 1]['t_peak']

    return events


def compute_J_feat(events_mod, events_obs, w_T=1.0, w_du=1.0):
    """
    Feature-based objective function.

      J = w_du * Σ_i (Δu_i_mod - Δu_i_obs)²
        + w_T  * Σ_i (T_i_mod  - T_i_obs )²

    Sums over min(len(events_mod), len(events_obs)) matched event pairs.
    Recurrence-interval terms are only included where both T values are non-None.
    """
    n_pairs = min(len(events_mod), len(events_obs))
    J = 0.0
    for i in range(n_pairs):
        J += w_du * (events_mod[i]['delta_u'] - events_obs[i]['delta_u']) ** 2
        if events_mod[i]['T'] is not None and events_obs[i]['T'] is not None:
            J += w_T * (events_mod[i]['T'] - events_obs[i]['T']) ** 2
    return J


def build_feat_adjoint_forcing(fwd, events_mod, events_obs, M, w_T=1.0, w_du=1.0):
    """
    Build the adjoint forcing sm(t) = dJ_feat/du(t) for the feature-based objective.

    Discretises dJ_feat/du as a vector compatible with the trapezoidal integration
    scheme used by adjoint_solve:  ∫ sm(t) δu(t) dt ≈ Σ_k sm[k] δu[k].
    Delta-function forcings are divided by their node's trapezoidal weight so that
    the RK integration produces the correct pointwise jump.

    Gradient sources
    ----------------
    Δu_i (exact):
        d(Δu_i)/du(t) = δ(t - t_end) - δ(t - t_start)
        → impulses ±coeff at idx_end / idx_start.

    T_i (approximate):
        During the interseismic period V ≈ V_bg, so
        T_i ≈ (u(t_peak_i) - u(t_peak_{i-1})) / V_bg.
        Therefore d(T_i)/du(t) ≈ [δ(t - t_peak_i) - δ(t - t_peak_{i-1})] / V_bg,
        giving impulses at consecutive event peak indices.

    Parameters
    ----------
    fwd        : forward solution dict
    events_mod : list of event dicts for current model (from detect_events)
    events_obs : list of target event dicts
    M          : model-parameter dict  (needs M['V_bg'])
    w_T, w_du  : objective weights

    Returns
    -------
    sm : ndarray, shape (n,)
    """
    t    = fwd['t']
    n    = len(t)
    V_bg = M['V_bg']

    # Trapezoidal node weights: ∫f dt ≈ Σ w[k] f[k]
    w = np.zeros(n)
    w[0]    = (t[1]  - t[0])   / 2.0
    w[-1]   = (t[-1] - t[-2])  / 2.0
    w[1:-1] = (t[2:] - t[:-2]) / 2.0

    sm      = np.zeros(n)
    n_pairs = min(len(events_mod), len(events_obs))

    for i in range(n_pairs):
        em = events_mod[i]
        eo = events_obs[i]

        # --- Δu forcing (exact impulses at event start/end) ---
        du_resid = em['delta_u'] - eo['delta_u']
        coeff_du = 2.0 * w_du * du_resid
        ie, is_  = em['idx_end'], em['idx_start']
        sm[ie]  += coeff_du / w[ie]
        sm[is_] -= coeff_du / w[is_]

        # --- T forcing (approximate impulses at consecutive event peaks) ---
        if em['T'] is not None and eo['T'] is not None:
            T_resid  = em['T'] - eo['T']
            coeff_T  = 2.0 * w_T * T_resid / V_bg
            ip       = em['idx_peak']
            ip_prev  = events_mod[i - 1]['idx_peak']
            sm[ip]      += coeff_T / w[ip]
            sm[ip_prev] -= coeff_T / w[ip_prev]

    return sm