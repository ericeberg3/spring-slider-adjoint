import numpy as np
from friction_derivs import *
from adapt_fwd_solve import *
from adjoint_solve import *
from compute_obj import *

M = {}
M['f0']=0.6; M['V0']=1e-6; M['a']=0.010; M['b']=0.015
M['dc']=1e-4; M['N']=50.0; M['eta']=2.7*3.5/2.0
k_crit=M['N']*(M['b']-M['a'])/M['dc']
M['k']=abs(0.9*k_crit); M['V_bg']=1e-9
V_init=1.0e-12
fss_bg=fss_fn(M['V_bg'],M)
psi_ss=M['a']*np.log(2.0*M['V0']/M['V_bg']*np.sinh(fss_bg/M['a']))
u_init=0.0; psi_init=psi_ss
M['tau0']=tau_fn(V_init,psi_init,M)+M['eta']*V_init

T=0.3e7; sigma_smooth=0.05*T
M_true=dict(M)

fwd_true=forward_solve_adaptive(M_true,T,u_init,psi_init)
t_obs_arr=fwd_true['t']; u_obs=fwd_true['u'].copy()

n_ref=int(T/(sigma_smooth/20))+1
t_ref=np.linspace(0.0,T,n_ref)
S_fixed=make_smoothing_matrix(t_ref,sigma_smooth)
u_obs_ref=np.interp(t_ref,t_obs_arr,u_obs)

def run_adj(a_val):
    Mc=dict(M_true); Mc['a']=a_val
    fwd=forward_solve_adaptive(Mc,T,u_init,psi_init)
    u_r=np.interp(t_ref,fwd['t'],fwd['u'])
    sm_ref=S_fixed.T@(S_fixed@u_r - S_fixed@u_obs_ref)
    sm_native=np.interp(fwd['t'],t_ref,sm_ref)
    adj=adjoint_solve(fwd,None,None,Mc,sigma_smooth,smooth_misfit=sm_native)
    return fwd, adj, sm_ref, sm_native

a_true_val=M_true['a']

# 1. Diagnose adjoint at a=0.009 — check p,r magnitudes at key times
print('=== Adjoint diagnostics at a=0.009 ===')
fwd9, adj9, sm_ref9, sm_nat9 = run_adj(0.009)
print('  fwd steps:', len(fwd9['t'])-1)
print('  V_max=%.3e at t=%.2f days' % (fwd9['V'].max(), fwd9['t'][np.argmax(fwd9['V'])]/86400))
print('  sm_ref range: [%.3e, %.3e]' % (sm_ref9.min(), sm_ref9.max()))
print('  sm_native range: [%.3e, %.3e]' % (sm_nat9.min(), sm_nat9.max()))
print('  p range: [%.3e, %.3e]' % (adj9['p'].min(), adj9['p'].max()))
print('  r range: [%.3e, %.3e]' % (adj9['r'].min(), adj9['r'].max()))
print('  lam range: [%.3e, %.3e]' % (adj9['lam'].min(), adj9['lam'].max()))
integrand9 = adj9['lam']*fwd9['dtau_da'] - adj9['r']*fwd9['dG_da']
print('  integrand range: [%.3e, %.3e]' % (integrand9.min(), integrand9.max()))
print('  gradient (integrated): %.4e' % np.trapz(integrand9, fwd9['t']))

print()
print('=== Adjoint diagnostics at a=0.011 ===')
fwd11, adj11, sm_ref11, sm_nat11 = run_adj(0.011)
print('  fwd steps:', len(fwd11['t'])-1)
print('  V_max=%.3e at t=%.2f days' % (fwd11['V'].max(), fwd11['t'][np.argmax(fwd11['V'])]/86400))
print('  sm_ref range: [%.3e, %.3e]' % (sm_ref11.min(), sm_ref11.max()))
print('  p range: [%.3e, %.3e]' % (adj11['p'].min(), adj11['p'].max()))
print('  r range: [%.3e, %.3e]' % (adj11['r'].min(), adj11['r'].max()))
integrand11 = adj11['lam']*fwd11['dtau_da'] - adj11['r']*fwd11['dG_da']
print('  integrand range: [%.3e, %.3e]' % (integrand11.min(), integrand11.max()))
print('  gradient (integrated): %.4e' % np.trapz(integrand11, fwd11['t']))

# 2. Check step sizes during earthquake at a=0.009 and stability |G_psi * dt|
print()
print('=== Step sizes and stability at a=0.009 earthquake ===')
t9 = fwd9['t']; dt9 = np.diff(t9)
# Find earthquake region (high V)
idx_eq = np.where(fwd9['V'] > 1e-3)[0]  # V > 1mm/s
if len(idx_eq) > 0:
    print('  Earthquake: t=[%.4f, %.4f] days (%d steps)' % (
        t9[idx_eq[0]]/86400, t9[idx_eq[-1]]/86400, len(idx_eq)))
    dt_eq = dt9[idx_eq[0]:idx_eq[-1]+1]
    Gpsi_eq = fwd9['G_psi'][idx_eq]
    stability = np.abs(Gpsi_eq[:-1]) * dt_eq
    print('  dt_eq: min=%.3e, max=%.3e, mean=%.3e s' % (dt_eq.min(), dt_eq.max(), dt_eq.mean()))
    print('  |G_psi| during eq: min=%.3e, max=%.3e s^-1' % (np.abs(Gpsi_eq).min(), np.abs(Gpsi_eq).max()))
    print('  |G_psi*dt| stability product: min=%.3e, max=%.3e, mean=%.3e' %
          (stability.min(), stability.max(), stability.mean()))
    print('  (should be < 1 for stability)')
else:
    print('  No high-V earthquake found (V>1e-3)')

# Same for a=0.011
print()
print('=== Step sizes and stability at a=0.011 earthquake ===')
t11 = fwd11['t']; dt11 = np.diff(t11)
idx_eq11 = np.where(fwd11['V'] > 1e-3)[0]
if len(idx_eq11) > 0:
    print('  Earthquake: t=[%.4f, %.4f] days (%d steps)' % (
        t11[idx_eq11[0]]/86400, t11[idx_eq11[-1]]/86400, len(idx_eq11)))
    dt_eq11 = dt11[idx_eq11[0]:idx_eq11[-1]+1]
    Gpsi_eq11 = fwd11['G_psi'][idx_eq11]
    stab11 = np.abs(Gpsi_eq11[:-1]) * dt_eq11
    print('  dt_eq: min=%.3e, max=%.3e, mean=%.3e s' % (dt_eq11.min(), dt_eq11.max(), dt_eq11.mean()))
    print('  |G_psi| during eq: min=%.3e, max=%.3e s^-1' % (np.abs(Gpsi_eq11).min(), np.abs(Gpsi_eq11).max()))
    print('  |G_psi*dt| stability product: min=%.3e, max=%.3e, mean=%.3e' %
          (stab11.min(), stab11.max(), stab11.mean()))
else:
    print('  No high-V earthquake found')

# 3. Test in the steep region a in [0.0100, 0.0110]
print()
print('=== FD vs Adjoint in steep region [0.010, 0.011] ===')
def _J_inv(a_val):
    Mc=dict(M_true); Mc['a']=a_val
    fwd=forward_solve_adaptive(Mc,T,u_init,psi_init)
    u_r=np.interp(t_ref,fwd['t'],fwd['u'])
    return 0.5*np.trapz((S_fixed@u_r - S_fixed@u_obs_ref)**2,t_ref)

def _adj_grad_inv(a_val):
    Mc=dict(M_true); Mc['a']=a_val
    fwd=forward_solve_adaptive(Mc,T,u_init,psi_init)
    u_r=np.interp(t_ref,fwd['t'],fwd['u'])
    sm_ref=S_fixed.T@(S_fixed@u_r - S_fixed@u_obs_ref)
    sm_native=np.interp(fwd['t'],t_ref,sm_ref)
    adj=adjoint_solve(fwd,None,None,Mc,sigma_smooth,smooth_misfit=sm_native)
    return compute_grad_a(fwd,adj,Mc)

da=a_true_val*1e-5
steep_a = [0.0100, 0.0102, 0.0104, 0.0106, 0.0108, 0.0110]
print('%10s  %12s  %12s  %10s  %6s' % ('a','FD grad','Adj grad','rel err','sign'))
print('-'*58)
for a_val in steep_a:
    gfd=(_J_inv(a_val+da)-_J_inv(a_val-da))/(2*da)
    gadj=_adj_grad_inv(a_val)
    rel=abs(gadj-gfd)/(abs(gfd)+1e-30)
    sign='OK' if np.sign(gadj)==np.sign(gfd) or abs(gfd)<1e-6 else 'WRONG'
    print('%10.4f  %12.4e  %12.4e  %10.2e  %6s' % (a_val,gfd,gadj,rel,sign))
