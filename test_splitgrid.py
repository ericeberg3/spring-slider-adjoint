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

T=0.3e7
sigma_smooth=0.05*T

M_true=dict(M)
print('Running true forward solve...')
fwd_true=forward_solve_adaptive(M_true,T,u_init,psi_init)
t_obs_arr=fwd_true['t']; u_obs=fwd_true['u'].copy()
print('  True solve: %d steps' % (len(fwd_true['t'])-1))

n_ref=int(T/(sigma_smooth/20))+1
t_ref=np.linspace(0.0,T,n_ref)
S_fixed=make_smoothing_matrix(t_ref,sigma_smooth)
u_obs_ref=np.interp(t_ref,t_obs_arr,u_obs)
print('  t_ref: %d uniform points' % n_ref)

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

a_true_val=M_true['a']
da=a_true_val*1e-5
a_test_vals=[a_true_val*f for f in [0.9, 1.0, 1.1, 1.2]]

print('')
print('%10s  %13s  %13s  %10s  %6s' % ('a','FD grad','Adj grad','rel err','sign'))
print('-'*58)
for a_val in a_test_vals:
    gfd=(_J_inv(a_val+da)-_J_inv(a_val-da))/(2*da)
    gadj=_adj_grad_inv(a_val)
    rel=abs(gadj-gfd)/(abs(gfd)+1e-30)
    sign='OK' if np.sign(gadj)==np.sign(gfd) else 'WRONG'
    print('%10.6f  %13.4e  %13.4e  %10.2e  %6s' % (a_val,gfd,gadj,rel,sign))
