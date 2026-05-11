import numpy as np
from friction_derivs import *
from adapt_fwd_solve import *

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
fwd_true=forward_solve_adaptive(M_true,T,u_init,psi_init)
t_obs_arr=fwd_true['t']; u_obs=fwd_true['u'].copy()

n_ref=int(T/(sigma_smooth/20))+1
t_ref=np.linspace(0.0,T,n_ref)
S_fixed=make_smoothing_matrix(t_ref,sigma_smooth)
u_obs_ref=np.interp(t_ref,t_obs_arr,u_obs)

def _J_inv(a_val):
    Mc=dict(M_true); Mc['a']=a_val
    fwd=forward_solve_adaptive(Mc,T,u_init,psi_init)
    u_r=np.interp(t_ref,fwd['t'],fwd['u'])
    return 0.5*np.trapz((S_fixed@u_r - S_fixed@u_obs_ref)**2,t_ref)

a_vals = np.linspace(0.007, 0.013, 31)
J_vals = []
V_max_times = []  # time of peak V (earthquake timing)

print('Scanning J landscape and earthquake timing...')
print('%8s  %12s  %12s  %8s' % ('a','J','t_Vmax(days)','n_steps'))
for a_val in a_vals:
    Mc=dict(M_true); Mc['a']=a_val
    fwd=forward_solve_adaptive(Mc,T,u_init,psi_init)
    u_r=np.interp(t_ref,fwd['t'],fwd['u'])
    J=0.5*np.trapz((S_fixed@u_r - S_fixed@u_obs_ref)**2,t_ref)
    idx_vmax=np.argmax(fwd['V'])
    t_vmax=fwd['t'][idx_vmax]/86400
    J_vals.append(J)
    V_max_times.append(fwd['t'][idx_vmax])
    print('%8.5f  %12.4e  %12.3f  %8d' % (a_val,J,t_vmax,len(fwd['t'])-1))

print('\nDone.')
# Save for potential plotting
np.save('/tmp/landscape_a.npy', a_vals)
np.save('/tmp/landscape_J.npy', np.array(J_vals))
np.save('/tmp/landscape_tvm.npy', np.array(V_max_times))
