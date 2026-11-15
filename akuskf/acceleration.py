"""Offline, acceleration-FIRST Kalman research for AKUS.

This is a *separate* one-state acceleration/residual estimator. It does NOT
pretend its fitted Q is Jan's two-state [position, velocity] process covariance.

Random-walk residual model:
    a_measured[k] = a_prior[k] + d[k] + measurement_noise[k]
    d[k+1] = d[k] + w[k], Var(w[k]) = q * dt

If no independent acceleration prior is available, a_prior=0, so d estimates
calibrated acceleration directly. In model-guided simulation the prior is
computed from Jan's deterministic linear physics using force inputs and never
from simulated ground truth or the accelerometer. In real recordings, torque
scaling / polarity have not been independently verified: model guide is optional.

Only real, observed stationary noise can give an empirical stationary R. Any
scalar Kalman method that smooths acceleration can attenuate *real* vibration;
innovation whiteness / held-out testing and bandwidth characterization matter.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from .jan_model import matrices, ModelParameters


def run_acceleration_filter(measured_ms2, dt, r, q, *, prior_ms2=None,
                            initial_residual=0.0, initial_variance=None):
    """One-state residual KF with transparent Joseph covariance update.

    q: continuous random-walk process spectral density [(m/s²)²/s].
    r: calibrated measurement noise variance [(m/s²)²] at THIS sampling rate.
    """
    y=np.asarray(measured_ms2,dtype=float).reshape(-1)
    if y.size<2 or not np.all(np.isfinite(y)):
        raise ValueError('At least two finite acceleration samples are required')
    if not (np.isfinite(dt) and dt>0 and np.isfinite(r) and r>0 and np.isfinite(q) and q>=0):
        raise ValueError('Invalid dt, R or q')
    baseline=np.zeros_like(y) if prior_ms2 is None else np.asarray(prior_ms2,dtype=float).reshape(-1)
    if baseline.shape != y.shape or not np.all(np.isfinite(baseline)):
        raise ValueError('prior_ms2 must match measurements and be finite')
    d=float(initial_residual)
    P=float(max(r,1e-12) if initial_variance is None else initial_variance)
    if not(np.isfinite(d) and np.isfinite(P) and P>0):
        raise ValueError('Invalid initial residual/covariance')
    Q=float(q*dt)
    estimated=np.zeros_like(y)
    gains=np.zeros_like(y)
    s_cov=np.zeros_like(y)
    innovations=np.zeros_like(y)
    variances=np.zeros_like(y)
    for k in range(y.size):
        # The prior dynamics are modelled as random walk around the supplied
        # deterministic acceleration. No simulated truth enters this recursion.
        P_minus=P+Q
        S=P_minus+r
        nu=(y[k]-baseline[k])-d
        K=P_minus/S
        d+=K*nu
        P=(1-K)**2 * P_minus + (K*K)*r
        estimated[k]=baseline[k]+d
        gains[k]=K
        s_cov[k]=S
        innovations[k]=nu
        variances[k]=P
    nis=innovations**2/s_cov
    loglike=float(np.sum(np.log(2*np.pi*s_cov)+nis))
    return {'estimated_acc_ms2':estimated,'K':gains,'S':s_cov,'Q_step':Q,
            'R':float(r),'P':variances,'innovation':innovations,'nis':nis,
            'negative_loglikelihood_twice':loglike,'prior_ms2':baseline}


def fit_acceleration_q(train_acc_ms2,dt,r, *, prior_ms2=None,
                       q_bounds=(1e-10,10.0)):
    """Constrained scalar maximum innovation likelihood, fit TRAINING only."""
    if len(train_acc_ms2)<50: raise ValueError('Need 50 or more training samples')
    lo,hi=map(float,q_bounds)
    if lo<=0 or hi<=lo: raise ValueError('Require 0 < low bound < high bound')
    def objective(log_q):
        return run_acceleration_filter(train_acc_ms2,dt,r,np.exp(log_q),
                                       prior_ms2=prior_ms2)['negative_loglikelihood_twice']
    fit=minimize_scalar(objective,bounds=(np.log(lo),np.log(hi)),
                        method='bounded',options={'xatol':1e-4})
    q=float(np.exp(fit.x))
    near_bound=(q/lo<1.15 or hi/q<1.15)
    return q,{'converged':bool(fit.success),'near_bound':bool(near_bound),
              'objective':float(fit.fun),'bounds':(lo,hi),
              'note':'Q here is scalar acceleration-residual Q=q*dt, NOT Jan position/velocity Q matrix'}


def estimate_acceleration_r(stationary_acc_ms2):
    """Initial stationary-sample R; independent noise assumption must be checked."""
    a=np.asarray(stationary_acc_ms2,float).reshape(-1)
    if len(a)<50 or not np.isfinite(a).all():raise ValueError('Need 50 stationary finite samples')
    r=float(np.var(a,ddof=1))
    if r<=0:raise ValueError('Zero apparent sensor noise: cannot infer R from constant samples')
    corr=float(np.corrcoef(a[:-1],a[1:])[0,1]) if len(a)>3 else float('nan')
    return r,corr


def jan_linear_acceleration_prior(force_input,dt,p=None,initial_z=None,initial_v=0.0):
    """Independent prior from deterministic Jan-style LINEAR physics + force.

    Inputs must be gravity-compensated net force u [N]. The result is NOT a
    reconstruction of Jan's nonlinear friction, and has no sensor/truth access.
    """
    if p is None:p=ModelParameters()
    p.validate()
    u=np.asarray(force_input,float).reshape(-1)
    if len(u)<2 or not np.isfinite(u).all():raise ValueError('Invalid force profile')
    A,B,C,D,_=matrices(dt,p)
    state=np.array([p.z_reference if initial_z is None else initial_z,initial_v],float)
    a=np.empty(len(u),float)
    for k in range(len(u)):
        a[k]=(u[k]-p.b_viscous*state[1])/p.effective_mass
        if k+1<len(u): state=A@state + B[:,0]*u[k]
    return a


def acceleration_diagnostics(filtered,truth=None,burn=20):
    """Held-out quality. Comparison to real hardware filtered PLC is NOT truth."""
    k=min(int(burn),len(filtered['nis'])//5)
    v=filtered['innovation'][k:]
    nis=filtered['nis'][k:]
    out={'mean_nis':float(np.mean(nis)),
         'innovation_mean_ms2':float(np.mean(v)),
         'innovation_lag1':float(np.corrcoef(v[:-1],v[1:])[0,1]) if len(v)>3 and np.std(v)>0 else float('nan'),
         'n_validation_samples':len(v)}
    if truth is not None:
        truth=np.asarray(truth,float).reshape(-1)
        if truth.shape!=filtered['estimated_acc_ms2'].shape:raise ValueError('Truth length mismatch')
        out['kalman_acc_rmse_ms2']=float(np.sqrt(np.mean((filtered['estimated_acc_ms2'][k:]-truth[k:])**2)))
    return out
