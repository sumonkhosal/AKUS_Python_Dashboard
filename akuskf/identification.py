"""Noise identification: separate sensor calibration from model fitting."""
import numpy as np
from scipy.optimize import minimize_scalar
from .filtering import run_filter


def estimate_R(stationary_residuals:np.ndarray):
    """Estimate covariance of simultaneous position and acceleration noise.

    The supplied residual signals must be calibrated and, for position,
    centered about a known stationary position. For acceleration, stationary
    gravity projection and bias must already be removed.
    """
    r=np.asarray(stationary_residuals,dtype=float)
    if r.ndim!=2 or r.shape[1]!=2 or len(r)<30 or not np.all(np.isfinite(r)):
        raise ValueError('Need >= 30 synchronized noise residual samples in SI units')
    R=np.cov(r,rowvar=False,ddof=1)
    if np.min(np.linalg.eigvalsh(R)) <= 0:raise ValueError('Estimated R is singular')
    return R


def estimate_q_by_likelihood(y,u,dt,model,R,train_slice=None,bounds=(1.e-9,1.)):
    """Estimate ONE positive physical spectral density q_a, keeping R fixed.

    Optimize innovation marginal Gaussian negative log-likelihood.
    Fits only model-observable process-uncertainty structure, not each Q entry.
    Fit on training segment; always independently validate later.
    """
    yl=np.asarray(y,dtype=float); ul=np.asarray(u,dtype=float)
    if train_slice is not None:
        yl=yl[train_slice]; ul=ul[train_slice]
    if len(yl)<40:raise ValueError('Need enough dynamic samples for identification')
    lo,hi=bounds
    if lo<=0 or lo>=hi:raise ValueError('Invalid q bounds')
    def objective(log_q):
        fit=run_filter(yl,ul,dt,model,R,np.exp(log_q),backend='reference')
        return float(np.nansum(fit['nll']))
    opt=minimize_scalar(objective,bounds=(np.log(lo),np.log(hi)),method='bounded',
                        options=dict(xatol=0.03,maxiter=80))
    if not opt.success:raise RuntimeError('q fit did not converge')
    return float(np.exp(opt.x)),dict(nll=float(opt.fun),iterations=int(opt.nfev),
                                    near_bound=bool(opt.x<=np.log(lo)+0.06 or opt.x>=np.log(hi)-0.06))


def validation_metrics(result,truth=None,first=25):
    """NIS and optional simulation-ground-truth position error diagnostics."""
    valid=np.asarray(result['nis'])[first:]
    valid=valid[np.isfinite(valid)]
    out=dict(mean_nis=float(np.mean(valid)),n_samples=int(len(valid)))
    inn=np.asarray(result['innovations'])[first:]
    for j,name in enumerate(('position','acceleration')):
        v=inn[:,j]; v=v[np.isfinite(v)]
        out[f'{name}_innovation_mean']=float(np.mean(v))
        out[f'{name}_lag1_correlation']=float(np.corrcoef(v[:-1],v[1:])[0,1]) if len(v)>3 and np.std(v)>0 else float('nan')
    if truth is not None:
        e=result['estimates'][first:]-np.asarray(truth)[first:]
        out['position_rmse_mm']=float(np.sqrt(np.mean(e[:,0]**2))*1000)
        out['velocity_rmse_mm_s']=float(np.sqrt(np.mean(e[:,1]**2))*1000)
    return out
