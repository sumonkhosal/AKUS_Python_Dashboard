"""Repeatable train/independent-validation AKUS covariance-identification benchmark.

The data used here are generated from documented model equations and therefore
have known ground truth. An identified Q/R is not a field calibration result.
The estimator is always the corrected linear two-state model; jan_euler tests
its robustness against Jan's original nonlinear friction force law.
"""
from dataclasses import asdict
import numpy as np
from scipy.stats import chi2

from .synthetic import generate
from .jan_model import ModelParameters
from .identification import estimate_R, estimate_q_by_likelihood, validation_metrics
from .filtering import run_filter


def _nees_diagnostics(result, truth, burn=20):
    """Posterior NEES and 95%-position-interval coverage (synthetic truth only)."""
    e=np.asarray(result['estimates'])[burn:]-np.asarray(truth)[burn:]
    ps=np.asarray(result['P'])[burn:]
    nees=np.array([float(ei@np.linalg.solve(pi,ei)) for ei,pi in zip(e,ps)])
    # 1.95996 SD positional credible interval; coverage assumes Gaussian model.
    coverage=float(np.mean(np.abs(e[:,0])<=1.959963984540054*np.sqrt(ps[:,0,0])))
    return float(nees.mean()),coverage


def _mean_chi2_interval(n,dim=2,alpha=.05):
    """95% CI for mean of iid chi-square draws; a diagnostic reference, NOT
    a valid CI if innovations are time-correlated or model assumptions fail.
    """
    return [float(chi2.ppf(alpha/2,n*dim)/n),
            float(chi2.ppf(1-alpha/2,n*dim)/n)]


def run_benchmark(*, dt=.02, n_train=850, n_validation=850,
                  plant='linear', q_true=0.0001, noise_z_mm=0.35,
                  noise_a_ms2=0.04, seed=42, backend='reference',
                  fit=True, q_fixed=1e-4, q_bounds=(1e-9,1.0),
                  model=None):
    """Identify stationary R, dynamic q on TRAINING run, hold out a separate run.

    `seed+7919` and a different drive-force profile make validation independent
    in both stochastic realizations and input excitation. Never uses the
    known truth values in fitting; they are for *validation only*.
    """
    if model is None: model=ModelParameters()
    model.validate()
    common=dict(dt=dt,p=model,sensor_std_z=noise_z_mm*1e-3,
                sensor_std_a=noise_a_ms2,true_q=q_true,plant=plant)
    train=generate(N=int(n_train),seed=int(seed),scenario='training',**common)
    holdout=generate(N=int(n_validation),seed=int(seed)+7919,scenario='validation',**common)
    R=estimate_R(train['calibration_residuals'])
    if fit:
        q,fit_info=estimate_q_by_likelihood(train['y'],train['u'],dt,model,R,bounds=q_bounds)
    else:
        q=float(q_fixed)
        if q<0: raise ValueError('q_fixed must be non-negative')
        fit_info={'mode':'manual q; no identification'}
    training_result=run_filter(train['y'],train['u'],dt,model,R,q,backend=backend)
    val_result=run_filter(holdout['y'],holdout['u'],dt,model,R,q,backend=backend)
    burn=min(30,int(n_validation)//10)
    metrics=validation_metrics(val_result,holdout['truth'],first=burn)
    nees,coverage=_nees_diagnostics(val_result,holdout['truth'],burn=burn)
    metrics['mean_nees']=nees
    metrics['position_95pct_coverage']=coverage
    metrics['raw_position_rmse_mm']=float(
        1e3*np.sqrt(np.mean((holdout['y'][burn:,0]-holdout['truth'][burn:,0])**2)))
    metrics['nis_mean_95pct_nominal_range']=_mean_chi2_interval(metrics['n_samples'])
    metrics['nees_mean_95pct_nominal_range']=_mean_chi2_interval(len(holdout['truth'])-burn)
    metrics['q_ratio_to_known_injected']=float(q/q_true) if q_true>0 else None
    metrics['R_acc_ratio_to_known_injected']=float(R[1,1]/train['R_true'][1,1])
    metrics['R_position_ratio_to_known_injected']=float(R[0,0]/train['R_true'][0,0])
    metrics['training_samples']=int(n_train)
    metrics['validation_samples']=int(n_validation)
    warnings=[]
    if fit and fit_info.get('near_bound'):
        warnings.append('q optimum is near the search boundary; Q is not identified reliably')
    if not (metrics['nis_mean_95pct_nominal_range'][0]<=metrics['mean_nis']<=
            metrics['nis_mean_95pct_nominal_range'][1]):
        warnings.append('Held-out mean NIS is outside nominal interval')
    if not (metrics['nees_mean_95pct_nominal_range'][0]<=nees<=
            metrics['nees_mean_95pct_nominal_range'][1]):
        warnings.append('Held-out mean NEES is outside nominal interval')
    if abs(metrics['acceleration_lag1_correlation'])>.15 or abs(metrics['position_lag1_correlation'])>.15:
        warnings.append('Held-out innovations appear temporally correlated')
    for name, dataset in (('training',train),('validation',holdout)):
        if np.min(dataset['truth'][:,0]) < 0.0 or np.max(dataset['truth'][:,0]) > 0.611111:
            warnings.append(f'{name} simulated truth leaves 0–611.1 mm AKUS nominal travel; physical feasibility not established')
    if plant=='jan_euler':
        warnings.append('Jan nonlinear Euler plant and linear estimator are intentionally mismatched')
    return dict(training=train,validation=holdout,R_est=R,q_est=q,
                train_result=training_result,validation_result=val_result,
                fit_info=fit_info,metrics=metrics,warnings=warnings,
                parameters=asdict(model),dt=dt,plant=plant,
                backend=backend,seed=int(seed))


def report_json_ready(out):
    """Compact JSON-safe report without storing huge arrays or truth in metadata."""
    def clean(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)): return [clean(v) for v in value]
        return value
    r=out['validation_result']
    return clean(dict(plant=out['plant'],backend=out['backend'],seed=out['seed'],
                      dt_seconds=out['dt'],parameters=out['parameters'],
                      q_injected=out['training']['q_true'],q_estimated=out['q_est'],
                      Q_injected=out['training']['Q_true'],Q_estimated=r['Q'],
                      R_injected=out['training']['R_true'],R_estimated=out['R_est'],
                      first_K=r['gains'][0],last_K=r['gains'][-1],
                      first_S=r['S'][0],last_S=r['S'][-1],
                      fit=out['fit_info'],metrics=out['metrics'],warnings=out['warnings'],
                      notes=['Synthetic, known model/noise; not validated against hardware',
                             'Validation uses independent random seed and force profile',
                             'NIS and NEES intervals assume temporally independent residuals']))
