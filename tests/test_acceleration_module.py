"""Tests of v4 acceleration-specific estimator (not the 2-state Jan Kalman)."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from akuskf.acceleration import (
    run_acceleration_filter, fit_acceleration_q, estimate_acceleration_r,
    jan_linear_acceleration_prior, acceleration_diagnostics,
)
from akuskf.synthetic import generate
from akuskf.jan_model import ModelParameters
from kalman_lab import _decimate


def test_known_acceleration_random_walk_recover_process_q_and_R():
    dt=.02; q_true=.001; r_true=.04**2
    rng=np.random.default_rng(15)
    n=3500
    a_true=np.cumsum(rng.normal(0,np.sqrt(q_true*dt),n))
    meas=a_true+rng.normal(0,np.sqrt(r_true),n)
    independent_stationary=rng.normal(0,np.sqrt(r_true),3000)
    r,corr=estimate_acceleration_r(independent_stationary)
    assert .90*r_true<r<1.10*r_true
    assert abs(corr)<.10
    q,info=fit_acceleration_q(meas[:2000],dt,r,q_bounds=(1e-6,.2))
    assert .35*q_true<q<2.5*q_true
    val=run_acceleration_filter(meas[2000:],dt,r,q)
    diagnostics=acceleration_diagnostics(val,a_true[2000:])
    assert diagnostics['kalman_acc_rmse_ms2'] < np.sqrt(np.mean((meas[2000:]-a_true[2000:])**2))
    assert val['S'].shape==(1500,) and val['K'].shape==(1500,)
    assert np.all((val['K']>0)&(val['K']<1))
    assert info['converged']


def test_linear_force_prior_does_not_use_measurements_or_truth():
    sim=generate(N=230,dt=.02,plant='linear',seed=123)
    prior=jan_linear_acceleration_prior(sim['u'],.02,ModelParameters())
    assert prior.shape==(230,) and np.isfinite(prior).all()
    # Independent, exact linear dynamics mean expected acceleration matches
    # up to stochastic disturbances; should remain close on this fixture.
    assert np.sqrt(np.mean((prior-sim['acc_truth'])**2))<.008


def test_acceleration_filter_rejects_invalid_noise():
    with pytest.raises(ValueError):run_acceleration_filter([0.,1.],.01,0.,1e-3)
    with pytest.raises(ValueError):run_acceleration_filter([0.,1.],.01,.01,-1)
    with pytest.raises(ValueError):run_acceleration_filter([0.,1.],.01,.01,1e-3,prior_ms2=[0])


def test_real_earth_replay_sample_noise_and_filter():
    root=Path(__file__).resolve().parents[1]
    stationary=pd.read_csv(root/'examples'/'akus_stationary_REAL.csv')
    motion=pd.read_csv(root/'examples'/'akus_movement_REAL.csv')
    st,dt=_decimate(stationary,.01)
    mo,dt2=_decimate(motion,.01)
    assert dt == pytest.approx(.01,abs=.001) and dt2 == pytest.approx(.01,abs=.001)
    baseline=stationary.fAccZUnfiltered.median()
    cal=(st.fAccZUnfiltered.to_numpy()-baseline)*9.80665
    y=(mo.fAccZUnfiltered.to_numpy()-baseline)*9.80665
    r,corr=estimate_acceleration_r(cal)
    assert 0<r<.01 and np.isfinite(corr)
    q,info=fit_acceleration_q(y[:500],dt,r)
    out=run_acceleration_filter(y,dt,r,q)
    assert np.isfinite(out['estimated_acc_ms2']).all()
    assert np.isfinite(out['nis']).all()
    assert len(out['estimated_acc_ms2'])==len(y)
