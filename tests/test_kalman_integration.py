from pathlib import Path
import numpy as np
from akus_sim import Simulator
from kalman_lab import (load_twincat_csv, make_real_observations,
                        make_simulation_observations, analyze_frequency,
                        estimate_and_validate)
ROOT = Path(__file__).parent.parent

def test_real_excerpt_parses_and_R_is_positive():
    stationary = load_twincat_csv(ROOT/'examples'/'akus_stationary_REAL.csv')
    moving = load_twincat_csv(ROOT/'examples'/'akus_movement_REAL.csv')
    assert len(stationary) > 10000 and len(moving) > 10000
    data=make_real_observations(moving,stationary,.02)
    assert data['y'].shape[1]==2
    assert 0.99 < data['baseline_g'] < 1.01
    assert np.all(np.linalg.eigvalsh(data['R'])>0)
    assert np.isfinite(data['u']).all()
    assert data['raw_R'][1,1] > data['R'][1,1]  # anti-alias filtering reduces high-f noise
    power=analyze_frequency(moving,3,stationary)
    assert 'stationary_psd' in power
    assert np.isfinite(power['acc_psd']).all()

def test_real_heldout_runs_and_Q_estimates_without_ground_truth_claim():
    stationary=load_twincat_csv(ROOT/'examples'/'akus_stationary_REAL.csv')
    moving=load_twincat_csv(ROOT/'examples'/'akus_movement_REAL.csv')
    data=make_real_observations(moving,stationary,.02)
    fit=estimate_and_validate(data,fit=True)
    assert fit['q'] > 0
    assert fit['result']['gains'].shape[1:]==(2,2)
    assert np.isfinite(fit['metrics']['mean_nis'])
    assert 'position_rmse_mm' not in fit['metrics']

def test_existing_carrier_history_connected_to_filter():
    sim=Simulator()
    sim.start_homing()
    sim.finish_operation()
    sim.move_absolute(100,2800,5000,5000)
    sim.finish_operation()
    data=make_simulation_observations(sim.history,sim.config,seed=123)
    assert np.isclose(data['truth'][-1,0], 0.1)
    fit=estimate_and_validate(data,fit=False)
    assert fit['result']['estimates'].shape==(len(data['y']),2)
    assert 'position_rmse_mm' in fit['metrics']
