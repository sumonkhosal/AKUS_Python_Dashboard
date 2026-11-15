import numpy as np
import pytest
from akuskf.benchmark import run_benchmark,report_json_ready
from akuskf.jan_model import ModelParameters,matrices,continuous_white_acceleration_Q
from akuskf.synthetic import _jan_euler_step,generate
from akuskf.filtering import run_filter


def test_jan_euler_friction_matches_uploaded_plc_source():
    p=ModelParameters()
    # Stopped, low force -> friction blocks motion, Jan's explicit Euler formula.
    z,v,a=_jan_euler_step(p.z_reference,0.,100.,.02,p)
    assert z==p.z_reference and v==a==0
    # Large drive -> Coulomb+viscous friction subtracted in drive-force direction.
    z,v,a=_jan_euler_step(p.z_reference,.05,500.,.02,p)
    expected=(500.-p.friction_coulomb-p.b_viscous*.05)/p.effective_mass
    assert a==pytest.approx(expected)
    assert z==pytest.approx(p.z_reference+.02*.05)
    assert v==pytest.approx(.05+.02*expected)
    # Negative drive + nonzero speed retains sign-specific friction.
    _,_,an=_jan_euler_step(p.z_reference,.05,-500.,.02,p)
    assert an==pytest.approx((-500.+p.friction_coulomb-p.b_viscous*.05)/p.effective_mass)


def test_matched_model_recovers_Q_R_and_validates_independent_motion():
    b=run_benchmark(plant='linear',dt=.02,n_train=850,n_validation=850,seed=42,q_true=1e-4)
    assert b['training']['scenario'] != b['validation']['scenario']
    assert 0.5 < b['q_est']/b['training']['q_true'] < 1.5
    np.testing.assert_allclose(np.diag(b['R_est']),np.diag(b['training']['R_true']),rtol=.13)
    m=b['metrics']
    assert m['position_rmse_mm'] < m['raw_position_rmse_mm']
    assert m['nis_mean_95pct_nominal_range'][0] < m['mean_nis'] < m['nis_mean_95pct_nominal_range'][1]
    assert m['nees_mean_95pct_nominal_range'][0] < m['mean_nees'] < m['nees_mean_95pct_nominal_range'][1]
    assert 0.90 < m['position_95pct_coverage'] < .99
    assert b['warnings']==[]


def test_nonlinear_mismatch_flagged_as_not_calibrated():
    b=run_benchmark(plant='jan_euler',dt=.02,n_train=850,n_validation=850,seed=42,q_true=1e-4)
    assert b['q_est'] > b['training']['q_true']*2
    assert b['metrics']['mean_nis']>2.5
    assert any('mismatched' in warning for warning in b['warnings'])


def test_report_has_all_Q_R_S_K_and_NO_perfect_data_claim():
    b=run_benchmark(n_train=210,n_validation=210,seed=14)
    report=report_json_ready(b)
    assert np.asarray(report['Q_estimated']).shape == (2,2)
    assert np.asarray(report['R_estimated']).shape == (2,2)
    assert np.asarray(report['last_S']).shape == (2,2)
    assert np.asarray(report['last_K']).shape == (2,2)
    assert 'hardware' in report['notes'][0]


def test_Q_discretization_symmetric_positive_semidefinite():
    p=ModelParameters(); A,B,C,D,Ac=matrices(.02,p)
    q=continuous_white_acceleration_Q(.02,Ac,1e-4)
    assert np.allclose(q,q.T,atol=1e-15)
    assert np.linalg.eigvalsh(q).min()>=-1e-15


def test_filterpy_matches_reference_when_installed():
    pytest.importorskip('filterpy')
    p=ModelParameters()
    a=generate(N=200,seed=19)
    R=a['R_true']
    ref=run_filter(a['y'],a['u'],.02,p,R,a['q_true'],backend='reference')
    fp=run_filter(a['y'],a['u'],.02,p,R,a['q_true'],backend='filterpy')
    np.testing.assert_allclose(fp['estimates'],ref['estimates'],rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(fp['P'],ref['P'],rtol=1e-7,atol=1e-10)
