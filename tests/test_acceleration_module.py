"""Comprehensive automated tests for the v4 Acceleration-First Kalman module (Section 14)."""
from pathlib import Path
import io
import py_compile
import numpy as np
import pandas as pd
import pytest

from akuskf.acceleration import (
    run_acceleration_filter,
    fit_acceleration_q,
    estimate_acceleration_r,
    analyze_acceleration_noise,
    evaluate_prior_accuracy,
    jan_linear_acceleration_prior,
    jan_nonlinear_acceleration_prior,
    acceleration_diagnostics,
    run_acceleration_benchmark,
)
from akuskf.twincat_export import (
    generate_twincat_st_code,
    verify_python_to_twincat_equivalence,
)
from akuskf.synthetic import generate
from akuskf.jan_model import ModelParameters
from kalman_lab import _decimate, load_twincat_csv


def test_known_acceleration_random_walk_recover_process_q_and_R():
    dt = 0.02
    q_true = 0.001
    r_true = 0.04 ** 2
    rng = np.random.default_rng(15)
    n = 3500
    a_true = np.cumsum(rng.normal(0, np.sqrt(q_true * dt), n))
    meas = a_true + rng.normal(0, np.sqrt(r_true), n)
    independent_stationary = rng.normal(0, np.sqrt(r_true), 3000)
    r, corr = estimate_acceleration_r(independent_stationary)
    assert 0.90 * r_true < r < 1.10 * r_true
    assert abs(corr) < 0.10
    q, info = fit_acceleration_q(meas[:2000], dt, r, q_bounds=(1e-6, 0.2))
    assert 0.35 * q_true < q < 2.5 * q_true
    val = run_acceleration_filter(meas[2000:], dt, r, q)
    diagnostics = acceleration_diagnostics(val, a_true[2000:], raw_measured=meas[2000:])
    assert diagnostics["kalman_acc_rmse_ms2"] < diagnostics["raw_acc_rmse_ms2"]
    assert diagnostics["kalman_acc_mae_ms2"] < diagnostics["raw_acc_mae_ms2"]
    assert val["S"].shape == (1500,) and val["K"].shape == (1500,)
    assert np.all((val["K"] > 0) & (val["K"] < 1))
    assert info["converged"]


@pytest.mark.parametrize("seed", [11, 29, 73])
def test_r_identification_approaches_injected_across_seeds(seed):
    r_injected = 0.035 ** 2
    rng = np.random.default_rng(seed)
    residuals = rng.normal(0.0, np.sqrt(r_injected), size=4000)
    report = analyze_acceleration_noise(residuals, dt=0.01, r_injected=r_injected)
    assert abs(report["relative_error"]) < 0.08
    assert np.isclose(report["std_estimated_ms2"] ** 2, report["r_estimated"])
    assert len(report["hist_centers"]) == 35
    assert len(report["psd_freq_hz"]) > 10


def test_filterpy_backend_matches_reference_exactly():
    pytest.importorskip("filterpy")
    rng = np.random.default_rng(99)
    y = np.cumsum(rng.normal(0.0, 0.005, size=400)) + rng.normal(0.0, 0.03, size=400)
    prior = 0.02 * np.sin(np.linspace(0, 6.0, 400))
    ref = run_acceleration_filter(y, 0.01, 0.03 ** 2, 0.002, prior_ms2=prior, backend="reference")
    fp = run_acceleration_filter(y, 0.01, 0.03 ** 2, 0.002, prior_ms2=prior, backend="filterpy")
    np.testing.assert_allclose(fp["estimated_acc_ms2"], ref["estimated_acc_ms2"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fp["K"], ref["K"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fp["S"], ref["S"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fp["P"], ref["P"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("seed", [42, 108])
def test_acceleration_benchmark_improves_rmse_and_mae_across_seeds(seed):
    out = run_acceleration_benchmark(
        plant="linear",
        dt=0.01,
        n_train=900,
        n_validation=900,
        q_true=0.001,
        sensor_std_a=0.04,
        seed=seed,
        use_prior=True,
        backend="filterpy",
        fit_q=True,
    )
    m = out["metrics"]
    assert m["kalman_acc_rmse_ms2"] < m["raw_acc_rmse_ms2"]
    assert m["kalman_acc_mae_ms2"] < m["raw_acc_mae_ms2"]
    assert m["backend_max_abs_diff"] < 1e-11
    # 1D NIS and 1D NEES should have expected mean ~ 1.0 (not 2.0)
    lo, hi = m["nis_95pct_interval_1d"]
    assert 0.75 < lo < 1.0 < hi < 1.30
    assert 0.65 < m["mean_nis"] < 1.45
    assert 0.55 < m["mean_nees"] < 1.65


def test_nonlinear_euler_mismatch_detected_in_acceleration_benchmark():
    out = run_acceleration_benchmark(
        plant="jan_euler",
        dt=0.01,
        n_train=900,
        n_validation=900,
        q_true=0.0002,
        sensor_std_a=0.03,
        seed=42,
        use_prior=True,
        prior_mode="linear",
    )
    # Linear force prior omits nonlinear Coulomb/position friction, so fitted q_lin inflates
    assert out["ablation"]["linear_prior_kf"]["q_estimated"] > out["metrics"]["q_injected"] * 1.5
    assert out["ablation"]["nonlinear_prior_kf"]["q_estimated"] < out["ablation"]["linear_prior_kf"]["q_estimated"] * 0.2


def test_em_and_mle_q_identification():
    rng = np.random.default_rng(321)
    dt = 0.01
    q_true = 0.002
    r_true = 0.03 ** 2
    a_true = np.cumsum(rng.normal(0.0, np.sqrt(q_true * dt), size=1200))
    y = a_true + rng.normal(0.0, np.sqrt(r_true), size=1200)
    q_mle, info_mle = fit_acceleration_q(y, dt, r_true, method="mle")
    q_em, info_em = fit_acceleration_q(y, dt, r_true, method="em")
    assert info_mle["converged"] and info_em["converged"]
    assert 0.3 * q_true < q_mle < 3.0 * q_true
    assert 0.3 * q_true < q_em < 3.0 * q_true


def test_linear_force_prior_does_not_use_measurements_or_truth():
    sim = generate(N=230, dt=0.02, plant="linear", seed=123)
    prior = jan_linear_acceleration_prior(sim["u"], 0.02, ModelParameters())
    assert prior.shape == (230,) and np.isfinite(prior).all()
    assert np.sqrt(np.mean((prior - sim["acc_truth"]) ** 2)) < 0.008


def test_acceleration_filter_rejects_invalid_noise():
    with pytest.raises(ValueError):
        run_acceleration_filter([0.0, 1.0], 0.01, 0.0, 1e-3)
    with pytest.raises(ValueError):
        run_acceleration_filter([0.0, 1.0], 0.01, 0.01, -1.0)
    with pytest.raises(ValueError):
        run_acceleration_filter([0.0, 1.0], 0.01, 0.01, 1e-3, prior_ms2=[0.0])


def test_real_earth_replay_sample_noise_and_filter():
    root = Path(__file__).resolve().parents[1]
    stationary = pd.read_csv(root / "examples" / "akus_stationary_REAL.csv")
    motion = pd.read_csv(root / "examples" / "akus_movement_REAL.csv")
    st, dt = _decimate(stationary, 0.01)
    mo, dt2 = _decimate(motion, 0.01)
    assert dt == pytest.approx(0.01, abs=0.001) and dt2 == pytest.approx(0.01, abs=0.001)
    baseline = stationary.fAccZUnfiltered.median()
    cal = (st.fAccZUnfiltered.to_numpy() - baseline) * 9.80665
    y = (mo.fAccZUnfiltered.to_numpy() - baseline) * 9.80665
    r, corr = estimate_acceleration_r(cal)
    assert 0 < r < 0.01 and np.isfinite(corr)
    q, info = fit_acceleration_q(y[:500], dt, r)
    out = run_acceleration_filter(y, dt, r, q)
    assert np.isfinite(out["estimated_acc_ms2"]).all()
    assert np.isfinite(out["nis"]).all()
    assert len(out["estimated_acc_ms2"]) == len(y)


def test_twincat_channel_discovery_aliases():
    df = pd.DataFrame({
        "MAIN.time_ms": np.arange(60) * 10.0,
        "MAIN.fPos1": np.zeros(60),
        "MAIN.fPos2": np.zeros(60),
        "MAIN.fVel1": np.zeros(60),
        "MAIN.fVel2": np.zeros(60),
        "MAIN.fTorque1": np.zeros(60),
        "MAIN.fTorque2": np.zeros(60),
        "MAIN.fAvg3702Z": np.ones(60),
        "MAIN.fAccZ": np.ones(60) * 9.80665,
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    parsed = load_twincat_csv(buf)
    assert "fAccZUnfiltered" in parsed.columns
    assert "fAvg3702Z" in parsed.columns
    assert len(parsed) == 60


def test_streamlit_app_compiles_cleanly():
    root = Path(__file__).resolve().parents[1]
    py_compile.compile(str(root / "app.py"), doraise=True)
    py_compile.compile(str(root / "desktop_app.py"), doraise=True)


def test_priority1_open_loop_prior_vs_ground_truth():
    p = ModelParameters()
    s_lin = generate(dt=0.01, N=950, p=p, plant="linear", seed=7961, scenario="validation", true_q=1e-4)
    pr_lin = jan_linear_acceleration_prior(s_lin["u"], 0.01, p)
    ev_lin = evaluate_prior_accuracy(pr_lin, s_lin["acc_truth"])
    assert ev_lin["prior_rmse_ms2"] < 0.01
    assert ev_lin["prior_r2_score"] > 0.95

    s_nl = generate(dt=0.01, N=950, p=p, plant="jan_euler", seed=7961, scenario="validation", true_q=1e-4)
    pr_nl = jan_linear_acceleration_prior(s_nl["u"], 0.01, p)
    ev_nl = evaluate_prior_accuracy(pr_nl, s_nl["acc_truth"])
    # Nonlinear Coulomb + position friction degrades the linear prior
    assert ev_nl["prior_rmse_ms2"] > 0.03


def test_priority2_innovation_autocorrelation_and_nees_consistency():
    out_lin = run_acceleration_benchmark(
        plant="linear", dt=0.01, n_train=950, n_validation=950, q_true=1e-4, sensor_std_a=0.04, seed=42
    )
    m_lin = out_lin["metrics"]
    assert abs(m_lin["innovation_lag1"]) <= m_lin["innovation_whiteness_bound_95"]
    assert m_lin["ljung_box_pvalue"] > 0.05
    assert 0.7 < m_lin["mean_nees"] < 1.5

    out_nl_mismatched = run_acceleration_benchmark(
        plant="jan_euler", dt=0.01, n_train=950, n_validation=950,
        q_true=1e-4, sensor_std_a=0.04, seed=42, prior_mode="linear"
    )
    m_nl_lin = out_nl_mismatched["metrics"]
    # Even when mean_nis is near 1.0 (~1.057), inflated Q makes posterior P over-conservative,
    # so mean_nees drops outside the 95% chi2(1) interval [0.911, 1.093]
    assert not m_nl_lin["nees_consistent_95pct"]


def test_priority3_model_guided_kf_outperforms_random_walk_kf_without_prior():
    out = run_acceleration_benchmark(
        plant="linear", dt=0.01, n_train=950, n_validation=950, q_true=1e-4, sensor_std_a=0.04, seed=42
    )
    abl = out["ablation"]
    rw = abl["random_walk_kf"]
    mg = abl["model_guided_kf"]
    # Model-guided KF should beat Random-Walk KF (prior_ms2=None) by > 45% RMSE on matched linear plant
    assert mg["rmse_ms2"] < rw["rmse_ms2"] * 0.55
    assert mg["q_estimated"] < rw["q_estimated"] * 0.1
    assert out["metrics"]["model_vs_rw_rmse_improvement_pct"] > 45.0


def test_nonlinear_force_prior_matches_jan_euler_truth_and_uses_no_measurements():
    p = ModelParameters()
    s_nl = generate(dt=0.01, N=950, p=p, plant="jan_euler", seed=7961, scenario="validation", true_q=1e-4)
    pr_lin = jan_linear_acceleration_prior(s_nl["u"], 0.01, p)
    pr_nl = jan_nonlinear_acceleration_prior(s_nl["u"], 0.01, p)
    ev_lin = evaluate_prior_accuracy(pr_lin, s_nl["acc_truth"])
    ev_nl = evaluate_prior_accuracy(pr_nl, s_nl["acc_truth"])
    assert ev_nl["prior_rmse_ms2"] < 0.007
    assert ev_nl["prior_r2_score"] > 0.94
    assert ev_nl["prior_rmse_ms2"] < ev_lin["prior_rmse_ms2"] * 0.2


def test_three_way_benchmark_on_nonlinear_plant_proves_nonlinear_prior_superiority():
    out = run_acceleration_benchmark(
        plant="jan_euler", dt=0.01, n_train=950, n_validation=950,
        q_true=1e-4, sensor_std_a=0.04, seed=42, prior_mode="nonlinear"
    )
    abl = out["ablation"]
    rw = abl["random_walk_kf"]
    lin = abl["linear_prior_kf"]
    nl = abl["nonlinear_prior_kf"]

    # 1) Nonlinear prior KF achieves > 45% lower RMSE than BOTH Ordinary RW-KF and Linear-Prior KF
    assert nl["rmse_ms2"] < rw["rmse_ms2"] * 0.55
    assert nl["rmse_ms2"] < lin["rmse_ms2"] * 0.58
    # 2) Nonlinear prior eliminates the >20x Q inflation caused by linear friction mismatch
    assert nl["q_estimated"] < lin["q_estimated"] * 0.05
    # 3) Nonlinear prior restores NEES calibration toward 1.0 (from 0.559 up to ~0.85) and 95% coverage (~97%)
    assert abs(nl["mean_nees"] - 1.0) < abs(lin["mean_nees"] - 1.0)
    assert 0.93 <= nl["coverage_95pct"] <= 0.985
    assert nl["lag1_white"] and nl["ljung_box_pvalue"] > 0.05


@pytest.mark.parametrize("prior_mode", [0, 1, 2])
def test_python_to_twincat_st_cycle_by_cycle_equivalence(prior_mode):
    p = ModelParameters()
    s = generate(dt=0.01, N=400, p=p, plant="jan_euler", seed=7961, scenario="validation", true_q=1e-4)
    st_text = generate_twincat_st_code(default_prior_mode=prior_mode)
    assert "FUNCTION_BLOCK FB_AkusNonlinearAccKF" in st_text
    assert "fQstep_ms4      := fProcessQ * fSampleDt;" in st_text
    eq = verify_python_to_twincat_equivalence(
        s["u"], s["y"][:, 1], dt=0.01, r=1.5884e-3, q=5.1957e-5, prior_mode=prior_mode, p=p
    )
    assert eq["is_bit_exact_lreal"]
    assert eq["max_abs_diff_acc_est_ms2"] < 1e-14
    assert eq["max_abs_diff_K"] < 1e-15
    assert eq["max_abs_diff_S_ms4"] < 1e-15



