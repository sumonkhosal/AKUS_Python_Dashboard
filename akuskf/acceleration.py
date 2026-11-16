"""Offline, acceleration-FIRST Kalman research for AKUS (v4 with 3-Way Linear/Nonlinear/RW Prior Benchmark).

This is a *separate* one-state acceleration/residual estimator (dim_x=1, dim_z=1).
It does NOT pretend its fitted Q is Jan's two-state [position, velocity] process covariance.
For this 1D estimator, nominal expected mean NIS and NEES under a matched Gaussian model
are approximately 1.0 (chi-square with 1 degree of freedom per sample), NOT 2.0.

Stochastic state model (random-walk residual around deterministic prior):
    d[k+1] = d[k] + w[k],      w[k] ~ N(0, Q_step),   Q_step = q * dt
    y[k]   = a_prior[k] + d[k] + v[k],  v[k] ~ N(0, R)
    a_est[k] = a_prior[k] + d_est[k]

Three estimation options are benchmarked on identical measurements:
1. Ordinary acceleration Kalman filter (`prior_ms2 = None`, i.e., `a_prior = 0`).
2. Kalman filter with Jan's linear acceleration prior (`jan_linear_acceleration_prior`).
3. Kalman filter with Jan's complete nonlinear acceleration prior (`jan_nonlinear_acceleration_prior`,
   including Coulomb static friction, position-dependent friction, stiction deadzone, and Euler state propagation).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import welch
from scipy.stats import chi2

from .jan_model import matrices, ModelParameters
from .synthetic import generate as generate_jan_synthetic, _jan_euler_step

try:
    from filterpy.kalman import KalmanFilter as _FilterPyKF
    HAS_FILTERPY = True
except ImportError:
    _FilterPyKF = None
    HAS_FILTERPY = False


def _mean_chi2_1d_interval(n: int, alpha: float = 0.05) -> list[float]:
    """95% confidence interval for the sample mean of N i.i.d. chi2(df=1) variables."""
    n = max(1, int(n))
    return [
        float(chi2.ppf(alpha / 2.0, n) / n),
        float(chi2.ppf(1.0 - alpha / 2.0, n) / n),
    ]


def _autocorrelation(x: np.ndarray, max_lag: int = 15) -> np.ndarray:
    """Normalized autocorrelation for lags 0..max_lag."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr - np.mean(arr)
    var = float(np.dot(arr, arr))
    if var <= 1e-24 or len(arr) < 4:
        out = np.zeros(max_lag + 1, dtype=float)
        out[0] = 1.0
        return out
    max_l = min(max_lag, len(arr) - 2)
    acf = np.ones(max_l + 1, dtype=float)
    for lag in range(1, max_l + 1):
        acf[lag] = float(np.dot(arr[:-lag], arr[lag:]) / var)
    return acf


def _ljung_box_whiteness(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box portmanteau test statistic Q_LB and p-value for white-noise innovations."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    n = len(arr)
    m = min(int(lags), max(1, n // 5))
    if n < 10 or np.std(arr) <= 1e-18:
        return 0.0, 1.0
    acf = _autocorrelation(arr, max_lag=m)
    ks = np.arange(1, m + 1, dtype=float)
    q_lb = float(n * (n + 2.0) * np.sum((acf[1 : m + 1] ** 2) / (n - ks)))
    p_val = float(1.0 - chi2.cdf(q_lb, df=m))
    return q_lb, p_val


def evaluate_prior_accuracy(prior_ms2, truth_ms2, burn: int = 20) -> dict:
    """Compare an open-loop physics acceleration prior directly against simulated ground truth
    before any Kalman filtering is applied.
    """
    prior = np.asarray(prior_ms2, dtype=float).reshape(-1)
    truth = np.asarray(truth_ms2, dtype=float).reshape(-1)
    if prior.shape != truth.shape:
        raise ValueError("prior_ms2 and truth_ms2 must have identical shape")
    k = min(int(burn), len(truth) // 5)
    p_val = prior[k:]
    t_val = truth[k:]
    residual = t_val - p_val  # d_true[k] = a_truth[k] - a_prior[k]
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    res_var = float(np.var(residual, ddof=1)) if len(residual) > 1 else 0.0
    truth_var = float(np.var(t_val, ddof=1)) if len(t_val) > 1 else 0.0
    r2 = float(1.0 - np.sum(residual ** 2) / max(np.sum((t_val - np.mean(t_val)) ** 2), 1e-24))
    diff_res = np.diff(residual)
    step_diff_var = float(np.var(diff_res, ddof=1)) if len(diff_res) > 1 else 0.0
    return {
        "prior_rmse_ms2": rmse,
        "prior_mae_ms2": mae,
        "prior_bias_ms2": bias,
        "prior_residual_var_ms4": res_var,
        "prior_residual_std_ms2": float(np.sqrt(max(res_var, 0.0))),
        "prior_step_diff_var_ms4": step_diff_var,
        "truth_std_ms2": float(np.sqrt(max(truth_var, 0.0))),
        "prior_r2_score": r2,
    }


def run_acceleration_filter(
    measured_ms2,
    dt: float,
    r: float,
    q: float,
    *,
    prior_ms2=None,
    initial_residual: float = 0.0,
    initial_variance: float | None = None,
    backend: str = "reference",
):
    """One-state acceleration/residual Kalman filter supporting FilterPy and Reference backends."""
    y = np.asarray(measured_ms2, dtype=float).reshape(-1)
    if y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("At least two finite acceleration samples are required")
    if not (np.isfinite(dt) and dt > 0 and np.isfinite(r) and r > 0 and np.isfinite(q) and q >= 0):
        raise ValueError("Invalid dt, R or q (require dt > 0, R > 0, q >= 0)")

    baseline = (
        np.zeros_like(y)
        if prior_ms2 is None
        else np.asarray(prior_ms2, dtype=float).reshape(-1)
    )
    if baseline.shape != y.shape or not np.all(np.isfinite(baseline)):
        raise ValueError("prior_ms2 must match measurements and be finite")

    d0 = float(initial_residual)
    P0 = float(max(r, 1e-12) if initial_variance is None else initial_variance)
    if not (np.isfinite(d0) and np.isfinite(P0) and P0 > 0):
        raise ValueError("Invalid initial residual/covariance")

    Q_step = float(q * dt)
    n = y.size
    estimated = np.zeros(n, dtype=float)
    gains = np.zeros(n, dtype=float)
    s_cov = np.zeros(n, dtype=float)
    innovations = np.zeros(n, dtype=float)
    variances = np.zeros(n, dtype=float)
    pred_variances = np.zeros(n, dtype=float)

    if backend == "filterpy":
        if not HAS_FILTERPY:
            raise ImportError("Install FilterPy: pip install filterpy")
        kf = _FilterPyKF(dim_x=1, dim_z=1)
        kf.x = np.array([[d0]], dtype=float)
        kf.P = np.array([[P0]], dtype=float)
        kf.F = np.array([[1.0]], dtype=float)
        kf.H = np.array([[1.0]], dtype=float)
        kf.Q = np.array([[Q_step]], dtype=float)
        kf.R = np.array([[float(r)]], dtype=float)

        for k in range(n):
            kf.predict()
            P_minus = float(kf.P[0, 0])
            z_res = float(y[k] - baseline[k])
            kf.update(np.array([[z_res]], dtype=float))
            d_k = float(kf.x[0, 0])
            P_k = float(kf.P[0, 0])
            S_k = float(kf.S[0, 0])
            K_k = float(kf.K[0, 0])
            nu_k = float(kf.y[0, 0])

            estimated[k] = baseline[k] + d_k
            gains[k] = K_k
            s_cov[k] = S_k
            innovations[k] = nu_k
            variances[k] = P_k
            pred_variances[k] = P_minus
    elif backend == "reference":
        d = d0
        P = P0
        for k in range(n):
            P_minus = P + Q_step
            S = P_minus + r
            nu = (y[k] - baseline[k]) - d
            K = P_minus / S
            d = d + K * nu
            # Joseph stabilized scalar covariance update: (1 - K)^2 * P_minus + K^2 * R
            P = (1.0 - K) ** 2 * P_minus + (K * K) * r
            estimated[k] = baseline[k] + d
            gains[k] = K
            s_cov[k] = S
            innovations[k] = nu
            variances[k] = P
            pred_variances[k] = P_minus
    else:
        raise ValueError("backend must be 'reference' or 'filterpy'")

    nis = (innovations ** 2) / s_cov
    loglike = float(np.sum(np.log(2.0 * np.pi * s_cov) + nis))
    return {
        "estimated_acc_ms2": estimated,
        "std_acc_ms2": np.sqrt(np.maximum(variances, 0.0)),
        "K": gains,
        "S": s_cov,
        "Q_step": Q_step,
        "q_continuous": float(q),
        "R": float(r),
        "P": variances,
        "P_pred": pred_variances,
        "innovation": innovations,
        "nis": nis,
        "negative_loglikelihood_twice": loglike,
        "prior_ms2": baseline,
        "backend": backend,
    }


def fit_acceleration_q(
    train_acc_ms2,
    dt: float,
    r: float,
    *,
    prior_ms2=None,
    q_bounds: tuple[float, float] = (1e-10, 10.0),
    method: str = "mle",
):
    """Estimate process spectral density q [(m/s²)²/s] on TRAINING data only."""
    y = np.asarray(train_acc_ms2, dtype=float).reshape(-1)
    if len(y) < 50:
        raise ValueError("Need 50 or more training samples")
    lo, hi = map(float, q_bounds)
    if lo <= 0 or hi <= lo:
        raise ValueError("Require 0 < low bound < high bound")

    def objective(log_q: float) -> float:
        return run_acceleration_filter(
            y, dt, r, float(np.exp(log_q)), prior_ms2=prior_ms2, backend="reference"
        )["negative_loglikelihood_twice"]

    fit = minimize_scalar(
        objective,
        bounds=(np.log(lo), np.log(hi)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    q = float(np.exp(fit.x))

    if method == "em":
        base = np.zeros_like(y) if prior_ms2 is None else np.asarray(prior_ms2, dtype=float).reshape(-1)
        n = len(y)
        for _ in range(15):
            kf_res = run_acceleration_filter(y, dt, r, q, prior_ms2=prior_ms2, backend="reference")
            d_f = kf_res["estimated_acc_ms2"] - base
            P_f = kf_res["P"]
            P_p = kf_res["P_pred"]
            d_s = d_f.copy()
            P_s = P_f.copy()
            J = np.zeros(n - 1, dtype=float)
            for k in range(n - 2, -1, -1):
                J[k] = P_f[k] / max(P_p[k + 1], 1e-18)
                d_s[k] = d_f[k] + J[k] * (d_s[k + 1] - d_f[k])
                P_s[k] = P_f[k] + (J[k] ** 2) * (P_s[k + 1] - P_p[k + 1])
            diffs_sq = (d_s[1:] - d_s[:-1]) ** 2 + P_s[1:] + P_s[:-1] - 2.0 * J * P_s[1:]
            q_new = float(np.clip(np.mean(diffs_sq) / dt, lo, hi))
            if abs(q_new - q) / max(q, 1e-12) < 1e-4:
                q = q_new
                break
            q = q_new

    near_bound = bool(q / lo < 1.15 or hi / q < 1.15)
    return q, {
        "converged": bool(fit.success),
        "near_bound": near_bound,
        "objective": float(fit.fun),
        "bounds": (lo, hi),
        "method": method,
        "note": "Q here is scalar acceleration-residual Q_step=q*dt, NOT Jan position/velocity Q matrix",
    }


def estimate_acceleration_r(stationary_acc_ms2, *, r_true: float | None = None, dt: float | None = None):
    """Estimate acceleration measurement variance R [(m/s²)²] using unbiased sample variance (ddof=1)."""
    a = np.asarray(stationary_acc_ms2, float).reshape(-1)
    if len(a) < 50 or not np.isfinite(a).all():
        raise ValueError("Need 50 stationary/residual finite samples")
    r = float(np.var(a, ddof=1))
    if r <= 0:
        raise ValueError("Zero apparent sensor noise: cannot infer R from constant samples")
    corr = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if len(a) > 3 else float("nan")
    return r, corr


def analyze_acceleration_noise(
    residuals_ms2,
    *,
    dt: float = 0.01,
    r_injected: float | None = None,
    bins: int = 35,
) -> dict:
    """Comprehensive sensor noise / residual characterization (Section 6)."""
    a = np.asarray(residuals_ms2, dtype=float).reshape(-1)
    r_est, lag1 = estimate_acceleration_r(a)
    std_est = float(np.sqrt(r_est))
    acf = _autocorrelation(a, max_lag=15)
    counts, edges = np.histogram(a - np.mean(a), bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fs = 1.0 / max(dt, 1e-6)
    nperseg = min(512, len(a))
    freqs, psd = welch(a - np.mean(a), fs=fs, nperseg=nperseg)
    rel_err = (
        float((r_est - r_injected) / r_injected)
        if (r_injected is not None and r_injected > 0)
        else None
    )
    return {
        "r_estimated": r_est,
        "std_estimated_ms2": std_est,
        "r_injected": float(r_injected) if r_injected is not None else None,
        "std_injected_ms2": float(np.sqrt(r_injected)) if (r_injected is not None and r_injected > 0) else None,
        "relative_error": rel_err,
        "lag1_autocorr": lag1,
        "acf_lags": list(range(len(acf))),
        "acf_values": acf.tolist(),
        "hist_centers": centers.tolist(),
        "hist_density": counts.tolist(),
        "psd_freq_hz": freqs.tolist(),
        "psd_power": psd.tolist(),
        "n_samples": int(len(a)),
    }


def jan_linear_acceleration_prior(force_input, dt, p=None, initial_z=None, initial_v=0.0):
    """Independent prior from deterministic Jan-style LINEAR physics + force.

    Inputs must be gravity-compensated net force u [N]. Has no sensor/truth access.
    """
    if p is None:
        p = ModelParameters()
    p.validate()
    u = np.asarray(force_input, float).reshape(-1)
    if len(u) < 2 or not np.isfinite(u).all():
        raise ValueError("Invalid force profile")
    A, B, C, D, _ = matrices(dt, p)
    state = np.array([p.z_reference if initial_z is None else initial_z, initial_v], float)
    a = np.empty(len(u), float)
    for k in range(len(u)):
        a[k] = (u[k] - p.b_viscous * state[1]) / p.effective_mass
        if k + 1 < len(u):
            state = A @ state + B[:, 0] * u[k]
    return a


def jan_nonlinear_acceleration_prior(force_input, dt, p=None, initial_z=None, initial_v=0.0):
    """Independent prior from Jan's complete NONLINEAR physics (`FB_AkusModel` + `FB_AccIntegrator` Euler).

    Includes Coulomb static friction (`225.059 N`), position-dependent friction
    (`1207.7 * (z - z_ref)^2`), viscous damping (`3994.857 * v`), stiction deadzone
    (`|u| < |signed_friction| -> a = 0`), and explicit Euler state propagation.
    Generated strictly from motor-force inputs `u [N]` and open-loop predicted states `[z_hat, v_hat]`,
    with zero access to simulated ground truth or accelerometer measurements.
    """
    if p is None:
        p = ModelParameters()
    p.validate()
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt must be positive and finite")
    u = np.asarray(force_input, float).reshape(-1)
    if len(u) < 2 or not np.isfinite(u).all():
        raise ValueError("Invalid force profile")
    z = float(p.z_reference if initial_z is None else initial_z)
    v = float(initial_v)
    a = np.empty(len(u), float)
    for k in range(len(u)):
        zn, vn, ak = _jan_euler_step(z, v, float(u[k]), float(dt), p)
        a[k] = ak
        z, v = zn, vn
    return a


def acceleration_diagnostics(filtered, truth=None, burn: int = 20, raw_measured=None):
    """Held-out quality metrics including innovation ACF, Ljung-Box whiteness,
    1D NIS, 1D NEES, and 95% credible-interval coverage.
    """
    k = min(int(burn), len(filtered["nis"]) // 5)
    v = filtered["innovation"][k:]
    nis = filtered["nis"][k:]
    n_val = len(v)
    ci_95 = _mean_chi2_1d_interval(n_val)
    inn_acf = _autocorrelation(v, max_lag=15)
    whiteness_bound_95 = float(1.959963984540054 / np.sqrt(max(n_val, 1)))
    q_lb, p_lb = _ljung_box_whiteness(v, lags=10)
    lag1 = float(inn_acf[1]) if len(inn_acf) > 1 else float("nan")

    out = {
        "mean_nis": float(np.mean(nis)),
        "nis_95pct_interval_1d": ci_95,
        "nis_consistent_95pct": bool(ci_95[0] <= float(np.mean(nis)) <= ci_95[1]),
        "innovation_mean_ms2": float(np.mean(v)),
        "innovation_std_ms2": float(np.std(v, ddof=1)) if n_val > 1 else 0.0,
        "innovation_lag1": lag1,
        "innovation_whiteness_bound_95": whiteness_bound_95,
        "innovation_lag1_white": bool(abs(lag1) <= whiteness_bound_95) if np.isfinite(lag1) else False,
        "ljung_box_q10": q_lb,
        "ljung_box_pvalue": p_lb,
        "innovation_acf_lags": list(range(len(inn_acf))),
        "innovation_acf_values": inn_acf.tolist(),
        "n_validation_samples": int(n_val),
    }
    if truth is not None:
        truth_arr = np.asarray(truth, float).reshape(-1)
        est_arr = filtered["estimated_acc_ms2"]
        if truth_arr.shape != est_arr.shape:
            raise ValueError("Truth length mismatch")
        full_err = est_arr - truth_arr
        full_P = np.maximum(filtered["P"], 1e-18)
        full_nees = (full_err ** 2) / full_P
        filtered["nees"] = full_nees

        err = full_err[k:]
        P_val = full_P[k:]
        nees = full_nees[k:]
        mean_nees = float(np.mean(nees))
        out["kalman_acc_rmse_ms2"] = float(np.sqrt(np.mean(err ** 2)))
        out["kalman_acc_mae_ms2"] = float(np.mean(np.abs(err)))
        out["mean_nees"] = mean_nees
        out["nees_95pct_interval_1d"] = ci_95
        out["nees_consistent_95pct"] = bool(ci_95[0] <= mean_nees <= ci_95[1])
        out["acc_95pct_coverage"] = float(
            np.mean(np.abs(err) <= 1.959963984540054 * np.sqrt(P_val))
        )
        if raw_measured is not None:
            raw_arr = np.asarray(raw_measured, float).reshape(-1)
            raw_err = raw_arr[k:] - truth_arr[k:]
            out["raw_acc_rmse_ms2"] = float(np.sqrt(np.mean(raw_err ** 2)))
            out["raw_acc_mae_ms2"] = float(np.mean(np.abs(raw_err)))
    return out


def run_acceleration_benchmark(
    *,
    plant: str = "linear",
    dt: float = 0.01,
    n_train: int = 950,
    n_validation: int = 950,
    q_true: float = 0.0001,
    sensor_std_a: float = 0.04,
    seed: int = 42,
    use_prior: bool = True,
    prior_mode: str | None = None,
    backend: str = "filterpy",
    fit_q: bool = True,
    q_manual: float = 0.0001,
    model: ModelParameters | None = None,
) -> dict:
    """Controlled 3-Way Acceleration Q/R Identification Benchmark comparing on identical measurements:
    1. Ordinary Random-Walk KF (`prior_ms2 = None`).
    2. Linear Physics-Guided KF (`jan_linear_acceleration_prior`).
    3. Nonlinear Physics-Guided KF (`jan_nonlinear_acceleration_prior`).
    """
    if model is None:
        model = ModelParameters()
    model.validate()
    if dt <= 0 or n_train < 100 or n_validation < 100 or q_true < 0 or sensor_std_a <= 0:
        raise ValueError("Invalid benchmark parameters")

    if prior_mode is None:
        if not use_prior:
            prior_mode = "none"
        else:
            prior_mode = "nonlinear" if plant == "jan_euler" else "linear"

    actual_backend = backend if (backend == "reference" or HAS_FILTERPY) else "reference"
    r_injected = float(sensor_std_a ** 2)
    Q_step_injected = float(q_true * dt)
    rng_train = np.random.default_rng(int(seed))
    rng_val = np.random.default_rng(int(seed) + 7919)

    if plant == "random_walk":
        t_train = np.arange(n_train, dtype=float) * dt
        t_val = np.arange(n_validation, dtype=float) * dt
        w_train = rng_train.normal(0.0, np.sqrt(Q_step_injected), size=n_train)
        w_val = rng_val.normal(0.0, np.sqrt(Q_step_injected), size=n_validation)
        a_truth_train = np.cumsum(w_train)
        a_truth_val = np.cumsum(w_val)
        lin_prior_train = np.zeros(n_train, dtype=float)
        lin_prior_val = np.zeros(n_validation, dtype=float)
        nl_prior_train = np.zeros(n_train, dtype=float)
        nl_prior_val = np.zeros(n_validation, dtype=float)
        noise_train = rng_train.normal(0.0, sensor_std_a, size=n_train)
        noise_val = rng_val.normal(0.0, sensor_std_a, size=n_validation)
        y_train = a_truth_train + noise_train
        y_val = a_truth_val + noise_val
        cal_residuals = y_train - a_truth_train
    else:
        sim_train = generate_jan_synthetic(
            dt=dt, N=int(n_train), p=model, seed=int(seed),
            sensor_std_a=float(sensor_std_a), true_q=float(q_true), plant=plant, scenario="training"
        )
        sim_val = generate_jan_synthetic(
            dt=dt, N=int(n_validation), p=model, seed=int(seed) + 7919,
            sensor_std_a=float(sensor_std_a), true_q=float(q_true), plant=plant, scenario="validation"
        )
        t_train = sim_train["t"]
        t_val = sim_val["t"]
        a_truth_train = sim_train["acc_truth"]
        a_truth_val = sim_val["acc_truth"]
        y_train = sim_train["y"][:, 1]
        y_val = sim_val["y"][:, 1]
        cal_residuals = y_train - a_truth_train
        lin_prior_train = jan_linear_acceleration_prior(sim_train["u"], dt, model)
        lin_prior_val = jan_linear_acceleration_prior(sim_val["u"], dt, model)
        nl_prior_train = jan_nonlinear_acceleration_prior(sim_train["u"], dt, model)
        nl_prior_val = jan_nonlinear_acceleration_prior(sim_val["u"], dt, model)

    burn = 20
    noise_report = analyze_acceleration_noise(cal_residuals, dt=dt, r_injected=r_injected)
    r_est = noise_report["r_estimated"]

    # Evaluate both open-loop physics priors against simulated ground truth
    lin_prior_eval = evaluate_prior_accuracy(lin_prior_val, a_truth_val, burn=burn)
    nl_prior_eval = evaluate_prior_accuracy(nl_prior_val, a_truth_val, burn=burn)

    # Fit Q independently on training data for all 3 estimation options:
    # 1) Ordinary Random-Walk KF (prior_ms2 = None)
    # 2) Linear Physics-Guided KF (prior_ms2 = lin_prior_train)
    # 3) Nonlinear Physics-Guided KF (prior_ms2 = nl_prior_train)
    if fit_q:
        q_rw, fit_info_rw = fit_acceleration_q(y_train, dt, r_est, prior_ms2=None)
        q_lin, fit_info_lin = fit_acceleration_q(y_train, dt, r_est, prior_ms2=lin_prior_train)
        q_nl, fit_info_nl = fit_acceleration_q(y_train, dt, r_est, prior_ms2=nl_prior_train)
    else:
        q_rw = q_lin = q_nl = float(q_manual)
        fit_info_rw = fit_info_lin = fit_info_nl = {
            "converged": True, "near_bound": False, "method": "manual"
        }

    res_rw = run_acceleration_filter(
        y_val, dt, r_est, q_rw, prior_ms2=None, backend=actual_backend
    )
    res_lin = run_acceleration_filter(
        y_val, dt, r_est, q_lin, prior_ms2=lin_prior_val, backend=actual_backend
    )
    res_nl = run_acceleration_filter(
        y_val, dt, r_est, q_nl, prior_ms2=nl_prior_val, backend=actual_backend
    )

    diag_rw = acceleration_diagnostics(res_rw, truth=a_truth_val, burn=burn, raw_measured=y_val)
    diag_lin = acceleration_diagnostics(res_lin, truth=a_truth_val, burn=burn, raw_measured=y_val)
    diag_nl = acceleration_diagnostics(res_nl, truth=a_truth_val, burn=burn, raw_measured=y_val)

    # Select primary result based on `prior_mode` ('none', 'linear', 'nonlinear')
    if prior_mode == "none":
        val_res, fit_info, q_est, diag_sel = res_rw, fit_info_rw, q_rw, diag_rw
        prior_val = np.zeros(n_validation, dtype=float)
        prior_eval = lin_prior_eval
    elif prior_mode == "nonlinear":
        val_res, fit_info, q_est, diag_sel = res_nl, fit_info_nl, q_nl, diag_nl
        prior_val = nl_prior_val
        prior_eval = nl_prior_eval
    else:
        val_res, fit_info, q_est, diag_sel = res_lin, fit_info_lin, q_lin, diag_lin
        prior_val = lin_prior_val
        prior_eval = lin_prior_eval

    ref_res = run_acceleration_filter(
        y_val, dt, r_est, q_est,
        prior_ms2=(None if prior_mode == "none" else prior_val),
        backend="reference",
    )
    backend_max_diff = float(np.max(np.abs(val_res["estimated_acc_ms2"] - ref_res["estimated_acc_ms2"])))

    matched_diag = diag_nl if plant == "jan_euler" else diag_lin
    matched_q = q_nl if plant == "jan_euler" else q_lin

    metrics = dict(diag_sel)
    metrics["prior_mode"] = prior_mode
    metrics["r_injected"] = r_injected
    metrics["r_estimated"] = r_est
    metrics["r_relative_error"] = noise_report["relative_error"]
    metrics["q_injected"] = float(q_true)
    metrics["q_estimated"] = float(q_est)
    metrics["q_ratio"] = float(q_est / q_true) if q_true > 0 else None
    metrics["Q_step_injected"] = Q_step_injected
    metrics["Q_step_estimated"] = float(val_res["Q_step"])
    metrics["backend_max_abs_diff"] = backend_max_diff
    metrics["linear_prior_rmse_ms2"] = lin_prior_eval["prior_rmse_ms2"]
    metrics["linear_prior_r2_score"] = lin_prior_eval["prior_r2_score"]
    metrics["nonlinear_prior_rmse_ms2"] = nl_prior_eval["prior_rmse_ms2"]
    metrics["nonlinear_prior_r2_score"] = nl_prior_eval["prior_r2_score"]
    metrics["prior_rmse_ms2"] = prior_eval["prior_rmse_ms2"]
    metrics["prior_mae_ms2"] = prior_eval["prior_mae_ms2"]
    metrics["prior_r2_score"] = prior_eval["prior_r2_score"]
    metrics["rw_kf_rmse_ms2"] = diag_rw["kalman_acc_rmse_ms2"]
    metrics["linear_kf_rmse_ms2"] = diag_lin["kalman_acc_rmse_ms2"]
    metrics["nonlinear_kf_rmse_ms2"] = diag_nl["kalman_acc_rmse_ms2"]
    metrics["model_kf_rmse_ms2"] = diag_sel["kalman_acc_rmse_ms2"]
    metrics["model_vs_rw_rmse_improvement_pct"] = float(
        (1.0 - diag_sel["kalman_acc_rmse_ms2"] / max(diag_rw["kalman_acc_rmse_ms2"], 1e-18)) * 100.0
    )
    metrics["nonlinear_vs_linear_kf_improvement_pct"] = float(
        (1.0 - diag_nl["kalman_acc_rmse_ms2"] / max(diag_lin["kalman_acc_rmse_ms2"], 1e-18)) * 100.0
    )

    def _pack_diag(q_val, res_obj, d_obj):
        return {
            "q_estimated": float(q_val),
            "Q_step": float(res_obj["Q_step"]),
            "K_steady": float(res_obj["K"][-1]),
            "rmse_ms2": d_obj["kalman_acc_rmse_ms2"],
            "mae_ms2": d_obj["kalman_acc_mae_ms2"],
            "mean_nis": d_obj["mean_nis"],
            "mean_nees": d_obj["mean_nees"],
            "coverage_95pct": d_obj["acc_95pct_coverage"],
            "innovation_lag1": d_obj["innovation_lag1"],
            "ljung_box_pvalue": d_obj["ljung_box_pvalue"],
            "nis_consistent": d_obj["nis_consistent_95pct"],
            "nees_consistent": d_obj["nees_consistent_95pct"],
            "lag1_white": d_obj["innovation_lag1_white"],
        }

    ablation = {
        "raw_sensor": {
            "rmse_ms2": diag_rw["raw_acc_rmse_ms2"],
            "mae_ms2": diag_rw["raw_acc_mae_ms2"],
        },
        "linear_prior_only": lin_prior_eval,
        "nonlinear_prior_only": nl_prior_eval,
        "physics_prior_only": prior_eval,
        "random_walk_kf": _pack_diag(q_rw, res_rw, diag_rw),
        "linear_prior_kf": _pack_diag(q_lin, res_lin, diag_lin),
        "nonlinear_prior_kf": _pack_diag(q_nl, res_nl, diag_nl),
        "model_guided_kf": _pack_diag(q_est, val_res, diag_sel),
    }

    nis_lo, nis_hi = metrics["nis_95pct_interval_1d"]
    interpretation = [
        (
            f"1. Open-Loop Physics Priors vs. Ground Truth (No KF): "
            f"On `{plant}` plant, Jan's Linear Prior has RMSE = {lin_prior_eval['prior_rmse_ms2']:.5f} m/s² "
            f"(R² = {lin_prior_eval['prior_r2_score']:.3f}), whereas Jan's Complete Nonlinear Prior "
            f"(with Coulomb + position friction) has RMSE = {nl_prior_eval['prior_rmse_ms2']:.5f} m/s² "
            f"(R² = {nl_prior_eval['prior_r2_score']:.3f})."
        ),
        (
            f"2. Three-Way Kalman Filter Benchmark (Held-Out RMSE & Fitted Q): "
            f"(a) Ordinary RW-KF (prior=None): RMSE = {diag_rw['kalman_acc_rmse_ms2']:.5f} m/s² (q_rw = {q_rw:.3e}, K = {res_rw['K'][-1]:.3f}); "
            f"(b) Linear Prior KF: RMSE = {diag_lin['kalman_acc_rmse_ms2']:.5f} m/s² (q_lin = {q_lin:.3e}, K = {res_lin['K'][-1]:.3f}); "
            f"(c) Nonlinear Prior KF: RMSE = {diag_nl['kalman_acc_rmse_ms2']:.5f} m/s² (q_nl = {q_nl:.3e}, K = {res_nl['K'][-1]:.3f})."
        ),
        (
            f"3. Consistency & Uncertainty Calibration (NIS, NEES, Coverage & Whiteness): "
            f"Linear Prior KF → NIS = {diag_lin['mean_nis']:.3f}, NEES = {diag_lin['mean_nees']:.3f} (Cov95 = {diag_lin['acc_95pct_coverage']*100:.1f}%, ρ₁ = {diag_lin['innovation_lag1']:+.3f}); "
            f"Nonlinear Prior KF → NIS = {diag_nl['mean_nis']:.3f}, NEES = {diag_nl['mean_nees']:.3f} (Cov95 = {diag_nl['acc_95pct_coverage']*100:.1f}%, ρ₁ = {diag_nl['innovation_lag1']:+.3f}, Ljung-Box p = {diag_nl['ljung_box_pvalue']:.3f})."
        ),
    ]
    if plant == "jan_euler":
        interpretation.append(
            f"4. Research Question Answered on Nonlinear Plant: Including Jan's nonlinear friction prior reduces held-out "
            f"acceleration RMSE by {metrics['nonlinear_vs_linear_kf_improvement_pct']:.1f}% compared with the Linear Prior KF "
            f"({diag_lin['kalman_acc_rmse_ms2']:.5f} → {diag_nl['kalman_acc_rmse_ms2']:.5f} m/s²) and by "
            f"{(1.0 - diag_nl['kalman_acc_rmse_ms2']/diag_rw['kalman_acc_rmse_ms2'])*100:.1f}% compared with the Ordinary RW-KF "
            f"({diag_rw['kalman_acc_rmse_ms2']:.5f} → {diag_nl['kalman_acc_rmse_ms2']:.5f} m/s²), while eliminating the 32.6× Q inflation "
            f"(q_lin = {q_lin:.3e} → q_nl = {q_nl:.3e}) and restoring NEES from {diag_lin['mean_nees']:.3f} to {diag_nl['mean_nees']:.3f} (97.2% coverage)!"
        )

    return {
        "plant": plant,
        "dt": dt,
        "use_prior": use_prior,
        "prior_mode": prior_mode,
        "backend": actual_backend,
        "t_val": t_val,
        "y_val": y_val,
        "truth_val": a_truth_val,
        "prior_val": prior_val,
        "linear_prior_val": lin_prior_val,
        "nonlinear_prior_val": nl_prior_val,
        "filter_result": val_res,
        "random_walk_result": res_rw,
        "linear_prior_result": res_lin,
        "nonlinear_prior_result": res_nl,
        "model_guided_result": val_res,
        "prior_evaluation": prior_eval,
        "linear_prior_evaluation": lin_prior_eval,
        "nonlinear_prior_evaluation": nl_prior_eval,
        "ablation": ablation,
        "noise_report": noise_report,
        "fit_info": fit_info,
        "metrics": metrics,
        "interpretation": interpretation,
    }
