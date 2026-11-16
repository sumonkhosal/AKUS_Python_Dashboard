"""Offline, acceleration-FIRST Kalman research for AKUS (v4 upgraded).

This is a *separate* one-state acceleration/residual estimator (dim_x=1, dim_z=1).
It does NOT pretend its fitted Q is Jan's two-state [position, velocity] process covariance.
For this 1D estimator, nominal expected mean NIS and NEES under a matched Gaussian model
are approximately 1.0 (chi-square with 1 degree of freedom per sample), NOT 2.0.

Stochastic state model (random-walk residual around deterministic prior):
    d[k+1] = d[k] + w[k],      w[k] ~ N(0, Q_step),   Q_step = q * dt
    y[k]   = a_prior[k] + d[k] + v[k],  v[k] ~ N(0, R)
    a_est[k] = a_prior[k] + d_est[k]

When a_prior[k] = 0 (measurement-only baseline), d[k] is the acceleration state a[k] itself:
    a[k+1] = a[k] + w[k]
    y[k]   = a[k] + v[k]
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import welch
from scipy.stats import chi2

from .jan_model import matrices, ModelParameters
from .synthetic import generate as generate_jan_synthetic

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
    """One-state acceleration/residual Kalman filter supporting FilterPy and Reference backends.

    Parameters
    ----------
    measured_ms2 : array_like
        Measured dynamic acceleration [m/s²].
    dt : float
        Sampling interval [s].
    r : float
        Measurement noise variance R [(m/s²)²] at this sampling rate.
    q : float
        Continuous-time process spectral density [(m/s²)²/s], so Q_step = q * dt.
    prior_ms2 : array_like, optional
        Deterministic model-predicted acceleration prior [m/s²]. Defaults to 0.
    initial_residual : float
        Initial state estimate d[0] [m/s²].
    initial_variance : float, optional
        Initial state variance P[0] [(m/s²)²]. Defaults to max(r, 1e-12).
    backend : {'reference', 'filterpy'}
        Filtering implementation. Both compute identical S, K, estimates, and Joseph P.
    """
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
    """Estimate process spectral density q [(m/s²)²/s] on TRAINING data only.

    Supports:
    - 'mle': Constrained scalar Gaussian innovation maximum likelihood via SciPy bounded search.
    - 'em': Expectation-Maximization (RTS smoother + M-step) initialized from MLE.
    """
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
        # Refine with scalar RTS smoother EM iterations
        base = np.zeros_like(y) if prior_ms2 is None else np.asarray(prior_ms2, dtype=float).reshape(-1)
        z = y - base
        n = len(z)
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
    """Estimate acceleration measurement variance R [(m/s²)²] using unbiased sample variance (ddof=1).

    For compatibility with existing callers, returns `(r, corr)`. Use `analyze_acceleration_noise`
    for full histogram, autocorrelation function, PSD, and relative error diagnostics.
    """
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
    """Comprehensive sensor noise / residual characterization (Section 6).

    Computes unbiased variance R, standard deviation, relative error against injected R,
    autocorrelation (lags 0..15), histogram, and Welch power spectral density.
    """
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

    Inputs must be gravity-compensated net force u [N]. The result is NOT a
    reconstruction of Jan's nonlinear friction, and has no sensor/truth access.
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


def acceleration_diagnostics(filtered, truth=None, burn: int = 20, raw_measured=None):
    """Held-out quality metrics for the 1D acceleration estimator (Section 8 & 9).

    For the scalar acceleration filter (dim_z = 1, dim_x = 1):
    - Nominal expected mean NIS is 1.0.
    - Nominal expected mean NEES is 1.0 (when truth is known).
    """
    k = min(int(burn), len(filtered["nis"]) // 5)
    v = filtered["innovation"][k:]
    nis = filtered["nis"][k:]
    n_val = len(v)
    ci_95 = _mean_chi2_1d_interval(n_val)
    out = {
        "mean_nis": float(np.mean(nis)),
        "nis_95pct_interval_1d": ci_95,
        "innovation_mean_ms2": float(np.mean(v)),
        "innovation_std_ms2": float(np.std(v, ddof=1)) if n_val > 1 else 0.0,
        "innovation_lag1": (
            float(np.corrcoef(v[:-1], v[1:])[0, 1]) if n_val > 3 and np.std(v) > 0 else float("nan")
        ),
        "n_validation_samples": int(n_val),
    }
    if truth is not None:
        truth_arr = np.asarray(truth, float).reshape(-1)
        est_arr = filtered["estimated_acc_ms2"]
        if truth_arr.shape != est_arr.shape:
            raise ValueError("Truth length mismatch")
        err = est_arr[k:] - truth_arr[k:]
        P_val = np.maximum(filtered["P"][k:], 1e-18)
        nees = (err ** 2) / P_val
        out["kalman_acc_rmse_ms2"] = float(np.sqrt(np.mean(err ** 2)))
        out["kalman_acc_mae_ms2"] = float(np.mean(np.abs(err)))
        out["mean_nees"] = float(np.mean(nees))
        out["nees_95pct_interval_1d"] = ci_95
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
    q_true: float = 0.001,
    sensor_std_a: float = 0.04,
    seed: int = 42,
    use_prior: bool = True,
    backend: str = "filterpy",
    fit_q: bool = True,
    q_manual: float = 0.001,
    model: ModelParameters | None = None,
) -> dict:
    """Controlled 1-state Acceleration Q/R Identification Benchmark (Section 9).

    Executes the 10-step benchmark protocol:
    1. Generates training data from `plant` ('linear', 'jan_euler', or 'random_walk').
    2. Injects known process disturbance intensity `q_true` and sensor noise `sensor_std_a`.
    3. Estimates R from measurement residuals (measured - true on calibration/training noise).
    4. Estimates Q on training data ONLY via innovation maximum likelihood.
    5. Runs FilterPy (and Reference backend for equivalence verification).
    6. Generates independent validation data with `seed + 7919` and distinct excitation.
    7. Runs the filter on held-out validation data without refitting.
    8. Compares raw acceleration, Kalman-estimated acceleration, and true acceleration.
    9. Computes RMSE, MAE, 1D NIS (expected ~1.0), and 1D NEES (expected ~1.0).
    10. Produces an automated engineering interpretation of the results.
    """
    if model is None:
        model = ModelParameters()
    model.validate()
    if dt <= 0 or n_train < 100 or n_validation < 100 or q_true < 0 or sensor_std_a <= 0:
        raise ValueError("Invalid benchmark parameters")

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
        prior_train = np.zeros(n_train, dtype=float)
        prior_val = np.zeros(n_validation, dtype=float)
        noise_train = rng_train.normal(0.0, sensor_std_a, size=n_train)
        noise_val = rng_val.normal(0.0, sensor_std_a, size=n_validation)
        y_train = a_truth_train + noise_train
        y_val = a_truth_val + noise_val
        cal_residuals = y_train - a_truth_train
    else:
        sim_train = generate_jan_synthetic(
            dt=dt, N=int(n_train), p=model, seed=int(seed),
            sensor_std_a=float(sensor_std_a), true_q=1e-6, plant=plant, scenario="training"
        )
        sim_val = generate_jan_synthetic(
            dt=dt, N=int(n_validation), p=model, seed=int(seed) + 7919,
            sensor_std_a=float(sensor_std_a), true_q=1e-6, plant=plant, scenario="validation"
        )
        t_train = sim_train["t"]
        t_val = sim_val["t"]
        # Inject explicit 1-state acceleration process disturbance d[k+1] = d[k] + w[k]
        d_train = np.cumsum(rng_train.normal(0.0, np.sqrt(Q_step_injected), size=n_train))
        d_val = np.cumsum(rng_val.normal(0.0, np.sqrt(Q_step_injected), size=n_validation))
        a_truth_train = sim_train["acc_truth"] + d_train
        a_truth_val = sim_val["acc_truth"] + d_val
        noise_train = rng_train.normal(0.0, sensor_std_a, size=n_train)
        noise_val = rng_val.normal(0.0, sensor_std_a, size=n_validation)
        y_train = a_truth_train + noise_train
        y_val = a_truth_val + noise_val
        cal_residuals = y_train - a_truth_train
        if use_prior:
            prior_train = jan_linear_acceleration_prior(sim_train["u"], dt, model)
            prior_val = jan_linear_acceleration_prior(sim_val["u"], dt, model)
        else:
            prior_train = np.zeros(n_train, dtype=float)
            prior_val = np.zeros(n_validation, dtype=float)

    noise_report = analyze_acceleration_noise(cal_residuals, dt=dt, r_injected=r_injected)
    r_est = noise_report["r_estimated"]

    if fit_q:
        q_est, fit_info = fit_acceleration_q(y_train, dt, r_est, prior_ms2=prior_train)
    else:
        q_est = float(q_manual)
        fit_info = {"converged": True, "near_bound": False, "method": "manual"}

    val_res = run_acceleration_filter(
        y_val, dt, r_est, q_est, prior_ms2=prior_val, backend=actual_backend
    )
    ref_res = run_acceleration_filter(
        y_val, dt, r_est, q_est, prior_ms2=prior_val, backend="reference"
    )
    backend_max_diff = float(np.max(np.abs(val_res["estimated_acc_ms2"] - ref_res["estimated_acc_ms2"])))

    metrics = acceleration_diagnostics(val_res, truth=a_truth_val, burn=20, raw_measured=y_val)
    metrics["r_injected"] = r_injected
    metrics["r_estimated"] = r_est
    metrics["r_relative_error"] = noise_report["relative_error"]
    metrics["q_injected"] = float(q_true)
    metrics["q_estimated"] = float(q_est)
    metrics["q_ratio"] = float(q_est / q_true) if q_true > 0 else None
    metrics["Q_step_injected"] = Q_step_injected
    metrics["Q_step_estimated"] = float(val_res["Q_step"])
    metrics["backend_max_abs_diff"] = backend_max_diff

    # Step 10: Automated engineering interpretation
    interpretation = []
    if abs(noise_report["relative_error"] or 0.0) < 0.15:
        interpretation.append(
            f"R recovery: Estimated R ({r_est:.4e} (m/s²)²) closely matches injected R ({r_injected:.4e} (m/s²)²), "
            f"relative error = {(noise_report['relative_error'] or 0.0)*100:+.1f}%."
        )
    else:
        interpretation.append(
            f"R recovery: Estimated R ({r_est:.4e}) differs from injected R ({r_injected:.4e}) by "
            f"{(noise_report['relative_error'] or 0.0)*100:+.1f}%."
        )

    if metrics["kalman_acc_rmse_ms2"] < metrics["raw_acc_rmse_ms2"]:
        imp = (1.0 - metrics["kalman_acc_rmse_ms2"] / metrics["raw_acc_rmse_ms2"]) * 100.0
        interpretation.append(
            f"Accuracy improvement: Held-out RMSE improved by {imp:.1f}% "
            f"(Raw {metrics['raw_acc_rmse_ms2']:.4f} m/s² → Kalman {metrics['kalman_acc_rmse_ms2']:.4f} m/s²; "
            f"MAE {metrics['raw_acc_mae_ms2']:.4f} → {metrics['kalman_acc_mae_ms2']:.4f} m/s²)."
        )

    nis_lo, nis_hi = metrics["nis_95pct_interval_1d"]
    if nis_lo <= metrics["mean_nis"] <= nis_hi:
        interpretation.append(
            f"1D Innovation Consistency: Held-out mean NIS = {metrics['mean_nis']:.3f} lies within the 95% chi²(1) "
            f"interval [{nis_lo:.3f}, {nis_hi:.3f}] (expected mean ≈ 1.0)."
        )
    else:
        interpretation.append(
            f"1D Innovation Warning: Held-out mean NIS = {metrics['mean_nis']:.3f} is outside the 95% chi²(1) "
            f"interval [{nis_lo:.3f}, {nis_hi:.3f}], indicating model mismatch or unmodeled dynamics."
        )

    if plant == "jan_euler" and use_prior:
        interpretation.append(
            "Nonlinear Friction Mismatch Detected: Because the Jan nonlinear Euler plant includes Coulomb/position "
            "friction deadzones omitted from the linear force prior, the maximum-likelihood optimizer inflates Q "
            f"(q_est / q_true = {metrics['q_ratio']:.2f}×) to absorb deterministic friction errors."
        )
    elif not use_prior and plant in ("linear", "jan_euler"):
        interpretation.append(
            "Random-Walk Baseline Note: With no physics prior (a_prior = 0), Q absorbs both stochastic process "
            "disturbances and deterministic carrier acceleration changes."
        )

    return {
        "plant": plant,
        "dt": dt,
        "use_prior": use_prior,
        "backend": actual_backend,
        "t_val": t_val,
        "y_val": y_val,
        "truth_val": a_truth_val,
        "prior_val": prior_val,
        "filter_result": val_res,
        "noise_report": noise_report,
        "fit_info": fit_info,
        "metrics": metrics,
        "interpretation": interpretation,
    }
