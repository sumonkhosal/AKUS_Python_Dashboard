# AKUS Control Studio v4 — Acceleration Kalman Filter Engineering Guide & Benchmark Report

> **Safety & Scope Notice:** Offline simulation and estimator research tool only. No physical Beckhoff TwinCAT PLC, EtherCAT drive, safety I/O, or Einstein Elevator hardware is commanded by this software.

---

## 1. Research Objective & Model Separation

This project maintains three distinct estimator research tabs in `app.py` (and `index.html`) alongside the original 8 operator tabs:

1. **`Kalman research` (Tab 8):** Jan Hoffmann's 2-state `[position z, velocity v]` damped linear physics estimator (`dim_x = 2`, `dim_z = 2`).
2. **`Simulation Q/R benchmark` (Tab 9):** Controlled train/validation benchmark for the 2-state `[z, v]` estimator comparing the Matched Linear plant against Jan's Nonlinear Euler plant (`nominal NIS/NEES mean ≈ 2.0`).
3. **`Acceleration KF` (Tab 10 — Primary Research Focus):** Dedicated 1-state acceleration-residual Kalman filter (`dim_x = 1`, `dim_z = 1`, `nominal NIS/NEES mean ≈ 1.0`) implemented in [`akuskf/acceleration.py`](akuskf/acceleration.py).

---

## 2. Mathematical Formulation (1-State Acceleration KF)

### Stochastic Residual State Model
For sampling interval $\Delta t > 0$:

$$d_{k+1} = d_k + w_k, \qquad w_k \sim \mathcal{N}(0,\, Q_{\text{step}}), \qquad Q_{\text{step}} = q \cdot \Delta t$$

$$y_k = a_{\text{prior},k} + d_k + v_k, \qquad v_k \sim \mathcal{N}(0,\, R)$$

$$\hat{a}_{k|k} = a_{\text{prior},k} + \hat{d}_{k|k}$$

- **Random-Walk Baseline (`a_prior = 0`):** $d_k \equiv a_k$ directly estimates dynamic acceleration without relying on uncalibrated motor-torque-to-force scaling.
- **Model-Informed Prior (`Jan linear physics force prediction`):** $a_{\text{prior},k} = (u_k - b_v \hat{v}_k) / (m_p + m_{\text{eq}})$ is predicted deterministically from Jan's linear equations of motion using net drive force $u_k$ (with zero access to simulated ground truth or accelerometer measurements), and $d_k$ tracks the stochastic/unmodeled acceleration residual.

### Prediction & Update Equations (FilterPy & Joseph-Form Reference)
At every sample $k$:

$$P_{k|k-1} = P_{k-1|k-1} + Q_{\text{step}}$$

$$S_k = P_{k|k-1} + R, \qquad K_k = \frac{P_{k|k-1}}{S_k}$$

$$\nu_k = (y_k - a_{\text{prior},k}) - \hat{d}_{k|k-1}, \qquad \hat{d}_{k|k} = \hat{d}_{k|k-1} + K_k \nu_k$$

$$P_{k|k} = (1 - K_k)^2 P_{k|k-1} + K_k^2 R \quad \text{(Joseph stabilized form)}$$

$$\text{NIS}_k = \frac{\nu_k^2}{S_k} \sim \chi^2_1, \qquad \text{NEES}_k = \frac{(\hat{a}_{k|k} - a_{\text{true},k})^2}{P_{k|k}} \sim \chi^2_1$$

Both `backend='filterpy'` (`filterpy.kalman.KalmanFilter(dim_x=1, dim_z=1)`) and `backend='reference'` are implemented and verified to agree to $< 10^{-15}\text{ m/s}^2$.

---

## 3. Controlled Acceleration Q/R Identification Benchmark Results

Generated reproducibly via `run_acceleration_benchmark` (`seed = 42` for training, `seed = 7961` for held-out validation, $\Delta t = 10\text{ ms}$, $N = 950$, saved in [`results/acceleration_benchmark/`](results/acceleration_benchmark/)):

| Metric (1-State Acceleration KF, `dim = 1`) | Matched Linear Plant | Jan Nonlinear Euler Plant |
|---|---:|---:|
| Injected sensor noise variance $R_{\text{true}}$ $[(\text{m/s}^2)^2]$ | `1.600e-3` ($\sigma_v = 0.04\text{ m/s}^2$) | `1.600e-3` ($\sigma_v = 0.04\text{ m/s}^2$) |
| Estimated sensor noise variance $\hat{R}$ $[(\text{m/s}^2)^2]$ | `1.637e-3` (`+2.3%` error) | `1.637e-3` (`+2.3%` error) |
| Injected process spectral density $q_{\text{true}}$ $[(\text{m/s}^2)^2/\text{s}]$ | `1.000e-3` | `1.000e-3` |
| Identified process spectral density $\hat{q}$ $[(\text{m/s}^2)^2/\text{s}]$ | `7.811e-4` ($0.78\times$) | `2.243e-3` (**$2.24\times$ inflation**) |
| Held-out Raw Acceleration RMSE $[\text{m/s}^2]$ | `0.0401` | `0.0401` |
| Held-out Kalman Acceleration RMSE $[\text{m/s}^2]$ | **`0.0121` (`-69.7%`)** | **`0.0131` (`-67.4%`)** |
| Held-out Raw $\rightarrow$ Kalman MAE $[\text{m/s}^2]$ | `0.0318` $\rightarrow$ **`0.0100`** | `0.0318` $\rightarrow$ **`0.0104`** |
| Held-out 1D Mean NIS (nominal $\approx 1.0$, 95% CI `[0.911, 1.093]`) | **`0.991`** | **`0.970`** |
| FilterPy vs. Reference max $\|\Delta \hat{a}\|$ $[\text{m/s}^2]$ | `5.55e-17` | `2.78e-17` |

### Comparison with Historical 2-State Position/Velocity Benchmark (`dim = 2`)
Do not confuse the 1-state Acceleration KF metrics above (`expected NIS/NEES ≈ 1.0`) with the previous 2-state `[position, velocity]` benchmark (`expected NIS/NEES ≈ 2.0`):

| Historical 2-State Benchmark Metric (`dim = 2`) | Matched Linear | Jan Nonlinear |
|---|---:|---:|
| Injected process intensity $q_a$ $[\text{m}^2/\text{s}^3]$ | `1.00e-4` | `1.00e-4` |
| Identified process intensity $\hat{q}_a$ $[\text{m}^2/\text{s}^3]$ | `6.70e-5` | `5.13e-4` |
| Held-out 2D Mean NIS (nominal $\approx 2.0$) | `2.107` | `3.648` |
| Nonlinear Raw $\rightarrow$ Filtered Position RMSE | — | `0.351 mm` $\rightarrow$ `0.274 mm` |

---

## 4. Verification Summary: What Is Proven vs. Remaining Uncertainty

### Verified in this release
1. **Unbiased $R$ recovery:** Stationary/residual sample variance reliably recovers injected sensor variance $R$ across seeds (`< 8%` error for $N \ge 950$).
2. **Data-driven $Q$ identification without leakage:** Both innovation MLE and RTS-smoother EM identify $q > 0$ on training data only and reduce held-out acceleration RMSE by `~67–70%` when sensor noise dominates process drift.
3. **1D Statistical consistency:** Held-out 1D NIS and NEES remain centered near `1.0` for matched linear dynamics.
4. **Nonlinear friction mismatch detection:** Under Jan's nonlinear Coulomb + position-dependent friction (`FB_AkusModel`), the linear force prior exhibits structured mismatch at velocity reversals/deadzones, causing MLE to inflate $\hat{q}$ (`2.24×` in 1D, `5.13×` in 2D).
5. **Real Earth-hardware replay:** Bundled `akus_stationary_REAL.csv` (2–10 s) reproduces the ~`0.00208 m/s²` decimated stationary standard deviation after $+1\text{ g}$ baseline removal (`g0 ≈ 0.99989 g`).

### Remaining physical & experimental uncertainties
1. **Earth moving data lacks independent acceleration ground truth:** On real hardware (`akus_movement_REAL.csv`), `fAccZ` (TwinCAT 10 Hz low-pass) is a comparison signal, not ground truth.
2. **Mechanical vibration vs. electronic sensor noise:** Moving spindle operation excites structural vibrations that violate white-noise assumptions (`|lag-1 autocorrelation| > 0.2` at high sample rates).
3. **Uncalibrated motor torque polarity & scaling:** Real `fTorque1`/`fTorque2` channels require physical bench calibration before trusting model-informed force priors on hardware.

---

## 5. Windows Launch & Test Instructions

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run the full 36-test automated verification suite
python -m pytest -v

# 3. Launch the Streamlit dashboard
python -m streamlit run app.py

# Or launch the native Windows desktop GUI (no Streamlit required):
python desktop_app.py
```
