# Acceleration KF (added after Control Studio v3)

Run `start_windows.bat`, open the **Acceleration KF** tab. This does **not** command hardware.

## What the new module does

- File: `akuskf/acceleration.py` (1-state **acceleration-first** estimator).
- Dashboard: `app.py` -> **Acceleration KF** tab.
- **Synthetic Jan linear** and **Synthetic Jan nonlinear Euler**: independent training and held-out runs; acceleration truth available only for checking RMSE. Calibration variance is estimated from separate synthetic stationary noise.
- **Real Earth hardware**: replay included stationary and moving excerpts or upload full `Test_KalmanFilter.csv`. The stationary signal establishes a local +1g baseline; the signal is converted to a dynamic acceleration in m/s². This is *not* a complete offset/scale calibration or free-fall measurement. Export acceleration traces and parameter report.
- Uses scalar process model `d[k+1]=d[k]+w[k]`, `a_obs[k]=a_prior[k]+d[k]+n[k]`. `a_prior` can be omitted (measurement-only) or calculated from deterministic linear Jan force equations (the latter is experimental for Earth data because motor scaling/sign remain unverified).
- Stationary measurement noise variance **R** estimated from stationary data **at the actual chosen estimator sample rate**, after resampling.
- Fit scalar continuous process intensity `q` with SciPy likelihood on training data; `Q_step=q*dt`, and **S, K** are calculated every update with a Joseph covariance update.
- Data export includes calibrated dynamic acceleration measurements, estimated acceleration, model prior, innovations, NIS, gains K and innovation covariance S.
- Prior modes and source type are clearly labeled. Source's original position and velocity Kalman experiment in the other tabs is unchanged.

## What this cannot prove

1. The 1-state acceleration Q is **not** the 2×2 Q of Jan's existing [position, velocity] KF. Do not transfer values between these models.
2. Scalar acceleration random walk is a deliberately **simple baseline**; it may attenuate real acceleration/vibration. Very smooth traces are not automatically accurate.
3. The real Earth Scope filtered channel is only a comparison, not ground truth.
4. Stationary sample R is a preliminary value (bias, correlation, vibration and sample-rate dependence matter). A low-pass or downsampled covariance is not identical to raw-sensor covariance.
5. For Earth experiments, a single recording split chronologically is not the same as an independent recording. A robust holdout uses separate runs.
6. In the model-guided prior, torque calibration and coordinate signs must be confirmed before interpreting residuals physically.
7. Kalman estimates are not guaranteed to resolve 10^-6 g in Einstein Elevator. This release is an *offline simulation/replay research tool*.

## Run tests

```powershell
py -m pip install -r requirements.txt
py -m pytest -q tests/test_acceleration_module.py
py -m streamlit run app.py
```
