# AKUS Phase 2 — Two-state Q/R/K identification and TwinCAT export

This is a **standalone add-on** for AKUS Python Control Studio V4. It does not replace or overwrite your newer Antigravity V4 modifications. The baseline `akuskf` code inside is copied from the attached V4 archive. **No direct PLC or ADS access.**

## What it does

- Uses Jan's corrected low-DOF **2-state** physical dynamics: `x = [position z (m), velocity v (m/s)]`, measurements `y = [encoder position z (m), gravity-corrected acceleration (m/s²)]`.
- Builds **A (2×2), B (2×1), C (2×2), D (2×1)** by exact sampling of Jan's damped linear equations.
- Constrains **Q (2×2)** to the physically meaningful positive-semidefinite continuous-acceleration-noise family, using the Van Loan matrix exponential, and fits its intensity `q_a` from training innovations by maximum likelihood.
- Estimates full **R (2×2)** from synchronized stationary position and acceleration noise, with an off-diagonal covariance (where data support it).
- Calculates full dynamic **K (2×2), P (2×2), S (2×2)** each filtering cycle with Joseph covariance update, not a hard-coded steady-state gain.
- Generates the editable **`FB_AKUS_TwoStateKF.st`** file, plus 950 rows of Python *reference* input/output vectors for later actual TwinCAT replay.
- Reads actual Earth-hardware recording **`Test_KalmanFilter.csv`** to separately compute empirical stationary sensor `R` at 100 Hz and export a 100 Hz motion excerpt.

## Run

Requires Python 3.10+, NumPy, SciPy (no FilterPy necessary for this reference backend).

```bash
pip install numpy scipy
python phase2_two_state.py --plant linear --dt 0.01 --real-csv "C:/path/to/Test_KalmanFilter.csv" --out results
python -m unittest discover -s tests -v
```

`--plant jan_euler` is an optional **deliberately mismatched nonlinear plant** benchmark: current exported two-state state-space equations remain linear; don't mistake the result for a full nonlinear EKF.

## Numerical example (10 ms, matched linear synthetic plant)

The script regenerates the numeric matrices and measured diagnostics into `results/two_state_simulation_report.json`. The output `FB_AKUS_TwoStateKF.st` contains the identified **simulation** `Q` and `R`; these are **not** a field calibration. Random seed, sample timing, model parameters, and validation excitation are documented in the JSON.

The `results/real_stationary_R_report.json` contains *real* stationary measurement-covariance estimates. It distinguishes empirical values from the **1-micron engineering floor** applied to position variance for a positive definite exploratory filter. The real sensor is oversampled at 2000 Hz and software-decimated to the target 100 Hz; real acceleration innovation may be temporally correlated. At rest, the encoder is quantized, so tiny observed stationary variance is **not** proof of micron/submicron absolute encoder accuracy.

**Critical:** real torque values, their conversion to spindle force and sign, accelerometer bias/scale, gravity convention, and friction/stiction have not been independently calibrated. The script intentionally does **not** claim it has reliably fitted the real physical `Q`. In simulation, injected independent sensor noise is known; in real data it is not. Never copy the synthetic 2-state `Q` and `R` to a real control loop without new validation.

## TwinCAT replay

1. Create a **new offline TwinCAT 3 PLC project**, add `FB_AKUS_TwoStateKF.st` as a function block. Avoid motor output mapping. The source is *generated* and has **not** been compiled inside TwinCAT here.
2. Run the PLC at **10 ms** (or regenerate with the exact configured sampling interval).
3. Enable `bEnable`, pulse `bReset` on the first record, pass `fPosMeas_m`, `fAccMeas_ms2`, `fNetForce_N` from each row of `twincat_2state_golden_vectors.csv`.
4. Log each scan's `fZEst_m`, `fVEst_ms`, `fK11`, `fK12`, `fK21`, `fK22`, S, P, innovations and NIS. Verify against all 950 reference rows.
5. Ensure `fNetForce_N` represents motor force **minus platform weight**, **before** linear viscous friction, and the encoder/gravity sign convention is identical to Python.
6. Check floating-point tolerances; declaring *bit-exact* output is unwarranted until actual PLC execution.

## Matrix dimensions

| Matrix | Shape | Role |
|---|---:|---|
| A | 2×2 | Position/velocity transition |
| B | 2×1 | Net-force input |
| C | 2×2 | Position and acceleration measurement relationships |
| D | 2×1 | Direct force-to-measured-acceleration term |
| Q | 2×2 | Prediction process covariance, **per sample** |
| R | 2×2 | Sensor measurement covariance, **per sample** |
| K | 2×2 | Kalman gain recalculated every scan |
| S | 2×2 | Innovation covariance |
| P | 2×2 | Posterior state covariance |

**Scope:** this Phase 2 is Jan's original **linear 2-state position/velocity** estimator, distinct from the previous **1-state acceleration-residual** filter. A full nonlinear state-estimation filter would need new Jacobians and dedicated validation.

## Nonlinear stress test

`results/nonlinear_plant_stress_test_report.json` repeats the estimation with Jan’s nonlinear Euler plant, while the baseline two-state estimator remains linear. Large innovation autocorrelation or inconsistent NIS/NEES means the identified `Q` absorbs nonlinear model mismatch; the nonlinear fitted `Q` is not a verified field covariance. The generated ST uses the **matched linear simulation fit**, not the nonlinear stress-test parameters.
