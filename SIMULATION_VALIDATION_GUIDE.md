# AKUS Python Studio v3 — Jan simulation Kalman benchmark

## Research objective (now, not later)

**Now:** Fit `R` from independent *synthetic stationary noise*, fit a structured
`Q` to an independently simulated **Jan-derived physical plant**, and validate
`S`, `K`, state estimation, NIS and NEES on a wholly different virtual motion run.

**Later:** Use the real AKUS/TwinCAT Scope recording to estimate empirical `R`
and analyze vibration, calibration and different models. Don't transfer synthetic
Q/R into real operating modes without new identification.

## How to use the dashboard

1. Extract `AKUS_Python_Control_Studio_v3.zip`.
2. Enter `AKUS_Python_Dashboard` and double click `start_windows.bat` (or
   `py -m pip install -r requirements.txt` and `py -m streamlit run app.py`).
3. The original animation, manual motion, experiments and **Kalman research**
   pages are unchanged. Open the new far-right tab **Simulation Q/R benchmark**.
4. First select **Matched linear plant**, click **Run train / held-out benchmark**.
   Compare *Injected q* to *Identified q*, *R injected* to *R estimated*, then
   inspect NIS/NEES and the true-vs-encoder-vs-Kalman position plot.
5. Without changing the noise levels, switch to **Jan nonlinear Euler plant**.
   The model mismatch should now show up in the innovation tests. These are
   **diagnostics**, not a reason to blindly increase Q.
6. Repeat with different seeds and sample intervals to judge stability.
7. Export the time-series CSV and research JSON; both include validation data.

For a reproducible headless run (no Streamlit needed):

```sh
python run_benchmark.py --plant linear --output results/matched_linear
python run_benchmark.py --plant jan_euler --output results/jan_euler
python -m pytest -q
```

The package contains already-generated, auditable baseline output in
`results/matched_linear/` and `results/jan_euler/`.

## Important details about Jan's actual TwinCAT project

I re-opened these original files in `akus-regelung-Jan_Hoffmann_Servos.zip`:

- `POUs/FB_AkusModel.TcPOU`: `fDrivingForce=(fTorque1+fTorque2)*2*pi/0.005`
  minus platform gravity (outside elevator), conditional signed Coulomb and
  position-dependent friction, plus viscous friction; zero resulting force when
  driving force does not exceed the signed friction threshold. Position and
  velocity are then integrated; acceleration output additionally receives +g
  on Earth, which represents an accelerometer-like specific-force convention.
- `POUs/FB_AccIntegrator.TcPOU`: the implemented Euler case updates velocity
  with previous acceleration and **position using previous velocity**.

This is faithfully reflected in the **offline Jan nonlinear Euler** option
for the friction force and one-step integration, with intentionally **added
stochastic process perturbations** so Q is identifiable. This benchmark does
**not** run the full TwinCAT task, servo drive, NC motion controller,
mechanical brakes, I/O, PLC filter activation, or timing, and is NOT an exact
real-time PLC digital twin. The earlier GUI **Dynamic force model** has its own
approximate virtual controller; the new benchmark isolates Jan's physics.

The **matched linear plant** solves the corrected damped model with the same
sampled matrices as the reference Kalman filter. That baseline answers whether
our identification implementation works when its assumptions hold.

## Mathematical definitions (units matter)

State: `x = [z (m), velocity (m/s)]`.  Input `u (N)` is net servo force **after
subtracting platform weight**. The baseline continuous model is:

```
zdot = v
vdot = - b_v / (m_platform+m_equivalent) * v + u/(m_platform+m_equivalent)
```

The measurements passed to FilterPy are `[encoder position m, dynamic
acceleration m/s²]` (gravity and bias already known/subtracted in the synthetic
case). Real accelerometers measure *specific force*, not dynamic acceleration
by default; the real-data page separately handles the 1g stationary baseline.

`Q = integral exp(Ac*s) diag(0,q_a) exp(Ac.T*s) ds`, with `q_a` the continuous
white-acceleration spectral density (m²/s³). `R` is the sampled measurement
covariance, with diagonal entries in m² and (m/s²)². Off-diagonal entries have
mixed position × acceleration units.

The process perturbations are applied to simulated motion **independently**
from the generated sensor noise. A separate stationary calibration sample
identifies `R` without consulting known truth `R`. Dynamic TRAINING motion
fits scalar `q_a` by minimizing innovation Gaussian negative log-likelihood.
The VALIDATION motion uses a different force-excitation profile and different
random seed, and is not used to fit anything. Hence the injected Q/R appear
in the report only for checking the recovered values, never as optimizer inputs.

`S_k = C P^- C.T + R` and `K_k = P^- C.T S_k^{-1}` are computed on each update;
we do **not** manually tune a K constant. Cholesky linear solves avoid explicit
matrix inversion in the reference filter. The covariance correction uses Joseph
form. The optional `FilterPy` backend runs the equivalent estimator.

## What counts as success?

- `R_est` approximately matches injected `R_true` when sensor noise is white;
  absolute error decreases as the independent calibration sample grows.
- `q_est` approximately matches injected `q_true` in the matched-model case;
  equality for a single random seed is **not expected**.
- Kalman position/velocity RMSE improves over noisy encoder measurements.
- Held-out mean **NIS** and **NEES** should be statistically compatible with
  expectations (~2 for the 2D measurement/state) under linear Gaussian
  assumptions. Also inspect the innovation autocorrelation and 95% position
  interval coverage. Reported chi-square bounds assume independent samples.
- When switching to Jan's nonlinear friction model, `q_est` may inflate to
  absorb model mismatch. NIS can fail even if position traces look smooth.
  This is an intended, scientifically useful **failure test**.

## Boundaries and things not yet identified

- *No recorded sensor* `R` here: the v3 synthetic benchmark identifies known
  simulated noise; the existing v2 Kalman research page remains available for
  actual TwinCAT CSV recordings.
- Sensor bias/drift, temporal correlation, structural vibration and
  accelerometer calibration are not represented by the baseline independent
  Gaussian sensor model. We can introduce them in controlled sensitivity tests
  once baseline identification is trustworthy.
- Jan's real motor torque sign/gear reduction is not identified from this
  benchmark; input is prescribed as **known net force**, specifically to
  isolate estimator reliability from torque conversion uncertainty.
- No read/write ADS, EtherCAT, hardware control, real E-STOP, or automatic
  deployment to TwinCAT. Benchmarks must not be used for safety decisions.
- The GUI layout has not been live-launched in the build environment because
  Streamlit could not be installed there. Numerical tests and source compilation
  were executed; launch locally as described above.
