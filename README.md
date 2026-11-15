# AKUS Python Control Studio — Offline Virtual HMI

A runnable **Python + Streamlit web dashboard** based on the uploaded AKUS Beckhoff TwinCAT 3 project. It visualizes the twin vertical spindle axes and simulated carrier motion and exposes useful operator controls, physical parameters, tests, and measurement plots.

> **Simulation only. This is not a drop-in replacement for TwinCAT, the drive controller, the safety PLC or the real operator HMI.** The app has **no ADS/EtherCAT/OPC-UA write capability**. The emergency-stop control is only an offline simulation control, not a machinery safety function. Never operate the actual carrier through this dashboard without a separate, validated industrial safety design and commissioning.

## Run on Windows

1. Extract the ZIP to a folder.
2. Make sure Python 3.10 or newer is installed, then open **PowerShell** in that folder.
3. Run:

   ```powershell
   py -m pip install -r requirements.txt
   py -m streamlit run app.py
   ```

   Or double-click `start_windows.bat`, which also installs the dependencies. If `py` is not available, use `python`.
4. Streamlit normally opens a browser page at `http://localhost:8501`. Use **Home both axes (simulation)**, then **Run sequence to completion** to jump to Ready quickly. From there, test Manual motion, Experiment, Identification, and so on.

Linux/macOS: `python3 -m pip install -r requirements.txt && python3 -m streamlit run app.py`.

## What works now

- Interactive graphical mechanism: two spindles, two motor symbols, moving carrier, moving height indicator, live position/velocity/acceleration/status.
- Homing (both existing MAIN routes, summarized as movement sequences): simulated upward travel to the NC TouchProbe (44,000 motor degrees), coordinate set to the reference, automatic return to 0 degrees, READY state.
- Manual absolute moves; directional **bounded timed** jog; go-to-start; operator stop; virtual E-STOP / reset; no movement allowed before referencing; upper and lower virtual travel guards.
- Ideal analytic trapezoidal move profile (position, velocity, acceleration); optional nonlinear force balance with rotor-equivalent mass, carrier gravity, Coulomb/viscous/position-dependent friction.
- Experiment planner: input acceleration in g, acceleration time or total time; calculate velocity, stroke, braking time; trigger delay, lock period, camera/recording status, auto return; collision-clearance and elevator-duration checks.
- Parameter identification sequence (start → move and record → return); controller configuration response test; persistent-style editable gains and model coefficients with JSON export/import.
- Vacuum control: schematic and simulated I/O statuses for two pumps and three valves, plus open/closed sensor status.
- Plotly scopes for position, velocity, acceleration, torque estimate and setpoints; CSV and JSON export; actual `.tcscopex` preset channel names extracted for reference.
- Event diagnostics and source-to-feature audit table.

## Source-derived constants

| Item | Value | Source |
|---|---:|---|
| Spindle pitch | 5 mm/revolution | `FB_AkusModel`, `MAIN` |
| Reference/top coordinate | 44,000° ≈ 611.11 mm | `GVL`, `MAIN` |
| Experiment soft guard | 44,000° − 720° = 43,280° ≈ 601.11 mm | `MAIN.TcPOU` |
| Homing velocity | 700°/s | `MAIN.TcPOU` |
| Homing return velocity | 2,800°/s | `MAIN.TcPOU` |
| Max experiment deceleration | 20,000°/s² | `MAIN.TcPOU` |
| Platform mass | 17.395 kg | `GVL_Persist` |
| Equivalent rotor/spindle mass | 4,154.59 kg | `GVL_Persist` |
| Coulomb friction | 225.059 N | `GVL_Persist` |
| Viscous friction | 3994.857 Ns/m | `GVL_Persist` |
| Position-dependent friction | 1207.7 N/m² | `GVL_Persist` |
| Friction center | 0.249715 m | `GVL_Persist` |
| Earth gravity | 9.80665 m/s² | `GVL` |

## Critical fidelity limitations

**Not an exact port:** The source project includes IEC Structured Text, NC PTP/PLCopen motion blocks, physical EtherCAT drive behavior, TwinCAT tasks, visualization objects, scope acquisition, sensor hardware, and custom controller details. The dashboard does **not** reproduce exact PLC scan ordering, timers, ADS buffers, every state transition, drive saturation, safety interlocks, low-level jog behavior, specialized acceleration and position controllers, or TCP input from the Raspberry Pi. The same numbers do **not** ensure the same trajectory on physical hardware.

**Two selectable model styles:**

1. **Ideal NC trajectory** (default): carrier follows an analytic trapezoid. This is an ideal motion profile, **not** the measured servo response. Mode switching changes labels/workflow, not real control-loop implementations. The same ideal position is used for both motor axes unless the operator adds a *visual-only* encoder offset.
2. **Dynamic force model:** uses the project's force balance and physical coefficients with a **simplified assumed feedback law**, a default artificial torque clamp of `2.5 Nm/motor`, and numerical integration. Assumed gains and torque clamps are **not verified drive settings**. At trajectory completion the virtual mechanism snaps to its target for workflow convenience; its terminal behavior is not a prediction of settling on the real system.

**Experiment formula:** The project's `MAIN.TcPOU` total-duration branch uses a deceleration distance term without the conventional ½ multiplier, while the acceleration-time branch correctly includes ½. This implementation deliberately uses the physically consistent ½ for both branches; accordingly it is not bit-for-bit identical to the buggy source.

**Sensor chain:** Project audit `AKUS_Kalman_Filter_and_Sensor_Processing_Audit.md` reports the Kalman filter is never enabled in the call site and flags measurement-unit and calibration bugs. The dashboard offers only synthetic acceleration display and a small display-only low-pass approximation; it does **not** claim a faithful EL3702 sensor, 10x oversampling, Butterworth filter, Kalman estimate or calibration process.

**Vacuum:** Switch states are mirrored digitally from commands. No pressure, valve actuation delay, pump model, limit switch faults, or physical I/O is represented.

**Visualization:** The mechanism is a simplified schematic built from uploaded project illustrations rather than an exact CAD geometry or pixel-perfect reproduction of the original TwinCAT visualizations.

**Out of scope:** DriveManager 2 parameter sets, hardware commissioning, remote camera/Pi feed, real-time ADS read/write, real NC/PTP tuning, physical axis decoupling/calibration, and control/safety certification.

## Test the numerical core

```powershell
py -m pip install pytest
py -m pytest -q
```

The numerical engine itself only uses Python standard-library modules and tests do not require Streamlit. Source-derived numerical behavior is tested. Passing tests does not constitute safe hardware commissioning.

## Project structure

- `app.py` — Streamlit interface and live SVG motion illustration
- `akus_sim.py` — simulation, trajectories, state progression, experiment calculation, virtual I/O
- `assets/` — four images from supplied TwinCAT visualization project
- `reference/PLC_mapping.md` — implementation mapping, important source names and intentional differences
- `reference/scope_catalog.json` — parsed channels from eight source Scope projects
- `tests/test_engine.py` — deterministic numerical tests
- `requirements.txt` — Streamlit, numpy, pandas, plotly

For next-stage engineering work, the safest progression is to validate offline generated trajectories against logged TwinCAT Scope data, add a **read-only** ADS monitoring adapter, and only then design a separately reviewed control integration supervised by the existing Beckhoff safety and motion system.

---

## v2: Kalman research integrated with the original virtual carrier

The original carrier animation, manual moves, homing, experiments, controller-tuning,
vacuum and scopes remain in the same `app.py`. Open the new **Kalman research** tab.

**Install/launch:** same commands as above. The updated `requirements.txt` adds SciPy
and FilterPy. There is **no hardware or ADS connection**. The Kalman estimate is
read-only and never drives the virtual or real controller; its input is the
simulator's recorded historical motion (not the instantaneous live motor loop).

Modes:

1. **Current carrier simulation**: Run a virtual movement first, then use the
   Kalman research tab to inject controlled *synthetic* encoder/acceleration noise
   into its recorded truth trajectory and estimate R/Q. The reference Kalman
   implementation and optional FilterPy implementation output estimates,
   innovations, S, K, NIS and CSV/JSON exports. The ideal NC profile is not
   generated by the same physical force equation; expect model mismatch.
2. **Included real AKUS recording**: The ZIP includes two **REAL** excerpts from
   the uploaded `Test_KalmanFilter.csv`: 2–10 seconds stationary for calibration
   and 230–240 seconds for motion. This is not the complete 296-second file.
3. **Upload TwinCAT Scope CSV**: Upload your full Scope export, optionally with
   a separate stationary calibration export, and select a 0.5–30 second motion
   excerpt. Upload of a ~134 MB CSV may take some time/memory. The fallback
   stationary calibration is the bundled November 2026 data and is **not**
   automatically valid for a different machine, calibration or date.

### How the real recording is processed

- Import the source TwinCAT semicolon-delimited CSV (metadata + 13 channels).
- `fPos1`, `fPos2` are in motor degrees; average and convert to **relative**
  vertical metres using 5 mm/revolution. The sign is configurable; absolute
  carrier height and encoder zero offsets cannot be recovered from this CSV.
- `fAccZUnfiltered` is expressed in g and is **already averaged over ten EL3702
  oversamples** in PLC source. Subtract a stationary median +1g baseline, then
  convert to m/s². This removes a static reference, **not sensor sensitivity,
  axis alignment, or temperature drift**.
- `fTorque1`, `fTorque2` are in nominal Nm according to `FeedbackControl`.
  Their force sign and scaling are not independently verified; they are
  configurable for exploration. A fitted q cannot repair bad torque scaling.
- Low-pass/resample all time series to the selected 5–50 ms estimator interval
  (`scipy.signal.resample_poly`). Derive R from the **same downsampling pipeline**
  applied to stationary measurements. The raw stationary R is also reported,
  but must not be used in place of resampled R. A 1e-12 SI numerical floor
  prevents singular encoder covariance; it is **not** an encoder accuracy claim.
- Run the corrected two-state **linear damped** reference model (state z,v).
  It excludes nonlinear dry and position-dependent friction by design; the
  simulator's dynamic plant still has that richer approximation.
- Estimate one physically structured continuous acceleration disturbance q_a by
  minimising the Gaussian innovation negative log-likelihood, holding R fixed.
- S (innovation covariance) and K (gain) are calculated every update, not
  fitted or typed as constants. Display NIS, gain, final Q/R/S/K and signals.
- Show moving-versus-stationary acceleration power spectra, plus coherence
  with torque and velocity at the original Scope sampling rate. This does
  **not** causally separate vibration from electronic sensor noise.

A train/held-out split of one continuous recording is an exploratory check, **not
independent validation**. Use physically separate experimental runs for an
actual validation campaign. FilterPy is optional at runtime (choose `reference`
if it is unavailable); the reference implementation uses SciPy Cholesky solves
and Joseph-form covariance update. Only the numerical backend and source
compilation were testable in the build environment; Streamlit was not installed
there and the end-to-end GUI needs a local launch test.

### Example real-data result and remaining research blocker

With the included motion segment, 20 ms processing, original Jan mass/viscous
friction coefficients, default torque orientation/scaling, and R from the
stationary excerpt, the Q likelihood optimiser hits its **upper search bound**.
This should be treated as **model mismatch or insufficient parameter
identifiability**, not a confirmed physically calibrated Q. Verify the torque
conversion, motor direction, encoder sign, accelerometer scale/gravity, and
nonlinear friction behaviour before deploying this estimator anywhere near
real control.

**Useful commands:**

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m pytest -q
```

Implementation files:
- `kalman_lab.py` — Scope parsing, sensor preprocessing, R, spectral analysis,
  existing-history simulation bridge, Q-fitting pipeline.
- `akuskf/` — corrected Jan linear matrices, exact discretisation, Van Loan Q,
  transparent reference + optional FilterPy core, likelihood optimisation.
- `app.py` — new last tab integrated into the original moving-machine HMI.
- `tests/test_kalman_integration.py` — real recording, R/PSD/coherence, Q/K/S,
  and simulator-history integration tests.
- `examples/akus_stationary_REAL.csv`, `examples/akus_movement_REAL.csv` —
  excerpts from the user's actual TwinCAT Scope recording.

---

## v3 — source-aligned simulated-physics Kalman validation (first research milestone)

The far-right **Simulation Q/R benchmark** tab adds an independently validated
identification experiment while preserving all v2 pages. Two truth plants are
available: the corrected matched linear damped model, and the **Jan nonlinear
Euler model** based on the real `FB_AkusModel` force/friction rules and
`FB_AccIntegrator`'s explicit Euler case. Both are offline physics
benchmarks, not the full TwinCAT drive/NC application.

The new workflow generates **known, separately injected** motion disturbances
and sensor noise, estimates R from an independent stationary calibration
sample, fits structured Q using only a dynamic training trajectory, and then
tests on a **different** force profile and random seed. The report includes
injected vs recovered Q/R, filter S and K, position/velocity RMSE, innovation
correlations, NIS, NEES and 95% position coverage, with warnings on mismatches.

Quick command-line reproductions:

```sh
python run_benchmark.py --plant linear --output results/matched_linear
python run_benchmark.py --plant jan_euler --output results/jan_euler
python -m pytest -q
```

Read `SIMULATION_VALIDATION_GUIDE.md` for exact source mapping, mathematical
units, experiment design and interpretation. All original controls and the
separate v2 **Kalman research** real-CSV mode are still present.


## Acceleration KF add-on (v4)

See [ACCELERATION_KF_GUIDE.md](ACCELERATION_KF_GUIDE.md). New dashboard tab **Acceleration KF** uses Jan simulated runs and the included **real Earth-hardware** excerpts, with an acceleration-first Kalman filter, scalar Q/R estimation, and S/K diagnostics. Existing motion controls and old position/velocity filter are preserved.
