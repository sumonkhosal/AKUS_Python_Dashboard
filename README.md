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
