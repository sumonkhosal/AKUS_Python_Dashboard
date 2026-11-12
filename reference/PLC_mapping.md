# PLC-to-Python traceability

The Python simulator was written based on the actual contents of the uploaded project, not solely the earlier prose summary.

## TwinCAT source symbols referenced

| TwinCAT source | Relevant behavior | Dashboard component |
|---|---|---|
| `GVL.TcGVL` | `fHomingPosition`, `fGravity`, `fTolerancePositionSyncPos`, `fToleranceVelocitySyncPos` | constants and diagnostics |
| `GVL_Persist.TcGVL` | `fMErs`, `fMPlatform`, `fBc`, `fBv`, `fBz`, `fZ0`, velocity/position PID tunings | `ControlConfig` and tuning page |
| `E_AkusState.TcDUT` | `Error`, `Init`, `Ready`, `Moving`, `ToStart`, `CtrlConfig`, `Calib`, `Ident` | status bar; many workflows approximated |
| `E_OpMode.TcDUT` | `SyncAccTorque=6`, `SyncPosTorque=7`, `SyncPos=8`, `SyncVel=9`, `SyncTorq=10`, `SyncTorqComm=11` | display/mode names; no real drive op-mode switching |
| `E_MoveType.TcDUT` | Position, Velocity, Beckhoff, Jog | Manual tab |
| `MAIN.TcPOU` | `nRefState` 20–25/1–11 | `Simulator.start_homing`, `_on_motion_done` |
| `MAIN.TcPOU` | `bMoveToStart`, `bStartManMove`, `bJogPos`, `bJogNeg`, `bStopAxis` | Manual and Experiment tabs |
| `MAIN.TcPOU` | `fGiveAcceleration`, `fAccTime`, `fCalcVel`, `fCalcDist`, `fMaxDeceleration`, `fMaxDurElevator` | `Simulator.experiment_plan`, `.start_experiment` |
| `MAIN.TcPOU` | `nPosExpStep`, `nBeckhoffExpStep`, `nAccExpStep` | simplified experiment delay/up/lock/return sequence |
| `MAIN.TcPOU` | identification state and configuration test variables | identification/control test actions |
| `MAIN.TcPOU` | `bSwitchOnTurboPump`, `bSwitchOnDiaphragmPump`, 3 valve command pairs | simulated vacuum state panel |
| `FeedbackControl.TcPOU` | `fbAxis1`, `fbAxis2`, actual motor states, `bTouchProbe1/2` | virtual status/probe readings |
| `FB_AkusModel.TcPOU` | torques → spindle force → gravity/friction → acceleration → integrated z/v | force model option |
| `FB_ConstAccSetpointGen.TcPOU` | triangular/trapezoidal setpoint | analytic trapezoid implementation, not exact ST integration |
| `FB_ScurveSetpointGen.TcPOU` | 7/8-stage jerk-limited S curve (incomplete stop logic in source) | Not implemented; no unsupported fidelity claim |
| `FB_AccIntegrator.TcPOU` | Euler/trapezoidal/RK4 selectable | simplified numerical integration only |
| `FB_AkusKalmanFilter.TcPOU` | Kalman filter bug audit | not included; caution in Diagnostics |
| `VISUs/Experiment.TcVIS` | experiment page | Experiment tab |
| `VISUs/Manual.TcVIS` | manual page | Manual motion tab |
| `VISUs/ControlConfigVisu.TcVIS` | controller tuning | Controller tuning tab |
| `VISUs/VacControl.TcVIS` | vacuum control page | Vacuum tab |
| `VISUs/Analysis.TcVIS` | line graphs | Analysis/Scopes tab |
| `Measurement/*.tcscopex` | channel names for 8 original Scope projects | reference/scope_catalog.json |

## Caveats while comparing with TwinCAT

* Project homing: two variants are exposed as one coherent up/reference/down sequence. The Python engine does not reproduce each per-scan flag, PLCopen `MC_Home` handshake, error handling, or drive enable sequence. Its virtual NC switch opens at 44000 degrees.
* User can inject a display-only axis skew for visualization. The plant dynamics remain single degree-of-freedom because the uploaded system model is 1-DOF and both spindles are mechanically tied. Such skew is NOT a physical model of bending/binding forces.
* Test and identification modes are trajectory workflows with sampled traces. They do not reproduce the full legacy ST conditionals, clocked TCP/ADS triggers or all physical control variants.
* The original project has unresolved bugs described in the included audit documents. Python does not silently mimic bugs that are unsafe or dimensionally inconsistent.
* All writes in this repository update Python objects exclusively. The repository does not import `pyads`, `pysoem`, `ads`, or connect to EtherCAT/PLC hardware.
