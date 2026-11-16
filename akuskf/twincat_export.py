"""Python-to-TwinCAT 3 Structured Text (IEC 61131-3) exporter & sample-by-sample equivalence verifier.

Generates `FB_AkusNonlinearAccKF` (.st and .TcPOU) implementing:
- Open-loop nonlinear Euler (or linear / random-walk) acceleration prediction `fAccPrior`
  using Jan's `FB_AkusModel` / `FB_AccIntegrator` equations (Coulomb, position-dependent,
  viscous friction, and stiction deadzone) without ground-truth access.
- Dynamic per-cycle covariance recursion:
    Q_step = fProcessQ * fSampleDt
    P_pred = fP + Q_step
    fS     = P_pred + fMeasR
    fK     = P_pred / fS
    fInnov = (fAccMeas - fAccPrior) - fResidualD
    fResidualD = fResidualD + fK * fInnov
    fP     = (1.0 - fK)*(1.0 - fK)*P_pred + fK*fK*fMeasR   // Joseph form
    fAccEst = fAccPrior + fResidualD
    fNIS   = (fInnov * fInnov) / fS
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .jan_model import ModelParameters, matrices
from .acceleration import (
    run_acceleration_filter,
    jan_linear_acceleration_prior,
    jan_nonlinear_acceleration_prior,
)


def generate_twincat_st_code(
    *,
    q_identified: float = 5.1957094e-5,
    r_identified: float = 1.5884078e-3,
    dt_s: float = 0.01,
    p: ModelParameters | None = None,
    default_prior_mode: int = 2,
) -> str:
    """Return complete IEC 61131-3 Structured Text source for `FB_AkusNonlinearAccKF`."""
    if p is None:
        p = ModelParameters()
    return f"""// ============================================================================
// Function Block: FB_AkusNonlinearAccKF
// Target:         Beckhoff TwinCAT 3 (IEC 61131-3 Structured Text, LREAL 64-bit)
// Purpose:        1-State Physics-Guided Acceleration Kalman Filter for AKUS.
//                 Computes S_k and K_k dynamically every PLC scan using the
//                 Joseph-stabilized covariance recursion (NOT a fixed gain K).
//                 Offline simulation / equivalence verification block.
// ============================================================================
FUNCTION_BLOCK FB_AkusNonlinearAccKF
VAR_INPUT
    bEnable         : BOOL := TRUE;
    bReset          : BOOL := FALSE;
    // Prior selection: 0 = Random-Walk (fAccPrior=0), 1 = Jan Linear, 2 = Jan Nonlinear Euler
    nPriorMode      : INT  := {int(default_prior_mode)};
    fSampleDt       : LREAL := {dt_s:.8g};          // [s] Filter sample period (Delta t)
    fProcessQ       : LREAL := {q_identified:.10e}; // [(m/s^2)^2/s] Continuous process spectral density q
    fMeasR          : LREAL := {r_identified:.10e}; // [(m/s^2)^2] Measurement noise variance R at fSampleDt
    fNetForce_N     : LREAL := 0.0;                 // [N] Gravity-compensated spindle drive force u_k
    fAccMeas_ms2    : LREAL := 0.0;                 // [m/s^2] Calibrated dynamic acceleration measurement y_k
    // Physical reference parameters (from GVL_Persist.TcGVL)
    fMassPlatform   : LREAL := {p.m_platform:.6f};      // [kg]
    fMassEquivalent : LREAL := {p.m_equivalent:.6f};    // [kg]
    fViscousB       : LREAL := {p.b_viscous:.6f};       // [Ns/m]
    fCoulombFc      : LREAL := {p.friction_coulomb:.6f};// [N]
    fPosFrictionBz  : LREAL := {p.friction_position:.6f};// [N/m^2]
    fZReference     : LREAL := {p.z_reference:.6f};     // [m]
END_VAR
VAR_OUTPUT
    bValid          : BOOL  := FALSE;
    fAccPrior_ms2   : LREAL := 0.0; // [m/s^2] Open-loop physics predicted acceleration a_prior[k]
    fAccEst_ms2     : LREAL := 0.0; // [m/s^2] Posterior Kalman-filtered acceleration a_est[k]
    fResidualD_ms2  : LREAL := 0.0; // [m/s^2] Estimated residual state d_est[k]
    fStdAcc_ms2     : LREAL := 0.0; // [m/s^2] Posterior standard deviation sqrt(P_k)
    fP_ms4          : LREAL := 0.0; // [(m/s^2)^2] Posterior state variance P_k
    fPpred_ms4      : LREAL := 0.0; // [(m/s^2)^2] Prior state variance P_k^-
    fQstep_ms4      : LREAL := 0.0; // [(m/s^2)^2] Per-sample process variance Q_step = q * dt
    fS_ms4          : LREAL := 0.0; // [(m/s^2)^2] Innovation variance S_k = P_k^- + R
    fK              : LREAL := 0.0; // [-] Dynamic Kalman gain K_k = P_k^- / S_k
    fInnovation_ms2 : LREAL := 0.0; // [m/s^2] Innovation nu_k = (y_k - a_prior_k) - d_pred
    fNIS            : LREAL := 0.0; // [-] 1D Normalized Innovation Squared (nu_k^2 / S_k)
    fPredZ_m        : LREAL := 0.0; // [m] Open-loop model-predicted position z_hat[k]
    fPredV_ms       : LREAL := 0.0; // [m/s] Open-loop model-predicted velocity v_hat[k]
END_VAR
VAR
    bInitialized    : BOOL  := FALSE;
    fZHat           : LREAL := 0.0;
    fVHat           : LREAL := 0.0;
    fMTotal         : LREAL := 0.0;
    fAlpha          : LREAL := 0.0;
    fExpAlphaDt     : LREAL := 0.0;
    fA01            : LREAL := 0.0;
    fB0             : LREAL := 0.0;
    fB1             : LREAL := 0.0;
    fPosTerm        : LREAL := 0.0;
    fSignedFriction : LREAL := 0.0;
    fResultingForce : LREAL := 0.0;
    fOneMinusK      : LREAL := 0.0;
    fNextZ          : LREAL := 0.0;
    fNextV          : LREAL := 0.0;
END_VAR

// ----------------------------------------------------------------------------
// Implementation
// ----------------------------------------------------------------------------
IF NOT bEnable THEN
    bValid := FALSE;
    RETURN;
END_IF;

fMTotal := fMassPlatform + fMassEquivalent;
IF (fSampleDt <= 0.0) OR (fMeasR <= 0.0) OR (fProcessQ < 0.0) OR (fMTotal <= 0.0) THEN
    bValid := FALSE;
    RETURN;
END_IF;

IF bReset OR (NOT bInitialized) THEN
    fZHat          := fZReference;
    fVHat          := 0.0;
    fResidualD_ms2 := 0.0;
    IF fMeasR > 1.0E-12 THEN
        fP_ms4 := fMeasR;
    ELSE
        fP_ms4 := 1.0E-12;
    END_IF;
    bInitialized := TRUE;
END_IF;

fPredZ_m  := fZHat;
fPredV_ms := fVHat;

// 1. Open-Loop Physics Acceleration Prior (Zero Ground-Truth or Sensor Access)
IF nPriorMode = 2 THEN
    // Jan Nonlinear Euler Prior (FB_AkusModel + FB_AccIntegrator Euler)
    fPosTerm := fCoulombFc + fPosFrictionBz * (fZHat - fZReference) * (fZHat - fZReference);
    IF fNetForce_N > 0.0 THEN
        fSignedFriction := fPosTerm + fViscousB * fVHat;
    ELSIF fNetForce_N < 0.0 THEN
        fSignedFriction := -fPosTerm + fViscousB * fVHat;
    ELSE
        fSignedFriction := 0.0;
    END_IF;

    IF ABS(fNetForce_N) < ABS(fSignedFriction) THEN
        fResultingForce := 0.0;
    ELSE
        fResultingForce := fNetForce_N - fSignedFriction;
    END_IF;

    fAccPrior_ms2 := fResultingForce / fMTotal;
    // Explicit Euler state propagation for next cycle (k -> k+1)
    fNextZ := fZHat + fSampleDt * fVHat;
    fNextV := fVHat + fSampleDt * fAccPrior_ms2;
    fZHat  := fNextZ;
    fVHat  := fNextV;

ELSIF nPriorMode = 1 THEN
    // Jan Linear Damped Physics Prior (Exact sampled matrix exponential)
    fAlpha      := fViscousB / fMTotal;
    fAccPrior_ms2 := (fNetForce_N - fViscousB * fVHat) / fMTotal;
    IF fAlpha > 1.0E-8 THEN
        fExpAlphaDt := EXP(-fAlpha * fSampleDt);
        fA01        := (1.0 - fExpAlphaDt) / fAlpha;
        fB0         := (fAlpha * fSampleDt - (1.0 - fExpAlphaDt)) / (fMTotal * fAlpha * fAlpha);
        fB1         := (1.0 - fExpAlphaDt) / (fMTotal * fAlpha);
        fNextZ      := fZHat + fA01 * fVHat + fB0 * fNetForce_N;
        fNextV      := fExpAlphaDt * fVHat + fB1 * fNetForce_N;
    ELSE
        fNextZ      := fZHat + fSampleDt * fVHat + 0.5 * fSampleDt * fSampleDt * fNetForce_N / fMTotal;
        fNextV      := fVHat + fSampleDt * fNetForce_N / fMTotal;
    END_IF;
    fZHat := fNextZ;
    fVHat := fNextV;

ELSE
    // Ordinary Random-Walk Baseline (No physics prior)
    fAccPrior_ms2 := 0.0;
END_IF;

// 2. Dynamic Scalar Kalman Prediction & Joseph-Form Correction
fQstep_ms4      := fProcessQ * fSampleDt;
fPpred_ms4      := fP_ms4 + fQstep_ms4;
fS_ms4          := fPpred_ms4 + fMeasR;
fK              := fPpred_ms4 / fS_ms4;
fInnovation_ms2 := (fAccMeas_ms2 - fAccPrior_ms2) - fResidualD_ms2;
fResidualD_ms2  := fResidualD_ms2 + fK * fInnovation_ms2;

// Joseph stabilized covariance update: P_k = (1 - K_k)^2 * P_pred + K_k^2 * R
fOneMinusK      := 1.0 - fK;
fP_ms4          := (fOneMinusK * fOneMinusK) * fPpred_ms4 + (fK * fK) * fMeasR;
fStdAcc_ms2     := SQRT(MAX(0.0, fP_ms4));

fAccEst_ms2     := fAccPrior_ms2 + fResidualD_ms2;
fNIS            := (fInnovation_ms2 * fInnovation_ms2) / fS_ms4;
bValid          := TRUE;
"""


def simulate_twincat_fb_cycle_by_cycle(
    u_force_n,
    y_acc_ms2,
    *,
    dt: float,
    r: float,
    q: float,
    prior_mode: int = 2,
    p: ModelParameters | None = None,
) -> dict[str, np.ndarray]:
    """Execute the exact line-by-line LREAL arithmetic of `FB_AkusNonlinearAccKF` in Python
    to verify sample-by-sample numerical equivalence with `run_acceleration_filter`.
    """
    if p is None:
        p = ModelParameters()
    u = np.asarray(u_force_n, dtype=float).reshape(-1)
    y = np.asarray(y_acc_ms2, dtype=float).reshape(-1)
    if u.shape != y.shape or len(y) < 2:
        raise ValueError("u_force_n and y_acc_ms2 must have identical length >= 2")

    n = len(y)
    fZHat = float(p.z_reference)
    fVHat = 0.0
    fResidualD = 0.0
    fP = float(max(r, 1e-12))
    fMTotal = float(p.m_platform + p.m_equivalent)

    fAccPrior = np.zeros(n, dtype=float)
    fAccEst = np.zeros(n, dtype=float)
    fResD = np.zeros(n, dtype=float)
    fStdAcc = np.zeros(n, dtype=float)
    fP_arr = np.zeros(n, dtype=float)
    fPpred_arr = np.zeros(n, dtype=float)
    fS_arr = np.zeros(n, dtype=float)
    fK_arr = np.zeros(n, dtype=float)
    fInnov_arr = np.zeros(n, dtype=float)
    fNIS_arr = np.zeros(n, dtype=float)
    fPredZ = np.zeros(n, dtype=float)
    fPredV = np.zeros(n, dtype=float)

    for k in range(n):
        fPredZ[k] = fZHat
        fPredV[k] = fVHat
        uk = float(u[k])
        yk = float(y[k])

        if prior_mode == 2:
            fPosTerm = p.friction_coulomb + p.friction_position * (fZHat - p.z_reference) * (fZHat - p.z_reference)
            if uk > 0.0:
                fSignedFriction = fPosTerm + p.b_viscous * fVHat
            elif uk < 0.0:
                fSignedFriction = -fPosTerm + p.b_viscous * fVHat
            else:
                fSignedFriction = 0.0
            if abs(uk) < abs(fSignedFriction):
                fResultingForce = 0.0
            else:
                fResultingForce = uk - fSignedFriction
            acc_prior = fResultingForce / fMTotal
            fNextZ = fZHat + dt * fVHat
            fNextV = fVHat + dt * acc_prior
            fZHat = fNextZ
            fVHat = fNextV
        elif prior_mode == 1:
            fAlpha = p.b_viscous / fMTotal
            acc_prior = (uk - p.b_viscous * fVHat) / fMTotal
            if fAlpha > 1e-8:
                fExpAlphaDt = math.exp(-fAlpha * dt)
                fA01 = (1.0 - fExpAlphaDt) / fAlpha
                fB0 = (fAlpha * dt - (1.0 - fExpAlphaDt)) / (fMTotal * fAlpha * fAlpha)
                fB1 = (1.0 - fExpAlphaDt) / (fMTotal * fAlpha)
                fNextZ = fZHat + fA01 * fVHat + fB0 * uk
                fNextV = fExpAlphaDt * fVHat + fB1 * uk
            else:
                fNextZ = fZHat + dt * fVHat + 0.5 * dt * dt * uk / fMTotal
                fNextV = fVHat + dt * uk / fMTotal
            fZHat = fNextZ
            fVHat = fNextV
        else:
            acc_prior = 0.0

        fQstep = float(q * dt)
        fPpred = fP + fQstep
        fS = fPpred + r
        fK = fPpred / fS
        fInnov = (yk - acc_prior) - fResidualD
        fResidualD = fResidualD + fK * fInnov
        fOneMinusK = 1.0 - fK
        fP = (fOneMinusK * fOneMinusK) * fPpred + (fK * fK) * r
        fStd = math.sqrt(max(0.0, fP))
        acc_est = acc_prior + fResidualD
        nis = (fInnov * fInnov) / fS

        fAccPrior[k] = acc_prior
        fAccEst[k] = acc_est
        fResD[k] = fResidualD
        fStdAcc[k] = fStd
        fP_arr[k] = fP
        fPpred_arr[k] = fPpred
        fS_arr[k] = fS
        fK_arr[k] = fK
        fInnov_arr[k] = fInnov
        fNIS_arr[k] = nis

    return {
        "fAccPrior_ms2": fAccPrior,
        "fAccEst_ms2": fAccEst,
        "fResidualD_ms2": fResD,
        "fStdAcc_ms2": fStdAcc,
        "fP_ms4": fP_arr,
        "fPpred_ms4": fPpred_arr,
        "fS_ms4": fS_arr,
        "fK": fK_arr,
        "fInnovation_ms2": fInnov_arr,
        "fNIS": fNIS_arr,
        "fPredZ_m": fPredZ,
        "fPredV_ms": fPredV,
    }


def verify_python_to_twincat_equivalence(
    u_force_n,
    y_acc_ms2,
    *,
    dt: float = 0.01,
    r: float = 1.5884078e-3,
    q: float = 5.1957094e-5,
    prior_mode: int = 2,
    p: ModelParameters | None = None,
) -> dict:
    """Compare Python `run_acceleration_filter(..., backend='filterpy')` against
    `simulate_twincat_fb_cycle_by_cycle` sample-by-sample across all outputs.
    """
    if p is None:
        p = ModelParameters()
    tc = simulate_twincat_fb_cycle_by_cycle(
        u_force_n, y_acc_ms2, dt=dt, r=r, q=q, prior_mode=prior_mode, p=p
    )
    if prior_mode == 2:
        py_prior = jan_nonlinear_acceleration_prior(u_force_n, dt, p)
    elif prior_mode == 1:
        py_prior = jan_linear_acceleration_prior(u_force_n, dt, p)
    else:
        py_prior = np.zeros_like(y_acc_ms2, dtype=float)

    py_kf = run_acceleration_filter(
        y_acc_ms2, dt, r, q, prior_ms2=py_prior, backend="filterpy"
    )

    max_diff_prior = float(np.max(np.abs(tc["fAccPrior_ms2"] - py_prior)))
    max_diff_est = float(np.max(np.abs(tc["fAccEst_ms2"] - py_kf["estimated_acc_ms2"])))
    max_diff_K = float(np.max(np.abs(tc["fK"] - py_kf["K"])))
    max_diff_S = float(np.max(np.abs(tc["fS_ms4"] - py_kf["S"])))
    max_diff_P = float(np.max(np.abs(tc["fP_ms4"] - py_kf["P"])))
    max_diff_innov = float(np.max(np.abs(tc["fInnovation_ms2"] - py_kf["innovation"])))
    max_diff_nis = float(np.max(np.abs(tc["fNIS"] - py_kf["nis"])))

    df_vectors = pd.DataFrame({
        "sample_k": np.arange(len(y_acc_ms2), dtype=int),
        "time_s": np.arange(len(y_acc_ms2), dtype=float) * dt,
        "fNetForce_N": np.asarray(u_force_n, dtype=float),
        "fAccMeas_ms2": np.asarray(y_acc_ms2, dtype=float),
        "py_acc_prior_ms2": py_prior,
        "tc_fAccPrior_ms2": tc["fAccPrior_ms2"],
        "py_acc_est_ms2": py_kf["estimated_acc_ms2"],
        "tc_fAccEst_ms2": tc["fAccEst_ms2"],
        "py_K": py_kf["K"],
        "tc_fK": tc["fK"],
        "py_S_ms4": py_kf["S"],
        "tc_fS_ms4": tc["fS_ms4"],
        "py_P_ms4": py_kf["P"],
        "tc_fP_ms4": tc["fP_ms4"],
        "py_NIS": py_kf["nis"],
        "tc_fNIS": tc["fNIS"],
        "abs_diff_acc_est_ms2": np.abs(tc["fAccEst_ms2"] - py_kf["estimated_acc_ms2"]),
    })

    return {
        "prior_mode": int(prior_mode),
        "n_samples": int(len(y_acc_ms2)),
        "dt_s": float(dt),
        "q": float(q),
        "r": float(r),
        "max_abs_diff_prior_ms2": max_diff_prior,
        "max_abs_diff_acc_est_ms2": max_diff_est,
        "max_abs_diff_K": max_diff_K,
        "max_abs_diff_S_ms4": max_diff_S,
        "max_abs_diff_P_ms4": max_diff_P,
        "max_abs_diff_innovation_ms2": max_diff_innov,
        "max_abs_diff_nis": max_diff_nis,
        "is_bit_exact_lreal": bool(
            max(max_diff_prior, max_diff_est, max_diff_K, max_diff_S, max_diff_P, max_diff_nis) < 1e-13
        ),
        "vectors_df": df_vectors,
    }
