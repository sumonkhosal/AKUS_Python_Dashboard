// ============================================================================
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
    nPriorMode      : INT  := 2;
    fSampleDt       : LREAL := 0.01;          // [s] Filter sample period (Delta t)
    fProcessQ       : LREAL := 5.1957094061e-05; // [(m/s^2)^2/s] Continuous process spectral density q
    fMeasR          : LREAL := 1.5884077765e-03; // [(m/s^2)^2] Measurement noise variance R at fSampleDt
    fNetForce_N     : LREAL := 0.0;                 // [N] Gravity-compensated spindle drive force u_k
    fAccMeas_ms2    : LREAL := 0.0;                 // [m/s^2] Calibrated dynamic acceleration measurement y_k
    // Physical reference parameters (from GVL_Persist.TcGVL)
    fMassPlatform   : LREAL := 17.395000;      // [kg]
    fMassEquivalent : LREAL := 4154.590000;    // [kg]
    fViscousB       : LREAL := 3994.857000;       // [Ns/m]
    fCoulombFc      : LREAL := 225.059000;// [N]
    fPosFrictionBz  : LREAL := 1207.700000;// [N/m^2]
    fZReference     : LREAL := 0.249715;     // [m]
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
