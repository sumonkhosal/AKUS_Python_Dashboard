(* AKUS offline 2-state position/velocity Kalman filter. No ADS/motor output.
   Generated coefficients: fixed dt. Input force u [N] is gravity-compensated.
   Position encoder [m], acceleration [m/s2]; offline calibration REQUIRED.
   Call every dt; do not pass previous sample twice.
   Python Golden CSV uses initial x=[first encoder,0], P=diag(max(R11,1e-10),0.02^2).
   This is NOT Jan's original PLC FB and does NOT add nonlinear friction.
*)
FUNCTION_BLOCK FB_AKUS_TwoStateKF
VAR_INPUT
    bEnable : BOOL := TRUE;
    bReset : BOOL;
    bMeasurementValid : BOOL := TRUE;
    fPosMeas_m : LREAL;
    fAccMeas_ms2 : LREAL;
    fNetForce_N : LREAL;
END_VAR
VAR_OUTPUT
    fZEst_m : LREAL;
    fVEst_ms : LREAL;
    fK11 : LREAL;
    fK12 : LREAL;
    fK21 : LREAL;
    fK22 : LREAL;
    fS11 : LREAL;
    fS12 : LREAL;
    fS22 : LREAL;
    fP11 : LREAL;
    fP12 : LREAL;
    fP22 : LREAL;
    fNuPos_m : LREAL;
    fNuAcc_ms2 : LREAL;
    fNIS : LREAL;
    bError : BOOL;
    nSamples : UDINT;
END_VAR
VAR
    (* Parameters generated from Jan's damped two-state model. *)
    a11 : LREAL := 1;
    a12 : LREAL := 0.0099522752762231329;
    a21 : LREAL := 0;
    a22 : LREAL := 0.99047026378255265;
    b1 : LREAL := 1.1946541209577074e-08;
    b2 : LREAL := 2.3855012125458581e-06;
    c11 : LREAL := 1;
    c12 : LREAL := 0;
    c21 : LREAL := 0;
    c22 : LREAL := -0.95754347151296071;
    d1 : LREAL := 0;
    d2 : LREAL := 0.0002396940545088249;
    q11 : LREAL := 2.9586531484217345e-11;
    q12 : LREAL := 4.4273744643775753e-09;
    q22 : LREAL := 8.8548165855377309e-07;
    r11 : LREAL := 1.287228976905685e-07;
    r12 : LREAL := -2.1900928124166311e-08;
    r22 : LREAL := 0.0015219094165495656;
    p011 : LREAL := 1.287228976905685e-07;
    p022 : LREAL := 0.00040000000000000002;
    x : ARRAY[1..2] OF LREAL;
    xp : ARRAY[1..2] OF LREAL;
    P : ARRAY[1..2,1..2] OF LREAL;
    Pp : ARRAY[1..2,1..2] OF LREAL;
    AA : ARRAY[1..2,1..2] OF LREAL;
    B : ARRAY[1..2] OF LREAL;
    H : ARRAY[1..2,1..2] OF LREAL;
    D : ARRAY[1..2] OF LREAL;
    QQ : ARRAY[1..2,1..2] OF LREAL;
    RR : ARRAY[1..2,1..2] OF LREAL;
    S : ARRAY[1..2,1..2] OF LREAL;
    K : ARRAY[1..2,1..2] OF LREAL;
    G : ARRAY[1..2,1..2] OF LREAL;
    T : ARRAY[1..2,1..2] OF LREAL;
    W : ARRAY[1..2,1..2] OF LREAL;
    J : ARRAY[1..2,1..2] OF LREAL;
    Rv : ARRAY[1..2] OF LREAL;
    detS : LREAL;
    inv11 : LREAL;
    inv12 : LREAL;
    inv22 : LREAL;
    sum : LREAL;
    oldForce : LREAL;
    initialized : BOOL;
    i : INT;
    j : INT;
    m : INT;
END_VAR
IF NOT bEnable THEN
    RETURN;
END_IF;
AA[1,1]:=a11; AA[1,2]:=a12; AA[2,1]:=a21; AA[2,2]:=a22;
B[1]:=b1; B[2]:=b2;
H[1,1]:=c11; H[1,2]:=c12; H[2,1]:=c21; H[2,2]:=c22;
D[1]:=d1; D[2]:=d2;
QQ[1,1]:=q11; QQ[1,2]:=q12; QQ[2,1]:=q12; QQ[2,2]:=q22;
RR[1,1]:=r11; RR[1,2]:=r12; RR[2,1]:=r12; RR[2,2]:=r22;
IF bReset OR NOT initialized THEN
    x[1]:=fPosMeas_m; x[2]:=0.0;
    P[1,1]:=p011; P[1,2]:=0.0; P[2,1]:=0.0; P[2,2]:=p022;
    initialized:=TRUE;
    nSamples:=0;
    oldForce:=fNetForce_N;
    bError:=FALSE;
END_IF;
(* Exactly like Python: k=0 starts without prediction. *)
IF nSamples=0 THEN
    FOR i:=1 TO 2 DO
        xp[i]:=x[i];
        FOR j:=1 TO 2 DO Pp[i,j]:=P[i,j]; END_FOR;
    END_FOR;
ELSE
    FOR i:=1 TO 2 DO
        xp[i]:=B[i]*oldForce;
        FOR j:=1 TO 2 DO xp[i]:=xp[i]+AA[i,j]*x[j]; END_FOR;
    END_FOR;
    (* T=AA*P, Pp=T*AA' + QQ *)
    FOR i:=1 TO 2 DO
        FOR j:=1 TO 2 DO
            T[i,j]:=0.0;
            FOR m:=1 TO 2 DO T[i,j]:=T[i,j]+AA[i,m]*P[m,j]; END_FOR;
        END_FOR;
    END_FOR;
    FOR i:=1 TO 2 DO
        FOR j:=1 TO 2 DO
            Pp[i,j]:=QQ[i,j];
            FOR m:=1 TO 2 DO Pp[i,j]:=Pp[i,j]+T[i,m]*AA[j,m]; END_FOR;
        END_FOR;
    END_FOR;
END_IF;
oldForce:=fNetForce_N;
IF NOT bMeasurementValid THEN
    FOR i:=1 TO 2 DO
        x[i]:=xp[i];
        FOR j:=1 TO 2 DO P[i,j]:=Pp[i,j]; END_FOR;
    END_FOR;
    nSamples:=nSamples+1;
    fZEst_m:=x[1]; fVEst_ms:=x[2];
    fP11:=P[1,1]; fP12:=P[1,2]; fP22:=P[2,2];
    RETURN;
END_IF;
Rv[1]:=fPosMeas_m-(H[1,1]*xp[1]+H[1,2]*xp[2]+D[1]*fNetForce_N);
Rv[2]:=fAccMeas_ms2-(H[2,1]*xp[1]+H[2,2]*xp[2]+D[2]*fNetForce_N);
(* G=Pp*H', S=H*G+R *)
FOR i:=1 TO 2 DO
    FOR j:=1 TO 2 DO
        G[i,j]:=0.0;
        FOR m:=1 TO 2 DO G[i,j]:=G[i,j]+Pp[i,m]*H[j,m]; END_FOR;
    END_FOR;
END_FOR;
FOR i:=1 TO 2 DO
    FOR j:=1 TO 2 DO
        S[i,j]:=RR[i,j];
        FOR m:=1 TO 2 DO S[i,j]:=S[i,j]+H[i,m]*G[m,j]; END_FOR;
    END_FOR;
END_FOR;
S[1,2]:=0.5*(S[1,2]+S[2,1]); S[2,1]:=S[1,2];
detS:=S[1,1]*S[2,2]-S[1,2]*S[1,2];
IF (S[1,1]<=0.0) OR (detS<=0.0) THEN
    bError:=TRUE;
    RETURN; (* Never divide by invalid innovation covariance *)
END_IF;
inv11:=S[2,2]/detS; inv12:=-S[1,2]/detS; inv22:=S[1,1]/detS;
FOR i:=1 TO 2 DO
    K[i,1]:=G[i,1]*inv11+G[i,2]*inv12;
    K[i,2]:=G[i,1]*inv12+G[i,2]*inv22;
END_FOR;
FOR i:=1 TO 2 DO
    x[i]:=xp[i]+K[i,1]*Rv[1]+K[i,2]*Rv[2];
END_FOR;
(* Joseph: P = (I-KH)Pp(I-KH)' + KRK' *)
FOR i:=1 TO 2 DO
    FOR j:=1 TO 2 DO
        J[i,j]:=0.0;
        IF i=j THEN J[i,j]:=1.0; END_IF;
        FOR m:=1 TO 2 DO J[i,j]:=J[i,j]-K[i,m]*H[m,j]; END_FOR;
    END_FOR;
END_FOR;
FOR i:=1 TO 2 DO
    FOR j:=1 TO 2 DO
        T[i,j]:=0.0; W[i,j]:=0.0;
        FOR m:=1 TO 2 DO
            T[i,j]:=T[i,j]+J[i,m]*Pp[m,j];
            W[i,j]:=W[i,j]+K[i,m]*RR[m,j];
        END_FOR;
    END_FOR;
END_FOR;
FOR i:=1 TO 2 DO
    FOR j:=1 TO 2 DO
        P[i,j]:=0.0;
        FOR m:=1 TO 2 DO
            P[i,j]:=P[i,j]+T[i,m]*J[j,m]+W[i,m]*K[j,m];
        END_FOR;
    END_FOR;
END_FOR;
P[1,2]:=0.5*(P[1,2]+P[2,1]); P[2,1]:=P[1,2];
fZEst_m:=x[1]; fVEst_ms:=x[2];
fK11:=K[1,1]; fK12:=K[1,2]; fK21:=K[2,1]; fK22:=K[2,2];
fS11:=S[1,1]; fS12:=S[1,2]; fS22:=S[2,2];
fP11:=P[1,1]; fP12:=P[1,2]; fP22:=P[2,2];
fNuPos_m:=Rv[1]; fNuAcc_ms2:=Rv[2];
fNIS:=Rv[1]*(inv11*Rv[1]+inv12*Rv[2])+Rv[2]*(inv12*Rv[1]+inv22*Rv[2]);
nSamples:=nSamples+1;
