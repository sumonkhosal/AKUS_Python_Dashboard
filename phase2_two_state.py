"""AKUS two-state [position,velocity] identification and TwinCAT export.

Source: V4 `akuskf` two-state research; Scope CSV may be real and/or synthetic.
Process intensity q_a fits a constrained PSD Q via exact damped Van-Loan discretization.
R comes from stationary measurements at the chosen filter sample period.
Do not conflate acceleration-only scalar Q/R from earlier phases with these matrices.

This research software has no PLC I/O and cannot assert hardware calibration.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from scipy.stats import chi2
from akuskf.jan_model import ModelParameters, matrices, continuous_white_acceleration_Q
from akuskf.filtering import run_filter
from akuskf.identification import estimate_q_by_likelihood, validation_metrics
from akuskf.benchmark import run_benchmark, report_json_ready

HEADERS = ['time_s','fPos1','fPos2','fTorque1','fTorque2','fAccZUnfiltered']

def portable(obj):
    if isinstance(obj, np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    if isinstance(obj,dict):return {str(k):portable(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [portable(v) for v in obj]
    if isinstance(obj,float) and not math.isfinite(obj):return None
    return obj

def read_scope_windows(csv_path, windows):
    """Streaming parser of TwinCAT 2 kHz Scope exports; doesn't load 136 MB at once."""
    regions={key:[] for key in windows}
    with open(csv_path,encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f,delimiter=';')
        for i,row in enumerate(reader):
            if i<13 or len(row)<12:continue
            try: t=float(row[0])/1000.0
            except (ValueError,IndexError):continue
            for key,(a,b) in windows.items():
                if a<=t<=b:
                    try: regions[key].append([t,float(row[1]),float(row[4]),float(row[3]),float(row[6]),float(row[11])])
                    except(ValueError,IndexError):pass
                    break
    return {key:np.asarray(value,dtype=float).reshape(-1,6) for key,value in regions.items()}

def decimate(raw,target_dt=.01):
    if len(raw)<100:raise ValueError('Insufficient scope samples')
    step=float(np.median(np.diff(raw[:,0])))
    ratio=int(round(target_dt/step))
    if ratio<1 or ratio>400 or not np.isclose(step*ratio,target_dt,rtol=1e-3):
        raise ValueError('Scope sampling not compatible with desired TwinCAT dt')
    filt=np.column_stack([resample_poly(raw[:,i],1,ratio,padtype='line') for i in range(1,6)])
    times=np.arange(len(filt))*step*ratio
    return np.column_stack((times,filt)),step*ratio

def initial_real_R(stationary_original,dt=.01,p=None,coordinate_sign=-1.0):
    if p is None:p=ModelParameters()
    st,actual_dt=decimate(stationary_original,dt)
    z=coordinate_sign*(st[:,1]+st[:,2])*0.5*p.pitch/360.0
    a=st[:,5]*p.gravity
    residuals=np.column_stack((z-z.mean(),a-a.mean()))
    empirical=np.cov(residuals,rowvar=False,ddof=1)
    # Noise floor is strictly a conservative engineering parameter, not measured R.
    # Encoder at rest has too little variation to identify every uncertainty source.
    floor_z=(1e-6)**2 # 1 micron position engineering floor (not a measurement)
    used=empirical.copy()
    used[0,0]=max(float(used[0,0]),floor_z)
    # Cap cross-correlation to ensure robust, strictly SPD covariance.
    max_cross=.8*np.sqrt(used[0,0]*used[1,1])
    used[0,1]=used[1,0]=np.clip(used[0,1],-max_cross,max_cross)
    if np.min(np.linalg.eigvalsh(used))<=0:raise ValueError('R is not SPD after floor')
    rho=np.corrcoef(residuals[:-1].T,residuals[1:].T)
    return portable(dict(dt=actual_dt,R_empirical=empirical,R_for_exploratory_filter=used,
       position_noise_sd_m=float(np.sqrt(empirical[0,0])),
       acceleration_noise_sd_ms2=float(np.sqrt(empirical[1,1])),
       acceleration_lag1=float(rho[1,3]),position_lag1=float(rho[0,2]),
       static_baseline_g=float(np.median(stationary_original[:,5])),
       floor_z_m=1e-6,window_s=[2,10],sample_count=len(st),
       warning='Engineering position-noise floor and decimation mean R_for_exploratory_filter is NOT a calibrated real R for all motion regimes.'))

def real_observation_data(motion_original,p,baseline_g,dt,coordinate_sign=-1,torque_scale=1.,torque_sign=1.):
    m,actual=decimate(motion_original,dt)
    pos=p.z_reference+coordinate_sign*((m[:,1]+m[:,2])/2.0-(m[0,1]+m[0,2])/2.0)*p.pitch/360.0
    a=(m[:,5]-baseline_g)*p.gravity
    force=torque_sign*torque_scale*p.motor_force(m[:,3],m[:,4])
    u=p.u_from_force(force)
    return dict(t=m[:,0],y=np.column_stack((pos,a)),u=np.asarray(u),dt=actual)

def _write_rows(path,header,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(header)
        for row in rows:w.writerow([f'{float(x):.17g}' for x in row])

def compare_reference_cyclewise(y,u,model,R,q,dt):
    """Independent 2x2 analytical-replay reference for the generated TwinCAT ST math.

    NOT execution of a TwinCAT binary; tests equations before actual PLC replay.
    """
    A,B,C,D,Ac=matrices(dt,model)
    Q=continuous_white_acceleration_Q(dt,Ac,q)
    P=np.diag([max(float(R[0,0]),1e-10),0.02**2])
    x=np.array([y[0,0],0.],float)
    I=np.eye(2)
    ret=[]
    for k in range(len(y)):
        if k:
            x=A@x+B[:,0]*u[k-1]
            P=A@P@A.T+Q
        nu=y[k]-(C@x+D[:,0]*u[k])
        S=C@P@C.T+R
        invS=np.array([[S[1,1],-S[0,1]],[-S[1,0],S[0,0]]])/np.linalg.det(S)
        K=(P@C.T)@invS
        x=x+K@nu
        M=I-K@C
        P=M@P@M.T+K@R@K.T
        P=.5*(P+P.T)
        nis=nu@invS@nu
        ret.append([x[0],x[1],*K.flatten(),S[0,0],S[0,1],S[1,1],P[0,0],P[0,1],P[1,1],*nu,float(nis)])
    return np.asarray(ret)

def golden_export(out,filename):
    d=out['validation']; res=out['validation_result'];
    ref=compare_reference_cyclewise(d['y'],d['u'],out['parameters_model'],out['R_est'],out['q_est'],out['dt'])
    expected=np.column_stack((res['estimates'],res['gains'].reshape(-1,4),res['S'][:,0,0],res['S'][:,0,1],res['S'][:,1,1],res['P'][:,0,0],res['P'][:,0,1],res['P'][:,1,1],res['innovations'],res['nis']))
    diff=float(np.nanmax(np.abs(expected-ref)))
    header=['t_s','u_N','encoder_z_m','measured_acc_ms2','reference_z_m','reference_v_ms','K11','K12','K21','K22','S11','S12','S22','P11','P12','P22','nu_pos','nu_acc','NIS']
    _write_rows(filename,header,np.column_stack((d['t'],d['u'],d['y'],expected)))
    return diff

def st_source(A,B,C,D,Q,R,dt,P0):
    """ST source for the 2-state linear model with per-cycle dynamic gain."""
    def f(v):return f'{float(v):.17g}'
    defaults='\n'.join(f'    {k} : LREAL := {f(v)};'
        for k,v in {'a11':A[0,0],'a12':A[0,1],'a21':A[1,0],'a22':A[1,1],
        'b1':B[0,0],'b2':B[1,0],'c11':C[0,0],'c12':C[0,1],'c21':C[1,0],'c22':C[1,1],
        'd1':D[0,0],'d2':D[1,0],'q11':Q[0,0],'q12':Q[0,1],'q22':Q[1,1],
        'r11':R[0,0],'r12':R[0,1],'r22':R[1,1],
        'p011':P0[0,0],'p022':P0[1,1]}.items())
    source=r'''(* AKUS offline 2-state position/velocity Kalman filter. No ADS/motor output.
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
__DEFAULTS__
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
'''
    return source.replace('__DEFAULTS__',defaults)

def run(args):
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    p=ModelParameters()
    benchmark=run_benchmark(dt=args.dt,n_train=950,n_validation=950,plant=args.plant,
                            q_true=1e-4,noise_z_mm=.35,noise_a_ms2=.04,seed=42)
    benchmark['parameters_model']=p
    sim_report=report_json_ready(benchmark)
    maxdiff=golden_export(benchmark,output/'twincat_2state_golden_vectors.csv')
    sim_report['replay_python_analytical_max_abs_difference']=maxdiff
    sim_report['replay_provenance']='Python analytical equations, NOT actual TwinCAT runtime'
    A,B,C,D,Ac=matrices(args.dt,p)
    Q=benchmark['validation_result']['Q'];R=benchmark['R_est']
    P0=np.diag([max(float(R[0,0]),1e-10),.02**2])
    (output/'two_state_simulation_report.json').write_text(json.dumps(portable(sim_report),indent=2),encoding='utf-8')
    (output/'FB_AKUS_TwoStateKF.st').write_text(st_source(A,B,C,D,Q,R,args.dt,P0),encoding='utf-8')
    real_report={'provided':bool(args.real_csv), 'status':'no real scope CSV supplied'}
    if args.real_csv:
        raw=read_scope_windows(args.real_csv,{'stationary':(2.,10.),'train':(100.,110.),'holdout':(230.,240.)})
        Rcal=initial_real_R(raw['stationary'],args.dt,p)
        real_report={'provided':True,'source':'Test_KalmanFilter.csv, real Earth hardware',
                     'stationary':Rcal,'physical_model_status':'Not validated: torque scaling/sign, acceleration calibration and 1-DOF model need independent checks',
                     'q_identified_from_real_data':False,
                     'notes':['Real R measured on stationary Earth hardware only; filtering at 100Hz changes its noise statistics',
                              'Real motor force is NOT used to claim a calibrated Q because torque conversion/polarity and stiction are unverified',
                              'Simulation-derived Q/R and K should NOT be deployed as real-hardware calibration']}
        obs=real_observation_data(raw['holdout'],p,Rcal['static_baseline_g'],args.dt)
        _write_rows(output/'real_acceleration_position_excerpt_100Hz.csv',
            ['time_s','pos_m_relative_to_first','acc_ms2_baseline_corrected','net_force_N_unverified'],
            np.column_stack((obs['t'],obs['y'][:,0]-p.z_reference,obs['y'][:,1],obs['u'])))
        # Audit the hold torque at static and contrast assumptions; not a calibration.
        static=raw['stationary']
        tf=p.motor_force(np.mean(static[:,3]),np.mean(static[:,4]))
        real_report['stationary_force_audit']={'raw_torque_sum_Nm':float(np.mean(static[:,3]+static[:,4])),
            'ideal_spindle_force_if_Nm':float(tf), 'platform_weight_N':p.m_platform*p.gravity,
            'gravity_balance_difference_N_if_ideal_force':float(tf-p.m_platform*p.gravity),
            'warning':'Stiction and unverified scaling mean this difference is NOT proof of erroneous torque calibration'}
    (output/'real_stationary_R_report.json').write_text(json.dumps(portable(real_report),indent=2),encoding='utf-8')
    print('Finished: files in',output)
    print('Two-state Q per cycle:',Q,'R simulation:',R,'K last:',benchmark['validation_result']['gains'][-1],sep='\n')
    print('Held-out synthetic metrics:',sim_report['metrics'])
    if args.real_csv:print('Real stationary R:',real_report['stationary']['R_empirical'])
    return sim_report,real_report

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--real-csv',type=Path,default=None)
    cli.add_argument('--out',type=Path,default=Path('results'))
    cli.add_argument('--plant',choices=['linear','jan_euler'],default='linear')
    cli.add_argument('--dt',type=float,default=.01)
    run(cli.parse_args())
