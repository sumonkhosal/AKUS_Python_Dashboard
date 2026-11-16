"""Read-only AKUS Kalman lab helpers for the existing virtual carrier dashboard.

No ADS, motor or physical I/O. CSV imports must be validated before physics inference.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, welch, coherence
from akuskf import ModelParameters, estimate_q_by_likelihood, run_filter
from akuskf.identification import validation_metrics
from akus_sim import ControlConfig, GRAVITY

REAL_CHANNELS = ('fPos1', 'fPos2', 'fVel1', 'fVel2', 'fTorque1', 'fTorque2',
                 'fAccZUnfiltered', 'fAccZ')
OPTIONAL_ACC_CHANNELS = ('fAccZg', 'fAvg3702Z')


def _discover_channel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Robust channel discovery for TwinCAT Scope CSV exports.

    Handles exact names as well as ADS symbol path prefixes (e.g., 'MAIN.fAccZUnfiltered')
    and preserves optional accelerometer channels (`fAccZg`, `fAvg3702Z`) when present.
    """
    rename_map = {}
    target_names = ('time_s', 'time_ms') + REAL_CHANNELS + OPTIONAL_ACC_CHANNELS
    existing = set(df.columns)
    for target in target_names:
        if target in existing:
            continue
        for col in df.columns:
            clean = str(col).strip()
            if clean.endswith('.' + target) or clean.lower() == target.lower():
                rename_map[col] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_twincat_csv(source):
    """Parse TwinCAT Scope semicolon-delimited export (6 metadata header rows).

    The first column is elapsed milliseconds; skip the Scope channel metadata.
    Also accept regular CSVs generated from this importer for easier reuse.
    """
    if hasattr(source, 'seek'):
        source.seek(0)
    if isinstance(source, (str, __import__("pathlib").Path)):
        with open(source, 'rb') as f:
            head = f.read(120)
    elif isinstance(source, bytes):
        head = source[:120]
    else:
        head = source.read(120)
        source.seek(0)
    if isinstance(head, str):
        head = head.encode('utf-8', errors='ignore')
    twin = head.startswith(b'Name;') and b'\nFile;' in head
    if twin:
        # 0-5: export metadata, 6: channel header, 7-12: channel metadata
        df = pd.read_csv(source, sep=';', header=0,
                         skiprows=lambda i: i < 6 or (7 <= i <= 12),
                         decimal='.', low_memory=False)
        if 'Name' not in df.columns:
            raise ValueError('TwinCAT CSV does not contain expected Name/time column')
        df = df.rename(columns={'Name': 'time_ms'})
        df['time_s'] = pd.to_numeric(df['time_ms'], errors='coerce') / 1000.0
    else:
        df = pd.read_csv(source, sep=None, engine='python')
        df = _discover_channel_columns(df)
        if 'time_s' not in df.columns:
            if 'time_ms' in df.columns:
                df['time_s'] = pd.to_numeric(df['time_ms'], errors='coerce') / 1000.0
            else:
                raise ValueError('CSV needs TwinCAT Scope header or time_s/time_ms')
    df = _discover_channel_columns(df)
    # Fallback: if fAccZUnfiltered is absent but fAvg3702Z or fAccZg is present
    if 'fAccZUnfiltered' not in df.columns:
        if 'fAvg3702Z' in df.columns:
            df['fAccZUnfiltered'] = df['fAvg3702Z']
        elif 'fAccZg' in df.columns:
            df['fAccZUnfiltered'] = df['fAccZg']
    needed = ('time_s',) + REAL_CHANNELS
    missing = [k for k in needed if k not in df.columns]
    if missing:
        raise ValueError('Missing required CSV channels: ' + ', '.join(missing))
    extra = [c for c in OPTIONAL_ACC_CHANNELS if c in df.columns]
    cols = ['time_s', *REAL_CHANNELS, *extra]
    # Original file has duplicate fTorque1/fTorque2 columns: pandas appends '.1';
    # first originals are already in Nm per FeedbackControl source.
    result = df[cols].copy()
    for name in cols:
        result[name] = pd.to_numeric(result[name], errors='coerce')
    result = result.dropna(subset=['time_s', 'fPos1', 'fPos2', 'fTorque1',
                                   'fTorque2', 'fAccZUnfiltered'])
    result = result.sort_values('time_s').drop_duplicates('time_s')
    if len(result) < 50:
        raise ValueError('At least 50 valid synchronized samples are required')
    return result.reset_index(drop=True)


def baseline_from_stationary(df):
    """Calibrate *zero-motion reference* only, NOT accelerometer scale/axis orientation."""
    baseline = float(df['fAccZUnfiltered'].median())
    if not (0.7 < baseline < 1.3):
        raise ValueError('Stationary signal is not near +1 g. Check orientation/calibration.')
    return baseline


def _decimate(df, target_dt):
    times = df['time_s'].to_numpy(dtype=float)
    dt_raw = float(np.median(np.diff(times)))
    if dt_raw <= 0 or not np.isfinite(dt_raw):
        raise ValueError('Invalid timestamps')
    # resample_poly low-pass filters before decimation; don't treat the original
    # 2 kHz R as the covariance of the resulting 50-200 Hz measurement stream.
    ratio = max(1, int(round(float(target_dt) / dt_raw)))
    if ratio > 1000:
        raise ValueError('Requested sample period much larger than the recording period')
    names = list(REAL_CHANNELS)
    if ratio == 1:
        vals = df[names].to_numpy(dtype=float)
        new_t = times - times[0]
    else:
        vals = np.column_stack([resample_poly(df[c].to_numpy(float), 1, ratio, padtype="line") for c in names])
        new_t = np.arange(len(vals), dtype=float) * dt_raw * ratio
    return pd.DataFrame(vals, columns=names).assign(time_s=new_t), dt_raw * ratio


def make_real_observations(motion, stationary, dt=0.02, coordinate_sign=-1,
                           torque_sign=1, torque_scale=1.0, config=None):
    """One coherent channel pipeline for real CSV.

    z is RELATIVE to initial averaged encoder degrees, not absolute physical
    height; polarities/signs require an independent calibration. The baseline
    static +1g is subtracted from raw acceleration for kinematic acceleration.
    """
    p = parameters_from_config(config)
    motion, actual_dt = _decimate(motion, dt)
    static, _ = _decimate(stationary, dt)
    baseline = baseline_from_stationary(stationary)
    av = (motion['fPos1'].to_numpy()+motion['fPos2'].to_numpy()) / 2.0
    position = float(coordinate_sign) * (av-av[0]) * p.pitch / 360.0
    acc = (motion['fAccZUnfiltered'].to_numpy() - baseline) * GRAVITY
    force = float(torque_sign)*float(torque_scale)*p.motor_force(
        motion['fTorque1'].to_numpy(), motion['fTorque2'].to_numpy())
    u = np.asarray(p.u_from_force(force), dtype=float)
    av_cal = (static['fPos1'].to_numpy()+static['fPos2'].to_numpy()) / 2.0
    znoise = float(coordinate_sign) * (av_cal-np.mean(av_cal))*p.pitch/360.0
    anoise = (static['fAccZUnfiltered'].to_numpy()-np.mean(static['fAccZUnfiltered'])) * GRAVITY
    residuals = np.column_stack((znoise, anoise))
    R = stable_R(residuals)
    y = np.column_stack((position, acc))
    return dict(t=motion['time_s'].to_numpy(), y=y, u=u, dt=actual_dt,
                R=R, p=p, truth=None, raw=motion, baseline_g=baseline,
                raw_R=stable_R(np.column_stack((
                    float(coordinate_sign)*((stationary['fPos1']+stationary['fPos2'])/2.0 -
                    float(((stationary['fPos1']+stationary['fPos2'])/2.0).mean()))*p.pitch/360,
                    (stationary['fAccZUnfiltered']-stationary['fAccZUnfiltered'].mean())*GRAVITY))),
                source='Real TwinCAT Scope CSV')


def parameters_from_config(config=None):
    c = config or ControlConfig()
    return ModelParameters(m_platform=float(c.m_platform_kg),
                           m_equivalent=float(c.m_equivalent_kg),
                           b_viscous=float(c.friction_viscous_ns_m),
                           friction_coulomb=float(c.friction_coulomb_n),
                           friction_position=float(c.friction_position_n_m2),
                           z_reference=float(c.friction_center_m))


def stable_R(residuals):
    """Sample covariance with a documented tiny numerical floor (not calibration)."""
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 2 or r.shape[1] != 2 or len(r) < 30 or not np.isfinite(r).all():
        raise ValueError('At least 30 finite calibration samples are required')
    R = np.cov(r, rowvar=False)
    # Encoder quantization can produce zero covariance when the machine is static.
    # This is a computational floor, not proof of true encoder precision.
    diag_floor = np.diag([1e-12, 1e-12])
    R = R + diag_floor
    eigen, vec = np.linalg.eigh((R+R.T)/2)
    eigen = np.maximum(eigen, 1e-12)
    return (vec*eigen)@vec.T


def make_simulation_observations(history, config=None, noise_position_mm=.20,
                                 noise_acc_ms2=.015, seed=42):
    """Inject *explicitly synthetic* independent measurement noise into existing
    engine history (same carrier trajectory, controller, and parameters).
    """
    df = pd.DataFrame(history).drop_duplicates('time_s').sort_values('time_s')
    if len(df) < 50:
        raise ValueError('Please move the simulated carrier and record at least 50 points')
    t = df['time_s'].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or np.max(abs(np.diff(t)-dt)) > .2*dt:
        # Resample to a uniform time base for the constant-dt filter.
        tnew = np.arange(t[0],t[-1] + 0.1*dt,dt)
        data = {k:np.interp(tnew,t,df[k].to_numpy(float)) for k in
                ('pos_mm','vel_mm_s','acc_ms2','torque1_nm','torque2_nm')}
        t = tnew
    else:
        data = {k:df[k].to_numpy(float) for k in
                ('pos_mm','vel_mm_s','acc_ms2','torque1_nm','torque2_nm')}
    p = parameters_from_config(config)
    truth = np.column_stack((data['pos_mm']/1000.,data['vel_mm_s']/1000.))
    rng = np.random.default_rng(int(seed))
    R_true = np.diag([(noise_position_mm/1000)**2,noise_acc_ms2**2])
    y = (np.column_stack((truth[:,0],data['acc_ms2'])) +
         rng.multivariate_normal(np.zeros(2), R_true, size=len(t)))
    u = np.asarray(p.u_from_force(p.motor_force(data['torque1_nm'],data['torque2_nm'])),float)
    # Independent dedicated simulated stationary calibration noise.
    calibration = rng.multivariate_normal(np.zeros(2),R_true,size=2500)
    return dict(t=t-t[0], y=y, u=u,dt=dt,p=p,R=stable_R(calibration),
                truth=truth, raw=df,baseline_g=1.,raw_R=None,
                source='Existing carrier simulation + synthetic noise')


def analyze_frequency(df, duration_s=3.0, stationary_df=None):
    """Welch PSD/coherence on ORIGINAL rate, with synchronized motor variables."""
    t = df['time_s'].to_numpy(float)
    dt=float(np.median(np.diff(t)))
    if len(t)<256 or dt<=0:
        raise ValueError('Need 256 samples for spectral analysis')
    block = df[(df['time_s']>=t[0]) & (df['time_s'] <= t[0]+duration_s)]
    fs = 1/dt
    x=block['fAccZUnfiltered'].to_numpy(float)*GRAVITY
    tq=(block['fTorque1']+block['fTorque2']).to_numpy(float)
    vel=(block['fVel1']+block['fVel2']).to_numpy(float)/2.0
    nperseg = min(2048,len(x))
    if nperseg<128:raise ValueError('Need a longer interval for frequency analysis')
    f,P=welch(x,fs=fs,nperseg=nperseg)
    _,Ct=coherence(x,tq,fs=fs,nperseg=nperseg)
    _,Cv=coherence(x,vel,fs=fs,nperseg=nperseg)
    out = pd.DataFrame({'frequency_hz':f,'acc_psd':P,'coherence_torque':Ct,
                        'coherence_velocity':Cv})
    if stationary_df is not None and len(stationary_df)>255:
        tt=stationary_df['time_s'].to_numpy(float)
        fs2=1/float(np.median(np.diff(tt)))
        f2,p2=welch(stationary_df['fAccZUnfiltered'].to_numpy(float)*GRAVITY,
                    fs=fs2,nperseg=min(2048,len(stationary_df)))
        out['stationary_psd'] = np.interp(f, f2, p2)
    return out


def estimate_and_validate(data, backend='reference', fit=True,
                          q_initial=1e-4, training_fraction=.60):
    y,u,dt,p,R = (data[k] for k in ('y','u','dt','p','R'))
    n=len(y)
    if n<80:raise ValueError('Need at least 80 samples to fit and validate')
    cut=max(60,min(n-30,int(n*training_fraction)))
    q = float(q_initial)
    fit_report = None
    if fit:
        # Cap identification work without changing the held-out period.
        stride = max(1,int(np.ceil(cut/1600)))
        train_y=y[:cut:stride]; train_u=u[:cut:stride]
        q,fit_report=estimate_q_by_likelihood(train_y,train_u,dt*stride,p,R)
    output=run_filter(y,u,dt,p,R,q,backend=backend)
    hold=run_filter(y[cut:],u[cut:],dt,p,R,q,backend=backend)
    truth=data.get('truth')
    metrics=validation_metrics(hold,truth[cut:] if truth is not None else None,first=min(25,len(hold['nis'])//10))
    return dict(result=output,heldout=hold,metrics=metrics,q=q,
                fit_report=fit_report,split=cut)
