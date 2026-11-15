"""Controlled simulations using Jan's AKUS physical parameters.

Modes
-----
linear: Exact discretization of the simplified, damped two-state physics, with
        independently generated continuous-white-acceleration process noise.
jan_euler: Source-aligned *force law* and explicit Euler step from the uploaded
        TwinCAT FB_AkusModel / FB_AccIntegrator (Euler case), plus the same
        artificial process noise. This is an offline numerical comparison,
        NOT the runtime PLC (no scheduler, NC or physical I/O).

Synthetic measurements: position z [m], *dynamic* acceleration [m/s^2], not
raw accelerometer specific-force (which would include gravity on Earth).
Gravity subtraction, calibration, and the sign convention are known exactly
for these synthetic measurements ONLY. Simulation truth is available by design.
"""
import numpy as np
from .jan_model import ModelParameters, matrices, continuous_white_acceleration_Q


def _input_profile(t, scenario):
    """Net drive force after gravity compensation, N; varied for holdout."""
    if scenario == 'training':
        return 390*np.sin(2*np.pi*t/4.0)+145*np.sin(2*np.pi*t/1.7)
    if scenario == 'validation':
        return 340*np.sin(2*np.pi*t/5.7 + 0.3)+185*np.sin(2*np.pi*t/2.3)
    if scenario == 'reversal':
        return 320*np.tanh(9*np.sin(2*np.pi*t/6.1))+110*np.sin(2*np.pi*t/1.3)
    raise ValueError('unknown excitation scenario')


def _jan_euler_step(z, v, net_drive, dt, p):
    """FB_AkusModel's source force/friction sign test and FB_AccIntegrator Euler.

    The PLC sets fResultingForce=0 if |driving|<|signed friction|, otherwise
    subtracts friction. It computes acceleration from the *previous* velocity
    and position; its Euler integrator updates position using the old velocity.
    Brake and elevator flags are intentionally disabled in this benchmark.
    """
    pos_term=p.friction_coulomb+p.friction_position*(z-p.z_reference)**2
    signed_friction=(np.sign(net_drive)*pos_term+p.b_viscous*v) if net_drive!=0 else 0.0
    resulting_force=(0.0 if abs(net_drive)<abs(signed_friction)
                     else net_drive-signed_friction)
    acc=resulting_force/p.effective_mass
    return z+dt*v, v+dt*acc, acc


def generate(dt=.02, N=1500, p=None, seed=10, sensor_std_z=0.00035,
             sensor_std_a=0.04, true_q=0.00004, plant='linear',
             scenario='training', calibration_count=2500):
    """Return known truth, noisy observations and *separate* calibration noise.

    Returns injected Q/R values so we can compare identification to ground
    truth. The calibration sensor samples are independent of dynamic samples.
    `nonlinear` remains an alias for v2 compatibility.
    """
    if p is None: p=ModelParameters()
    p.validate()
    if plant=='nonlinear': plant='jan_euler'
    if plant not in ('linear','jan_euler'):
        raise ValueError('plant must be linear or jan_euler')
    if N<30 or dt<=0 or true_q<0 or sensor_std_z<=0 or sensor_std_a<=0:
        raise ValueError('invalid sample, timing or noise parameters')
    rng=np.random.default_rng(seed)
    A,B,C,D,Ac=matrices(dt,p)
    Q=continuous_white_acceleration_Q(dt,Ac,true_q)
    time=np.arange(N)*dt
    u=_input_profile(time,scenario)
    total_spindle_force=u+p.m_platform*p.gravity
    total_spindle_torque_nm=total_spindle_force*p.pitch/(2*np.pi)
    true=np.zeros((N,2)); true[0,0]=p.z_reference
    acc_truth=np.zeros(N)
    for k in range(N):
        if k>0:
            if plant=='linear':
                # Exact sampled solution + known Brownian acceleration disturbance.
                true[k]=A@true[k-1]+B[:,0]*u[k-1]+rng.multivariate_normal([0.,0.],Q)
            else:
                z,v=true[k-1]
                zn,vn,_=_jan_euler_step(z,v,u[k-1],dt,p)
                true[k]=np.array([zn,vn])+rng.multivariate_normal([0.,0.],Q)
        if plant=='linear':
            acc_truth[k]=(u[k]-p.b_viscous*true[k,1])/p.effective_mass
        else:
            _,_,acc_truth[k]=_jan_euler_step(true[k,0],true[k,1],u[k],dt,p)
    R=np.diag([sensor_std_z**2,sensor_std_a**2])
    noise=rng.multivariate_normal([0,0],R,size=N)
    y=np.column_stack((true[:,0],acc_truth))+noise
    # These samples are from a controlled independent stationary noise test.
    calibration=rng.multivariate_normal([0,0],R,size=calibration_count)
    return dict(t=time,u=u,force=total_spindle_force,torque=total_spindle_torque_nm,
                y=y,truth=true,acc_truth=acc_truth,calibration_residuals=calibration,
                R_true=R,Q_true=Q,q_true=true_q,plant=plant,scenario=scenario)
