"""Offline AKUS twin-axis virtual test bench.

Project-derived constants and mode names are attributed to the user's TwinCAT PLC
project. This is a Python behavioral model, NOT an IEC ST interpreter, NC PTP
emulator, EtherCAT drive controller, or safety function. NO HARDWARE I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from collections import deque
import csv
import io
import math
from typing import Optional

LEAD_M = 0.005                 # ball screw pitch from project
DEG_TO_M = LEAD_M / 360.0
GRAVITY = 9.80665
TOP_DEG = 44000.0
NORMAL_TOP_DEG = TOP_DEG - 720.0    # MAIN experiment clearance guard
HOME_VEL = 700.0
HOME_RETURN_VEL = 4 * HOME_VEL
MAX_DEC = 20000.0

OP_MODES = {
    'SyncAccTorque': 6, 'SyncPosTorque': 7, 'SyncPos': 8,
    'SyncVel': 9, 'SyncTorq': 10, 'SyncTorqComm': 11,
}
STATES = ('Init', 'Ready', 'Moving', 'ToStart', 'CtrlConfig', 'Calib', 'Ident', 'Error')


@dataclass
class ControlConfig:
    # GVL_Persist.TcGVL (same initial values and units)
    pos_outer_kp: float = 10.0
    pos_outer_tn: float = 0.0
    pos_outer_tv: float = 0.0
    pos_inner_kp: float = 0.582
    pos_inner_tn: float = 0.015
    pos_inner_tv: float = 0.0
    pos_inner_limit_up: float = 100.0
    pos_inner_limit_low: float = -100.0
    acc_outer_kp: float = 10.0
    acc_outer_tn: float = 0.0
    acc_outer_tv: float = 0.0
    acc_inner_kp: float = 0.582
    acc_inner_tn: float = 0.015
    acc_inner_tv: float = 0.0
    acc_inner_limit_up: float = 100.0
    acc_inner_limit_low: float = -100.0
    torque_ramp: float = 1000.0
    velocity_limit_added: float = 0.1
    acceleration_limit_added: float = 0.1
    m_equivalent_kg: float = 4154.59
    m_platform_kg: float = 17.395
    friction_coulomb_n: float = 225.059
    friction_viscous_ns_m: float = 3994.857
    friction_position_n_m2: float = 1207.7
    friction_center_m: float = 0.249715
    acc_gradient: float = 1.0
    acc_offset_g: float = 0.0
    acc_use_offset: bool = False
    # Simulation-only assumptions, NOT extracted drive ratings:
    max_motor_torque_nm: float = 2.5
    simulated_position_kp_s2: float = 13.0
    simulated_velocity_kp_s: float = 11.0


@dataclass
class Trajectory:
    start_deg: float
    target_deg: float
    velocity_deg_s: float
    acceleration_deg_s2: float
    deceleration_deg_s2: float
    t_acc: float
    t_flat: float
    t_dec: float
    v_peak: float
    sign: int
    elapsed: float = 0.0
    kind: str = 'move'

    @classmethod
    def make(cls, start: float, target: float, vel: float, acc: float,
             dec: float, kind: str = 'move') -> 'Trajectory':
        if not all(math.isfinite(x) for x in (start, target, vel, acc, dec)):
            raise ValueError('Non-finite trajectory parameter')
        if vel <= 0 or acc <= 0 or dec <= 0:
            raise ValueError('Velocity, acceleration and deceleration must be > 0')
        delta = target - start
        sign = 1 if delta >= 0 else -1
        length = abs(delta)
        v_tri = math.sqrt(max(0, 2 * length * acc * dec / (acc + dec)))
        vpk = min(vel, v_tri)
        ta = vpk / acc
        td = vpk / dec
        ds = 0.5 * vpk * ta + 0.5 * vpk * td
        tf = max(0.0, (length - ds) / vpk) if vpk > 0 else 0.0
        return cls(start, target, vel, acc, dec, ta, tf, td, vpk, sign, kind=kind)

    @property
    def duration(self):
        return self.t_acc + self.t_flat + self.t_dec

    def sample(self, t: float) -> tuple[float, float, float]:
        """Analytic trapezoid, output [deg, deg/s, deg/s²]."""
        if t >= self.duration or self.duration < 1e-12:
            return self.target_deg, 0.0, 0.0
        if t <= 0:
            return self.start_deg, 0.0, 0.0
        a, d, v = self.acceleration_deg_s2, self.deceleration_deg_s2, self.v_peak
        if t < self.t_acc:
            distance = 0.5 * a * t**2
            vel = a * t
            acc = a
        elif t < self.t_acc + self.t_flat:
            td = t - self.t_acc
            distance = 0.5 * v * self.t_acc + v * td
            vel, acc = v, 0.0
        else:
            td = t - self.t_acc - self.t_flat
            distance = 0.5 * v * self.t_acc + v * self.t_flat + v * td - 0.5 * d * td**2
            vel, acc = max(0.0, v - d * td), -d
        return self.start_deg + self.sign * distance, self.sign * vel, self.sign * acc


@dataclass
class Simulator:
    config: ControlConfig = field(default_factory=ControlConfig)
    t_s: float = 0.0
    pos_deg: float = 0.0
    vel_deg_s: float = 0.0
    acc_deg_s2: float = 0.0
    torque1_nm: float = 0.0
    torque2_nm: float = 0.0
    set_pos_deg: float = 0.0
    set_vel_deg_s: float = 0.0
    set_acc_deg_s2: float = 0.0
    state: str = 'Init'
    phase: str = 'Unreferenced'
    op_mode: str = 'SyncPos'
    simulation_type: str = 'Ideal NC trajectory'
    homed: bool = False
    enabled: bool = False
    running: bool = False
    e_stop: bool = False
    fault: str = ''
    in_elevator: bool = False
    camera_on: bool = False
    recording: bool = False
    current: Optional[Trajectory] = None
    operation: str = ''
    next_step: str = ''
    wait_until: float = 0.0
    # Visual-only axis deviation to illustrate synchronization errors, default 0
    offset2_deg: float = 0.0
    logging: bool = True
    history: list = field(default_factory=list)
    events: list = field(default_factory=list)
    pump_turbo: bool = False
    pump_diaphragm: bool = False
    valve_big: bool = False
    valve_normal: bool = False
    valve_diaphragm: bool = False
    sensor_filter: bool = False
    filtered_acc_g: float = 1.0
    _last_log: float = -1000.0

    def __post_init__(self):
        self.event('Virtual bench initialized (OFFLINE, no EtherCAT/ADS writes)')
        self.log(force=True)

    @property
    def pos_mm(self):
        return self.pos_deg * 5.0 / 360.0

    @property
    def pos2_deg(self):
        return self.pos_deg + self.offset2_deg

    @property
    def vel_mm_s(self):
        return self.vel_deg_s * 5.0 / 360.0

    @property
    def acc_ms2(self):
        return self.acc_deg_s2 * DEG_TO_M

    @property
    def acc_g(self):
        # PCB accelerometer measured static +1g; in elevator zero-g convention differs.
        value = self.acc_ms2 / GRAVITY + (0.0 if self.in_elevator else 1.0)
        if self.config.acc_use_offset:
            value += self.config.acc_offset_g
        return value

    @property
    def switch_top_nc(self):
        return self.pos_deg < TOP_DEG - 1e-6

    @property
    def axis_deviation_deg(self):
        return self.offset2_deg

    def event(self, message: str):
        self.events.append({'time_s': round(self.t_s, 3), 'event': message})
        if len(self.events) > 1500:
            del self.events[:200]

    def log(self, force=False):
        if self.logging and (force or self.t_s - self._last_log >= 0.05 - 1e-8):
            self.history.append({
                'time_s': round(self.t_s, 4), 'state': self.state, 'phase': self.phase,
                'pos1_deg': self.pos_deg, 'pos2_deg': self.pos2_deg,
                'set_pos_deg': self.set_pos_deg, 'pos_mm': self.pos_mm,
                'vel1_deg_s': self.vel_deg_s, 'vel2_deg_s': self.vel_deg_s,
                'set_vel_deg_s': self.set_vel_deg_s, 'vel_mm_s': self.vel_mm_s,
                'acc_ms2': self.acc_ms2, 'acc_g': self.acc_g,
                'acc_filtered_g': self.filtered_acc_g,
                'set_acc_deg_s2': self.set_acc_deg_s2,
                'torque1_nm': self.torque1_nm, 'torque2_nm': self.torque2_nm,
                'top_switch_nc': self.switch_top_nc,
                'emergency_stop': self.e_stop, 'camera_on': self.camera_on,
                'recording': self.recording,
            })
            self._last_log = self.t_s
            if len(self.history) > 20000:
                del self.history[:2000]

    def clear_history(self):
        self.history.clear(); self._last_log = -1000; self.log(force=True)
        self.event('Measurement history cleared')

    def _require_ready(self):
        if self.e_stop or self.state == 'Error':
            raise ValueError('Fault latched. Reset first, then home the axes.')
        if not self.homed or self.state != 'Ready':
            raise ValueError('Homing required; controller must be READY.')
        if self.current or self.operation:
            raise ValueError('Another movement is in progress.')

    def reset(self):
        if self.running:
            raise ValueError('Stop motion before resetting the virtual controller.')
        self.e_stop = False
        self.fault = ''
        self.state = 'Init'
        self.phase = 'Unreferenced'
        self.homed = False
        self.enabled = False
        self.current = None
        self.operation = ''
        self.camera_on = False
        self.recording = False
        self.vel_deg_s = 0.0
        self.acc_deg_s2 = 0.0
        self.event('Faults reset; referencing required')
        self.log(force=True)

    def emergency_stop(self):
        self.e_stop = True
        self.fault = 'Virtual emergency stop'
        self.state = 'Error'
        self.phase = 'E-STOP latched'
        self.running = False
        self.enabled = False
        self.current = None
        self.operation = ''
        self.vel_deg_s = 0.0
        self.acc_deg_s2 = 0.0
        self.torque1_nm = self.torque2_nm = 0.0
        self.recording = self.camera_on = False
        self.event('VIRTUAL E-STOP pressed (not a physical safety stop)')
        self.log(force=True)

    def stop(self):
        if self.e_stop:
            return
        if self.current or self.operation:
            self.event(f'Virtual controlled stop; canceled {self.operation or self.current.kind}')
        self.current = None
        self.operation = ''
        self.next_step = ''
        self.vel_deg_s = 0.0
        self.acc_deg_s2 = 0.0
        self.set_vel_deg_s = self.set_acc_deg_s2 = 0.0
        self.set_pos_deg = self.pos_deg
        self.recording = self.camera_on = False
        self.state = 'Ready' if self.homed else 'Init'
        self.phase = 'Stopped by operator'
        self.running = False
        self.log(force=True)

    def _motion(self, target_deg, velocity, acceleration, deceleration, kind):
        upper = TOP_DEG if kind == 'homing-search' else NORMAL_TOP_DEG
        if not (0 <= target_deg <= upper):
            raise ValueError(f'Commanded position {target_deg:.1f}° outside simulated [0, {upper:.0f}°] soft bounds')
        self.current = Trajectory.make(self.pos_deg, target_deg, abs(velocity), abs(acceleration), abs(deceleration), kind)
        self.set_pos_deg = target_deg
        self.enabled = True
        self.running = self.current.duration > 0
        self.event(f'{kind}: {self.pos_deg:.1f}° → {target_deg:.1f}° @ {velocity:.0f}°/s')

    def start_homing(self, beckhoff=True):
        if self.e_stop:
            raise ValueError('Reset fault before homing')
        if self.current or self.operation:
            raise ValueError('Stop existing movement first')
        self.state = 'Init'
        self.homed = False
        self.operation = 'homing'
        self.op_mode = 'SyncPos' if beckhoff else 'SyncPosTorque'
        self.phase = '21 Enable / 22-23 upward search' if beckhoff else '3-5 torque search'
        self._motion(TOP_DEG, HOME_VEL, 200.0, 10000.0, 'homing-search')

    def move_absolute(self, position_mm: float, velocity_deg_s: float = 2800.0,
                      acceleration_deg_s2: float = 400.0,
                      deceleration_deg_s2: float = 500.0,
                      controller: str = 'SyncPos'):
        self._require_ready()
        if not (0 <= position_mm * 72 <= NORMAL_TOP_DEG):
            raise ValueError('Target is outside normal soft travel bounds')
        self.op_mode = controller
        self.state = 'Moving'
        self.phase = 'Manual positioning'
        self.operation = 'manual'
        self._motion(position_mm * 72.0, velocity_deg_s, acceleration_deg_s2,
                     deceleration_deg_s2, 'manual-absolute')

    def jog_for(self, direction: int, duration_s: float, velocity_deg_s: float):
        self._require_ready()
        if duration_s <= 0 or duration_s > 20:
            raise ValueError('Jog duration must be in (0, 20] seconds')
        if velocity_deg_s <= 0:
            raise ValueError('Jog velocity must be > 0')
        # Timed jog is simulated as a bounded point-to-point segment.
        distance = direction * velocity_deg_s * duration_s
        target = min(NORMAL_TOP_DEG, max(0, self.pos_deg + distance))
        if abs(target-self.pos_deg) < 1e-8:
            raise ValueError('Jog would violate a position boundary')
        self.state = 'Moving'
        self.phase = 'Timed jog (software approximation)'
        self.operation = 'manual'
        self._motion(target, velocity_deg_s, 10000.0, 10000.0, 'jog')

    def move_to_start(self):
        self._require_ready()
        self.state = 'ToStart'
        self.phase = 'Return to start'
        self.operation = 'to_start'
        self.op_mode = 'SyncPos'
        self._motion(0.0, HOME_RETURN_VEL, 200.0, 100.0, 'to-start')

    @staticmethod
    def experiment_plan(acceleration_g: float, acc_time_s: float,
                        max_deceleration_deg_s2: float = MAX_DEC,
                        total_duration_s: Optional[float] = None) -> dict:
        """MAIN.TcPOU experiment setpoint calculation, with corrected decel-distance term."""
        if acceleration_g <= 0 or not math.isfinite(acceleration_g):
            raise ValueError('Requested positive acceleration in g is required')
        if max_deceleration_deg_s2 <= 0:
            raise ValueError('Deceleration must be > 0')
        a = acceleration_g * GRAVITY / DEG_TO_M
        if total_duration_s is not None:
            if total_duration_s <= 0:
                raise ValueError('Total duration must be > 0')
            vmax = total_duration_s * a * max_deceleration_deg_s2 / (a + max_deceleration_deg_s2)
            acc_time_s = vmax / a
        if acc_time_s <= 0:
            raise ValueError('Acceleration duration must be > 0')
        vmax = a * acc_time_s
        dec_time = vmax / max_deceleration_deg_s2
        distance = .5 * a * acc_time_s**2 + .5 * max_deceleration_deg_s2 * dec_time**2
        return {
            'a_deg_s2': a, 'v_peak_deg_s': vmax, 't_acc_s': acc_time_s,
            't_dec_s': dec_time, 'total_duration_s': acc_time_s + dec_time,
            'distance_deg': distance, 'distance_mm': distance * 5 / 360,
            'max_dec_deg_s2': max_deceleration_deg_s2,
        }

    def start_experiment(self, acceleration_g: float, acc_time_s: float,
                         mode: str = 'PositionBeckhoff',
                         total_duration_s: Optional[float] = None,
                         delay_s: float = 0.0, lock_s: float = 8.0,
                         return_to_start: bool = True):
        self._require_ready()
        plan = self.experiment_plan(acceleration_g, acc_time_s, total_duration_s=total_duration_s)
        target = self.pos_deg + plan['distance_deg']
        if target > NORMAL_TOP_DEG:
            raise ValueError(f'Insufficient stroke. Need {plan["distance_mm"]:.1f} mm; top clearance ends at {NORMAL_TOP_DEG*5/360:.1f} mm')
        if mode not in ('Position', 'PositionBeckhoff', 'Acceleration'):
            raise ValueError('Unsupported experiment feedback mode')
        if self.in_elevator and plan['total_duration_s'] > 3.6:
            raise ValueError('TwinCAT fMaxDurElevator exceeded: 3.6 s')
        self.op_mode = {'Position':'SyncPosTorque', 'PositionBeckhoff':'SyncPos',
                        'Acceleration':'SyncAccTorque'}[mode]
        self.state = 'Moving'
        self.operation = 'experiment'
        self.phase = 'Trigger delay' if delay_s > 0 else 'Experiment accelerating/decelerating'
        self._exp_params = (target, plan, max(0.0, lock_s), return_to_start)
        self.camera_on = True
        self.recording = True
        if delay_s > 0:
            self.wait_until = self.t_s + delay_s
            self.next_step = 'experiment-up'
            self.running = True
        else:
            self._exp_up()
        return plan

    def _exp_up(self):
        target, plan, lock_s, do_return = self._exp_params
        self.phase = 'Experiment upward trajectory'
        self._motion(target, plan['v_peak_deg_s'], plan['a_deg_s2'],
                     plan['max_dec_deg_s2'], 'experiment-up')

    def start_identification(self, start_mm: float, end_mm: float, velocity_deg_s: float,
                             acceleration_deg_s2: float):
        self._require_ready()
        if not 0 <= start_mm <= NORMAL_TOP_DEG*5/360:
            raise ValueError('Identification start position exceeds range')
        if not 0 <= end_mm <= NORMAL_TOP_DEG*5/360:
            raise ValueError('Identification end position exceeds range')
        if start_mm == end_mm:
            raise ValueError('Identification needs nonzero displacement')
        self.operation = 'identification'
        self.state = 'Ident'
        self.phase = 'To identification start'
        self._ident_params = (start_mm*72, end_mm*72, velocity_deg_s, acceleration_deg_s2)
        self._motion(start_mm*72, velocity_deg_s, acceleration_deg_s2, acceleration_deg_s2, 'ident-to-start')

    def start_control_test(self, start_mm: float, end_mm: float, velocity_deg_s: float,
                           acceleration_deg_s2: float):
        self._require_ready()
        if not (0 <= start_mm <= NORMAL_TOP_DEG*5/360 and 0 <= end_mm <= NORMAL_TOP_DEG*5/360):
            raise ValueError('Controller test positions outside bounds')
        self.operation = 'control_test'
        self.state = 'CtrlConfig'
        self.phase = 'Control test: initial position'
        self._test_params = (start_mm*72, end_mm*72, velocity_deg_s, acceleration_deg_s2)
        self._motion(start_mm*72, velocity_deg_s, acceleration_deg_s2, acceleration_deg_s2, 'control-test-to-start')

    def _on_motion_done(self, kind: str):
        self.vel_deg_s = self.acc_deg_s2 = 0.0
        self.set_vel_deg_s = self.set_acc_deg_s2 = 0.0
        self.set_pos_deg = self.pos_deg
        self.running = False
        self.event(f'{kind} complete at {self.pos_mm:.2f} mm')
        if self.operation == 'homing':
            if kind == 'homing-search':
                self.phase = '24 TouchProbe + MC_Home'
                self.pos_deg = TOP_DEG
                self.event('Top limit detected (normally closed switch opens); position set to 44000°')
                self.phase = '25 Return to starting position'
                self._motion(0.0, HOME_RETURN_VEL, 200, 100, 'homing-return')
            else:
                self.operation = ''
                self.homed = True
                self.state = 'Ready'
                self.phase = 'Referenced / READY'
                self.event('Reference cycle complete; both axes READY')
        elif self.operation == 'experiment':
            if kind == 'experiment-up':
                self.phase = 'Lock after acceleration'
                _, _, lock_s, do_return = self._exp_params
                self.wait_until = self.t_s + lock_s
                self.next_step = 'experiment-return' if do_return else 'experiment-end'
                self.event(f'Experiment upward stage finished; lock {lock_s:.1f} s')
                self.running = True
            elif kind == 'experiment-return':
                self._finish_experiment()
        elif self.operation == 'identification':
            s,e,v,a = self._ident_params
            if kind == 'ident-to-start':
                self.phase = 'Identification measurement'
                self.recording = True
                self._motion(e, v, a, a, 'ident-measure')
            elif kind == 'ident-measure':
                self.recording = False
                self.phase = 'Identification return'
                self._motion(s, v, a, a, 'ident-return')
            else:
                self.operation = ''
                self.state = 'Ready'
                self.phase = 'Identification completed'
                self.event('Identification sequence done; CSV can be exported')
        elif self.operation == 'control_test':
            s,e,v,a = self._test_params
            if kind == 'control-test-to-start':
                self.phase = 'Controller response test'
                self.recording = True
                self._motion(e, v, a, a, 'control-test-measure')
            elif kind == 'control-test-measure':
                self.recording = False
                self.operation = ''
                self.state = 'Ready'
                self.phase = 'Controller test done'
                self.event('Controller response test completed')
        else:
            self.operation = ''
            self.state = 'Ready' if self.homed else 'Init'
            self.phase = 'Movement completed'

    def _finish_experiment(self):
        self.operation = ''
        self.recording = False
        self.camera_on = False
        self.state = 'Ready'
        self.phase = 'Experiment complete'
        self.event('Experiment complete; record export available')

    def _sample_torque(self, desired_a: float, desired_v: float):
        cfg = self.config
        # Torque estimate from the force balance, not a drive emulator
        z, v = self.pos_deg*DEG_TO_M, self.vel_deg_s*DEG_TO_M
        force = (cfg.m_equivalent_kg+cfg.m_platform_kg)*desired_a*DEG_TO_M
        if not self.in_elevator:
            force += cfg.m_platform_kg*GRAVITY
        direction = 1 if desired_v > 0 or (desired_v == 0 and desired_a > 0) else -1 if desired_v < 0 or desired_a < 0 else 0
        force += direction*(cfg.friction_coulomb_n + cfg.friction_position_n_m2*(z-cfg.friction_center_m)**2)
        force += cfg.friction_viscous_ns_m*v
        tau = force * LEAD_M / (4*math.pi)
        return tau, tau

    def _dynamic_step(self, dt, p_des, v_des, a_des):
        """Reference nonlinear plant; controller here is *not* exact TwinCAT PID/NC."""
        c = self.config
        z, v = self.pos_deg*DEG_TO_M, self.vel_deg_s*DEG_TO_M
        dp, dv, da = p_des*DEG_TO_M, v_des*DEG_TO_M, a_des*DEG_TO_M
        cmd_acc = da + c.simulated_position_kp_s2*(dp-z) + c.simulated_velocity_kp_s*(dv-v)
        mass = c.m_equivalent_kg+c.m_platform_kg
        if mass <= 0:
            raise ValueError('Equivalent mass + platform mass must be > 0')
        direction = 1 if dv>0 else -1 if dv<0 else 1 if cmd_acc>0 else -1 if cmd_acc<0 else 0
        coulomb = c.friction_coulomb_n+c.friction_position_n_m2*(z-c.friction_center_m)**2
        gravity = 0.0 if self.in_elevator else c.m_platform_kg*GRAVITY
        force_cmd = mass*cmd_acc + gravity + direction*coulomb + c.friction_viscous_ns_m*v
        tau = min(c.max_motor_torque_nm,max(-c.max_motor_torque_nm,force_cmd*LEAD_M/(4*math.pi)))
        driving = 2*tau*2*math.pi/LEAD_M-gravity
        friction = (coulomb if driving>0 else -coulomb if driving<0 else 0) + c.friction_viscous_ns_m*v
        residual = 0.0 if abs(driving)<abs(friction) and abs(v)<1e-5 else driving-friction
        self.acc_deg_s2 = residual/mass/DEG_TO_M
        self.vel_deg_s += self.acc_deg_s2*dt
        self.pos_deg += self.vel_deg_s*dt
        self.torque1_nm = self.torque2_nm = tau

    def step(self, dt: float):
        if dt <= 0:
            return
        if dt > 1000:
            raise ValueError('Requested single advancement is too long')
        n = max(1, math.ceil(dt/0.025))
        h = dt/n
        for _ in range(n):
            self._step_one(h)

    def _step_one(self, dt):
        self.t_s += dt
        if self.e_stop:
            self.log()
            return
        if self.current:
            m = self.current
            m.elapsed = min(m.elapsed+dt, m.duration)
            p,v,a = m.sample(m.elapsed)
            self.set_pos_deg, self.set_vel_deg_s, self.set_acc_deg_s2 = p,v,a
            if self.simulation_type == 'Dynamic force model':
                self._dynamic_step(dt,p,v,a)
            else:
                self.pos_deg = p
                self.vel_deg_s = v
                self.acc_deg_s2 = a
                self.torque1_nm,self.torque2_nm=self._sample_torque(a,v)
            # Homing-search simulated TouchProbe at the upper stop. The ideal NC
            # mode reaches it exactly; in the dynamic mode trigger on crossing.
            if m.kind == 'homing-search' and self.pos_deg >= TOP_DEG:
                self.pos_deg = TOP_DEG
                self.vel_deg_s = self.acc_deg_s2 = 0.0
                self.current = None
                self._on_motion_done(m.kind)
                self.filtered_acc_g += min(1.0, dt*12)*(self.acc_g-self.filtered_acc_g)
                self.log()
                return
            # Physical limits in this offline virtual bench. No hardware safety claims.
            if self.pos_deg < -0.01 or self.pos_deg > TOP_DEG + 0.01:
                self.fault = 'Virtual travel limit exceeded'
                self.emergency_stop()
                return
            if m.elapsed >= m.duration:
                if self.simulation_type == 'Dynamic force model' and (abs(self.pos_deg-m.target_deg)>0.25 or abs(self.vel_deg_s)>0.5):
                    # Continue tracking final setpoint, while enforcing timeout.
                    if m.elapsed - m.duration < 5.0:
                        pass
                    # Intentionally finalize by ideal NC target to bound tracking mismatch;
                    # must not be mistaken for physical validation.
                self.pos_deg = m.target_deg
                self.vel_deg_s = self.acc_deg_s2 = 0.0
                self.current = None
                self._on_motion_done(m.kind)
        elif self.next_step and self.t_s >= self.wait_until:
            action = self.next_step
            self.next_step = ''
            if action == 'experiment-up':
                self._exp_up()
            elif action == 'experiment-return':
                self.phase = 'Automatic experiment return'
                self.op_mode = 'SyncPosTorque'
                self._motion(0.0, 3*HOME_VEL, 400, 500, 'experiment-return')
            elif action == 'experiment-end':
                self._finish_experiment()
        # Mimic low-pass sensor display, not the shipped EL3702/TcFilter signal chain.
        self.filtered_acc_g += min(1.0, dt*12)*(self.acc_g-self.filtered_acc_g)
        self.log()

    def fast_forward(self, seconds: float):
        self.step(float(seconds))

    def finish_operation(self, max_seconds: float = 300.0):
        """Discrete deterministic run, useful for the demo and numerical tests."""
        elapsed = 0.0
        while (self.current is not None or self.next_step or self.running or self.operation) and elapsed < max_seconds:
            self.step(0.1)
            elapsed += 0.1
        if elapsed >= max_seconds:
            raise ValueError('Fast-forward hit its configured maximum simulated duration')
        return elapsed

    def set_vacuum(self, pump_turbo=None, pump_diaphragm=None,
                   valve_big=None, valve_normal=None, valve_diaphragm=None):
        for k,v in locals().copy().items():
            if k not in ('self',) and v is not None:
                setattr(self,k,bool(v))
                self.event(f'Virtual vacuum {k} → {"ON" if v else "OFF"}')

    def csv_bytes(self) -> bytes:
        if not self.history:
            return b''
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(self.history[0].keys()))
        writer.writeheader()
        writer.writerows(self.history)
        return output.getvalue().encode('utf-8-sig')

    def snapshot(self):
        return {
            'time_s': round(self.t_s,3), 'state': self.state,'phase': self.phase,
            'op_mode': self.op_mode,'simulation_type': self.simulation_type,
            'homed': self.homed,'enabled': self.enabled,'e_stop': self.e_stop,
            'pos1_deg': self.pos_deg,'pos2_deg': self.pos2_deg,
            'pos_mm': self.pos_mm,'vel_deg_s': self.vel_deg_s,
            'vel_mm_s': self.vel_mm_s,'acc_ms2':self.acc_ms2,
            'acc_g':self.acc_g,'torque1_nm':self.torque1_nm,
            'torque2_nm':self.torque2_nm,'top_probe_nc':self.switch_top_nc,
            'running': self.running,'fault':self.fault,
            'config':asdict(self.config),
        }
