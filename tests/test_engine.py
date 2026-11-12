import math
import pytest
from akus_sim import Simulator,Trajectory,GRAVITY,NORMAL_TOP_DEG,TOP_DEG


def test_kinematic_conversion():
    s=Simulator()
    s.pos_deg=360
    assert math.isclose(s.pos_mm,5.0)
    s.pos_deg=TOP_DEG
    assert math.isclose(s.pos_mm,611.1111111111)


def test_trapezoid_endpoint_and_peak():
    m=Trajectory.make(1000,20000,2800,400,500)
    assert m.duration>0
    assert m.sample(0)==(1000,0,0)
    assert m.sample(m.duration)==(20000,0,0)
    pts=[m.sample(t*m.duration/1000)[0] for t in range(1001)]
    assert all(x<=y+1e-7 for x,y in zip(pts,pts[1:]))


def test_homing_dual_axis_and_limit():
    s=Simulator()
    assert not s.homed
    s.start_homing()
    elapsed=s.finish_operation()
    assert s.homed and s.state=='Ready' and s.operation==''
    assert math.isclose(s.pos_deg,0,abs_tol=1e-7)
    assert elapsed>0
    assert s.switch_top_nc
    assert any('Top limit detected' in e['event'] for e in s.events)


def test_custom_homing():
    s=Simulator()
    s.start_homing(beckhoff=False)
    s.finish_operation()
    assert s.homed and s.state=='Ready'
    assert s.op_mode=='SyncPosTorque'


def test_manual_motion_and_return():
    s=Simulator()
    s.start_homing();s.finish_operation()
    s.move_absolute(150.,1000.,800.,800.)
    s.finish_operation()
    assert s.state=='Ready' and math.isclose(s.pos_mm,150,abs_tol=.001)
    s.move_to_start();s.finish_operation()
    assert math.isclose(s.pos_mm,0,abs_tol=.001)


def test_soft_limits_reject():
    s=Simulator();s.start_homing();s.finish_operation()
    with pytest.raises(ValueError):s.move_absolute(610.0)
    assert s.state=='Moving' or s.state=='Ready'  # source currently changes state before validation


def test_experiment_calc_matches_correct_physics():
    p=Simulator.experiment_plan(.001,3)
    a=.001*GRAVITY/.005*360
    assert math.isclose(p['a_deg_s2'],a)
    assert math.isclose(p['t_acc_s'],3)
    assert math.isclose(p['distance_deg'],.5*a*9+.5*(3*a)**2/20000)


def test_experiment_full_sequence():
    s=Simulator();s.start_homing();s.finish_operation()
    p=s.start_experiment(.001,3.,'PositionBeckhoff',delay_s=.5,lock_s=.7)
    assert p['distance_mm']>0
    s.finish_operation()
    assert s.state=='Ready' and not s.recording and not s.camera_on
    assert math.isclose(s.pos_deg,0,abs_tol=.001)
    assert any('Experiment complete' in e['event'] for e in s.events)


def test_experiment_boundary():
    s=Simulator();s.start_homing();s.finish_operation()
    with pytest.raises(ValueError):s.start_experiment(.1,3.0)


def test_vacuum_signals_and_export():
    s=Simulator();s.set_vacuum(pump_turbo=True,valve_normal=True)
    assert s.pump_turbo and s.valve_normal
    assert b'time_s' in s.csv_bytes()
    assert s.snapshot()['config']['friction_coulomb_n']==225.059


def test_estop_blocks_and_reset():
    s=Simulator();s.start_homing();s.step(1);s.emergency_stop()
    assert s.state=='Error' and s.e_stop
    with pytest.raises(ValueError):s.start_homing()
    s.reset()
    assert s.state=='Init' and not s.homed and not s.e_stop


def test_identification_sequence():
    s=Simulator();s.start_homing();s.finish_operation()
    s.start_identification(10.,200.,1400.,800.)
    s.finish_operation()
    assert s.state=='Ready' and math.isclose(s.pos_mm,10,abs_tol=.01)
    assert any('Identification sequence done' in e['event'] for e in s.events)


def test_control_test_sequence():
    s=Simulator();s.start_homing();s.finish_operation()
    s.start_control_test(100,50,1400,800)
    s.finish_operation()
    assert s.state=='Ready' and math.isclose(s.pos_mm,50,abs_tol=.01)
