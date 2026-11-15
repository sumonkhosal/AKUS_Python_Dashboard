"""AKUS offline virtual HMI, built with Streamlit.

Run: streamlit run app.py

The hardware interface is deliberately NOT connected: actions are simulations only.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from kalman_lab import (
    load_twincat_csv, make_real_observations, make_simulation_observations,
    analyze_frequency, estimate_and_validate,
)
from akuskf.benchmark import run_benchmark, report_json_ready
from akuskf import ModelParameters
from akuskf.acceleration import (
    run_acceleration_filter, fit_acceleration_q, estimate_acceleration_r,
    jan_linear_acceleration_prior, acceleration_diagnostics
)
from akuskf.synthetic import generate as generate_jan_synthetic
from kalman_lab import _decimate
from akus_sim import (
    Simulator, ControlConfig, OP_MODES, TOP_DEG, NORMAL_TOP_DEG,
    HOME_VEL, MAX_DEC, DEG_TO_M,
)

ROOT=Path(__file__).parent
st.set_page_config(page_title='AKUS | Virtual Control Bench', page_icon='⚙️', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
  .stApp {background: #0b1220;color: #e6edf7;}
  [data-testid="stSidebar"] {background:#101b2b;}
  h1,h2,h3 {color:#ecf4ff;letter-spacing:-.02em;}
  .block-container {padding-top:1.1rem; max-width:1580px;}
  div[data-testid="stMetric"] {border:1px solid #263449;border-radius:11px;padding:11px;background:#131f30;}
  div[data-testid="stMetricLabel"] {color:#a9bbd0;}
  .stButton button[kind="primary"] {background:#1f78cb;}
  .kicker {color:#67cbfd;font-size:.75rem;font-weight:700;letter-spacing:.13em;}
  .muted {color:#9eb2c9;font-size:.88rem;}
  .offline {background:#163a39;color:#8af2cc;padding:5px 10px;border-radius:100px;font-size:.74rem;font-weight:800;display:inline-block;}
</style>''', unsafe_allow_html=True)

if 'sim' not in st.session_state:
    st.session_state.sim = Simulator()
if 'live_clock' not in st.session_state:
    st.session_state.live_clock = True
if 'wall_clock' not in st.session_state:
    st.session_state.wall_clock = time.monotonic()
if 'speed' not in st.session_state:
    st.session_state.speed = 12

sim: Simulator=st.session_state.sim

def invoke(action, *args, **kwargs):
    try:
        result = getattr(sim,action)(*args,**kwargs)
        st.session_state.last_message = f'Action accepted in simulator: {action}'
        return result
    except (ValueError, TypeError) as ex:
        st.session_state.last_message = f'⚠ {ex}'
        return None

def carrier_svg(position_mm, top_mm=TOP_DEG*5/360, skew=0):
    """Twin-axis simplified mechanism; live vertical position, not a CAD reconstruction."""
    pos_ratio=min(1,max(0,position_mm/top_mm))
    y_top,y_bottom=42,395
    y=y_bottom - pos_ratio*(y_bottom-y_top)
    y_right=y-max(-17,min(17,skew/30))
    return f'''<!doctype html><html><head><style>
    body{{background:transparent;margin:0; font-family:Inter,Segoe UI,sans-serif; color:#d1e2f4;}}
    .lbl{{font-size:13px;fill:#bfd1e8;}} .tiny{{font-size:11px;fill:#8da2ba;}}
    </style></head><body><svg width="100%" viewBox="0 0 475 472" xmlns="http://www.w3.org/2000/svg">
    <defs><linearGradient id="steel" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#72879b"/><stop offset=".52" stop-color="#e2e8ee"/><stop offset="1" stop-color="#6e839a"/></linearGradient>
    <linearGradient id="pl" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#92cced"/><stop offset="1" stop-color="#2f5875"/></linearGradient></defs>
    <rect x="65" y="17" width="342" height="430" rx="15" fill="#101b2a" stroke="#2c455d"/>
    <rect x="98" y="24" width="20" height="402" rx="4" fill="url(#steel)"/>
    <rect x="354" y="24" width="20" height="402" rx="4" fill="url(#steel)"/>
    <line x1="130" x2="130" y1="32" y2="419" stroke="#cba43b" stroke-width="8"/>
    <line x1="342" x2="342" y1="32" y2="419" stroke="#cba43b" stroke-width="8"/>
    <g opacity=".2">{''.join(f'<line x1="130" x2="342" y1="{i}" y2="{i}" stroke="#8ba5b9" stroke-dasharray="5 7"/>' for i in range(52,415,45))}</g>
    <rect x="99" y="28" width="18" height="13" rx="2" fill="#6fe1a2"/>
    <rect x="355" y="28" width="18" height="13" rx="2" fill="#6fe1a2"/>
    <rect x="99" y="410" width="18" height="16" rx="2" fill="#c87979"/>
    <rect x="355" y="410" width="18" height="16" rx="2" fill="#c87979"/>
    <g><rect x="109" y="{y-17:.1f}" width="32" height="35" rx="4" fill="#c8dae5"/>
       <rect x="330" y="{y_right-17:.1f}" width="32" height="35" rx="4" fill="#c8dae5"/>
       <path d="M141 {y-9:.1f} L330 {y_right-9:.1f} L330 {y_right+9:.1f} L141 {y+9:.1f}Z" fill="url(#pl)" stroke="#b2e7f7" stroke-width="2"/>
       <path d="M209 {y-9:.1f} L209 {y-55:.1f} Q236 {y-73:.1f} 265 {y-55:.1f} L265 {y_right-9:.1f}" fill="#8fa7b8" stroke="#d6edf6" stroke-width="2"/>
       <rect x="190" y="{y+10:.1f}" width="90" height="14" rx="3" fill="#829cad"/>
       <circle cx="237" cy="{y-31:.1f}" r="6" fill="#df6d56"/></g>
    <g><rect x="91" y="431" width="34" height="24" rx="5" fill="#223b60" stroke="#477ec0"/>
      <rect x="348" y="431" width="34" height="24" rx="5" fill="#223b60" stroke="#477ec0"/>
      <text x="108" y="448" font-size="10" text-anchor="middle" fill="white">M1</text>
      <text x="365" y="448" font-size="10" text-anchor="middle" fill="white">M2</text></g>
    <text x="386" y="32" class="tiny">TOP</text><text x="386" y="419" class="tiny">START</text>
    <line x1="412" x2="412" y1="39" y2="402" stroke="#415f7e"/>
    <circle cx="412" cy="{y:.1f}" r="7" fill="#4fcbef" stroke="#d1ffff"/>
    <text x="237" y="466" text-anchor="middle" class="lbl">SIMULATED CARRIER · {position_mm:.1f} mm</text>
    </svg></body></html>'''

def plot_history(df, charts=('pos_mm','vel_mm_s','acc_g'), scope_title='AKUS recorded signals'):
    fig=make_subplots(rows=len(charts),cols=1,shared_xaxes=True,vertical_spacing=.09,
                      subplot_titles=[{'pos_mm':'Position [mm]','vel_mm_s':'Velocity [mm/s]','acc_g':'Acceleration [g]',
                                       'torque1_nm':'Motor torque [Nm]','pos1_deg':'Encoder position [deg]','set_pos_deg':'Setpoint [deg]'}
                                      .get(x,x) for x in charts])
    palette=['#5ac8fa','#7ce3ab','#ffbb66','#db9bff']
    for i,c in enumerate(charts,1):
        if c not in df.columns: continue
        fig.add_trace(go.Scatter(x=df['time_s'],y=df[c],mode='lines',name=c,line=dict(color=palette[(i-1)%4],width=2)),row=i,col=1)
        if c=='pos_mm':
            fig.add_hline(y=TOP_DEG*5/360,line_dash='dot',line_color='#ed7a82',row=i,col=1)
    fig.update_layout(title=scope_title,template='plotly_dark',height=max(350,230*len(charts)),
                      margin=dict(l=30,r=16,t=68,b=30),paper_bgcolor='#101b2a',plot_bgcolor='#101b2a',
                      showlegend=False,hovermode='x unified')
    fig.update_xaxes(title='Simulation time [s]',gridcolor='#213146')
    fig.update_yaxes(gridcolor='#213146',zerolinecolor='#3a4f65')
    return fig

with st.sidebar:
    st.markdown('### <span style="color:#8ed6fa">AKUS</span> Control Studio',unsafe_allow_html=True)
    st.markdown('<span class="offline">● SIMULATION ONLY</span>',unsafe_allow_html=True)
    st.caption('Derived from the uploaded TwinCAT 3 PLC / Visualization / Scope project')
    st.divider()
    sim.simulation_type=st.selectbox('Numerical plant', ['Ideal NC trajectory','Dynamic force model'],
                                    index=['Ideal NC trajectory','Dynamic force model'].index(sim.simulation_type),
                                    help='Dynamic model reproduces the force equation with assumed feedback gains and torque rating. Ideal mode follows an analytic trajectory.')
    st.session_state.speed=st.slider('Simulation speed × real time',1,60,st.session_state.speed,1)
    st.session_state.live_clock=st.toggle('Live simulation clock',value=st.session_state.live_clock)
    sim.in_elevator=st.toggle('Elevator mode (zero gravitational term)',value=sim.in_elevator)
    st.caption('Run to completion calculates a full motion immediately without waiting for the animated playback.')
    c1,c2=st.columns(2)
    with c1:
        if st.button('⏹ Stop',use_container_width=True): invoke('stop')
    with c2:
        if st.button('⚠ E-STOP',use_container_width=True,type='primary'): invoke('emergency_stop')
    if st.button('Reset virtual controller',use_container_width=True): invoke('reset')
    if st.button('Run sequence to completion',use_container_width=True):
        invoke('finish_operation',300.0)
    if st.button('Advance +5 simulated seconds',use_container_width=True): invoke('fast_forward',5.0)
    st.divider()
    st.caption('Source mechanism illustration')
    st.image(str(ROOT/'assets'/'linear_axes_carrier.png'),use_container_width=True)
    st.caption('Important: no safety PLC, drives, I/O or real emergency stop are controlled from this app.')

st.markdown('<div class="kicker">INDUSTRIAL MOTION · OFFLINE DIGITAL TWIN</div>',unsafe_allow_html=True)
st.title('AKUS · Two-Axis Carrier Control')
st.markdown('<div class="muted">TwinCAT-inspired Python HMI · positions / velocity / acceleration / model / tests / data</div>',unsafe_allow_html=True)
if 'last_message' in st.session_state:
    msg=st.session_state.last_message
    if msg.startswith('⚠'):st.warning(msg)
    else:st.caption(msg)

@st.fragment(run_every='250ms')
def live_monitor():
    now=time.monotonic()
    elapsed=now-st.session_state.wall_clock
    st.session_state.wall_clock=now
    if st.session_state.live_clock and (sim.current or sim.next_step or sim.operation):
        sim.step(max(0.0,min(elapsed,0.6))*st.session_state.speed)
    cols=st.columns(6)
    labels=[('STATE',sim.state),('HEIGHT',f'{sim.pos_mm:.1f} mm'),('VELOCITY',f'{sim.vel_mm_s:.2f} mm/s'),
            ('ACCELERATION',f'{sim.acc_g:.3f} g'),('AXIS SKEW',f'{sim.axis_deviation_deg:.1f}°'),('TIME',f'{sim.t_s:.1f} s')]
    for c,(label,val) in zip(cols,labels):
        c.metric(label,val)
    left,right=st.columns([.89,1.11],gap='large')
    with left:
        components.html(carrier_svg(sim.pos_mm,skew=sim.offset2_deg),height=455,scrolling=False)
    with right:
        status={
            'Operational state':sim.state, 'Active step':sim.phase,
            'Mode (E_OpMode)':f'{sim.op_mode} = {OP_MODES.get(sim.op_mode,"—")}',
            'Axis 1 [deg]':f'{sim.pos_deg:,.2f}',
            'Axis 2 [deg]':f'{sim.pos2_deg:,.2f}',
            'Upper TouchProbe': 'CLOSED (normal)' if sim.switch_top_nc else 'OPEN (actuated)',
            'Reference': 'Done' if sim.homed else 'Required',
            'Motors': 'Enabled (virtual)' if sim.enabled else 'Disabled',
            'Recording': 'ON' if sim.recording else 'OFF',
            'Camera trigger': 'ON' if sim.camera_on else 'OFF',
        }
        st.dataframe(pd.DataFrame([{'Signal':k,'Value':v} for k,v in status.items()]),hide_index=True,use_container_width=True,height=388)
    if sim.fault:st.error(sim.fault)
    return

live_monitor()

tabs=st.tabs(['Experiment','Manual motion','Controller tuning','Identification','Vacuum','Analysis / Scopes','Diagnostics','Project reference','Kalman research','Simulation Q/R benchmark','Acceleration KF'])

with tabs[0]:
    st.subheader('Automatic acceleration experiment · Flug')
    st.caption('Calculation follows MAIN.TcPOU; the alternate total-time formula is corrected to the intended trapezoidal distance.')
    with st.form('experiment'):
        a,b,c=st.columns(3)
        with a:
            exp_g=st.number_input('Set acceleration [g]',min_value=.00001,max_value=5.0,value=.001,format='%.5f')
            exp_mode=st.selectbox('Feedback control',('PositionBeckhoff','Position','Acceleration'))
            duration_mode=st.radio('Time specification',('Acceleration time','Total acceleration + braking time'))
        with b:
            exp_t=st.number_input('Requested duration [s]',min_value=.01,max_value=60.0,value=3.0,step=.1)
            exp_delay=st.number_input('Trigger delay [s]',min_value=0.,max_value=10.,value=0.,step=.1)
            exp_lock=st.number_input('Brake/lock duration [s]',min_value=0.,max_value=20.,value=8.,step=.5)
        with c:
            return_start=st.checkbox('Return automatically to start',True)
            st.markdown('**Travel guard:** motion must remain below `44000° - 720°`.')
            st.markdown('**Elevator guard:** total motion duration ≤ 3.6 s.')
        submitted=st.form_submit_button('Start simulated experiment',type='primary')
    try:
        plan=sim.experiment_plan(exp_g,exp_t,
            total_duration_s=(exp_t if duration_mode.startswith('Total') else None))
        cp=st.columns(4)
        for cc,(n,v) in zip(cp,[('Peak speed',f'{plan["v_peak_deg_s"]:.1f} °/s'),
                 ('Stroke needed',f'{plan["distance_mm"]:.2f} mm'),
                 ('Accel phase',f'{plan["t_acc_s"]:.3f} s'),
                 ('Decel phase',f'{plan["t_dec_s"]:.3f} s')]): cc.metric(n,v)
        if submitted: invoke('start_experiment',exp_g,exp_t,exp_mode,
            total_duration_s=(exp_t if duration_mode.startswith('Total') else None),
            delay_s=exp_delay,lock_s=exp_lock,return_to_start=return_start)
    except ValueError as e: st.warning(str(e))
    st.divider()
    x,y=st.columns(2)
    with x:
        st.markdown('#### Referencing / initialization')
        homing_mode=st.radio('Reference strategy',('Beckhoff NC · SyncPos (20–25)', 'Custom cascade · SyncPosTorque (1–11)'))
        if st.button('Home both axes (simulation)',use_container_width=True):
            invoke('start_homing',beckhoff=homing_mode.startswith('Beckhoff'))
    with y:
        st.markdown('#### Start position')
        st.write('The original procedure searches the upper limit, assigns 44,000°, then returns to 0°.')
        if st.button('Move carrier to 0 mm',use_container_width=True):invoke('move_to_start')

with tabs[1]:
    st.subheader('Manual operation · Einzelbewegung / Jog')
    st.caption('Position commands use the original motor-degree convention: 360° motor rotation = 5 mm travel.')
    with st.form('move'):
        m1,m2,m3=st.columns(3)
        with m1: pos_mm=st.number_input('Target carrier position [mm]',min_value=0.0,max_value=float(NORMAL_TOP_DEG*5/360),value=100.0,step=5.)
        with m2: move_v=st.number_input('Velocity [°/s]',min_value=1.,max_value=40000.,value=2800.,step=100.)
        with m3: controller=st.selectbox('Motion backend',('SyncPos','SyncPosTorque','SyncVel'))
        m4,m5=st.columns(2)
        with m4: move_a=st.number_input('Acceleration [°/s²]',min_value=1.,max_value=100000.,value=400.)
        with m5: move_d=st.number_input('Deceleration [°/s²]',min_value=1.,max_value=100000.,value=500.)
        if st.form_submit_button('Move to position',type='primary'):
            invoke('move_absolute',pos_mm,move_v,move_a,move_d,controller)
    st.markdown('#### Bounded timed jog')
    j1,j2,j3=st.columns(3)
    with j1: jog_vel=st.number_input('Jog velocity [°/s]',min_value=1.,value=700.)
    with j2: jog_dur=st.number_input('Jog duration [s]',min_value=.05,max_value=20.,value=1.,step=.25)
    with j3:
        if st.button('↑ Jog up',use_container_width=True):invoke('jog_for',1,jog_dur,jog_vel)
        if st.button('↓ Jog down',use_container_width=True):invoke('jog_for',-1,jog_dur,jog_vel)
    st.caption('Timed jog is deliberately bounded; this is not MC_Jog dead-man hardware behavior.')
    st.markdown('#### Synchronization illustration')
    sim.offset2_deg=st.slider('Inject Axis 2 position offset [°] (visual-only)',-50.,50.,float(sim.offset2_deg),1.)
    if abs(sim.offset2_deg)>40:st.warning('Simulated axis mismatch exceeds GVL position synchronization tolerance (40°).')

with tabs[2]:
    st.subheader('ControlConfig · PID + mechanical plant parameters')
    st.caption('Initial numerical values were read from GVL_Persist.TcGVL. Tuning affects the dynamic plant approximation only; the ideal trajectory ignores plant feedback gains.')
    cfg=asdict(sim.config)
    clusters={
        'Position cascade':[k for k in cfg if k.startswith('pos_')],
        'Acceleration cascade':[k for k in cfg if k.startswith('acc_')],
        'Torque limits':[k for k in cfg if k.startswith('torque_') or k.endswith('_added')],
        'Plant physics':[k for k in cfg if k.startswith('m_') or k.startswith('friction_')],
        'Acceleration calibration':[k for k in cfg if k.startswith('acc_gradient') or k.startswith('acc_offset')],
        'Offline-only feedback assumptions':[k for k in cfg if k.startswith('max_motor') or k.startswith('simulated_')],
    }
    group=st.selectbox('Parameter group',list(clusters))
    keys=clusters[group]
    edited=st.data_editor(pd.DataFrame([{'Parameter':k,'Value':float(cfg[k])} for k in keys]),
            column_config={'Parameter':st.column_config.TextColumn(disabled=True),
                           'Value':st.column_config.NumberColumn(format='%.6f')},
            hide_index=True,use_container_width=True,key='ed_'+group)
    q1,q2=st.columns(2)
    with q1:
        if st.button('Apply parameter changes',use_container_width=True):
            if sim.current or sim.operation:
                st.warning('Stop the virtual machine before editing plant parameters.')
            else:
                try:
                    changes={r['Parameter']:float(r['Value']) for _,r in edited.iterrows()}
                    if not all(math.isfinite(v) for v in changes.values()):raise ValueError('Parameters must be finite')
                    if any(v <= 0 for k,v in changes.items() if k in ('m_equivalent_kg','m_platform_kg','max_motor_torque_nm')):
                        raise ValueError('Masses and motor torque limit must be > 0')
                    for k,v in changes.items():setattr(sim.config,k,v)
                    sim.event('Applied virtual parameter group: '+group)
                    st.success('Virtual parameters updated')
                except ValueError as e:st.warning(str(e))
    with q2:
        st.download_button('Export parameters JSON',data=json.dumps(asdict(sim.config),indent=2),file_name='akus_parameters.json',mime='application/json',use_container_width=True)
    upload=st.file_uploader('Import parameter JSON (offline only)',type='json')
    if upload and st.button('Load JSON into simulator'):
        try:
            data=json.load(upload)
            allowed=asdict(sim.config)
            for k,v in data.items():
                if k in allowed and isinstance(v,(float,int)) and math.isfinite(v):setattr(sim.config,k,float(v))
            sim.event('Control configuration imported from JSON')
            st.success('Parameters loaded; inspect values before running another test')
        except (ValueError,TypeError) as e:st.error(str(e))
    st.divider()
    st.markdown('#### Simulate controller response / configuration test')
    with st.form('control_test'):
        ct1,ct2,ct3,ct4=st.columns(4)
        with ct1:ct_start=st.number_input('Start [mm]',value=550.,min_value=0.,max_value=600.,key='ct_start')
        with ct2:ct_end=st.number_input('End [mm]',value=50.,min_value=0.,max_value=600.,key='ct_end')
        with ct3:ct_vel=st.number_input('Velocity [°/s]',value=2000.,min_value=1.,key='ct_vel')
        with ct4:ct_acc=st.number_input('Acceleration [°/s²]',value=1000.,min_value=1.,key='ct_acc')
        if st.form_submit_button('Run offline controller test'):invoke('start_control_test',ct_start,ct_end,ct_vel,ct_acc)
    st.image(str(ROOT/'assets'/'force_model.png'),caption='Source project: AKUS dynamical force diagram',use_container_width=False,width=560)

with tabs[3]:
    st.subheader('Parameter identification · IdentTrajectory')
    st.caption('Moves to start, records an excitation trajectory, returns to start and supports CSV export. Not a full replica of every parameter-identification state in MAIN.TcPOU.')
    with st.form('ident_form'):
        q1,q2,q3,q4=st.columns(4)
        with q1:id_start=st.number_input('Start [mm]',0.,600.,value=50.,key='id_start')
        with q2:id_end=st.number_input('End [mm]',0.,600.,value=500.,key='id_end')
        with q3:id_v=st.number_input('Velocity [°/s]',min_value=1.,value=2400.,key='id_v')
        with q4:id_a=st.number_input('Acceleration [°/s²]',min_value=1.,value=1500.,key='id_a')
        if st.form_submit_button('Run simulated identification',type='primary'):
            invoke('start_identification',id_start,id_end,id_v,id_a)
    st.markdown('**Plant equations**')
    st.latex(r'F_{motor}= (T_1+T_2)\frac{2\pi}{h},\quad F_{drive}=F_{motor}-m_p g')
    st.latex(r'F_{friction}=\operatorname{sgn}(F_{drive})\left[B_c+B_z(z-z_0)^2\right]+B_v\dot z')
    st.latex(r'\ddot z=\frac{F_{drive}-F_{friction}}{m_{ers}+m_p}')
    st.info('Torque traces are inferred, not actual measured torque signals. Dynamic-force mode uses a simplified feedback controller and should not be used to predict real stability.')

with tabs[4]:
    st.subheader('Vacuum test panel · Ventilsteuerung')
    st.caption('Offline signal-state panel based on MAIN vacuum I/O symbols. No pump, valve, interlock or pressure sensor is connected.')
    left,right=st.columns([1.1,1])
    with left:
        st.image(str(ROOT/'assets'/'vacuum_valves.png'),use_container_width=True,caption='Valve connections from source project')
    with right:
        with st.form('vacuum_controls'):
            p1=st.checkbox('Turbo pump',value=sim.pump_turbo)
            p2=st.checkbox('Diaphragm pump',value=sim.pump_diaphragm)
            v1=st.checkbox('Big valve',value=sim.valve_big)
            v2=st.checkbox('Normal valve',value=sim.valve_normal)
            v3=st.checkbox('Diaphragm valve',value=sim.valve_diaphragm)
            if st.form_submit_button('Apply virtual I/O states'):
                sim.set_vacuum(pump_turbo=p1,pump_diaphragm=p2,
                               valve_big=v1,valve_normal=v2,valve_diaphragm=v3)
        st.dataframe(pd.DataFrame([
            {'Sensor':'bBigValveOpened','Value':sim.valve_big},
            {'Sensor':'bBigValveClosed','Value':not sim.valve_big},
            {'Sensor':'bNormalValveOpened','Value':sim.valve_normal},
            {'Sensor':'bNormalValveClosed','Value':not sim.valve_normal},
            {'Sensor':'bDiaphragmValveOpened','Value':sim.valve_diaphragm},
            {'Sensor':'bDiaphragmValveClosed','Value':not sim.valve_diaphragm},
        ]),hide_index=True,use_container_width=True)
        st.warning('Valve feedback is synthetically mirrored from commands. No vacuum dynamics or real end switches are modeled.')

with tabs[5]:
    st.subheader('Measurement & analysis · TwinCAT Scope equivalents')
    presets=json.loads((ROOT/'reference'/'scope_catalog.json').read_text(encoding='utf8'))
    scope_name=st.selectbox('Original TwinCAT Scope preset', list(presets))
    with st.expander('Original Scope variables included in this preset'):
        st.write(', '.join(presets[scope_name]) or '(No channels in source XML)')
    df=pd.DataFrame(sim.history)
    if len(df)>1:
        window=st.slider('Recent simulation history [seconds]',10,300,60,10)
        df=df[df['time_s']>=df['time_s'].max()-window]
        chart_choices={'Position':'pos_mm','Velocity':'vel_mm_s','Acceleration':'acc_g','Torque':'torque1_nm','Motor position':'pos1_deg','Setpoint':'set_pos_deg'}
        chart_selection=st.multiselect('Signals to plot',list(chart_choices),default=['Position','Velocity','Acceleration'])
        if chart_selection:
            st.plotly_chart(plot_history(df,[chart_choices[x] for x in chart_selection],scope_name),use_container_width=True)
        st.download_button('Export full simulated measurement CSV',sim.csv_bytes(),
                file_name='akus_simulated_scope.csv',mime='text/csv',use_container_width=True)
        if st.button('Clear simulated recording',use_container_width=True):invoke('clear_history')
    else:st.info('Run homing or a motion first to populate scope traces.')
    st.caption('These are software-simulated traces, not historical TwinCAT Scope measurements.')

with tabs[6]:
    st.subheader('Diagnostics, safety & trace')
    st.markdown('**Read-only virtual status**')
    s=sim.snapshot()
    diagnostic_rows=[
        ('Controller state',s['state']),('Reference complete',s['homed']),('Fault latched',s['e_stop']),
        ('TouchProbe 1, NC',s['top_probe_nc']),('TouchProbe 2, NC',s['top_probe_nc']),
        ('Motor 1 torque, modeled',f'{sim.torque1_nm:.3f} Nm'),
        ('Motor 2 torque, modeled',f'{sim.torque2_nm:.3f} Nm'),
        ('Position sync error',f'{sim.offset2_deg:.2f}°'),
        ('Experiment camera signal',sim.camera_on),('Virtual recording',sim.recording),
    ]
    st.dataframe(pd.DataFrame(diagnostic_rows,columns=['Item','Current value']),hide_index=True,use_container_width=True)
    st.markdown('#### Event log')
    events=pd.DataFrame(sim.events)
    st.dataframe(events.iloc[::-1],hide_index=True,use_container_width=True,height=280)
    st.download_button('Export event log JSON',json.dumps(sim.events,indent=2),
                       file_name='akus_simulation_events.json',mime='application/json')
    st.markdown('#### Sensor / calibration caveat')
    st.warning('Source audit reports the Kalman FB is not enabled by its call site, measurement-unit mismatches, and inconsistent acceleration calibration. This dashboard uses a basic display low-pass only; it does not claim to reproduce an operational Kalman filter or the EL3702 oversampling pipeline.')

with tabs[7]:
    st.subheader('Source-to-dashboard coverage')
    mapping=[
        ('MAIN.TcPOU','Init / Ready / Moving / ToStart / Ident / CtrlConfig','Workflow approximation'),
        ('MAIN.TcPOU','Upward reference search, MC_Home coordinate, return','Behaviorally simulated'),
        ('MAIN.TcPOU','Manual position, velocity, timed Jog','Kinematic approximation'),
        ('MAIN.TcPOU','Experiment calculation, delay, lock, automatic return','Core workflow implemented'),
        ('MAIN.TcPOU','Calibration / single-axis mode','Not implemented'),
        ('FeedbackControl.TcPOU','E_OpMode 6–11 mappings','Labels only; no NC/drive equivalence'),
        ('FB_AkusModel.TcPOU','Mass, gravity, nonlinear dry/viscous/position friction','Simplified dynamics option'),
        ('FB_AccIntegrator.TcPOU','Euler / trapezoid / RK4 selectability','Not individually replicated'),
        ('GVL_Persist.TcGVL','Default parameter values','Imported as editable Python config'),
        ('VISUs/*.TcVIS','Original page concepts','Recreated as modern dashboard, not pixel-perfect'),
        ('Measurement/*.tcscopex','Scope channel catalogs','Catalogs extracted; synthetic traces where supported'),
        ('FB_AkusKalmanFilter.TcPOU','State estimation / sensor chain','Corrected two-state Python reference + optional FilterPy in Kalman research; not TwinCAT scan-exact'),
        ('FB_TCPServer.TcPOU','Raspberry Pi accelerometer data','Not connected'),
        ('Beckhoff PLCopen MC_* / EtherCAT','Real motor drives / torque / homing / E-STOP','NOT connected for safety'),
    ]
    st.dataframe(pd.DataFrame(mapping,columns=['Source','Feature','Current coverage']),
                 hide_index=True,use_container_width=True)
    st.caption('Source project was used as the implementation reference. No claim of exact TwinCAT scan-equivalence or safety equivalence.')
    st.markdown('#### Units & constraints')
    st.code('1 motor revolution = 360° = 5 mm carrier travel\n44,000° = 611.11 mm simulated upper limit\nExperiment travel guard = 43,280° = 601.11 mm\nNominal gravitation = 9.80665 m/s²',language='text')
    st.markdown('See `README.md`, `reference/PLC_mapping.md` and `reference/scope_catalog.json` for source mapping, limitations, and test instructions.')


# Offline Kalman research integrated into existing live simulator and Visu.
with tabs[8]:
    st.subheader('Kalman research · Simulation + TwinCAT sensor recordings')
    st.caption('Jan two-state damped physics model · R from a stationary sample · '
               'structured Q via innovation likelihood · S and K calculated per update. '
               'OFFLINE only; the estimator never changes motor commands.')
    st.info('Use **Current carrier simulation** to compare ground truth against noisy measurements '
            'while the carrier visualization moves. Use **Real AKUS sample** or upload the '
            'full TwinCAT CSV to test measured signals. This is an estimator research tool, '
            'not a calibrated replacement for a physical sensor or a safety controller.')
    lab_a,lab_b=st.columns([1.1,1])
    with lab_a:
        lab_mode=st.radio('Data source',
            ['Current carrier simulation','Included real AKUS recording','Upload TwinCAT Scope CSV'],
            horizontal=False, key='kalman_source')
    with lab_b:
        backend=st.selectbox('Filter algorithm', ['reference','filterpy'],
                 help='Reference uses SciPy Cholesky solves and Joseph covariance update. '
                      'FilterPy uses an established Python Kalman-filter implementation.',key='kf_backend')
        fit_q=st.checkbox('Estimate Q from training measurements (innovation likelihood)',True,key='kf_fit')
        manual_q=st.number_input('Provisional qₐ [m²/s³] when not fitting',
                      min_value=1e-10,max_value=100.,value=1e-4,format='%.6g',key='kf_q')
    datafile=None
    calfile=None
    dt_target=.02
    sign_z=-1
    sign_tau=1
    force_factor=1.
    calibration_note=''
    if lab_mode=='Current carrier simulation':
        c1,c2,c3=st.columns(3)
        with c1: enc_sd=st.number_input('Synthetic encoder noise SD [mm]',min_value=0.001,
                                         value=0.2,format='%.3f',key='kf_enc')
        with c2: acc_sd=st.number_input('Synthetic acceleration noise SD [m/s²]',
                        min_value=.00001,value=.015,format='%.5f',key='kf_acc')
        with c3: seed=st.number_input('Random seed',min_value=0,value=42,step=1,key='kf_seed')
        st.caption(f'Available current carrier samples: {len(sim.history)}. '
                   'Run homing, a manual motion or an experiment in the other tabs first. '
                   'The position and velocity in sim.history are the simulation ground truth; '
                   'noise is injected only for the estimator.')
        if sim.simulation_type=='Ideal NC trajectory':
            st.warning('Ideal NC motion is not generated by Jan’s force equation: '
                       'a large Q or imperfect innovation consistency is expected.')
    else:
        upload_start=230.0
        upload_duration=10.0
        if lab_mode=='Included real AKUS recording':
            st.caption('Bundled samples from November 5, 2026: '
                      '230–240 s (real moving experiment) and 2–10 s (real stationary calibration). '
                      'The files are extracts; the original 296 s recording is not embedded.')
            datafile=ROOT/'examples'/'akus_movement_REAL.csv'
            calfile=ROOT/'examples'/'akus_stationary_REAL.csv'
        else:
            datafile=st.file_uploader('Upload full Test_KalmanFilter.csv or compatible export',
                                     type=['csv'],key='kf_motion_upload')
            calfile=st.file_uploader('Optional separate stationary calibration recording',
                                    type=['csv'],key='kf_cal_upload')
            t1,t2=st.columns(2)
            with t1:upload_start=st.number_input('Analysis start in uploaded recording [s]',
                                 min_value=0.,value=230.,step=1.,key='kf_record_start')
            with t2:upload_duration=st.number_input('Analysis duration [s] (max 30 s)',
                                 min_value=.5,max_value=30.,value=10.,step=.5,key='kf_record_duration')
            if calfile is None:
                calfile=ROOT/'examples'/'akus_stationary_REAL.csv'
                calibration_note=('No separate stationary recording supplied; using the bundled '
                                  'November 5, 2026 stationary sample. Verify identical sensor '
                                  'configuration and operating conditions before interpreting R.')
        x1,x2,x3=st.columns(3)
        with x1:dt_target=st.selectbox('Kalman sample period [ms]',[5,10,20,50],index=2,key='kf_dt')/1000
        with x2:sign_z=st.selectbox('Encoder-to-vertical direction',[-1,1],
                    help='Jan’s Kalman source uses a negative encoder angle sign. '
                         'Check against known direction of motion.',key='kf_sign_z')
        with x3:sign_tau=st.selectbox('Torque-to-force direction',[-1,1],index=1,
                    help='Nm alone does not establish the mechanical force polarity.',key='kf_sign_tau')
        force_factor=st.number_input('Exploratory force scale multiplier (NOT calibration)',
                        min_value=0.01,max_value=100.0,value=1.,step=.1,key='kf_force_scale')
        st.caption('Real position is reconstructed from the two encoder angles relative '
                   'to the beginning of the chosen segment. fAccZUnfiltered is in g; '
                   'the stationary baseline is subtracted and converted to m/s². '
                   'Raw 2 kHz signals are low-pass resampled before estimation; '
                   'R is estimated on the same resampled calibration signals.')
        if calibration_note:st.warning(calibration_note)
    with st.form('run_akus_kalman_lab'):
        run_kf=st.form_submit_button('Analyze Q / R / S / K using selected data',type='primary')
    if run_kf:
        try:
            if lab_mode=='Current carrier simulation':
                data=make_simulation_observations(sim.history,sim.config,enc_sd,acc_sd,int(seed))
                spectrum=None
            else:
                if datafile is None:
                    raise ValueError('Upload the motion CSV first')
                motion=load_twincat_csv(datafile)
                station=load_twincat_csv(calfile)
                if lab_mode=='Upload TwinCAT Scope CSV':
                    st.caption(f'Imported {len(motion):,} samples; {len(station):,} stationary samples')
                    start=float(motion['time_s'].min());end=float(motion['time_s'].max())
                    # Restrict interactive 2 kHz recording to a user-selected interval.
                    slice_start=float(upload_start)
                    slice_end=slice_start+float(upload_duration)
                    subset=motion[(motion['time_s']>=slice_start)&(motion['time_s']<=slice_end)]
                    if len(subset)<100:
                        raise ValueError(f'No 100+ samples between {slice_start:.2f} and {slice_end:.2f} s; '
                                         f'file range {start:.2f}–{end:.2f} s')
                    motion=subset.copy()
                data=make_real_observations(motion,station,dt_target,
                        sign_z,sign_tau,force_factor,sim.config)
                spectrum=analyze_frequency(motion,min(5.,float(motion['time_s'].max()-motion['time_s'].min())),station)
            if len(data['y']) > 12000:
                raise ValueError('Too many estimator samples. Choose a longer sampling period or a shorter recording.')
            output=estimate_and_validate(data,backend=backend,fit=fit_q,q_initial=manual_q)
            st.session_state.kf_lab_report=(data,output,spectrum,lab_mode)
        except Exception as error:
            st.session_state.pop('kf_lab_report',None)
            st.error(f'Cannot analyze the requested recording: {error}')
    if 'kf_lab_report' in st.session_state:
        data,analysis,spectrum,which=st.session_state.kf_lab_report
        f=analysis['result'];met=analysis['metrics']
        q=analysis['q']; R=data['R']; cut=analysis['split']
        st.markdown(f'#### Analysis results — {data["source"]}')
        if which!=lab_mode:
            st.warning('These results are from a previous data source. Press Analyze to refresh.')
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Identified qₐ [m²/s³]',f'{q:.4g}')
        c2.metric('R position [m²]',f'{R[0,0]:.3g}')
        c3.metric('R accel [(m/s²)²]',f'{R[1,1]:.3g}')
        c4.metric('Held-out mean NIS',f'{met["mean_nis"]:.3f}')
        if analysis['fit_report'] and analysis['fit_report']['near_bound']:
            st.warning('Q optimization reached an allowed bound. The fitted value is not '
                       'well identified; check physical model, torque scaling, units, '
                       'timing and noise assumptions. Do NOT treat q as calibrated.')
        if which!='Current carrier simulation':
            st.caption(f'Stationary acceleration reference: {data["baseline_g"]:.7f} g. '
                       f'Original 2 kHz stationary acceleration variance: '
                       f'{data["raw_R"][1,1]:.3e} (m/s²)²; '
                       f'after low-pass resampling: {R[1,1]:.3e} (m/s²)².')
            st.warning('REAL MEASUREMENTS: no independent platform-position ground truth. '
                       'The encoder is itself a Kalman measurement. R from stationary '
                       'scatter is only a preliminary estimate; vibration and sensor '
                       'correlation may invalidate white-noise assumptions. The fitted '
                       'Q may compensate for omitted Coulomb/static friction and torque bias.')
        st.caption(f'{len(data["y"]):,} estimator samples at {data["dt"]*1000:.2f} ms; '
                   f'first {cut:,} used to fit Q, remaining {len(data["y"])-cut:,} '
                   'held out for a within-recording check (NOT an independent experiment).')
        results=pd.DataFrame({
            'time_s':data['t'],
            'encoder_position_mm':data['y'][:,0]*1000,
            'kf_position_mm':f['estimates'][:,0]*1000,
            'kf_velocity_mm_s':f['estimates'][:,1]*1000,
            'acceleration_resampled_ms2':data['y'][:,1],
            'NIS':f['nis'],
            'K_position_from_encoder':f['gains'][:,0,0],
            'K_velocity_from_accel':f['gains'][:,1,1],
        })
        if data.get('truth') is not None:
            results['true_position_mm']=data['truth'][:,0]*1000
            results['true_velocity_mm_s']=data['truth'][:,1]*1000
        st.plotly_chart(go.Figure([
            go.Scatter(x=results['time_s'],y=results[c],name=c,mode='lines')
            for c in ('true_position_mm','encoder_position_mm','kf_position_mm')
            if c in results.columns]).update_layout(
            title='Carrier position — reference / encoder / Kalman',
            template='plotly_dark',xaxis_title='Time [s]',yaxis_title='Position [mm]',height=370
        ),use_container_width=True)
        a,b=st.columns(2)
        with a:
            st.line_chart(results.set_index('time_s')[['acceleration_resampled_ms2']],
                          height=260)
        with b:
            st.line_chart(results.set_index('time_s')[['NIS']],height=260)
        st.markdown('##### Covariance matrices and filter gain')
        cols=st.columns(4)
        for col,(name,mat) in zip(cols,[('R',f['R']),('Q',f['Q']),
                       ('Last S',f['S'][-1]),('Last K',f['gains'][-1])]):
            with col:
                st.markdown(f'**{name}**')
                st.dataframe(pd.DataFrame(mat).style.format('{:.4e}'),hide_index=True)
        st.markdown('##### Validation diagnostics')
        st.json({k:round(float(v),6) if np.isfinite(v) else None for k,v in met.items()})
        if spectrum is not None:
            with st.expander('Frequency analysis · acceleration PSD + torque/velocity coherence',expanded=True):
                g1,g2=st.columns(2)
                with g1:
                    fig1=go.Figure()
                    fig1.add_trace(go.Scatter(x=spectrum['frequency_hz'],y=spectrum['acc_psd'],name='Moving'))
                    if 'stationary_psd' in spectrum:
                        fig1.add_trace(go.Scatter(x=spectrum['frequency_hz'],y=spectrum['stationary_psd'],name='Stationary'))
                    fig1.update_layout(template='plotly_dark',title='Welch acceleration PSD',
                                       xaxis_title='Frequency [Hz]',yaxis_title='(m/s²)²/Hz',
                                       yaxis_type='log',height=310)
                    st.plotly_chart(fig1,use_container_width=True)
                with g2:
                    fig2=go.Figure()
                    for key,title in [('coherence_torque','Torque'),('coherence_velocity','Velocity')]:
                        fig2.add_trace(go.Scatter(x=spectrum['frequency_hz'],
                                  y=spectrum[key],name=title))
                    fig2.update_layout(template='plotly_dark',title='Magnitude-squared coherence',
                                xaxis_title='Frequency [Hz]',yaxis_range=[0,1],height=310)
                    st.plotly_chart(fig2,use_container_width=True)
                st.caption('Coherence indicates spectral association, not proof of '
                           'mechanical vibration. Strong vibration during movement '
                           'must not automatically be treated as white sensor noise.')
        d1,d2=st.columns(2)
        with d1:
            st.download_button('Export estimator time series CSV',
                results.to_csv(index=False).encode('utf8'),
                file_name='AKUS_Kalman_Results.csv',mime='text/csv')
        with d2:
            report={'source':data['source'],'sample_period_s':data['dt'],
                    'n_samples':len(data['y']),'Q_intensity_q_a':q,
                    'Q':f['Q'].tolist(),'R':f['R'].tolist(),
                    'K_last':f['gains'][-1].tolist(),'S_last':f['S'][-1].tolist(),
                    'model':{k:float(v) for k,v in vars(data['p']).items()},
                    'validation':{k:float(v) for k,v in met.items()},
                    'fit_status':analysis['fit_report']}
            st.download_button('Export identification report JSON',
                 json.dumps(report,indent=2),file_name='AKUS_Kalman_Identification.json',
                 mime='application/json')
    else:
        st.caption('Press Analyze to run the Kalman model using the selected source. '
                   'The carrier animation and all original operator functions remain available.')

# Explicitly physics-based, known-noise benchmark. Separate from v2's recorded
# carrier history and real recordings. None of this runs on Beckhoff hardware.
with tabs[9]:
    st.subheader('Jan physics · controlled Q/R identification benchmark')
    st.caption('Source-aligned offline plant vs. corrected linear Kalman estimator. '
               'Known synthetic truth permits an accuracy and covariance-calibration test. '
               'Run independent training and validation experiments; this page does not change the carrier controls.')
    st.info('For the FIRST research milestone choose **Matched linear plant**. '
            'When Q/R recovery and NIS/NEES look reasonable, choose **Jan nonlinear Euler plant** '
            'to measure what changes when the estimator no longer perfectly matches the physics.')
    with st.form('controlled_kalman_benchmark'):
        ca,cb,cc=st.columns(3)
        with ca:
            plant_choice=st.selectbox('Truth plant', ['Matched linear plant', 'Jan nonlinear Euler plant'],
                help='Linear is deliberately matched; nonlinear reproduces the FB_AkusModel friction '
                     'sign/threshold force rule and the Euler update in FB_AccIntegrator.')
            sample_dt_ms=st.selectbox('Simulation / Kalman sample time [ms]', [5,10,20,50],index=2)
            samples_train=st.number_input('Training samples',min_value=200,max_value=3000,value=850,step=100)
        with cb:
            inj_q=st.number_input('Known process intensity qₐ [m²/s³]',
                  min_value=1e-8,max_value=.5,value=.0001,format='%.6g')
            sd_position=st.number_input('Known encoder noise SD [mm]',
                   min_value=.001,max_value=5.,value=.35,format='%.4f')
            sd_acc=st.number_input('Known acceleration noise SD [m/s²]',
                   min_value=.0001,max_value=1.,value=.04,format='%.5f')
        with cc:
            samples_val=st.number_input('Independent validation samples',
                    min_value=200,max_value=3000,value=850,step=100)
            seed_benchmark=st.number_input('Reproducible random seed',min_value=0,value=42,step=1)
            algorithm_bench=st.selectbox('Kalman implementation', ['reference','filterpy'],
                 help='Both use the same model and data. FilterPy requires pip install filterpy.')
            identify_bench=st.checkbox('Estimate q from training data',value=True,
                   help='If unchecked q uses known injected value (oracle sanity check).')
        submit_benchmark=st.form_submit_button('Run train / held-out benchmark',type='primary')
    if submit_benchmark:
        try:
            with st.spinner('Generating truth, estimating R/Q and independently validating...'):
                bench=run_benchmark(
                    plant='linear' if plant_choice.startswith('Matched') else 'jan_euler',
                    dt=float(sample_dt_ms)*1e-3,
                    n_train=int(samples_train),n_validation=int(samples_val),
                    q_true=float(inj_q),noise_z_mm=float(sd_position),noise_a_ms2=float(sd_acc),
                    seed=int(seed_benchmark),backend=algorithm_bench,fit=identify_bench,
                    q_fixed=float(inj_q),model=ModelParameters(
                        m_platform=float(sim.config.m_platform_kg),
                        m_equivalent=float(sim.config.m_equivalent_kg),
                        b_viscous=float(sim.config.friction_viscous_ns_m),
                        friction_coulomb=float(sim.config.friction_coulomb_n),
                        friction_position=float(sim.config.friction_position_n_m2),
                        z_reference=float(sim.config.friction_center_m)))
                st.session_state.known_noise_benchmark=bench
        except Exception as error:
            st.session_state.pop('known_noise_benchmark',None)
            st.error(f'Benchmark could not finish: {error}')
    if 'known_noise_benchmark' in st.session_state:
        bench=st.session_state.known_noise_benchmark
        m=bench['metrics']; val=bench['validation']; f=bench['validation_result']
        st.markdown('#### Independently held-out validation')
        a1,a2,a3,a4=st.columns(4)
        a1.metric('Injected process qₐ',f'{bench["training"]["q_true"]:.4g}')
        a2.metric('Identified process qₐ',f'{bench["q_est"]:.4g}')
        a3.metric('Mean held-out NIS',f'{m["mean_nis"]:.3f}')
        a4.metric('Mean held-out NEES',f'{m["mean_nees"]:.3f}')
        st.caption(f'Nominal 95% NIS mean interval: {m["nis_mean_95pct_nominal_range"][0]:.3f}'
                   f'–{m["nis_mean_95pct_nominal_range"][1]:.3f}; '
                   f'NEES mean interval: {m["nees_mean_95pct_nominal_range"][0]:.3f}'
                   f'–{m["nees_mean_95pct_nominal_range"][1]:.3f}. '
                   'Intervals assume white Gaussian innovations and independent time samples.')
        b1,b2,b3,b4=st.columns(4)
        b1.metric('Encoder raw position RMSE',f'{m["raw_position_rmse_mm"]:.3f} mm')
        b2.metric('Kalman position RMSE',f'{m["position_rmse_mm"]:.3f} mm')
        b3.metric('Kalman velocity RMSE',f'{m["velocity_rmse_mm_s"]:.3f} mm/s')
        b4.metric('Position 95% interval coverage',f'{m["position_95pct_coverage"]*100:.1f}%')
        for message in bench['warnings']: st.warning(message)
        if not bench['warnings']:
            st.success('No diagnostic flags in this specific benchmark run. '
                       'This is not a certification; repeat with other seeds and disturbances.')
        st.markdown('#### Recovered covariance and Kalman gain')
        dc=st.columns(4)
        for col,(name,matrix) in zip(dc,[
            ('R injected',bench['training']['R_true']),
            ('R estimated',bench['R_est']),
            ('Q injected',bench['training']['Q_true']),
            ('Q estimated',f['Q'])]):
            with col:
                st.markdown(f'**{name}**')
                st.dataframe(pd.DataFrame(matrix).style.format('{:.5e}'),hide_index=True)
        st.caption('R is estimated exclusively from independent stationary calibration noise. '
                   'Q is fitted on a separate dynamic training run using Gaussian innovation '
                   'likelihood. S and K are calculated at every measurement update.')
        with st.expander('Final innovation covariance S and gain K',expanded=False):
            cc1,cc2=st.columns(2)
            with cc1:
                st.markdown('**S, final validation sample**')
                st.dataframe(pd.DataFrame(f['S'][-1]).style.format('{:.6e}'),hide_index=True)
            with cc2:
                st.markdown('**K, final validation sample**')
                st.dataframe(pd.DataFrame(f['gains'][-1]).style.format('{:.6e}'),hide_index=True)
        chart_df=pd.DataFrame(dict(
            time_s=val['t'],truth_mm=val['truth'][:,0]*1000,
            measured_mm=val['y'][:,0]*1000,kalman_mm=f['estimates'][:,0]*1000,
            truth_velocity_mm_s=val['truth'][:,1]*1000,
            kalman_velocity_mm_s=f['estimates'][:,1]*1000,
            truth_acc_ms2=val['acc_truth'],measured_acc_ms2=val['y'][:,1],
            kalman_acc_ms2=f['C'][1]@f['estimates'].T+f['D'][1,0]*val['u'],
            NIS=f['nis'],K_z_from_encoder=f['gains'][:,0,0],
            K_velocity_from_acceleration=f['gains'][:,1,1]))
        st.plotly_chart(go.Figure([
            go.Scatter(x=chart_df['time_s'],y=chart_df[key],name=label,mode='lines')
            for key,label in [('truth_mm','Known plant truth'),('measured_mm','Noisy encoder'),
                              ('kalman_mm','Kalman estimate')]
            ]).update_layout(template='plotly_dark',title='Validation carrier position',
                             xaxis_title='Time [s]',yaxis_title='Position [mm]',height=360),
                        use_container_width=True)
        left,right=st.columns(2)
        with left:
            st.plotly_chart(go.Figure([
                go.Scatter(x=chart_df['time_s'],y=chart_df[key],name=label)
                for key,label in [('truth_velocity_mm_s','Known true velocity'),
                                  ('kalman_velocity_mm_s','Kalman estimated velocity')]
                ]).update_layout(template='plotly_dark',height=300,
                       title='Velocity estimation',xaxis_title='Time [s]',
                       yaxis_title='Velocity [mm/s]'),use_container_width=True)
        with right:
            st.plotly_chart(go.Figure([
                go.Scatter(x=chart_df['time_s'],y=chart_df[key],name=label)
                for key,label in [('truth_acc_ms2','Plant acceleration'),
                                  ('measured_acc_ms2','Noisy accelerometer'),
                                  ('kalman_acc_ms2','Linear-model estimated acceleration')]
                ]).update_layout(template='plotly_dark',height=300,
                       title='Acceleration measurement vs model',
                       xaxis_title='Time [s]',yaxis_title='Acceleration [m/s²]'),
                       use_container_width=True)
        left,right=st.columns(2)
        with left:
            st.line_chart(chart_df.set_index('time_s')[['NIS']],height=240)
        with right:
            st.line_chart(chart_df.set_index('time_s')[
                ['K_z_from_encoder','K_velocity_from_acceleration']],height=240)
        st.markdown('#### Export the independently validated experiment')
        e1,e2=st.columns(2)
        with e1:
            st.download_button('Export validation time series CSV',
                   chart_df.to_csv(index=False).encode('utf-8'),
                   'AKUS_v3_synthetic_validation.csv',mime='text/csv')
        with e2:
            st.download_button('Export Q/R/S/K research report JSON',
                   json.dumps(report_json_ready(bench),indent=2).encode('utf-8'),
                   'AKUS_v3_synthetic_benchmark_report.json',mime='application/json')
        st.caption('This benchmark runs offline; the truth is artificially generated. '
                   'The nonlinear case intentionally violates the estimator assumptions. '
                   'A good match here does not prove the real Beckhoff drives or sensors are calibrated.')
    else:
        st.caption('Use the default matched linear run first. You can then switch to the '
                   'Jan nonlinear Euler plant without changing the synthetic noise settings.')


# v4: standalone acceleration-first Kalman estimator and Earth-hardware replay.
# NOT Jan's existing two-state position/velocity filter; Q and R differ by model.
with tabs[10]:
    st.subheader('Acceleration KF · simulated truth and real Earth-hardware replay')
    st.caption('Primary output: acceleration [m/s²]. The existing carrier animation, '
               'position/velocity Kalman research, and other operator pages remain unchanged.')
    st.info('This is a separate 1-state acceleration-residual Kalman filter, with '
            'Q_step = q × sample_period, R from stationary acceleration noise, '
            'and S/K calculated at every update. These are NOT the 2×2 matrices '
            'from Jan’s position/velocity Kalman model.')
    source_acc=st.radio('Acceleration dataset',
        ['Synthetic · Jan linear physics', 'Synthetic · Jan nonlinear Euler physics',
         'Real Earth hardware · included Scope excerpt', 'Real Earth hardware · upload CSV'],
        horizontal=False,key='v4_acc_source')
    is_synthetic=source_acc.startswith('Synthetic')
    with st.form('v4_acceleration_form'):
        f1,f2,f3=st.columns(3)
        with f1:
            dt_acc=st.selectbox('Analysis sample period (ms)', [0.5,2,5,10,20,50],index=3,
                help='The 0.5ms setting applies mainly to real 2kHz recordings; synthetic runs match their own chosen sampling period.')
            prior_choice=st.selectbox('Acceleration prediction',
                ['Measured acceleration only (random-walk model)',
                 'Jan linear physics force prediction (experimental)'],
                index=1 if is_synthetic else 0)
        with f2:
            sd_acc=st.number_input('Synthetic sensor noise SD [m/s²]',.0001,1.0,.04,format='%.5f',disabled=not is_synthetic)
            n_samples=st.number_input('Samples per synthetic run',min_value=200,max_value=3000,value=950,step=50,disabled=not is_synthetic)
        with f3:
            seed_acc=st.number_input('Synthetic random seed',0,1000000,42,1,disabled=not is_synthetic)
            use_fit=st.checkbox('Identify acceleration Q on training interval',value=True)
        uploaded_acc=None
        if source_acc.endswith('upload CSV'):
            uploaded_acc=st.file_uploader('TwinCAT CSV · 134 MB full file is supported by Streamlit when upload limit permits',type=['csv'],key='acc_uploaded')
            start_acc=st.number_input('Start of moving interval [s]',min_value=0.0,value=230.0,step=1.0)
            duration_acc=st.number_input('Duration [s]',min_value=1.0,max_value=180.,value=10.,step=1.)
        else:
            start_acc=230.;duration_acc=10.
        go_acc=st.form_submit_button('Estimate R / Q and filter acceleration',type='primary')
    if go_acc:
        try:
            with st.spinner('Estimating acceleration uncertainty and validating...'):
                dt_target=dt_acc*0.001
                if is_synthetic:
                    params=ModelParameters(
                        m_platform=float(sim.config.m_platform_kg),
                        m_equivalent=float(sim.config.m_equivalent_kg),
                        b_viscous=float(sim.config.friction_viscous_ns_m),
                        friction_coulomb=float(sim.config.friction_coulomb_n),
                        friction_position=float(sim.config.friction_position_n_m2),
                        z_reference=float(sim.config.friction_center_m))
                    plant='linear' if 'linear physics' in source_acc else 'jan_euler'
                    train_acc=generate_jan_synthetic(dt=dt_target,N=int(n_samples),p=params,
                        plant=plant,seed=int(seed_acc),scenario='training',sensor_std_a=float(sd_acc))
                    hold_acc=generate_jan_synthetic(dt=dt_target,N=int(n_samples),p=params,
                        plant=plant,seed=int(seed_acc)+7919,scenario='validation',sensor_std_a=float(sd_acc))
                    Racc,corr=estimate_acceleration_r(train_acc['calibration_residuals'][:,1])
                    a_train=train_acc['y'][:,1]; a_hold=hold_acc['y'][:,1]
                    t_hold=hold_acc['t']; truth_acc=hold_acc['acc_truth']
                    compare_acc=None
                    if prior_choice.startswith('Jan linear'):
                        prior_train=jan_linear_acceleration_prior(train_acc['u'],dt_target,params)
                        prior_hold=jan_linear_acceleration_prior(hold_acc['u'],dt_target,params)
                    else:
                        prior_train=prior_hold=None
                    source_note='Synthetic known-truth, independent training/validation seeds and force profiles'
                    actual_dt=dt_target
                else:
                    stationary_acc=pd.read_csv(ROOT/'examples'/'akus_stationary_REAL.csv')
                    if uploaded_acc is None and source_acc.endswith('upload CSV'):
                        raise ValueError('Upload the TwinCAT Scope CSV first')
                    if uploaded_acc is None:
                        motion_acc=pd.read_csv(ROOT/'examples'/'akus_movement_REAL.csv')
                    else:
                        full_acc=load_twincat_csv(uploaded_acc)
                        motion_acc=full_acc.loc[(full_acc.time_s>=start_acc)&
                            (full_acc.time_s<start_acc+duration_acc)].copy()
                        if len(motion_acc)<50:raise ValueError('Selected interval has insufficient samples')
                    # Decimate with anti-alias low-pass *before* comparing samples
                    # or estimating R at the chosen estimator sample rate.
                    motion_ds,actual_dt=_decimate(motion_acc,dt_target)
                    stationary_ds,_=_decimate(stationary_acc,dt_target)
                    g0=float(stationary_acc['fAccZUnfiltered'].median())
                    # The raw sensor channel is specific force in g, ~1g at rest.
                    # Subtracting a stationary baseline gives an Earth-relative
                    # dynamic acceleration estimate, NOT a full sensor calibration.
                    a_hold=(motion_ds['fAccZUnfiltered'].to_numpy(float)-g0)*9.80665
                    cal_acc=(stationary_ds['fAccZUnfiltered'].to_numpy(float)-g0)*9.80665
                    Racc,corr=estimate_acceleration_r(cal_acc)
                    t_hold=motion_ds['time_s'].to_numpy(float)
                    truth_acc=None
                    compare_acc=motion_ds['fAccZ'].to_numpy(float)-float(stationary_ds['fAccZ'].mean())
                    if prior_choice.startswith('Jan linear'):
                        p=ModelParameters()
                        # Torque-to-force is NOT calibrated. The optional prior is
                        # exploratory only. Use by explicit operator selection.
                        force=p.motor_force(motion_ds['fTorque1'].to_numpy(float),
                                            motion_ds['fTorque2'].to_numpy(float))
                        u=p.u_from_force(force)
                        prior_hold=jan_linear_acceleration_prior(u,actual_dt,p)
                    else:
                        prior_hold=None
                    source_note='Real Earth hardware; no independent acceleration truth. Scope low-pass is NOT ground truth.'
                    # Time-wise training/validation on selected motion interval;
                    # this is less independent than two distinct experiment runs.
                    cut=max(60,min(len(a_hold)-30,int(len(a_hold)*.55)))
                    a_train=a_hold[:cut]
                    prior_train=prior_hold[:cut] if prior_hold is not None else None
                if use_fit:
                    q_acc,fit_acc=fit_acceleration_q(a_train,actual_dt,Racc,
                        prior_ms2=prior_train)
                else:
                    q_acc=1e-4
                    fit_acc={'near_bound':False,'converged':True,'note':'Manually selected q, not identified'}
                result_acc=run_acceleration_filter(a_hold,actual_dt,Racc,q_acc,prior_ms2=prior_hold)
                # For real hardware use the second interval as diagnostics, not
                # an independent experiment. For synthetic run use held-out run.
                burn_acc=(0 if is_synthetic else cut)
                metrics_acc=acceleration_diagnostics(result_acc,truth_acc,burn=burn_acc)
                if truth_acc is not None:
                    metrics_acc['raw_acc_rmse_ms2']=float(np.sqrt(np.mean((a_hold-truth_acc)**2)))
                export_acc=pd.DataFrame({'time_s':t_hold,'acc_measured_ms2':a_hold,
                    'acc_kalman_ms2':result_acc['estimated_acc_ms2'],
                    'acc_prior_ms2':result_acc['prior_ms2'],
                    'innovation_ms2':result_acc['innovation'],'K':result_acc['K'],
                    'S':result_acc['S'],'NIS':result_acc['nis']})
                if truth_acc is not None:export_acc['acc_truth_ms2']=truth_acc
                if compare_acc is not None:export_acc['twinCAT_filtered_acc_ms2_centered']=compare_acc
                st.session_state.v4_acc_result={'df':export_acc,'R':Racc,'q':q_acc,
                    'Q':result_acc['Q_step'],'corr':corr,'fit':fit_acc,'metrics':metrics_acc,
                    'source':source_note,'synthetic':is_synthetic,'training_cut':burn_acc}
        except Exception as ex:
            st.session_state.pop('v4_acc_result',None)
            st.error(f'Acceleration research run failed: {ex}')
    if 'v4_acc_result' in st.session_state:
        ar=st.session_state.v4_acc_result
        st.caption(ar['source'])
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Acceleration R [(m/s²)²]',f'{ar["R"]:.5g}')
        c2.metric('Identified q [(m/s²)²/s]',f'{ar["q"]:.5g}')
        c3.metric('Q per sample [(m/s²)²]',f'{ar["Q"]:.5g}')
        c4.metric('Last Kalman gain K',f'{ar["df"]["K"].iloc[-1]:.4g}')
        if ar['fit'].get('near_bound'):
            st.warning('Estimated q is near the optimization boundary: not reliably identified. '
                       'Check model mismatch, bias, filtering bandwidth and measurement assumptions.')
        if abs(ar['corr'])>.2:
            st.warning('Stationary acceleration samples are correlated at this chosen sample rate. '
                       'R is only a preliminary covariance under a white-noise assumption.')
        if not ar['synthetic']:
            st.warning('Hardware replay is from Earth, not free fall. A stationary +1g baseline '
                       'has been subtracted, but gain, offset drift and torque direction are not calibrated. '
                       'The TwinCAT filtered channel is a comparison, never ground truth.')
        st.markdown('#### Acceleration: measurement vs Kalman estimate')
        ac=ar['df'];fig=go.Figure()
        series=[('acc_measured_ms2','Raw averaged accelerometer'),('acc_kalman_ms2','Kalman acceleration'),
                ('acc_truth_ms2','Synthetic ground truth'),('twinCAT_filtered_acc_ms2_centered','TwinCAT 10 Hz low-pass')]
        for k,name in series:
            if k in ac.columns:fig.add_trace(go.Scatter(x=ac.time_s,y=ac[k],name=name,mode='lines'))
        fig.update_layout(template='plotly_dark',height=430,xaxis_title='Time [s]',
                          yaxis_title='Dynamic acceleration [m/s²]',hovermode='x unified')
        st.plotly_chart(fig,use_container_width=True)
        ca,cb=st.columns(2)
        with ca:
            st.line_chart(ac.set_index('time_s')[['K','S']],height=250)
        with cb:
            st.line_chart(ac.set_index('time_s')[['NIS']],height=250)
        st.json({k:float(v) if isinstance(v,(int,float,np.floating)) else v
                 for k,v in ar['metrics'].items()})
        if 'kalman_acc_rmse_ms2' in ar['metrics']:
            st.caption('Synthetic ground-truth RMSE: raw '
                       f'{ar["metrics"]["raw_acc_rmse_ms2"]:.5g} m/s² vs. Kalman '
                       f'{ar["metrics"]["kalman_acc_rmse_ms2"]:.5g} m/s².')
        st.caption('Innovation diagnostics require white residuals to interpret nominal NIS '
                   'thresholds. Smoothness alone does not prove accuracy. '
                   'This 1-state Q must not be copied into Jan’s 2-state position/velocity filter.')
        da,db=st.columns(2)
        with da:
            st.download_button('Download acceleration output CSV',
                ac.to_csv(index=False).encode('utf-8'),file_name='AKUS_Acceleration_Filter.csv',mime='text/csv')
        with db:
            report={'R_acc':ar['R'],'q_acc':ar['q'],'Q_acc_per_sample':ar['Q'],
                    'K_last':float(ac.K.iloc[-1]),'S_last':float(ac.S.iloc[-1]),
                    'stationary_lag1':ar['corr'],'fit':ar['fit'],'metrics':ar['metrics'],
                    'caveat':ar['source']}
            st.download_button('Download acceleration identification JSON',
                json.dumps(report,indent=2),file_name='AKUS_Acceleration_Identification.json',mime='application/json')
    else:
        st.caption('Choose simulated Jan physics or real Earth-hardware replay, then run '
                   'the acceleration identification above. This does not drive the real servos.')
