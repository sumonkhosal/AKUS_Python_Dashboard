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

tabs=st.tabs(['Experiment','Manual motion','Controller tuning','Identification','Vacuum','Analysis / Scopes','Diagnostics','Project reference'])

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
        ('FB_AkusKalmanFilter.TcPOU','State estimation / sensor chain','Not replicated; source bugs require fixing'),
        ('FB_TCPServer.TcPOU','Raspberry Pi accelerometer data','Not connected'),
        ('Beckhoff PLCopen MC_* / EtherCAT','Real motor drives / torque / homing / E-STOP','NOT connected for safety'),
    ]
    st.dataframe(pd.DataFrame(mapping,columns=['Source','Feature','Current coverage']),
                 hide_index=True,use_container_width=True)
    st.caption('Source project was used as the implementation reference. No claim of exact TwinCAT scan-equivalence or safety equivalence.')
    st.markdown('#### Units & constraints')
    st.code('1 motor revolution = 360° = 5 mm carrier travel\n44,000° = 611.11 mm simulated upper limit\nExperiment travel guard = 43,280° = 601.11 mm\nNominal gravitation = 9.80665 m/s²',language='text')
    st.markdown('See `README.md`, `reference/PLC_mapping.md` and `reference/scope_catalog.json` for source mapping, limitations, and test instructions.')
