"""Run reproducible, fully offline source-derived AKUS Q/R validation.

Examples:
  python run_benchmark.py --plant linear --output results/linear
  python run_benchmark.py --plant jan_euler --output results/jan_euler
  python run_benchmark.py --plant linear --backend filterpy

Q/R are fitted only to synthetic training measurements; a different stochastic
seed and force profile constitute a separate independent validation experiment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from akuskf.benchmark import run_benchmark,report_json_ready


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plant',choices=['linear','jan_euler'],default='linear')
    ap.add_argument('--backend',choices=['reference','filterpy'],default='reference')
    ap.add_argument('--dt',type=float,default=.02,help='sample period in seconds')
    ap.add_argument('--train',type=int,default=850)
    ap.add_argument('--validation',type=int,default=850)
    ap.add_argument('--q',type=float,default=1e-4,help='known injected process spectral intensity')
    ap.add_argument('--position-noise-mm',type=float,default=.35)
    ap.add_argument('--acceleration-noise-ms2',type=float,default=.04)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output',type=Path,default=Path('results')/'sim_matched_linear')
    args=ap.parse_args()
    result=run_benchmark(plant=args.plant,backend=args.backend,dt=args.dt,
             n_train=args.train,n_validation=args.validation,q_true=args.q,
             noise_z_mm=args.position_noise_mm,noise_a_ms2=args.acceleration_noise_ms2,
             seed=args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    truth=result['validation']; f=result['validation_result']
    pd.DataFrame({
      'time_s':truth['t'], 'force_input_N':truth['u'],
      'true_position_m':truth['truth'][:,0],
      'true_velocity_m_s':truth['truth'][:,1],
      'true_acceleration_m_s2':truth['acc_truth'],
      'noisy_encoder_position_m':truth['y'][:,0],
      'noisy_acceleration_m_s2':truth['y'][:,1],
      'KF_position_m':f['estimates'][:,0],
      'KF_velocity_m_s':f['estimates'][:,1],
      'K_z_from_encoder':f['gains'][:,0,0],
      'K_velocity_from_acceleration':f['gains'][:,1,1],
      'NIS':f['nis'],
      'S_00':f['S'][:,0,0], 'S_01':f['S'][:,0,1],
      'S_11':f['S'][:,1,1],
    }).to_csv(args.output/'validation.csv',index=False)
    with (args.output/'report.json').open('w') as h:
        json.dump(report_json_ready(result),h,indent=2,allow_nan=False)
    m=result['metrics']
    print('Plant:',args.plant)
    print(f'Injected q={args.q:.6g}; identified q={result["q_est"]:.6g}')
    print('Injected R diag:', np.diag(result['training']['R_true']))
    print('Estimated R diag:', np.diag(result['R_est']))
    print(f'Validation position RMSE: raw={m["raw_position_rmse_mm"]:.3f} mm, KF={m["position_rmse_mm"]:.3f} mm')
    print(f'Validation NIS={m["mean_nis"]:.3f}, NEES={m["mean_nees"]:.3f}, position coverage={m["position_95pct_coverage"]:.3%}')
    print('Warnings:', '; '.join(result['warnings']) or 'None for this run')
    print('Saved:',args.output.resolve())


if __name__ == '__main__':
    main()
