"""Standalone sample recording generator; no GUI or third-party packages required."""
from pathlib import Path
from akus_sim import Simulator

if __name__ == '__main__':
    sim=Simulator()
    sim.start_homing(beckhoff=True)
    sim.finish_operation()
    sim.move_absolute(125.0,velocity_deg_s=2400., acceleration_deg_s2=700.,deceleration_deg_s2=700.)
    sim.finish_operation()
    sim.move_to_start()
    sim.finish_operation()
    sim.start_experiment(.001,3.,'PositionBeckhoff',lock_s=1.0)
    sim.finish_operation()
    target=Path(__file__).parent/'example_scope_recording.csv'
    target.write_bytes(sim.csv_bytes())
    print(f'Generated {target} with {len(sim.history)} samples and {len(sim.events)} events.')
