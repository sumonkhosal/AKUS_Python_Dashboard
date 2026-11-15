"""Corrected *linear* two-state matrices derived from Jan's FB_AkusKalmanFilter.

Coordinates: z [m] positive upward; v [m/s] positive upward.
Input u [N]: spindle linear force MINUS platform weight; nonlinear dry/position
friction is deliberately excluded from the baseline filter, but can appear
in the separate nonlinear virtual plant for robustness assessment.
Raw NC ActTorque is NOT directly an input here. Verify drive torque scaling.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm

@dataclass(frozen=True)
class ModelParameters:
    m_platform: float = 17.395
    m_equivalent: float = 4154.59
    b_viscous: float = 3994.857
    friction_coulomb: float = 225.059
    friction_position: float = 1207.7
    z_reference: float = 0.249715
    pitch: float = 0.005
    gravity: float = 9.80665

    def validate(self):
        if self.m_platform <= 0 or self.m_equivalent < 0 or self.b_viscous < 0 or self.pitch <= 0:
            raise ValueError('Unphysical model parameters')

    @property
    def effective_mass(self):
        return self.m_platform + self.m_equivalent

    def motor_force(self, spindle_torque1_nm, spindle_torque2_nm):
        """Ideal spindle relationship; assumes torques measured AT spindle, not raw percent feedback."""
        return 2.0*np.pi*(np.asarray(spindle_torque1_nm)+np.asarray(spindle_torque2_nm))/self.pitch

    def u_from_force(self, total_spindle_force_newtons):
        """Known-input net force before viscous/dry friction, SI units."""
        return np.asarray(total_spindle_force_newtons) - self.m_platform*self.gravity


def matrices(dt: float, p: ModelParameters):
    """Exact sampled A,B for damped force model; H=C and D for z, acceleration.

    Continuous A = [[0,1],[0,-b/M]], B = [[0],[1/M]].
    Block matrix exponential is stable as b tends to zero.
    """
    p.validate()
    if dt <= 0: raise ValueError('dt must be positive')
    M=p.effective_mass; alpha=p.b_viscous/M
    A_c=np.array([[0.,1.],[0.,-alpha]])
    B_c=np.array([[0.],[1./M]])
    augmented=np.zeros((3,3)); augmented[:2,:2]=A_c; augmented[:2,2:3]=B_c
    F=expm(augmented*dt)
    A=F[:2,:2]; B=F[:2,2:3]
    C=np.array([[1.,0.],[0.,-alpha]])
    D=np.array([[0.],[1./M]])
    return A,B,C,D,A_c


def continuous_white_acceleration_Q(dt:float, A_c:np.ndarray, q_a:float):
    """Exact Van-Loan discretization of continuous acceleration-disturbance spectral density.

    Qd = integral_0^dt exp(A*s) [0,0;0,q_a] exp(A.T*s) ds.
    q_a in m^2/s^3; Q position variance in m^2, velocity in (m/s)^2.
    """
    if q_a < 0: raise ValueError('q_a must be nonnegative')
    V=np.zeros((2,2)); V[1,1]=float(q_a)
    block=np.zeros((4,4)); block[:2,:2]=A_c; block[:2,2:]=V; block[2:,2:]=-A_c.T
    E=expm(block*dt)
    Q=E[:2,2:]@E[:2,:2].T
    return 0.5*(Q+Q.T)
