"""FilterPy implementation plus a transparent SciPy/NumPy reference implementation.

FilterPy is deliberately a normal pip dependency, not a bundled or copied library.
Choice of backend can be checked with tests. Joseph-form covariance in reference.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from .jan_model import matrices,continuous_white_acceleration_Q


def run_filter(y, u, dt, p, R, q_a, initial_state=None, P0=None, backend='reference'):
    """Inputs y (N,2) = [encoder z m, calibrated acceleration m/s^2],
    u (N,) known force [N] excluding platform gravity.
    y[k] = C*x[k] + D*u[k] + sensor noise, x[k]=A*x[k-1]+B*u[k-1].

    Missing measurements supported per channel with np.nan by reference backend only.
    """
    y=np.asarray(y,dtype=float); u=np.asarray(u,dtype=float)
    if y.ndim != 2 or y.shape[1]!=2 or u.shape!=(len(y),):
        raise ValueError('y shape must be (N,2), u shape (N,)')
    R=np.asarray(R,dtype=float)
    if R.shape!=(2,2) or not np.allclose(R,R.T,atol=1.e-12) or np.min(np.linalg.eigvalsh(R))<=0:
        raise ValueError('R must be 2x2 symmetric positive-definite')
    A,B,C,D,A_c=matrices(dt,p)
    Q=continuous_white_acceleration_Q(dt,A_c,q_a)
    x=np.array([y[0,0] if np.isfinite(y[0,0]) else p.z_reference,0.],dtype=float) if initial_state is None else np.asarray(initial_state,dtype=float).copy()
    P=np.diag([max(R[0,0],1.e-10),0.02**2]) if P0 is None else np.asarray(P0,dtype=float).copy()
    if x.shape!=(2,) or P.shape!=(2,2):raise ValueError('initial state / P0 shape mismatch')
    if backend=='filterpy':
        try: from filterpy.kalman import KalmanFilter
        except ImportError as exc: raise ImportError('Install optional FilterPy: pip install filterpy') from exc
        if not np.isfinite(y).all(): raise ValueError('Use reference backend for missing channels')
        kf=KalmanFilter(dim_x=2,dim_z=2)
        kf.x=x.reshape((2,1)); kf.P=P; kf.F=A; kf.B=B; kf.H=C; kf.Q=Q; kf.R=R
    elif backend != 'reference':
        raise ValueError('backend must be reference or filterpy')
    N=len(u)
    estimates=np.full((N,2),np.nan); innov=np.full((N,2),np.nan)
    nis=np.full(N,np.nan); nll=np.full(N,np.nan)
    ks=np.full((N,2,2),np.nan); ss=np.full((N,2,2),np.nan); ps=np.full((N,2,2),np.nan)
    I=np.eye(2)
    for k in range(N):
        if backend=='filterpy':
            if k>0: kf.predict(u=float(u[k-1]))
            xpred=np.ravel(kf.x); Ppred=kf.P.copy()
        else:
            if k>0:
                x=A@x+B[:,0]*u[k-1]
                P=A@P@A.T+Q
            xpred=x.copy(); Ppred=P.copy()
        indices=np.flatnonzero(np.isfinite(y[k]))
        if not len(indices):
            if backend=='reference':
                estimates[k]=x; ps[k]=P
            else:
                estimates[k]=xpred; ps[k]=Ppred
            continue
        H=C[indices]; Rc=R[np.ix_(indices,indices)]; Dc=D[indices,0]
        r=y[k,indices]-(H@xpred+Dc*u[k]); S=H@Ppred@H.T+Rc
        try:
            fact=cho_factor(S,lower=True,check_finite=True)
            G=Ppred@H.T
            gain=cho_solve(fact,G.T).T
            Sinv_r=cho_solve(fact,r)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Innovation covariance not positive definite') from exc
        err=float(r@Sinv_r)
        # Observations dimensionality is n_obs for missing-channel cases.
        logdet=2*float(np.log(np.diag(fact[0])).sum())
        if backend=='filterpy':
            # FilterPy subtracts input feedthrough D*u from the measurement.
            kf.update((y[k]-D[:,0]*u[k]).reshape((2,1)))
            x=np.ravel(kf.x); P=kf.P.copy()
        else:
            x=xpred+gain@r
            M=I-gain@H
            P=M@Ppred@M.T+gain@Rc@gain.T  # Joseph update
            P=0.5*(P+P.T)
        estimates[k]=x; ps[k]=P
        innov[k,indices]=r
        nis[k]=err; nll[k]=0.5*(len(indices)*np.log(2*np.pi)+logdet+err)
        ks[k][:,indices]=gain; ss[k][np.ix_(indices,indices)]=S
    return dict(estimates=estimates,innovations=innov,nis=nis,nll=nll,
                gains=ks,S=ss,P=ps,Q=Q,R=R,A=A,B=B,C=C,D=D)
