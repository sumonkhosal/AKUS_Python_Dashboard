"""The 2-state matrices, likelihood, and independent replay math are tested offline.
Do NOT confuse tests of Python-generated ST equations with running actual TwinCAT PLC.
"""
import unittest
import numpy as np
from akuskf.synthetic import generate
from akuskf.jan_model import ModelParameters, matrices, continuous_white_acceleration_Q
from akuskf.identification import estimate_q_by_likelihood, estimate_R
from akuskf.filtering import run_filter
from phase2_two_state import compare_reference_cyclewise, st_source, initial_real_R

class TwoStatePhaseTests(unittest.TestCase):
    def setUp(self):
        self.dt=.01
        self.p=ModelParameters()
        self.data=generate(dt=self.dt,N=150,p=self.p,seed=9,true_q=1e-4,calibration_count=2500,plant='linear')
        self.R=estimate_R(self.data['calibration_residuals'])
    def test_Q_and_R_matrices(self):
        A,B,C,D,Ac=matrices(self.dt,self.p)
        Q=continuous_white_acceleration_Q(self.dt,Ac,.0001)
        self.assertEqual(Q.shape,(2,2))
        self.assertEqual(self.R.shape,(2,2))
        self.assertGreater(np.min(np.linalg.eigvalsh(Q)),0)
        self.assertGreater(np.min(np.linalg.eigvalsh(self.R)),0)
        np.testing.assert_allclose(Q,Q.T,atol=1e-16)
        self.assertEqual(A.shape,(2,2))
        self.assertEqual(B.shape,(2,1))
        self.assertEqual(C.shape,(2,2))
        self.assertEqual(D.shape,(2,1))
    def test_analytical_replay_matches_reference(self):
        y,u=self.data['y'],self.data['u']
        result=run_filter(y,u,self.dt,self.p,self.R,1e-4)
        replay=compare_reference_cyclewise(y,u,self.p,self.R,1e-4,self.dt)
        expected=np.column_stack((result['estimates'],result['gains'].reshape(-1,4),result['S'][:,0,0],result['S'][:,0,1],result['S'][:,1,1],result['P'][:,0,0],result['P'][:,0,1],result['P'][:,1,1],result['innovations'],result['nis']))
        self.assertLess(np.max(np.abs(expected-replay)),1e-9)
        self.assertTrue(np.isfinite(replay).all())
    def test_ST_generator_has_dynamic_gain_and_joseph(self):
        A,B,C,D,Ac=matrices(self.dt,self.p)
        Q=continuous_white_acceleration_Q(self.dt,Ac,1e-4)
        source=st_source(A,B,C,D,Q,self.R,self.dt,np.diag([max(self.R[0,0],1e-10),.02**2]))
        self.assertIn('FUNCTION_BLOCK FB_AKUS_TwoStateKF',source)
        self.assertIn('detS:=',source)
        self.assertIn('J[i,j]:=',source)
        self.assertIn('K[i,1]:=',source)
        self.assertNotIn('__DEFAULTS__',source)
    def test_likelihood_finds_positive_q(self):
        q,rep=estimate_q_by_likelihood(self.data['y'],self.data['u'],self.dt,self.p,self.R)
        self.assertTrue(np.isfinite(q))
        self.assertGreater(q,0)
        self.assertTrue(np.isfinite(rep['nll']))

if __name__=='__main__': unittest.main()
