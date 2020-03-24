import numpy as np
import pandas as pd

from pharmpy import Model
from pharmpy.parameter_sampling import sample_from_covariance_matrix


def test_sample_from_covariance_matrix(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    np.random.seed(318)
    samples = sample_from_covariance_matrix(model, n=3)
    correct = pd.DataFrame({'THETA(1)': [0.004965, 0.004811, 0.004631],
                            'THETA(2)': [0.979979, 1.042210, 0.962791],
                            'THETA(3)': [0.007825, -0.069350, 0.052367],
                            'OMEGA(1,1)': [0.019811, 0.059127, 0.030619],
                            'OMEGA(2,2)': [0.025248, 0.029088, 0.019749],
                            'SIGMA(1,1)': [0.014700, 0.014347, 0.011470]})
    pd.testing.assert_frame_equal(samples, correct, check_less_precise=True)
