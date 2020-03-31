import numpy as np
import pandas as pd

from pharmpy import Model
from pharmpy.methods.frem.results import FREMResults


def test_frem_results(testdata):
    model = Model(testdata / 'nonmem' / 'model_4.mod')
    np.random.seed(39)
    res = FREMResults(model, continuous=['APGR', 'WGT'], samples=10)
    correct = pd.DataFrame({
        'parameter': ['0', '0', '0', '0', '1', '1', '1', '1'],
        'covariate': ['APGR', 'APGR', 'WGT', 'WGT', 'APGR', 'APGR', 'WGT', 'WGT'],
        'condition': ['5th', '95th', '5th', '95th', '5th', '95th', '5th', '95th'],
        '5th': [1.021471, 0.855472, 0.862255, 0.976016, 0.813598, 1.020936, 0.942373, 0.915405],
        'mean': [1.159994, 0.935780, 0.939846, 1.145692, 0.876731, 1.065829, 1.005247, 0.993865],
        '95th': [1.389830, 0.990048, 1.013764, 1.359351, 0.957428, 1.103065, 1.044549, 1.129945]})
    pd.testing.assert_frame_equal(res.covariate_effects, correct)
