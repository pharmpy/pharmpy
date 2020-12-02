import numpy as np
import pandas as pd
import pytest

from pharmpy import Model
from pharmpy.parameter_sampling import sample_from_covariance_matrix, sample_individual_estimates


def test_sample_from_covariance_matrix(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    np.random.seed(318)
    samples = sample_from_covariance_matrix(model, n=3)
    correct = pd.DataFrame(
        {
            'THETA(1)': [0.004965, 0.004811, 0.004631],
            'THETA(2)': [0.979979, 1.042210, 0.962791],
            'THETA(3)': [0.007825, -0.069350, 0.052367],
            'OMEGA(1,1)': [0.019811, 0.059127, 0.030619],
            'OMEGA(2,2)': [0.025248, 0.029088, 0.019749],
            'SIGMA(1,1)': [0.014700, 0.014347, 0.011470],
        }
    )
    pd.testing.assert_frame_equal(samples, correct, atol=1e-6)
    # Make cov matrix non-posdef
    model.modelfit_results.covariance_matrix['THETA(1)']['THETA(1)'] = -1
    with pytest.warns(UserWarning):
        sample_from_covariance_matrix(model, n=1, force_posdef_covmatrix=True)


def test_sample_individual_estimates(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    np.random.seed(86)
    samples = sample_individual_estimates(model)
    assert len(samples) == 59 * 100
    assert list(samples.columns) == ['ETA(1)', 'ETA(2)']
    assert pytest.approx(samples.iloc[0]['ETA(1)'], 1e-5) == 0.0418399
    assert pytest.approx(samples.iloc[0]['ETA(2)'], 1e-5) == -0.0587623

    restricted = sample_individual_estimates(model, parameters=['ETA(2)'], samples_per_id=1)
    assert len(restricted) == 59
    assert restricted.columns == ['ETA(2)']
    assert pytest.approx(restricted.iloc[0]['ETA(2)'], 1e-5) == -0.0167246
