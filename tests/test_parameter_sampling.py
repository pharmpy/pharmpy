import numpy as np
import pandas as pd
import pytest

from pharmpy import Model
from pharmpy.parameter_sampling import sample_from_covariance_matrix, sample_individual_estimates


def test_sample_from_covariance_matrix(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    rng = np.random.default_rng(318)
    samples = sample_from_covariance_matrix(model, n=3, seed=rng)
    correct = pd.DataFrame(
        {
            'THETA(1)': [0.004489330033579095, 0.004866193232279955, 0.004619661658761273],
            'THETA(2)': [0.9720563045663096, 1.0217868717352445, 0.9662036500731115],
            'THETA(3)': [0.19927467608338267, 0.237140948854298, 0.1979609848931148],
            'OMEGA(1,1)': [0.012012933626520568, 0.03859989956899462, 0.03228178862778379],
            'OMEGA(2,2)': [0.03718187653238525, 0.036766142234483934, 0.02433717922068797],
            'SIGMA(1,1)': [0.00962550646345379, 0.01311348785596405, 0.014054031573722888],
        }
    )
    pd.testing.assert_frame_equal(samples, correct, atol=1e-6)
    # Make cov matrix non-posdef
    model.modelfit_results.covariance_matrix['THETA(1)']['THETA(1)'] = -1
    with pytest.warns(UserWarning):
        sample_from_covariance_matrix(model, n=1, force_posdef_covmatrix=True)


def test_sample_individual_estimates(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    rng = np.random.default_rng(86)
    samples = sample_individual_estimates(model, seed=rng)
    assert len(samples) == 59 * 100
    assert list(samples.columns) == ['ETA(1)', 'ETA(2)']
    assert pytest.approx(samples.iloc[0]['ETA(1)'], 1e-5) == 0.21179186940672637
    assert pytest.approx(samples.iloc[0]['ETA(2)'], 1e-5) == -0.05771736555248238

    restricted = sample_individual_estimates(
        model, parameters=['ETA(2)'], samples_per_id=1, seed=rng
    )
    assert len(restricted) == 59
    assert restricted.columns == ['ETA(2)']
    assert pytest.approx(restricted.iloc[0]['ETA(2)'], 1e-5) == 0.06399039578129821
