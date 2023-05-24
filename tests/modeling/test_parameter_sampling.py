import numpy as np
import pandas as pd
import pytest

from pharmpy.modeling import (
    create_rng,
    sample_individual_estimates,
    sample_parameters_from_covariance_matrix,
    sample_parameters_uniformly,
)
from pharmpy.tools import read_modelfit_results


def test_create_rng():
    rng = create_rng(23)
    assert rng.standard_normal() == 0.5532605888887387

    rng = create_rng(23.0)
    assert rng.standard_normal() == 0.5532605888887387


def test_sample_parameters_uniformly(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    rng = create_rng(23)
    df = sample_parameters_uniformly(model, res.parameter_estimates, n=3, rng=rng)
    assert df['PTVCL'][0] == 0.004877674495376137


def test_sample_parameter_from_covariance_matrix(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    rng = np.random.default_rng(318)
    pe = res.parameter_estimates
    cm = res.covariance_matrix
    samples = sample_parameters_from_covariance_matrix(
        model,
        pe,
        cm,
        n=3,
        rng=rng,
    )
    correct = pd.DataFrame(
        {
            'PTVCL': [0.004489330033579095, 0.004866193232279955, 0.004619661658761273],
            'PTVV': [0.9720563045663096, 1.0217868717352445, 0.9662036500731115],
            'THETA_3': [0.19927467608338267, 0.237140948854298, 0.1979609848931148],
            'IVCL': [0.012012933626520568, 0.03859989956899462, 0.03228178862778379],
            'IVV': [0.03718187653238525, 0.036766142234483934, 0.02433717922068797],
            'SIGMA_1_1': [0.00962550646345379, 0.01311348785596405, 0.014054031573722888],
        }
    )
    pd.testing.assert_frame_equal(samples, correct, atol=1e-6)
    # Make cov matrix non-posdef
    cm2 = cm.copy()
    cm2['PTVCL']['PTVCL'] = -1
    with pytest.warns(UserWarning):
        sample_parameters_from_covariance_matrix(
            model,
            pe,
            cm2,
            n=1,
            force_posdef_covmatrix=True,
        )


def test_sample_individual_estimates(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    rng = np.random.default_rng(86)
    ie = res.individual_estimates
    iec = res.individual_estimates_covariance
    samples = sample_individual_estimates(model, ie, iec, rng=rng)
    assert len(samples) == 59 * 100
    assert list(samples.columns) == ['ETA_1', 'ETA_2']
    assert pytest.approx(samples.iloc[0]['ETA_1'], 1e-5) == 0.21179186940672637
    assert pytest.approx(samples.iloc[0]['ETA_2'], 1e-5) == -0.05771736555248238

    restricted = sample_individual_estimates(
        model, ie, iec, parameters=['ETA_2'], samples_per_id=1, rng=rng
    )
    assert len(restricted) == 59
    assert restricted.columns == ['ETA_2']
    assert pytest.approx(restricted.iloc[0]['ETA_2'], 1e-5) == 0.06399039578129821
