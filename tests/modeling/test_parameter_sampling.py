import numpy as np
import pandas as pd
import pytest

from pharmpy.modeling import (
    create_rng,
    load_example_model,
    sample_individual_estimates,
    sample_parameters_from_covariance_matrix,
    sample_parameters_uniformly,
)


def test_create_rng():
    rng = create_rng(23)
    assert rng.standard_normal() == 0.5532605888887387

    rng = create_rng(23.0)
    assert rng.standard_normal() == 0.5532605888887387


def test_sample_parameters_uniformly():
    model = load_example_model("pheno")
    rng = create_rng(23)
    df = sample_parameters_uniformly(
        model, model.modelfit_results.parameter_estimates, n=3, rng=rng
    )
    assert df['THETA(1)'][0] == 0.004877674495376137


def test_sample_parameter_from_covariance_matrix(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    rng = np.random.default_rng(318)
    pe = model.modelfit_results.parameter_estimates
    cm = model.modelfit_results.covariance_matrix
    samples = sample_parameters_from_covariance_matrix(
        model,
        pe,
        cm,
        n=3,
        rng=rng,
    )
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
    cm2 = cm.copy()
    cm2['THETA(1)']['THETA(1)'] = -1
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
    rng = np.random.default_rng(86)
    ie = model.modelfit_results.individual_estimates
    iec = model.modelfit_results.individual_estimates_covariance
    samples = sample_individual_estimates(model, ie, iec, rng=rng)
    assert len(samples) == 59 * 100
    assert list(samples.columns) == ['ETA(1)', 'ETA(2)']
    assert pytest.approx(samples.iloc[0]['ETA(1)'], 1e-5) == 0.21179186940672637
    assert pytest.approx(samples.iloc[0]['ETA(2)'], 1e-5) == -0.05771736555248238

    restricted = sample_individual_estimates(
        model, ie, iec, parameters=['ETA(2)'], samples_per_id=1, rng=rng
    )
    assert len(restricted) == 59
    assert restricted.columns == ['ETA(2)']
    assert pytest.approx(restricted.iloc[0]['ETA(2)'], 1e-5) == 0.06399039578129821
