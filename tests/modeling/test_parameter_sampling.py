import numpy as np
import pytest

from pharmpy.deps import pandas as pd
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
    df = sample_parameters_uniformly(model, res.parameter_estimates, n=3, seed=rng)
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
        seed=rng,
    )
    correct = pd.DataFrame(
        {
            'PTVCL': [0.004799509454006033, 0.004876509525000064, 0.004797517013599597],
            'PTVV': [0.9921305867883562, 0.9327192430072004, 0.9982429841528833],
            'THETA_3': [0.043042621151496346, 0.2444558430725377, 0.16779782800563542],
            'IVCL': [0.04254874182372164, 0.03036056519335301, 0.025479286881310285],
            'IVV': [0.03563760569946331, 0.02807739530058964, 0.02727242745988005],
            'SIGMA_1_1': [0.015486135732046637, 0.009797367739126846, 0.014683495078833424],
        }
    )
    pd.testing.assert_frame_equal(samples, correct, atol=1e-6)
    # Make cov matrix non-posdef
    cm2 = cm.copy()
    cm2.loc['PTVCL', 'PTVCL'] = -1
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
    samples = sample_individual_estimates(model, ie, iec, seed=rng)
    assert len(samples) == 59 * 100
    assert list(samples.columns) == ['ETA_1', 'ETA_2']
    assert pytest.approx(samples.iloc[0]['ETA_1'], 1e-5) == 0.21179186940672637
    assert pytest.approx(samples.iloc[0]['ETA_2'], 1e-5) == -0.05771736555248238

    restricted = sample_individual_estimates(
        model, ie, iec, parameters=['ETA_2'], samples_per_id=1, seed=rng
    )
    assert len(restricted) == 59
    assert restricted.columns == ['ETA_2']
    assert pytest.approx(restricted.iloc[0]['ETA_2'], 1e-5) == 0.06399039578129821
