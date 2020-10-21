import numpy as np
import pandas as pd
from pytest import approx

import pharmpy.methods.cdd.results as cdd
from pharmpy import Model
from pharmpy.methods.psn_helpers import model_paths, options_from_command


def test_psn_cdd_options(testdata):
    opts = cdd.psn_cdd_options(testdata / 'nonmem' / 'cdd' / 'pheno_real_bin10')
    assert opts['case_column'] == 'ID'
    assert (
        opts['model_path']
        == '/home/PMX/Projects/github/pharmpy/tests/testdata/nonmem/pheno_real.mod'
    )


def test_psn_options():
    cmd = 'cmd_line: /opt/bin/cdd pheno.mod -ignore -bins=10 -case=WGT -clean=3 -dir=caseWGTbin10'
    assert options_from_command(cmd) == {
        'ignore': '',
        'bins': '10',
        'case': 'WGT',
        'clean': '3',
        'dir': 'caseWGTbin10',
    }


def test_cdd_psn(testdata):
    path = testdata / 'nonmem' / 'cdd' / 'pheno_real_bin10'
    base_model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    cdd_models = [Model(p) for p in model_paths(path, 'cdd_*.mod')]
    skipped_individuals = cdd.psn_cdd_skipped_individuals(path)

    cdd_bin_id = cdd.calculate_results(base_model, cdd_models, 'ID', skipped_individuals)

    correct = pd.DataFrame(
        {
            'cook_score': [
                0.6002247,
                0.7681936,
                1.054763,
                0.8994126,
                0.9405288,
                0.6945769,
                0.7631572,
                0.7947513,
                0.7535628,
                0.5651823,
            ],
            'jackknife_cook_score': [
                0.55860759,
                0.90000807,
                1.00565998,
                0.92285602,
                0.74943641,
                0.83529442,
                0.89191965,
                0.85322234,
                0.6674856,
                0.70395028,
            ],
            'delta_ofv': [
                0.27067,
                0.35146,
                0.79115,
                0.84426,
                0.86966,
                0.37618,
                0.69805,
                0.49343,
                0.31184,
                0.31729,
            ],
            'covariance_ratio': [
                1.218597,
                1.389662,
                0.906991,
                2.152456,
                1.166288,
                1.214472,
                1.314591,
                1.085536,
                1.337584,
                1.342893,
            ],
            'skipped_individuals': [
                ['1', '2', '3', '4', '5', '6'],
                ['7', '8', '9', '10', '11', '12'],
                ['13', '14', '15', '16', '17', '18'],
                ['19', '20', '21', '22', '23', '24'],
                ['25', '26', '27', '28', '29', '30'],
                ['31', '32', '33', '34', '35', '36'],
                ['37', '38', '39', '40', '41', '42'],
                ['43', '44', '45', '46', '47', '48'],
                ['49', '50', '51', '52', '53', '54'],
                ['55', '56', '57', '58', '59'],
            ],
        },
        index=[
            'cdd_1',
            'cdd_2',
            'cdd_3',
            'cdd_4',
            'cdd_5',
            'cdd_6',
            'cdd_7',
            'cdd_8',
            'cdd_9',
            'cdd_10',
        ],
    )
    pd.testing.assert_frame_equal(cdd_bin_id.case_results, correct, rtol=1e-4)


def test_cdd_calculate_results(testdata):
    path = testdata / 'nonmem' / 'cdd' / 'pheno_real_bin10'
    skipped_individuals = cdd.psn_cdd_skipped_individuals(path)
    base_model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    cdd_model_paths = model_paths(path, 'cdd_*.mod')

    cdd_models = [Model(p) for p in cdd_model_paths]

    # Results for plain PsN run
    delta_ofv = cdd.compute_delta_ofv(base_model, cdd_models, skipped_individuals)

    assert delta_ofv == approx(
        [0.27067, 0.35146, 0.79115, 0.84426, 0.86966, 0.37618, 0.69805, 0.49343, 0.31184, 0.31729],
        abs=1e-5,
    )

    cdd_estimates = pd.DataFrame(
        data=[
            pd.Series(m.modelfit_results.parameter_estimates, name=m.name)
            for m in cdd_models
            if m.modelfit_results
        ]
    )

    jackknife = cdd.compute_jackknife_covariance_matrix(cdd_estimates)
    jackpsnvalues = [
        3.30415232e-08,
        -1.02486433e-06,
        -3.42829379e-07,
        -6.31160018e-07,
        -6.54383718e-08,
        4.16059589e-08,
        -1.02486433e-06,
        7.66862781e-04,
        -8.00400982e-04,
        7.94700855e-05,
        7.30565442e-06,
        4.83776339e-06,
        -3.42829379e-07,
        -8.00400982e-04,
        8.97549534e-03,
        -2.91263334e-04,
        2.17285280e-04,
        3.56099773e-05,
        -6.31160018e-07,
        7.94700855e-05,
        -2.91263334e-04,
        1.62624661e-04,
        -5.66902777e-05,
        3.77862212e-06,
        -6.54383718e-08,
        7.30565442e-06,
        2.17285280e-04,
        -5.66902777e-05,
        5.45538028e-05,
        -2.00496073e-06,
        4.16059589e-08,
        4.83776339e-06,
        3.56099773e-05,
        3.77862212e-06,
        -2.00496073e-06,
        2.62509124e-06,
    ]

    assert jackknife.values.flatten() == approx(jackpsnvalues, rel=1e-6)

    cook_scores = cdd.compute_cook_scores(
        base_model.modelfit_results.parameter_estimates,
        cdd_estimates,
        base_model.modelfit_results.covariance_matrix,
    )

    assert cook_scores == approx(
        [
            0.6002247,
            0.7681936,
            1.054763,
            0.8994126,
            0.9405288,
            0.6945769,
            0.7631572,
            0.7947513,
            0.7535628,
            0.5651823,
        ],
        abs=1e-5,
    )

    cov_ratios = cdd.compute_covariance_ratios(
        cdd_models, base_model.modelfit_results.covariance_matrix
    )

    assert cov_ratios == approx(
        [
            1.218597,
            1.389662,
            0.906991,
            2.152456,
            1.166288,
            1.214472,
            1.314591,
            1.085536,
            1.337584,
            1.342893,
        ],
        abs=1e-5,
    )

    # Replace three estimated cdd_models with fake models without estimates
    # and recompute results to verify handling of missing output
    cdd_models[0] = Model(path / 'm1' / 'rem_1.mod')
    cdd_models[1] = Model(path / 'm1' / 'rem_2.mod')
    cdd_models[3] = Model(path / 'm1' / 'rem_4.mod')

    res = cdd.calculate_results(base_model, cdd_models, 'ID', skipped_individuals)

    assert res.case_results.delta_ofv.values == approx(
        [np.nan, np.nan, 0.79115, np.nan, 0.86966, 0.37618, 0.69805, 0.49343, 0.31184, 0.31729],
        abs=1e-5,
        nan_ok=True,
    )

    # pandas.testing.assert_frame_equal  rtol atol

    assert res.case_results.covariance_ratio.values == approx(
        [
            np.nan,
            np.nan,
            0.906991,
            np.nan,
            1.166288,
            1.214472,
            1.314591,
            1.085536,
            1.337584,
            1.342893,
        ],
        abs=1e-5,
        nan_ok=True,
    )

    assert res.case_results.cook_score.values == approx(
        [
            np.nan,
            np.nan,
            1.054763,
            np.nan,
            0.9405288,
            0.6945769,
            0.7631572,
            0.7947513,
            0.7535628,
            0.5651823,
        ],
        abs=1e-5,
        nan_ok=True,
    )


# tox -e py38 -- pytest -s test_parameter.py
