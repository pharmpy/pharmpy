from pharmpy.tools.resmod.results import psn_resmod_results


def test_resmod_results(testdata):
    res = psn_resmod_results(testdata / 'psn' / 'resmod_dir1')
    assert list(res.models['dOFV']) == [
        -1.31,
        -3.34,
        -13.91,
        -18.54,
        -8.03,
        -4.20,
        -0.25,
        -1.17,
        -0.00,
        -0.09,
        -2.53,
        -3.12,
        -3.60,
        -25.62,
        -7.66,
        -0.03,
        -5.53,
    ]


def test_resmod_results_dvid(testdata):
    res = psn_resmod_results(testdata / 'psn' / 'resmod_dir2')
    df = res.models
    assert df['dOFV'].loc[1, '1', 'autocorrelation'] == -0.74
    assert df['dOFV'].loc[1, 'sum', 'tdist'] == -35.98
