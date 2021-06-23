import pytest

from pharmpy.tools.crossval import psn_crossval_results


def test_psn_crossval_results(testdata):
    res = psn_crossval_results(testdata / 'psn' / 'crossval_dir1')
    assert list(res.runs['estimation_ofv']) == pytest.approx(
        [545.863847, 512.953185, 547.843632, 579.109635]
    )
    assert list(res.runs['prediction_ofv']) == pytest.approx(
        [186.192727, 223.694832, 184.022535, 153.596550]
    )
    assert res.prediction_ofv_sum == pytest.approx(747.506644594254)
