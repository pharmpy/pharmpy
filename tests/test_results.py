import numpy as np
import pytest

from pharmpy import Model


def test_individual_parameter_statistics(testdata):
    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'pheno.mod')
    rng = np.random.default_rng(103)
    stats = model.modelfit_results.individual_parameter_statistics('CL/V', seed=rng)

    assert stats['mean'][0] == pytest.approx(0.004700589484324183)
    assert stats['variance'][0] == pytest.approx(8.086653508585209e-06)
    assert stats['stderr'][0] == pytest.approx(0.0035089729730046304, abs=1e-6)

    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'run1.mod')
    rng = np.random.default_rng(5678)
    stats = model.modelfit_results.individual_parameter_statistics('CL/V', seed=rng)
    assert stats['mean'][0] == pytest.approx(0.0049100899539843)
    assert stats['variance'][0] == pytest.approx(7.391076132098555e-07)
    assert stats['stderr'][0] == pytest.approx(0.0009425952783595735, abs=1e-6)

    covmodel = Model(testdata / 'nonmem' / 'secondary_parameters' / 'run2.mod')
    rng = np.random.default_rng(8976)
    stats = covmodel.modelfit_results.individual_parameter_statistics('K = CL/V', seed=rng)
    assert stats['mean']['K', 'median'] == pytest.approx(0.004525842355027405)
    assert stats['variance']['K', 'median'] == pytest.approx(2.9540381716908423e-06)
    assert stats['stderr']['K', 'median'] == pytest.approx(0.001804371451706786, abs=1e-6)
    assert stats['mean']['K', 'p5'] == pytest.approx(0.0033049497924269385)
    assert stats['variance']['K', 'p5'] == pytest.approx(1.5730213328583985e-06)
    assert stats['stderr']['K', 'p5'] == pytest.approx(0.0013102577338191103, abs=1e-6)
    assert stats['mean']['K', 'p95'] == pytest.approx(0.014625434302866478)
    assert stats['variance']['K', 'p95'] == pytest.approx(3.090546438695198e-05)
    assert stats['stderr']['K', 'p95'] == pytest.approx(0.0069971678916412716, abs=1e-6)


def test_pk_parameters(testdata):
    model = Model(testdata / 'nonmem' / 'models' / 'mox1.mod')
    rng = np.random.default_rng(103)
    df = model.modelfit_results.pk_parameters(seed=rng)
    assert df['mean'].loc['t_max', 'median'] == pytest.approx(1.5999856886869577)
    assert df['variance'].loc['t_max', 'median'] == pytest.approx(0.29728565293669557)
    assert df['stderr'].loc['t_max', 'median'] == pytest.approx(0.589128711884761)
    assert df['mean'].loc['C_max_dose', 'median'] == pytest.approx(0.6306393134647171)
    assert df['variance'].loc['C_max_dose', 'median'] == pytest.approx(0.012194951111832641)
    assert df['stderr'].loc['C_max_dose', 'median'] == pytest.approx(0.11328030491204867)
