import numpy as np
import pytest

from pharmpy import Model


def test_individual_parameter_statistics(testdata):
    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'pheno.mod')
    np.random.seed(103)
    stats = model.modelfit_results.individual_parameter_statistics('CL/V')

    assert stats['mean'][0] == pytest.approx(0.004703788314066429)
    assert stats['variance'][0] == pytest.approx(8.098952786418367e-06)
    assert stats['stderr'][0] == pytest.approx(0.003412994638115151, abs=1e-6)

    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'run1.mod')
    np.random.seed(5678)
    stats = model.modelfit_results.individual_parameter_statistics('CL/V')
    assert stats['mean'][0] == pytest.approx(0.0049080779503763595)
    assert stats['variance'][0] == pytest.approx(7.374752150926833e-07)
    assert stats['stderr'][0] == pytest.approx(0.0009118101946374003, abs=1e-6)

    covmodel = Model(testdata / 'nonmem' / 'secondary_parameters' / 'run2.mod')
    np.random.seed(8976)
    stats = covmodel.modelfit_results.individual_parameter_statistics('CL/V')
    assert stats['mean']['median'] == pytest.approx(0.004529896229540338)
    assert stats['variance']['median'] == pytest.approx(2.9660484933157016e-06)
    assert stats['stderr']['median'] == pytest.approx(0.0017836906149599476, abs=1e-6)
    assert stats['mean']['p5'] == pytest.approx(0.003304609270246549)
    assert stats['variance']['p5'] == pytest.approx(1.573924162061832e-06)
    assert stats['stderr']['p5'] == pytest.approx(0.0013977445360638522, abs=1e-6)
    assert stats['mean']['p95'] == pytest.approx(0.014618964871102277)
    assert stats['variance']['p95'] == pytest.approx(3.0746876963740084e-05)
    assert stats['stderr']['p95'] == pytest.approx(0.007186921810266427, abs=1e-6)


def test_pk_parameters(testdata):
    model = Model(testdata / 'nonmem' / 'models' / 'mox1.mod')
    np.random.seed(103)
    df = model.modelfit_results.pk_parameters()
    assert df['mean'].loc['Tmax', 'median'] == pytest.approx(1.6012919051165513)
    assert df['variance'].loc['Tmax', 'median'] == pytest.approx(0.2992886152583829)
    assert df['stderr'].loc['Tmax', 'median'] == pytest.approx(0.5565206657252156)
    assert df['mean'].loc['Cmax/dose', 'median'] == pytest.approx(0.630597022391845)
    assert df['variance'].loc['Cmax/dose', 'median'] == pytest.approx(0.012201307770739035)
    assert df['stderr'].loc['Cmax/dose', 'median'] == pytest.approx(0.11151092285209177)
