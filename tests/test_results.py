import numpy as np
import pytest

from pharmpy import Model
from pharmpy.results import individual_parameter_statistics


def test_individual_parameter_statistics(testdata):
    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'pheno.mod')
    np.random.seed(103)
    stats = individual_parameter_statistics(model, 'CL/V')

    assert stats['mean'] == pytest.approx(0.004703788314066429)
    assert stats['variance'] == pytest.approx(8.098952786418367e-06)
    assert stats['stderr'] == pytest.approx(0.003412994638115151, abs=1e-6)

    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'run1.mod')
    np.random.seed(5678)
    stats = individual_parameter_statistics(model, 'CL/V')
    assert stats['mean'] == pytest.approx(0.0049080779503763595)
    assert stats['variance'] == pytest.approx(7.374752150926833e-07)
    assert stats['stderr'] == pytest.approx(0.0009118101946374003, abs=1e-6)
