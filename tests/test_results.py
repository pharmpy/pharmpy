import numpy as np
import pytest

from pharmpy import Model
from pharmpy.results import individual_parameter_statistics


def test_individual_parameter_statistics(testdata):
    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'pheno.mod')
    np.random.seed(103)
    stats = individual_parameter_statistics(model, 'CL/V')

    assert stats['mean'] == pytest.approx(0.00470525776968202)
    assert stats['variance'] == pytest.approx(8.12398122254498e-6)
    assert stats['stderr'] == pytest.approx(0.00344872, abs=1e-5)

    model = Model(testdata / 'nonmem' / 'secondary_parameters' / 'run1.mod')
    np.random.seed(5678)
    # stats = individual_parameter_statistics(model, 'CL/V')
