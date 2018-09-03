
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pysn.generic import PopulationParameter as Param


@pytest.fixture
def pheno_params():
    return {
        'CL': Param(0.00469307, fix=False, lower=0),
        'V': Param(1.00916, fix=False, lower=0),
        'APGR_V': Param(.1, fix=False, lower=-0.99),
        'OMEGA_1': np.array((
            (Param(0.0309626), Param(0, True)),
            (Param(0, True), Param(0.031128)),
        ))
    }


def test_pheno_inits(nonmem, pheno_real, pheno_params):
    model = nonmem.Model(pheno_real)
    pop = model.parameters.population
    assert pop[0] == pheno_params['CL']
    assert pop[1] == pheno_params['V']
    assert pop[2] == pheno_params['APGR_V']
    assert_array_equal(pop[3], pheno_params['OMEGA_1'])
