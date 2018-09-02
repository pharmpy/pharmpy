
import pytest

from pysn.generic import PopulationParameter


@pytest.fixture
def pheno_params():
    return {
        'CL': PopulationParameter(lower=0, init=0.00469307, upper=float('INF'), fixed=False),
        'V': PopulationParameter(lower=0, init=1.00916, upper=float('INF'), fixed=False),
        'APGR_V': PopulationParameter(lower=-0.99, init=.1, upper=float('INF'), fixed=False),
    }


def test_pheno_inits(nonmem, pheno_real, pheno_params):
    model = nonmem.Model(pheno_real)
    pop = model.parameters.population
    assert pop[0] == pheno_params['CL']
    assert pop[1] == pheno_params['V']
    assert pop[2] == pheno_params['APGR_V']
