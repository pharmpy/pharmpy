
import pytest

from pysn.generic import PopulationParameter


@pytest.fixture
def pheno_params():
    return {
        'CL': PopulationParameter(0.00469307, fix=False, lower=0),
        'V': PopulationParameter(1.00916, fix=False, lower=0),
        'APGR_V': PopulationParameter(.1, fix=False, lower=-0.99),
    }


def test_pheno_inits(nonmem, pheno_real, pheno_params):
    model = nonmem.Model(pheno_real)
    pop = model.parameters.population
    assert pop[0] == pheno_params['CL']
    assert pop[1] == pheno_params['V']
    assert pop[2] == pheno_params['APGR_V']
