
import pytest

from pysn.parameter_model import Scalar, CovarianceMatrix


@pytest.fixture
def pheno_params():
    return {
        'CL': Scalar(0.00469307, fix=False, lower=0),
        'V': Scalar(1.00916, fix=False, lower=0),
        'APGR_V': Scalar(.1, fix=False, lower=-0.99),
        'OMEGA_1': CovarianceMatrix(data=(
            (Scalar(0.0309626), Scalar(0, None)),
            (Scalar(0, None), Scalar(0.031128)),
        ))
    }


def test_pheno_inits(nonmem, pheno_real, pheno_params):
    model = nonmem.Model(pheno_real)
    pop = model.parameters.population
    assert pop[0] == pheno_params['CL']
    assert pop[1] == pheno_params['V']
    assert pop[2] == pheno_params['APGR_V']
    assert pop[3] == pheno_params['OMEGA_1']
