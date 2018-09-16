# -*- encoding: utf-8 -*-

import pytest

from pharmpy.parameters import CovarianceMatrix
from pharmpy.parameters import Scalar


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


def test_pheno_inits(nonmem, pheno, pheno_params):
    pop = pheno.parameters.inits
    assert pop[0] == pheno_params['CL']
    assert pop[1] == pheno_params['V']
    assert pop[2] == pheno_params['APGR_V']
    assert pop[3] == pheno_params['OMEGA_1']
