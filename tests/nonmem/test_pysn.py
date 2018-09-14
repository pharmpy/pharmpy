
from copy import deepcopy

from pysn import Model


def test_model(pheno_real):
    with open(str(pheno_real), 'r') as f:
        buf = f.read()
    pheno = Model(pheno_real)
    assert str(pheno) == buf


def test_model_copy(pheno_real):
    pheno = Model(pheno_real)
    copy = deepcopy(pheno)
    assert pheno is not copy
    assert pheno.path.samefile(copy.path)
    assert pheno.path is not copy.path
    for api in ['input', 'output', 'parameters', 'execute']:
        assert getattr(pheno, api) is not getattr(copy, api)
