from pharmpy import Model
from pharmpy.modeling import ninds


def test_read_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    assert ninds(model) == 59
