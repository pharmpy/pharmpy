from pharmpy import Model
from pharmpy.plugins.nlmixr import convert_model


def test_model(testdata):
    nmmodel = Model(testdata / 'nonmem' / 'pheno.mod')
    model = convert_model(nmmodel)
    assert 'ini' in str(model)


def test_pheno_real(testdata):
    nmmodel = Model(testdata / 'nonmem' / 'pheno_real.mod')
    model = convert_model(nmmodel)
    print(str(model))
    assert '} else {' in str(model)
