from pharmpy import Model
from pharmpy.plugins.nlmixr import convert_model


def test_model(testdata):
    nmmodel = Model.create_model(testdata / 'nonmem' / 'pheno.mod')
    model = convert_model(nmmodel)
    assert 'ini' in model.model_code


def test_pheno_real(testdata):
    nmmodel = Model.create_model(testdata / 'nonmem' / 'pheno_real.mod')
    model = convert_model(nmmodel)
    assert '} else {' in model.model_code
