from pharmpy import Model
from pharmpy.modeling import error_model


def test_remove_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    error_model(model, 'none')
    model.update_source()
    assert(str(model).split('\n')[11] == 'Y = F')


def test_additive_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    error_model(model, 'additive')
    model.update_source()
    assert(str(model).split('\n')[11] == 'Y = EPS(1) + F')
    assert(str(model).split('\n')[17] == '$SIGMA  0.1 ; sigma')
