from pharmpy import Model
from pharmpy.modeling import additive_error, combined_error, proportional_error, remove_error


def test_remove_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    remove_error(model)
    model.update_source()
    assert str(model).split('\n')[11] == 'Y = F'


def test_additive_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    additive_error(model)
    model.update_source()
    assert str(model).split('\n')[11] == 'Y = EPS(1) + F'
    assert str(model).split('\n')[17] == '$SIGMA  0.1 ; sigma'


def test_proportional_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    proportional_error(model)
    model.update_source()
    assert str(model).split('\n')[11] == 'Y=F+F*EPS(1)'
    assert str(model).split('\n')[17] == '$SIGMA  0.1 ; sigma'


def test_combined_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    combined_error(model)
    model.update_source()
    assert str(model).split('\n')[11] == 'Y = EPS(1)*F + EPS(2) + F'
    assert str(model).split('\n')[17] == '$SIGMA  0.1 ; sigma_prop'
    assert str(model).split('\n')[18] == '$SIGMA  0.1 ; sigma_add'
