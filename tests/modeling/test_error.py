from pharmpy import Model
from pharmpy.modeling import error_model


def test_fix_parameters(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    error_model(model, 'none')
    model.update_source()
    assert(str(model).split('\n')[11] == 'Y = F')
