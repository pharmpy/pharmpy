from pharmpy.modeling.blq import transform_blq


def test_transform_blq(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = transform_blq(model, 0.1)

    y_above_lloq = 'Y = F + EPS(1)*W'
    y_below_lloq = 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'

    assert y_above_lloq in model.model_code
    assert y_below_lloq in model.model_code
    assert 'LAPLACE' in model.model_code
    assert 'DV.GE.LLOQ' in model.model_code
