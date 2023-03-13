from pharmpy.deps import numpy as np
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

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')

    df_blq = model.dataset
    df_blq['BLQ'] = np.random.randint(0, 2, df_blq.shape[0])
    model = model.replace(dataset=df_blq)
    blq_col = model.datainfo['BLQ'].replace(type='blq')
    di_blq = model.datainfo.set_column(blq_col)
    model = model.replace(datainfo=di_blq)

    model_blq = transform_blq(model)

    assert y_above_lloq in model_blq.model_code
    assert y_below_lloq in model_blq.model_code
    assert 'LAPLACE' in model_blq.model_code
    assert 'BLQ.EQ.1' in model_blq.model_code

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')

    df_lloq = model.dataset
    df_lloq['LLOQ'] = np.random.random(df_lloq.shape[0])
    model = model.replace(dataset=df_lloq)
    lloq_col = model.datainfo['LLOQ'].replace(type='lloq')
    di_lloq = model.datainfo.set_column(lloq_col)
    model = model.replace(datainfo=di_lloq)

    model_lloq = transform_blq(model)

    assert y_above_lloq in model_lloq.model_code
    assert y_below_lloq in model_lloq.model_code
    assert 'LAPLACE' in model_lloq.model_code
    assert 'DV.GE.LLOQ' in model_lloq.model_code
    assert 'LLOQ = ' not in model_lloq.model_code
