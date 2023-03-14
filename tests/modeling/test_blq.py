from pharmpy.deps import numpy as np
from pharmpy.model.model import update_datainfo
from pharmpy.modeling.blq import transform_blq


def test_transform_blq(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model_m4_float = transform_blq(model, 0.1)

    y_above_lloq = 'Y = F + EPS(1)*W'
    y_below_lloq = 'Y = (CUMD - CUMDZ)/(1 - CUMDZ)'

    assert y_above_lloq in model_m4_float.model_code
    assert y_below_lloq in model_m4_float.model_code
    assert 'LAPLACE' in model_m4_float.model_code
    assert 'DV.GE.LLOQ' in model_m4_float.model_code

    df_blq = model.dataset
    df_blq['BLQ'] = np.random.randint(0, 2, df_blq.shape[0])
    di_blq = update_datainfo(model.datainfo, df_blq)
    blq_col = di_blq['BLQ'].replace(type='blq')
    di_blq = di_blq.set_column(blq_col)
    model_blq = model.replace(dataset=df_blq, datainfo=di_blq)

    model_m4_blq = transform_blq(model_blq)

    assert y_above_lloq in model_m4_blq.model_code
    assert y_below_lloq in model_m4_blq.model_code
    assert 'LAPLACE' in model_m4_blq.model_code
    assert 'BLQ.EQ.1' in model_m4_blq.model_code

    df_lloq = model.dataset
    df_lloq['LLOQ'] = np.random.random(df_lloq.shape[0])
    di_lloq = update_datainfo(model.datainfo, df_lloq)
    lloq_col = di_lloq['LLOQ'].replace(type='lloq')
    di_lloq = di_lloq.set_column(lloq_col)
    model_lloq = model.replace(dataset=df_lloq, datainfo=di_lloq)

    model_m4_lloq = transform_blq(model_lloq)

    assert y_above_lloq in model_m4_lloq.model_code
    assert y_below_lloq in model_m4_lloq.model_code
    assert 'LAPLACE' in model_m4_lloq.model_code
    assert 'DV.GE.LLOQ' in model_m4_lloq.model_code
    assert 'LLOQ = ' not in model_m4_lloq.model_code

    model_m1_float = transform_blq(model, lloq=10.0, method='m1')
    assert len(model_m1_float.dataset) < len(model.dataset)
