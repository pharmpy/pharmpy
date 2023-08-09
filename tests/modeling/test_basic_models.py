from pharmpy.internals.immutable import frozenmapping
from pharmpy.modeling import convert_model, create_basic_pk_model, set_zero_order_absorption


def test_create_basic_pk_model(testdata):
    model = create_basic_pk_model('iv')
    assert len(model.parameters) == 6

    model = create_basic_pk_model('oral')
    # print(model.parameters)
    assert len(model.parameters) == 8

    dataset_path = testdata / 'nonmem/pheno.dta'
    model = create_basic_pk_model(
        modeltype='oral', dataset_path=dataset_path, cl_init=0.01, vc_init=1.0, mat_init=0.1
    )
    model = convert_model(model, 'nonmem')
    run1 = set_zero_order_absorption(model)
    assert not model.dataset.empty
    assert not run1.dataset.empty
    assert type(model.dependent_variables) == frozenmapping
