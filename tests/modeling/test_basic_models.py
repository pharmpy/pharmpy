from pharmpy.internals.immutable import frozenmapping
from pharmpy.modeling import convert_model, create_basic_pk_model, set_zero_order_absorption


def test_create_basic_pk_model(testdata):
    model = create_basic_pk_model('iv')
    assert len(model.parameters) == 6

    model = create_basic_pk_model('oral')
    assert len(model.parameters) == 8

    dataset_path = testdata / 'nonmem/pheno.dta'
    model = create_basic_pk_model(
        administration='oral', dataset_path=dataset_path, cl_init=0.01, vc_init=1.0, mat_init=0.1
    )
    model = convert_model(model, 'nonmem')
    run1 = set_zero_order_absorption(model)
    assert not model.dataset.empty
    assert not run1.dataset.empty
    assert isinstance(model.dependent_variables, frozenmapping)

    dataset_path = testdata / 'nonmem/pheno_pd.csv'
    model = create_basic_pk_model(
        administration='oral', dataset_path=dataset_path, cl_init=0.01, vc_init=1.0, mat_init=0.1
    )
    assert not model.dataset.empty
    assert model.datainfo._separator == r'\s+'

    model = create_basic_pk_model('ivoral')
    assert len(model.statements.ode_system.dosing_compartments) == 2
    assert model.datainfo.id_column
