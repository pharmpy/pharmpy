from pharmpy.model import Bolus, Infusion
from pharmpy.modeling import create_basic_pk_model


def test_create_start_model(testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    model = create_basic_pk_model(dataset_path=path, administration='iv')
    sep = model.datainfo.separator
    assert sep == "\\s+"
    assert len(model.dataset.columns) == 8
    assert len(model.parameters) == 6
    assert 'POP_CL' in model.parameters
    assert 'POP_MAT' not in model.parameters
    assert model.statements.ode_system.dosing_compartments[0].doses[0] == Bolus.create("AMT")
    model = create_basic_pk_model(dataset_path=path, administration='oral')
    assert 'IIV_MAT' in model.parameters
    assert 'POP_CL' in model.parameters
    assert 'POP_MAT' in model.parameters

    path_2 = testdata / 'nonmem' / 'modeling' / 'pheno_zero_order.csv'
    model = create_basic_pk_model(dataset_path=path_2, administration='iv')
    assert model.statements.ode_system.dosing_compartments[0].doses[0] == Infusion.create(
        "AMT", duration="D1"
    )
