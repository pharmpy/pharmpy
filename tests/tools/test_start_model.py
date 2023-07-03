from pharmpy.model import Bolus, Infusion
from pharmpy.tools.amd.funcs import create_start_model


def test_create_start_model(testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    model = create_start_model(path, modeltype='pk_iv')
    sep = model.datainfo.separator
    assert sep == "\\s+"
    assert len(model.dataset.columns) == 8
    assert len(model.parameters) == 6
    assert 'POP_CL' in model.parameters
    assert 'POP_MAT' not in model.parameters
    assert model.statements.ode_system.dosing_compartment[0].dose == Bolus.create("AMT")
    model = create_start_model(path, modeltype='pk_oral')
    assert 'IIV_MAT' in model.parameters
    assert 'POP_CL' in model.parameters
    assert 'POP_MAT' in model.parameters

    path_2 = testdata / 'nonmem' / 'modeling' / 'pheno_zero_order.csv'
    model = create_start_model(path_2, modeltype='pk_iv')
    assert model.statements.ode_system.dosing_compartment[0].dose == Infusion.create(
        "AMT", duration="D1"
    )
