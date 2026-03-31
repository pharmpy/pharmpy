import pytest

from pharmpy.basic import Unit
from pharmpy.model import AddColumn, Assignment, Drop
from pharmpy.modeling import (
    add_iiv,
    add_lag_time,
    convert_unit,
    create_basic_pk_model,
    create_joint_distribution,
    get_unit_of,
    remove_iiv,
    set_property,
    set_unit,
)


def test_get_unit_of(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_real.mod")
    assert get_unit_of(model, "Y") == Unit("mg/L")
    assert get_unit_of(model, "V") == Unit("L")
    assert get_unit_of(model, "WGT") == Unit("kg")
    assert get_unit_of(model, "CL") == Unit("L/h")
    assert get_unit_of(model, "ETA_1") == Unit(1)

    m2 = add_lag_time(model)
    assert get_unit_of(m2, "MDT") == Unit("h")

    m3 = create_joint_distribution(model, ("ETA_1", "ETA_2"))
    assert get_unit_of(m3, "IIV_CL_IIV_V") == Unit(1)

    m4 = remove_iiv(model, "ETA_1")
    m4 = add_iiv(m4, "CL", "add")
    m4 = create_joint_distribution(m4, ("ETA_CL", "ETA_2"))
    assert get_unit_of(m4, "ETA_CL") == Unit("L/h")
    assert get_unit_of(m4, "ETA_2") == Unit("1")
    assert get_unit_of(m4, "IIV_CL") == Unit("L^2/h^2")
    assert get_unit_of(m4, "IVV") == Unit("1")
    assert get_unit_of(m4, "IIV_V_IIV_CL") == Unit("L^2/h^2")

    m = create_basic_pk_model("ivoral")
    m = set_unit(m, "DV", "mg/L")
    m = set_unit(m, "AMT", "mg")
    m = set_unit(m, "TIME", "h")
    assert get_unit_of(m, "F_BIO") == Unit("")
    assert get_unit_of(m, "ETA_BIO") == Unit("")

    with pytest.raises(ValueError):
        get_unit_of(m, "UNKNOWNNAME")


def test_convert_unit(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    m2 = convert_unit(model, "WGT", "g")
    assert m2.statements[0] == Assignment.create("SCALED_WGT", "1000*WGT")
    assert len(m2.datainfo.provenance) == 1
    m2 = convert_unit(model, "WGT", "g", in_dataset=True)
    assert m2.statements[0].symbol.name == 'TVCL'
    assert m2.dataset['WGT'][1] == 1400.0
    assert Drop.create('WGT') in m2.datainfo.provenance
    assert AddColumn.create('WGT') in m2.datainfo.provenance
    assert len(m2.datainfo.provenance) == 3

    m2 = convert_unit(model, "WGT", "kg", in_dataset=True)
    assert model == m2
    assert len(m2.datainfo.provenance) == 1

    m2 = set_property(model, "DV", "molar_mass", 136.0)
    m2 = convert_unit(m2, "DV", "M")
    assert float(m2.statements.before_odes[-1].expression) == pytest.approx(
        7.35294117647059e-6, abs=1e-7
    )

    with pytest.raises(ValueError):
        convert_unit(m2, "TIME", "s")


@pytest.mark.parametrize(
    'variable,unit,dv_unit,amt_unit,values',
    [
        ("WGT", "g", None, None, (1400.0, 1400.0)),
        ("AMT", "ug", "ug/L", "ug", (25000.0, 0.0)),
        ("AMT", "mg", "mg/L", "mg", (25.0, 0.0)),
        ("DV", "g/L", "g/L", "g", (0.0, 0.0173)),
        ("DV", "g/mL", "g/mL", "g", (0.0, 0.0000173)),
        ("DV", "mg/mL", "mg/mL", "mg", (0.0, 0.0173)),
    ],
)
def test_convert_unit_in_dataset(
    load_example_model_for_test, variable, unit, dv_unit, amt_unit, values
):
    model = load_example_model_for_test("pheno")
    m2 = convert_unit(model, variable, unit, in_dataset=True)
    assert m2.datainfo[variable].variable.properties['unit'] == Unit(unit)
    if dv_unit is not None:
        assert m2.datainfo.dv_column.variable.properties['unit'] == Unit(dv_unit)
    if amt_unit is not None:
        assert m2.datainfo.typeix['dose'][0].variable.properties['unit'] == Unit(amt_unit)
    assert m2.dataset[variable].iloc[0] == values[0]
    assert m2.dataset[variable].iloc[1] == values[1]
