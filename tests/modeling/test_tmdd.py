import pytest

from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.model import Assignment
from pharmpy.modeling import (
    add_peripheral_compartment,
    create_basic_pk_model,
    create_rng,
    set_mixed_mm_fo_elimination,
    set_tmdd,
)
from pharmpy.modeling.tmdd import _validate_dv_types


def S(x):
    return Expr.symbol(x)


def test_full(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="full")
    assert model.statements.ode_system.eqs[0].rhs == Expr(
        'A_CENTRAL(t)*(-A_TARGET(t)*KON/V - CL/V) + A_COMPLEX(t)*KOFF'
    )


def test_qss(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="qss")
    assert model.statements.ode_system.eqs[0].rhs == Expr(
        '-A_CENTRAL(t)*CL*LAFREE/V - A_TARGET(t)*KINT*LAFREE/(KD + LAFREE) - CL*LAFREE/V'
    )


def test_qss_1c(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_peripheral_compartment(model)
    tmdd_model = set_tmdd(model, type="qss")
    assert (
        str(tmdd_model.statements.ode_system.eqs[0].rhs)
        == """-CL*LAFREE/V1 - KINT*LAFREE*A_TARGET(t)/(KD + LAFREE) - LAFREE*QP1/V1 + QP1*A_PERIPHERAL1(t)/VP1"""
    )
    assert (
        str(tmdd_model.statements.ode_system.eqs[1].rhs)
        == """LAFREE*QP1/V1 - QP1*A_PERIPHERAL1(t)/VP1"""
    )
    assert (
        str(tmdd_model.statements.ode_system.eqs[2].rhs)
        == """KSYN*V1 + (-KDEG - LAFREE*(-KDEG + KINT)/(KD + LAFREE))*A_TARGET(t)"""
    )

    model2 = add_peripheral_compartment(model)
    tmdd_model = set_tmdd(model2, type="qss")


def test_cr(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="cr")
    assert model.statements.ode_system.eqs[1].rhs == Expr(
        '(-KINT - KOFF)*A_COMPLEX(t) + (KON*R_0 - KON*A_COMPLEX(t)/V)*A_CENTRAL(t)'
    )


def test_crib(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="crib")
    assert model.statements.ode_system.eqs[1].rhs == Expr(
        '-KINT*A_COMPLEX(t) + (KON*R_0 - KON*A_COMPLEX(t)/V)*A_CENTRAL(t)'
    )


def test_ib(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="ib")
    assert model.statements.ode_system.eqs[2].rhs == Expr(
        '-KDEG*A_TARGET(t) - KON*A_CENTRAL(t)*A_TARGET(t)/V + KSYN*V'
    )


def test_mmapp(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_peripheral_compartment(model)
    model = set_tmdd(model, type="mmapp")
    assert (
        str(model.statements.ode_system.eqs[0].rhs)
        == """QP1*A_PERIPHERAL1(t)/VP1 + (-CL/V1 - KINT*A_TARGET(t)/(KM + A_CENTRAL(t)/V1) - QP1/V1)*A_CENTRAL(t)"""
    )


def test_wagner(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_peripheral_compartment(model)
    model = set_tmdd(model, type="wagner")


def test_tmdd_with_mixed_fo_mm_elimination(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = add_peripheral_compartment(model)
    model = set_mixed_mm_fo_elimination(model)
    qss = set_tmdd(model, 'qss')
    assert qss


def test_wagner_1c(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_peripheral_compartment(model)
    wagner = set_tmdd(model, type="wagner")
    assert (
        str(wagner.statements.ode_system.eqs[0].rhs)
        == """-CL*LAFREE/V1 + KINT*LAFREE - KINT*A_CENTRAL(t) - LAFREE*QP1/V1 + QP1*A_PERIPHERAL1(t)/VP1"""
    )
    assert (
        str(wagner.statements.ode_system.eqs[1].rhs)
        == """LAFREE*QP1/V1 - QP1*A_PERIPHERAL1(t)/VP1"""
    )

    model = add_peripheral_compartment(model)
    wagner = set_tmdd(model, type="wagner")
    assert (
        str(wagner.statements.ode_system.eqs[1].rhs)
        == """LAFREE*QP1/V1 - QP1*A_PERIPHERAL1(t)/VP1"""
    )


@pytest.mark.parametrize(
    ('model_name', 'dv_types', 'expected'),
    (
        (
            'full',
            {'drug': 1, 'target': 2, 'complex': 3},
            {S('Y'): 1, S('Y_TARGET'): 2, S('Y_COMPLEX'): 3},
        ),
        (
            'qss',
            {'drug': 1, 'target': 2, 'complex': 3},
            {S('Y'): 1, S('Y_TARGET'): 2, S('Y_COMPLEX'): 3},
        ),
        (
            'ib',
            {'drug': 1, 'target': 2, 'complex': 3},
            {S('Y'): 1, S('Y_TARGET'): 2, S('Y_COMPLEX'): 3},
        ),
        ('mmapp', {'drug': 1, 'target': 2}, {S('Y'): 1, S('Y_TARGET'): 2}),
        ('cr', {'drug': 1, 'complex': 2}, {S('Y'): 1, S('Y_COMPLEX'): 2}),
        ('wagner', {'drug': 1, 'complex': 2}, {S('Y'): 1, S('Y_COMPLEX'): 2}),
        ('crib', {'drug': 1, 'complex': 2}, {S('Y'): 1, S('Y_COMPLEX'): 2}),
        ('crib', {'drug': 1, 'complex': 3}, {S('Y'): 1, S('Y_COMPLEX'): 3}),
        (
            'full',
            {'target': 2, 'complex': 3},
            {S('Y'): 1, S('Y_TARGET'): 2, S('Y_COMPLEX'): 3},
        ),
        (
            'cr',
            {'drug': 1, 'complex': 2, 'target': 3},
            {S('Y'): 1, S('Y_COMPLEX'): 2},
        ),
        (
            'mmapp',
            {'drug': 1, 'target': 2, 'complex': 3},
            {S('Y'): 1, S('Y_TARGET'): 2},
        ),
    ),
    ids=repr,
)
def test_full_multiple_dvs(pheno_path, load_model_for_test, model_name, dv_types, expected):
    model = load_model_for_test(pheno_path)
    model = _add_random_dvids(model)
    model = set_tmdd(model, model_name, dv_types)
    assert model.dependent_variables == expected
    assert len(model.random_variables.epsilons) > 1


def test_multiple_dvs(load_model_for_test, pheno_path, testdata):
    central_amount = Expr.function("A_CENTRAL", S('t'))
    complex_amount = Expr.function("A_COMPLEX", S('t'))

    model = load_model_for_test(pheno_path)
    model = _add_random_dvids(model)
    model1 = set_tmdd(model, 'qss', {'complex': 2, 'target_tot': 3})
    ass1 = Assignment.create(S("F"), S("LAFREEF") / S("V"))
    assert model1.statements.find_assignment("F") == ass1
    assert S("Y_COMPLEX") in model1.statements.free_symbols
    assert S("Y_TOTTARGET") in model1.statements.free_symbols

    model2 = set_tmdd(model, 'qss', {'drug_tot': 1, 'complex': 2, 'target_tot': 3})
    ass2 = Assignment.create(S("F"), central_amount / S("V"))
    assert model2.statements.find_assignment("F") == ass2
    assert S("Y_COMPLEX") in model2.statements.free_symbols
    assert S("Y_TOTTARGET") in model2.statements.free_symbols

    model1 = set_tmdd(model, 'full', {'complex': 2, 'target_tot': 3})
    ass1 = Assignment.create(S("F"), central_amount / S("V"))
    assert model1.statements.find_assignment("F") == ass1
    assert S("Y_COMPLEX") in model1.statements.free_symbols
    assert S("Y_TOTTARGET") in model1.statements.free_symbols

    model2 = set_tmdd(model, 'full', {'drug_tot': 1, 'complex': 2, 'target_tot': 3})
    ass2 = Assignment.create(S("F"), (central_amount + complex_amount) / S("V"))
    assert model2.statements.find_assignment("F") == ass2
    assert S("Y_COMPLEX") in model2.statements.free_symbols
    assert S("Y_TOTTARGET") in model2.statements.free_symbols

    model1 = set_tmdd(model, 'cr', {'complex': 2})
    ass1 = Assignment.create(S("F"), central_amount / S("V"))
    assert model1.statements.find_assignment("F") == ass1
    assert S("Y_COMPLEX") in model1.statements.free_symbols

    model2 = set_tmdd(model, 'cr', {'drug_tot': 1, 'complex': 2})
    ass2 = Assignment.create(S("F"), (central_amount + complex_amount) / S("V"))
    assert model2.statements.find_assignment("F") == ass2
    assert S("Y_COMPLEX") in model2.statements.free_symbols

    model1 = set_tmdd(model, 'wagner', {'drug_tot': 1, 'complex': 2})
    ass1 = Assignment.create(S("F"), central_amount / S("V"))
    assert model1.statements.find_assignment("F") == ass1
    assert S("Y_COMPLEX") in model1.statements.free_symbols

    model1 = set_tmdd(model, 'mmapp', {'drug': 1, 'target_tot': 3})
    ass1 = Assignment.create(S("F"), central_amount / S("V"))
    assert model1.statements.find_assignment("F") == ass1
    assert S("Y_TOTTARGET") in model1.statements.free_symbols

    model = create_basic_pk_model('iv', testdata / 'nonmem' / 'pheno.dta')
    model = _add_random_dvids(model)
    model1 = set_tmdd(model, 'qss', {'complex': 2, 'target_tot': 3})
    ass1 = Assignment.create(S("IPRED"), S("LAFREEF") / S("VC"))
    assert model1.statements.find_assignment("IPRED") == ass1
    assert S("Y_COMPLEX") in model1.statements.free_symbols
    assert S("Y_TOTTARGET") in model1.statements.free_symbols

    model2 = set_tmdd(model, 'qss', {'drug_tot': 1, 'complex': 2, 'target_tot': 3})
    ass2 = Assignment.create(S("IPRED"), central_amount / S("VC"))
    assert model2.statements.find_assignment("IPRED") == ass2
    assert S("Y_COMPLEX") in model2.statements.free_symbols
    assert S("Y_TOTTARGET") in model2.statements.free_symbols


def test_validation(load_model_for_test, testdata):
    _validate_dv_types(dv_types={'drug_tot': 1, 'target_tot': 2, 'complex': 3})
    _validate_dv_types(dv_types={'drug': 1, 'target_tot': 2, 'complex': 3})
    _validate_dv_types(dv_types={'drug': 1, 'target': 2, 'complex': 3})

    with pytest.raises(AssertionError):
        _validate_dv_types(dv_types={'drug': 1, 'target': 1, 'complex': 2})
    with pytest.raises(
        ValueError, match='Only drug can have DVID = 1. Please choose another DVID.'
    ):
        _validate_dv_types(dv_types={'target': 1, 'complex': 2})
    with pytest.raises(
        ValueError,
        match='Invalid dv_types key "taget". Allowed keys are: "drug", "target", "complex", '
        '"drug_tot" and "target_tot".',
    ):
        _validate_dv_types(dv_types={'taget': 1})


def test_dataset_multiple_dvs(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = _add_random_dvids(model)

    model1 = set_tmdd(model, type="cr", dv_types={'drug': 1, 'target': 2})
    unique_values = model1.dataset['DVID'].unique()
    assert 1 in unique_values and 2 not in unique_values
    model2 = set_tmdd(model, type="cr", dv_types={'drug': 1, 'complex': 3})
    unique_values = model2.dataset['DVID'].unique()
    assert 1 in unique_values and 3 in unique_values
    model3 = set_tmdd(model, type="wagner", dv_types={'drug': 1, 'target': 2})
    unique_values = model3.dataset['DVID'].unique()
    assert 1 in unique_values and 2 not in unique_values
    model4 = set_tmdd(model, type="wagner", dv_types={'drug': 1, 'complex': 3})
    unique_values = model4.dataset['DVID'].unique()
    assert 1 in unique_values and 3 in unique_values
    model5 = set_tmdd(model, type="mmapp", dv_types={'drug': 1, 'complex': 2})
    unique_values = model5.dataset['DVID'].unique()
    assert 1 in unique_values and 2 not in unique_values
    model6 = set_tmdd(model, type="mmapp", dv_types={'drug': 1, 'target': 2})
    unique_values = model6.dataset['DVID'].unique()
    assert 1 in unique_values and 2 in unique_values
    model7 = set_tmdd(model, type="full", dv_types={'drug': 1, 'target': 2, 'complex': 3})
    unique_values = model7.dataset['DVID'].unique()
    assert 1 in unique_values and 2 in unique_values and 3 in unique_values


def _add_random_dvids(model):
    df = model.dataset.copy()
    rng = create_rng(23)
    df['DVID'] = np.array([rng.integers(1, 6) for i in range(len(df))])
    return model.replace(dataset=df)
