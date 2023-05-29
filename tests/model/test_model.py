from pathlib import Path

import pytest

from pharmpy.deps import sympy
from pharmpy.model import Assignment, Model
from pharmpy.model.external.nonmem.dataset import read_nonmem_dataset
from pharmpy.modeling import convert_model, create_symbol, load_example_model

tabpath = Path(__file__).resolve().parent.parent / 'testdata' / 'nonmem' / 'pheno_real_linbase.tab'
lincorrect = read_nonmem_dataset(
    tabpath,
    ignore_character='@',
    colnames=['ID', 'G11', 'G21', 'H11', 'CIPREDI', 'DV', 'PRED', 'RES', 'WRES'],
)


@pytest.mark.parametrize(
    'stem,force_numbering,symbol_name', [('DV', False, 'DV1'), ('X', False, 'X'), ('X', True, 'X1')]
)
def test_create_symbol(load_model_for_test, testdata, stem, force_numbering, symbol_name):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    symbol = create_symbol(model, stem=stem, force_numbering=force_numbering)

    assert symbol.name == symbol_name


def test_to_generic_model(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    nm_model = load_model_for_test(path)
    model = convert_model(nm_model, 'generic')

    assert model.parameters == nm_model.parameters
    assert model.random_variables == nm_model.random_variables
    assert model.name == nm_model.name
    assert model.statements == nm_model.statements
    assert type(model) == Model


def test_model_equality():
    pheno1 = load_example_model("pheno")
    assert pheno1 == pheno1

    pheno2 = load_example_model("pheno")
    assert pheno2 == pheno2

    assert pheno1 == pheno2

    pheno_linear1 = load_example_model("pheno_linear")
    assert pheno_linear1 == pheno_linear1

    pheno_linear2 = load_example_model("pheno_linear")
    assert pheno_linear2 == pheno_linear2

    assert pheno_linear1 == pheno_linear2

    assert pheno1 != pheno_linear1
    assert pheno1 != pheno_linear2
    assert pheno2 != pheno_linear1
    assert pheno2 != pheno_linear2

    assert hash(pheno1) != hash(pheno_linear1)


def test_replace(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    sset = model.statements
    cl = sset.find_assignment('CL')
    sset_new = sset.reassign(cl.symbol, cl.expression + sympy.Symbol('x'))
    with pytest.raises(ValueError, match='Symbol x is not defined'):
        model.replace(statements=sset_new)

    x_assignment = Assignment(sympy.Symbol('x'), sympy.Float(1))
    model.replace(statements=x_assignment + sset_new)

    sset_new = sset_new.before_odes + x_assignment + sset_new.ode_system + sset_new.after_odes
    with pytest.raises(ValueError, match='Symbol x defined after being used'):
        model.replace(statements=sset_new)

    sset_new = sset.reassign(cl.symbol, cl.expression + sympy.Symbol('TIME'))
    model.replace(statements=sset_new)


def test_dict(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    d = model.to_dict()
    model2 = Model.from_dict(d)
    assert model == model2
