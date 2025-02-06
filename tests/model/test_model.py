import re
from pathlib import Path

import pytest

from pharmpy.basic import Expr
from pharmpy.deps import pandas as pd
from pharmpy.model import (
    Assignment,
    ExecutionSteps,
    JointNormalDistribution,
    Model,
    Parameter,
    Parameters,
    RandomVariables,
    Statements,
    get_and_check_dataset,
    get_and_check_odes,
)
from pharmpy.model.external.nonmem.dataset import read_nonmem_dataset
from pharmpy.model.model import ModelInternals
from pharmpy.modeling import convert_model, create_basic_pk_model, create_symbol

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


def test_init_model_internals():
    mi = ModelInternals()
    assert isinstance(mi, ModelInternals)


def test_to_generic_model(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    nm_model = load_model_for_test(path)
    model = convert_model(nm_model, 'generic')

    assert model.parameters == nm_model.parameters
    assert model.random_variables == nm_model.random_variables
    assert model.name == nm_model.name
    assert model.statements == nm_model.statements
    assert isinstance(model, Model)


def test_model_equality(load_example_model_for_test):
    pheno1 = load_example_model_for_test("pheno")
    assert pheno1 == pheno1

    pheno2 = load_example_model_for_test("pheno")
    assert pheno2 == pheno2

    assert pheno1 == pheno2

    pheno_linear1 = load_example_model_for_test("pheno_linear")
    assert pheno_linear1 == pheno_linear1

    pheno_linear2 = load_example_model_for_test("pheno_linear")
    assert pheno_linear2 == pheno_linear2

    assert pheno_linear1 == pheno_linear2

    assert pheno1 != pheno_linear1
    assert pheno1 != pheno_linear2
    assert pheno2 != pheno_linear1
    assert pheno2 != pheno_linear2

    assert hash(pheno1) != hash(pheno_linear1)

    assert pheno1 != 'model'

    model1 = Model.create('model1')
    model2 = Model.create('model2')
    assert model1 == model2

    model2 = Model.create('model2', random_variables=pheno1.random_variables)
    assert model1 != model2

    ass = Assignment.create('a', 0)
    model2 = Model.create('model2', statements=model1.statements + ass)
    assert model1 != model2

    model2 = Model.create('model2', dependent_variables=pheno1.dependent_variables)
    assert model1 != model2

    model2 = Model.create('model2', observation_transformation=pheno1.observation_transformation)
    assert model1 != model2

    model2 = Model.create('model2', datainfo=pheno1.datainfo)
    assert model1 != model2

    model1 = model1.replace(value_type='LIKELIHOOD')
    model2 = Model.create('model2', value_type='PREDICTION')
    assert model1 != model2

    model1 = Model.create('model1')
    model2 = Model.create('model2', execution_steps=pheno1.execution_steps)
    assert model1 != model2

    model2 = Model.create('model2', initial_individual_estimates=pd.DataFrame({'a': [1, 2, 3]}))
    assert not model1 == model2
    assert not model2 == model1

    model1 = Model.create('model1', initial_individual_estimates=pd.DataFrame({'a': [1, 2, 3]}))
    assert model1 == model2

    model1 = Model.create('model1', initial_individual_estimates=pd.DataFrame({'a': [1, 2, 4]}))
    assert not model1 == model2


def test_replace(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    sset = model.statements
    cl = sset.find_assignment('CL')
    sset_new = sset.reassign(cl.symbol, cl.expression + Expr.symbol('x'))
    with pytest.raises(ValueError, match='Symbol x is not defined'):
        model.replace(statements=sset_new)

    x_assignment = Assignment(Expr.symbol('x'), Expr.float(1))
    model.replace(statements=x_assignment + sset_new)

    sset_new = sset_new.before_odes + x_assignment + sset_new.ode_system + sset_new.after_odes
    with pytest.raises(ValueError, match='Symbol x defined after being used'):
        model.replace(statements=sset_new)

    sset_new = sset.reassign(cl.symbol, cl.expression + Expr.symbol('TIME'))
    model.replace(statements=sset_new)

    with pytest.raises(TypeError, match='model.datainfo must be of DataInfo type'):
        model.replace(datainfo=9)

    with pytest.raises(ValueError, match=re.escape("Invalid keywords given : ['unknown_arg']")):
        model.replace(unknown_arg=9)


def test_dict(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    d = model.to_dict()
    model2 = Model.from_dict(d)
    assert model == model2

    model = Model.create('model', initial_individual_estimates=pd.DataFrame({'a': [1, 2, 3]}))
    d = model.to_dict()
    assert d['initial_individual_estimates'] is not None

    model_from_dict = Model.from_dict(d)
    assert model_from_dict.initial_individual_estimates is not None


def test_dict_generic(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.dta'
    model = create_basic_pk_model('iv', dataset_path=path)
    assert model.code


def test_create_model():
    m = Model.create(name='model')
    assert isinstance(m.random_variables, RandomVariables)
    assert isinstance(m.parameters, Parameters)
    assert isinstance(m.execution_steps, ExecutionSteps)
    assert isinstance(m.statements, Statements)
    assert m.dependent_variables == {Expr.symbol('y'): 1}

    # datainfo
    with pytest.raises(TypeError, match='model.datainfo must be of DataInfo type'):
        Model.create(name='model', datainfo=None)

    # value_type
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot set value_type to A. Must be one of ('PREDICTION', 'LIKELIHOOD', '-2LL') or a symbol"
        ),
    ):
        Model.create(name='model', value_type='A')

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Can only set value_type to one of ('PREDICTION', 'LIKELIHOOD', '-2LL') or a symbol"
        ),
    ):
        Model.create(name='model', value_type=Expr.float(1))

    m = Model.create(name='model', value_type=Expr.symbol('A'))

    # parameter estimates
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [[Expr.symbol('x'), Expr.symbol('y')], [Expr.symbol('y'), Expr.symbol('z')]],
    )
    rvs = RandomVariables.create([dist1])
    px = Parameter('x', -10)
    py = Parameter('y', 0)
    pz = Parameter('z', -1)
    params = Parameters((px, py, pz))
    m = Model.create(name='model', random_variables=rvs, parameters=params)
    assert m.parameters.inits == {'x': 0.0, 'y': 0.0, 'z': 0.0}

    # random variables
    with pytest.raises(TypeError, match='model.random_variables must be of RandomVariables type'):
        Model.create(name='model', random_variables='a')

    # statements
    with pytest.raises(TypeError, match='model.statements must be of Statements type'):
        Model.create(name='model', statements='a')

    # dependent_variables
    with pytest.raises(TypeError, match='Dependent variable values must be of int type'):
        Model.create(name='model', dependent_variables={'a': 'a'})
    with pytest.raises(TypeError, match='Dependent variable keys must be a string or a symbol'):
        Model.create(name='model', dependent_variables={1: 2})

    m = Model.create(name='model', dependent_variables={'y2': 2})
    assert m.dependent_variables == {'y2': 2}
    m = Model.create(name='model', dependent_variables={Expr.symbol('y2'): 2})
    assert m.dependent_variables == {Expr.symbol('y2'): 2}

    # name
    with pytest.raises(TypeError, match='Name of a model has to be of string type'):
        Model.create(name=1)

    # parameters
    with pytest.raises(TypeError, match='parameters must be of Parameters type'):
        Model.create(name='name', parameters='a')

    # estimation steps
    with pytest.raises(TypeError, match='model.execution_steps must be of ExecutionSteps type'):
        Model.create(name='name', execution_steps='a')


def test_repr():
    model = Model.create('model1')
    assert model.__repr__() == "<Pharmpy model object model1>"


def test_repr_html(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    html_code = model._repr_html_()
    assert 'TVCL &=' in html_code
    assert 'POP_CL' in html_code
    assert 'eta_{CL}' in html_code


def test_has_same_dataset_as(load_example_model_for_test):
    pheno = load_example_model_for_test("pheno")
    model1 = Model.create('model1')
    model2 = Model.create('model2')
    assert model1.has_same_dataset_as(model2)

    model2 = Model.create('model2', dataset=pheno.dataset)
    assert not model1.has_same_dataset_as(model2)
    assert not model2.has_same_dataset_as(model1)

    model1 = Model.create('model1', dataset=pd.DataFrame({'a': [1, 2, 3]}))
    model2 = Model.create('model2', dataset=pd.DataFrame({'b': [1, 2, 3]}))
    assert not model1.has_same_dataset_as(model2)


def test_write_files():
    model = Model.create('model')
    assert isinstance(model.write_files(), Model)


def test_statements(load_example_model_for_test):
    pheno = load_example_model_for_test('pheno')
    model = Model.create(
        'model',
        parameters=pheno.parameters,
        datainfo=pheno.datainfo,
        statements=pheno.statements,
        random_variables=pheno.random_variables,
    )
    assert model.statements == pheno.statements

    with pytest.raises(ValueError):
        model = Model.create(
            'model',
            parameters=pheno.parameters,
            datainfo=pheno.datainfo,
            statements=pheno.statements,
        )
    with pytest.raises(ValueError):
        model = Model.create(
            'model',
            parameters=pheno.parameters,
            statements=pheno.statements,
            random_variables=pheno.random_variables,
        )
    with pytest.raises(ValueError):
        model = Model.create(
            'model',
            datainfo=pheno.datainfo,
            statements=pheno.statements,
            random_variables=pheno.random_variables,
        )
    with pytest.raises(ValueError):
        model = Model.create(
            'model',
            parameters=pheno.parameters,
            statements=pheno.statements,
            random_variables=pheno.random_variables,
        )
    with pytest.raises(ValueError):
        model = Model.create('model', statements=pheno.statements)
    with pytest.raises(ValueError):
        model = Model.create('model', parameters=pheno.parameters, statements=pheno.statements)
    with pytest.raises(ValueError):
        model = Model.create('model', datainfo=pheno.datainfo, statements=pheno.statements)
    with pytest.raises(ValueError):
        model = Model.create(
            'model', statements=pheno.statements, random_variables=pheno.random_variables
        )

    assign = Assignment.create(Expr.symbol('A'), Expr.symbol('B'))
    with pytest.raises(ValueError, match='Symbol B is not defined'):
        model.replace(statements=model.statements + assign)

    assign = Assignment.create(Expr.symbol('A'), Expr.symbol('t'))
    with pytest.raises(ValueError, match='Symbol t is not defined'):
        Model.create('model', statements=Statements() + assign)

    assign = Assignment.create(Expr.function('A', Expr.symbol('t')), Expr.symbol('t'))
    Model.create('model', statements=Statements() + assign)

    assign = Assignment.create(Expr.function('A', 0), Expr.symbol('t'))
    with pytest.raises(ValueError, match='Symbol t is not defined'):
        Model.create('model', statements=Statements() + assign)

    assign = Assignment.create(Expr.function('A', 0), Expr.symbol('0'))
    with pytest.raises(ValueError, match='Symbol 0 is not defined'):
        Model.create('model', statements=Statements() + assign)

    assign1 = Assignment(Expr.symbol('A'), Expr.symbol('B'))
    assign2 = Assignment(Expr.symbol('B'), Expr.symbol('0'))
    with pytest.raises(ValueError, match='Symbol B defined after being used'):
        Model.create('model', statements=Statements() + assign1 + assign2)

    df = pheno.dataset.rename(columns={'FA1': 'CL'})
    with pytest.raises(ValueError):
        pheno.replace(dataset=df)

    df = pheno.dataset.rename(columns={'FA1': '12'})
    with pytest.raises(ValueError):
        pheno.replace(dataset=df)


def test_get_and_check_odes(load_example_model_for_test, load_model_for_test, testdata):
    pheno = load_example_model_for_test('pheno')
    assert get_and_check_odes(pheno) is pheno.statements.ode_system
    model_min = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    with pytest.raises(ValueError):
        get_and_check_odes(model_min)


def test_get_and_check_dataset(load_example_model_for_test):
    pheno = load_example_model_for_test('pheno')
    assert get_and_check_dataset(pheno) is pheno.dataset
    m = pheno.replace(dataset=None)
    with pytest.raises(ValueError):
        get_and_check_dataset(m)
