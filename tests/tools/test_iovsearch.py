import pytest

from pharmpy.basic import Expr
from pharmpy.modeling import add_iov, fix_parameters, remove_iov
from pharmpy.tools.iovsearch.tool import (
    _get_iiv_etas_with_corresponding_iov,
    _get_nonfixed_iivs,
    create_workflow,
    validate_input,
)
from pharmpy.workflows import Workflow


def S(x):
    return Expr.symbol(x)


def test_ignore_fixed_iiv(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert len(model.random_variables.iiv.names) == 2
    model = fix_parameters(model, 'IVCL')
    nonfixed_iivs = _get_nonfixed_iivs(model)
    assert len(nonfixed_iivs.names) == 1


def test_iovsearch_github_issues_976(load_model_for_test, testdata):
    m = load_model_for_test(testdata / 'nonmem' / 'pheno_multivariate_piecewise.mod')
    assert not m.random_variables.iov
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == set()

    m = add_iov(m, 'FA1', distribution='same-as-iiv')
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == set(
        map(lambda rv: S(rv), m.random_variables.iiv.names)
    )

    m = remove_iov(m, 'ETA_IOV_1_1')
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == {S('ETA_2')}

    m = remove_iov(m, 'ETA_IOV_2_1')
    assert not m.random_variables.iov
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == set()


def test_create_workflow():
    assert isinstance(create_workflow(), Workflow)


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert isinstance(create_workflow(model=model, column='APGR'), Workflow)


def test_validate_input():
    validate_input()


def test_validate_input_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    validate_input(model=model, column='APGR')


def test_validate_input_with_model_and_list_of_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    validate_input(model=model, column='APGR', list_of_parameters=['CL', 'V'])
    validate_input(model=model, column='APGR', list_of_parameters=[['V'], ['CL']])


@pytest.mark.parametrize(
    (
        'model_path',
        'arguments',
        'exception',
        'match',
    ),
    [
        (None, dict(column=1), TypeError, 'Invalid `column`'),
        (None, dict(list_of_parameters='CL'), TypeError, 'Invalid `list_of_parameters`'),
        (None, dict(rank_type=1), ValueError, 'Invalid `rank_type`'),
        (None, dict(rank_type='bi'), ValueError, 'Invalid `rank_type`'),
        (None, dict(cutoff='1'), TypeError, 'Invalid `cutoff`'),
        (None, dict(distribution=['same-as-iiv']), ValueError, 'Invalid `distribution`'),
        (None, dict(distribution='same'), ValueError, 'Invalid `distribution`'),
        (
            ('nonmem', 'pheno.mod'),
            dict(column='OCC'),
            ValueError,
            'Invalid `column`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(column='APGR', list_of_parameters=['K']),
            ValueError,
            'Invalid `list_of_parameters`',
        ),
        (
            None,
            dict(model=1),
            TypeError,
            'Invalid `model`',
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test,
    testdata,
    model_path,
    arguments,
    exception,
    match,
):
    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    kwargs = {'model': model, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
