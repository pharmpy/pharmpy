import pytest
from sympy import Symbol as S

from pharmpy.modeling import add_iov, remove_iov
from pharmpy.tools.iovsearch.tool import (
    _get_iiv_etas_with_corresponding_iov,
    create_workflow,
    validate_input,
)
from pharmpy.workflows import Workflow


def test_iovsearch_github_issues_976(load_model_for_test, testdata):

    m = load_model_for_test(testdata / 'nonmem' / 'pheno_multivariate_piecewise.mod')
    assert not m.random_variables.iov
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == set()

    add_iov(m, 'FA1', distribution='same-as-iiv')
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == set(
        map(lambda rv: S(rv), m.random_variables.iiv.names)
    )

    remove_iov(m, 'ETA_IOV_1_1')
    assert set(_get_iiv_etas_with_corresponding_iov(m)) == {S('ETA(2)')}

    remove_iov(m, 'ETA_IOV_2_1')
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


@pytest.mark.parametrize(
    ('model_path', 'column', 'list_of_parameters', 'rank_type', 'cutoff', 'distribution'),
    [
        (
            None,
            1,
            None,
            'bic',
            None,
            'same-as-iiv',
        ),
        (
            None,
            'OCC',
            'CL',
            'bic',
            None,
            'same-as-iiv',
        ),
        (
            None,
            'OCC',
            None,
            'bi',
            None,
            'same-as-iiv',
        ),
        (
            None,
            'OCC',
            None,
            'bic',
            '1',
            'same-as-iiv',
        ),
        (
            None,
            'OCC',
            None,
            'bic',
            None,
            'same',
        ),
        (
            ('nonmem', 'pheno.mod'),
            'OCC',
            None,
            'bic',
            None,
            'same-as-iiv',
        ),
        (
            ('nonmem', 'pheno.mod'),
            'APGR',
            ['K'],
            'bic',
            None,
            'same-as-iiv',
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test,
    testdata,
    model_path,
    column,
    list_of_parameters,
    rank_type,
    distribution,
    cutoff,
):

    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    with pytest.raises((ValueError, TypeError)):
        validate_input(
            column=column,
            list_of_parameters=list_of_parameters,
            rank_type=rank_type,
            cutoff=cutoff,
            distribution=distribution,
            model=model,
        )


def test_validate_input_raises_on_wrong_model_type():
    with pytest.raises(TypeError, match='Invalid model'):
        validate_input(model=1)
