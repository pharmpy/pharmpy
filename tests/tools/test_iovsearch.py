from functools import partial

import pytest

from pharmpy.basic import Expr
from pharmpy.modeling import (
    add_iov,
    create_joint_distribution,
    fix_parameters,
    remove_iiv,
    remove_iov,
)
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.iovsearch.tool import (
    _get_iiv_etas_with_corresponding_iov,
    _get_nonfixed_iivs,
    create_candidate_model_entry,
    create_iov_base_model_entry,
    create_workflow,
    prepare_list_of_parameters,
    validate_input,
)
from pharmpy.workflows import ModelEntry, Workflow


def S(x):
    return Expr.symbol(x)


@pytest.mark.parametrize(
    (
        'param_fix',
        'param_input',
        'param_ref',
    ),
    [
        (None, None, {'ETA_1', 'ETA_2', 'ETA_3'}),
        (None, ['CL'], {'CL'}),
        (['IIV_CL'], None, {'ETA_2', 'ETA_3'}),
    ],
)
def test_prepare_list_of_parameters(
    load_model_for_test, testdata, param_fix, param_input, param_ref
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    if param_fix:
        model = fix_parameters(model, param_fix)
    assert set(prepare_list_of_parameters(model, param_input)) == param_ref


@pytest.mark.parametrize(
    ('func', 'param_input', 'dist', 'no_of_iov_etas', 'no_of_iov_omegas', 'desc_ref'),
    [
        (None, None, 'same-as-iiv', 6, 3, 'IIV([CL]+[VC]+[MAT]);IOV([CL]+[VC]+[MAT])'),
        (
            partial(create_joint_distribution, rvs=['ETA_1', 'ETA_2']),
            None,
            'same-as-iiv',
            6,
            4,
            'IIV([CL,VC]+[MAT]);IOV([CL,VC]+[MAT])',
        ),
        (
            partial(create_joint_distribution, rvs=['ETA_1', 'ETA_2']),
            None,
            'disjoint',
            6,
            3,
            'IIV([CL,VC]+[MAT]);IOV([CL]+[VC]+[MAT])',
        ),
        (None, None, 'joint', 6, 6, 'IIV([CL]+[VC]+[MAT]);IOV([CL,VC,MAT])'),
    ],
)
def test_create_iov_base_model_entry(
    load_model_for_test,
    testdata,
    func,
    param_input,
    dist,
    no_of_iov_etas,
    no_of_iov_omegas,
    desc_ref,
):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    if func:
        model_start = func(model_start)
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)
    me_base = create_iov_base_model_entry(me_start, 'VISI', param_input, dist)
    model_base = me_base.model
    rvs_base = model_base.random_variables

    assert len(rvs_base.iov.names) == no_of_iov_etas
    assert len(rvs_base.iov.parameter_names) == no_of_iov_omegas
    assert model_base.description == desc_ref


@pytest.mark.parametrize(
    ('func', 'etas', 'no_of_iiv', 'no_of_iov', 'desc_ref'),
    [
        (remove_iiv, ['ETA_1'], 2, 6, 'IIV([VC]+[MAT]);IOV([CL]+[VC]+[MAT])'),
        (remove_iov, ['ETA_IOV_1_1'], 3, 4, 'IIV([CL]+[VC]+[MAT]);IOV([VC]+[MAT])'),
    ],
)
def test_create_candidate_model_entry(
    load_model_for_test,
    testdata,
    func,
    etas,
    no_of_iiv,
    no_of_iov,
    desc_ref,
):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_start = add_iov(model_start, 'VISI', distribution='same-as-iiv')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)
    me_cand = create_candidate_model_entry(func, me_start, etas, 1)
    model_cand = me_cand.model
    rvs_base = model_cand.random_variables

    assert len(rvs_base.iiv.names) == no_of_iiv
    assert len(rvs_base.iov.names) == no_of_iov
    assert model_cand.description == desc_ref


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


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    assert isinstance(create_workflow(model=model, results=results, column='APGR'), Workflow)


def test_validate_input_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    validate_input(model=model, results=results, column='APGR')


def test_validate_input_with_model_and_list_of_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    validate_input(model=model, results=results, column='APGR', list_of_parameters=['CL', 'V'])
    validate_input(model=model, results=results, column='APGR', list_of_parameters=[['V'], ['CL']])


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
        (None, {'rank_type': 'ofv', 'E': 1.0}, ValueError, 'E can only be provided'),
        (None, {'rank_type': 'mbic'}, ValueError, 'Value `E` must be provided when using mbic'),
        (None, {'rank_type': 'mbic', 'E': 0.0}, ValueError, 'Value `E` must be more than 0'),
        (None, {'rank_type': 'mbic', 'E': '10'}, ValueError, 'Value `E` must be denoted with `%`'),
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
    if not model_path:
        model_path = ('nonmem/pheno.mod',)
    path = testdata.joinpath(*model_path)
    model = load_model_for_test(path)
    results = read_modelfit_results(path)

    kwargs = {'model': model, 'results': results, **arguments}
    if 'column' not in arguments.keys():
        kwargs['column'] = 'APGR'

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
