from functools import partial

import pytest

from pharmpy.mfl import ModelFeatures
from pharmpy.modeling import (
    add_iiv,
    add_lag_time,
    add_peripheral_compartment,
    add_pk_iiv,
    create_joint_distribution,
    fix_parameters,
    remove_iiv,
)
from pharmpy.modeling.mfl import get_model_features
from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.tools.iivsearch.algorithms import (
    create_candidate_linearized,
    create_description,
    get_covariance_combinations,
    get_iiv_combinations,
)
from pharmpy.tools.iivsearch.tool import (
    create_base_model_entry,
    create_param_mapping,
    create_workflow,
    needs_base_model,
    prepare_algorithms,
    prepare_input_model_entry,
    prepare_rank_options,
    update_linearized_base_model,
    validate_input,
)
from pharmpy.tools.linearize.tool import create_derivative_model, create_linearized_model
from pharmpy.tools.run import read_modelfit_results
from pharmpy.workflows import ModelEntry, Workflow

MINIMAL_VALID_MFL_STRING = 'IIV(@PK,exp)'


def test_update_linearized_base_model_mfl(load_model_for_test, testdata, model_entry_factory):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_start = remove_iiv(model_start, ['ETA_3'])
    description = '[CL]+[VC]'
    model_start = model_start.replace(description=description)
    me_start = model_entry_factory([model_start])[0]

    model_updated = update_linearized_base_model(False, me_start, me_start)
    assert len(model_updated.parameters) == len(model_start.parameters)
    assert model_updated.description == description

    model_base = add_iiv(model_start, ['MAT'], 'exp')
    model_base = fix_parameters(model_base, parameter_names=['IIV_MAT'])
    description = '[CL]+[VC]+[MAT]'
    model_base = model_base.replace(description=description)
    me_base = model_entry_factory([model_base])[0]
    model_updated = update_linearized_base_model(False, me_start, me_base)
    assert len(model_updated.parameters) > len(model_start.parameters)
    assert len(model_base.parameters.fixed) > len(model_updated.parameters.fixed)
    assert len(model_updated.parameters) == len(model_base.parameters)
    assert model_updated.description == description

    description = '[CL,VC,MAT]'
    model_fullblock = model_base.replace(description=description)
    me_fullblock = model_entry_factory([model_fullblock])[0]
    model_fullblock = update_linearized_base_model(True, me_start, me_fullblock)
    assert len(model_fullblock.parameters) > len(model_updated.parameters)
    assert len(model_fullblock.random_variables.iiv) == 1
    assert model_fullblock.description == description


def test_prepare_input_model_entry(load_model_for_test, testdata):
    model_start = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')

    me = prepare_input_model_entry(model_start, res_start)
    assert me.model.name == 'input'
    assert me.model.description == '[CL]+[MAT]+[VC]'
    assert me.modelfit_results is not None
    assert me.parent is None


@pytest.mark.parametrize(
    'algorithm, correlation_algorithm, list_of_algorithms',
    [
        ('top_down_exhaustive', 'skip', ['td_exhaustive_no_of_etas']),
        ('bottom_up_stepwise', 'skip', ['bu_stepwise_no_of_etas']),
        (
            'top_down_exhaustive',
            'top_down_exhaustive',
            ['td_exhaustive_no_of_etas', 'td_exhaustive_block_structure'],
        ),
        ('skip', 'top_down_exhaustive', ['td_exhaustive_block_structure']),
        (
            'top_down_exhaustive',
            None,
            ['td_exhaustive_no_of_etas', 'td_exhaustive_block_structure'],
        ),
        ('bottom_up_stepwise', None, ['bu_stepwise_no_of_etas', 'td_exhaustive_block_structure']),
    ],
)
def test_prepare_algorithms(algorithm, correlation_algorithm, list_of_algorithms):
    assert prepare_algorithms(algorithm, correlation_algorithm) == list_of_algorithms


def test_create_param_mapping(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'models' / 'mox2.mod'
    model = load_model_for_test(path)
    param_mapping = create_param_mapping(model)
    assert param_mapping == {'ETA_1': 'CL', 'ETA_2': 'VC', 'ETA_3': 'MAT'}


@pytest.mark.parametrize(
    'mfl, type, expected',
    [
        ('IIV([CL,MAT,VC],EXP)', 'iiv', '[CL]+[MAT]+[VC]'),
        ('IIV(CL,EXP);IIV([MAT,VC],ADD)', 'iiv', '[CL]+[MAT]+[VC] (ADD:MAT,VC)'),
        ('IIV(CL,EXP);IIV(MAT,ADD);IIV(VC,LOG)', 'iiv', '[CL]+[MAT]+[VC] (ADD:MAT;LOG:VC)'),
        ('IIV([CL,MAT,VC],EXP);COVARIANCE(IIV,[CL,VC])', 'iiv', '[CL,VC]+[MAT]'),
        ('IIV(CL,EXP);IIV(MAT,ADD);IIV(VC,LOG)', 'covariance', '[CL]+[MAT]+[VC]'),
        ('IIV([CL,MAT,VC],EXP);COVARIANCE(IIV,[CL,MAT,VC])', 'iiv', '[CL,MAT,VC]'),
    ],
)
def test_create_description_mfl(mfl, type, expected):
    mfl = ModelFeatures.create(mfl)
    assert create_description(mfl, type) == expected


@pytest.mark.parametrize(
    'mfl, base_features, expected',
    [
        ('IIV(CL,EXP);IIV?([MAT,VC],EXP)', 'IIV([CL,MAT,VC],EXP)', 3),
        ('IIV?([CL,MAT,VC],EXP)', 'IIV([CL,MAT,VC],EXP)', 7),
        ('IIV?([CL,MAT,VC],EXP)', [], 7),
        ('IIV(CL,EXP);IIV?([MAT,VC],[EXP,ADD])', 'IIV([CL,MAT,VC],EXP)', 8),
    ],
)
def test_get_iiv_combinations(mfl, base_features, expected):
    mfl = ModelFeatures.create(mfl)
    mfl_base = ModelFeatures.create(base_features)
    combinations = get_iiv_combinations(mfl, mfl_base)
    assert len(combinations) == expected
    assert mfl_base not in combinations


@pytest.mark.parametrize(
    'mfl, base_features, expected',
    [
        ('COVARIANCE?(IIV,[CL,MAT,VC])', 'COVARIANCE(IIV,[CL,MAT,VC])', 4),
        ('COVARIANCE?(IIV,[CL,MAT,VC,MDT])', 'COVARIANCE(IIV,[CL,MAT,VC,MDT])', 11),
        ('COVARIANCE?(IIV,[CL,MAT,VC,MDT])', 'COVARIANCE(IIV,[CL,MAT,VC])', 11),
        ('COVARIANCE?(IIV,[CL,MAT,VC])', [], 4),
        ('COVARIANCE(IIV,[CL,MAT]);COVARIANCE?(IIV,[VC,MDT])', 'COVARIANCE(IIV,[CL,MAT])', 1),
    ],
)
def test_get_covariance_combinations(mfl, base_features, expected):
    mfl = ModelFeatures.create(mfl)
    mfl_base = ModelFeatures.create(base_features)
    combinations = get_covariance_combinations(mfl, mfl_base)
    assert len(combinations) == expected
    mfl_forced = mfl - mfl.filter(filter_on='optional')
    mf_empty = ModelFeatures(tuple())
    if mfl_forced:
        assert mf_empty not in combinations
    else:
        assert mfl_base not in combinations
        assert mf_empty in combinations if base_features else mf_empty not in combinations


@pytest.mark.parametrize(
    'func, mfl, type, description, no_of_params',
    [
        (None, 'IIV([CL,MAT],exp)', 'iiv', '[CL]+[MAT]', 2),
        (None, 'IIV([CL],exp)', 'iiv', '[CL]', 1),
        (None, '', 'iiv', '', 1),
        (add_lag_time, 'IIV([CL,VC,MDT],exp)', 'iiv', '[CL]+[MDT]+[VC]', 3),
        (None, 'COVARIANCE(IIV,[CL,VC,MAT])', 'covariance', '[CL,MAT,VC]', 6),
        (None, 'COVARIANCE(IIV,[CL,VC])', 'covariance', '[CL,VC]+[MAT]', 4),
        (add_lag_time, 'COVARIANCE(IIV,[CL,VC,MAT])', 'covariance', '[CL,MAT,VC]+[MDT]', 7),
        (
            add_lag_time,
            'COVARIANCE(IIV,[CL,VC]);COVARIANCE(IIV,[MAT,MDT])',
            'covariance',
            '[CL,VC]+[MAT,MDT]',
            6,
        ),
    ],
)
def test_create_candidate_linearized(
    load_model_for_test, model_entry_factory, testdata, func, mfl, type, description, no_of_params
):
    input_model = load_model_for_test(testdata / "nonmem" / "models" / "mox2.mod")
    if func:
        input_model = func(input_model)
        input_model = add_pk_iiv(input_model)
    input_model_entry = model_entry_factory([input_model])[0]

    derivative_model_entry = create_derivative_model(input_model_entry)
    derivative_model_entry = model_entry_factory([derivative_model_entry.model])[0]

    linbase_model_entry = create_linearized_model(
        "linbase", "", input_model, derivative_model_entry
    )
    linbase_model_entry = model_entry_factory([linbase_model_entry.model])[0]

    param_mapping = {'ETA_1': 'CL', 'ETA_2': 'VC', 'ETA_3': 'MAT', 'ETA_MDT': 'MDT'}

    mfl = ModelFeatures.create(mfl)
    candidate_model_entry = create_candidate_linearized(
        'cand1', mfl, type, param_mapping, linbase_model_entry
    )
    assert candidate_model_entry.parent == linbase_model_entry.model
    assert candidate_model_entry.modelfit_results is None

    candidate_model = candidate_model_entry.model
    assert candidate_model.description == description
    assert len(candidate_model.random_variables.iiv.parameter_names) == no_of_params
    if mfl:
        assert (
            candidate_model.parameters.inits['IIV_CL']
            != linbase_model_entry.model.parameters.inits['IIV_CL']
        )


@pytest.mark.parametrize(
    'funcs, mfl, as_fullblock, algorithm, expected',
    [
        ([], 'IIV([CL,MAT,VC],EXP)', False, 'td', False),
        ([], 'IIV?([CL,MAT,VC],EXP)', False, 'td', False),
        ([], 'IIV?([CL,MAT,VC],EXP);COVARIANCE(IIV,[CL,VC])', False, 'td', True),
        ([add_peripheral_compartment], 'IIV?([CL,MAT,VC],EXP)', False, 'td', False),
        ([add_peripheral_compartment], 'IIV?([CL,MAT,VC,QP1],EXP)', False, 'td', False),
        (
            [
                add_peripheral_compartment,
                partial(add_iiv, list_of_parameters=['QP1'], expression='exp'),
            ],
            'IIV?([CL,MAT,VC,QP1],EXP)',
            False,
            'td',
            False,
        ),
        ([], 'IIV([CL,MAT,VC],EXP);COVARIANCE?(IIV,[CL,VC])', False, 'td', False),
        ([], 'IIV([CL,MAT,VC],EXP)', True, 'td', True),
        ([create_joint_distribution], 'IIV([CL,MAT,VC],EXP)', True, 'td', False),
        ([], 'IIV(CL,EXP);IIV?([MAT,VC],[EXP,ADD])', False, 'td', False),
        ([], 'IIV([CL,MAT,VC],EXP)', False, 'bu', True),
    ],
)
def test_needs_base_model(
    load_model_for_test, testdata, funcs, mfl, as_fullblock, algorithm, expected
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)

    mfl = ModelFeatures.create(mfl)
    assert needs_base_model(model, mfl, as_fullblock, algorithm) == expected


@pytest.mark.parametrize(
    'type, mfl, as_fullblock, iiv_expected, cov_expected',
    [
        ('td', 'IIV?([CL,VC],EXP)', False, 'IIV([CL,VC],EXP)', ''),
        ('td', 'IIV?([CL,MAT,VC],EXP)', False, 'IIV([CL,MAT,VC],EXP)', ''),
        (
            'td',
            'IIV?([CL,MAT,VC],EXP)',
            True,
            'IIV([CL,MAT,VC],EXP)',
            'COVARIANCE(IIV,[CL,MAT,VC])',
        ),
        (
            'td',
            'IIV(CL,EXP);IIV?([CL,MAT,VC],EXP);COVARIANCE?(IIV,[CL,MAT,VC])',
            False,
            'IIV([CL,MAT,VC],EXP)',
            '',
        ),
        ('td', 'IIV(CL,EXP);IIV?([MAT,VC],ADD)', False, 'IIV(CL,EXP);IIV([MAT,VC],ADD)', ''),
        ('td', 'IIV(CL,[EXP,ADD]);IIV?([MAT,VC],[EXP,ADD])', False, 'IIV([CL,MAT,VC],EXP)', ''),
        (
            'td',
            'IIV(CL,[EXP,ADD]);IIV?([MAT,VC],[EXP,ADD]);COVARIANCE?(IIV,[CL,MAT,VC])',
            True,
            'IIV([CL,MAT,VC],EXP)',
            'COVARIANCE(IIV,[CL,MAT,VC])',
        ),
        ('bu', 'IIV?([CL,MAT,VC],EXP)', False, '', ''),
        ('bu', 'IIV(CL,EXP);IIV?([MAT,VC],EXP)', False, 'IIV(CL,EXP)', ''),
        ('bu', 'IIV([CL,MAT],EXP);IIV?(VC,EXP)', False, 'IIV([CL,MAT],EXP)', ''),
        ('bu', 'IIV(CL,EXP);IIV?([MAT,VC],EXP)', True, 'IIV(CL,EXP)', ''),
        (
            'bu',
            'IIV([CL,MAT],EXP);IIV?(VC,EXP)',
            True,
            'IIV([CL,MAT],EXP)',
            'COVARIANCE(IIV,[CL,MAT])',
        ),
        ('linearize', 'IIV?([CL,VC],EXP)', False, 'IIV([CL,VC],EXP)', ''),
        ('linearize', 'IIV?([CL,MAT,VC],EXP)', False, 'IIV([CL,MAT,VC],EXP)', ''),
    ],
)
def test_create_base_model(
    load_model_for_test,
    testdata,
    model_entry_factory,
    type,
    mfl,
    as_fullblock,
    iiv_expected,
    cov_expected,
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = remove_iiv(model, to_remove=['ETA_3'])
    model_entry = model_entry_factory([model])[0]
    mfl = ModelFeatures.create(mfl)
    base_model_entry = create_base_model_entry(type, mfl, as_fullblock, model_entry)
    base_model = base_model_entry.model
    assert repr(get_model_features(base_model, type='iiv')) == iiv_expected
    assert repr(get_model_features(base_model, type='covariance')) == cov_expected
    assert base_model_entry.modelfit_results is None
    assert base_model_entry.parent == model

    if type == 'linearize':
        params_new = base_model.parameters - model.parameters
        assert all(p.init == 0.000001 for p in params_new)


def test_create_base_model_raises(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model_entry = ModelEntry.create(model=model, modelfit_results=res)
    mfl = ModelFeatures.create('')
    with pytest.raises(ValueError):
        create_base_model_entry('x', mfl, True, model_entry)


@pytest.mark.parametrize(
    'kwargs, expected',
    [
        (
            {
                'rank_type': 'ofv',
                'cutoff': 3.84,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E_p': None,
                'E_q': None,
            },
            {
                'rank_type': 'ofv',
                'cutoff': 3.84,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E': None,
            },
        ),
        (
            {
                'rank_type': 'bic',
                'cutoff': None,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E_p': None,
                'E_q': None,
            },
            {
                'rank_type': 'bic_iiv',
                'cutoff': None,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E': None,
            },
        ),
        (
            {
                'rank_type': 'mbic',
                'cutoff': None,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E_p': 0.5,
                'E_q': 0.5,
            },
            {
                'rank_type': 'mbic_iiv',
                'cutoff': None,
                'strictness': 'minimization_successful',
                'parameter_uncertainty_method': None,
                'E': (0.5, 0.5),
            },
        ),
    ],
)
def test_prepare_rank_options(kwargs, expected):
    rank_options = prepare_rank_options(**kwargs)
    for key, value in expected.items():
        assert getattr(rank_options, key) == value


def test_create_workflow_with_model(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    results = parse_modelfit_results(model, path)
    assert isinstance(
        create_workflow(
            model=model,
            results=results,
            algorithm='top_down_exhaustive',
            search_space='IIV(@IIV,exp)',
        ),
        Workflow,
    )


def test_validate_input_with_model(load_model_for_test, testdata):
    path = testdata / 'nonmem' / 'pheno.mod'
    model = load_model_for_test(path)
    results = parse_modelfit_results(model, path)
    validate_input(
        model=model,
        results=results,
        algorithm='top_down_exhaustive',
        search_space=MINIMAL_VALID_MFL_STRING,
    )


@pytest.mark.parametrize(
    ('model_path', 'arguments', 'exception', 'match'),
    [
        (None, dict(search_space='x'), ValueError, 'Could not parse `search_space`'),
        (None, dict(algorithm=1), ValueError, 'Invalid `algorithm`'),
        (None, dict(algorithm='brute_force_no_of_eta'), ValueError, 'Invalid `algorithm`'),
        (None, dict(rank_type=1), ValueError, 'Invalid `rank_type`'),
        (None, dict(rank_type='bi'), ValueError, 'Invalid `rank_type`'),
        (None, dict(cutoff='1'), TypeError, 'Invalid `cutoff`'),
        (
            None,
            dict(model=1),
            TypeError,
            'Invalid `model`',
        ),
        (
            ('nonmem/pheno.mod',),
            dict(strictness='rse'),
            ValueError,
            '`parameter_uncertainty_method` not set',
        ),
        (
            None,
            dict(algorithm='skip', correlation_algorithm='skip'),
            ValueError,
            'Both algorithm and correlation_algorithm',
        ),
        (
            None,
            dict(algorithm='skip', correlation_algorithm=None),
            ValueError,
            'correlation_algorithm need to be specified',
        ),
        (None, {'rank_type': 'ofv', 'E_p': 1.0}, ValueError, 'E_p and E_q can only be provided'),
        (None, {'rank_type': 'ofv', 'E_q': 1.0}, ValueError, 'E_p and E_q can only be provided'),
        (
            None,
            {'rank_type': 'mbic', 'algorithm': 'top_down_exhaustive'},
            ValueError,
            'Value `E_p` must be provided for `algorithm`',
        ),
        (
            None,
            {
                'rank_type': 'mbic',
                'algorithm': 'skip',
                'correlation_algorithm': 'top_down_exhaustive',
            },
            ValueError,
            'Value `E_q` must be provided for `correlation_algorithm`',
        ),
        (None, {'rank_type': 'mbic', 'E_p': 0.0}, ValueError, 'Value `E_p` must be more than 0'),
        (
            None,
            {
                'rank_type': 'mbic',
                'algorithm': 'skip',
                'correlation_algorithm': 'top_down_exhaustive',
                'E_q': 0.0,
            },
            ValueError,
            'Value `E_q` must be more than 0',
        ),
        (
            None,
            {'rank_type': 'mbic', 'E_p': '10'},
            ValueError,
            'Value `E_p` must be denoted with `%`',
        ),
        (
            None,
            {
                'rank_type': 'mbic',
                'algorithm': 'skip',
                'correlation_algorithm': 'top_down_exhaustive',
                'E_q': '10',
            },
            ValueError,
            'Value `E_q` must be denoted with `%`',
        ),
        (('nonmem/qa/boxcox.mod',), dict(), ValueError, 'Invalid `model`'),
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
    results = parse_modelfit_results(model, path)

    harmless_arguments = dict(
        algorithm='top_down_exhaustive',
    )

    kwargs = {'model': model, 'results': results, **harmless_arguments, **arguments}

    if 'search_space' not in kwargs.keys():
        kwargs['search_space'] = MINIMAL_VALID_MFL_STRING

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
