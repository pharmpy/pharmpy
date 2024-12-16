from dataclasses import replace
from functools import partial

import pytest

from pharmpy.modeling import (
    add_allometry,
    add_covariate_effect,
    add_peripheral_compartment,
    get_covariate_effects,
    remove_covariate_effect,
    set_name,
)
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.covsearch.tool import (
    AdaptiveStep,
    AddEffect,
    Candidate,
    Effect,
    ForwardStep,
    SearchState,
    _greedy_search,
    _start,
    create_result_tables,
    create_workflow,
    extract_nonsignificant_effects,
    filter_effects,
    get_best_candidate_so_far,
    get_effect_funcs_and_base_model,
    get_exploratory_covariates,
    is_model_in_search_space,
    prepare_mfls,
    task_add_covariate_effect,
    validate_input,
)
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.workflows import ModelEntry, Workflow

MINIMAL_INVALID_MFL_STRING = ''
MINIMAL_VALID_MFL_STRING = 'LET(x, 0)'
LARGE_VALID_MFL_STRING = 'COVARIATE?(@IIV, @CONTINUOUS, *);COVARIATE?(@IIV, @CATEGORICAL, CAT)'


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    assert isinstance(
        create_workflow(model=model, results=results, search_space=MINIMAL_VALID_MFL_STRING),
        Workflow,
    )


@pytest.mark.parametrize(
    ('model_path',), ((('nonmem', 'pheno.mod'),), (('nonmem', 'ruvsearch', 'mox3.mod'),))
)
def test_validate_input_with_model(load_model_for_test, testdata, model_path):
    path = testdata.joinpath(*model_path)
    model = load_model_for_test(path)
    results = read_modelfit_results(path)
    validate_input(model=model, results=results, search_space=LARGE_VALID_MFL_STRING)


@pytest.mark.parametrize(
    'funcs, search_space, is_in_search_space',
    [
        ([], 'COVARIATE?([CL,VC],WT,EXP)', True),
        ([], 'COVARIATE([CL,VC],WT,EXP)', False),
        (
            [partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp')],
            'COVARIATE?([CL,VC],WT,EXP)',
            False,
        ),
        (
            [partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp')],
            'COVARIATE([CL,VC],WT,EXP)',
            False,
        ),
        (
            [add_peripheral_compartment, add_allometry],
            'COVARIATE([QP1,CL,VC,VP1],WT,POW);COVARIATE?([CL,VC],AGE,[EXP,LIN]);COVARIATE?([CL,VC],SEX,CAT)',
            True,
        ),
    ],
)
def test_is_model_in_search_space(
    load_model_for_test, testdata, funcs, search_space, is_in_search_space
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)

    ss_mfl = ModelFeatures.create_from_mfl_string(search_space)
    model_mfl = ModelFeatures.create_from_mfl_string(get_model_features(model))
    model_mfl = ModelFeatures.create_from_mfl_statement_list(
        model_mfl.mfl_statement_list(["covariate"])
    )

    assert is_model_in_search_space(model, model_mfl, ss_mfl) == is_in_search_space


@pytest.mark.parametrize(
    'search_space, no_of_exploratory_covs',
    [
        ('COVARIATE?([CL,VC],WT,EXP)', 2),
        ('COVARIATE(CL,WT,EXP);COVARIATE?(VC,WT,EXP)', 1),
    ],
)
def test_get_exploratory_covariates(search_space, no_of_exploratory_covs):
    search_space = ModelFeatures.create_from_mfl_string(search_space)
    assert len(get_exploratory_covariates(search_space)) == no_of_exploratory_covs


def test_filter_effects():
    search_space = 'COVARIATE?([CL,VC],[WT,AGE],EXP)'
    mfl = ModelFeatures.create_from_mfl_string(search_space)
    effect_funcs = get_exploratory_covariates(mfl)
    assert len(effect_funcs) == 4
    effect_args_1 = ('CL', 'WT', 'exp', '*')
    last_effect = Effect(*effect_args_1)
    filtered_1 = filter_effects(effect_funcs, last_effect, {})
    assert len(filtered_1) == 3
    assert effect_args_1 in effect_funcs.keys()
    assert effect_args_1 not in filtered_1.keys()
    nonsignificant_effects = {effect_args_1: effect_funcs[effect_args_1]}
    effect_args_2 = ('CL', 'AGE', 'exp', '*')
    last_effect = Effect(*effect_args_2)
    filtered_2 = filter_effects(effect_funcs, last_effect, nonsignificant_effects)
    assert len(filtered_2) == 2


@pytest.mark.parametrize(
    'p_value, no_of_nonsignificant_effects',
    [(0, 4), (100000000000000000, 0), (10**-8, 2)],
)
def test_extract_nonsignificant_effects(
    load_model_for_test, testdata, model_entry_factory, p_value, no_of_nonsignificant_effects
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    modelres = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    parent_modelentry = ModelEntry(model, modelfit_results=modelres)

    search_space = 'COVARIATE?([CL,VC],[WT, AGE],EXP)'
    mfl = ModelFeatures.create_from_mfl_string(search_space)
    effect_funcs = get_exploratory_covariates(mfl)
    models = [func(model) for func in effect_funcs.values()]
    model_entries = model_entry_factory(models, ref_val=modelres.ofv)
    steps = [ForwardStep(p_value, Effect(*key)) for key in effect_funcs.keys()]
    candidates = [Candidate(me, (step,)) for me, step in zip(model_entries, steps)]
    nonsignificant_effects = extract_nonsignificant_effects(
        parent_modelentry, candidates, effect_funcs, p_value
    )

    assert len(nonsignificant_effects) == no_of_nonsignificant_effects


@pytest.mark.parametrize(
    'search_space, no_of_covariates',
    [
        ('COVARIATE?([CL,VC],WT,EXP)', 1),
        (
            'LET(CONTINUOUS,[AGE,WT]);LET(CATEGORICAL,SEX)\n'
            'COVARIATE?([CL,VC],@CONTINUOUS,exp,*)\n'
            'COVARIATE?([CL,VC],@CATEGORICAL,cat,*)',
            2,
        ),
    ],
)
def test_prepare_mfls(load_model_for_test, testdata, search_space, no_of_covariates):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    ss_mfl, model_mfl = prepare_mfls(model, search_space)
    assert model_mfl.absorption is None
    assert len(ss_mfl.covariate) == no_of_covariates
    assert prepare_mfls(model, ss_mfl)[0] == ss_mfl


@pytest.mark.parametrize(
    (
        'model_path',
        'arguments',
        'exception',
        'match',
    ),
    [
        (
            None,
            dict(p_forward='x'),
            TypeError,
            'Invalid `p_forward`',
        ),
        (
            None,
            dict(p_forward=1.05),
            ValueError,
            'Invalid `p_forward`',
        ),
        (
            None,
            dict(p_backward=[]),
            TypeError,
            'Invalid `p_backward`',
        ),
        (
            None,
            dict(p_backward=1.01),
            ValueError,
            'Invalid `p_backward`',
        ),
        (
            None,
            dict(max_steps=1.2),
            TypeError,
            'Invalid `max_steps`',
        ),
        (None, dict(algorithm=()), ValueError, 'Invalid `algorithm`'),
        (
            None,
            dict(algorithm='scm-backward'),
            ValueError,
            'Invalid `algorithm`',
        ),
        (('nonmem', 'pheno.mod'), dict(search_space=1), TypeError, 'Invalid `search_space`'),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space=MINIMAL_INVALID_MFL_STRING),
            ValueError,
            'Invalid `search_space`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='LAGTIME(ON)'),
            ValueError,
            'Invalid `search_space`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='COVARIATE([CL, VC], WGT, EXP)'),
            ValueError,
            'Invalid `search_space` because of invalid parameter',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='COVARIATE([CL, V], SEX, EXP)'),
            ValueError,
            'Invalid `search_space` because of invalid covariate',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='COVARIATE([CL, V], WGT, [EXP, CUSTOM])'),
            ValueError,
            'Invalid `search_space` because of invalid effect function',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='COVARIATE(CL, WGT, exp, -)'),
            ValueError,
            'Invalid `search_space`',
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
    if not model_path:
        model_path = ('nonmem/pheno.mod',)
    path = testdata.joinpath(*model_path)
    model = load_model_for_test(path)
    results = read_modelfit_results(path)

    harmless_arguments = dict(
        search_space=MINIMAL_VALID_MFL_STRING,
    )

    kwargs = {'model': model, 'results': results, **harmless_arguments, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)


def test_covariate_filtering(load_model_for_test, testdata):
    search_space = 'COVARIATE(@IIV, @CONTINUOUS, lin);COVARIATE?(@IIV, @CATEGORICAL, CAT)'
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    orig_cov = get_covariate_effects(model)
    assert len(orig_cov) == 3

    eff, filtered_model = get_effect_funcs_and_base_model(search_space, model)
    assert len(eff) == 2
    expected_cov_eff = set((('CL', 'APGR', 'cat', '*'), ('V', 'APGR', 'cat', '*')))
    assert set(eff.keys()) == expected_cov_eff
    assert len(get_covariate_effects(filtered_model)) == 2

    for cov_effect in get_covariate_effects(model):
        model = remove_covariate_effect(model, cov_effect[0], cov_effect[1].name)

    model = add_covariate_effect(model, 'CL', 'WGT', 'pow', '*')
    assert len(get_covariate_effects(model)) == 1
    search_space = 'COVARIATE([CL, V],WGT,pow,*)'
    eff, filtered_model = get_effect_funcs_and_base_model(search_space, model)
    assert len(get_covariate_effects(filtered_model)) == 2
    assert len(eff) == 0

    # exploratory covariates should be removed before covsearch if present in model
    for cov_effect in get_covariate_effects(model):
        model = remove_covariate_effect(model, cov_effect[0], cov_effect[1].name)

    model = add_covariate_effect(model, 'CL', 'WGT', 'pow', '*')
    assert len(get_covariate_effects(model)) == 1
    search_space = 'COVARIATE?([CL, V], WGT, pow, *)'
    eff, filtered_model = get_effect_funcs_and_base_model(search_space, model)
    assert len(get_covariate_effects(filtered_model)) == 0
    assert len(eff) == 2


def test_max_eval(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    modelres = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')

    no_max_eval_model_entry = _start(model, modelres, max_eval=False)
    assert no_max_eval_model_entry.model == model

    max_eval_model_entry = _start(model, modelres, max_eval=True)
    assert max_eval_model_entry.model != model
    assert max_eval_model_entry.model.execution_steps[0].maximum_evaluations == round(
        3.1 * modelres.function_evaluations_iterations.loc[1]
    )


def _mock_fit(effect, parent, adaptive_step):
    if effect[0][0] == "CL":
        if effect[0][1] == "WGT" and effect[0][2] == "pow":
            ofv = parent.modelentry.modelfit_results.ofv + 100 * -2
        else:
            if not adaptive_step:
                if (
                    len(parent.steps) != 0
                    and effect[0][0] == "CL"
                    and effect[0][1] == "APGR"
                    and effect[0][2] == "pow"
                ):
                    ofv = parent.modelentry.modelfit_results.ofv + 100 * -2
                else:
                    ofv = parent.modelentry.modelfit_results.ofv + 100 * -1
            else:
                if len(parent.steps) != 0:
                    ofv = parent.modelentry.modelfit_results.ofv + 100
                else:
                    ofv = parent.modelentry.modelfit_results.ofv + 100 * -1
    else:
        ofv = parent.modelentry.modelfit_results.ofv + 100

    return ofv


@pytest.mark.parametrize(('adaptive_step',), ((True,), (False,)))
def test_adaptive_scope_reduction(load_model_for_test, testdata, adaptive_step):
    p_value = 0.01
    search_space = "COVARIATE?([CL,V],[WGT,APGR],[exp,lin,pow])"

    # Mock version of handle_effects() within covsearch
    def _mock_handle_effects(
        step: int,
        parent: Candidate,
        candidate_effect_funcs: dict,
        index_offset: int,
    ):
        new_candidate_modelentries = []
        for i, effect in enumerate(candidate_effect_funcs.items(), 1):
            candidate_model_entry = task_add_covariate_effect(
                parent.modelentry,
                parent,
                effect,
                index_offset + i,
            )

            # Fake new ofv value of model
            ofv = _mock_fit(effect, parent, adaptive_step)
            new_modelfit = replace(parent.modelentry.modelfit_results, ofv=ofv)

            candidate_model_entry = ModelEntry.create(
                model=candidate_model_entry.model,
                modelfit_results=new_modelfit,
                parent=candidate_model_entry.parent,
            )
            new_candidate_modelentries.append(candidate_model_entry)
        return [
            Candidate(modelentry, parent.steps + (ForwardStep(p_value, AddEffect(*effect)),))
            for modelentry, effect in zip(new_candidate_modelentries, candidate_effect_funcs.keys())
        ]

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    modelres = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')

    candidate_effect_funcs, start = get_effect_funcs_and_base_model(search_space, model)
    start = _start(start, modelres, False)
    candidate = Candidate(start, ())
    state = SearchState(start, start, candidate, [candidate])

    res = _greedy_search(
        state,
        _mock_handle_effects,
        candidate_effect_funcs,
        p_value,
        6,
        "minimization_successful or (rounding_errors and sigdigs>=0.1)",
        True,
    )

    best_candidate = res.all_candidates_so_far[0]
    for c in res.all_candidates_so_far:
        if c.modelentry.modelfit_results.ofv < best_candidate.modelentry.modelfit_results.ofv:
            best_candidate = c

    for c in res.all_candidates_so_far:
        if len(c.steps) == 1:
            assert (
                best_candidate.modelentry.modelfit_results.ofv <= c.modelentry.modelfit_results.ofv
            )
        elif len(c.steps) == 2:
            assert best_candidate.steps[0] == c.steps[0]
        elif len(c.steps) == 3 and adaptive_step:
            assert isinstance(c.steps[-2], AdaptiveStep)
        elif len(c.steps) == 3 and not adaptive_step:
            assert c.steps[-1].effect.parameter == "V"
            assert (
                c.modelentry.modelfit_results.ofv > best_candidate.modelentry.modelfit_results.ofv
            )
        elif len(c.steps) == 4:
            assert c.steps[-1].effect.parameter == "V"
            assert (
                c.modelentry.modelfit_results.ofv > best_candidate.modelentry.modelfit_results.ofv
            )


def _create_candidates(model_entry_factory, funcs, parent_cand, i, p_value):
    parent_model = parent_cand.modelentry.model
    parent_res = parent_cand.modelentry.modelfit_results
    candidates = []
    for key, func in funcs.items():
        cand_model = func(parent_model)
        cand_model = set_name(cand_model, f'run{i}')
        cand_me = model_entry_factory([cand_model], ref_val=parent_res.ofv, parent=parent_model)[0]
        steps = parent_cand.steps + (ForwardStep(p_value, Effect(*key)),)
        cand = Candidate(cand_me, steps)
        candidates.append(cand)
        i += 1
    return candidates


def test_get_best_model_so_far(load_model_for_test, testdata, model_entry_factory):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    modelres = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    parent_model_entry = ModelEntry(model, modelfit_results=modelres)
    parent_cand = Candidate(parent_model_entry, steps=tuple())

    search_space = 'COVARIATE?([CL,VC,MAT],WT,EXP)'
    mfl = ModelFeatures.create_from_mfl_string(search_space)
    effect_funcs = get_exploratory_covariates(mfl)

    p_value = 0.01
    strictness = 'minimization_successful'

    candidates = [Candidate(parent_model_entry, steps=tuple())]
    candidates_step_1 = _create_candidates(
        model_entry_factory, effect_funcs, parent_cand, 1, p_value
    )
    candidates.extend(candidates_step_1)
    model_entries_step_1 = [cand.modelentry for cand in candidates_step_1]

    best_candidate_so_far = get_best_candidate_so_far(
        parent_model_entry, model_entries_step_1, candidates, strictness, p_value
    )
    model_entry_lowest_ofv = min(model_entries_step_1, key=lambda me: me.modelfit_results.ofv)
    assert best_candidate_so_far.modelentry == model_entry_lowest_ofv

    candidates_step_2 = _create_candidates(
        model_entry_factory, effect_funcs, parent_cand, len(candidates_step_1) + 1, p_value
    )
    candidates.extend(candidates_step_2)
    model_entries_step_2 = [cand.modelentry for cand in candidates_step_2]

    best_candidate_so_far = get_best_candidate_so_far(
        parent_model_entry, model_entries_step_2, candidates, strictness, p_value
    )
    model_entry_lowest_ofv = min(model_entries_step_2, key=lambda me: me.modelfit_results.ofv)
    assert best_candidate_so_far.modelentry == model_entry_lowest_ofv


def test_create_result_tables(load_model_for_test, testdata, model_entry_factory):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    modelres = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    parent_model_entry = ModelEntry(model, modelfit_results=modelres)
    parent_cand = Candidate(parent_model_entry, steps=tuple())

    search_space = 'COVARIATE?([CL,VC],[WT,AGE],EXP)'
    mfl = ModelFeatures.create_from_mfl_string(search_space)
    effect_funcs = get_exploratory_covariates(mfl)

    p_value = 0.01
    strictness = 'minimization_successful'

    candidates_step_1 = _create_candidates(
        model_entry_factory, effect_funcs, parent_cand, 1, p_value
    )
    selected_cand = min(candidates_step_1, key=lambda cand: cand.modelentry.modelfit_results.ofv)
    eff = selected_cand.steps[0].effect
    key = (eff.parameter, eff.covariate, eff.fp, eff.operation)
    effect_funcs.pop(key)

    candidates_step_2 = _create_candidates(
        model_entry_factory, effect_funcs, selected_cand, len(candidates_step_1) + 1, p_value
    )
    candidates = [parent_cand] + candidates_step_1 + candidates_step_2
    candidate_model_entries = [cand.modelentry for cand in candidates_step_1 + candidates_step_2]

    tables = create_result_tables(
        candidates,
        candidates[-1].modelentry,
        parent_model_entry,
        parent_model_entry,
        candidate_model_entries,
        (p_value, None),
        strictness,
    )

    summary_models = tables['summary_models']
    assert len(summary_models) == len(candidates)
    steps = set(summary_models.index.get_level_values('step'))
    assert steps == {0, 1, 2}

    summary_tool = tables['summary_tool']
    assert len(summary_tool) == len(candidates)
    steps = set(summary_tool.index.get_level_values('step'))
    assert steps == {0, 1, 2}
    d_params = summary_tool['d_params'].values
    assert set(d_params) == {0, 1, 2}

    steps = tables['steps']
    assert len(steps) == len(candidates)
    assert 'CL' in steps.index[1] and 'AGE' in steps.index[1]
    assert 'CL' in steps.index[2] and 'AGE' not in steps.index[2]

    ofv_summary = tables['ofv_summary']
    assert len(ofv_summary) == 3

    candidate_summary = tables['candidate_summary']
    assert len(candidate_summary) == 4
