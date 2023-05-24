from collections import Counter, defaultdict
from dataclasses import astuple, dataclass, replace
from itertools import count
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import add_covariate_effect, get_pk_parameters
from pharmpy.modeling.covariate_effect import get_covariates_allowed_in_covariate_effect
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.modeling.lrt import p_value as lrt_p_value
from pharmpy.modeling.lrt import test as lrt_test
from pharmpy.results import ModelfitResults
from pharmpy.tools import summarize_modelfit_results
from pharmpy.tools.common import create_results, update_initial_estimates
from pharmpy.tools.mfl.feature.covariate import (
    EffectLiteral,
    InputSpec,
    all_covariate_effects,
    parse_spec,
    spec,
)
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.scm.results import candidate_summary_dataframe, ofv_summary_dataframe
from pharmpy.workflows import Task, Workflow, call_workflow

from ..mfl.filter import covsearch_statement_types
from .results import COVSearchResults

NAME_WF = 'covsearch'

DataFrame = Any  # NOTE should be pd.DataFrame but we want lazy loading


@dataclass(frozen=True)
class Effect:
    parameter: str
    covariate: str
    fp: str
    operation: str


class AddEffect(Effect):
    pass


class RemoveEffect(Effect):
    pass


@dataclass(frozen=True)
class Step:
    alpha: float
    effect: Effect


class ForwardStep(Step):
    pass


class BackwardStep(Step):
    pass


def _added_effects(steps: Tuple[Step, ...]) -> Iterable[Effect]:
    added_effects = defaultdict(list)
    for i, step in enumerate(steps):
        if isinstance(step, ForwardStep):
            added_effects[astuple(step.effect)].append(i)
        else:
            assert isinstance(step, BackwardStep)
            added_effects[astuple(step.effect)].pop()

    pos = {effect: set(indices) for effect, indices in added_effects.items()}

    for i, step in enumerate(steps):
        if isinstance(step, ForwardStep) and i in pos[astuple(step.effect)]:
            yield step.effect


@dataclass
class Candidate:
    model: Model
    steps: Tuple[Step, ...]


@dataclass
class SearchState:
    start_model: Model
    best_candidate_so_far: Candidate
    all_candidates_so_far: List[Candidate]


ALGORITHMS = frozenset(['scm-forward', 'scm-forward-then-backward'])


def create_workflow(
    effects: Union[str, Sequence[InputSpec]],
    p_forward: float = 0.05,
    p_backward: float = 0.01,
    max_steps: int = -1,
    algorithm: str = 'scm-forward-then-backward',
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
):
    """Run COVsearch tool. For more details, see :ref:`covsearch`.

    Parameters
    ----------
    effects : str
        MFL of covariate effects to try
    p_forward : float
        The p-value to use in the likelihood ratio test for forward steps
    p_backward : float
        The p-value to use in the likelihood ratio test for backward steps
    max_steps : int
        The maximum number of search steps to make
    algorithm : str
        The search algorithm to use. Currently 'scm-forward' and
        'scm-forward-then-backward' are supported.
    results : ModelfitResults
        Results of model
    model : Model
        Pharmpy model

    Returns
    -------
    COVSearchResults
        COVsearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import run_covsearch, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> effects = 'COVARIATE([CL, V], [AGE, WT], EXP)'
    >>> res = run_covsearch(effects, model=model, results=results)      # doctest: +SKIP
    """

    wf = Workflow()
    wf.name = NAME_WF

    init_task = init(model)
    wf.add_task(init_task)

    forward_search_task = Task(
        'forward-search',
        task_greedy_forward_search,
        effects,
        p_forward,
        max_steps,
    )

    wf.add_task(forward_search_task, predecessors=init_task)
    search_output = wf.output_tasks

    if algorithm == 'scm-forward-then-backward':
        backward_search_task = Task(
            'backward-search',
            task_greedy_backward_search,
            p_backward,
            max_steps,
        )

        wf.add_task(backward_search_task, predecessors=search_output)
        search_output = wf.output_tasks

    results_task = Task(
        'results',
        task_results,
        p_forward,
        p_backward,
    )

    wf.add_task(results_task, predecessors=search_output)

    return wf


def _init_search_state(model: Model) -> SearchState:
    candidate = Candidate(model, ())
    return SearchState(model, candidate, [candidate])


def init(model: Union[Model, None]):
    return (
        Task('init', _init_search_state)
        if model is None
        else Task('init', _init_search_state, model)
    )


def task_greedy_forward_search(
    context,
    effects: str,
    p_forward: float,
    max_steps: int,
    state: SearchState,
) -> SearchState:
    candidate = state.best_candidate_so_far
    assert state.all_candidates_so_far == [candidate]
    effect_spec = spec(candidate.model, mfl_parse(effects))
    candidate_effects = sorted(set(parse_spec(effect_spec)))

    def handle_effects(
        step: int, parent: Candidate, candidate_effects: List[EffectLiteral], index_offset: int
    ):
        wf = wf_effects_addition(parent.model, parent, candidate_effects, index_offset)
        new_candidate_models = call_workflow(wf, f'{NAME_WF}-effects_addition-{step}', context)

        return [
            Candidate(model, parent.steps + (ForwardStep(p_forward, AddEffect(*effect)),))
            for model, effect in zip(new_candidate_models, candidate_effects)
        ]

    return _greedy_search(
        state,
        handle_effects,
        candidate_effects,
        p_forward,
        max_steps,
    )


def task_greedy_backward_search(
    context,
    p_backward: float,
    max_steps: int,
    state: SearchState,
) -> SearchState:
    def handle_effects(
        step: int, parent: Candidate, candidate_effects: List[EffectLiteral], index_offset: int
    ):
        wf = wf_effects_removal(state.start_model, parent, candidate_effects, index_offset)
        new_candidate_models = call_workflow(wf, f'{NAME_WF}-effects_removal-{step}', context)

        return [
            Candidate(model, parent.steps + (BackwardStep(p_backward, RemoveEffect(*effect)),))
            for model, effect in zip(new_candidate_models, candidate_effects)
        ]

    candidate_effects = list(map(astuple, _added_effects(state.best_candidate_so_far.steps)))

    n_removable_effects = max(0, len(state.best_candidate_so_far.steps) - 1)

    return _greedy_search(
        state,
        handle_effects,
        candidate_effects,
        p_backward,
        min(max_steps, n_removable_effects) if max_steps >= 0 else n_removable_effects,
    )


def _greedy_search(
    state: SearchState,
    handle_effects: Callable[[int, Candidate, List[EffectLiteral], int], List[Candidate]],
    candidate_effects: List[EffectLiteral],
    alpha: float,
    max_steps: int,
) -> SearchState:
    best_candidate_so_far = state.best_candidate_so_far
    all_candidates_so_far = list(state.all_candidates_so_far)  # NOTE this includes start model

    steps = range(1, max_steps + 1) if max_steps >= 0 else count(1)

    for step in steps:
        if not candidate_effects:
            break

        new_candidates = handle_effects(
            step, best_candidate_so_far, candidate_effects, len(all_candidates_so_far) - 1
        )

        all_candidates_so_far.extend(new_candidates)
        new_candidate_models = list(map(lambda candidate: candidate.model, new_candidates))

        parent = best_candidate_so_far.model
        ofvs = [
            np.nan if (mfr := model.modelfit_results) is None else mfr.ofv
            for model in new_candidate_models
        ]
        # NOTE We assume parent.modelfit_results is not None
        assert parent.modelfit_results is not None
        best_model_so_far = lrt_best_of_many(
            parent, new_candidate_models, parent.modelfit_results.ofv, ofvs, alpha
        )

        if best_model_so_far is parent:
            break

        best_candidate_so_far = next(
            filter(lambda candidate: candidate.model is best_model_so_far, all_candidates_so_far)
        )

        # NOTE Filter out incompatible effects
        last_step_effect = best_candidate_so_far.steps[-1].effect

        candidate_effects = [
            effect
            for effect in candidate_effects
            if effect[0] != last_step_effect.parameter or effect[1] != last_step_effect.covariate
        ]

    return SearchState(
        state.start_model,
        best_candidate_so_far,
        all_candidates_so_far,
    )


def wf_effects_addition(
    model: Model, candidate: Candidate, candidate_effects: List[EffectLiteral], index_offset: int
):
    wf = Workflow()

    for i, effect in enumerate(candidate_effects, 1):
        task = Task(
            repr(effect),
            task_add_covariate_effect,
            model,
            candidate,
            effect,
            index_offset + i,
        )
        wf.add_task(task)

    wf_fit = create_fit_workflow(n=len(candidate_effects))
    wf.insert_workflow(wf_fit)

    task_gather = Task('gather', lambda *models: models)
    wf.add_task(task_gather, predecessors=wf.output_tasks)
    return wf


def task_add_covariate_effect(
    model: Model, candidate: Candidate, effect: EffectLiteral, effect_index: int
):
    name = f'covsearch_run{effect_index}'
    description = _create_description(effect, candidate.steps)
    model_with_added_effect = model.replace(name=name, description=description, parent_model=name)
    model_with_added_effect = update_initial_estimates(model_with_added_effect)
    model_with_added_effect = add_covariate_effect(
        model_with_added_effect, *effect, allow_nested=True
    )
    return model_with_added_effect


def _create_description(
    effect_new: Union[Tuple, Effect], steps_prev: Tuple[Step, ...], forward: bool = True
):
    # Will create this type of description: '(CL-AGE-exp);(MAT-AGE-exp);(MAT-AGE-exp-+)'
    def _create_effect_str(effect):
        if isinstance(effect, Tuple):
            param, cov, fp, op = effect
        elif isinstance(effect, Effect):
            param, cov, fp, op = effect.parameter, effect.covariate, effect.fp, effect.operation
        else:
            raise ValueError('Effect must be a tuple or Effect dataclass')
        effect_base = f'{param}-{cov}-{fp}'
        if op == '+':
            effect_base += f'-{op}'
        return f'({effect_base})'

    effect_new_str = _create_effect_str(effect_new)
    effects = []
    for effect_prev in _added_effects(steps_prev):
        effect_prev_str = _create_effect_str(effect_prev)
        if not forward and effect_prev_str == effect_new_str:
            continue
        effects.append(effect_prev_str)
    if forward:
        effects.append(effect_new_str)
    return ';'.join(effects)


def wf_effects_removal(
    base_model: Model, parent: Candidate, candidate_effects: List[EffectLiteral], index_offset: int
):
    wf = Workflow()

    for i, effect in enumerate(candidate_effects, 1):
        task = Task(
            repr(effect),
            task_remove_covariate_effect,
            base_model,
            parent,
            effect,
            index_offset + i,
        )
        wf.add_task(task)

    wf_fit = create_fit_workflow(n=len(candidate_effects))
    wf.insert_workflow(wf_fit)

    task_gather = Task('gather', lambda *models: models)
    wf.add_task(task_gather, predecessors=wf.output_tasks)
    return wf


def task_remove_covariate_effect(
    base_model: Model, candidate: Candidate, effect: EffectLiteral, effect_index: int
):
    model = candidate.model
    name = f'covsearch_run{effect_index}'
    description = _create_description(effect, candidate.steps, forward=False)
    model_with_removed_effect = base_model.replace(
        name=name, description=description, parent_model=model.name
    )

    for kept_effect in _added_effects((*candidate.steps, BackwardStep(-1, RemoveEffect(*effect)))):
        model_with_removed_effect = add_covariate_effect(
            model_with_removed_effect, *astuple(kept_effect), allow_nested=True
        )

    model_with_removed_effect = update_initial_estimates(model_with_removed_effect)
    return model_with_removed_effect


def task_results(p_forward: float, p_backward: float, state: SearchState):
    candidates = state.all_candidates_so_far
    models = list(map(lambda candidate: candidate.model, candidates))
    base_model, *res_models = models
    assert base_model is state.start_model
    best_model = state.best_candidate_so_far.model

    res = create_results(
        COVSearchResults, base_model, base_model, res_models, 'lrt', (p_forward, p_backward)
    )

    steps = _make_df_steps(best_model, candidates)
    res = replace(
        res,
        steps=steps,
        candidate_summary=candidate_summary_dataframe(steps),
        ofv_summary=ofv_summary_dataframe(steps, final_included=True, iterations=True),
        summary_tool=_modify_summary_tool(res.summary_tool, steps),
        summary_models=_summarize_models(models, steps),
    )

    return res


def _modify_summary_tool(summary_tool, steps):
    step_cols_to_keep = ['step', 'pvalue', 'goal_pvalue', 'is_backward', 'selected', 'model']
    steps_df = steps.reset_index()[step_cols_to_keep].set_index(['step', 'model'])

    summary_tool_new = steps_df.join(summary_tool)
    column_to_move = summary_tool_new.pop('description')

    summary_tool_new.insert(0, 'description', column_to_move)
    return summary_tool_new.drop(['rank'], axis=1)


def _summarize_models(models, steps):
    summary_models = summarize_modelfit_results([model.modelfit_results for model in models])
    summary_models['step'] = steps.reset_index().set_index(['model'])['step']

    return summary_models.reset_index().set_index(['step', 'model'])


def _make_df_steps(best_model: Model, candidates: List[Candidate]):
    models_dict = {candidate.model.name: candidate.model for candidate in candidates}
    children_count = Counter(candidate.model.parent_model for candidate in candidates)

    data = (
        _make_df_steps_row(models_dict, children_count, best_model, candidate)
        for candidate in candidates
    )

    return pd.DataFrame.from_records(
        data,
        index=['step', 'parameter', 'covariate', 'extended_state'],
    )


def _make_df_steps_row(
    models_dict: dict, children_count: Counter, best_model: Model, candidate: Candidate
):
    model = candidate.model
    parent_model = models_dict[model.parent_model]
    reduced_ofv = np.nan if (mfr := parent_model.modelfit_results) is None else mfr.ofv
    extended_ofv = np.nan if (mfr := model.modelfit_results) is None else mfr.ofv
    ofv_drop = reduced_ofv - extended_ofv
    if candidate.steps:
        last_step = candidate.steps[-1]
        last_effect = last_step.effect
        parameter, covariate = last_effect.parameter, last_effect.covariate
        extended_state = f'{last_effect.operation} {last_effect.fp}'
        is_backward = isinstance(last_step, BackwardStep)
        alpha = last_step.alpha
        extended_significant = lrt_test(
            parent_model,
            candidate.model,
            reduced_ofv,
            extended_ofv,
            alpha,
        )
    else:
        parameter, covariate, extended_state = '', '', ''
        is_backward = False
        alpha, extended_significant = np.nan, np.nan

    p_value = lrt_p_value(parent_model, model, reduced_ofv, extended_ofv)
    selected = children_count[candidate.model.name] >= 1 or candidate.model is best_model

    assert not selected or (model is parent_model) or extended_significant
    return {
        'step': len(candidate.steps),
        'parameter': parameter,
        'covariate': covariate,
        'extended_state': extended_state,
        'reduced_ofv': reduced_ofv,
        'extended_ofv': extended_ofv,
        'ofv_drop': ofv_drop,
        'delta_df': len(model.parameters.nonfixed) - len(parent_model.parameters.nonfixed),
        'pvalue': p_value,
        'goal_pvalue': alpha,
        'is_backward': is_backward,
        'extended_significant': extended_significant,
        'selected': selected,
        'model': model.name,
        'covariate_effects': np.nan,
    }


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(effects, p_forward, p_backward, algorithm, model):
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f'Invalid `algorithm`: got `{algorithm}`, must be one of {sorted(ALGORITHMS)}.'
        )

    if not 0 < p_forward <= 1:
        raise ValueError(
            f'Invalid `p_forward`: got `{p_forward}`, must be a float in range (0, 1].'
        )

    if not 0 < p_backward <= 1:
        raise ValueError(
            f'Invalid `p_backward`: got `{p_backward}`, must be a float in range (0, 1].'
        )

    if model is not None:
        try:
            statements = mfl_parse(effects)
        except:  # noqa E722
            raise ValueError(f'Invalid `effects`, could not be parsed: `{effects}`')

        bad_statements = list(
            filter(
                lambda statement: not isinstance(statement, covsearch_statement_types),
                statements,
            )
        )

        if bad_statements:
            raise ValueError(
                f'Invalid `effects`: found unknown statement of type {type(bad_statements[0]).__name__}.'
            )
        effect_spec = spec(model, statements)

        candidate_effects = map(lambda x: Effect(*x), sorted(set(parse_spec(effect_spec))))

        allowed_covariates = get_covariates_allowed_in_covariate_effect(model)
        allowed_parameters = set(get_pk_parameters(model)).union(
            str(statement.symbol) for statement in model.statements.before_odes
        )
        allowed_covariate_effects = set(all_covariate_effects)
        allowed_ops = set(['*', '+'])

        for effect in candidate_effects:
            if effect.covariate not in allowed_covariates:
                raise ValueError(
                    f'Invalid `effects` because of invalid covariate found in'
                    f' effects: got `{effect.covariate}`,'
                    f' must be in {sorted(allowed_covariates)}.'
                )
            if effect.parameter not in allowed_parameters:
                raise ValueError(
                    f'Invalid `effects` because of invalid parameter found in'
                    f' effects: got `{effect.parameter}`,'
                    f' must be in {sorted(allowed_parameters)}.'
                )
            if effect.fp not in allowed_covariate_effects:
                raise ValueError(
                    f'Invalid `effects` because of invalid effect function found in'
                    f' effects: got `{effect.fp}`,'
                    f' must be in {sorted(allowed_covariate_effects)}.'
                )
            if effect.operation not in allowed_ops:
                raise ValueError(
                    f'Invalid `effects` because of invalid effect operation found in'
                    f' effects: got `{effect.operation}`,'
                    f' must be in {sorted(allowed_ops)}.'
                )
