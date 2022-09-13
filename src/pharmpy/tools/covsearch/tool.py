from collections import Counter, defaultdict
from dataclasses import astuple, dataclass
from itertools import count
from typing import Any, Callable, Iterable, List, Tuple, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.modeling import add_covariate_effect, copy_model
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.modeling.lrt import p_value as lrt_p_value
from pharmpy.modeling.lrt import test as lrt_test
from pharmpy.tools.common import create_results, update_initial_estimates
from pharmpy.tools.mfl.feature.covariate import EffectLiteral, Spec, parse_spec, spec
from pharmpy.tools.mfl.parse import parse
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.scm.results import candidate_summary_dataframe, ofv_summary_dataframe
from pharmpy.workflows import Task, Workflow, call_workflow

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
    effects: Union[str, List[Spec]],
    p_forward: float = 0.05,
    p_backward: float = 0.01,
    max_steps: int = -1,
    algorithm: str = 'scm-forward-then-backward',
    model: Union[Model, None] = None,
):
    """Run COVsearch tool. For more details, see :ref:`covsearch`.

    Parameters
    ----------
    effects : str | list
        The list of candidate parameter-covariate effects to try, either as a
        MFL sentence or in (optionally compact) tuple form.
    p_forward : float
        The p-value to use in the likelihood ratio test for forward steps
    p_backward : float
        The p-value to use in the likelihood ratio test for backward steps
    max_steps : int
        The maximum number of search steps to make
    algorithm : str
        The search algorithm to use. Currently 'scm-forward' and
        'scm-forward-then-backward' are supported.
    model : Model
        Pharmpy model

    Returns
    -------
    COVSearchResults
        COVsearch tool result object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> from pharmpy.tools import run_covsearch  # doctest: +SKIP
    >>> res = run_covsearch([
    ...     ('CL', 'WGT', 'exp', '*'),
    ...     ('CL', 'APGR', 'exp', '*'),
    ...     ('V', 'WGT', 'exp', '*'),
    ...     ('V', 'APGR', 'exp', '*'),
    ... ], model=model)      # doctest: +SKIP

    """

    if algorithm not in ALGORITHMS:
        raise ValueError(f'covsearch only supports algorithm in {sorted(ALGORITHMS)}')

    if p_forward <= 0:
        raise ValueError(f'p_forward must be positive (got {p_forward})')

    if p_backward <= 0:
        raise ValueError(f'p_backward must be positive (got {p_backward})')

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
    effects: Union[str, List[Spec]],
    p_forward: float,
    max_steps: int,
    state: SearchState,
) -> SearchState:
    candidate = state.best_candidate_so_far
    assert state.all_candidates_so_far == [candidate]
    effect_spec = spec(candidate.model, parse(effects)) if isinstance(effects, str) else effects
    candidate_effects = sorted(set(parse_spec(effect_spec)))

    def handle_effects(step: int, parent: Candidate, candidate_effects: List[EffectLiteral]):

        wf = wf_effects_addition(parent.model, candidate_effects)
        new_candidate_models = call_workflow(wf, f'{NAME_WF}-effects_addition-{step}')

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
    p_backward: float,
    max_steps: int,
    state: SearchState,
) -> SearchState:
    def handle_effects(step: int, parent: Candidate, candidate_effects: List[EffectLiteral]):

        wf = wf_effects_removal(state.start_model, parent, candidate_effects)
        new_candidate_models = call_workflow(wf, f'{NAME_WF}-effects_removal-{step}')

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
    handle_effects: Callable[[int, Candidate, List[EffectLiteral]], List[Candidate]],
    candidate_effects: List[EffectLiteral],
    alpha: float,
    max_steps: int,
) -> SearchState:

    best_candidate_so_far = state.best_candidate_so_far
    all_candidates_so_far = list(state.all_candidates_so_far)

    steps = range(1, max_steps + 1) if max_steps >= 0 else count(1)

    for step in steps:
        if not candidate_effects:
            break

        new_candidates = handle_effects(step, best_candidate_so_far, candidate_effects)

        all_candidates_so_far.extend(new_candidates)
        new_candidate_models = map(lambda candidate: candidate.model, new_candidates)

        parent = best_candidate_so_far.model
        best_model_so_far = lrt_best_of_many(parent, new_candidate_models, alpha)

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


def wf_effects_addition(model: Model, candidate_effects: List[EffectLiteral]):
    wf = Workflow()

    for i, effect in enumerate(candidate_effects, 1):
        task = Task(
            repr(effect),
            task_add_covariate_effect,
            model,
            effect,
            i,
        )
        wf.add_task(task)

    wf_fit = create_fit_workflow(n=len(candidate_effects))
    wf.insert_workflow(wf_fit)

    task_gather = Task('gather', lambda *models: models)
    wf.add_task(task_gather, predecessors=wf.output_tasks)
    return wf


def task_add_covariate_effect(model: Model, effect: EffectLiteral, effect_index: int):
    model_with_added_effect = copy_model(model, name=f'{model.name}+{effect_index}')
    model_with_added_effect.description = (
        f'add_covariate_effect(<{model.description or model.name}>, {", ".join(map(str,effect))})'
    )
    model_with_added_effect.parent_model = model.name
    update_initial_estimates(model_with_added_effect)
    add_covariate_effect(model_with_added_effect, *effect)
    return model_with_added_effect


def wf_effects_removal(
    base_model: Model, parent: Candidate, candidate_effects: List[EffectLiteral]
):
    wf = Workflow()

    for i, effect in enumerate(candidate_effects, 1):
        task = Task(
            repr(effect),
            task_remove_covariate_effect,
            base_model,
            parent,
            effect,
            i,
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
    model_with_removed_effect = copy_model(base_model, name=f'{model.name}-{effect_index}')
    model_with_removed_effect.description = (
        'remove_covariate_effect'
        f'(<{model.description or model.name}>, {", ".join(map(str,effect))})'
    )
    model_with_removed_effect.parent_model = model.name

    for kept_effect in _added_effects((*candidate.steps, BackwardStep(-1, RemoveEffect(*effect)))):
        add_covariate_effect(model_with_removed_effect, *astuple(kept_effect))

    update_initial_estimates(model_with_removed_effect)
    return model_with_removed_effect


def task_results(state: SearchState):
    candidates = state.all_candidates_so_far
    models = list(map(lambda candidate: candidate.model, candidates))
    base_model, *res_models = models
    assert base_model is state.start_model
    best_model = state.best_candidate_so_far.model

    res = create_results(COVSearchResults, base_model, base_model, res_models, 'bic', None)

    res.steps = _make_df_steps(best_model, candidates)
    res.candidate_summary = candidate_summary_dataframe(res.steps)
    res.ofv_summary = ofv_summary_dataframe(res.steps, final_included=True, iterations=True)

    return res


def _make_df_steps(best_model: Model, candidates: List[Candidate]) -> DataFrame:
    models_dict = {candidate.model.name: candidate.model for candidate in candidates}
    children_count = Counter(candidate.model.parent_model for candidate in candidates)

    data = (
        _make_df_steps_row(models_dict, children_count, best_model, candidate)
        for candidate in candidates
        if candidate.steps
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
    reduced_ofv = (
        np.nan if parent_model.modelfit_results is None else parent_model.modelfit_results.ofv
    )
    extended_ofv = np.nan if model.modelfit_results is None else model.modelfit_results.ofv
    ofv_drop = reduced_ofv - extended_ofv
    last_step = candidate.steps[-1]
    last_effect = last_step.effect
    is_backward = isinstance(last_step, BackwardStep)
    p_value = lrt_p_value(parent_model, model)
    alpha = last_step.alpha
    selected = children_count[candidate.model.name] >= 1 or candidate.model is best_model
    extended_significant = lrt_test(parent_model, candidate.model, alpha)
    assert not selected or (candidate.model is parent_model) or extended_significant
    return {
        'step': len(candidate.steps),
        'parameter': last_effect.parameter,
        'covariate': last_effect.covariate,
        'extended_state': f'{last_effect.operation} {last_effect.fp}',
        'reduced_ofv': reduced_ofv,
        'extended_ofv': extended_ofv,
        'ofv_drop': ofv_drop,
        'delta_df': len(model.parameters) - len(parent_model.parameters),
        'pvalue': p_value,
        'goal_pvalue': alpha,
        'is_backward': is_backward,
        'extended_significant': extended_significant,
        'selected': selected,
        'directory': str(candidate.model.database.path),
        'model': candidate.model.name,
        'covariate_effects': np.nan,
    }
