from collections import Counter
from dataclasses import dataclass
from itertools import count
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from pharmpy.model import Model
from pharmpy.modeling import add_covariate_effect, copy_model
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.modeling.lrt import best_of_subtree as lrt_best_of_subtree
from pharmpy.modeling.lrt import p_value as lrt_p_value
from pharmpy.tools.common import create_results, update_initial_estimates
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.scm.results import candidate_summary_dataframe, ofv_summary_dataframe
from pharmpy.workflows import Task, Workflow, call_workflow

from .effects import EffectLiteral, Effects, Spec, parse_spec
from .results import COVSearchResults

NAME_WF = 'covsearch'


@dataclass
class Effect:
    parameter: str
    covariate: str
    fp: str
    operation: str


class AddEffect(Effect):
    pass


class RemoveEffect(Effect):
    pass


@dataclass
class Step:
    alpha: float
    effect: Effect


class ForwardStep(Step):
    pass


class BackwardStep(Step):
    pass


@dataclass
class Candidate:
    model: Model
    steps: Tuple[Step, ...]


def create_workflow(
    effects: Union[str, List[Spec]],
    p_forward: float = 0.05,
    max_steps: int = -1,
    algorithm: str = 'scm-forward',
    model: Union[Model, None] = None,
):
    """Run COVsearch tool. For more details, see :ref:`covsearch`.

    Parameters
    ----------
    effects : str | list
        The list of candidates to try, either in DSL str form or in
        (optionally compact) tuple form.
    p_forward : float
        The p-value to use in the likelihood ratio test for forward steps
    max_steps : int
        The maximum number of search steps to make
    algorithm : str
        The search algorithm to use. Currently only 'scm-forward' is supported.
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

    if algorithm != 'scm-forward':
        raise ValueError('covsearch only supports algorithm="scm-forward"')

    wf = Workflow()
    wf.name = NAME_WF

    init_task = init(model)
    wf.add_task(init_task)

    search_task = Task(
        'search',
        task_greedy_forward_search,
        effects,
        p_forward,
        max_steps,
    )

    wf.add_task(search_task, predecessors=init_task)
    search_output = wf.output_tasks

    results_task = Task(
        'results',
        task_results,
        p_forward,
    )

    wf.add_task(results_task, predecessors=search_output)

    return wf


def init(model: Union[Model, None]):
    return (
        Task('init', lambda model: model)
        if model is None
        else Task('init', lambda model: model, model)
    )


def task_greedy_forward_search(
    effects: Union[str, List[Spec]],
    p_forward: float,
    max_steps: int,
    model: Model,
):
    effect_spec = Effects(effects).spec(model) if isinstance(effects, str) else effects
    candidate_effects = sorted(set(parse_spec(effect_spec)))

    best_candidate_so_far = Candidate(model, ())
    all_candidates_so_far = [best_candidate_so_far]

    steps = range(1, max_steps + 1) if max_steps >= 0 else count(1)

    for step in steps:
        if not candidate_effects:
            break

        wf = wf_effects_addition(best_candidate_so_far.model, candidate_effects)
        new_candidate_models = call_workflow(wf, f'{NAME_WF}-effects_addition-{step}')

        all_candidates_so_far.extend(
            Candidate(
                model, best_candidate_so_far.steps + (ForwardStep(p_forward, AddEffect(*effect)),)
            )
            for model, effect in zip(new_candidate_models, candidate_effects)
        )

        parent = best_candidate_so_far.model
        best_model_so_far = lrt_best_of_many(parent, new_candidate_models, p_forward)

        if best_model_so_far is parent:
            return all_candidates_so_far

        best_candidate_so_far = next(
            filter(lambda candidate: candidate.model is best_model_so_far, all_candidates_so_far)
        )

        # NOTE Filter out incompatible effects
        added_effect = best_candidate_so_far.steps[-1].effect

        candidate_effects = [
            effect
            for effect in candidate_effects
            if effect[0] != added_effect.parameter or effect[1] != added_effect.covariate
        ]

    return all_candidates_so_far


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
    model_with_added_effect = copy_model(model, name=f'{model.name}-{effect_index}')
    model_with_added_effect.description = (
        f'add_covariate_effect(<{model.description or model.name}>, {", ".join(map(str,effect))})'
    )
    model_with_added_effect.parent_model = model.name
    update_initial_estimates(model_with_added_effect)
    add_covariate_effect(model_with_added_effect, *effect)
    return model_with_added_effect


def task_results(p_forward: float, candidates: List[Candidate]):
    models = list(map(lambda candidate: candidate.model, candidates))
    base_model, *res_models = models

    res = create_results(COVSearchResults, base_model, base_model, res_models, 'bic', None)

    best_model = lrt_best_of_subtree(base_model, res_models, p_forward)

    res.steps = _make_df_steps(best_model, candidates)
    res.candidate_summary = candidate_summary_dataframe(res.steps)
    res.ofv_summary = ofv_summary_dataframe(res.steps, final_included=True, iterations=True)

    return res


def _make_df_steps(best_model: Model, candidates: List[Candidate]) -> pd.DataFrame:
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
    extended_significant = p_value <= alpha
    assert not selected or extended_significant
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
