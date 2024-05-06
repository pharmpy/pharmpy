from collections import Counter, defaultdict
from dataclasses import astuple, dataclass, replace
from itertools import count
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import (
    get_pk_parameters,
    has_covariate_effect,
    remove_covariate_effect,
    set_estimation_step,
)
from pharmpy.modeling.covariate_effect import get_covariates_allowed_in_covariate_effect
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.modeling.lrt import p_value as lrt_p_value
from pharmpy.modeling.lrt import test as lrt_test
from pharmpy.tools import is_strictness_fulfilled
from pharmpy.tools.common import create_results, update_initial_estimates
from pharmpy.tools.mfl.feature.covariate import EffectLiteral, all_covariate_effects
from pharmpy.tools.mfl.feature.covariate import features as covariate_features
from pharmpy.tools.mfl.feature.covariate import parse_spec, spec
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.mfl.statement.feature.covariate import Covariate
from pharmpy.tools.mfl.statement.feature.symbols import Wildcard
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import summarize_modelfit_results_from_entries
from pharmpy.tools.scm.results import candidate_summary_dataframe, ofv_summary_dataframe
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.hashing import ModelHash
from pharmpy.workflows.results import ModelfitResults

from ..mfl.filter import COVSEARCH_STATEMENT_TYPES
from ..mfl.parse import ModelFeatures, get_model_features
from .results import COVSearchResults

NAME_WF = 'covsearch'

DataFrame = Any  # NOTE: should be pd.DataFrame but we want lazy loading


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


class DummyEffect(Effect):
    pass


@dataclass(frozen=True)
class Step:
    alpha: float
    effect: Effect


class ForwardStep(Step):
    pass


class BackwardStep(Step):
    pass


class AdaptiveStep(Step):
    pass


def _added_effects(steps: Tuple[Step, ...]) -> Iterable[Effect]:
    added_effects = defaultdict(list)
    for i, step in enumerate(steps):
        if isinstance(step, ForwardStep):
            added_effects[astuple(step.effect)].append(i)
        elif isinstance(step, BackwardStep):
            added_effects[astuple(step.effect)].pop()
        elif isinstance(step, AdaptiveStep):
            pass
        else:
            raise ValueError("Unknown step ({step}) added")

    pos = {effect: set(indices) for effect, indices in added_effects.items()}

    for i, step in enumerate(steps):
        if isinstance(step, ForwardStep) and i in pos[astuple(step.effect)]:
            yield step.effect


@dataclass
class Candidate:
    modelentry: ModelEntry
    steps: Tuple[Step, ...]


@dataclass
class SearchState:
    user_input_modelentry: ModelEntry
    start_modelentry: ModelEntry
    best_candidate_so_far: Candidate
    all_candidates_so_far: List[Candidate]


ALGORITHMS = ('scm-forward', 'scm-forward-then-backward')


def create_workflow(
    search_space: Union[str, ModelFeatures],
    p_forward: float = 0.01,
    p_backward: float = 0.001,
    max_steps: int = -1,
    algorithm: Literal[ALGORITHMS] = 'scm-forward-then-backward',
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    max_eval: bool = False,
    adaptive_scope_reduction: bool = False,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    naming_index_offset: Optional[int] = 0,
):
    """Run COVsearch tool. For more details, see :ref:`covsearch`.

    Parameters
    ----------
    search_space : str
        MFL of covariate effects to try
    p_forward : float
        The p-value to use in the likelihood ratio test for forward steps
    p_backward : float
        The p-value to use in the likelihood ratio test for backward steps
    max_steps : int
        The maximum number of search steps to make
    algorithm : {'scm-forward', 'scm-forward-then-backward'}
        The search algorithm to use. Currently, 'scm-forward' and
        'scm-forward-then-backward' are supported.
    results : ModelfitResults
        Results of model
    model : Model
        Pharmpy model
    max_eval : bool
        Limit the number of function evaluations to 3.1 times that of the
        base model. Default is False.
    adaptive_scope_reduction : bool
        Stash all non-significant parameter-covariate effects to be tested
        after all significant effects have been tested. Once all these have been
        tested, try adding the stashed effects once more with a regular forward approach.
        Default is False
    strictness : str or None
        Strictness criteria
    naming_index_offset: int
        index offset for naming of runs. Default is 0.

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
    >>> search_space = 'COVARIATE([CL, V], [AGE, WT], EXP)'
    >>> res = run_covsearch(search_space, model=model, results=results)      # doctest: +SKIP
    """

    wb = WorkflowBuilder(name=NAME_WF)

    # FIXME : Handle when model is None
    start_task = Task("create_modelentry", _start, model, results, max_eval)
    init_task = Task("init", _init_search_state, search_space)
    wb.add_task(init_task, predecessors=start_task)

    forward_search_task = Task(
        'forward-search',
        task_greedy_forward_search,
        p_forward,
        max_steps,
        naming_index_offset,
        strictness,
        adaptive_scope_reduction,
    )

    wb.add_task(forward_search_task, predecessors=init_task)
    search_output = wb.output_tasks

    if algorithm == 'scm-forward-then-backward':
        backward_search_task = Task(
            'backward-search',
            task_greedy_backward_search,
            p_backward,
            max_steps,
            naming_index_offset,
            strictness,
        )

        wb.add_task(backward_search_task, predecessors=search_output)
        search_output = wb.output_tasks

    results_task = Task(
        'results',
        task_results,
        p_forward,
        p_backward,
        strictness,
    )

    wb.add_task(results_task, predecessors=search_output)

    return Workflow(wb)


def _start(model, results, max_eval):
    if max_eval:
        max_eval_number = round(3.1 * results.function_evaluations_iterations.loc[1])
        # Change last instead of first?
        first_es = model.execution_steps[0]
        model = set_estimation_step(model, first_es.method, 0, maximum_evaluations=max_eval_number)
    return ModelEntry.create(model=model, parent=None, modelfit_results=results)


def _init_search_state(context, search_space: str, modelentry: ModelEntry) -> SearchState:
    model = modelentry.model
    effect_funcs, filtered_model = filter_search_space_and_model(search_space, model)
    if filtered_model != model:
        filtered_modelentry = ModelEntry.create(model=filtered_model)
        filtered_fit_wf = create_fit_workflow(modelentries=[filtered_modelentry])
        filtered_modelentry = call_workflow(filtered_fit_wf, 'fit_filtered_model', context)
    else:
        filtered_modelentry = modelentry
    candidate = Candidate(filtered_modelentry, ())
    return (effect_funcs, SearchState(modelentry, filtered_modelentry, candidate, [candidate]))


def filter_search_space_and_model(search_space, model):
    filtered_model = model.replace(name="filtered_input_model", parent_model="filtered_input_model")
    if isinstance(search_space, str):
        search_space = ModelFeatures.create_from_mfl_string(search_space)
    ss_mfl = search_space.expand(filtered_model)  # Expand to remove LET/REF
    model_mfl = ModelFeatures.create_from_mfl_string(get_model_features(filtered_model))

    # Remove all covariates not part of the search space
    to_be_removed = model_mfl - ss_mfl
    covariate_list = to_be_removed.mfl_statement_list(["covariate"])
    description = []
    if len(covariate_list) != 0:
        description.append("REMOVED")
        for cov_effect in parse_spec(spec(filtered_model, covariate_list)):
            if cov_effect[2].lower() == "custom":
                try:
                    filtered_model = remove_covariate_effect(
                        filtered_model, cov_effect[0], cov_effect[1]
                    )
                    description.append(f'({cov_effect[0]}-{cov_effect[1]}-{cov_effect[2]})')

                except AssertionError:
                    # All custom effects are grouped and therefor might not exist
                    # FIXME : remove_covariate_effect not raise if non-existing ?
                    pass
            else:
                filtered_model = remove_covariate_effect(
                    filtered_model, cov_effect[0], cov_effect[1]
                )
                description.append(f'({cov_effect[0]}-{cov_effect[1]}-{cov_effect[2]})')
        filtered_model = filtered_model.replace(description=';'.join(description))

    all_forced_cov = tuple([c for c in ss_mfl.covariate if not c.optional.option])
    all_forced = all_funcs(Model(), all_forced_cov)
    added_comb = set(k[1:3] for k in all_forced.keys())

    ss_cov = ss_mfl - model_mfl
    forced_add_cov = tuple([c for c in ss_cov.covariate if not c.optional.option])
    mandatory_funcs = all_funcs(Model(), forced_add_cov)

    optional_cov_list = tuple(c for c in ss_cov.covariate if c.optional.option)
    optional_cov = ModelFeatures.create(covariate=optional_cov_list)
    optional_funcs = optional_cov.convert_to_funcs()
    optional_funcs = {k: v for k, v in optional_funcs.items() if not k[1:3] in added_comb}
    optional_remove = {k: v for k, v in optional_funcs.items() if k[-1] == "REMOVE"}
    optional_add = {k: v for k, v in optional_funcs.items() if k[-1] == "ADD"}

    def func_description(effect_funcs, model=None, add=True):
        d = []
        for eff_descriptor, _ in effect_funcs.items():
            if model:
                if (
                    has_covariate_effect(model, eff_descriptor[1], eff_descriptor[2])
                    if add
                    else not has_covariate_effect(model, eff_descriptor[1], eff_descriptor[2])
                ):
                    d.append(f'({eff_descriptor[1]}-{eff_descriptor[2]}-{eff_descriptor[3]})')
            else:
                d.append(f'({eff_descriptor[1]}-{eff_descriptor[2]}-{eff_descriptor[3]})')
        return d

    # Remove all optional covariates
    if len(optional_remove) != 0:
        potential_extension = func_description(optional_remove, filtered_model, add=True)
        for _, optional_func in optional_remove.items():
            filtered_model = optional_func(filtered_model)
        if len(potential_extension) != 0:
            if not description:
                description.append("REMOVED")
            description.extend(potential_extension)

    # Add all mandatory covariates
    if len(mandatory_funcs) != 0:
        change_description = True
        for eff_descriptor, mandatory_func in mandatory_funcs.items():
            if not has_covariate_effect(filtered_model, eff_descriptor[1], eff_descriptor[2]):
                if change_description:
                    description.append("ADDED")
                    description.extend(func_description(mandatory_funcs, filtered_model, add=False))
                    change_description = False
                filtered_model = mandatory_func(filtered_model)

    # Filter unneccessary keys from fuctions
    optional_add = {k[1:-1]: v for k, v in optional_add.items()}

    if len(description) > 1:
        filtered_model = filtered_model.replace(description=';'.join(description))
        return (optional_add, filtered_model)
    else:
        return (optional_add, model)


def task_greedy_forward_search(
    context,
    p_forward: float,
    max_steps: int,
    naming_index_offset: int,
    strictness: Optional[str],
    adaptive_scope_reduction: bool,
    state_and_effect: Tuple[SearchState, dict],
) -> SearchState:
    for temp in state_and_effect:
        if isinstance(temp, SearchState):
            state = temp
        else:
            candidate_effect_funcs = temp
    candidate = state.best_candidate_so_far
    assert state.all_candidates_so_far == [candidate]

    def handle_effects(
        step: int,
        parent: Candidate,
        candidate_effect_funcs: dict,
        index_offset: int,
    ):
        index_offset = index_offset + naming_index_offset
        wf = wf_effects_addition(parent.modelentry, parent, candidate_effect_funcs, index_offset)
        new_candidate_modelentries = call_workflow(
            wf, f'{NAME_WF}-effects_addition-{step}', context
        )
        return [
            Candidate(modelentry, parent.steps + (ForwardStep(p_forward, AddEffect(*effect)),))
            for modelentry, effect in zip(new_candidate_modelentries, candidate_effect_funcs.keys())
        ]

    return _greedy_search(
        state,
        handle_effects,
        candidate_effect_funcs,
        p_forward,
        max_steps,
        strictness,
        adaptive_scope_reduction,
    )


def task_greedy_backward_search(
    context,
    p_backward: float,
    max_steps: int,
    naming_index_offset,
    strictness: Optional[str],
    state: SearchState,
) -> SearchState:
    def handle_effects(
        step: int,
        parent: Candidate,
        candidate_effect_funcs: List[EffectLiteral],
        index_offset: int,
    ):
        index_offset = index_offset + naming_index_offset
        wf = wf_effects_removal(parent, candidate_effect_funcs, index_offset)
        new_candidate_modelentries = call_workflow(wf, f'{NAME_WF}-effects_removal-{step}', context)

        return [
            Candidate(modelentry, parent.steps + (BackwardStep(p_backward, RemoveEffect(*effect)),))
            for modelentry, effect in zip(new_candidate_modelentries, candidate_effect_funcs.keys())
        ]

    optional_effects = list(map(astuple, _added_effects(state.best_candidate_so_far.steps)))
    candidate_effect_funcs = dict(
        covariate_features(
            state.best_candidate_so_far.modelentry.model,
            ModelFeatures.create_from_mfl_string(
                get_model_features(state.best_candidate_so_far.modelentry.model)
            ).covariate,
            remove=True,
        )
    )

    candidate_effect_funcs = {
        k[1:-1]: v for k, v in candidate_effect_funcs.items() if (*k[1:-1],) in optional_effects
    }

    n_removable_effects = max(0, len(state.best_candidate_so_far.steps) - 1)

    return _greedy_search(
        state,
        handle_effects,
        candidate_effect_funcs,
        p_backward,
        min(max_steps, n_removable_effects) if max_steps >= 0 else n_removable_effects,
        strictness,
    )


def _greedy_search(
    state: SearchState,
    handle_effects: Callable[[int, Candidate, List[EffectLiteral], int], List[Candidate]],
    candidate_effect_funcs: dict,
    alpha: float,
    max_steps: int,
    strictness: Optional[str],
    adaptive_scope_reduction: bool = False,
) -> SearchState:
    best_candidate_so_far = state.best_candidate_so_far
    all_candidates_so_far = list(
        state.all_candidates_so_far
    )  # NOTE: This includes start model/filtered model

    steps = range(1, max_steps + 1) if max_steps >= 0 else count(1)

    nonsignificant_effects, all_candidates_so_far, best_candidate_so_far = perform_step_procedure(
        steps,
        candidate_effect_funcs,
        handle_effects,
        all_candidates_so_far,
        best_candidate_so_far,
        strictness,
        alpha,
        adaptive_scope_reduction,
    )

    if nonsignificant_effects and adaptive_scope_reduction:

        # TODO : Different number of steps for adaptive part (?)
        steps = range(1, max_steps + 1) if max_steps >= 0 else count(1)

        # Filter incompatible effects
        parameter_steps_taken = [step.effect.parameter for step in best_candidate_so_far.steps]
        cov_steps_taken = [step.effect.covariate for step in best_candidate_so_far.steps]

        nonsignificant_effects = {
            effect_description: effect_func
            for effect_description, effect_func in nonsignificant_effects.items()
            if effect_description[0] not in parameter_steps_taken
            or effect_description[1] not in cov_steps_taken
        }

        if nonsignificant_effects:
            # Add adaptive step if the best candidate model is not part of the latest step
            max_steps_taken = max([len(c.steps) for c in all_candidates_so_far])
            add_adaptive_step = len(best_candidate_so_far.steps) != max_steps_taken

            _, all_candidates_so_far, best_candidate_so_far = perform_step_procedure(
                steps,
                nonsignificant_effects,
                handle_effects,
                all_candidates_so_far,
                best_candidate_so_far,
                strictness,
                alpha,
                adaptive_scope_reduction=False,
                add_adaptive_step=add_adaptive_step,
            )

    return SearchState(
        state.user_input_modelentry,
        state.start_modelentry,
        best_candidate_so_far,
        all_candidates_so_far,
    )


def perform_step_procedure(
    steps,
    candidate_effect_funcs,
    handle_effects,
    all_candidates_so_far,
    best_candidate_so_far,
    strictness,
    alpha,
    adaptive_scope_reduction,
    add_adaptive_step=False,
):

    nonsignificant_effects = {}

    for step in steps:
        if not candidate_effect_funcs:
            break
        if add_adaptive_step and step == 1:
            temp_best_candidate_so_far = Candidate(
                best_candidate_so_far.modelentry,
                best_candidate_so_far.steps + (AdaptiveStep(alpha, DummyEffect("", "", "", "")),),
            )
            new_candidates = handle_effects(
                step,
                temp_best_candidate_so_far,
                candidate_effect_funcs,
                len(all_candidates_so_far) - 1,
            )
        else:
            new_candidates = handle_effects(
                step, best_candidate_so_far, candidate_effect_funcs, len(all_candidates_so_far) - 1
            )

        all_candidates_so_far.extend(new_candidates)
        new_candidate_modelentries = list(
            map(lambda candidate: candidate.modelentry, new_candidates)
        )

        parent_modelentry = best_candidate_so_far.modelentry
        ofvs = [
            (
                np.nan
                if modelentry.modelfit_results is None
                or not is_strictness_fulfilled(
                    modelentry.modelfit_results, modelentry.model, strictness
                )
                else modelentry.modelfit_results.ofv
            )
            for modelentry in new_candidate_modelentries
        ]
        # NOTE: We assume parent_modelentry.modelfit_results is not None
        assert parent_modelentry.modelfit_results is not None
        best_model_so_far = lrt_best_of_many(
            parent_modelentry,
            new_candidate_modelentries,
            parent_modelentry.modelfit_results.ofv,
            ofvs,
            alpha,
        )

        if best_model_so_far is parent_modelentry:
            break

        best_candidate_so_far = next(
            filter(
                lambda candidate: candidate.modelentry is best_model_so_far, all_candidates_so_far
            )
        )

        # TODO : Find all non-significant models and stash the most recently added effect
        if adaptive_scope_reduction:
            last_step = best_candidate_so_far.steps[-1]
            is_backward = isinstance(last_step, BackwardStep)
            if not is_backward:
                for new_cand in new_candidates:
                    if alpha <= lrt_p_value(
                        parent_modelentry.model,
                        new_cand.modelentry.model,
                        parent_modelentry.modelfit_results.ofv,
                        new_cand.modelentry.modelfit_results.ofv,
                    ):
                        last_step_effect = new_cand.steps[-1].effect
                        key = (
                            last_step_effect.parameter,
                            last_step_effect.covariate,
                            last_step_effect.fp,
                            last_step_effect.operation,
                        )
                        nonsignificant_effects[key] = candidate_effect_funcs[key]

        # NOTE: Filter out incompatible effects

        # Filter effects with same parameter or covariate
        last_step_effect = best_candidate_so_far.steps[-1].effect

        candidate_effect_funcs = {
            effect_description: effect_func
            for effect_description, effect_func in candidate_effect_funcs.items()
            if effect_description[0] != last_step_effect.parameter
            or effect_description[1] != last_step_effect.covariate
        }

        # Filter away any stashed effects as well
        nonsig_param_cov_eff = tuple((eff[0], eff[1]) for eff in nonsignificant_effects.keys())

        candidate_effect_funcs = {
            effect_description: effect_func
            for effect_description, effect_func in candidate_effect_funcs.items()
            if (effect_description[0], effect_description[1]) not in nonsig_param_cov_eff
        }

    return nonsignificant_effects, all_candidates_so_far, best_candidate_so_far


def wf_effects_addition(
    modelentry: ModelEntry,
    candidate: Candidate,
    candidate_effect_funcs: dict,
    index_offset: int,
):
    wb = WorkflowBuilder()

    for i, effect in enumerate(candidate_effect_funcs.items(), 1):
        task = Task(
            repr(effect[0]),
            task_add_covariate_effect,
            modelentry,
            candidate,
            effect,
            index_offset + i,
        )
        wb.add_task(task)

    wf_fit = create_fit_workflow(n=len(candidate_effect_funcs))
    wb.insert_workflow(wf_fit)

    task_gather = Task('gather', lambda *models: models)
    wb.add_task(task_gather, predecessors=wb.output_tasks)
    return Workflow(wb)


def task_add_covariate_effect(
    modelentry: ModelEntry, candidate: Candidate, effect: dict, effect_index: int
):
    model = modelentry.model
    name = f'covsearch_run{effect_index}'
    description = _create_description(effect[0], candidate.steps)
    model_with_added_effect = model.replace(name=name, description=description)
    model_with_added_effect = update_initial_estimates(
        model_with_added_effect, modelentry.modelfit_results
    )
    func = effect[1]
    model_with_added_effect = func(model_with_added_effect, allow_nested=True)
    return ModelEntry.create(
        model=model_with_added_effect, parent=candidate.modelentry.model, modelfit_results=None
    )


def _create_description(effect_new: dict, steps_prev: Tuple[Step, ...], forward: bool = True):
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
    parent: Candidate,
    candidate_effect_funcs: dict,
    index_offset: int,
):
    wb = WorkflowBuilder()

    for i, effect in enumerate(candidate_effect_funcs.items(), 1):
        task = Task(
            repr(effect[0]),
            task_remove_covariate_effect,
            parent,
            effect,
            index_offset + i,
        )
        wb.add_task(task)

    wf_fit = create_fit_workflow(n=len(candidate_effect_funcs))
    wb.insert_workflow(wf_fit)

    task_gather = Task('gather', lambda *models: models)
    wb.add_task(task_gather, predecessors=wb.output_tasks)
    return Workflow(wb)


def task_remove_covariate_effect(candidate: Candidate, effect: dict, effect_index: int):
    model = candidate.modelentry.model
    name = f'covsearch_run{effect_index}'
    description = _create_description(effect[0], candidate.steps, forward=False)
    model_with_removed_effect = model.replace(
        name=name, description=description, parent_model=model.name
    )

    func = effect[1]
    model_with_removed_effect = func(model_with_removed_effect)

    model_with_removed_effect = update_initial_estimates(
        model_with_removed_effect, candidate.modelentry.modelfit_results
    )
    return ModelEntry.create(
        model=model_with_removed_effect, parent=candidate.modelentry.model, modelfit_results=None
    )


def task_results(context, p_forward: float, p_backward: float, strictness: str, state: SearchState):
    candidates = state.all_candidates_so_far
    modelentries = list(map(lambda candidate: candidate.modelentry, candidates))
    base_modelentry, *res_modelentries = modelentries
    assert base_modelentry is state.start_modelentry
    best_modelentry = state.best_candidate_so_far.modelentry
    user_input_modelentry = state.user_input_modelentry
    if user_input_modelentry != base_modelentry:
        modelentries = [user_input_modelentry] + modelentries

    res = create_results(
        COVSearchResults,
        base_modelentry,
        base_modelentry,
        res_modelentries,
        'lrt',
        (p_forward, p_backward),
        context=context,
    )

    steps = _make_df_steps(best_modelentry, candidates)
    res = replace(
        res,
        final_model=best_modelentry.model,
        steps=steps,
        candidate_summary=candidate_summary_dataframe(steps),
        ofv_summary=ofv_summary_dataframe(steps, final_included=True, iterations=True),
        summary_tool=_modify_summary_tool(res.summary_tool, steps),
        summary_models=_summarize_models(modelentries, steps),
    )

    context.store_key("final", ModelHash(best_modelentry.model))

    return res


def _modify_summary_tool(summary_tool, steps):
    step_cols_to_keep = ['step', 'pvalue', 'goal_pvalue', 'is_backward', 'selected', 'model']
    steps_df = steps.reset_index()[step_cols_to_keep].set_index(['step', 'model'])

    summary_tool_new = steps_df.join(summary_tool)
    column_to_move = summary_tool_new.pop('description')

    summary_tool_new.insert(0, 'description', column_to_move)
    return summary_tool_new.drop(['rank'], axis=1)


def _summarize_models(modelentries, steps):
    summary_models = summarize_modelfit_results_from_entries(modelentries)
    summary_models['step'] = steps.reset_index().set_index(['model'])['step']

    return summary_models.reset_index().set_index(['step', 'model'])


def _make_df_steps(best_modelentry: ModelEntry, candidates: List[Candidate]):
    best_model = best_modelentry.model
    modelentries_dict = {
        candidate.modelentry.model.name: candidate.modelentry for candidate in candidates
    }
    children_count = Counter(
        candidate.modelentry.parent.name for candidate in candidates if candidate.modelentry.parent
    )

    # Find if longest forward is also the input for the backwards search
    # otherwise add an index offset to _make_df_steps_function
    forward_candidates = [
        fc for fc in candidates if fc.steps == tuple() or isinstance(fc.steps[-1], ForwardStep)
    ]
    largest_forward_step = max([len(fc.steps) for fc in forward_candidates])
    largest_forward_candidates = [
        fc for fc in forward_candidates if len(fc.steps) == largest_forward_step
    ]
    index_offset = 0
    if not any(
        children_count[c.modelentry.model.name] >= 1 or c.modelentry.model is best_model
        for c in largest_forward_candidates
    ):
        index_offset = 1

    data = (
        _make_df_steps_row(
            modelentries_dict, children_count, best_model, candidate, index_offset=index_offset
        )
        for candidate in candidates
    )

    return pd.DataFrame.from_records(
        data,
        index=['step', 'parameter', 'covariate', 'extended_state'],
    )


def _make_df_steps_row(
    modelentries_dict: dict,
    children_count: Counter,
    best_model: Model,
    candidate: Candidate,
    index_offset=0,
):
    modelentry = candidate.modelentry
    model = modelentry.model
    parent_name = modelentry.parent.name if modelentry.parent else model.name
    parent_modelentry = modelentries_dict[parent_name]
    parent_model = parent_modelentry.model
    if candidate.steps:
        last_step = candidate.steps[-1]
        last_effect = last_step.effect
        parameter, covariate = last_effect.parameter, last_effect.covariate
        extended_state = f'{last_effect.operation} {last_effect.fp}'
        is_backward = isinstance(last_step, BackwardStep)
        if not is_backward:
            reduced_ofv = np.nan if (mfr := parent_modelentry.modelfit_results) is None else mfr.ofv
            extended_ofv = np.nan if (mfr := modelentry.modelfit_results) is None else mfr.ofv
            alpha = last_step.alpha
            extended_significant = lrt_test(
                parent_model,
                model,
                reduced_ofv,
                extended_ofv,
                alpha,
            )
        else:
            extended_ofv = (
                np.nan if (mfr := parent_modelentry.modelfit_results) is None else mfr.ofv
            )
            reduced_ofv = np.nan if (mfr := modelentry.modelfit_results) is None else mfr.ofv
            alpha = last_step.alpha
            extended_significant = lrt_test(
                model,
                parent_model,
                reduced_ofv,
                extended_ofv,
                alpha,
            )
        ofv_drop = reduced_ofv - extended_ofv
    else:
        parameter, covariate, extended_state = '', '', ''
        is_backward = False
        reduced_ofv = np.nan if (mfr := parent_modelentry.modelfit_results) is None else mfr.ofv
        extended_ofv = np.nan if (mfr := modelentry.modelfit_results) is None else mfr.ofv
        ofv_drop = reduced_ofv - extended_ofv
        alpha, extended_significant = np.nan, np.nan

    selected = children_count[model.name] >= 1 or model.name == best_model.name
    if not is_backward:
        p_value = lrt_p_value(parent_model, model, reduced_ofv, extended_ofv)
        assert not selected or (model is parent_model) or extended_significant
    else:
        p_value = lrt_p_value(model, parent_model, reduced_ofv, extended_ofv)
        assert not selected or (model is parent_model) or not extended_significant

    return {
        'step': (
            len(candidate.steps)
            if candidate.steps == tuple()
            or isinstance(candidate.steps[-1], ForwardStep)
            or isinstance(candidate.steps[-1], AdaptiveStep)
            else len(candidate.steps) + index_offset
        ),
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
def validate_input(
    search_space, p_forward, p_backward, algorithm, model, strictness, naming_index_offset
):
    if not 0 < p_forward <= 1:
        raise ValueError(
            f'Invalid `p_forward`: got `{p_forward}`, must be a float in range (0, 1].'
        )

    if not 0 < p_backward <= 1:
        raise ValueError(
            f'Invalid `p_backward`: got `{p_backward}`, must be a float in range (0, 1].'
        )

    if model is not None:
        if isinstance(search_space, str):
            try:
                statements = mfl_parse(search_space)
            except:  # noqa E722
                raise ValueError(f'Invalid `search_space`, could not be parsed: `{search_space}`')
        else:
            if not search_space.covariate:
                raise ValueError(
                    f'Invalid `search_space`, no covariate effect could be found in: `{search_space}`'
                )
            statements = search_space.covariate

        bad_statements = list(
            filter(
                lambda statement: not isinstance(statement, COVSEARCH_STATEMENT_TYPES),
                statements,
            )
        )
        if bad_statements:
            raise ValueError(
                f'Invalid `search_space`: found unknown statement of type {type(bad_statements[0]).__name__}.'
            )

        for s in statements:
            if isinstance(s, Covariate) and isinstance(s.fp, Wildcard) and not s.optional.option:
                raise ValueError(
                    f'Invalid `search_space` due to non-optional covariate'
                    f' defined with WILDCARD as effect in {s}'
                    f' Only single effect allowed for mandatory covariates'
                )

        effect_spec = spec(model, statements)

        candidate_effects = map(
            lambda x: Effect(*x[:-1]), sorted(set(parse_spec(effect_spec)))
        )  # Ignore OPTIONAL attribute

        allowed_covariates = get_covariates_allowed_in_covariate_effect(model)
        allowed_parameters = set(get_pk_parameters(model)).union(
            str(statement.symbol) for statement in model.statements.before_odes
        )
        allowed_covariate_effects = set(all_covariate_effects)
        allowed_ops = set(['*', '+'])

        for effect in candidate_effects:
            if effect.covariate not in allowed_covariates:
                raise ValueError(
                    f'Invalid `search_space` because of invalid covariate found in'
                    f' search_space: got `{effect.covariate}`,'
                    f' must be in {sorted(allowed_covariates)}.'
                )
            if effect.parameter not in allowed_parameters:
                raise ValueError(
                    f'Invalid `search_space` because of invalid parameter found in'
                    f' search_space: got `{effect.parameter}`,'
                    f' must be in {sorted(allowed_parameters)}.'
                )
            if effect.fp not in allowed_covariate_effects:
                raise ValueError(
                    f'Invalid `search_space` because of invalid effect function found in'
                    f' search_space: got `{effect.fp}`,'
                    f' must be in {sorted(allowed_covariate_effects)}.'
                )
            if effect.operation not in allowed_ops:
                raise ValueError(
                    f'Invalid `search_space` because of invalid effect operation found in'
                    f' search_space: got `{effect.operation}`,'
                    f' must be in {sorted(allowed_ops)}.'
                )
    if strictness is not None and "rse" in strictness.lower():
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                'parameter_uncertainty_method not set for model, cannot calculate relative standard errors.'
            )
    if not isinstance(naming_index_offset, int) or naming_index_offset < 0:
        raise ValueError('naming_index_offset need to be a postive (>=0) integer.')
