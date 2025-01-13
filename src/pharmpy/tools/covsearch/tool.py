from collections import Counter, defaultdict
from dataclasses import astuple, dataclass
from itertools import count
from typing import Any, Callable, Iterable, Literal, Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import get_pk_parameters, remove_covariate_effect, set_estimation_step
from pharmpy.modeling.covariate_effect import get_covariates_allowed_in_covariate_effect
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.modeling.lrt import p_value as lrt_p_value
from pharmpy.modeling.lrt import test as lrt_test
from pharmpy.tools import is_strictness_fulfilled
from pharmpy.tools.common import (
    create_plots,
    summarize_tool,
    table_final_eta_shrinkage,
    update_initial_estimates,
)
from pharmpy.tools.covsearch.samba import samba_workflow
from pharmpy.tools.mfl.feature.covariate import EffectLiteral
from pharmpy.tools.mfl.feature.covariate import features as covariate_features
from pharmpy.tools.mfl.feature.covariate import parse_spec, spec
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.mfl.statement.definition import Let
from pharmpy.tools.mfl.statement.feature.covariate import Covariate
from pharmpy.tools.mfl.statement.feature.symbols import Option, Wildcard
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import summarize_errors_from_entries, summarize_modelfit_results_from_entries
from pharmpy.tools.scm.results import candidate_summary_dataframe, ofv_summary_dataframe
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

from ..mfl.parse import ModelFeatures, get_model_features
from .results import COVSearchResults

COVSEARCH_STATEMENT_TYPES = (
    Let,
    Covariate,
)

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


def _added_effects(steps: tuple[Step, ...]) -> Iterable[Effect]:
    added_effects = defaultdict(list)
    for i, step in enumerate(steps):
        if isinstance(step, ForwardStep):
            added_effects[astuple(step.effect)].append(i)
        elif isinstance(step, BackwardStep):
            added_effects[astuple(step.effect)].pop()
        elif isinstance(step, AdaptiveStep):
            pass
        else:
            raise ValueError(f"Unknown step ({step}) added")

    pos = {effect: set(indices) for effect, indices in added_effects.items()}

    for i, step in enumerate(steps):
        if isinstance(step, ForwardStep) and i in pos[astuple(step.effect)]:
            yield step.effect


@dataclass
class Candidate:
    modelentry: ModelEntry
    steps: tuple[Step, ...]


@dataclass
class SearchState:
    user_input_modelentry: ModelEntry
    start_modelentry: ModelEntry
    best_candidate_so_far: Candidate
    all_candidates_so_far: list[Candidate]


def create_workflow(
    model: Model,
    results: ModelfitResults,
    search_space: Union[str, ModelFeatures],
    p_forward: float = 0.01,
    p_backward: float = 0.001,
    max_steps: int = -1,
    algorithm: Literal[
        'scm-forward', 'scm-forward-then-backward', 'samba', 'samba-foce'
    ] = 'scm-forward-then-backward',
    max_eval: bool = False,
    adaptive_scope_reduction: bool = False,
    strictness: str = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    naming_index_offset: Optional[int] = 0,
    nsamples: int = 10,
    _samba_max_covariates: Optional[int] = 3,
    _samba_selection_criterion: Literal['bic', 'lrt'] = 'bic',
    _samba_linreg_method: Literal['ols', 'wls', 'lme'] = 'ols',
    _samba_stepwise_lcs: Optional[bool] = None,
):
    """Run COVsearch tool. For more details, see :ref:`covsearch`.

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results of model
    search_space : str
        MFL of covariate effects to try
    p_forward : float
        The p-value to use in the likelihood ratio test for forward steps
    p_backward : float
        The p-value to use in the likelihood ratio test for backward steps
    max_steps : int
        The maximum number of search steps to make
    algorithm : {'scm-forward', 'scm-forward-then-backward', 'samba'}
        The search algorithm to use. Currently, 'scm-forward' and
        'scm-forward-then-backward' are supported.
    max_eval : bool
        Limit the number of function evaluations to 3.1 times that of the
        base model. Default is False.
    adaptive_scope_reduction : bool
        Stash all non-significant parameter-covariate effects to be tested
        after all significant effects have been tested. Once all these have been
        tested, try adding the stashed effects once more with a regular forward approach.
        Default is False
    strictness : str
        Strictness criteria
    naming_index_offset : int
        index offset for naming of runs. Default is 0.
    nsamples : int
        Number of samples from individual parameter conditional distribution for linear covariate model selection.
        Default is 10, i.e. generating 10 samples per subject
    _samba_max_covariates: int or None
        Maximum number of covariate inclusion allowed in linear covariate screening for each parameter.
    _samba_linreg_method: str
        Method used to fit linear covariate models. Currently, Ordinary Least Squares (ols),
        Weighted Least Squares (wls), and Linear Mixed-Effects (lme) are supported.
    _samba_selection_criterion: str
        Method used for linear and nonlinear model selection in SAMBA methods. Currently, BIC and LRT are
        supported.
    _samba_stepwise_lcs: bool or None
        Use stepwise linear covariate screening or not. By default, SAMBA methods use stepwise LCS whereas SCM-LCS uses
        non-stepwise LCS.

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
    >>> res = run_covsearch(model=model, results=results, search_space=search_space)      # doctest: +SKIP
    """
    if algorithm in ["samba", "samba-foce"]:
        return samba_workflow(
            model=model,
            results=results,
            search_space=search_space,
            max_steps=max_steps,
            p_forward=p_forward,
            p_backward=p_backward,
            max_eval=max_eval,
            algorithm=algorithm,
            nsamples=nsamples,
            max_covariates=_samba_max_covariates,
            selection_criterion=_samba_selection_criterion,
            linreg_method=_samba_linreg_method,
            stepwise_lcs=_samba_stepwise_lcs,
            strictness=strictness,
        )

    wb = WorkflowBuilder(name=NAME_WF)

    # FIXME : Handle when model is None
    store_task = Task("store_input_model", _store_input_model, model, results, max_eval)
    start_task = Task("create_modelentry", _start, model, results)
    wb.add_task(start_task, predecessors=store_task)
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


def _store_input_model(context, model, results, max_eval):
    context.log_info("Starting tool covsearch")
    model = model.replace(name="input", description="")
    me = ModelEntry.create(model=model, modelfit_results=results)
    context.store_input_model_entry(me)
    return max_eval


def _start(model, results, max_eval):
    if max_eval:
        max_eval_number = round(3.1 * results.function_evaluations_iterations.loc[1])
        # Change last instead of first?
        first_es = model.execution_steps[0]
        model = set_estimation_step(model, first_es.method, 0, maximum_evaluations=max_eval_number)
    return ModelEntry.create(
        model=model.replace(name="input", description=""), parent=None, modelfit_results=results
    )


def _init_search_state(
    context, search_space: Union[str, ModelFeatures], modelentry: ModelEntry
) -> tuple[dict[tuple[str], callable], SearchState]:
    model = modelentry.model
    effect_funcs, base_model = get_effect_funcs_and_base_model(search_space, model)

    if base_model != model:
        base_modelentry = ModelEntry.create(model=base_model)
        base_fit_wf = create_fit_workflow(modelentries=[base_modelentry])
        base_modelentry = context.call_workflow(base_fit_wf, 'fit_filtered_model')
    else:
        base_modelentry = modelentry
    base_candidate = Candidate(base_modelentry, ())
    search_state_init = SearchState(modelentry, base_modelentry, base_candidate, [base_candidate])

    return effect_funcs, search_state_init


def get_effect_funcs_and_base_model(search_space, model):
    ss_mfl, model_mfl = prepare_mfls(model, search_space)
    exploratory_cov_funcs = get_exploratory_covariates(ss_mfl)

    if is_model_in_search_space(model, model_mfl, ss_mfl):
        return exploratory_cov_funcs, model

    filtered_model = model.replace(name="filtered_input_model")
    # covariate effects not in search space, should be kept as it is
    covariate_to_keep = model_mfl - ss_mfl
    # covariate effects in both model and search space, should be removed for exploration in future searching steps
    covariate_to_remove = model_mfl - covariate_to_keep
    covariate_to_remove = covariate_to_remove.mfl_statement_list(["covariate"])
    description = []
    if len(covariate_to_remove) != 0:
        description.append("REMOVED")
        for cov_effect in parse_spec(spec(filtered_model, covariate_to_remove)):
            filtered_model = remove_covariate_effect(filtered_model, cov_effect[0], cov_effect[1])
            description.append('({}-{}-{})'.format(cov_effect[0], cov_effect[1], cov_effect[2]))
    # Remove all custom effects
    covariate_to_keep = covariate_to_keep.mfl_statement_list(["covariate"])
    for cov_effect in parse_spec(spec(filtered_model, covariate_to_keep)):
        if cov_effect[2].lower() == "custom":
            filtered_model = remove_covariate_effect(filtered_model, cov_effect[0], cov_effect[1])
            description.append('({}-{}-{})'.format(cov_effect[0], cov_effect[1], cov_effect[2]))

    filtered_model = filtered_model.replace(description=';'.join(description))

    # Add structural covariates in search space if any
    structural_cov = tuple([c for c in ss_mfl.covariate if not c.optional.option])
    structural_cov_funcs = all_funcs(Model(), structural_cov)
    if len(structural_cov_funcs) != 0:
        description.append("ADDED")
        for cov_effect, cov_func in structural_cov_funcs.items():
            filtered_model = cov_func(filtered_model)
            description.append('({}-{}-{})'.format(cov_effect[0], cov_effect[1], cov_effect[2]))
        filtered_model = filtered_model.replace(description=";".join(description))

    return (exploratory_cov_funcs, filtered_model)


def prepare_mfls(model, search_space):
    if isinstance(search_space, str):
        search_space = ModelFeatures.create_from_mfl_string(search_space)
    ss_mfl = search_space.expand(model)  # Expand to remove LET/REF
    model_mfl = ModelFeatures.create_from_mfl_string(get_model_features(model))

    ss_mfl = ModelFeatures.create_from_mfl_statement_list(ss_mfl.mfl_statement_list(["covariate"]))
    model_mfl = ModelFeatures.create_from_mfl_statement_list(
        model_mfl.mfl_statement_list(["covariate"])
    )

    return ss_mfl, model_mfl


def get_exploratory_covariates(ss_mfl):
    exploratory_cov = tuple(c for c in ss_mfl.covariate if c.optional.option)
    cov_funcs = all_funcs(Model(), exploratory_cov)
    exploratory_cov_funcs = dict()
    for cov_effect, cov_func in cov_funcs.items():
        if cov_effect[-1] == "ADD":
            effect = cov_effect[1:-1]  # Everything except "ADD", e.g. ('CL', 'WT', 'exp', '*')
            exploratory_cov_funcs[effect] = cov_func
    # Sort by effect
    exploratory_cov_funcs = dict(sorted(exploratory_cov_funcs.items()))
    return exploratory_cov_funcs


def is_model_in_search_space(model, model_mfl, cov_mfl):
    def _is_optional(cov):
        return cov.optional == Option(True)

    cov_struct = [cov for cov in cov_mfl.covariate if not _is_optional(cov)]
    cov_struct_mfl = ModelFeatures.create_from_mfl_statement_list(cov_struct)

    # Check if all obligatory covariates are in model
    if not model_mfl.contain_subset(cov_struct_mfl, model=model):
        return False
    elif model_mfl.covariate:
        # Check if all covariates in model are in original search space
        if not cov_mfl.contain_subset(model_mfl, model=model):
            return False
        # FIXME: workaround, check if model is simplest model in search space
        cov_exploratory = cov_mfl - cov_struct_mfl
        if cov_exploratory.contain_subset(model_mfl, model=model):
            return False
    return True


def task_greedy_forward_search(
    context,
    p_forward: float,
    max_steps: int,
    naming_index_offset: int,
    strictness: str,
    adaptive_scope_reduction: bool,
    state_and_effect: tuple[SearchState, dict],
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
        new_candidate_modelentries = context.call_workflow(wf, f'{NAME_WF}-effects_addition-{step}')
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
    strictness: str,
    state: SearchState,
) -> SearchState:
    def handle_effects(
        step: int,
        parent: Candidate,
        candidate_effect_funcs: list[EffectLiteral],
        index_offset: int,
    ):
        index_offset = index_offset + naming_index_offset
        wf = wf_effects_removal(parent, candidate_effect_funcs, index_offset)
        new_candidate_modelentries = context.call_workflow(wf, f'{NAME_WF}-effects_removal-{step}')

        return [
            Candidate(modelentry, parent.steps + (BackwardStep(p_backward, RemoveEffect(*effect)),))
            for modelentry, effect in zip(new_candidate_modelentries, candidate_effect_funcs.keys())
        ]

    # TODO : When only backwards search is supported, use get_model_features to extract removeable effects.
    optional_effects = list(map(astuple, _added_effects(state.best_candidate_so_far.steps)))

    def _extract_sublist(lst, n, iterable=False):
        if iterable:
            return [(item[n],) for item in lst]
        else:
            return [item[n] for item in lst]

    candidate_effect_funcs = dict(
        covariate_features(
            state.best_candidate_so_far.modelentry.model,
            tuple(
                map(
                    Covariate,
                    _extract_sublist(optional_effects, 0, True),
                    _extract_sublist(optional_effects, 1, True),
                    _extract_sublist(optional_effects, 2, True),
                    _extract_sublist(optional_effects, 3),
                )
            ),
            remove=True,
        )
    )
    candidate_effect_funcs = {k[1:-1]: v for k, v in candidate_effect_funcs.items()}

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
    handle_effects: Callable[[int, Candidate, list[EffectLiteral], int], list[Candidate]],
    candidate_effect_funcs: dict,
    alpha: float,
    max_steps: int,
    strictness: str,
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
        # NOTE: We assume parent_modelentry.modelfit_results is not None
        parent_modelentry = best_candidate_so_far.modelentry
        assert parent_modelentry.modelfit_results is not None

        best_candidate_so_far = get_best_candidate_so_far(
            parent_modelentry, new_candidate_modelentries, all_candidates_so_far, strictness, alpha
        )

        if best_candidate_so_far.modelentry is parent_modelentry:
            break

        # TODO : Find all non-significant models and stash the most recently added effect
        if adaptive_scope_reduction:
            last_step = best_candidate_so_far.steps[-1]
            is_backward = isinstance(last_step, BackwardStep)
            if not is_backward:
                nonsignificant_effects_step = extract_nonsignificant_effects(
                    parent_modelentry, new_candidates, candidate_effect_funcs, alpha
                )
                nonsignificant_effects.update(nonsignificant_effects_step)

        # NOTE: Filter out incompatible effects
        last_step_effect = best_candidate_so_far.steps[-1].effect
        candidate_effect_funcs = filter_effects(
            candidate_effect_funcs, last_step_effect, nonsignificant_effects
        )

    return nonsignificant_effects, all_candidates_so_far, best_candidate_so_far


def get_best_candidate_so_far(
    parent_modelentry, new_candidate_modelentries, all_candidates_so_far, strictness, alpha
):
    ofvs = [
        (
            np.nan
            if modelentry.modelfit_results is None
            or not is_strictness_fulfilled(
                modelentry.model, modelentry.modelfit_results, strictness
            )
            else modelentry.modelfit_results.ofv
        )
        for modelentry in new_candidate_modelentries
    ]
    best_model_so_far = lrt_best_of_many(
        parent_modelentry,
        new_candidate_modelentries,
        parent_modelentry.modelfit_results.ofv,
        ofvs,
        alpha,
    )

    best_candidate_so_far = next(
        filter(lambda candidate: candidate.modelentry is best_model_so_far, all_candidates_so_far)
    )

    return best_candidate_so_far


def filter_effects(effect_funcs, last_step_effect, nonsignificant_effects):
    candidate_effect_funcs = {
        effect_description: effect_func
        for effect_description, effect_func in effect_funcs.items()
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

    return candidate_effect_funcs


def extract_nonsignificant_effects(parent_modelentry, new_candidates, effect_funcs, alpha):
    nonsignificant_effects = {}
    for new_cand in new_candidates:
        p_value = lrt_p_value(
            parent_modelentry.model,
            new_cand.modelentry.model,
            parent_modelentry.modelfit_results.ofv,
            new_cand.modelentry.modelfit_results.ofv,
        )
        if alpha <= p_value:
            last_step_effect = new_cand.steps[-1].effect
            key = (
                last_step_effect.parameter,
                last_step_effect.covariate,
                last_step_effect.fp,
                last_step_effect.operation,
            )
            nonsignificant_effects[key] = effect_funcs[key]
    return nonsignificant_effects


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


def _create_description(effect_new: dict, steps_prev: tuple[Step, ...], forward: bool = True):
    # Will create this type of description: '(CL-AGE-exp);(MAT-AGE-exp);(MAT-AGE-exp-+)'
    def _create_effect_str(effect):
        if isinstance(effect, tuple):
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
    model_with_removed_effect = model.replace(name=name, description=description)

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
    tables = create_result_tables(
        candidates,
        best_modelentry,
        user_input_modelentry,
        base_modelentry,
        res_modelentries,
        (p_forward, p_backward),
        strictness,
    )
    plots = create_plots(best_modelentry.model, best_modelentry.modelfit_results)

    res = COVSearchResults(
        final_model=best_modelentry.model,
        final_results=best_modelentry.modelfit_results,
        summary_models=tables['summary_models'],
        summary_tool=tables['summary_tool'],
        summary_errors=tables['summary_errors'],
        steps=tables['steps'],
        ofv_summary=tables['ofv_summary'],
        candidate_summary=tables['candidate_summary'],
        final_model_dv_vs_ipred_plot=plots['dv_vs_ipred'],
        final_model_dv_vs_pred_plot=plots['dv_vs_pred'],
        final_model_cwres_vs_idv_plot=plots['cwres_vs_idv'],
        final_model_abs_cwres_vs_ipred_plot=plots['abs_cwres_vs_ipred'],
        final_model_eta_distribution_plot=plots['eta_distribution'],
        final_model_eta_shrinkage=table_final_eta_shrinkage(
            best_modelentry.model, best_modelentry.modelfit_results
        ),
    )

    context.store_final_model_entry(best_modelentry)
    context.log_info("Finishing tool covsearch")
    return res


def create_result_tables(
    candidates,
    best_modelentry,
    input_modelentry,
    base_modelentry,
    res_modelentries,
    cutoff,
    strictness,
):
    steps = _make_df_steps(best_modelentry, candidates)
    model_entries = [base_modelentry] + res_modelentries
    if input_modelentry != base_modelentry:
        model_entries.insert(0, input_modelentry)
    sum_models = _summarize_models(model_entries, steps)
    sum_tool = summarize_tool(
        res_modelentries,
        base_modelentry,
        rank_type='lrt',
        cutoff=cutoff,
        strictness=strictness,
    )
    sum_tool = _modify_summary_tool(sum_tool, steps)
    sum_errors = summarize_errors_from_entries(model_entries)
    ofv_summary = ofv_summary_dataframe(steps, final_included=True, iterations=True)
    sum_cand = candidate_summary_dataframe(steps)
    tables = {
        'summary_tool': sum_tool,
        'summary_models': sum_models,
        'summary_errors': sum_errors,
        'steps': steps,
        'ofv_summary': ofv_summary,
        'candidate_summary': sum_cand,
    }
    return tables


def _create_proxy_model_table(candidates, steps, proxy_models):
    step_cols_to_keep = ['step', 'pvalue', 'model']
    steps_df = steps.reset_index()[step_cols_to_keep].set_index(['step', 'model'])

    steps_df = steps_df.reset_index()
    steps_df = steps_df[steps_df['model'].isin(proxy_models)]
    steps_df = steps_df.set_index(['step', 'model'])

    return steps_df


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


def _make_df_steps(best_modelentry: ModelEntry, candidates: list[Candidate]):
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
            if effect.fp == "custom":
                raise ValueError(
                    f'Invalid `search_space` because of invalid effect function found in'
                    f' search_space: `{effect.fp}` is not a supported type.'
                )
            if effect.operation not in allowed_ops:
                raise ValueError(
                    f'Invalid `search_space` because of invalid effect operation found in'
                    f' search_space: got `{effect.operation}`,'
                    f' must be in {sorted(allowed_ops)}.'
                )
    if "rse" in strictness.lower():
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                'parameter_uncertainty_method not set for model, cannot calculate relative standard errors.'
            )
    if not isinstance(naming_index_offset, int) or naming_index_offset < 0:
        raise ValueError('naming_index_offset need to be a postive (>=0) integer.')
