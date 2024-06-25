from collections import Counter, defaultdict
from dataclasses import astuple, dataclass, replace
from itertools import count
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

from pharmpy.basic.expr import Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import (
    Assignment,
    EstimationStep,
    ExecutionSteps,
    Model,
    NormalDistribution,
    Parameter,
    Parameters,
    RandomVariables,
    Statements,
)
from pharmpy.modeling import (
    convert_model,
    get_parameter_rv,
    get_pk_parameters,
    get_sigmas,
    has_covariate_effect,
    mu_reference_model,
    remove_covariate_effect,
    set_estimation_step,
)
from pharmpy.modeling.covariate_effect import (
    add_covariate_effect,
    get_covariates_allowed_in_covariate_effect,
)
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.modeling.lrt import p_value as lrt_p_value
from pharmpy.modeling.lrt import test as lrt_test
from pharmpy.tools import is_strictness_fulfilled
from pharmpy.tools.common import create_results, update_initial_estimates
from pharmpy.tools.mfl.feature.covariate import EffectLiteral
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


class SambaStep(Step):
    pass


def _added_effects(steps: Tuple[Step, ...]) -> Iterable[Effect]:
    added_effects = defaultdict(list)
    for i, step in enumerate(steps):
        if isinstance(step, ForwardStep):
            added_effects[astuple(step.effect)].append(i)
        elif isinstance(step, BackwardStep):
            added_effects[astuple(step.effect)].pop()
        elif isinstance(step, (AdaptiveStep, SambaStep)):
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


ALGORITHMS = ('scm-forward', 'scm-forward-then-backward', 'SAMBA')


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
    algorithm : {'scm-forward', 'scm-forward-then-backward', 'SAMBA'}
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
    store_task = Task("store_input_model", _store_input_model, model, results, max_eval)
    start_task = Task("create_modelentry", _start, model, results)
    wb.add_task(start_task, predecessors=store_task)
    init_task = Task("init", _init_search_state, search_space, algorithm)
    wb.add_task(init_task, predecessors=start_task)

    if algorithm != "SAMBA":
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
    else:
        forward_samba_task = Task(
            'forward-samba',
            samba_search,
            max_steps,
            p_forward,
            strictness,
            adaptive_scope_reduction,
        )
        # FIXME : Adaptive scope reduction?
        wb.add_task(forward_samba_task, predecessors=init_task)
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
        context, search_space: str, algorithm: str, modelentry: ModelEntry
) -> SearchState:
    model = modelentry.model
    effect_funcs, filtered_model = filter_search_space_and_model(search_space, model)
    if algorithm == "SAMBA":
        # Add IIV to all individual parameters?
        # FIXME : Add IIV to all parameters given in the covariate search space if not already there

        filtered_model = filtered_model.replace(name="Base_SAMBA_model")

        # Switch to SAEM (and hence mean posterior etas)
        filtered_model = set_estimation_step(
            filtered_model,
            "SAEM",
            maximum_evaluations=9999,  # Fix ValueError: MAXEVAL already set as attribute in estimation method object
            tool_options={'NITER': 1000, 'AUTO': 1, 'PHITYPE': 1},
        )

        # Mu-reference model
        filtered_model = mu_reference_model(filtered_model)

    if filtered_model != model:
        filtered_modelentry = ModelEntry.create(model=filtered_model)
        filtered_fit_wf = create_fit_workflow(modelentries=[filtered_modelentry])
        filtered_modelentry = call_workflow(filtered_fit_wf, 'fit_filtered_model', context)
    else:
        filtered_modelentry = modelentry
    candidate = Candidate(filtered_modelentry, ())
    return (effect_funcs, SearchState(modelentry, filtered_modelentry, candidate, [candidate]))


def filter_search_space_and_model(search_space, model):
    filtered_model = model.replace(name="filtered_input_model")
    if isinstance(search_space, str):
        search_space = ModelFeatures.create_from_mfl_string(search_space)
    ss_mfl = search_space.expand(filtered_model)  # Expand to remove LET/REF

    # Clean up all covariate effect in model
    model_mfl = ModelFeatures.create_from_mfl_string(get_model_features(filtered_model))
    covariate_to_remove = model_mfl.mfl_statement_list(["covariate"])
    description = []
    if len(covariate_to_remove) != 0:
        description.append("REMOVED")
        for cov_effect in parse_spec(spec(filtered_model, covariate_to_remove)):
            filtered_model = remove_covariate_effect(
                filtered_model, cov_effect[0], cov_effect[1]
            )
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

    # Exploratory covariates
    exploratory_cov = tuple(c for c in ss_mfl.covariate if c.optional.option)
    exploratory_cov_funcs = all_funcs(Model(), exploratory_cov)
    exploratory_cov_funcs = {cov_effect[1:-1]: cov_func
                             for cov_effect, cov_func in exploratory_cov_funcs.items()
                             if cov_effect[-1] == "ADD"}

    return (exploratory_cov_funcs, filtered_model)


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
                    modelentry.model, modelentry.modelfit_results, strictness
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
    model_with_removed_effect = model.replace(name=name, description=description)

    func = effect[1]
    model_with_removed_effect = func(model_with_removed_effect)

    model_with_removed_effect = update_initial_estimates(
        model_with_removed_effect, candidate.modelentry.modelfit_results
    )
    return ModelEntry.create(
        model=model_with_removed_effect, parent=candidate.modelentry.model, modelfit_results=None
    )


def samba_search(
        context,
        max_steps,
        p_forward,
        strictness,
        adaptive_scope_reduction,
        state_and_effect: Tuple[SearchState, dict],
):
    for temp in state_and_effect:
        if isinstance(temp, SearchState):
            state = temp
        else:
            candidate_effect_funcs = temp

    param_effect_funcs = {}
    param_cov = {}  # FIXME : Extract from param_effect_funcs instead as needed
    for key, func in candidate_effect_funcs.items():
        param = key[0]
        if param in param_effect_funcs.keys():
            param_effect_funcs[param].update({key: func})
            param_cov[param].append(key[1])
        else:
            param_effect_funcs[param] = {key: func}
            param_cov[param] = [key[1]]

    steps = range(1, max_steps + 1) if max_steps >= 0 else count(1)

    best_candidate = state.best_candidate_so_far
    all_nonsignificant_effects = {}
    all_proxy_models = {}
    for current_step in steps:
        for param, funcs in param_effect_funcs.items():
            base_proxy_model = _create_samba_proxy_model(
                best_candidate.modelentry, param, param_cov[param], current_step
            )
            # Run base proxy model
            base_proxy_modelentry = ModelEntry.create(model=base_proxy_model)
            base_proxy_fit_wf = create_fit_workflow(modelentries=[base_proxy_modelentry])
            base_proxy_modelentry = call_workflow(base_proxy_fit_wf, 'fit_base_proxy', context)

            base_candidate = Candidate(
                base_proxy_modelentry,
                best_candidate.steps + (SambaStep(p_forward, DummyEffect("", "", "", "")),),
            )
            state.all_candidates_so_far.extend([base_candidate])

            def handle_samba_effects(step, parent, candidate_effect_funcs, index_offset):
                # index_offset = index_offset + naming_index_offset
                wf = wf_effects_addition(
                    parent.modelentry, parent, candidate_effect_funcs, index_offset
                )
                new_candidate_modelentries = call_workflow(
                    wf, f'{NAME_WF}-effects_addition-{step}', context
                )
                return [
                    Candidate(
                        modelentry, parent.steps + (ForwardStep(p_forward, AddEffect(*effect)),)
                    )
                    for modelentry, effect in zip(
                        new_candidate_modelentries, candidate_effect_funcs.keys()
                    )
                ]

            no_candidates_so_far = len(state.all_candidates_so_far)
            nonsignificant_effects, all_candidates_so_far, best_candidate_so_far = (
                perform_step_procedure(
                    [current_step],
                    funcs,
                    handle_samba_effects,
                    state.all_candidates_so_far,
                    base_candidate,
                    strictness,
                    p_forward,
                    adaptive_scope_reduction,
                )
            )

            new_proxy_candidates = [base_candidate] + all_candidates_so_far[
                                                      no_candidates_so_far:
                                                      ]  # Ignore previous candidates
            all_candidates_so_far = (
                    all_candidates_so_far[:no_candidates_so_far]
                    + [base_candidate]
                    + all_candidates_so_far[no_candidates_so_far:]
            )  # ONÖDIGT
            all_nonsignificant_effects.update(nonsignificant_effects)
            all_proxy_models[current_step] = new_proxy_candidates

        # Determine the best proxy model
        # Should be the best_candidate_so_far -> ONÖDIGT
        proxy_models_of_this_step = all_proxy_models[current_step]
        first_proxy_model_entry = proxy_models_of_this_step[0].modelentry
        rest_of_proxy_models = [m.modelentry for m in proxy_models_of_this_step[1:]]
        rest_of_proxy_models_ofv = [
            m.modelentry.modelfit_results.ofv for m in proxy_models_of_this_step[1:]
        ]
        best_proxy_model = lrt_best_of_many(
            first_proxy_model_entry,
            rest_of_proxy_models,
            first_proxy_model_entry.modelfit_results.ofv,
            rest_of_proxy_models_ofv,
            p_forward,
        )
        best_proxy_candidate_so_far = next(
            filter(lambda candidate: candidate.modelentry is best_proxy_model, new_proxy_candidates)
        )

        # Check that this is NOT the same as the base model.
        # Should break if it is

        # Add to base model and estimate
        latest_added_effect = best_proxy_candidate_so_far.steps[-1].effect
        steps_wo_samba = [
            s for s in best_proxy_candidate_so_far.steps if not isinstance(s, SambaStep)
        ]
        candidate_best_model = best_candidate.modelentry.model.replace(
            name=f"best_{current_step}",
            description=_create_description(steps_wo_samba[-1].effect, steps_wo_samba[:-1]),
        )
        candidate_best_model = add_covariate_effect(
            candidate_best_model,
            latest_added_effect.parameter,
            latest_added_effect.covariate,
            latest_added_effect.fp,
            latest_added_effect.operation,
        )
        candidate_best_model_entry = ModelEntry.create(
            model=candidate_best_model, parent=best_candidate.modelentry.model
        )
        candidate_fit_wf = create_fit_workflow(modelentries=[candidate_best_model_entry])
        candidate_best_model_entry = call_workflow(
            candidate_fit_wf, 'fit_candidate_best_model', context
        )

        last_step = best_proxy_candidate_so_far.steps[-1]
        state.all_candidates_so_far.extend(
            [Candidate(candidate_best_model_entry, best_candidate.steps + (last_step,))]
        )

        if not lrt_test(
                best_candidate.modelentry.model,
                candidate_best_model_entry.model,
                best_candidate.modelentry.modelfit_results.ofv,
                candidate_best_model_entry.modelfit_results.ofv,
                p_forward,
        ):
            break

        # Create new best_candidate
        last_step = best_proxy_candidate_so_far.steps[-1]
        best_candidate = Candidate(candidate_best_model_entry, best_candidate.steps + (last_step,))

        # Change the state to match the current iteration
        state = SearchState(
            state.user_input_modelentry,
            state.start_modelentry,
            best_candidate,
            state.all_candidates_so_far,
        )

    # RETURN THE CORRECT THING
    return state


def _create_samba_proxy_model(modelentry, param, covariates, step):
    dataset = _create_samba_dataset(modelentry, param, covariates)

    model = modelentry.model

    theta = Parameter('theta', 0.1)  # FIXME : Something else?
    sigma = get_sigmas(model)[0]
    params = Parameters((theta, sigma))

    sigma_name = 'epsilon'
    sigma = NormalDistribution.create(sigma_name, 'ruv', 0, sigma.symbol)
    rvs = RandomVariables.create([sigma])

    base = Assignment.create(Expr.symbol(param), theta.symbol)
    ipred = Assignment.create(Expr.symbol("IPRED"), Expr.symbol(param))
    y = Assignment.create(Expr.symbol('Y'), Expr.symbol("IPRED") + Expr.symbol(sigma_name))
    statements = Statements([base, ipred, y])

    name = f'samba_{step}_{param}'

    est = EstimationStep.create('foce')

    base_model = Model.create(
        parameters=params,
        random_variables=rvs,
        statements=statements,
        name=name,
        description=name,
        execution_steps=ExecutionSteps.create([est]),
        dependent_variables={y.symbol: 1},
    )

    base_model = base_model.replace(dataset=dataset)

    di = base_model.datainfo
    di = di.set_dv_column("DV")
    di = di.set_id_column("ID")

    base_model = base_model.replace(datainfo=di)

    base_model = convert_model(base_model, "nonmem")

    return base_model


def _create_samba_dataset(model_entry, param, covariates):
    # Store ETA as DV
    # Store all covariates (that are needed)

    eta_name = get_parameter_rv(model_entry.model, param)[0]

    eta_column = model_entry.modelfit_results.individual_estimates[eta_name]
    eta_column = eta_column.rename("DV")
    covariate_columns = model_entry.model.dataset[["ID"] + covariates]

    dataset = covariate_columns.join(eta_column, "ID")

    return dataset


def task_results(context, p_forward: float, p_backward: float, strictness: str, state: SearchState):
    candidates = state.all_candidates_so_far
    modelentries = list(map(lambda candidate: candidate.modelentry, candidates))
    base_modelentry, *res_modelentries = modelentries
    assert base_modelentry is state.start_modelentry
    best_modelentry = state.best_candidate_so_far.modelentry
    user_input_modelentry = state.user_input_modelentry
    if user_input_modelentry != base_modelentry:
        modelentries = [user_input_modelentry] + modelentries

    steps = _make_df_steps(best_modelentry, candidates)

    # Drop all proxy models from SAMBA algorithm
    to_be_dropped = []
    for c in candidates:
        if any(isinstance(s, SambaStep) for s in c.steps):
            to_be_dropped.append(c.modelentry.model.name)

    # Create a new table with all the proxy models from each step.
    # if to_be_dropped:
    #     proxy_model_table = _create_proxy_model_table(candidates, steps, to_be_dropped)
    # FIXME : Add proxy_model_table to the results object.

    res = create_results(
        COVSearchResults,
        base_modelentry,
        base_modelentry,
        res_modelentries,
        'lrt',
        (p_forward, p_backward),
        context=context,
    )

    res = replace(
        res,
        final_model=best_modelentry.model,
        steps=steps,
        candidate_summary=candidate_summary_dataframe(steps),
        ofv_summary=ofv_summary_dataframe(steps, final_included=True, iterations=True),
        summary_tool=_modify_summary_tool(res.summary_tool, steps, to_be_dropped),
        summary_models=_summarize_models(modelentries, steps, to_be_dropped),
    )

    context.store_final_model_entry(best_modelentry)
    return res


def _create_proxy_model_table(candidates, steps, proxy_models):
    step_cols_to_keep = ['step', 'pvalue', 'model']
    steps_df = steps.reset_index()[step_cols_to_keep].set_index(['step', 'model'])

    steps_df = steps_df.reset_index()
    steps_df = steps_df[steps_df['model'].isin(proxy_models)]
    steps_df = steps_df.set_index(['step', 'model'])

    return steps_df


def _modify_summary_tool(summary_tool, steps, to_be_dropped):
    step_cols_to_keep = ['step', 'pvalue', 'goal_pvalue', 'is_backward', 'selected', 'model']
    steps_df = steps.reset_index()[step_cols_to_keep].set_index(['step', 'model'])

    summary_tool_new = steps_df.join(summary_tool)
    column_to_move = summary_tool_new.pop('description')

    summary_tool_new.insert(0, 'description', column_to_move)
    if to_be_dropped:
        summary_tool_new = summary_tool_new.reset_index()
        summary_tool_new = summary_tool_new[~summary_tool_new['model'].isin(to_be_dropped)]
        summary_tool_new = summary_tool_new.set_index(['step', 'model'])

    return summary_tool_new.drop(['rank'], axis=1)


def _summarize_models(modelentries, steps, to_be_dropped):
    summary_models = summarize_modelfit_results_from_entries(modelentries)
    summary_models['step'] = steps.reset_index().set_index(['model'])['step']

    if to_be_dropped:
        summary_models = summary_models.reset_index()
        summary_models = summary_models[~summary_models['model'].isin(to_be_dropped)]
        summary_models = summary_models.set_index(['step', 'model'])

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
               or isinstance(candidate.steps[-1], SambaStep)
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
    if strictness is not None and "rse" in strictness.lower():
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            raise ValueError(
                'parameter_uncertainty_method not set for model, cannot calculate relative standard errors.'
            )
    if not isinstance(naming_index_offset, int) or naming_index_offset < 0:
        raise ValueError('naming_index_offset need to be a postive (>=0) integer.')
