# isolate the samba algorithm from tool.py
from functools import partial
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
    get_covariate_effects,
    load_example_model,
    convert_model,
    get_parameter_rv,
    get_pk_parameters,
    get_sigmas,
    has_covariate_effect,
    has_mu_reference,
    mu_reference_model,
    remove_covariate_effect,
    add_estimation_step,
    set_estimation_step,
    remove_estimation_step,
    unconstrain_parameters,
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

from pharmpy.tools.mfl.filter import COVSEARCH_STATEMENT_TYPES
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.covsearch.results import COVSearchResults

NAME_WF = 'samba_covsearch'

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


def samba_workflow(
        search_space: Union[str, ModelFeatures],
        p_forward: float = 0.01,
        max_steps: int = -1,
        results: Optional[ModelfitResults] = None,
        model: Optional[Model] = None,
        max_eval: bool = False,
        naming_index_offset: Optional[int] = 0,
):
    """
    Workflow builder for SAMBA covariate search algorithm.
    """

    wb = WorkflowBuilder(name=NAME_WF)

    # Initiate model and search state
    store_task = Task("store_input_model", _store_input_model, model, results)
    wb.add_task(store_task)

    init_task = Task("init", _init_search_state, p_forward, search_space)
    wb.add_task(init_task, predecessors=store_task)

    # SAMBA search tasks
    samba_search_task = Task("samba_search", samba_search,
                             max_steps,
                             )
    wb.add_task(samba_search_task, predecessors=init_task)
    search_output = wb.output_tasks

    # # Results tasks
    # results_task = Task("results", task_results,
    #                     p_forward,
    #                     # p_backward, strictness,
    #                     )
    # wb.add_task(results_task, predecessors=search_output)

    return Workflow(wb)


# Commonly used
# Tested: it returns the same as separately running _store_input_model and _start
# thus, it is safe to modify the samba workflow
def _store_input_model(context, model, results):
    # store the input model
    model = model.replace(name="input_model", description="")
    input_me = ModelEntry.create(model=model, modelfit_results=results)
    context.store_input_model_entry(input_me)

    return input_me


# TESTING MODE
def _init_search_state(
        # context,
        alpha: float,
        search_space: str,
        modelentry: ModelEntry
):
    model = modelentry.model
    exploratory_cov_funcs, linear_cov_funcs, filtered_model = filter_search_space_and_model(search_space,
                                                                                            model)  # exploratory effect functions, filtered model (cleaned cov effects)

    # nonlinear search state
    filtered_model = filtered_model.replace(name="samba_start_model")
    filtered_model = mu_reference_model(filtered_model)
    filtered_model = remove_estimation_step(filtered_model, idx=0)
    filtered_model = add_estimation_step(filtered_model, method="SAEM", idx=0,
                                         tool_options={'NITER': 1000, 'AUTO': 1, 'PHITYPE': 1},
                                         )

    # TESTING STARTS =======================
    filtered_model_result = fit(filtered_model)
    filtered_modelentry = ModelEntry.create(model=filtered_model, modelfit_results=filtered_model_result)
    # TESTING ENDS  ========================
    # REPLACE TEST CHUNK ===================
    # filtered_modelentry = ModelEntry.create(model=filtered_model)
    # filtered_fit_wf = create_fit_workflow(modelentries=[filtered_modelentry])
    # filtered_modelentry = call_workflow(filtered_fit_wf, 'fit_filtered_model', context)
    nonlinear_model_candidate = Candidate(filtered_modelentry, ())
    nonlinear_search_state = SearchState(modelentry, filtered_modelentry, nonlinear_model_candidate,
                                         [nonlinear_model_candidate])

    # linear_modelentries dict {param: [modelentries]}
    param_indexed_funcs = {}  # container for cov_effect : cov_func pairs indexed by parameters {param: {cov_effect: cov_func}}
    param_cov_list = {}  # {param : [covariates]}
    for cov_effect, cov_func in linear_cov_funcs.items():
        param = cov_effect[0]
        if param not in param_indexed_funcs.keys():
            param_indexed_funcs[param] = {cov_effect: cov_func}
            param_cov_list[param] = [cov_effect[1]]
        else:
            param_indexed_funcs[param].update({cov_effect: cov_func})
            param_cov_list[param].append(cov_effect[1])

    # create param_base_model
    proxy_model_dict = dict.fromkeys(param_cov_list.keys(), None)
    for param, covariates in param_cov_list.items():
        param_base_model = _create_samba_base_model(filtered_modelentry, param, covariates)
        param_base_modelentry = ModelEntry.create(
            model=param_base_model)  # update dataset and fit all linear models in samba step
        proxy_model_dict[param] = [param_base_modelentry]

        # create linear covariate models for each parameter ("lin", "+")
        for cov_effect, linear_func in param_indexed_funcs[param].items():
            param_cov_model = linear_func(model=param_base_model)
            name = "-".join(cov_effect[0:3])
            param_cov_model = param_cov_model.replace(name=name)
            param_cov_modelentry = ModelEntry.create(model=param_cov_model)
            proxy_model_dict[param].append(param_cov_modelentry)

    return (nonlinear_search_state, proxy_model_dict, exploratory_cov_funcs, param_cov_list)


# init linear search state first, then use model = model.replace(dataset) to update the linear search state
def samba_step(state_and_funcs):
    nonlinear_search_state, proxy_model_dict, exploratory_cov_funcs, param_cov_list = state_and_effect
    best_nlme_modelentry = nonlinear_search_state.best_candidate_so_far.modelentry
    print(best_nlme_modelentry.modelfit_results.ofv)
    # update dataset (etas) for all linear covariate candidate models
    for param, covariates in param_cov_list.items():
        # update dataset
        updated_dataset = _create_samba_dataset(best_nlme_modelentry, param, covariates)
        # update linear covariate models
        proxy_model_dict[param] = [ModelEntry.create(model=modelentry.model.replace(dataset=updated_dataset))
                                   for modelentry in proxy_model_dict[param]]
    selected_explor_cov_funcs = []
    # fit linear covariate models
    # FIXME: MINIMIZATION TERMINATED with potentially wrong OVF values
    # TODO: Transform to Task() and Workflow()
    for param, linear_modelentries in proxy_model_dict.items():
        # MANUAL RUN CHUNK STARTS =======
        linear_modelentries = [modelentry.attach_results(fit(modelentry.model))
                               for modelentry in linear_modelentries]
        # MANUAL RUN CHUNK ENDS   =======
        ###### Task() and Workflow() STARTS
        # linear_fit_wf = create_fit_workflow(linear_modelentries)
        # linear_modelentries = call_workflow(linear_fit_wf, 'fit_linear_models', context)
        ###### Task() and Workflow() ENDS
        proxy_model_dict[param] = linear_modelentries

        # covariate model selection: best of many

        # for param, linear_modelentries in proxy_model_dict.items():
        ofvs = [modelentry.modelfit_results.ofv
                if modelentry.modelfit_results is not None
                else np.nan
                for modelentry in linear_modelentries]
        selected_cov = lrt_best_of_many(parent=linear_modelentries[0], models=linear_modelentries[1:],
                                        parent_ofv=ofvs[0], model_ofvs=ofvs[1:], alpha=0.5)
        cov_model_index = selected_cov.model.name
        print("Selected Covariate Effect: ", cov_model_index)
        if "base" not in cov_model_index:
            selected_explor_cov_funcs.append(exploratory_cov_funcs[cov_model_index])
        else:
            selected_explor_cov_funcs.append(None)

    # nonlinear mixed effect model selection: best of two
    if any(selected_explor_cov_funcs):
        # TODO: Transform to Task() and Workflow()
        selected_explor_cov_funcs = [f for f in selected_explor_cov_funcs if f is not None]
        new_nonlin_models = list(map(lambda cov_func: cov_func(best_nlme_modelentry.model), selected_explor_cov_funcs))
        new_model_results = list(map(lambda model: fit(model), new_nonlin_models))
        ofvs = [results.ofv for results in new_model_results]
        new_nonlin_modelentries = [ModelEntry.create(model=model, modelfit_results=result) for model, result in
                                   zip(new_nonlin_models, new_model_results)]
        # Task() and Workflow() STARTS =======
        # new_nonlin_modelentries = list(map(lambda model: ModelEntry.create(model), new_nonlin_models))
        # nonlin_fit_wf = create_fit_workflow(new_nonlin_modelentries)
        # new_nonlin_modelentries = call_workflow(nonlin_fit_wf, 'fit_nonlinear_models', context)
        # ofvs = [modelentry.modelfit_results.ofv 
        #         if modelentry.modelfit_results is not None
        #         else np.nan
        #         for modelentry in new_nonlin_modelentries]
        # Task() and Workflow() ENDS   =======

        best_nlme_modelentry = lrt_best_of_many(parent=best_nlme_modelentry,
                                                models=new_nonlin_modelentries,
                                                parent_ofv=best_nlme_modelentry.modelfit_results.ofv,
                                                model_ofvs=ofvs,
                                                alpha=0.05)
        print(best_nlme_modelentry.modelfit_results.ofv)

    # update search states
    best_candidate_so_far = Candidate(best_nlme_modelentry, ())
    nonlinear_search_state = replace(nonlinear_search_state,
                                     best_candidate_so_far=best_candidate_so_far)
    nonlinear_search_state.all_candidates_so_far.extend([best_candidate_so_far])

    return (nonlinear_search_state, proxy_model_dict, exploratory_cov_funcs, param_cov_list)


def samba_search(max_steps, state_and_effect):
    steps = range(1, max_steps + 1)
    for step in steps:
        state_and_effect = samba_step(state_and_effect)
    return state_and_effect


# linear covariate model selection: parent=param_base_model

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

    # Exploratory covariates and cov_funcs
    exploratory_cov = tuple(c for c in ss_mfl.covariate if c.optional.option)
    exploratory_cov_funcs = all_funcs(Model(), exploratory_cov)
    exploratory_cov_funcs = {cov_effect[1:-1]: cov_func
                             for cov_effect, cov_func in exploratory_cov_funcs.items()
                             if cov_effect[-1] == "ADD"}

    # cov_funcs for linear covariate models
    linear_cov_funcs = {cov_effect: partial(add_covariate_effect,
                                            parameter=cov_effect[0],
                                            covariate=cov_effect[1],
                                            effect="lin",
                                            operation="+")
                        for cov_effect in exploratory_cov_funcs.keys()}

    # indexed exploratory cov_funcs
    indexed_explor_cov_funcs = {"-".join(cov_effect[0:3]): cov_func
                                for cov_effect, cov_func in exploratory_cov_funcs.items()}

    return (indexed_explor_cov_funcs, linear_cov_funcs, filtered_model)


def _create_samba_dataset(model_entry, param, covariates):
    # Get ETA values associated with the interested model parameter
    eta_name = get_parameter_rv(model_entry.model, param)[0]
    eta_column = model_entry.modelfit_results.individual_estimates[eta_name]
    eta_column = eta_column.rename("DV")

    # Extract the covariates dataset
    covariates = list(set(covariates))  # drop duplicated covariates
    covariate_columns = model_entry.model.dataset[["ID"] + covariates]
    covariate_columns.loc[:, covariates] = covariate_columns.loc[:, covariates].apply(
        np.log)  # for linear proxy model, covariates need to be log transformed
    covariate_columns = covariate_columns.replace([np.inf, -np.inf], 1)
    # Merge the ETAs and Covariate dataset
    dataset = covariate_columns.join(eta_column, "ID")

    return dataset


def _create_samba_base_model(
        modelentry,
        param,
        covariates,
):
    """
    Create base model [Y ~ THETA(1) + ERR(1)] for the model parameters,
    with ETA values associated with these model parameters as DV.
    The OFVs of these base models are used as the basis of proxy model selection.
    """
    dataset = _create_samba_dataset(modelentry, param, covariates)

    # parameters
    theta = Parameter(name="theta", init=0.1)
    sigma = Parameter(name="sigma", init=0.2)
    params = Parameters((theta, sigma))

    # random variables
    eps_dist = NormalDistribution.create(name="epsilon", level="ruv",
                                         mean=0, variance=sigma.symbol)
    random_vars = RandomVariables.create(dists=[eps_dist])

    # assignments
    base = Assignment.create(symbol=Expr.symbol(param), expression=theta.symbol)
    ipred = Assignment.create(symbol=Expr.symbol("IPRED"), expression=base.symbol)
    y = Assignment.create(symbol=Expr.symbol("Y"),
                          expression=Expr.symbol("IPRED") + Expr.symbol("epsilon"))
    statements = Statements([base, ipred, y])

    name = f"samba_{param}_base"
    est = EstimationStep.create(method="FO", maximum_evaluations=9999,
                                tool_options={"NSIG": 6, "PRINT": 1, "NOHABORT": 0})

    base_model = Model.create(name=name, parameters=params, random_variables=random_vars,
                              statements=statements,
                              dataset=dataset,
                              description=name,
                              execution_steps=ExecutionSteps.create([est]),
                              dependent_variables={y.symbol: 1})

    di = base_model.datainfo
    di = di.set_dv_column("DV")
    di = di.set_id_column("ID")
    base_model = base_model.replace(datainfo=di)

    base_model = convert_model(base_model, to_format="nonmem")
    return base_model


def _create_proxy_model_table(candidates, steps, proxy_models):
    step_cols_to_keep = ['step', 'pvalue', 'model']
    steps_df = steps.reset_index()[step_cols_to_keep].set_index(['step', 'model'])

    steps_df = steps_df.reset_index()
    steps_df = steps_df[steps_df['model'].isin(proxy_models)]
    steps_df = steps_df.set_index(['step', 'model'])

    return steps_df


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


if __name__ == "__main__":
    #%%
    from pharmpy.modeling import load_example_model, print_model_code, read_model
    from pharmpy.tools import load_example_modelfit_results, fit
    import os
    import shutil

    for name in os.listdir():
        if name.startswith("modelfit"):
            shutil.rmtree(name)
    #%%
    # model = load_example_model("pheno")
    # results = load_example_modelfit_results("pheno")
    # modelentry = ModelEntry.create(model=model, modelfit_results=results)
    # search_space = "COVARIATE?([CL, VC], WGT, [EXP, POW])"
    model = read_model("sambas.ctl")
    modelentry = ModelEntry.create(model=model)
    search_space = "COVARIATE?([CL, V], [WT, AGE], POW); COVARIATE?([CL,V], SEX, EXP)"

    #%%
    state_and_effect = _init_search_state(0.05, search_space, modelentry)
    #%%
    # step = samba_step(state_and_effect)

    for i in range(3):
        state_and_effect = samba_step(state_and_effect)

    # final_step = samba_search(max_steps=3, state_and_effect=state_and_effect)  # FIXME: not working because the state_and_effect doesn't update itself at each step

# %%
