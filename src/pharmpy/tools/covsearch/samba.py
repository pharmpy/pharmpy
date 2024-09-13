from dataclasses import replace, dataclass
from functools import partial
from itertools import count
from typing import Literal, Optional, Union

import pandas as pd
import statsmodels.formula.api as smf

import pharmpy.tools.covsearch.tool as scm_tool
from pharmpy.basic.expr import Expr
from pharmpy.deps import numpy as np
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
    add_estimation_step,
    convert_model,
    get_parameter_rv,
    mu_reference_model,
    remove_covariate_effect,
    remove_estimation_step,
    unconstrain_parameters,
)
from pharmpy.modeling.covariate_effect import add_covariate_effect
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.modeling.lrt import best_of_two as lrt_best_of_two
from pharmpy.tools.common import update_initial_estimates
from pharmpy.tools.covsearch.util import (
    Candidate,
    DummyEffect,
    ForwardStep,
    SearchState,
    set_maxevals,
)
from pharmpy.tools.mfl.feature.covariate import parse_spec, spec
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

NAME_WF = 'covsearch'


@dataclass
class StateAndEffect:
    search_state: SearchState
    effect_funcs: dict


@dataclass
class LinStateAndEffect(StateAndEffect):
    linear_modelentries: dict
    param_cov_list: dict


def samba_workflow(
    search_space: Union[str, ModelFeatures],
    max_steps: int = -1,
    alpha: float = 0.05,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    max_eval: bool = False,
    statsmodels: bool = True,
    algorithm: Literal['samba'] = 'samba',
    nsamples: int = 10,
    weighted_linreg: bool = False,
    lin_filter: int = 2,
):
    """
    Workflow builder for SAMBA covariate search algorithm.
    """

    wb = WorkflowBuilder(name=NAME_WF)

    # Initiate model and search state
    store_task = Task("store_input_model", store_input_model, model, results, max_eval)
    wb.add_task(store_task)

    init_task = Task("init", init_search_state, search_space, algorithm, nsamples, weighted_linreg)
    wb.add_task(init_task, predecessors=store_task)

    # SAMBA search task
    samba_search_task = Task(
        "samba_search",
        samba_search,
        max_steps,
        alpha,
        statsmodels,
        nsamples,
        lin_filter,
        algorithm,
    )
    wb.add_task(samba_search_task, predecessors=init_task)
    search_output = wb.output_tasks

    # Results task
    results_task = Task(
        "results",
        samba_task_results,
        alpha,
    )
    wb.add_task(results_task, predecessors=search_output)

    return Workflow(wb)


def store_input_model(context, model, results, max_eval):
    """Store the input model"""
    model = model.replace(name="input", description="")
    if max_eval:
        input_modelentry = set_maxevals(model, results)
    else:
        input_modelentry = ModelEntry.create(model=model, modelfit_results=results)
    context.store_input_model_entry(input_modelentry)
    return input_modelentry


def init_search_state(context, search_space, algorithm, nsamples, modelentry):
    model = modelentry.model
    effect_funcs, filtered_model = filter_search_space_and_model(search_space, model)
    search_state = init_nonlinear_search_state(
        context, modelentry, filtered_model, algorithm, nsamples
    )
    return StateAndEffect(search_state=search_state, effect_funcs=effect_funcs)


def init_linear_state_and_effect(nsamples, algorithm, weighted_linreg, state_and_effect):
    """
    initialize the elements required for linear covariate model selection
    """
    effect_funcs, search_state = state_and_effect.effect_funcs, state_and_effect.search_state
    effect_funcs, linear_effect_funcs = linearize_coveffects(effect_funcs)
    linear_modelentries, param_covariate_lst = create_linear_covmodels(
        linear_effect_funcs, search_state.start_modelentry, nsamples, algorithm, weighted_linreg
    )
    return LinStateAndEffect(
        effect_funcs=effect_funcs,
        search_state=search_state,
        linear_models=linear_modelentries,
        param_cov_list=param_covariate_lst,
    )


def linearize_coveffects(exploratory_cov_funcs):
    """
    change covariate effect function's effect to "lin" and operation to "+"
    """
    explor_cov_funcs = {}
    linear_cov_funcs = {}
    for cov_effect, cov_func in exploratory_cov_funcs.items():
        param_index = "_".join(cov_effect[0:2])
        if param_index not in explor_cov_funcs:
            explor_cov_funcs[param_index] = [cov_func]
            linear_cov_funcs[cov_effect[0:2]] = partial(
                add_covariate_effect,
                parameter=cov_effect[0],
                covariate=cov_effect[1],
                effect="lin",
                operation="+",
            )
        else:
            explor_cov_funcs[param_index].append(cov_func)
    return (explor_cov_funcs, linear_cov_funcs)


def init_nonlinear_search_state(context, input_modelentry, filtered_model, algorithm, nsamples):
    if algorithm == "samba":
        filtered_model = mu_reference_model(filtered_model)
        filtered_model = remove_estimation_step(filtered_model, idx=0)
        filtered_model = add_estimation_step(
            filtered_model,
            method="ITS",
            idx=0,
            interaction=True,
            auto=True,
            niter=5,
        )
        filtered_model = add_estimation_step(
            filtered_model,
            method="SAEM",
            idx=1,
            interaction=True,
            auto=True,
            niter=200,
            keep_every_nth_iter=50,
            tool_options={'PHITYPE': "1", 'FNLETA': "0"},
        )
        if nsamples > 0:
            filtered_model = add_estimation_step(
                filtered_model,
                method="SAEM",
                idx=2,
                niter=0,
                tool_options={
                    "EONLY": "1",
                    "NBURN": "0",
                    "MASSRESET": "0",
                    "ETASAMPLES": "1",
                    "ISAMPLE": f"{nsamples}",
                },
            )
        if nsamples == 1:
            filtered_model = add_estimation_step(
                filtered_model,
                method="IMP",
                idx=3,
                auto=True,
                interaction=True,
                niter=20,
                tool_options={
                    "EONLY": "1",
                    "MASSRESET": "1",
                    "ISAMPLE": "1000",
                },
            )
    if algorithm != "samba" and filtered_model.execution_steps[0].method != "FOCE":
        filtered_model = remove_estimation_step(filtered_model, idx=0)
        filtered_model = add_estimation_step(
            filtered_model,
            method="FOCE",
            idx=0,
            interaction=True,
            auto=True,
        )

    # nonlinear mixed effect modelentry creation and fit
    if filtered_model != input_modelentry.model:
        filtered_modelentry = ModelEntry.create(model=filtered_model)
        filtered_fit_wf = create_fit_workflow(modelentries=[filtered_modelentry])
        filtered_modelentry = call_workflow(filtered_fit_wf, 'fit_filtered_model', context)
    else:
        filtered_modelentry = input_modelentry

    candidate = Candidate(filtered_modelentry, ())
    return SearchState(input_modelentry, filtered_modelentry, candidate, [candidate])


def create_linear_covmodels(linear_cov_funcs, modelentry, nsamples, algorithm, weighted_linreg):
    param_indexed_funcs = {}  # {param: {cov_effect: cov_func}}
    param_covariate_lst = {}  # {param: [covariates]}
    for cov_effect, cov_func in linear_cov_funcs.items():
        param = cov_effect[0]
        if param not in param_indexed_funcs.keys():
            param_indexed_funcs[param] = {cov_effect: cov_func}
            param_covariate_lst[param] = [cov_effect[1]]
        else:
            param_indexed_funcs[param].update({cov_effect: cov_func})
            param_covariate_lst[param].append(cov_effect[1])

    # linear_modelentry_dict: {param: [linear_base, linear_covariate]}
    linear_modelentry_dict = dict.fromkeys(param_covariate_lst.keys(), None)
    # create param_base_covmodel
    for param, covariates in param_covariate_lst.items():
        data = create_covmodel_dataset(modelentry, param, covariates, nsamples, algorithm)
        param_base_model = create_base_covmodel(data, param, nsamples, weighted_linreg)
        param_base_modelentry = ModelEntry.create(model=param_base_model)
        linear_modelentry_dict[param] = [param_base_modelentry]

        # create linear covariate models for each parameter ("lin", "+")
        for cov_effect, linear_func in param_indexed_funcs[param].items():
            param_cov_model = linear_func(model=param_base_model)
            param_cov_model = unconstrain_parameters(param_cov_model, f"POP_{cov_effect[0]}{cov_effect[1]}")
            description = "_".join(cov_effect[0:2])
            param_cov_model = param_cov_model.replace(description=description)
            param_cov_modelentry = ModelEntry.create(model=param_cov_model)
            linear_modelentry_dict[param].append(param_cov_modelentry)
    return linear_modelentry_dict, param_covariate_lst


def samba_step(
    context, step, alpha, statsmodels, nsamples, lin_filter, algorithm, state_and_effect
):

    # LINEAR COVARIATE MODEL PROCESSING #####################
    selected_explor_cov_funcs, linear_modelentry_dict = linear_model_selection(
        context, step, alpha, state_and_effect, statsmodels, nsamples, lin_filter, algorithm
    )

    # NONLINEAR MIXED EFFECT MODEL PROCESSING #####################
    state_and_effect = nonlinear_model_selection(
        context, step, alpha, state_and_effect, selected_explor_cov_funcs
    )

    return state_and_effect


def statsmodels_linear_selection(
    step,
    alpha,
    linear_state_and_effect,
    nsamples,
    lin_filter,
    algorithm,
):
    effect_funcs = linear_state_and_effect.effect_funcs
    search_state = linear_state_and_effect.search_state
    param_cov_list = linear_state_and_effect.param_cov_list
    best_modelentry = search_state.best_candidate_so_far.modelentry
    selected_effect_funcs = []
    selected_lin_model_ofv = []

    for param, covariates in param_cov_list.items():
        # update dataset
        updated_data = create_covmodel_dataset(best_modelentry, param, covariates, nsamples, algorithm)
        covs = ["1"] + covariates
        if algorithm == "samba" and nsamples > 1:
            linear_models = [
                smf.mixedlm(f"DV~{cov}", data=updated_data, groups=updated_data["ID"])
                for cov in covs
            ]
        elif algorithm == "samba" and nsamples == 1:
            linear_models = [smf.ols(f"DV~{cov}", data=updated_data) for cov in covs]
        else:
            linear_models = [
                smf.wls(f"DV~{cov}", data=updated_data, weights=1.0 / updated_data["ETC"])
                for cov in covs
            ]
        linear_fitres = [model.fit() for model in linear_models]
        ofvs = [-2 * res.llf for res in linear_fitres]

        selected_effect_funcs, selected_lin_model_ofv = _lin_filter_option(
            lin_filter,
            linear_models,
            ofvs,
            alpha,
            param,
            selected_effect_funcs,
            effect_funcs,
            selected_lin_model_ofv,
        )
    # select the best linear model (covariate effect) with the largest drop-off in ofv
    if selected_lin_model_ofv:
        best_index = np.nanargmax(selected_lin_model_ofv)
        selected_effect_funcs = selected_effect_funcs[best_index]
        if not isinstance(selected_effect_funcs, list):
            selected_effect_funcs = [selected_effect_funcs]
    if selected_effect_funcs:
        selected_effect_funcs = {
            tuple(func.keywords.values()): func for func in selected_effect_funcs
        }

    return StateAndEffect(effect_funcs=selected_effect_funcs, search_state=search_state)


def nonmem_linear_selection(
    context,
    step,
    alpha,
    linear_state_and_effect,
    nsamples,
    lin_filter,
    algorithm,
):
    effect_funcs = linear_state_and_effect.effect_funcs
    search_state = linear_state_and_effect.search_state
    linear_modelentry_dict = linear_state_and_effect.linear_models
    param_cov_list = linear_state_and_effect.param_cov_list
    best_modelentry = search_state.best_candidate_so_far.modelentry
    selected_effect_funcs = []
    selected_lin_model_ofv = []

    for param, linear_modelentries in linear_modelentry_dict.items():
        wb = WorkflowBuilder(name="linear model selection")
        covariates = param_cov_list[param]
        # update dataset
        updated_dataset = create_covmodel_dataset(
            best_modelentry, param, covariates, nsamples, algorithm
        )
        covs = ["Base"] + covariates
        linear_modelentries = list(linear_modelentries)
        for i, me in enumerate(linear_modelentries):
            linear_modelentries[i] = ModelEntry.create(
                model=me.model.replace(
                    dataset=updated_dataset, name=f"step {step}_lin_{param}_{covs[i]}"
                )
            )
            task = Task("fit_lin_mes", lambda me: me, linear_modelentries[i])
            wb.add_task(task)
        # fit linear covariate models
        linear_fit_wf = create_fit_workflow(n=len(linear_modelentries))
        wb.insert_workflow(linear_fit_wf)
        task_gather = Task("gather", lambda *models: models)
        wb.add_task(task_gather, predecessors=wb.output_tasks)
        linear_modelentries = call_workflow(Workflow(wb), 'fit_linear_models', context)
        linear_modelentry_dict[param] = linear_modelentries

        # linear covariate model selection
        ofvs = [
            (
                modelentry.modelfit_results.ofv
                if modelentry.modelfit_results is not None
                else np.nan
            )
            for modelentry in linear_modelentries
        ]

        selected_effect_funcs, selected_lin_model_ofv = _lin_filter_option(
            lin_filter,
            linear_modelentries,
            ofvs,
            alpha,
            param,
            selected_effect_funcs,
            effect_funcs,
            selected_lin_model_ofv,
        )

    # select the best linear model (covariate effect) with the largest drop-off in ofv
    if selected_lin_model_ofv:
        best_index = np.nanargmax(selected_lin_model_ofv)
        selected_effect_funcs = selected_effect_funcs[best_index]
        if not isinstance(selected_effect_funcs, list):
            selected_effect_funcs = [selected_effect_funcs]
    if selected_effect_funcs:
        selected_effect_funcs = {
            tuple(func.keywords.values()): func for func in selected_effect_funcs
        }
    return StateAndEffect(effect_funcs=selected_effect_funcs, search_state=search_state)


def _lin_filter_option(
    lin_filter,
    linear_models,
    ofvs,
    alpha,
    param,
    selected_effect_funcs,
    effect_funcs,
    selected_lin_model_ofv,
):
    if lin_filter in [1, 2]:
        selected_cov = lrt_best_of_many(
            parent=linear_models[0],
            models=linear_models[1:],
            parent_ofv=ofvs[0],
            model_ofvs=ofvs[1:],
            alpha=alpha,
        )
        if isinstance(linear_models[0], ModelEntry):
            cov_model_index = selected_cov.model.description
        else:
            cov_model_index = "_".join([param, selected_cov.exog_names[-1]])

        if ("Base" in cov_model_index) or ("Intercept" in cov_model_index):
            cov_model_index = None

        if cov_model_index and lin_filter == 2:
            selected_effect_funcs.append(effect_funcs[cov_model_index])
            # calculate the drop-off of ofv for selected linear models
            parent_ofv = ofvs[0]
            child_ofv = ofvs[1:]
            best_index = np.nanargmin(child_ofv)
            ofv_drop = parent_ofv - child_ofv[best_index]
            selected_lin_model_ofv.append(ofv_drop)
        if cov_model_index and lin_filter == 1:
            selected_effect_funcs.extend(effect_funcs[cov_model_index])

    elif lin_filter == 0:
        selected_cov = [
            lrt_best_of_two(linear_models[0], me, ofvs[0], ofv, alpha)
            for me, ofv in zip(linear_models[1:], ofvs[1:])
        ]
        if isinstance(linear_models[0], ModelEntry):
            cov_model_index = [me.model.description for me in selected_cov]
        else:
            cov_model_index = ["_".join([param, model.exog_names[-1]]) for model in selected_cov]
        for cm_index in cov_model_index:
            if ("Base" not in cm_index) and ("Intercept" not in cm_index):
                selected_effect_funcs.extend(effect_funcs[cm_index])
    else:
        raise ValueError("lin_filter must be one from the list {0, 1, 2}")

    return selected_effect_funcs, selected_lin_model_ofv


def nonlinear_model_selection(context, step, alpha, state_and_effect, selected_explor_cov_funcs):
    nonlinear_search_state, linear_modelentry_dict, exploratory_cov_funcs, param_cov_list = (
        state_and_effect
    )
    best_nlme_candidate = nonlinear_search_state.best_candidate_so_far
    best_nlme_modelentry = nonlinear_search_state.best_candidate_so_far.modelentry
    # nonlinear mixed effect model selection
    if selected_explor_cov_funcs:
        new_nonlin_models = []
        for cov_func in selected_explor_cov_funcs:
            cov_func_args = cov_func.keywords
            cov_effect = f'{cov_func_args["parameter"]}-{cov_func_args["covariate"]}-{cov_func_args["effect"]}'

            nonlin_model_added_effect = best_nlme_modelentry.model.replace(
                name=f"step {step}_NLin_{cov_effect}",
                description=f"{best_nlme_modelentry.model.description};({cov_effect.lower()})",
            )
            nonlin_model_added_effect = update_initial_estimates(
                nonlin_model_added_effect, best_nlme_modelentry.modelfit_results
            )
            nonlin_model_added_effect = cov_func(nonlin_model_added_effect)
            new_nonlin_models.append(nonlin_model_added_effect)

        new_modelentries = [
            ModelEntry.create(model, best_nlme_modelentry.model) for model in new_nonlin_models
        ]
        nonlin_fit_wf = create_fit_workflow(modelentries=new_modelentries)
        wb = WorkflowBuilder(nonlin_fit_wf)
        task_gather = Task("gather", lambda *models: models)
        wb.add_task(task_gather, predecessors=wb.output_tasks)
        new_nonlin_modelentries = call_workflow(Workflow(wb), 'fit_nonlinear_models', context)

        for me, cov_func in zip(new_nonlin_modelentries, selected_explor_cov_funcs):
            cov_func_args = cov_func.keywords
            nlme_candidate = Candidate(
                me,
                steps=best_nlme_candidate.steps
                + (
                    ForwardStep(
                        alpha,
                        DummyEffect(
                            cov_func_args["parameter"],
                            cov_func_args["covariate"],
                            cov_func_args["effect"],
                            cov_func_args["operation"],
                        ),
                    ),
                ),
            )
            nonlinear_search_state.all_candidates_so_far.extend([nlme_candidate])

        ofvs = [
            modelentry.modelfit_results.ofv if modelentry.modelfit_results is not None else np.nan
            for modelentry in new_nonlin_modelentries
        ]
        new_best_nlme_modelentry = lrt_best_of_many(
            parent=best_nlme_modelentry,
            models=new_nonlin_modelentries,
            parent_ofv=best_nlme_modelentry.modelfit_results.ofv,
            model_ofvs=ofvs,
            alpha=alpha,
        )

        if new_best_nlme_modelentry != best_nlme_modelentry:
            # update search states
            context.store_model_entry(
                ModelEntry.create(
                    model=new_best_nlme_modelentry.model.replace(name=f"step {step}_selection")
                )
            )
            best_candidate_so_far = next(
                filter(
                    lambda candidate: candidate.modelentry is new_best_nlme_modelentry,
                    nonlinear_search_state.all_candidates_so_far,
                )
            )
            nonlinear_search_state = replace(
                nonlinear_search_state, best_candidate_so_far=best_candidate_so_far
            )
    state_and_effect = tuple(
        (nonlinear_search_state, linear_modelentry_dict, exploratory_cov_funcs, param_cov_list)
    )
    return state_and_effect


def samba_search(
    context,
    max_steps,
    alpha,
    statsmodels,
    nsamples,
    lin_filter,
    algorithm,
    state_and_effect,
):
    steps = range(1, max_steps + 1) if max_steps >= 1 else count(1)
    for step in steps:
        prev_best = state_and_effect[0].best_candidate_so_far

        state_and_effect = samba_step(
            context,
            step,
            alpha,
            statsmodels,
            nsamples,
            lin_filter,
            algorithm,
            state_and_effect,
        )

        new_best = state_and_effect[0].best_candidate_so_far
        if new_best is prev_best:
            break

    return state_and_effect[0]


def filter_search_space_and_model(search_space, model):
    """Prepare the input model based on the search space.
    Preparation includes:
    (1) cleaning up all covariate effect present in the input model
    (2) adding structural covariates in the search space if any
    (3) preparing the exploratory covariate effect indexed covariate functions for nonlinear mixed effect model
    (4) preparing the exploratory covariate effect indexed covariate functions for linear covariate models
    """
    filtered_model = model.replace(name="filtered_input_model")
    if isinstance(search_space, str):
        search_space = ModelFeatures.create_from_mfl_string(search_space)
    ss_mfl = search_space.expand(filtered_model)  # Expand to remove LET/REF

    # Clean up all covariate effect in model
    model_mfl = ModelFeatures.create_from_mfl_string(get_model_features(filtered_model))
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
        filtered_model = filtered_model.replace(description=';'.join(description))

    # Add structural covariates in search space if any
    structural_cov = tuple([c for c in ss_mfl.covariate if not c.optional.option])
    structural_cov_funcs = all_funcs(Model(), structural_cov)
    if len(structural_cov_funcs) != 0:
        description.append("ADDED")
        for cov_effect, cov_func in structural_cov_funcs.items():
            filtered_model = cov_func(filtered_model)
            description.append('({}-{}-{})'.format(cov_effect[0], cov_effect[1], cov_effect[2]))
    # Remove custom effects
    covariate_to_keep = covariate_to_keep.mfl_statement_list(["covariate"])
    for cov_effect in parse_spec(spec(filtered_model, covariate_to_keep)):
        if cov_effect[2].lower() == "custom":
            filtered_model = remove_covariate_effect(filtered_model, cov_effect[0], cov_effect[1])
            description.append('({}-{}-{})'.format(cov_effect[0], cov_effect[1], cov_effect[2]))

    filtered_model = filtered_model.replace(description=";".join(description))

    # Exploratory covariates and cov_funcs
    exploratory_cov = tuple(c for c in ss_mfl.covariate if c.optional.option)
    exploratory_cov_funcs = all_funcs(Model(), exploratory_cov)
    exploratory_cov_funcs = {
        cov_effect[1:-1]: cov_func
        for cov_effect, cov_func in exploratory_cov_funcs.items()
        if cov_effect[-1] == "ADD"
    }
    # indexed exploratory cov_funcs for nonlinear mixed effect models
    indexed_explor_cov_funcs = {}
    linear_cov_funcs = {}
    for cov_effect, cov_func in exploratory_cov_funcs.items():
        param_index = "_".join(cov_effect[0:2])
        if param_index not in indexed_explor_cov_funcs:
            indexed_explor_cov_funcs[param_index] = [cov_func]
            # cov_funcs for linear covariate models
            linear_cov_funcs[cov_effect[0:2]] = partial(
                add_covariate_effect,
                parameter=cov_effect[0],
                covariate=cov_effect[1],
                effect="lin",
                operation="+",
            )
        else:
            indexed_explor_cov_funcs[param_index].append(cov_func)

    return (indexed_explor_cov_funcs, linear_cov_funcs, filtered_model)


def create_covmodel_dataset(model_entry, param, covariates, nsamples, algorithm):

    eta_name = get_parameter_rv(model_entry.model, param)[0]
    # Extract individual parameters
    if algorithm == "samba" and nsamples >= 1:
        eta_column = model_entry.modelfit_results.individual_eta_samples[eta_name]
    else:
        eta_column = model_entry.modelfit_results.individual_estimates[eta_name]
    eta_column = eta_column.rename("DV")

    # Extract the covariates dataset
    covariates = list(set(covariates))  # drop duplicated covariates
    covariate_columns = model_entry.model.dataset[["ID"] + covariates]
    covariate_columns = covariate_columns.drop_duplicates()  # drop duplicated rows
    # Log-transform covariates with only positive values
    columns_to_trans = covariate_columns.columns[(covariate_columns > 0).all(axis=0)]
    columns_to_trans = columns_to_trans.drop("ID")
    covariate_columns.loc[:, columns_to_trans] = covariate_columns.loc[:, columns_to_trans].apply(
        np.log
    )
    # Merge the ETAs and Covariate dataset
    dataset = covariate_columns.merge(eta_column, on="ID")

    # Extract the conditional variance (ETC) for individual parameters
    etc = [
        subset.loc[eta_name, eta_name].squeeze()
        for subset in model_entry.modelfit_results.individual_estimates_covariance
    ]
    subject_id = model_entry.modelfit_results.individual_estimates.index
    etc_columns = pd.DataFrame({"ID": subject_id, "ETC": etc})
    dataset = dataset.merge(etc_columns, on="ID")

    return dataset


def create_base_covmodel(data, parameter, nsamples, weighted_linreg=False):
    """
    Create linear base model [Y ~ THETA(1) + ERR(1)] for the parameters to be explored.
    ETA values associated with these model parameters are set as dependent variable (DV).
    The OFVs of these base models are used as the basis of linear covariate model selection.
    """
    if nsamples>1:
        base_model = _mixed_effects_base_model(data, parameter)
    else:
        base_model = _linear_base_model(data, parameter, weighted_linreg)

    di = base_model.datainfo
    di = di.set_dv_column("DV")
    di = di.set_id_column("ID")
    base_model = base_model.replace(datainfo=di)

    base_model = convert_model(base_model, to_format="nonmem")
    return base_model


def _linear_base_model(data, parameter, weighted_linreg=False):
    # parameters
    theta = Parameter(name="theta", init=0.1)
    sigma = Parameter(name="sigma", init=0.2)
    params = Parameters((theta, sigma))
    # random variables
    eps_dist = NormalDistribution.create(name="epsilon", level="ruv", mean=0, variance=sigma.symbol)
    random_vars = RandomVariables.create(dists=[eps_dist])
    # assignments
    base = Assignment.create(symbol=Expr.symbol(parameter), expression=theta.symbol)
    ipred = Assignment.create(symbol=Expr.symbol("IPRED"), expression=base.symbol)
    if weighted_linreg:
        y = Assignment.create(
            symbol=Expr.symbol("Y"), expression=Expr.symbol("IPRED") +
                                                Expr.symbol("epsilon") * Expr.sqrt(Expr.symbol("ETC"))
        )
        name = f"{parameter}_Weighted_Base"
    else:
        y = Assignment.create(
            symbol=Expr.symbol("Y"), expression=Expr.symbol("IPRED") +
                                                Expr.symbol("epsilon")
        )
        name = f"{parameter}_Base"
    statements = Statements([base, ipred, y])

    est = EstimationStep.create(
        method="FO", maximum_evaluations=9999, tool_options={"NSIG": 6, "PRINT": 1, "NOHABORT": 0}
    )
    base_model = Model.create(
        name=name,
        parameters=params,
        random_variables=random_vars,
        statements=statements,
        dataset=data,
        description=name,
        execution_steps=ExecutionSteps.create([est]),
        dependent_variables={y.symbol: 1},
    )
    return base_model


def _mixed_effects_base_model(data, parameter):
    # parameters
    theta = Parameter(name="theta", init=0.1)
    sigma = Parameter(name="sigma", init=0.2)
    omega0 = Parameter(name="DUMMYOMEGA", init=0, fix=True)
    omega1 = Parameter(name="OMEGA_ETA_INT", init=0.1)
    omega2 = Parameter(name="OMEGA_ETA_EPS", init=0.1)
    params = Parameters((theta, sigma, omega0, omega1, omega2))
    # random variables
    eps_dist = NormalDistribution.create(name="epsilon", level="ruv", mean=0, variance=sigma.symbol)
    eta0_dist = NormalDistribution.create(name="DUMMYETA", level="iiv", mean=0, variance=omega0.symbol)
    eta1_dist = NormalDistribution.create(name="ETA_INT", level="iiv", mean=0, variance=omega1.symbol)
    eta2_dist = NormalDistribution.create(name="ETA_EPS", level="iiv", mean=0, variance=omega2.symbol)
    random_vars = RandomVariables.create(dists=[eps_dist, eta0_dist, eta1_dist, eta2_dist])
    # assignments
    base = Assignment.create(symbol=Expr.symbol(parameter), expression=theta.symbol)
    ipred = Assignment.create(symbol=Expr.symbol("IPRED"), expression=base.symbol * Expr.exp(Expr.symbol("DUMMYETA")))
    y = Assignment.create(
        symbol=Expr.symbol("Y"), expression=Expr.symbol("IPRED") +
                                            Expr.symbol("ETA_INT") +
                                            Expr.symbol("epsilon") * Expr.exp(Expr.symbol("ETA_EPS"))
    )
    # y = Assignment.create(
    #     symbol=Expr.symbol("Y"), expression=Expr.symbol("IPRED") +
    #                                         Expr.symbol("ETA_INT") +
    #                                         Expr.symbol("epsilon")*(1+Expr.symbol("ETA_EPS"))
    # )
    statements = Statements([base, ipred, y])
    name = f"{parameter}_Mixed_Effects_Base"
    est = EstimationStep.create(
        method="FOCE", maximum_evaluations=9999, interaction=True,
        tool_options={"NSIG": 6, "PRINT": 1, "NOHABORT": 0}
    )
    base_model = Model.create(
        name=name,
        parameters=params,
        random_variables=random_vars,
        statements=statements,
        dataset=data,
        description=name,
        execution_steps=ExecutionSteps.create([est]),
        dependent_variables={y.symbol: 1},
    )

    return base_model


def samba_task_results(context, p_forward, state):
    # set p_backward and strictness to None
    return scm_tool.task_results(context, p_forward, p_backward=None, strictness=None, state=state)
