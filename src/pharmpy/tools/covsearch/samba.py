# isolate the samba algorithm from tool.py
from functools import partial
from collections import Counter, defaultdict
from dataclasses import astuple, dataclass, replace
from itertools import count
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

from pharmpy.basic.expr import Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
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
    mu_reference_model,
    remove_covariate_effect,
    add_estimation_step,
    remove_estimation_step,
)
from pharmpy.modeling.covariate_effect import add_covariate_effect
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.tools.mfl.feature.covariate import EffectLiteral
from pharmpy.tools.mfl.feature.covariate import parse_spec, spec
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

from pharmpy.tools.mfl.filter import COVSEARCH_STATEMENT_TYPES
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.covsearch.results import COVSearchResults
from pharmpy.tools.covsearch.tool import task_results

NAME_WF = 'samba_covsearch'

DataFrame = Any  # NOTE: should be pd.DataFrame but we want lazy loading


@dataclass(frozen=True)
class Effect:
    parameter: str
    covariate: str
    fp: str
    operation: str


class DummyEffect(Effect):
    pass


@dataclass(frozen=True)
class Step:
    alpha: float
    effect: Effect


class SambaStep(Step):
    pass


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
        max_steps: int = -1,
        alpha: float = 0.05,
        results: Optional[ModelfitResults] = None,
        model: Optional[Model] = None,
):
    """
    Workflow builder for SAMBA covariate search algorithm.
    """

    wb = WorkflowBuilder(name=NAME_WF)

    # Initiate model and search state
    store_task = Task("store_input_model", _store_input_model, model, results)
    wb.add_task(store_task)

    init_task = Task("init", _init_search_state, search_space)
    wb.add_task(init_task, predecessors=store_task)

    # SAMBA search task
    samba_search_task = Task("samba_search", samba_search,
                             max_steps,
                             alpha,
                             )
    wb.add_task(samba_search_task, predecessors=init_task)
    search_output = wb.output_tasks

    # Results task
    results_task = Task("results", samba_task_results,
                        alpha,
                        # p_backward, 
                        # strictness,
                        )
    wb.add_task(results_task, predecessors=search_output)

    return Workflow(wb)


def _store_input_model(context, model, results):
    """Store the input model"""
    model = model.replace(name="input_model", description="input")
    input_modelentry = ModelEntry.create(model=model, modelfit_results=results)
    context.store_input_model_entry(input_modelentry)

    return input_modelentry


def _init_search_state(
        context,
        search_space: str,
        modelentry: ModelEntry
):
    """ Initialize SAMBA covariate search"""
    model = modelentry.model
    exploratory_cov_funcs, linear_cov_funcs, filtered_model = filter_search_space_and_model(search_space,
                                                                                            model)

    # init nonlinear search state
    nonlinear_search_state = _init_nonlinear_search_state(context, filtered_model)
    filtered_modelentry = nonlinear_search_state.start_modelentry

    # create linear covariate models
    linear_modelentry_dict, param_cov_list = _param_indexed_linear_modelentries(linear_cov_funcs, filtered_modelentry)

    return (nonlinear_search_state, linear_modelentry_dict, exploratory_cov_funcs, param_cov_list)


def _init_nonlinear_search_state(context, filtered_model):
    # nonlinear mixed effect model setup
    filtered_model = filtered_model.replace(name="samba_start_model", description="start")
    filtered_model = mu_reference_model(filtered_model)
    filtered_model = remove_estimation_step(filtered_model, idx=0)
    filtered_model = add_estimation_step(filtered_model, method="SAEM", idx=0,
                                         tool_options={'NITER': 1000, 'AUTO': 1, 'PHITYPE': 1},
                                         )
    
    # nonlinear mixed effect modelentry creation and fit
    filtered_modelentry = ModelEntry.create(model=filtered_model)
    filtered_fit_wf = create_fit_workflow(modelentries=[filtered_modelentry])
    filtered_modelentry = call_workflow(filtered_fit_wf, 'fit_filtered_model', context)
    
    # nonlinear search state
    nonlinear_model_candidate = Candidate(filtered_modelentry, ())
    nonlinear_search_state = SearchState(modelentry, filtered_modelentry, nonlinear_model_candidate,
                                         [nonlinear_model_candidate])
    return nonlinear_search_state


def _param_indexed_linear_modelentries(linear_cov_funcs, filtered_modelentry):
    param_indexed_funcs = {}  # {param: {cov_effect: cov_func}}
    param_cov_list = {}       # {param: [covariates]}
    for cov_effect, cov_func in linear_cov_funcs.items():
        param = cov_effect[0]
        if param not in param_indexed_funcs.keys():
            param_indexed_funcs[param] = {cov_effect: cov_func}
            param_cov_list[param] = [cov_effect[1]]
        else:
            param_indexed_funcs[param].update({cov_effect: cov_func})
            param_cov_list[param].append(cov_effect[1])

    # linear_modelentry_dict: {param: [linear_base, linear_covariate]}
    linear_modelentry_dict = dict.fromkeys(param_cov_list.keys(), None)
    # create param_base_model
    for param, covariates in param_cov_list.items():
        param_base_model = _create_samba_base_model(filtered_modelentry, param, covariates)
        param_base_modelentry = ModelEntry.create(
            model=param_base_model)
        linear_modelentry_dict[param] = [param_base_modelentry]

        # create linear covariate models for each parameter ("lin", "+")
        for cov_effect, linear_func in param_indexed_funcs[param].items():
            param_cov_model = linear_func(model=param_base_model)
            description = "_".join(cov_effect[0:3])
            param_cov_model = param_cov_model.replace(description=description)
            param_cov_modelentry = ModelEntry.create(model=param_cov_model)
            linear_modelentry_dict[param].append(param_cov_modelentry)
    return linear_modelentry_dict, param_cov_list


def samba_step(
    context, 
    step,
    alpha, 
    state_and_effect,
):
    nonlinear_search_state, linear_modelentry_dict, exploratory_cov_funcs, param_cov_list = state_and_effect
    best_nlme_candidate = nonlinear_search_state.best_candidate_so_far
    best_nlme_modelentry = nonlinear_search_state.best_candidate_so_far.modelentry
    print(best_nlme_modelentry.modelfit_results.ofv)
    
    ################# LINEAR COVARIATE MODEL PROCESSING #####################
    # update dataset (etas) for all linear covariate candidate models
    for param, covariates in param_cov_list.items():
        # update dataset
        updated_dataset = _create_samba_dataset(best_nlme_modelentry, param, covariates, log_transform=False)
        # update linear covariate models
        covs = ["Base"] + covariates
        linear_modelentry_dict[param] = [ModelEntry.create(model=me.model.replace(dataset=updated_dataset, 
                                                                            name=f"step {step}_Lin_{param}_{covs[i]}"))
                                   for i, me in enumerate(linear_modelentry_dict[param])]
    selected_explor_cov_funcs = []
    # fit linear covariate models
    # NOTE: MINIMIZATION TERMINATED may result in potentially wrong OVF values
    for param, linear_modelentries in linear_modelentry_dict.items():
        # TODO: optimize the clumsy workflow building chuncks, a wf_func for similar code chunk (if not isinstance(input, modelentry): to_modelentry)
        wb = WorkflowBuilder()
        for i in linear_modelentries:
            task = Task("fit_linear_mes", lambda x: x, i)
            wb.add_task(task)
        linear_fit_wf = create_fit_workflow(n=len(linear_modelentries))
        wb.insert_workflow(linear_fit_wf)
        task_gather = Task("gather", lambda *models: models)
        wb.add_task(task_gather, predecessors=wb.output_tasks)
        linear_modelentries = call_workflow(Workflow(wb), 'fit_linear_models', context)
        linear_modelentry_dict[param] = linear_modelentries

        # covariate model selection: best of many
        ofvs = [modelentry.modelfit_results.ofv
                if modelentry.modelfit_results is not None
                else np.nan
                for modelentry in linear_modelentries]
        selected_cov = lrt_best_of_many(parent=linear_modelentries[0], models=linear_modelentries[1:],
                                        parent_ofv=ofvs[0], model_ofvs=ofvs[1:], alpha=alpha)
        cov_model_index = selected_cov.model.description
        print("Selected Covariate Effect: ", cov_model_index)
        if "Base" not in cov_model_index:
            selected_explor_cov_funcs.append(exploratory_cov_funcs[cov_model_index])

    ################# NONLINEAR MIXED EFFECT MODEL PROCESSING #####################
    # nonlinear mixed effect model selection
    if selected_explor_cov_funcs:  # if any linear cov_model is selected, create corresponding nonlinear models
        new_nonlin_models = []
        for cov_func in selected_explor_cov_funcs:
            cov_func_args = cov_func.keywords
            cov_effect = f"{cov_func_args["parameter"]}-{cov_func_args["covariate"]}-{cov_func_args["effect"]}"
            nonlin_model_added_effect = cov_func(best_nlme_modelentry.model)
            print(f"{best_nlme_modelentry.model.description};({cov_effect.lower()})")
            nonlin_model_added_effect = nonlin_model_added_effect.replace(name=f"step {step}_NLin_{cov_effect}",
                                                                            description=f"{best_nlme_modelentry.model.description};({cov_effect.lower()})")
            new_nonlin_models.append(nonlin_model_added_effect)
        
        # TODO: a wf_func for similar code chunk (if not isinstance(input, modelentry): to_modelentry)
        wb = WorkflowBuilder()
        model_to_modelentry = lambda model: ModelEntry.create(model, parent=best_nlme_modelentry.model)
        for nonlin_model in new_nonlin_models:
            nonlin_me_task = Task("to_nonlin_modelentry", model_to_modelentry, nonlin_model)
            wb.add_task(nonlin_me_task)
        nonlin_fit_wf = create_fit_workflow(n=len(new_nonlin_models))
        wb.insert_workflow(nonlin_fit_wf)
        task_gather = Task("gather", lambda *models: models)
        wb.add_task(task_gather, predecessors=wb.output_tasks)
        new_nonlin_modelentries = call_workflow(Workflow(wb), 'fit_nonlinear_models', context)

        for me, cov_func in zip(new_nonlin_modelentries, selected_explor_cov_funcs):
            cov_func_args = cov_func.keywords
            nlme_candidate = Candidate(me, steps=best_nlme_candidate.steps + 
                                       (SambaStep(alpha, DummyEffect(cov_func_args["parameter"],
                                                                     cov_func_args["covariate"],
                                                                     cov_func_args["effect"],
                                                                     cov_func_args["operation"])),))
            nonlinear_search_state.all_candidates_so_far.extend([nlme_candidate])
        
        ofvs = [modelentry.modelfit_results.ofv 
                if modelentry.modelfit_results is not None
                else np.nan
                for modelentry in new_nonlin_modelentries]

        new_best_nlme_modelentry = lrt_best_of_many(parent=best_nlme_modelentry,
                                                models=new_nonlin_modelentries,
                                                parent_ofv=best_nlme_modelentry.modelfit_results.ofv,
                                                model_ofvs=ofvs,
                                                alpha=alpha)
        print(new_best_nlme_modelentry.modelfit_results.ofv)
        
        if new_best_nlme_modelentry != best_nlme_modelentry:
            # update search states
            context.store_model_entry(ModelEntry.create(model=new_best_nlme_modelentry.model.replace(name=f"step {step}_selection")))
            best_candidate_so_far = next(
                filter(
                    lambda candidate: candidate.modelentry is new_best_nlme_modelentry,
                    nonlinear_search_state.all_candidates_so_far
                )
            )
            nonlinear_search_state = replace(nonlinear_search_state,
                                            best_candidate_so_far=best_candidate_so_far)

    return (nonlinear_search_state, linear_modelentry_dict, exploratory_cov_funcs, param_cov_list)


def _update_linear_covariate_dataset():
    """ Update the dataset of linear covariate models based on the current best nlme model.
        Return updated linear_modelentry_dict
    """
    pass


def _select_linear_covariate_model():
    """ Linear Covariate Model Selection
        Return selected covariate effects
    """
    pass


def _select_nonlin_model():
    pass


def samba_search(
    context, 
    max_steps, 
    alpha,
    state_and_effect,
):
    steps = range(1, max_steps + 1) if max_steps>=1 else count(1)
    for step in steps:
        prev_best = state_and_effect[0].best_candidate_so_far
        
        state_and_effect = samba_step(context, 
                                      step, 
                                      alpha,
                                      state_and_effect)
        
        new_best = state_and_effect[0].best_candidate_so_far
        if new_best is prev_best:
            print(f"Search stops at step {step} due to no signif decrease in ofv of nlme models.")
            break
    
    return state_and_effect[0]


def filter_search_space_and_model(search_space, model):
    """ Prepare the input model based on the search space.
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
    # indexed exploratory cov_funcs for nonlinear mixed effect models
    indexed_explor_cov_funcs = {"_".join(cov_effect[0:3]): cov_func
                                for cov_effect, cov_func in exploratory_cov_funcs.items()}
    
    # cov_funcs for linear covariate models
    linear_cov_funcs = {cov_effect: partial(add_covariate_effect,
                                            parameter=cov_effect[0],
                                            covariate=cov_effect[1],
                                            effect="lin",
                                            operation="+")
                        for cov_effect in exploratory_cov_funcs.keys()}

    return (indexed_explor_cov_funcs, linear_cov_funcs, filtered_model)


def _create_samba_dataset(model_entry, param, covariates, log_transform=True):
    # Get ETA values associated with the interested model parameter
    eta_name = get_parameter_rv(model_entry.model, param)[0]
    eta_column = model_entry.modelfit_results.individual_estimates[eta_name]
    eta_column = eta_column.rename("DV")

    # Extract the covariates dataset
    covariates = list(set(covariates))  # drop duplicated covariates
    covariate_columns = model_entry.model.dataset[["ID"] + covariates]
    if log_transform:
        covariate_columns.loc[:, covariates] = covariate_columns.loc[:, covariates].apply(
            np.log)  # for linear covariate models, covariates need to be log transformed
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
    Create linear base model [Y ~ THETA(1) + ERR(1)] for the parameters to be explored.
    ETA values associated with these model parameters are set as dependent variable (DV).
    The OFVs of these base models are used as the basis of linear covariate model selection.
    """
    dataset = _create_samba_dataset(modelentry, param, covariates, log_transform=False)

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

    name = f"samba_{param}_Base_Lin"
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


def samba_task_results(context, p_forward, state):
    # set p_backward and strictness to None
    return task_results(context, p_forward, p_backward=None, strictness=None, state=state)


if __name__ == "__main__":
    #%%
    from pharmpy.modeling import load_example_model, print_model_code, read_model
    from pharmpy.tools import load_example_modelfit_results, fit
    from pharmpy.workflows import execute_workflow  # , default_context
    import os
    import shutil

    for name in os.listdir():
        if name.startswith("modelfit"):
            shutil.rmtree(name)
    
    for name in os.listdir():
        if name.startswith("samba_covsearch"):
            shutil.rmtree(name)
    
    #%%
    # model = load_example_model("pheno")
    # results = load_example_modelfit_results("pheno")
    # modelentry = ModelEntry.create(model=model, modelfit_results=results)
    # search_space = "COVARIATE?([CL, VC], WGT, [EXP, POW])"
    model = read_model("sambas.ctl")
    modelentry = ModelEntry.create(model=model)
    search_space = "COVARIATE?([CL, V], [WT, AGE], POW); COVARIATE?([CL,V], SEX, EXP)"
    
    #%% WORKFLOW TESTING
    # SAMBA WORKFLOW INIT
    wf = samba_workflow(search_space=search_space,
                        max_steps=-1,
                        alpha=0.05,
                        model=model,
                        )
    #%% WORKFLOW TESTING
    # SAMBA WORKFLOW EXECUTE
    res = execute_workflow(wf)
