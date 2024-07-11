# isolate the samba algorithm from tool.py
from dataclasses import dataclass, replace
from functools import partial
from itertools import count
from typing import Any, List, Optional, Tuple, Union

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
)
from pharmpy.modeling.covariate_effect import add_covariate_effect
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.tools.mfl.feature.covariate import parse_spec, spec
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

NAME_WF = 'covsearch'

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


class ForwardStep(Step):
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
    samba_search_task = Task(
        "samba_search",
        samba_search,
        max_steps,
        alpha,
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


def _store_input_model(context, model, results):
    """Store the input model"""
    model = model.replace(name="input_model", description="input")
    input_modelentry = ModelEntry.create(model=model, modelfit_results=results)
    context.store_input_model_entry(input_modelentry)

    return input_modelentry


def _init_search_state(context, search_space: str, modelentry: ModelEntry):
    """Initialize SAMBA covariate search"""
    model = modelentry.model
    exploratory_cov_funcs, linear_cov_funcs, filtered_model = filter_search_space_and_model(
        search_space, model
    )

    # init nonlinear search state
    nonlinear_search_state = _init_nonlinear_search_state(context, modelentry, filtered_model)
    filtered_modelentry = nonlinear_search_state.start_modelentry

    # create linear covariate models
    linear_modelentry_dict, param_cov_list = _param_indexed_linear_modelentries(
        linear_cov_funcs, filtered_modelentry, context
    )

    return (nonlinear_search_state, linear_modelentry_dict, exploratory_cov_funcs, param_cov_list)


def _init_nonlinear_search_state(context, input_modelentry, filtered_model):
    # nonlinear mixed effect model setup
    filtered_model = filtered_model.replace(name="samba_start_model", description="start")
    filtered_model = mu_reference_model(filtered_model)
    filtered_model = remove_estimation_step(filtered_model, idx=0)
    filtered_model = add_estimation_step(
        filtered_model,
        method="SAEM",
        idx=0,
        tool_options={'NITER': 1000, 'AUTO': 1, 'PHITYPE': 1},
    )
    # TESTING ###################
    # TODO: IMP est step for suitable OFV for LRT (TBD: EONLY setting)
    # FIXME: Adding IMP est step bugs the summary_tool's ofv and dofv (empty)
    filtered_model = add_estimation_step(filtered_model, method="IMP", idx=1, isample=3000, niter=5, interaction=True,
                                         tool_options={"EONLY": "2", "MAPITER": "0", "SIGL": "8", "PRINT":"1"})
    # TESTING ###################

    # nonlinear mixed effect modelentry creation and fit
    filtered_modelentry = ModelEntry.create(model=filtered_model)
    filtered_fit_wf = create_fit_workflow(modelentries=[filtered_modelentry])
    filtered_modelentry = call_workflow(filtered_fit_wf, 'fit_filtered_model', context)

    # nonlinear search state
    nonlinear_model_candidate = Candidate(filtered_modelentry, ())
    nonlinear_search_state = SearchState(
        input_modelentry,
        filtered_modelentry,
        nonlinear_model_candidate,
        [nonlinear_model_candidate],
    )
    return nonlinear_search_state


def _param_indexed_linear_modelentries(linear_cov_funcs, filtered_modelentry, context):
    param_indexed_funcs = {}  # {param: {cov_effect: cov_func}}
    param_cov_list = {}  # {param: [covariates]}
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
        param_base_model = _create_samba_base_model(filtered_modelentry, param, covariates, context)
        param_base_modelentry = ModelEntry.create(model=param_base_model)
        linear_modelentry_dict[param] = [param_base_modelentry]

        # create linear covariate models for each parameter ("lin", "+")
        for cov_effect, linear_func in param_indexed_funcs[param].items():
            param_cov_model = linear_func(model=param_base_model)
            description = "_".join(cov_effect[0:2])
            param_cov_model = param_cov_model.replace(description=description)
            param_cov_modelentry = ModelEntry.create(model=param_cov_model)
            linear_modelentry_dict[param].append(param_cov_modelentry)
    return linear_modelentry_dict, param_cov_list


def samba_step(context, step, alpha, state_and_effect):
    nonlinear_search_state, linear_modelentry_dict, exploratory_cov_funcs, param_cov_list = (
        state_and_effect
    )
    best_nlme_candidate = nonlinear_search_state.best_candidate_so_far
    best_nlme_modelentry = nonlinear_search_state.best_candidate_so_far.modelentry

    selected_explor_cov_funcs = []
    # LINEAR COVARIATE MODEL PROCESSING #####################
    # TODO: try statsmodels linear
    # update dataset (etas) for all linear covariate candidate models
    for param, linear_modelentries in linear_modelentry_dict.items():
        wb = WorkflowBuilder(name="linear model selection")
        covariates = param_cov_list[param]
        # update dataset
        updated_dataset = _create_samba_dataset(
            best_nlme_modelentry, param, covariates, context, log_transform=True
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
            modelentry.modelfit_results.ofv if modelentry.modelfit_results is not None else np.nan
            for modelentry in linear_modelentries
        ]
        selected_cov = lrt_best_of_many(
            parent=linear_modelentries[0],
            models=linear_modelentries[1:],
            parent_ofv=ofvs[0],
            model_ofvs=ofvs[1:],
            alpha=alpha,
        )
        cov_model_index = selected_cov.model.description
        print("Selected Covariate Effect: ", cov_model_index)
        if "Base" not in cov_model_index:
            selected_explor_cov_funcs.extend(exploratory_cov_funcs[cov_model_index])

    # NONLINEAR MIXED EFFECT MODEL PROCESSING #####################
    # nonlinear mixed effect model selection
    if selected_explor_cov_funcs:
        new_nonlin_models = []
        for cov_func in selected_explor_cov_funcs:
            cov_func_args = cov_func.keywords
            cov_effect = f'{cov_func_args["parameter"]}-{cov_func_args["covariate"]}-{cov_func_args["effect"]}'
            nonlin_model_added_effect = cov_func(best_nlme_modelentry.model)
            nonlin_model_added_effect = nonlin_model_added_effect.replace(
                name=f"step {step}_NLin_{cov_effect}",
                description=f"{best_nlme_modelentry.model.description};({cov_effect.lower()})",
            )
            new_nonlin_models.append(nonlin_model_added_effect)

        wb = WorkflowBuilder(name="nonlinear model selection")

        def model_to_modelentry(model, parent=best_nlme_modelentry.model):
            return ModelEntry.create(model, parent=parent)

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
        # NOTE: some NLME models' OFV can be really off, due to poor convergence after averaging SAEM burn-in steps
        ofvs = [
            modelentry.modelfit_results.ofv if modelentry.modelfit_results is not None else np.nan
            for modelentry in new_nonlin_modelentries
        ]
        print(ofvs)
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

    return (nonlinear_search_state, linear_modelentry_dict, exploratory_cov_funcs, param_cov_list)


def samba_search(context, max_steps, alpha, state_and_effect):
    steps = range(1, max_steps + 1) if max_steps >= 1 else count(1)
    for step in steps:
        prev_best = state_and_effect[0].best_candidate_so_far

        state_and_effect = samba_step(context, step, alpha, state_and_effect)

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
    covariate_to_remove = model_mfl.mfl_statement_list(["covariate"])
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
        filtered_model = filtered_model.replace(description=";".join(description))

    # Exploratory covariates and cov_funcs
    exploratory_cov = tuple(c for c in ss_mfl.covariate if c.optional.option)
    exploratory_cov_funcs = all_funcs(Model(), exploratory_cov)
    exploratory_cov_funcs = {
        cov_effect[1:-1]: cov_func
        for cov_effect, cov_func in exploratory_cov_funcs.items()
        if cov_effect[-1] == "ADD"
    }
    # TODO: if algorithm=="SAMBA": INDEXED CODE CHUNK return (ind, lin, mod)
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


def _create_samba_dataset(model_entry, param, covariates, context, log_transform=True):
    # if model_entry.model.execution_steps[-1].method != "SAEM":
    #     m = remove_estimation_step(model_entry.model, idx=1)
    #     me = ModelEntry.create(model=m)
    #     me_fit = create_fit_workflow(modelentries=[me])
    #     model_entry = call_workflow(me_fit, "fit_for_dataset", context)
    # Get ETA values associated with the interested model parameter
    eta_name = get_parameter_rv(model_entry.model, param)[0]
    eta_column = model_entry.modelfit_results.individual_estimates[eta_name]
    eta_column = eta_column.rename("DV")

    # Extract the covariates dataset
    covariates = list(set(covariates))  # drop duplicated covariates
    covariate_columns = model_entry.model.dataset[["ID"] + covariates]
    # TESTING ###################
    if log_transform:
        columns_to_trans = covariate_columns.columns[(covariate_columns>0).all(axis=0)]
        columns_to_trans = columns_to_trans.drop("ID")
        covariate_columns.loc[:, columns_to_trans] = covariate_columns.loc[:, columns_to_trans].apply(
            np.log
        )
    # TESTING ###################
    # Merge the ETAs and Covariate dataset
    dataset = covariate_columns.join(eta_column, "ID")

    return dataset


def _create_samba_base_model(modelentry, param, covariates, context):
    """
    Create linear base model [Y ~ THETA(1) + ERR(1)] for the parameters to be explored.
    ETA values associated with these model parameters are set as dependent variable (DV).
    The OFVs of these base models are used as the basis of linear covariate model selection.
    """
    dataset = _create_samba_dataset(modelentry, param, covariates, context, log_transform=True)

    # parameters
    theta = Parameter(name="theta", init=0.1)
    sigma = Parameter(name="sigma", init=0.2)
    params = Parameters((theta, sigma))

    # random variables
    eps_dist = NormalDistribution.create(name="epsilon", level="ruv", mean=0, variance=sigma.symbol)
    random_vars = RandomVariables.create(dists=[eps_dist])

    # assignments
    base = Assignment.create(symbol=Expr.symbol(param), expression=theta.symbol)
    ipred = Assignment.create(symbol=Expr.symbol("IPRED"), expression=base.symbol)
    y = Assignment.create(
        symbol=Expr.symbol("Y"), expression=Expr.symbol("IPRED") + Expr.symbol("epsilon")
    )
    statements = Statements([base, ipred, y])

    name = f"samba_{param}_Base_Lin"
    est = EstimationStep.create(
        method="FO", maximum_evaluations=9999, tool_options={"NSIG": 6, "PRINT": 1, "NOHABORT": 0}
    )

    base_model = Model.create(
        name=name,
        parameters=params,
        random_variables=random_vars,
        statements=statements,
        dataset=dataset,
        description=name,
        execution_steps=ExecutionSteps.create([est]),
        dependent_variables={y.symbol: 1},
    )

    di = base_model.datainfo
    di = di.set_dv_column("DV")
    di = di.set_id_column("ID")
    base_model = base_model.replace(datainfo=di)

    base_model = convert_model(base_model, to_format="nonmem")
    return base_model


def samba_task_results(context, p_forward, state):
    # set p_backward and strictness to None
    return scm_tool.task_results(context, p_forward, p_backward=None, strictness=None, state=state)


if __name__ == "__main__":
    from pharmpy.modeling import *
    from pharmpy.tools import *
    from pharmpy.workflows import execute_workflow
    import time
    #%%
    model = read_model("sambas.ctl")
    search_space = "COVARIATE?([CL, V], [WT, AGE], POW); COVARIATE?([CL,V], SEX, EXP)"
    #%%
    wf = samba_workflow(search_space=search_space, model=model, max_steps=5)

    t1 = time.perf_counter()
    res = execute_workflow(wf)
    t2 = time.perf_counter()
    print(f"runtime: t2 - t1")

    