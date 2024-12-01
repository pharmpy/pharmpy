from dataclasses import replace
from functools import partial
from itertools import count

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
    add_covariate_effect,
    convert_model,
    get_parameter_rv,
    unconstrain_parameters,
)
from pharmpy.modeling.lrt import best_of_many as lrt_best_of_many
from pharmpy.modeling.lrt import best_of_two as lrt_best_of_two
from pharmpy.tools.common import update_initial_estimates
from pharmpy.tools.covsearch.util import (
    Candidate,
    DummyEffect,
    ForwardStep,
    LinStateAndEffect,
    StateAndEffect,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder


def fast_forward(
    context,
    p_forward,
    max_steps,
    algorithm,
    nsamples,
    weighted_linreg,
    statsmodels,
    lin_filter,
    state_and_effect,
):
    lin_state_and_effect = init_linear_state_and_effect(
        nsamples, algorithm, weighted_linreg, state_and_effect
    )

    steps = range(1, max_steps + 1) if max_steps >= 1 else count(1)
    search_state = state_and_effect.search_state
    for step in steps:
        prev_best = search_state.best_candidate_so_far
        if statsmodels:
            state_and_effect = statsmodels_linear_selection(
                step,
                p_forward,
                lin_state_and_effect,
                nsamples,
                lin_filter,
                algorithm,
                weighted_linreg,
            )
        else:
            state_and_effect = nonmem_linear_selection(
                context, step, p_forward, lin_state_and_effect, nsamples, lin_filter, algorithm
            )
        search_state = nonlinear_model_selection(context, step, p_forward, state_and_effect)
        new_best = search_state.best_candidate_so_far
        if new_best is prev_best:
            break
        else:
            lin_state_and_effect = replace(lin_state_and_effect, search_state=search_state)

    return search_state


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
            param_cov_model = unconstrain_parameters(
                param_cov_model, f"POP_{cov_effect[0]}{cov_effect[1]}"
            )
            description = "_".join(cov_effect[0:2])
            param_cov_model = param_cov_model.replace(description=description)
            param_cov_modelentry = ModelEntry.create(model=param_cov_model)
            linear_modelentry_dict[param].append(param_cov_modelentry)
    return linear_modelentry_dict, param_covariate_lst


def statsmodels_linear_selection(
    step,
    alpha,
    linear_state_and_effect,
    nsamples,
    lin_filter,
    algorithm,
    weighted_linreg,
):
    import statsmodels.formula.api as smf

    effect_funcs = linear_state_and_effect.effect_funcs
    search_state = linear_state_and_effect.search_state
    param_cov_list = linear_state_and_effect.param_cov_list
    best_modelentry = search_state.best_candidate_so_far.modelentry
    selected_effect_funcs = []
    selected_lin_model_ofv = []

    for param, covariates in param_cov_list.items():
        # update dataset
        updated_data = create_covmodel_dataset(
            best_modelentry, param, covariates, nsamples, algorithm
        )
        covs = ["1"] + covariates

        if "samba" in algorithm and nsamples > 1:
            linear_models = [
                smf.mixedlm(f"DV~{cov}", data=updated_data, groups=updated_data["ID"])
                for cov in covs
            ]
        elif weighted_linreg:
            linear_models = [
                smf.wls(f"DV~{cov}", data=updated_data, weights=1.0 / updated_data["ETC"])
                for cov in covs
            ]
        else:
            linear_models = [smf.ols(f"DV~{cov}", data=updated_data) for cov in covs]

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
        linear_modelentries = context.call_workflow(Workflow(wb), 'fit_linear_models')
        linear_modelentry_dict[param] = linear_modelentries

        # linear covariate model selection
        ofvs = [
            (modelentry.modelfit_results.ofv if modelentry.modelfit_results is not None else np.nan)
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


def nonlinear_model_selection(context, step, p_forward, state_and_effect):
    effect_funcs, search_state = state_and_effect.effect_funcs, state_and_effect.search_state
    best_candidate = search_state.best_candidate_so_far
    best_model = best_candidate.modelentry.model
    if effect_funcs:
        new_models = []
        for cov_effect, cov_func in effect_funcs.items():
            name = f"step {step}_NLin_" + "-".join(cov_effect[:3])
            desc = best_model.description + f";({'-'.join(cov_effect[:3])})"
            model = best_model.replace(name=name, description=desc)
            model = update_initial_estimates(model, best_candidate.modelentry.modelfit_results)
            model = cov_func(model)
            new_models.append(model)

        new_modelentries = [
            ModelEntry.create(model=model, parent=best_model) for model in new_models
        ]
        fit_wf = create_fit_workflow(modelentries=new_modelentries)
        wb = WorkflowBuilder(fit_wf)
        task_gather = Task("gather", lambda *models: models)
        wb.add_task(task_gather, predecessors=wb.output_tasks)
        new_modelentries = context.call_workflow(Workflow(wb), 'fit_nonlinear_models')
        new_candidates = [
            Candidate(me, best_candidate.steps + (ForwardStep(p_forward, DummyEffect(*effect)),))
            for me, effect in zip(new_modelentries, effect_funcs.keys())
        ]
        search_state.all_candidates_so_far.extend(new_candidates)
        ofvs = [
            me.modelfit_results.ofv if me.modelfit_results is not None else np.nan
            for me in new_modelentries
        ]
        new_best_modelentry = lrt_best_of_many(
            parent=best_candidate.modelentry,
            models=new_modelentries,
            parent_ofv=best_candidate.modelentry.modelfit_results.ofv,
            model_ofvs=ofvs,
            alpha=p_forward,
        )
        if new_best_modelentry != best_candidate.modelentry:
            context.store_model_entry(
                ModelEntry.create(
                    model=new_best_modelentry.model.replace(name=f"step {step}_selection")
                )
            )
            best_candidate_so_far = next(
                filter(
                    lambda candidate: candidate.modelentry is new_best_modelentry,
                    search_state.all_candidates_so_far,
                )
            )
            search_state = replace(search_state, best_candidate_so_far=best_candidate_so_far)
    return search_state


def create_base_covmodel(data, parameter, nsamples, weighted_linreg=False):
    if nsamples > 1:
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
            symbol=Expr.symbol("Y"),
            expression=Expr.symbol("IPRED")
            + Expr.symbol("epsilon") * Expr.sqrt(Expr.symbol("ETC")),
        )
        name = f"{parameter}_Weighted_Base"
    else:
        y = Assignment.create(
            symbol=Expr.symbol("Y"), expression=Expr.symbol("IPRED") + Expr.symbol("epsilon")
        )
        name = f"{parameter}_Base"
    statements = Statements([base, ipred, y])

    est = EstimationStep.create(
        method="FO", maximum_evaluations=99999, tool_options={"NSIG": 2, "PRINT": 1, "NOHABORT": 0}
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
    eta0_dist = NormalDistribution.create(
        name="DUMMYETA", level="iiv", mean=0, variance=omega0.symbol
    )
    eta1_dist = NormalDistribution.create(
        name="ETA_INT", level="iiv", mean=0, variance=omega1.symbol
    )
    eta2_dist = NormalDistribution.create(
        name="ETA_EPS", level="iiv", mean=0, variance=omega2.symbol
    )
    random_vars = RandomVariables.create(dists=[eps_dist, eta0_dist, eta1_dist, eta2_dist])
    # assignments
    base = Assignment.create(symbol=Expr.symbol(parameter), expression=theta.symbol)
    ipred = Assignment.create(
        symbol=Expr.symbol("IPRED"), expression=base.symbol * Expr.exp(Expr.symbol("DUMMYETA"))
    )
    y = Assignment.create(
        symbol=Expr.symbol("Y"),
        expression=Expr.symbol("IPRED")
        + Expr.symbol("ETA_INT")
        + Expr.symbol("epsilon") * Expr.exp(Expr.symbol("ETA_EPS")),
    )
    statements = Statements([base, ipred, y])
    name = f"{parameter}_Mixed_Effects_Base"
    est = EstimationStep.create(
        method="FOCE",
        maximum_evaluations=99999,
        interaction=True,
        tool_options={"NSIG": 2, "PRINT": 1, "NOHABORT": 0},
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


def create_covmodel_dataset(modelentry, param, covariates, nsamples, algorithm):

    eta_name = get_parameter_rv(modelentry.model, param)[0]
    # Extract the conditional means (ETA) for individual parameters
    if "samba" in algorithm and nsamples >= 1:
        eta_column = modelentry.modelfit_results.individual_eta_samples[eta_name]
    else:
        eta_column = modelentry.modelfit_results.individual_estimates[eta_name]
    eta_column = eta_column.rename("DV")

    # Extract the covariates dataset
    covariates = list(set(covariates))  # drop duplicated covariates
    covariate_columns = modelentry.model.dataset[["ID"] + covariates]
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
        for subset in modelentry.modelfit_results.individual_estimates_covariance
    ]
    subject_id = modelentry.modelfit_results.individual_estimates.index
    etc_columns = pd.DataFrame({"ID": subject_id, "ETC": etc})
    dataset = dataset.merge(etc_columns, on="ID")

    return dataset
