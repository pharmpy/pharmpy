from typing import Optional

from pharmpy.basic import Expr
from pharmpy.model import Assignment, Model, Statements
from pharmpy.modeling import (
    add_estimation_step,
    add_predictions,
    get_observations,
    get_omegas,
    get_sigmas,
    remove_parameter_uncertainty_step,
    set_estimation_step,
    set_initial_estimates,
)
from pharmpy.modeling.estimation_steps import add_derivative
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults

from .results import calculate_results


def create_workflow(
    model: Optional[Model] = None,
    results: Optional[ModelfitResults] = None,
    model_name: str = "linbase",
    description: str = "",
):
    """
    Linearize a model

    Parameters
    ----------
    model : Model
        Pharmpy model.
    results : ModelfitResults
        Results of estimation of model
    model_name : str, optional
        New name of linearized model. The default is "linbase".
    description : str, optional
        Description of linearized model. The default is "".

    Returns
    -------
    LinearizeResults
        Linearize tool results object.

    """
    wb = WorkflowBuilder(name="linearize")

    if model is not None:
        start_task = Task('start_linearize', start_linearize, model, results)
    else:
        start_task = Task('start_linearize', start_linearize)

    wb.add_task(start_task)

    # No need to run input model before adding derivatives
    der_task = Task("create_derivative_model", create_derivative_model)
    wb.add_task(der_task, predecessors=wb.output_tasks)

    wf_fit_deriv = create_fit_workflow(n=1)
    wb.insert_workflow(wf_fit_deriv, predecessors=[der_task])
    fit_deriv_task = wb.output_tasks

    lin_task = Task(
        "create_linearized_model", _create_linearized_model, model_name, description, model
    )
    wb.add_task(lin_task, predecessors=fit_deriv_task)

    wf_fit_lin = create_fit_workflow(n=1)
    wb.insert_workflow(wf_fit_lin, predecessors=[lin_task])
    fit_lin_task = wb.output_tasks

    postprocess_task = Task("results", postprocess, model_name)
    wb.add_task(postprocess_task, predecessors=fit_deriv_task + fit_lin_task)

    return Workflow(wb)


def start_linearize(context, model, results):
    context.log_info("Starting linearize")
    start_model_entry = ModelEntry.create(model=model, modelfit_results=results)

    # Create links to input model
    context.store_input_model_entry(start_model_entry)

    return start_model_entry


def postprocess(context, model_name, *modelentries):
    for me in modelentries:
        if me.model.name == model_name:
            linbase = me
        else:
            base = me

    res = calculate_results(
        base.model, base.modelfit_results, linbase.model, linbase.modelfit_results
    )

    context.log_info(f"OFV of input model:                {res.ofv['base']:.3f}")
    context.log_info(f"OFV of evaluated linearized model: {res.ofv['lin_evaluated']:.3f}")
    context.log_info(f"OFV of estimated linearized model: {res.ofv['lin_estimated']:.3f}")

    res.to_csv(context.path / "results.csv")

    # Create links to final model
    context.store_final_model_entry(res.final_model)

    context.log_info("Finishing linearize")
    return res


def create_derivative_model(context, modelentry):
    der_model = modelentry.model.replace(name="derivatives")
    if (
        modelentry.modelfit_results is not None
        and modelentry.modelfit_results.parameter_estimates is not None
    ):
        der_model = set_initial_estimates(
            der_model, modelentry.modelfit_results.parameter_estimates
        )
    if (
        modelentry.modelfit_results is not None
        and modelentry.modelfit_results.individual_estimates is not None
    ):
        der_model = der_model.replace(
            initial_individual_estimates=modelentry.modelfit_results.individual_estimates
        )
    der_model = add_predictions(der_model, ["CIPREDI"])
    der_model = add_derivative(der_model)
    der_model = set_estimation_step(der_model, "FOCE", 0, evaluation=True)
    context.log_info("Running derivative model")
    der_model = remove_parameter_uncertainty_step(der_model)
    return ModelEntry.create(model=der_model)


def _create_linearized_model(context, model_name, description, model, derivative_model_entry):
    if derivative_model_entry.modelfit_results is None:
        context.abort_workflow("Error while running the derivative model")
    df = cleanup_columns(derivative_model_entry)

    derivative_model = derivative_model_entry.model
    linbase = Model.create(
        parameters=get_omegas(derivative_model) + get_sigmas(derivative_model),
        random_variables=derivative_model_entry.model.random_variables,
        dependent_variables={list(derivative_model_entry.model.dependent_variables.keys())[0]: 1},
        dataset=df,
        name=model_name,
        description=description,
    )
    di = linbase.datainfo
    di = di.set_dv_column("DV")
    di = di.set_id_column("ID")
    di = di.set_idv_column("TIME")
    linbase = linbase.replace(
        datainfo=di,
        initial_individual_estimates=derivative_model_entry.modelfit_results.individual_estimates,
    )

    linbase = set_initial_estimates(
        linbase, derivative_model_entry.modelfit_results.parameter_estimates, strict=False
    )
    linbase = add_estimation_step(linbase, "FOCE", maximum_evaluations=999999, interaction=True)

    statements = _create_linearized_model_statements(linbase, model)
    linbase = linbase.replace(statements=statements)

    context.log_info("Running linearized model")
    return ModelEntry.create(model=linbase)


def _create_linearized_model_statements(linbase, model):
    ms = []
    base_terms_sum = 0
    for i, eta in enumerate(model.random_variables.etas.names, start=1):
        deta = Expr.symbol(f"D_ETA_{i}")
        oeta = Expr.symbol(f"OETA_{i}")
        base = Assignment(Expr.symbol(f'BASE{i}'), deta * (Expr.symbol(eta) - oeta))
        ms.append(base)
        base_terms_sum += base.symbol

    base_terms = Assignment(Expr.symbol('BASE_TERMS'), base_terms_sum)
    ms.append(base_terms)
    ipred = Assignment(Expr.symbol('IPRED'), Expr.symbol('OPRED') + base_terms.symbol)
    ms.append(ipred)

    i = 1
    err_terms_sum = 0
    for epsno, eps in enumerate(model.random_variables.epsilons, start=1):
        err = Assignment(
            Expr.symbol(f'ERR{epsno}'), Expr.symbol(eps.names[0]) * Expr.symbol(f'D_EPS_{epsno}')
        )
        err_terms_sum += err.symbol
        ms.append(err)
        i += 1
        for etano, eta in enumerate(model.random_variables.etas.names, start=1):
            inter = Assignment(
                Expr.symbol(f'ERR{i}'),
                Expr.symbol(eps.names[0])
                * Expr.symbol(f'D_EPSETA_{epsno}_{etano}')
                * (Expr.symbol(eta) - Expr.symbol(f'OETA_{etano}')),
            )
            err_terms_sum += inter.symbol
            ms.append(inter)
            i += 1
    error_terms = Assignment(Expr.symbol('ERROR_TERMS'), err_terms_sum)
    ms.append(error_terms)

    # FIXME: Handle other DVs?
    y = list(model.dependent_variables.keys())[0]
    y_assignment = Assignment.create(y, ipred.symbol + error_terms.symbol)

    ms.append(y_assignment)

    return Statements(ms)


def cleanup_columns(modelentry):
    predictions = modelentry.modelfit_results.predictions
    model = modelentry.model
    if predictions is None:
        raise ValueError("Require PREDICTIONS to be calculated")
    else:
        if "CIPREDI" in predictions:
            predcol = "CIPREDI"
        elif "IPRED" in predictions:
            predcol = "IPRED"
        else:
            raise ValueError("Cannot find IPRED or CIPREDI for the input model")
        ipred = predictions[predcol]

    df = model.dataset[["ID", "TIME", "DV"]].copy()  # Assume existence
    df['OPRED'] = ipred

    derivatives = modelentry.modelfit_results.derivatives
    derivative_name_subs = {}
    for der_col in derivatives.columns:
        names = der_col.split(";")
        if 1 <= len(names) <= 2:
            derivative_name_subs[der_col] = create_derivative_name(model, names)
        else:
            if len(names) == 0:
                raise ValueError(f"Unsupported derivative {der_col} ModelfitResults object")
    derivatives = derivatives.rename(derivative_name_subs, axis=1)
    df = df.join(derivatives)

    etas = modelentry.modelfit_results.individual_estimates
    eta_name_subs = {}
    for n, eta in enumerate(model.random_variables.iiv.names, start=1):
        eta_name_subs[eta] = f"OETA_{n}"
    etas = etas.rename(eta_name_subs, axis=1)
    df = df.join(etas, on="ID")

    df = df.reset_index(drop=True)
    obs = get_observations(model, keep_index=True)
    df = df.loc[obs.index]
    return df


def create_derivative_name(model, param_list):
    param_names = ""
    param_numbers = []
    for name in param_list:
        if name in model.random_variables.etas:
            param_names += "ETA"
            param_numbers.append(str(model.random_variables.etas.names.index(name) + 1))
        elif name in model.random_variables.epsilons:
            param_names += "EPS"
            param_numbers.append(str(model.random_variables.epsilons.names.index(name) + 1))
        else:
            raise ValueError(f"Derivatives with respect to parameter {name} not supported.")

    return "D_" + param_names + "_" + "_".join(param_numbers)
