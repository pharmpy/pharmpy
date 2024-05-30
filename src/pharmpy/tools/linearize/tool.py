from typing import Optional

from pharmpy.basic import Expr
from pharmpy.deps import pandas as pd
from pharmpy.model import Assignment, DataInfo, EstimationStep, ExecutionSteps, Model, Statements
from pharmpy.modeling import (
    add_predictions,
    append_estimation_step_options,
    get_mdv,
    set_estimation_step,
    set_initial_estimates,
)
from pharmpy.modeling.estimation_steps import add_derivative
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder

from .results import calculate_results


def create_workflow(
    model: Optional[Model] = None, model_name: str = "linbase", description: str = ""
):
    """
    Run linaerization procedure

    Parameters
    ----------
    model : Model
        Pharmpy model.
    model_name : str, optional
        New name of linearized model. The default is "linbase".
    description : str, optional
        Description of linaerized model. The default is "".

    Returns
    -------
    LinearizeResults
        Linaerize tool results object.

    """
    wb = WorkflowBuilder(name="linearize")

    if model is not None:
        start_task = Task('start_linearize', start_linearize, model)
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


def start_linearize(context, model):
    start_model_entry = ModelEntry.create(model=model)

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

    res.to_csv(context.path / "results.csv")

    # Create links to final model
    context.store_final_model_entry(res.final_model)

    return res


def create_derivative_model(modelentry):
    der_model = modelentry.model.replace(name="derivative_model")
    der_model = add_derivative(der_model)
    first_es = der_model.execution_steps[0]
    der_model = set_estimation_step(der_model, first_es.method, 0, maximum_evaluations=1)
    der_model = add_predictions(der_model, ["CIPREDI"])
    return ModelEntry.create(model=der_model)


def _create_linearized_model(model_name, description, model, derivative_model_entry):
    new_input_file = cleanup_columns(derivative_model_entry)
    new_datainfo = DataInfo.create(list(new_input_file.columns))
    new_datainfo = new_datainfo.set_dv_column("DV")
    new_datainfo = new_datainfo.set_id_column("ID")
    new_datainfo = new_datainfo.set_idv_column("TIME")

    linbase = Model(
        parameters=derivative_model_entry.model.parameters,
        random_variables=derivative_model_entry.model.random_variables,
        dependent_variables={list(derivative_model_entry.model.dependent_variables.keys())[0]: 1},
        datainfo=model.datainfo,
        name=model_name,
        description=description,
    )
    linbase = set_initial_estimates(
        linbase, derivative_model_entry.modelfit_results.parameter_estimates
    )

    linbase = linbase.replace(dataset=new_input_file)
    linbase = linbase.replace(datainfo=new_datainfo)

    linbase = _create_linearized_model_statements(linbase, model)

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
            Expr.symbol(f'ERR_{epsno}'), Expr.symbol(eps.names[0]) * Expr.symbol(f'D_EPS_{epsno}')
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

    est = EstimationStep.create('foce', interaction=True)
    linbase = linbase.replace(execution_steps=ExecutionSteps.create([est]))
    linbase = linbase.replace(statements=Statements(ms))
    linbase = append_estimation_step_options(linbase, tool_options={"MCETA": 1000}, idx=0)

    from pharmpy.modeling import convert_model

    linbase = convert_model(linbase, "nonmem")

    return linbase


def cleanup_columns(modelentry):
    predictions = modelentry.modelfit_results.predictions
    if predictions is None:
        raise ValueError("Require PREDICTIONS to be calculated")
    else:
        pred_found = False
        for p in ["IPRED", "CIPREDI"]:
            try:
                pred_col = predictions[p]  # TODO : Allow other prediction values?
                predictions = pd.DataFrame(pred_col)
                predictions = predictions.rename({p: "OPRED"}, axis=1)
                pred_found = True
                break
            except KeyError:
                pass

        if not pred_found:
            raise ValueError("Cannot determine OPRED to use for linearized model")

    derivatives = modelentry.modelfit_results.derivatives
    amt_dv = modelentry.model.dataset[["ID", "TIME", "AMT", "DV"]]  # Assume existance
    derivative_name_subs = {}
    for der_col in derivatives.columns:
        names = der_col.split(";")
        if 1 <= len(names) <= 2:
            derivative_name_subs[der_col] = create_derivative_name(modelentry.model, names)
        else:
            if len(names) == 0:
                raise ValueError(f"Unsupported derivaitve {der_col} ModelfitResults object")
    derivatives = derivatives.rename(derivative_name_subs, axis=1)
    new_input_file = predictions.join(amt_dv).join(derivatives)

    etas = modelentry.modelfit_results.individual_estimates
    eta_name_subs = {}
    for n, eta in enumerate(modelentry.model.random_variables.iiv.names, start=1):
        eta_name_subs[eta] = f"OETA_{n}"
    etas = etas.rename(eta_name_subs, axis=1)
    new_input_file = new_input_file.join(etas, on="ID")

    new_input_file = new_input_file.reset_index(drop=True)
    new_input_file["MDV"] = get_mdv(modelentry.model)

    return new_input_file


def create_derivative_name(model, param_list):
    param_names = ""
    param_numbers = []
    for name in param_list:
        if name in model.random_variables.etas:
            param_names += "ETA"
            param_numbers.append(
                str(model.random_variables.etas.index(model.random_variables.etas[name]) + 1)
            )
        elif name in model.random_variables.epsilons:
            param_names += "EPS"
            param_numbers.append(
                str(
                    model.random_variables.epsilons.index(model.random_variables.epsilons[name]) + 1
                )
            )
        else:
            raise ValueError(f"Derivatives with respect to parameter {name} not supported.")

    return "D_" + param_names + "_" + "_".join(param_numbers)
