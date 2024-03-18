import re
from typing import Dict, Optional

import pharmpy.model
from pharmpy.basic import Expr
from pharmpy.deps import pandas as pd
from pharmpy.model import Assignment, DataInfo, EstimationStep, EstimationSteps, Model, Statements
from pharmpy.modeling import (
    add_iiv,
    create_joint_distribution,
    get_mdv,
    get_omegas,
    is_linearized,
    remove_iiv,
    remove_unused_parameters_and_rvs,
    unfix_parameters,
    update_inits,
)
from pharmpy.modeling.estimation_steps import add_derivative
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import (
    ModelEntry,
    Task,
    ToolDatabase,
    Workflow,
    WorkflowBuilder,
    execute_workflow,
)

from .results import calculate_results


def create_workflow(model=None, name="linbase", description="", db=None):
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
        "create_linearized_model", _create_linearized_model, name, description, db, model
    )
    wb.add_task(lin_task, predecessors=fit_deriv_task)

    wf_fit_lin = create_fit_workflow(n=1)
    wb.insert_workflow(wf_fit_lin, predecessors=[lin_task])
    fit_lin_task = wb.output_tasks

    postprocess_task = Task("results", postprocess, db.path / "results.csv")
    wb.add_task(postprocess_task, predecessors=fit_deriv_task + fit_lin_task)

    return Workflow(wb)


def start_linearize(model):
    return ModelEntry.create(model=model)


def postprocess(path, *modelentries):
    for me in modelentries:
        if me.model.name == "linbase":
            linbase = me
        else:
            base = me

    res = calculate_results(
        base.model, base.modelfit_results, linbase.model, linbase.modelfit_results
    )

    res.to_csv(path)

    return res


def create_derivative_model(modelentry):
    # Create derivative model
    der_model = modelentry.model.replace(name="derivative_model")
    der_model = add_derivative(der_model)
    return ModelEntry.create(model=der_model)


def _create_linearized_model(name, description, db, model, derivative_model_entry):
    der_table = derivative_model_entry.model.internals.control_stream.get_records("TABLE")
    filename = der_table[-1].get_option("FILE")
    if filename is None:
        filename = f"{model.name}.tab"
    filename_path = db.path / "models" / "derivative_model" / filename
    result_dataset = pd.read_csv(filename_path, skiprows=1, sep=r'\s+')
    new_input_file = cleanup_columns(result_dataset, derivative_model_entry)

    new_datainfo = DataInfo.create(list(new_input_file.columns))
    new_datainfo = new_datainfo.set_dv_column("DV")
    new_datainfo = new_datainfo.set_id_column("ID")
    new_datainfo = new_datainfo.set_idv_column("TIME")

    linbase = pharmpy.model.Model(
        parameters=derivative_model_entry.model.parameters,
        random_variables=derivative_model_entry.model.random_variables,
        datainfo=model.datainfo,
        name=name,
        description=description,
    )
    linbase = update_inits(linbase, derivative_model_entry.modelfit_results.parameter_estimates)

    linbase = linbase.replace(dataset=new_input_file)
    linbase = linbase.replace(datainfo=new_datainfo)  # CANNOT CHANGE BOTH AT THE SAME TIME

    linbase = _create_linearized_model_statements(linbase, model)

    return ModelEntry.create(model=linbase)


def _create_linearized_model_statements(linbase, model):
    ms = []
    base_terms_sum = 0
    for i, eta in enumerate(model.random_variables.etas.names, start=1):
        deta = Expr.symbol(f"D_ETA{i}")
        oeta = Expr.symbol(f"OETA{i}")
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
            Expr.symbol(f'ERR{epsno}'), Expr.symbol(eps.names[0]) * Expr.symbol(f'D_EPS{epsno}')
        )
        err_terms_sum += err.symbol
        ms.append(err)
        i += 1
        for etano, eta in enumerate(model.random_variables.etas.names, start=1):
            inter = Assignment(
                Expr.symbol(f'ERR{i}'),
                Expr.symbol(eps.names[0])
                * Expr.symbol(f'D_EPSETA{epsno}_{etano}')
                * (Expr.symbol(eta) - Expr.symbol(f'OETA{etano}')),
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
    linbase = linbase.replace(name='linbase', estimation_steps=EstimationSteps.create([est]))
    linbase = linbase.replace(statements=Statements(ms))

    from pharmpy.modeling import convert_model

    linbase = convert_model(linbase, "nonmem")

    return linbase


def cleanup_columns(table, modelentry):

    rename_columns = {}
    columns_to_drop = []
    for c in table.columns:
        if match := re.match(r'H(\d+)1', c):
            rename_columns[c] = f"D_EPS{int(match.group(1))}"
        elif match := re.match(r'G(\d+)1', c):
            rename_columns[c] = f"D_ETA{int(match.group(1))}"
        elif c.startswith("D_ETAEPS_"):
            match = re.match(r'D_ETAEPS_(\d+)_(\d+)', c)
            rename_columns[c] = f"D_EPSETA{int(match.group(2))}_{int(match.group(1))}"
        elif c.startswith("D_EPSETA_"):
            match = re.match(r'D_ETAEPS_(\d+)_(\d+)', c)
            rename_columns[c] = f"D_EPSETA{int(match.group(1))}_{int(match.group(2))}"
        elif c in ["TIME", "ID", "AMT", "DV"]:
            pass
        else:
            columns_to_drop.append(c)
    new_input_file = table.drop(columns_to_drop, axis=1)
    new_input_file = new_input_file.rename(rename_columns, axis=1)

    for attr in ["TIME", "ID", "AMT", "DV"]:
        if attr not in new_input_file.columns:
            dv_col = modelentry.model.dataset[attr]
            new_input_file[attr] = dv_col
    new_input_file = new_input_file.set_index(["ID", "TIME"])

    etas = modelentry.modelfit_results.individual_estimates
    eta_name_subs = {}
    for n, eta in enumerate(modelentry.model.random_variables.iiv.names, start=1):
        eta_name_subs[eta] = f"OETA{n}"
    etas = etas.rename(eta_name_subs, axis=1)
    new_input_file = new_input_file.join(etas, on="ID")

    pred = modelentry.modelfit_results.predictions
    if pred is None:
        raise ValueError("Require PREDICTIONS to be calculated")
    else:
        try:
            pred_col = pred["IPRED"]
            pred_col = pd.DataFrame(pred_col)
            pred_col = pred_col.rename({"IPRED": "OPRED"}, axis=1)
        except KeyError:  # HOW TO HANDLE ?
            raise ValueError("Cannot determine OPRED to use for linearized model")
        new_input_file = new_input_file.join(pred_col)

    new_input_file = new_input_file.reset_index()
    if "MDV" not in new_input_file.columns:
        new_input_file["MDV"] = get_mdv(modelentry.model)
    return new_input_file


def create_linearized_model(
    model: Model,
    name: str = "linbase",
    description: str = "",
    return_wf: bool = False,
    context: ToolDatabase = None,
):
    # TODO : Integrate within run tool once new database is in place.

    if is_linearized(model):
        return model

    from pharmpy.tools.run import _get_run_setup

    if context is None:
        dispatcher, context = _get_run_setup({}, "Linearize")
    else:
        dispatcher, _ = _get_run_setup({}, "Linearize")

    wf = create_workflow(model, name, description, context)

    if return_wf:
        return wf
    else:
        linearized_model = execute_workflow(wf, dispatcher=dispatcher, database=context)
        return (
            linearized_model  # Return model instead of ModelEntry (for users) OR a results object
        )


def delinearize_model(
    linearized_model: Model, base_model: Model, param_mapping: Optional[Dict] = None
):
    """
    Delinearize a model given a base_model to linearize to. If param_mapping is
    set, then the new model will get new ETAs based on this mapping.
    E.g param_mapping = {"ETA_1": "CL", "ETA_2": "V"}
    Otherwise, all ETAs are assumed to be the same in the both models and
    only the initial estimates will be updated.

    Parameters
    ----------
    linearized_model : Model
        Linearized model
    base_model : Model
        Model to use for the different
    param_mapping : None, Dict
        Use special mapping, given as a dict. The default is None.

    Returns
    -------
    Model.

    """

    if len(linearized_model.random_variables) != len(base_model.random_variables):
        if not param_mapping:
            raise ValueError(
                "Cannot de-linearize model with different set"
                " of random variables without param_mapping"
            )
    if param_mapping:
        # TODO : Assert all mapping parameters are in the model
        # TODO : Assert all ETAs in linearized model is in param_mapping
        dl_model = remove_iiv(base_model)  # Remove all IIV and then add based on linearized model
        for block in linearized_model.random_variables.etas:
            if len(block) > 1:
                # Add diagonal elements
                for eta in block:
                    eta_name = eta.variance
                    parameter = param_mapping[eta_name]
                    initial_estimate = linearized_model.parameters[eta_name].init
                    dl_model = add_iiv(
                        dl_model, parameter, "exp", initial_estimate=initial_estimate
                    )
                added_etas = dl_model.random_variables.etas[-len(block) :]
                added_etas_names = [eta.names[0] for eta in added_etas]

                # Create the join_normal_distribution
                dl_model = create_joint_distribution(dl_model, added_etas_names)
                new_matrix = dl_model.random_variables.etas[-1].variance
                new_initial_matrix = block.variance
                off_diagonal_updates = {}
                for row in range(1, len(added_etas)):
                    for col in range(1, row + 1):
                        param_name = new_matrix[row, col]  # STRING???
                        param_value = new_initial_matrix[row, col]
                        off_diagonal_updates[param_name] = param_value
            else:
                # Single ETA
                eta_name = block.names[0]
                eta_variance = block.variance
                parameter = param_mapping[eta_name]
                initial_estimate = linearized_model.parameters[eta_variance].init
                dl_model = add_iiv(dl_model, parameter, "exp", initial_estimate=initial_estimate)

        dl_model = unfix_parameters(dl_model, "DUMMYOMEGA")
        dl_model = remove_iiv(dl_model, "eta_dummy")
    else:
        # Assume all parameter names are the same ?
        # TODO : Raise if parameter names differ in any way
        dl_model = update_inits(base_model, get_omegas(linearized_model).inits)
    dl_model = remove_unused_parameters_and_rvs(dl_model)
    return dl_model.update_source()
