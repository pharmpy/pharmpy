import pandas as pd

import pharmpy.model
import pharmpy.tools
import pharmpy.tools.modelfit as modelfit
from pharmpy import Parameter, Parameters, RandomVariable, RandomVariables
from pharmpy.estimation import EstimationMethod
from pharmpy.modeling import get_mdv, set_iiv_on_ruv, set_power_on_ruv
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.tools.modelfit import create_multiple_fit_workflow
from pharmpy.workflows import Task, Workflow

from .results import calculate_results


def create_workflow(model=None):
    wf = Workflow()
    wf.name = "resmod"  # FIXME: Could have as input to Workflow

    if model is not None:
        start_task = Task('start_resmod', start, model)
    else:
        start_task = Task('start_resmod', start)

    task_base_model = Task('create_base_model', _create_base_model)
    wf.add_task(task_base_model, predecessors=start_task)

    task_iiv = Task('create_iiv_on_ruv_model', _create_iiv_on_ruv_model)
    wf.add_task(task_iiv, predecessors=task_base_model)
    task_power = Task('create_power_model', _create_power_model)
    wf.add_task(task_power, predecessors=task_base_model)

    fit_wf = create_multiple_fit_workflow(n=3)
    wf.insert_workflow(fit_wf, predecessors=[task_base_model, task_iiv, task_power])

    task_post_process = Task('post_process', post_process)
    wf.add_task(task_post_process, predecessors=[start_task] + fit_wf.output_tasks)

    task_unpack = Task('unpack', _unpack)
    wf.add_task(task_unpack, predecessors=[task_post_process])

    fit_final = modelfit.create_single_fit_workflow()
    wf.insert_workflow(fit_final, predecessors=[task_unpack])

    task_results = Task('results', _results)
    wf.add_task(task_results, predecessors=[task_post_process] + fit_final.output_tasks)

    return wf


def start(model):
    return model


def post_process(start_model, *models):
    res = calculate_results(
        base_model=_find_model(models, 'base'),
        iiv_on_ruv=_find_model(models, 'iiv_on_ruv'),
        power=_find_model(models, 'power'),
    )
    best_model = _create_best_model(start_model, res)
    res.best_model = best_model
    return res


def _results(res, best_model):
    res.best_model = best_model
    return res


def _unpack(res):
    return res.best_model


def _find_model(models, name):
    for model in models:
        if model.name == name:
            return model


def _create_base_model(input_model):
    base_model = pharmpy.model.Model()
    theta = Parameter('theta', 0.1)
    omega = Parameter('omega', 0.01, lower=0)
    sigma = Parameter('sigma', 1, lower=0)
    params = Parameters([theta, omega, sigma])
    base_model.parameters = params

    eta = RandomVariable.normal('eta', 'iiv', 0, omega.symbol)
    sigma = RandomVariable.normal('epsilon', 'ruv', 0, sigma.symbol)
    rvs = RandomVariables([eta, sigma])
    base_model.random_variables = rvs

    y = Assignment('Y', theta.symbol + eta.symbol + sigma.symbol)
    stats = ModelStatements([y])
    base_model.statements = stats

    base_model.dependent_variable = y.symbol
    base_model.name = 'base'
    base_model.dataset = _create_dataset(input_model)
    base_model.database = input_model.database

    est = EstimationMethod('foce', interaction=True)
    base_model.estimation_steps = [est]
    return base_model


def _create_iiv_on_ruv_model(input_model):
    base_model = input_model
    model = base_model.copy()
    set_iiv_on_ruv(model)
    model.name = 'iiv_on_ruv'
    return model


def _create_power_model(input_model):
    base_model = input_model
    model = base_model.copy()
    set_power_on_ruv(model, ipred='IPRED')
    model.name = 'power'
    return model


def _create_dataset(input_model):
    residuals = input_model.modelfit_results.residuals
    cwres = residuals['CWRES'].reset_index(drop=True)
    predictions = input_model.modelfit_results.predictions
    if 'CIPREDI' in predictions:
        ipredcol = 'CIPREDI'
    elif 'IPRED' in predictions:
        ipredcol = 'IPRED'
    else:
        raise ValueError("Need CIPREDI or IPRED")
    ipred = predictions[ipredcol].reset_index(drop=True)
    mdv = get_mdv(input_model)
    df_ipred = pd.concat([mdv, ipred], axis=1).rename(columns={ipredcol: 'IPRED'})
    df_ipred = df_ipred[df_ipred['MDV'] == 0].reset_index(drop=True)
    df = pd.concat([cwres, df_ipred['IPRED']], axis=1).rename(columns={'CWRES': 'DV'})
    return df


def _create_best_model(model, res):
    model = model.copy()
    if any(res.models['dofv'] > 3.84):
        idx = res.models['dofv'].idxmax()
        name = idx[0]
        if name == 'power':
            set_power_on_ruv(model)
            model.parameters.inits = {
                'power1': res.models['parameters'].loc['power', 1, 1].get('theta')
            }
        else:
            set_iiv_on_ruv(model)
            model.parameters.inits = {
                'IIV_RUV1': res.models['parameters'].loc['IIV_on_RUV', 1, 1].get('omega')
            }
        model.update_source()
    return model
