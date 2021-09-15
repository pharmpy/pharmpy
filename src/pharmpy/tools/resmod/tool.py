import pandas as pd
import sympy

import pharmpy.model
import pharmpy.tools
from pharmpy import Parameter, Parameters, RandomVariable, RandomVariables
from pharmpy.modeling import set_iiv_on_ruv, set_power_on_ruv
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.tools.modelfit import create_multiple_fit_workflow
from pharmpy.tools.workflows import Task, Workflow

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
    wf.add_task(task_iiv, predecessors=start_task)
    task_power = Task('create_power_model', _create_power_model)
    wf.add_task(task_power, predecessors=start_task)

    fit_wf = create_multiple_fit_workflow(n=3)
    wf.insert_workflow(fit_wf, predecessors=[task_base_model, task_iiv, task_power])

    task_results = Task('results', post_process)
    wf.add_task(task_results, predecessors=fit_wf.output_tasks)
    return wf


def start(model):
    return model


def post_process(*models):
    res = calculate_results(
        base_model=_find_model(models, 'base'),
        iiv_on_ruv=_find_model(models, 'iiv_on_ruv'),
        power=_find_model(models, 'power'),
    )
    return res


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
    return base_model


def _create_iiv_on_ruv_model(input_model):
    base_model = _create_base_model(input_model)  # FIXME: could be done only once in the workflow
    model = base_model.copy()
    #    model.database = base_model.database  # FIXME: Should be unnecessary
    set_iiv_on_ruv(model)
    model.name = 'iiv_on_ruv'
    return model


def _create_power_model(input_model):
    base_model = _create_base_model(input_model)  # FIXME: could be done only once in the workflow
    model = base_model.copy()
    model.individual_prediction_symbol = sympy.Symbol('IPRED')
    set_power_on_ruv(model)
    model.name = 'power'
    return model


def _create_dataset(input_model):
    residuals = input_model.modelfit_results.residuals
    cwres = residuals['CWRES']
    predictions = input_model.modelfit_results.predictions
    ipred = predictions['IPRED'].reindex(cwres.index)
    df = pd.concat([cwres, ipred], axis=1).rename(columns={'CWRES': 'DV'})
    return df
