from functools import partial

import pandas as pd
import sympy
from scipy.stats import chi2

import pharmpy.model
import pharmpy.tools
from pharmpy import Parameter, Parameters, RandomVariable, RandomVariables
from pharmpy.estimation import EstimationStep, EstimationSteps
from pharmpy.modeling import (
    add_time_after_dose,
    create_symbol,
    get_mdv,
    set_combined_error_model,
    set_iiv_on_ruv,
    set_power_on_ruv,
)
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow

from ...modeling.error import remove_error_model, set_time_varying_error_model
from .results import calculate_results


def create_workflow(model=None, groups=4, p_value=0.05, skip=None):
    cutoff = float(chi2.isf(q=p_value, df=1))
    if skip is None:
        skip = []
    wf = Workflow()
    wf.name = "resmod"  # FIXME: Could have as input to Workflow

    if model is not None:
        start_task = Task('start_resmod', start, model)
    else:
        start_task = Task('start_resmod', start)

    task_base_model = Task('create_base_model', _create_base_model)
    wf.add_task(task_base_model, predecessors=start_task)

    tasks = []
    if 'IIV_on_RUV' not in skip:
        task_iiv = Task('create_iiv_on_ruv_model', _create_iiv_on_ruv_model)
        tasks.append(task_iiv)
        wf.add_task(task_iiv, predecessors=task_base_model)

    if 'power' not in skip and 'combined' not in skip:
        task_power = Task('create_power_model', _create_power_model)
        wf.add_task(task_power, predecessors=task_base_model)
        tasks.append(task_power)
        task_combined = Task('create_combined_error_model', _create_combined_model)
        wf.add_task(task_combined, predecessors=task_base_model)
        tasks.append(task_combined)

    if 'time_varying' not in skip:
        for i in range(1, groups):
            tvar = partial(_create_time_varying_model, groups=groups, i=i)
            task = Task(f"create_time_varying_model{i}", tvar)
            tasks.append(task)
            wf.add_task(task, predecessors=task_base_model)

    fit_wf = create_fit_workflow(n=1 + len(tasks))
    wf.insert_workflow(fit_wf, predecessors=[task_base_model] + tasks)
    post_pro = partial(post_process, cutoff=cutoff)
    task_post_process = Task('post_process', post_pro)
    wf.add_task(task_post_process, predecessors=[start_task] + fit_wf.output_tasks)

    task_unpack = Task('unpack', _unpack)
    wf.add_task(task_unpack, predecessors=[task_post_process])

    fit_final = create_fit_workflow(n=1)
    wf.insert_workflow(fit_final, predecessors=[task_unpack])
    _results = partial(_compare_full_models_results, cutoff=cutoff)
    task_results = Task('results', _results)
    wf.add_task(
        task_results, predecessors=[start_task] + fit_final.output_tasks + [task_post_process]
    )

    return wf


def start(model):
    return model


def post_process(start_model, *models, cutoff):
    res = calculate_results(
        base_model=_find_models(models)[0],
        tvar_models=_find_models(models)[1],
        other_models=_find_models(models)[2],
    )
    best_model = _create_best_model(start_model, res, cutoff=cutoff)
    res.best_model = best_model[0]
    res.selected_model_name = best_model[1]
    return res


def _unpack(res):
    return res.best_model


def _compare_full_models_results(start_model, best_resmod, res, cutoff):
    delta_ofv = start_model.modelfit_results.ofv - best_resmod.modelfit_results.ofv

    if delta_ofv <= cutoff:
        res.best_model = start_model
        res.selected_model_name = 'base'
    return res


def _find_models(models):
    base_model = None
    tvar_models = []
    other_models = []
    for model in models:
        if model.name == 'base':
            base_model = model
        elif model.name[:12] == 'time_varying':
            tvar_models.append(model)
        else:
            other_models.append(model)
    return base_model, tvar_models, other_models


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
    base_model.observation_transformation = y.symbol
    base_model.name = 'base'
    base_model.dataset = _create_dataset(input_model)
    base_model.database = input_model.database

    est = EstimationStep('foce', interaction=True)
    base_model.estimation_steps = EstimationSteps([est])
    return base_model


def _create_iiv_on_ruv_model(input_model):
    base_model = input_model
    model = base_model.copy()
    set_iiv_on_ruv(model)
    model.name = 'IIV_on_RUV'
    return model


def _create_power_model(input_model):
    base_model = input_model
    model = base_model.copy()
    set_power_on_ruv(model, ipred='IPRED', lower_limit=None, zero_protection=True)
    model.name = 'power'
    return model


def _create_time_varying_model(input_model, groups, i):
    base_model = input_model
    model = base_model.copy()
    quantile = i / groups
    cutoff = model.dataset['TAD'].quantile(q=quantile)
    set_time_varying_error_model(model, cutoff=cutoff, idv='TAD')
    model.name = f"time_varying{i}"
    return model


def _create_combined_model(input_model):
    base_model = input_model
    model = base_model.copy()
    remove_error_model(model)
    s = model.statements[0]
    ruv_prop = create_symbol(model, 'epsilon_p')
    ruv_add = create_symbol(model, 'epsilon_a')
    ipred = sympy.Symbol('IPRED')
    s.expression = s.expression + ruv_prop + ruv_add / ipred

    sigma_prop = Parameter('sigma_prop', 1, lower=0)
    model.parameters.append(sigma_prop)
    model.dataset['IPRED'].replace(0, 2.225e-307, inplace=True)
    ipred_min = model.dataset['IPRED'].min()
    sigma_add_init = ipred_min / 2
    sigma_add = Parameter('sigma_add', sigma_add_init, lower=0)
    model.parameters.append(sigma_add)

    eps_prop = RandomVariable.normal(ruv_prop.name, 'ruv', 0, sigma_prop.symbol)
    model.random_variables.append(eps_prop)
    eps_add = RandomVariable.normal(ruv_add.name, 'ruv', 0, sigma_add.symbol)
    model.random_variables.append(eps_add)

    model.name = 'combined'
    return model


def _create_dataset(input_model):
    input_model = input_model.copy()
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
    mdv = mdv.reset_index(drop=True)
    label_id = input_model.datainfo.id_column.name
    input_id = input_model.dataset[label_id].astype('int64').squeeze().reset_index(drop=True)
    _time_after_dose(input_model)
    tad = input_model.dataset['TAD'].squeeze().reset_index(drop=True)
    df = pd.concat([mdv, input_id, tad, ipred], axis=1)
    df = df[df['MDV'] == 0].reset_index(drop=True)
    df = pd.concat([df, cwres], axis=1).rename(columns={'CWRES': 'DV', ipredcol: 'IPRED'})
    return df


def _time_after_dose(model):
    if 'TAD' in model.dataset:
        pass
    else:
        add_time_after_dose(model)
    return model


def _create_best_model(model, res, groups=4, cutoff=3.84):
    model = model.copy()
    model.name = 'best_resmod'
    selected_model_name = 'base'
    if any(res.models['dofv'] > cutoff):
        idx = res.models['dofv'].idxmax()
        name = idx[0]

        if name == 'power':
            set_power_on_ruv(model)
            model.parameters.inits = {
                'power1': res.models['parameters'].loc['power', 1, 1].get('theta') + 1
            }
        elif name == 'IIV_on_RUV':
            set_iiv_on_ruv(model)
            model.parameters.inits = {
                'IIV_RUV1': res.models['parameters'].loc['IIV_on_RUV', 1, 1].get('omega')
            }
        elif name[:12] == 'time_varying':
            _time_after_dose(model)
            i = int(name[-1])
            quantile = i / groups
            df = _create_dataset(model)
            tad = df['TAD']
            cutoff_tvar = tad.quantile(q=quantile)
            set_time_varying_error_model(model, cutoff=cutoff_tvar, idv='TAD')
            model.parameters.inits = {
                'time_varying': res.models['parameters']
                .loc[f"time_varying{i}", 1, 1]
                .get(f"theta_tvar{i}")
            }
        else:
            set_combined_error_model(model)
            model.parameters.inits = {
                'sigma_prop': res.models['parameters'].loc['combined', 1, 1].get('sigma_prop'),
                'sigma_add': res.models['parameters'].loc['combined', 1, 1].get('sigma_add'),
            }
        selected_model_name = name
        model.update_source()
    return model, selected_model_name
