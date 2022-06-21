from functools import partial

import pandas as pd
from sympy import Symbol

import pharmpy.plugins as plugins
from pharmpy.modeling.common import convert_model
from pharmpy.modeling.data import remove_loq_data
from pharmpy.modeling.results import summarize_errors, write_results
from pharmpy.results import Results
from pharmpy.workflows import default_tool_database

from .run import fit, run_tool


class AMDResults(Results):
    def __init__(
        self,
        final_model=None,
        summary_tool=None,
        summary_models=None,
        summary_individuals_count=None,
        summary_errors=None,
    ):
        self.final_model = final_model
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals_count = summary_individuals_count
        self.summary_errors = summary_errors


def run_amd(
    input,
    modeltype='pk_oral',
    cl_init=0.01,
    vc_init=1,
    mat_init=0.1,
    search_space=None,
    lloq=None,
    order=None,
    categorical=None,
    continuous=None,
    allometric_variable=None,
    occasion=None,
):
    """Run Automatic Model Development (AMD) tool

    Runs structural modelsearch, IIV building, and resmod

    Parameters
    ----------
    input : Model
        Read model object/Path to a dataset
    modeltype : str
        Type of model to build. Either 'pk_oral' or 'pk_iv'
    cl_init : float
        Initial estimate for the population clearance
    vc_init : float
        Initial estimate for the central compartment population volume
    mat_init : float
        Initial estimate for the mean absorption time (not for iv models)
    search_space : str
        MFL for search space for structural model
    lloq : float
        Lower limit of quantification. LOQ data will be removed.
    order : list
        Runorder of components
    categorical : list
        List of categorical covariates
    continuous : list
        List of continuous covariates
    allometric_variable: str or Symbol
        Variable to use for allometry
    occasion: str
        Name of occasion column

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> from pharmpy.tools import run_amd # doctest: +SKIP
    >>> run_amd(model)      # doctest: +SKIP

    See also
    --------
    run_iiv
    run_tool

    """
    if type(input) is str:
        from pharmpy.tools.amd.funcs import create_start_model

        model = create_start_model(
            input, modeltype=modeltype, cl_init=cl_init, vc_init=vc_init, mat_init=mat_init
        )
        model = convert_model(model, 'nonmem')  # FIXME: Workaround for results retrieval system
    elif type(input) is plugins.nonmem.model.Model:
        model = input
        model.name = 'start'
    else:
        raise TypeError('Only model and dataset are supported currently')

    if lloq is not None:
        remove_loq_data(model, lloq=lloq)

    default_order = ['structural', 'iivsearch', 'iovsearch', 'residual', 'allometry', 'covariates']
    if order is None:
        order = default_order

    if search_space is None:
        if modeltype == 'pk_oral':
            search_space = (
                'ABSORPTION(SEQ-ZO-FO);'
                'ELIMINATION([MM,MIX-FO-MM]);'
                'LAGTIME();'
                'TRANSITS([1,3,10],*);'
                'PERIPHERALS(1)'
            )
        else:
            search_space = 'ELIMINATION([MM,MIX-FO-MM]);' 'PERIPHERALS([1,2])'

    db = default_tool_database(toolname='amd')
    fit(model)
    run_subfuncs = dict()
    for section in order:
        if section == 'structural':
            func = partial(_run_modelsearch, search_space=search_space, path=db.path)
            run_subfuncs['modelsearch'] = func
        elif section == 'iivsearch':
            func = partial(_run_iiv, path=db.path)
            run_subfuncs['iivsearch'] = func
        elif section == 'iovsearch':
            func = partial(_run_iov, occasion=occasion, path=db.path)
            run_subfuncs['iovsearch'] = func
        elif section == 'residual':
            func = partial(_run_resmod, path=db.path)
            run_subfuncs['resmod'] = func
        elif section == 'allometry':
            func = partial(_run_allometry, allometric_variable=allometric_variable, path=db.path)
            run_subfuncs['allometry'] = func
        elif section == 'covariates':
            func = partial(
                _run_covariates, continuous=continuous, categorical=categorical, path=db.path
            )
            run_subfuncs['covsearch'] = func
        else:
            raise ValueError(
                f"Unrecognized section {section} in order. Must be one of {default_order}"
            )

    run_tool('modelfit', model, path=db.path / 'modelfit')
    next_model = model
    sum_tools, sum_models, sum_inds_counts, sum_amd = [], [], [], []
    for func in run_subfuncs.values():
        subresults = func(next_model)
        next_model = subresults.best_model
        if hasattr(subresults, 'summary_tool'):
            sum_tools.append(subresults.summary_tool.reset_index()),
        sum_models.append(subresults.summary_models.reset_index()),
        sum_inds_counts.append(subresults.summary_individuals_count.reset_index()),

    for sums in [sum_tools, sum_models, sum_inds_counts]:
        sums = pd.concat(
            sums, keys=list(run_subfuncs.keys()), names=['tool', 'default index']
        ).reset_index()
        if 'step' in sums.columns:
            sums['step'] = sums['step'].fillna(1).astype('int64')
        else:
            sums['step'] = 1

        sums.set_index(['tool', 'step', 'model'], inplace=True)
        sums.drop('default index', axis=1, inplace=True)
        sum_amd.append(sums)

    summary_errors = summarize_errors(next_model)
    res = AMDResults(
        final_model=next_model,
        summary_tool=sum_amd[0],
        summary_models=sum_amd[1],
        summary_individuals_count=sum_amd[2],
        summary_errors=summary_errors,
    )
    write_results(results=res, path=db.path / 'results.json')
    write_results(results=res, path=db.path / 'results.csv', csv=True)
    return res


def _run_modelsearch(model, search_space, path):
    res_modelsearch = run_tool(
        'modelsearch',
        search_space=search_space,
        algorithm='reduced_stepwise',
        model=model,
        path=path / 'modelsearch',
    )
    return res_modelsearch


def _run_iiv(model, path):
    res_iiv = run_tool(
        'iivsearch', 'brute_force', iiv_strategy='fullblock', model=model, path=path / 'iivsearch'
    )
    return res_iiv


def _run_resmod(model, path):
    res_resmod = run_tool('resmod', model, path=path / 'resmod')
    return res_resmod


def _run_covariates(model, continuous, categorical, path):
    if continuous is None:
        continuous = []
        for col in model.datainfo:
            if col.type == 'covariate' and col.continuous is True:
                continuous.append(col.name)
    con_covariates = [Symbol(item) for item in continuous]

    if categorical is None:
        categorical = []
        for col in model.datainfo:
            if col.type == 'covariate' and col.continuous is False:
                categorical.append(col.name)
    cat_covariates = [Symbol(item) for item in categorical]

    if continuous is not None or categorical is not None:
        covariates_search_space = (
            f'CONTINUOUS({con_covariates}); CATEGORICAL({cat_covariates})\n'
            f'COVARIATE(@IIV, @CONTINUOUS, exp, *)\n'
            f'COVARIATE(@IIV, @CATEGORICAL, cat, *)'
        )
        res_cov = run_tool(
            'covsearch', covariates_search_space, model=model, path=path / 'covsearch'
        )
        return res_cov


def _run_allometry(model, allometric_variable, path):
    if allometric_variable is None:
        for col in model.datainfo:
            if col.descriptor == 'body weight':
                allometric_variable = col.name
                break

    if allometric_variable is not None:
        res_allometry = run_tool(
            'allometry', model, allometric_variable=allometric_variable, path=path / 'allometry'
        )
        return res_allometry


def _run_iov(model, occasion, path):
    if occasion is not None:
        res_iov = run_tool('iovsearch', model=model, column=occasion, path=path / 'iovsearch')
        return res_iov
