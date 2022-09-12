import warnings
from functools import partial

import pharmpy.plugins as plugins
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.modeling.common import convert_model
from pharmpy.modeling.data import remove_loq_data
from pharmpy.modeling.eta_additions import get_occasion_levels
from pharmpy.modeling.results import summarize_errors, write_results
from pharmpy.workflows import default_tool_database

from ..run import fit, run_tool
from .results import AMDResults


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

    Runs structural modelsearch, IIV building, and ruvsearch

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
        raise TypeError('Only NONMEM model or standalone dataset are supported currently')

    if lloq is not None:
        remove_loq_data(model, lloq=lloq)

    default_order = ['structural', 'iivsearch', 'residual', 'iovsearch', 'allometry', 'covariates']
    if order is None:
        order = default_order

    if search_space is None:
        if modeltype == 'pk_oral':
            search_space = (
                'ABSORPTION([ZO,SEQ-ZO-FO]);'
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
            func = partial(_run_ruvsearch, path=db.path)
            run_subfuncs['ruvsearch'] = func
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
        if subresults is None:
            sum_tools.append(None)
            sum_models.append(None)
            sum_inds_counts.append(None)
        else:
            next_model = subresults.best_model
            if hasattr(subresults, 'summary_tool'):
                sum_tools.append(subresults.summary_tool.reset_index()),
            else:
                sum_tools.append(None)
            sum_models.append(subresults.summary_models.reset_index()),
            sum_inds_counts.append(subresults.summary_individuals_count.reset_index()),

    for sums in [sum_tools, sum_models, sum_inds_counts]:
        filtered_results = list(
            zip(*filter(lambda t: t[1] is not None, zip(list(run_subfuncs.keys()), sums)))
        )

        if not filtered_results:
            sum_amd.append(None)
            continue

        sums = pd.concat(
            filtered_results[1], keys=list(filtered_results[0]), names=['tool', 'default index']
        ).reset_index()
        if 'step' in sums.columns:
            sums['step'] = sums['step'].fillna(1).astype('int64')
        else:
            sums['step'] = 1

        sums.set_index(['tool', 'step', 'model'], inplace=True)
        sums.drop('default index', axis=1, inplace=True)
        sum_amd.append(sums)

    summary_tool, summary_models, summary_individuals_count = sum_amd

    if summary_tool is None:
        warnings.warn(
            'AMDResults.summary_tool is None because none of the tools yielded a summary.'
        )

    if summary_models is None:
        warnings.warn(
            'AMDResults.summary_models is None because none of the tools yielded a summary.'
        )

    if summary_individuals_count is None:
        warnings.warn(
            'AMDResults.summary_individuals_count is None because none of the tools yielded '
            'a summary.'
        )

    summary_errors = summarize_errors(next_model)
    res = AMDResults(
        final_model=next_model,
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_individuals_count=summary_individuals_count,
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


def _run_ruvsearch(model, path):
    res_ruvsearch = run_tool('ruvsearch', model, path=path / 'ruvsearch')
    return res_ruvsearch


def _run_covariates(model, continuous, categorical, path):
    if continuous is None:
        continuous = []
        for col in model.datainfo:
            if col.type == 'covariate' and col.continuous is True:
                continuous.append(col.name)
    con_covariates = [sympy.Symbol(item) for item in continuous]

    if categorical is None:
        categorical = []
        for col in model.datainfo:
            if col.type == 'covariate' and col.continuous is False:
                categorical.append(col.name)
    cat_covariates = [sympy.Symbol(item) for item in categorical]

    if not continuous and not categorical:
        warnings.warn(
            'Skipping COVsearch because continuous and/or categorical are None'
            ' and could not be inferred through .datainfo via "covariate" type'
            ' and "continuous" flag.'
        )
        return None

    covariates_search_space = (
        f'LET(CONTINUOUS, {con_covariates}); LET(CATEGORICAL, {cat_covariates})\n'
        f'COVARIATE(@IIV, @CONTINUOUS, exp, *)\n'
        f'COVARIATE(@IIV, @CATEGORICAL, cat, *)'
    )
    res_cov = run_tool('covsearch', covariates_search_space, model=model, path=path / 'covsearch')
    return res_cov


def _run_allometry(model, allometric_variable, path):
    if allometric_variable is None:
        for col in model.datainfo:
            if col.descriptor == 'body weight':
                allometric_variable = col.name
                break

    if allometric_variable is None:
        warnings.warn(
            'Skipping Allometry because allometric_variable is None and could'
            ' not be inferred through .datainfo via "body weight" descriptor.'
        )
        return None

    res_allometry = run_tool(
        'allometry', model, allometric_variable=allometric_variable, path=path / 'allometry'
    )
    return res_allometry


def _run_iov(model, occasion, path):
    if occasion is None:
        warnings.warn('Skipping IOVsearch because occasion is None.')
        return None

    if occasion not in model.dataset:
        # TODO Check this upstream and raise instead of warn
        warnings.warn(f'Skipping IOVsearch because dataset is missing column "{occasion}".')
        return None

    categories = get_occasion_levels(model.dataset, occasion)
    if len(categories) < 2:
        warnings.warn(
            f'Skipping IOVsearch because there are less than two '
            f'occasion categories in column "{occasion}": {categories}.'
        )
        return None

    res_iov = run_tool('iovsearch', model=model, column=occasion, path=path / 'iovsearch')
    return res_iov
