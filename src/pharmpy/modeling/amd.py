from functools import partial

import pharmpy.tools.scm as scm
from pharmpy.results import Results
from pharmpy.tools.amd.funcs import create_start_model
from pharmpy.workflows import default_tool_database

from .common import convert_model
from .data import remove_loq_data
from .run import fit, run_tool


class AMDResults(Results):
    def __init__(self, final_model=None):
        self.final_model = final_model


def run_amd(
    dataset_path,
    modeltype='pk_oral',
    cl_init=0.01,
    vc_init=1,
    mat_init=0.1,
    search_space=None,
    lloq=None,
    order=None,
    categorical=None,
    continuous=None,
):
    """Run Automatic Model Development (AMD) tool

    Runs structural modelsearch, IIV building, and resmod

    Parameters
    ----------
    dataset_path : Model
        Path to a dataset
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
        List of continuouts covariates

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> run_amd(model)      # doctest: +SKIP

    See also
    --------
    run_iiv
    run_tool

    """

    model = create_start_model(
        dataset_path, modeltype=modeltype, cl_init=cl_init, vc_init=vc_init, mat_init=mat_init
    )
    model = convert_model(model, 'nonmem')  # FIXME: Workaround for results retrieval system

    if lloq is not None:
        remove_loq_data(model, lloq=lloq)

    default_order = ['structural', 'iivsearch', 'residual', 'allometry', 'covariates']
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

    run_funcs = []
    for section in order:
        if section == 'structural':
            func = partial(_run_modelsearch, search_space=search_space, path=db.path)
            run_funcs.append(func)
        elif section == 'iivsearch':
            func = partial(_run_iiv, path=db.path)
            run_funcs.append(func)
        elif section == 'residual':
            func = partial(_run_resmod, path=db.path)
            run_funcs.append(func)
        elif section == 'allometry':
            func = partial(_run_allometry, allometric_variable=None, path=db.path)
            run_funcs.append(func)
        elif section == 'covariates':
            if scm.have_scm() and (continuous is not None or categorical is not None):
                func = partial(
                    _run_covariates, continuous=continuous, categorical=categorical, path=db.path
                )
                run_funcs.append(func)
        else:
            raise ValueError(
                f"Unrecognized section {section} in order. Must be one of {default_order}"
            )

    run_tool('modelfit', model, path=db.path / 'modelfit')

    next_model = model
    for func in run_funcs:
        next_model = func(next_model)

    res = AMDResults(final_model=next_model)
    return res


def _run_modelsearch(model, search_space, path):
    res_modelsearch = run_tool(
        'modelsearch',
        search_space=search_space,
        algorithm='exhaustive_stepwise',
        model=model,
        path=path / 'modelsearch',
    )
    selected_model = res_modelsearch.best_model
    return selected_model


def _run_iiv(model, path):
    res_iiv = run_tool('iivsearch', 'brute_force', iiv_strategy=2, model=model, path=path)
    selected_iiv_model = res_iiv.best_model
    return selected_iiv_model


def _run_resmod(model, path):
    res_resmod = run_tool('resmod', model, path=path / 'resmod')
    selected_model = res_resmod.best_model
    return selected_model


def _run_covariates(model, continuous, categorical, path):
    parameters = ['CL', 'VC', 'KMM', 'CLMM', 'MDT', 'MAT', 'QP1', 'QP2', 'VP1', 'VP2']

    if continuous is None:
        covariates = categorical
    elif categorical is None:
        covariates = continuous
    else:
        covariates = continuous + categorical

    relations = dict()
    for p in parameters:
        if model.statements.find_assignment(p):
            expr = model.statements.before_odes.full_expression(p)
            for eta in model.random_variables.etas:
                if eta.symbol in expr.free_symbols:
                    relations[p] = covariates
                    break
    res = scm.run_scm(model, relations, continuous=continuous, categorical=categorical, path=path)
    return res.final_model


def _run_allometry(model, allometric_variable, path):
    if allometric_variable is None:
        for col in model.datainfo:
            if col.descriptor == 'body weight':
                allometric_variable = col.name
                break

    if allometric_variable is not None:
        res = run_tool(
            'allometry', model, allometric_variable=allometric_variable, path=path / 'allometry'
        )
        return res
