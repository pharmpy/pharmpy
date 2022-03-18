from functools import partial

import pharmpy.tools.scm as scm
from pharmpy.results import Results
from pharmpy.tools.amd.funcs import create_start_model
from pharmpy.workflows import default_tool_database

from .allometry import add_allometry
from .common import convert_model, read_model, write_model
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
    mfl=None,
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
    mfl : str
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

    default_order = ['structural', 'iiv', 'residual', 'allometry', 'covariates']
    if order is None:
        order = default_order

    if mfl is None:
        if modeltype == 'pk_oral':
            mfl = (
                'ABSORPTION([ZO,SEQ-ZO-FO]);'
                'ELIMINATION([MM,MIX-FO-MM]);'
                'LAGTIME();'
                'TRANSITS([1,3,10],*);'
                'PERIPHERALS([1,2])'
            )
        else:
            mfl = 'ELIMINATION([MM,MIX-FO-MM]);' 'PERIPHERALS([1,2])'

    db = default_tool_database(toolname='amd')
    # FIXME: Workaround for nonmem<->generic eta names gets lost somewhere
    write_model(model, db.path)
    model = read_model(db.path / (model.name + '.ctl'))
    fit(model)

    run_funcs = []
    for section in order:
        if section == 'structural':
            func = partial(_run_modelsearch, mfl=mfl, path=db.path)
            run_funcs.append(func)
        elif section == 'iiv':
            func = partial(_run_iiv, path=db.path)
            run_funcs.append(func)
        elif section == 'residual':
            func = partial(_run_resmod, path=db.path)
            run_funcs.append(func)
        elif section == 'allometry':
            func = partial(_run_allometry, allometric_variable=None)
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


def _run_modelsearch(model, mfl, path):
    res_modelsearch = run_tool(
        'modelsearch', 'exhaustive_stepwise', mfl=mfl, model=model, path=path / 'modelsearch'
    )
    selected_model = res_modelsearch.best_model
    return selected_model


def _run_iiv(model, path):
    res_iiv = run_iiv(model, path=path)
    selected_iiv_model = res_iiv.best_model
    return selected_iiv_model


def _run_resmod(model, path):
    skip = []

    res_resmod = run_tool('resmod', model, path=path / 'resmod1')
    selected_model = res_resmod.best_model
    name = res_resmod.selected_model_name
    if name == 'base':
        return selected_model
    elif name[:12] == 'time_varying':
        skip.append('time_varying')
    else:
        skip.append(name)

    res_resmod = run_tool('resmod', selected_model, skip=skip, path=path / 'resmod2')
    selected_model = res_resmod.best_model
    name = res_resmod.selected_model_name
    if name == 'base':
        return selected_model
    elif name[:12] == 'time_varying':
        skip.append('time_varying')
    else:
        skip.append(name)

    res_resmod = run_tool('resmod', selected_model, skip=skip, path=path / 'resmod3')
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


def _run_allometry(model, allometric_variable):
    if allometric_variable is None:
        for col in model.datainfo:
            if col.descriptor == 'body weight':
                allometric_variable = col.name
                break

    if allometric_variable is not None:
        add_allometry(model, allometric_variable=allometric_variable)
        model.name = "scaled_model"
        fit(model)
    return model


def run_iiv(model, add_iivs=False, iiv_as_fullblock=False, rankfunc='ofv', cutoff=None, path=None):
    """Run IIV tool

    Runs two IIV workflows: testing the number of etas and testing which block structure

    Parameters
    ----------
    model : Model
        Pharmpy model
    add_iivs : bool
        Whether to add IIV on structural parameters. Default is False
    iiv_as_fullblock : bool
        Whether added etas should be as a fullblock. Default is False
    rankfunc : str
        Which ranking function should be used (OFV, AIC, BIC). Default is OFV
    cutoff : float
        Cutoff for which value of the ranking function that is considered significant. Default
        is 3.84
    path : Path
        Path of rundirectory

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> run_iiv(model)      # doctest: +SKIP

    See also
    --------
    run_amd
    run_tool

    """
    if path:
        path1 = path / 'iiv1'
        path2 = path / 'iiv2'
    else:
        path1 = path
        path2 = path

    res_no_of_etas = run_tool(
        'iiv',
        'brute_force_no_of_etas',
        add_iivs=add_iivs,
        iiv_as_fullblock=iiv_as_fullblock,
        rankfunc=rankfunc,
        cutoff=cutoff,
        model=model,
        path=path1,
    )
    res_block_structure = run_tool(
        'iiv',
        'brute_force_block_structure',
        rankfunc=rankfunc,
        cutoff=cutoff,
        model=res_no_of_etas.best_model,
        path=path2,
    )

    from pharmpy.modeling import summarize_modelfit_results

    summary_models = summarize_modelfit_results(
        [model] + res_no_of_etas.models + res_block_structure.models
    )

    from pharmpy.tools.iiv.tool import IIVResults

    res = IIVResults(
        summary_tool=[res_no_of_etas.summary_tool, res_block_structure.summary_tool],
        summary_models=summary_models,
        best_model=res_block_structure.best_model,
        models=res_no_of_etas.models + res_block_structure.models,
        start_model=model,
    )

    return res
