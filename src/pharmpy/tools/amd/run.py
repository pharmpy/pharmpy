import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.model import Model, Results
from pharmpy.modeling.blq import has_blq_transformation, transform_blq
from pharmpy.modeling.common import convert_model
from pharmpy.modeling.covariate_effect import get_covariates_allowed_in_covariate_effect
from pharmpy.modeling.parameter_variability import get_occasion_levels
from pharmpy.results import ModelfitResults
from pharmpy.tools import retrieve_final_model, summarize_errors, write_results
from pharmpy.tools.allometry.tool import validate_allometric_variable
from pharmpy.tools.mfl.feature.covariate import covariates as extract_covariates
from pharmpy.tools.mfl.feature.covariate import spec as covariate_spec
from pharmpy.tools.mfl.filter import covsearch_statement_types, modelsearch_statement_types
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.mfl.statement.feature.absorption import Absorption
from pharmpy.tools.mfl.statement.feature.covariate import Covariate, Ref
from pharmpy.tools.mfl.statement.feature.elimination import Elimination
from pharmpy.tools.mfl.statement.feature.lagtime import LagTime
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.feature.symbols import Name, Wildcard
from pharmpy.tools.mfl.statement.feature.transits import Transits
from pharmpy.tools.mfl.statement.statement import Statement
from pharmpy.tools.mfl.stringify import stringify as mfl_stringify
from pharmpy.workflows import default_tool_database

from ..run import run_tool
from .results import AMDResults


def run_amd(
    input: Union[Model, Path, str],
    results: Optional[ModelfitResults] = None,
    modeltype: str = 'basic_pk',
    administration: str = 'oral',
    cl_init: float = 0.01,
    vc_init: float = 1.0,
    mat_init: float = 0.1,
    search_space: Optional[str] = None,
    lloq_method: Optional[str] = None,
    lloq_limit: Optional[str] = None,
    order: Optional[List[str]] = None,
    allometric_variable: Optional[Union[str, sympy.Symbol]] = None,
    occasion: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    resume: Optional[bool] = False,
):
    """Run Automatic Model Development (AMD) tool

    Runs structural modelsearch, IIV building, and ruvsearch

    Parameters
    ----------
    input : Model or Path
        Read model object/Path to a dataset
    results : ModelfitResults
        Reults of input if input is a model
    modeltype : str
        Type of model to build. Either 'basic_pk' or 'tmdd'
    administration : str
        Route of administration. Either 'iv' or 'oral'
    cl_init : float
        Initial estimate for the population clearance
    vc_init : float
        Initial estimate for the central compartment population volume
    mat_init : float
        Initial estimate for the mean absorption time (not for iv models)
    search_space : str
        MFL for search space for structural model
    lloq_method : str
        Method for how to remove LOQ data. See `transform_blq` for list of available methods
    lloq_limit : float
        Lower limit of quantification. If None LLOQ column from dataset will be used
    order : list
        Runorder of components
    allometric_variable: str or Symbol
        Variable to use for allometry
    occasion : str
        Name of occasion column
    path : str or Path
        Path to run AMD in
    resume : bool
        Whether to allow resuming previous run

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import run_amd, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> run_amd(model, results=results)      # doctest: +SKIP

    See also
    --------
    run_iiv
    run_tool

    """
    from pharmpy.model.external import nonmem  # FIXME We should not depend on NONMEM

    # FIXME: temporary until modeltype and administration is fully supported in e.g. create_start_model
    if modeltype == 'basic_pk' and administration == 'iv':
        modeltype = 'pk_iv'
    elif modeltype == 'basic_pk' and administration == 'oral':
        modeltype = 'pk_oral'

    if type(input) is str:
        from pharmpy.tools.amd.funcs import create_start_model

        model = create_start_model(
            input, modeltype=modeltype, cl_init=cl_init, vc_init=vc_init, mat_init=mat_init
        )
        model = convert_model(model, 'nonmem')  # FIXME: Workaround for results retrieval system
    elif type(input) is nonmem.model.Model:
        model = input
        model = model.replace(name='start')
    else:
        raise TypeError(
            f'Invalid input: got `{input}` of type {type(input)},'
            f' only NONMEM model or standalone dataset are supported currently.'
        )

    if lloq_method is not None:
        model = transform_blq(
            model,
            method=lloq_method,
            lloq=lloq_limit,
        )

    default_order = ['structural', 'iivsearch', 'residual', 'iovsearch', 'allometry', 'covariates']
    if order is None:
        order = default_order

    try:
        input_search_space_features = [] if search_space is None else mfl_parse(search_space)
    except:  # noqa E722
        raise ValueError(f'Invalid `search_space`, could not be parsed: "{search_space}"')

    modelsearch_features = tuple(
        filter(
            lambda statement: isinstance(statement, modelsearch_statement_types),
            input_search_space_features,
        )
    )
    if not modelsearch_features:
        if modeltype == 'pk_oral':
            modelsearch_features = (
                Absorption((Name('ZO'), Name('SEQ-ZO-FO'))),
                Elimination((Name('MM'), Name('MIX-FO-MM'))),
                LagTime(),
                Transits((1, 3, 10), Wildcard()),
                Peripherals((1,)),
            )
        else:
            modelsearch_features = (
                Elimination((Name('MM'), Name('MIX-FO-MM'))),
                Peripherals((1, 2)),
            )

    covsearch_features = tuple(
        filter(
            lambda statement: isinstance(statement, covsearch_statement_types),
            input_search_space_features,
        )
    )
    if not any(map(lambda statement: isinstance(statement, Covariate), covsearch_features)):
        covsearch_features = (
            Covariate(Ref('IIV'), Ref('CONTINUOUS'), ('exp',), '*'),
            Covariate(Ref('IIV'), Ref('CATEGORICAL'), ('cat',), '*'),
        ) + covsearch_features

    db = default_tool_database(toolname='amd', path=path, exist_ok=resume)
    run_subfuncs = {}
    for section in order:
        if section == 'structural':
            func = _subfunc_modelsearch(search_space=modelsearch_features, path=db.path)
            run_subfuncs['modelsearch'] = func
        elif section == 'iivsearch':
            func = _subfunc_iiv(path=db.path)
            run_subfuncs['iivsearch'] = func
        elif section == 'iovsearch':
            func = _subfunc_iov(amd_start_model=model, occasion=occasion, path=db.path)
            run_subfuncs['iovsearch'] = func
        elif section == 'residual':
            func = _subfunc_ruvsearch(path=db.path)
            run_subfuncs['ruvsearch'] = func
        elif section == 'allometry':
            func = _subfunc_allometry(
                amd_start_model=model, input_allometric_variable=allometric_variable, path=db.path
            )
            run_subfuncs['allometry'] = func
        elif section == 'covariates':
            func = _subfunc_covariates(
                amd_start_model=model, search_space=covsearch_features, path=db.path
            )
            run_subfuncs['covsearch'] = func
        else:
            raise ValueError(
                f"Unrecognized section {section} in order. Must be one of {default_order}"
            )

    model = run_tool('modelfit', model, path=db.path / 'modelfit', resume=resume)
    next_model = model
    sum_subtools, sum_models, sum_inds_counts, sum_amd = [], [], [], []
    sum_subtools.append(_create_sum_subtool('start', model))
    for tool_name, func in run_subfuncs.items():
        subresults = func(next_model)
        if subresults is None:
            sum_models.append(None)
            sum_inds_counts.append(None)
        else:
            if subresults.final_model_name != next_model.name:
                next_model = retrieve_final_model(subresults)
            sum_subtools.append(_create_sum_subtool(tool_name, next_model))
            sum_models.append(subresults.summary_models.reset_index())
            sum_inds_counts.append(subresults.summary_individuals_count.reset_index())

    for sums in [sum_models, sum_inds_counts]:
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

    summary_models, summary_individuals_count = sum_amd
    summary_tool = _create_tool_summary(sum_subtools)

    if summary_models is None:
        warnings.warn(
            'AMDResults.summary_models is None because none of the tools yielded a summary.'
        )

    if summary_individuals_count is None:
        warnings.warn(
            'AMDResults.summary_individuals_count is None because none of the tools yielded '
            'a summary.'
        )

    summary_errors = summarize_errors(next_model.modelfit_results)
    res = AMDResults(
        final_model=next_model.name,
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_individuals_count=summary_individuals_count,
        summary_errors=summary_errors,
    )
    write_results(results=res, path=db.path / 'results.json')
    write_results(results=res, path=db.path / 'results.csv', csv=True)
    return res


def _create_sum_subtool(tool_name, selected_model):
    return {
        'tool': tool_name,
        'selected_model': selected_model.name,
        'description': selected_model.description,
        'n_params': len(selected_model.parameters.nonfixed),
        'ofv': selected_model.modelfit_results.ofv,
    }


def _create_tool_summary(rows):
    summary_prev = None
    rows_updated = []
    for summary in rows:
        summary_updated = summary
        if not summary_prev:
            summary_updated['d_params'] = 0
            summary_updated['dofv'] = 0
        else:
            summary_updated['d_params'] = summary['n_params'] - summary_prev['n_params']
            summary_updated['dofv'] = summary_prev['ofv'] - summary['ofv']
        rows_updated.append(summary_updated)
        summary_prev = summary

    columns = ['tool', 'selected_model', 'description', 'ofv', 'dofv', 'n_params', 'd_params']
    df = pd.DataFrame.from_records(rows_updated, columns=columns).set_index(['tool'])
    return df


SubFunc = Callable[[Model], Optional[Results]]


def noop_subfunc(_: Model):
    return None


def _subfunc_modelsearch(search_space: Tuple[Statement, ...], path) -> SubFunc:
    def _run_modelsearch(model):
        res = run_tool(
            'modelsearch',
            search_space=mfl_stringify(search_space),
            algorithm='reduced_stepwise',
            model=model,
            path=path / 'modelsearch',
        )
        assert isinstance(res, Results)
        return res

    return _run_modelsearch


def _subfunc_iiv(path) -> SubFunc:
    def _run_iiv(model):
        res = run_tool(
            'iivsearch',
            'brute_force',
            iiv_strategy='fullblock',
            model=model,
            path=path / 'iivsearch',
        )
        assert isinstance(res, Results)
        return res

    return _run_iiv


def _subfunc_ruvsearch(path) -> SubFunc:
    def _run_ruvsearch(model):
        if has_blq_transformation(model):
            skip, max_iter = ['IIV_on_RUV', 'time_varying'], 1
        else:
            skip, max_iter = [], 3
        res = run_tool('ruvsearch', model, skip=skip, max_iter=max_iter, path=path / 'ruvsearch')
        assert isinstance(res, Results)
        return res

    return _run_ruvsearch


def _subfunc_covariates(
    amd_start_model: Model, search_space: Tuple[Statement, ...], path
) -> SubFunc:
    covariates = set(extract_covariates(amd_start_model, search_space))
    if covariates:
        allowed_covariates = get_covariates_allowed_in_covariate_effect(amd_start_model)
        for covariate in sorted(covariates):
            if covariate not in allowed_covariates:
                raise ValueError(
                    f'Invalid `search_space` because of invalid covariate found in'
                    f' search_space: got `{covariate}`,'
                    f' must be in {sorted(allowed_covariates)}.'
                )
    else:
        warnings.warn(
            'COVsearch will most likely be skipped because no covariates could be found.'
            ' Check search_space definition'
            ' and .datainfo usage of "covariate" type and "continuous" flag.'
        )

    def _run_covariates(model):
        effects = list(covariate_spec(model, search_space))

        if not effects:
            warnings.warn(
                'Skipping COVsearch because no effect candidates could be generated.'
                ' Check search_space definition'
                ' and .datainfo usage of "covariate" type and "continuous" flag.'
            )
            return None

        res = run_tool(
            'covsearch', mfl_stringify(search_space), model=model, path=path / 'covsearch'
        )
        assert isinstance(res, Results)
        return res

    return _run_covariates


def _subfunc_allometry(amd_start_model: Model, input_allometric_variable, path) -> SubFunc:
    def _allometric_variable(model: Model):
        if input_allometric_variable is not None:
            return input_allometric_variable

        for col in model.datainfo:
            if col.descriptor == 'body weight':
                return col.name

        return None

    allometric_variable = _allometric_variable(amd_start_model)

    if allometric_variable is None:
        warnings.warn(
            'Allometry will most likely be skipped because allometric_variable is None and could'
            ' not be inferred through .datainfo via "body weight" descriptor.'
        )

    else:
        validate_allometric_variable(amd_start_model, allometric_variable)

    def _run_allometry(model):
        allometric_variable = _allometric_variable(model)

        if allometric_variable is None:
            warnings.warn(
                'Skipping Allometry because allometric_variable is None and could'
                ' not be inferred through .datainfo via "body weight" descriptor.'
            )
            return None

        res = run_tool(
            'allometry', model, allometric_variable=allometric_variable, path=path / 'allometry'
        )
        assert isinstance(res, Results)
        return res

    return _run_allometry


def _subfunc_iov(amd_start_model, occasion, path) -> SubFunc:
    if occasion is None:
        warnings.warn('IOVsearch will be skipped because occasion is None.')
        return noop_subfunc

    if occasion not in amd_start_model.dataset:
        raise ValueError(
            f'Invalid `occasion`: got `{occasion}`,'
            f' must be one of {sorted(amd_start_model.datainfo.names)}.'
        )

    def _run_iov(model):
        if occasion not in model.dataset:
            warnings.warn(f'Skipping IOVsearch because dataset is missing column "{occasion}".')
            return None

        categories = get_occasion_levels(model.dataset, occasion)
        if len(categories) < 2:
            warnings.warn(
                f'Skipping IOVsearch because there are less than two '
                f'occasion categories in column "{occasion}": {categories}.'
            )
            return None

        res = run_tool('iovsearch', model=model, column=occasion, path=path / 'iovsearch')
        assert isinstance(res, Results)
        return res

    return _run_iov
