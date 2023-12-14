import warnings
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.model import Model
from pharmpy.modeling import (
    has_mixed_mm_fo_elimination,
    plot_abs_cwres_vs_ipred,
    plot_cwres_vs_idv,
    plot_dv_vs_ipred,
    plot_dv_vs_pred,
)
from pharmpy.modeling.blq import has_blq_transformation, transform_blq
from pharmpy.modeling.common import convert_model, filter_dataset
from pharmpy.modeling.covariate_effect import get_covariates_allowed_in_covariate_effect
from pharmpy.modeling.parameter_variability import get_occasion_levels
from pharmpy.reporting import generate_report
from pharmpy.tools import retrieve_models, summarize_errors, write_results
from pharmpy.tools.allometry.tool import validate_allometric_variable
from pharmpy.tools.mfl.feature.covariate import covariates as extract_covariates
from pharmpy.tools.mfl.feature.covariate import spec as covariate_spec
from pharmpy.tools.mfl.filter import (
    COVSEARCH_STATEMENT_TYPES,
    STRUCTSEARCH_STATEMENT_TYPES,
    mfl_filtering,
)
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.mfl.statement.feature.absorption import Absorption
from pharmpy.tools.mfl.statement.feature.covariate import Covariate, Ref
from pharmpy.tools.mfl.statement.feature.elimination import Elimination
from pharmpy.tools.mfl.statement.feature.lagtime import LagTime
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.feature.symbols import Name, Option, Wildcard
from pharmpy.tools.mfl.statement.feature.transits import Transits
from pharmpy.tools.mfl.statement.statement import Statement
from pharmpy.tools.mfl.stringify import stringify as mfl_stringify
from pharmpy.workflows import Results, default_tool_database
from pharmpy.workflows.results import ModelfitResults

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
    b_init: Optional[Union[int, float]] = None,
    emax_init: Optional[Union[int, float]] = None,
    ec50_init: Optional[Union[int, float]] = None,
    met_init: Optional[Union[int, float]] = None,
    search_space: Optional[str] = None,
    lloq_method: Optional[str] = None,
    lloq_limit: Optional[str] = None,
    order: Optional[List[str]] = None,
    allometric_variable: Optional[Union[str, sympy.Symbol]] = None,
    occasion: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    resume: bool = False,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    dv_types: Optional[dict[str, int]] = None,
    mechanistic_covariates: Optional[List[str]] = None,
    retries_strategy: Literal["final", "all_final", "skip"] = "final",
    seed: Optional[Union[np.random.Generator, int]] = None,
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
        Type of model to build. Valid strings are 'basic_pk', 'pkpd', 'drug_metabolite' and 'tmdd'
    administration : str
        Route of administration. Either 'iv', 'oral' or 'ivoral'
    cl_init : float
        Initial estimate for the population clearance
    vc_init : float
        Initial estimate for the central compartment population volume
    mat_init : float
        Initial estimate for the mean absorption time (not for iv models)
    b_init : float
        Initial estimate for the baseline (PKPD model)
    emax_init : float
        Initial estimate for E_max (PKPD model)
    ec50_init : float
        Initial estimate for EC_50 (PKPD model)
    met_init : float
        Initial estimate for mean equilibration time (PKPD model)
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
    strictness : str or None
        Strictness criteria
    dv_types : dict or None
        Dictionary of DV types for TMDD models with multiple DVs.
    mechanistic_covariates : list
        List of covariates to run in a separate proioritized covsearch run. The effects are extracted
        from the search space for covsearch
    retries_strategy: str
        Weither or not to run retries tool. Valid options are 'skip', 'all_final' or 'final'.
        Default is 'final'.
    seed : int or rng
        Random number generator or seed to be used.

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

    from pharmpy.model.external import nonmem  # FIXME: We should not depend on NONMEM

    if administration not in ['iv', 'oral', 'ivoral']:
        raise ValueError(f'Invalid input: "{administration}" as administration is not supported')
    if modeltype not in ['basic_pk', 'pkpd', 'drug_metabolite', 'tmdd']:
        raise ValueError(f'Invalid input: "{modeltype}" as modeltype is not supported')

    if modeltype == 'pkpd':
        dv = 2
        iiv_strategy = 'pd_fullblock'

        try:
            input_search_space_features = [] if search_space is None else mfl_parse(search_space)
        except:  # noqa E722
            raise ValueError(f'Invalid `search_space`, could not be parsed: "{search_space}"')

        structsearch_features = tuple(
            filter(
                lambda statement: isinstance(statement, STRUCTSEARCH_STATEMENT_TYPES),
                input_search_space_features,
            )
        )
        if not structsearch_features:
            structsearch_features = "DIRECTEFFECT(*);EFFECTCOMP(*);INDIRECTEFFECT(*,*)"
        else:
            structsearch_features = mfl_stringify(structsearch_features)
    else:
        dv = None
        iiv_strategy = 'fullblock'

    if type(input) is str:
        from pharmpy.modeling import create_basic_pk_model

        model = create_basic_pk_model(
            administration,
            dataset_path=input,
            cl_init=cl_init,
            vc_init=vc_init,
            mat_init=mat_init,
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
        if modeltype == 'pkpd':
            order = ['structural', 'iivsearch', 'residual', 'iovsearch', 'covariates']
        else:
            order = default_order

    try:
        input_search_space_features = [] if search_space is None else mfl_parse(search_space)
    except:  # noqa E722
        raise ValueError(f'Invalid `search_space`, could not be parsed: "{search_space}"')

    modelsearch_features = mfl_filtering(input_search_space_features, "modelsearch")
    if modeltype in ['pkpd', 'drug_metabolite']:
        structsearch_features = mfl_filtering(input_search_space_features, "structsearch")
        if search_space is None:
            if modeltype == 'pkpd':
                structsearch_features = "DIRECTEFFECT(*);EFFECTCOMP(*);INDIRECTEFFECT(*,*)"
            else:
                if administration in ['oral', 'ivoral']:
                    structsearch_features = "METABOLITE([PSC, BASIC]);PERIPHERALS([0,1], MET)"
                else:
                    structsearch_features = "METABOLITE([BASIC]);PERIPHERALS([0,1], MET)"

    if not modelsearch_features:
        if modeltype in ('basic_pk', 'drug_metabolite') and administration == 'oral':
            modelsearch_features = (
                Absorption((Name('FO'), Name('ZO'), Name('SEQ-ZO-FO'))),
                Elimination((Name('FO'),)),
                LagTime((Name('OFF'), Name('ON'))),
                Transits((0, 1, 3, 10), Wildcard()),
                Peripherals((0, 1)),
            )
        elif modeltype in ('basic_pk', 'drug_metabolite') and administration == 'ivoral':
            modelsearch_features = (
                Absorption((Name('FO'), Name('ZO'), Name('SEQ-ZO-FO'))),
                Elimination((Name('FO'),)),
                LagTime((Name('OFF'), Name('ON'))),
                Transits((0, 1, 3, 10), Wildcard()),
                Peripherals((0, 1, 2)),
            )
        else:
            modelsearch_features = (
                Elimination((Name('FO'),)),
                Peripherals((0, 1, 2)),
            )

    covsearch_features = tuple(
        filter(
            lambda statement: isinstance(statement, COVSEARCH_STATEMENT_TYPES),
            input_search_space_features,
        )
    )
    if not any(map(lambda statement: isinstance(statement, Covariate), covsearch_features)):
        def_cov_search_feature = (
            Covariate(Ref('IIV'), Ref('CONTINUOUS'), ('exp',), '*', Option(True)),
            Covariate(Ref('IIV'), Ref('CATEGORICAL'), ('cat',), '*', Option(True)),
        )
        if modeltype == 'basic_pk' and administration == 'ivoral':
            def_cov_search_feature = def_cov_search_feature + (
                Covariate(('RUV',), ('ADMID',), ('cat',), '*', Option(True)),
            )

        covsearch_features = def_cov_search_feature + covsearch_features

    if modeltype == "tmdd":
        orig_dataset = model.dataset
        if dv_types is not None:
            model = filter_dataset(model, 'DVID < 2')

    db = default_tool_database(toolname='amd', path=path, exist_ok=resume)
    run_subfuncs = {}
    # Always add?
    run_subfuncs['structural_covariates'] = _subfunc_structural_covariates(
        amd_start_model=model,
        search_space=covsearch_features,
        strictness=strictness,
        path=db.path,
    )
    for section in order:
        if section == 'structural':
            if modeltype == 'pkpd':
                func = _subfunc_structsearch(
                    type=modeltype,
                    search_space=structsearch_features,
                    b_init=b_init,
                    emax_init=emax_init,
                    ec50_init=ec50_init,
                    met_init=met_init,
                    strictness=strictness,
                    path=db.path,
                )
                run_subfuncs['structsearch'] = func
            elif modeltype == 'tmdd':
                func = _subfunc_structsearch_tmdd(
                    search_space=modelsearch_features,
                    type=modeltype,
                    strictness=strictness,
                    dv_types=dv_types,
                    orig_dataset=orig_dataset,
                    path=db.path,
                )
                run_subfuncs['structsearch'] = func
            else:
                func = _subfunc_modelsearch(
                    search_space=modelsearch_features, strictness=strictness, path=db.path
                )
                run_subfuncs['modelsearch'] = func
            # Perfomed 'after' modelsearch
            if modeltype == 'drug_metabolite':
                func = _subfunc_structsearch(
                    type=modeltype,
                    search_space=structsearch_features,
                    path=db.path,
                )
                run_subfuncs['structsearch'] = func
        elif section == 'iivsearch':
            func = _subfunc_iiv(iiv_strategy=iiv_strategy, strictness=strictness, path=db.path)
            run_subfuncs['iivsearch'] = func
        elif section == 'iovsearch':
            func = _subfunc_iov(
                amd_start_model=model, occasion=occasion, strictness=strictness, path=db.path
            )
            run_subfuncs['iovsearch'] = func
        elif section == 'residual':
            if modeltype == 'drug_metabolite':
                # FIXME : Assume the dv number?
                # Perform two searches
                # One for the drug
                func = _subfunc_ruvsearch(
                    dv=1, strictness=strictness, path=db.path / 'ruvsearch_drug'
                )
                run_subfuncs['ruvsearch_drug'] = func
                # And one for the metabolite
                func = _subfunc_ruvsearch(
                    dv=2, strictness=strictness, path=db.path / 'ruvsearch_metabolite'
                )
                run_subfuncs['ruvsearch_metabolite'] = func
            elif modeltype == 'tmdd' and dv_types is not None:
                # Run for DV = 1
                func = _subfunc_ruvsearch(
                    dv=1, strictness=strictness, path=db.path / 'ruvsearch_drug'
                )
                run_subfuncs['ruvsearch_drug'] = func
                # Run for second dv which is the first key in dv_types that is not 'drug'
                if list(dv_types.keys())[0] == 'drug':
                    second_dv_type = list(dv_types.keys())[1]
                else:
                    second_dv_type = list(dv_types.keys())[0]
                func = _subfunc_ruvsearch(
                    dv=dv_types[second_dv_type],
                    strictness=strictness,
                    path=db.path / f"ruvsearch_{second_dv_type}",
                )
                run_subfuncs[f'ruvsearch_{second_dv_type}'] = func
            else:
                func = _subfunc_ruvsearch(dv=dv, strictness=strictness, path=db.path)
                run_subfuncs['ruvsearch'] = func
        elif section == 'allometry':
            func = _subfunc_allometry(
                amd_start_model=model, input_allometric_variable=allometric_variable, path=db.path
            )
            run_subfuncs['allometry'] = func
        elif section == 'covariates':
            func = _subfunc_mechanistic_exploratory_covariates(
                amd_start_model=model,
                search_space=covsearch_features,
                mechanistic_covariates=mechanistic_covariates,
                strictness=strictness,
                path=db.path,
            )
            run_subfuncs['covsearch'] = func
        else:
            raise ValueError(
                f"Unrecognized section {section} in order. Must be one of {default_order}"
            )
        if retries_strategy == 'all_final':
            func = _subfunc_retires(tool=section, strictness=strictness, seed=seed, path=db.path)
            run_subfuncs[f'{section}_retries'] = func

    if retries_strategy == 'final':
        func = _subfunc_retires(tool="", strictness=strictness, seed=seed, path=db.path)
        run_subfuncs['retries'] = func

    # Filter data to only contain dvid=1
    if modeltype == "drug_metabolite":
        orig_dataset = model.dataset
        # FIXME : remove
        model = filter_dataset(model, 'DVID != 2')

    if results is None:
        results = run_tool('modelfit', model, path=db.path / 'modelfit', resume=resume)
    model = model.replace(modelfit_results=results)
    next_model = model
    sum_subtools, sum_models, sum_inds_counts, sum_amd = [], [], [], []
    sum_subtools.append(_create_sum_subtool('start', model))
    for tool_name, func in run_subfuncs.items():
        if modeltype == 'drug_metabolite' and tool_name == "structsearch":
            next_model = next_model.replace(dataset=orig_dataset)
        subresults = func(next_model)

        if subresults is None:
            sum_models.append(None)
            sum_inds_counts.append(None)
        else:
            if subresults.final_model.name != next_model.name:
                if tool_name == "allometry" and 'allometry' in order[: order.index('covariates')]:
                    cov_before = ModelFeatures.create_from_mfl_string(
                        get_model_features(next_model)
                    )
                    cov_after = ModelFeatures.create_from_mfl_string(
                        get_model_features(subresults.final_model)
                    )
                    cov_differences = (cov_after - cov_before).covariate
                    if cov_differences:
                        covsearch_features += cov_differences
                        func = _subfunc_mechanistic_exploratory_covariates(
                            amd_start_model=model,
                            search_space=covsearch_features,
                            strictness=strictness,
                            mechanistic_covariates=mechanistic_covariates,
                            path=db.path,
                        )
                        run_subfuncs['covsearch'] = func
                next_model = subresults.final_model
                results = subresults.tool_database.model_database.retrieve_modelfit_results(
                    subresults.final_model.name
                )
                next_model = next_model.replace(modelfit_results=results)
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

    final_model = next_model
    final_results = next_model.modelfit_results
    summary_errors = summarize_errors(final_results)
    if final_results.predictions is not None:
        dv_vs_ipred_plot = plot_dv_vs_ipred(model, final_results.predictions)
        dv_vs_pred_plot = plot_dv_vs_pred(model, final_results.predictions)
    else:
        dv_vs_ipred_plot = None
        dv_vs_pred_plot = None
    if final_results.residuals is not None:
        cwres_vs_idv_plot = plot_cwres_vs_idv(model, final_results.residuals)
    else:
        cwres_vs_idv_plot = None
    if final_results.predictions is not None and final_results.residuals is not None:
        abs_cwres_vs_ipred_plot = plot_abs_cwres_vs_ipred(
            model, predictions=final_results.predictions, residuals=final_results.residuals
        )
    else:
        abs_cwres_vs_ipred_plot = None
    res = AMDResults(
        final_model=final_model.name,
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_individuals_count=summary_individuals_count,
        summary_errors=summary_errors,
        final_model_parameter_estimates=_table_final_parameter_estimates(
            model, final_results.parameter_estimates_sdcorr, final_results.standard_errors_sdcorr
        ),
        final_model_dv_vs_ipred_plot=dv_vs_ipred_plot,
        final_model_dv_vs_pred_plot=dv_vs_pred_plot,
        final_model_cwres_vs_idv_plot=cwres_vs_idv_plot,
        final_model_abs_cwres_vs_ipred_plot=abs_cwres_vs_ipred_plot,
    )
    # Since we are outside of the regular tools machinery the following is needed
    results_path = db.path / 'results.json'
    write_results(results=res, path=results_path)
    write_results(results=res, path=db.path / 'results.csv', csv=True)
    rst_path = Path(__file__).parent / 'report.rst'
    target_path = db.path / 'results.html'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate_report(rst_path, results_path, target_path)
    return res


def _table_final_parameter_estimates(model: Model, parameter_estimates, ses):
    rse = ses / parameter_estimates
    rse.name = "RSE"
    df = pd.concat([parameter_estimates, rse], axis=1)
    return df


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


def _subfunc_retires(tool, strictness, seed, path):
    def _run_retries(model):
        res = run_tool(
            'retries',
            model=model,
            results=model.modelfit_results,
            strictness=strictness,
            scale='UCP',
            prefix_name=tool,
            seed=seed,
            path=path / f'{tool}_retries',
        )
        assert isinstance(res, Results)
        return res

    return _run_retries


def _subfunc_modelsearch(search_space: Tuple[Statement, ...], strictness, path) -> SubFunc:
    def _run_modelsearch(model):
        res = run_tool(
            'modelsearch',
            search_space=mfl_stringify(search_space),
            algorithm='reduced_stepwise',
            model=model,
            strictness=strictness,
            results=model.modelfit_results,
            path=path / 'modelsearch',
        )
        assert isinstance(res, Results)

        return res

    return _run_modelsearch


def _subfunc_structsearch(path, **kwargs) -> SubFunc:
    def _run_structsearch(model):
        res = run_tool(
            'structsearch',
            model=model,
            results=model.modelfit_results,
            **kwargs,
            path=path / 'structsearch',
        )
        assert isinstance(res, Results)
        return res

    return _run_structsearch


def _subfunc_structsearch_tmdd(
    search_space, type, strictness, dv_types, orig_dataset, path
) -> SubFunc:
    def _run_structsearch_tmdd(model):
        res = run_tool(
            'modelsearch',
            search_space=mfl_stringify(search_space),
            algorithm='reduced_stepwise',
            model=model,
            strictness=strictness,
            results=model.modelfit_results,
            path=path / 'modelsearch',
        )

        final_model = res.final_model

        if not has_mixed_mm_fo_elimination(final_model):
            # Only select models that have mixed MM FO elimination
            # If no model with mixed MM FO then final model from modelsearch will be used
            model_db = res.tool_database.model_database
            all_models = [model_db.retrieve_model(model) for model in model_db.list_models()]
            models_mixed_mm_fo_el = [
                model.name for model in all_models if has_mixed_mm_fo_elimination(model)
            ]
            if len(models_mixed_mm_fo_el) > 0:
                rank_all = res.summary_tool.dropna(subset='bic')[['rank']]
                rank_filtered = rank_all.query('model in @models_mixed_mm_fo_el')
                if len(rank_filtered) > 0:
                    rank_filtered = rank_filtered.sort_values(by=['rank'])
                    highest_ranked = rank_filtered.index[0]
                    final_model = retrieve_models(path / 'modelsearch', names=[highest_ranked])[0]

        extra_model = None
        extra_model_results = None
        n_peripherals = len(final_model.statements.ode_system.find_peripheral_compartments())
        modelfeatures = ModelFeatures.create_from_mfl_string(get_model_features(final_model))
        # Model features - 1 peripheral compartment
        modelfeatures_minus = modelfeatures.replace(peripherals=Peripherals((n_peripherals - 1,)))
        # Loop through all models and find one with same features
        models = [
            model.name
            for model in all_models
            if ModelFeatures.create_from_mfl_string(get_model_features(model))
            == modelfeatures_minus
        ]
        if len(models) > 0:
            # Find highest ranked model
            rank_all = res.summary_tool.dropna(subset='bic')[['rank']]
            rank_filtered = rank_all.query('model in @models')
            if len(rank_filtered) > 0:
                rank_filtered = rank_filtered.sort_values(by=['rank'])
                highest_ranked = rank_filtered.index[0]
                extra_model = retrieve_models(path / 'modelsearch', names=[highest_ranked])[0]
                if dv_types is not None:
                    extra_model = extra_model.replace(dataset=orig_dataset)
                extra_model_results = extra_model.modelfit_results

        # Replace original dataset if multiple DVs
        if dv_types is not None:
            final_model = final_model.replace(dataset=orig_dataset)

        res = run_tool(
            'structsearch',
            type=type,
            model=final_model,
            results=final_model.modelfit_results,
            extra_model=extra_model,
            extra_model_results=extra_model_results,
            strictness=strictness,
            dv_types=dv_types,
            path=path / 'structsearch',
        )
        assert isinstance(res, Results)
        return res

    return _run_structsearch_tmdd


def _subfunc_iiv(iiv_strategy, strictness, path) -> SubFunc:
    def _run_iiv(model):
        res = run_tool(
            'iivsearch',
            'brute_force',
            iiv_strategy=iiv_strategy,
            model=model,
            results=model.modelfit_results,
            strictness=strictness,
            keep=['CL'],
            path=path / 'iivsearch',
        )
        assert isinstance(res, Results)
        return res

    return _run_iiv


def _subfunc_ruvsearch(dv, strictness, path) -> SubFunc:
    def _run_ruvsearch(model):
        if has_blq_transformation(model):
            skip, max_iter = ['IIV_on_RUV', 'time_varying'], 1
        else:
            skip, max_iter = [], 3
        res = run_tool(
            'ruvsearch',
            model,
            results=model.modelfit_results,
            skip=skip,
            max_iter=max_iter,
            dv=dv,
            strictness=strictness,
            path=path / 'ruvsearch',
        )
        assert isinstance(res, Results)
        return res

    return _run_ruvsearch


def _subfunc_structural_covariates(
    amd_start_model: Model,
    search_space: Tuple[Statement, ...],
    strictness,
    path,
) -> SubFunc:
    def _run_structural_covariates(model):
        from pharmpy.modeling import get_pk_parameters

        allowed_parameters = allowed_parameters = set(get_pk_parameters(model)).union(
            str(statement.symbol) for statement in model.statements.before_odes
        )
        # Extract all forced
        mfl = ModelFeatures.create_from_mfl_statement_list(search_space)
        mfl_covariates = mfl.expand(model).covariate
        structural_searchspace = []
        skipped_parameters = set()
        for cov_statement in mfl_covariates:
            if not cov_statement.optional.option:
                filtered_parameters = tuple(
                    [p for p in cov_statement.parameter if p in allowed_parameters]
                )
                # Not optional -> Add to all search spaces (was added in structural run)
                structural_searchspace.append(
                    Covariate(
                        filtered_parameters,
                        cov_statement.covariate,
                        cov_statement.fp,
                        cov_statement.op,
                        cov_statement.optional,
                    )
                )
                skipped_parameters.union(set(cov_statement.parameter) - set(filtered_parameters))

        # Ignore warning?
        if skipped_parameters:
            warnings.warn(
                f'{skipped_parameters} missing in start model and structural covariate effect cannot be added'
                ' Might be added during a later COVsearch step if possible.'
            )
        if not structural_searchspace and not skipped_parameters:
            # Uneccessary to warn (?)
            return None
        elif not structural_searchspace:
            warnings.warn(
                'No applicable structural covariates found in search space. Skipping structural_COVsearch'
            )
            return None

        res = run_tool(
            'covsearch',
            mfl_stringify(structural_searchspace),
            model=model,
            strictness=strictness,
            results=model.modelfit_results,
            path=path / 'covsearch_structural',
        )
        assert isinstance(res, Results)
        return res

    return _run_structural_covariates


def _subfunc_mechanistic_exploratory_covariates(
    amd_start_model: Model,
    search_space: Tuple[Statement, ...],
    mechanistic_covariates,
    strictness,
    path,
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

    def _run_mechanistic_exploratory_covariates(model):
        index_offset = 0  # For naming runs

        effects = list(covariate_spec(model, search_space))

        if not effects:
            warnings.warn(
                'Skipping COVsearch because no effect candidates could be generated.'
                ' Check search_space definition'
                ' and .datainfo usage of "covariate" type and "continuous" flag.'
            )
            return None

        if mechanistic_covariates:
            # Extract them and all forced
            mfl = ModelFeatures.create_from_mfl_statement_list(search_space)
            mfl_covariates = mfl.expand(model).covariate
            filtered_searchspace = []
            mechanistic_searchspace = []
            structural_searchspace = []
            for cov_statement in mfl_covariates:
                if not cov_statement.optional.option:
                    # Not optional -> Add to all search spaces (was added in structural run)
                    structural_searchspace.append(cov_statement)
                    mechanistic_searchspace.append(cov_statement)
                    filtered_searchspace.append(cov_statement)
                else:
                    current_cov = []
                    for cov in cov_statement.covariate:
                        if cov in mechanistic_covariates:
                            current_cov.append(cov)
                    if current_cov:
                        mechanistic_cov = Covariate(
                            cov_statement.parameter,
                            tuple(current_cov),
                            cov_statement.fp,
                            cov_statement.op,
                            cov_statement.optional,
                        )
                        mechanistic_searchspace.append(mechanistic_cov)
                        if filtered_cov_set := tuple(
                            set(cov_statement.covariate) - set(current_cov)
                        ):
                            filtered_cov = Covariate(
                                cov_statement.parameter,
                                filtered_cov_set,
                                cov_statement.fp,
                                cov_statement.op,
                                cov_statement.optional,
                            )
                            filtered_searchspace.append(filtered_cov)
                    else:
                        filtered_searchspace.append(cov_statement)

            if not mechanistic_searchspace:
                warnings.warn(
                    'No covariate effect for given mechanistic covariates found.'
                    ' Skipping mechanistic COVsearch.'
                )
            else:
                res = run_tool(
                    'covsearch',
                    mfl_stringify(mechanistic_searchspace),
                    model=model,
                    strictness=strictness,
                    results=model.modelfit_results,
                    path=path / 'covsearch_mechanistic',
                )
                index_offset = int(
                    res.summary_tool.iloc[-1].name[-1][13:]
                )  # Get largest number of run
                if res.final_model.name != model.name:
                    model = res.final_model
                    model_results = res.tool_database.model_database.retrieve_modelfit_results(
                        res.final_model.name
                    )
                    model = model.replace(modelfit_results=model_results)
                    added_covs = ModelFeatures.create_from_mfl_string(
                        get_model_features(model)
                    ).covariate
                    filtered_searchspace.extend(
                        added_covs
                    )  # Avoid removing added cov in exploratory
        else:
            filtered_searchspace = search_space

        res = run_tool(
            'covsearch',
            mfl_stringify(filtered_searchspace),
            model=model,
            strictness=strictness,
            results=model.modelfit_results,
            path=path / 'covsearch_exploratory',
            naming_index_offset=index_offset,
        )
        assert isinstance(res, Results)
        return res

    return _run_mechanistic_exploratory_covariates


def _allometric_variable(model: Model, input_allometric_variable):
    if input_allometric_variable is not None:
        return input_allometric_variable

    for col in model.datainfo:
        if col.descriptor == 'body weight':
            return col.name

    return None


def _subfunc_allometry(amd_start_model: Model, input_allometric_variable, path) -> SubFunc:
    allometric_variable = _allometric_variable(amd_start_model, input_allometric_variable)

    if allometric_variable is None:
        warnings.warn(
            'Allometry will most likely be skipped because allometric_variable is None and could'
            ' not be inferred through .datainfo via "body weight" descriptor.'
        )

    else:
        validate_allometric_variable(amd_start_model, allometric_variable)

    def _run_allometry(model):
        allometric_variable = _allometric_variable(model, input_allometric_variable)

        if allometric_variable is None:
            warnings.warn(
                'Skipping Allometry because allometric_variable is None and could'
                ' not be inferred through .datainfo via "body weight" descriptor.'
            )
            return None

        res = run_tool(
            'allometry',
            model,
            results=model.modelfit_results,
            allometric_variable=allometric_variable,
            path=path / 'allometry',
        )
        assert isinstance(res, Results)
        return res

    return _run_allometry


def _subfunc_iov(amd_start_model, occasion, strictness, path) -> SubFunc:
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

        res = run_tool(
            'iovsearch',
            model=model,
            results=model.modelfit_results,
            column=occasion,
            strictness=strictness,
            path=path / 'iovsearch',
        )
        assert isinstance(res, Results)
        return res

    return _run_iov
