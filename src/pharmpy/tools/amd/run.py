import re
import warnings
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

from pharmpy.basic import TSymbol
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.type import check_list, with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import (
    add_parameter_uncertainty_step,
    add_predictions,
    add_residuals,
    create_basic_pk_model,
    find_clearance_parameters,
    get_central_volume_and_clearance,
    get_pk_parameters,
    has_mixed_mm_fo_elimination,
    plot_abs_cwres_vs_ipred,
    plot_cwres_vs_idv,
    plot_dv_vs_ipred,
    plot_dv_vs_pred,
    plot_eta_distributions,
    set_simulation,
    vpc_plot,
)
from pharmpy.modeling.blq import has_blq_transformation, transform_blq
from pharmpy.modeling.common import convert_model, filter_dataset
from pharmpy.modeling.covariate_effect import get_covariates_allowed_in_covariate_effect
from pharmpy.modeling.parameter_variability import get_occasion_levels
from pharmpy.modeling.tmdd import DV_TYPES
from pharmpy.reporting import generate_report
from pharmpy.tools import retrieve_models, write_results
from pharmpy.tools.allometry.tool import validate_allometric_variable
from pharmpy.tools.common import table_final_eta_shrinkage
from pharmpy.tools.mfl.feature.covariate import covariates as extract_covariates
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.mfl.statement.feature.covariate import Covariate
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.statement import Statement
from pharmpy.tools.run import summarize_errors_from_entries
from pharmpy.workflows import ModelEntry, Results, default_context
from pharmpy.workflows.model_database.local_directory import get_modelfit_results
from pharmpy.workflows.results import ModelfitResults

from ..run import run_tool
from .results import AMDResults

ALLOWED_STRATEGY = ["default", "reevaluation", "SIR", "SRI", "RSI"]
ALLOWED_ADMINISTRATION = ["iv", "oral", "ivoral"]
ALLOWED_MODELTYPE = ['basic_pk', 'pkpd', 'drug_metabolite', 'tmdd']
RETRIES_STRATEGIES = ["final", "all_final", "skip"]


def run_amd(
    input: Union[Model, Path, str],
    results: Optional[ModelfitResults] = None,
    modeltype: str = 'basic_pk',
    administration: str = 'oral',
    strategy: str = "default",
    cl_init: Optional[float] = None,
    vc_init: Optional[float] = None,
    mat_init: Optional[float] = None,
    b_init: Optional[float] = None,
    emax_init: Optional[float] = None,
    ec50_init: Optional[float] = None,
    met_init: Optional[float] = None,
    search_space: Optional[str] = None,
    lloq_method: Optional[str] = None,
    lloq_limit: Optional[str] = None,
    allometric_variable: Optional[TSymbol] = None,
    occasion: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    resume: bool = False,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    dv_types: Optional[dict[Literal[DV_TYPES], int]] = None,
    mechanistic_covariates: Optional[List[Union[str, tuple[str]]]] = None,
    retries_strategy: Literal["final", "all_final", "skip"] = "all_final",
    seed: Optional[Union[np.random.Generator, int]] = None,
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
    ignore_datainfo_fallback: bool = False,
):
    """Run Automatic Model Development (AMD) tool

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
    strategy : str
        Run algorithm for AMD procedure. Valid options are 'default', 'reevaluation'. Default is 'default'
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
        List of covariates or tuple of covariate and parameter combination to run in a
        separate proioritized covsearch run. For instance ["WT", ("CRCL", "CL")].
        The effects are extracted from the search space for covsearch.
    retries_strategy: str
        Whether or not to run retries tool. Valid options are 'skip', 'all_final' or 'final'.
        Default is 'final'.
    seed : int or rng
        Random number generator or seed to be used.
    parameter_uncertainty_method: {'SANDWICH', 'SMAT', 'RMAT', 'EFIM'} or None
        Parameter uncertainty method.
    ignore_datainfo_fallback : bool
        Ignore using datainfo to get information not given by the user. Default is False

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
    args = locals()

    from pharmpy.model.external import nonmem  # FIXME: We should not depend on NONMEM

    if search_space is not None:
        try:
            ss_mfl = mfl_parse(search_space, True)
        except:  # noqa E722
            raise ValueError(f'Invalid `search_space`, could not be parsed: "{search_space}"')
    else:
        ss_mfl = ModelFeatures()

    if modeltype == 'pkpd':
        dv = 2
        iiv_strategy = 'pd_fullblock'
    else:
        dv = None
        iiv_strategy = 'fullblock'

    if isinstance(input, str):
        model = create_basic_pk_model(
            administration,
            dataset_path=input,
            cl_init=cl_init,
            vc_init=vc_init,
            mat_init=mat_init,
        )
        model = convert_model(model, 'nonmem')  # FIXME: Workaround for results retrieval system
    elif isinstance(input, nonmem.model.Model):
        model = input
        model = model.replace(name='start')
    else:
        # Redundant with validation
        raise TypeError(
            f'Invalid input: got `{input}` of type {type(input)},'
            f' only NONMEM model or standalone dataset are supported currently.'
        )

    if 'dvid' in model.datainfo.types:
        dvid_name = model.datainfo.typeix['dvid'][0].name
    elif 'DVID' in model.datainfo.names:
        dvid_name = 'DVID'
    else:
        dvid_name = None

    model = add_predictions(model, ['PRED', 'CIPREDI'])
    model = add_residuals(model, ['CWRES'])

    # FIXME : Handle validation differently?
    # AMD start model (dataset) is required before validation
    args['input'] = model
    to_be_skipped = validate_input(**args)

    if parameter_uncertainty_method is not None:
        model = add_parameter_uncertainty_step(model, parameter_uncertainty_method)

    if lloq_method is not None:
        model = transform_blq(
            model,
            method=lloq_method,
            lloq=lloq_limit,
        )

    if strategy == "default":
        order = ['structural', 'iivsearch', 'residual', 'iovsearch', 'allometry', 'covariates']
    elif strategy == "reevaluation":
        order = [
            'structural',
            'iivsearch',
            'residual',
            'iovsearch',
            'allometry',
            'covariates',
            'iivsearch',
            'residual',
        ]
    elif strategy == 'SIR':
        order = ['structural', 'iivsearch', 'residual']
    elif strategy == 'SRI':
        order = ['structural', 'residual', 'iivsearch']
    elif strategy == 'RSI':
        order = ['residual', 'structural', 'iivsearch']

    if modeltype == 'pkpd' and 'allometry' in order:
        warnings.warn('Skipping allometry since modeltype is "pkpd"')
        order.remove('allometry')

    if to_be_skipped:
        order = [tool for tool in order if tool not in to_be_skipped]

    if modeltype in ['pkpd', 'drug_metabolite']:
        if modeltype == 'pkpd':
            structsearch_features = ss_mfl.filter("pd")
        else:
            structsearch_features = ss_mfl.filter("metabolite")
        if len(structsearch_features.mfl_statement_list()) == 0:
            if modeltype == 'pkpd':
                structsearch_features = mfl_parse(
                    "DIRECTEFFECT(*);EFFECTCOMP(*);INDIRECTEFFECT(*,*)", True
                )
            else:
                if administration in ['oral', 'ivoral']:
                    structsearch_features = mfl_parse(
                        "METABOLITE([PSC, BASIC]);PERIPHERALS([0,1], MET)", True
                    )
                else:
                    structsearch_features = mfl_parse(
                        "METABOLITE([BASIC]);PERIPHERALS([0,1], MET)", True
                    )

    modelsearch_features = ss_mfl.filter("pk")
    if len(modelsearch_features.mfl_statement_list()) == 0:
        if modeltype in ('basic_pk', 'drug_metabolite') and administration == 'oral':
            modelsearch_features = mfl_parse(
                "ABSORPTION([FO,ZO,SEQ-ZO-FO]);"
                "ELIMINATION(FO);"
                "LAGTIME([OFF,ON]);"
                "TRANSITS([0,1,3,10],*);"
                "PERIPHERALS(0..1)",
                True,
            )
        elif modeltype in ('basic_pk', 'drug_metabolite') and administration == 'ivoral':
            modelsearch_features = mfl_parse(
                "ABSORPTION([FO,ZO,SEQ-ZO-FO]);"
                "ELIMINATION(FO);"
                "LAGTIME([OFF,ON]);"
                "TRANSITS([0,1,3,10],*);"
                "PERIPHERALS(0..2)",
                True,
            )
        elif modeltype == 'tmdd' and administration == 'oral':
            modelsearch_features = mfl_parse(
                "ABSORPTION([FO,ZO,SEQ-ZO-FO]);"
                "ELIMINATION([MM, MIX-FO-MM]);"
                "LAGTIME([OFF,ON]);"
                "TRANSITS([0,1,3,10],*);"
                "PERIPHERALS(0..1)",
                True,
            )
        elif modeltype == 'tmdd' and administration == 'ivoral':
            modelsearch_features = mfl_parse(
                "ABSORPTION([FO,ZO,SEQ-ZO-FO]);"
                "ELIMINATION([MM, MIX-FO-MM]);"
                "LAGTIME([OFF,ON]);"
                "TRANSITS([0,1,3,10],*);"
                "PERIPHERALS(0..2)",
                True,
            )
        else:
            modelsearch_features = mfl_parse("ELIMINATION(FO);" "PERIPHERALS(0..2)", True)

    covsearch_features = ModelFeatures.create(covariate=ss_mfl.covariate)
    if not covsearch_features.covariate:
        if modeltype != 'pkpd':
            cov_ss = mfl_parse(
                "COVARIATE?(@IIV, @CONTINUOUS, EXP);" "COVARIATE?(@IIV,@CATEGORICAL, CAT)", True
            )
        else:
            cov_ss = mfl_parse(
                "COVARIATE?(@PD_IIV, @CONTINUOUS, EXP);" "COVARIATE?(@PD_IIV,@CATEGORICAL, CAT)",
                True,
            )
        covsearch_features = covsearch_features.replace(covariate=cov_ss.covariate)
        if modeltype == 'basic_pk' and administration == 'ivoral':
            # FIXME : Allow addition between search space with reference values in COVARITATE statement
            covsearch_features = mfl_parse(str(cov_ss) + ";COVARIATE?(RUV,ADMID,CAT)", True)

    if modeltype == "tmdd":
        orig_dataset = model.dataset
        if dv_types is not None:
            model = filter_dataset(model, f'{dvid_name} < 2')

    n = 1
    while True:
        name = f"amd{n}"
        if not default_context.exists(name):
            ctx = default_context(name)
            break
        n += 1

    ctx = default_context(name, ref=path)
    run_subfuncs = {}

    for section in order:
        if section == 'structural':
            if modeltype != 'pkpd':
                run_subfuncs['structural_covariates'] = _subfunc_structural_covariates(
                    amd_start_model=model,
                    search_space=covsearch_features,
                    strictness=strictness,
                    ctx=ctx,
                )
            if modeltype == 'pkpd':
                func = _subfunc_structsearch(
                    type=modeltype,
                    search_space=structsearch_features,
                    b_init=b_init,
                    emax_init=emax_init,
                    ec50_init=ec50_init,
                    met_init=met_init,
                    strictness=strictness,
                    ctx=ctx,
                )
                run_subfuncs['structsearch'] = func
            elif modeltype == 'tmdd':
                func = _subfunc_structsearch_tmdd(
                    search_space=modelsearch_features,
                    type=modeltype,
                    strictness=strictness,
                    dv_types=dv_types,
                    orig_dataset=orig_dataset,
                    ctx=ctx,
                )
                run_subfuncs['structsearch'] = func
            else:
                func = _subfunc_modelsearch(
                    search_space=modelsearch_features, strictness=strictness, ctx=ctx
                )
                run_subfuncs['modelsearch'] = func
            # Perfomed 'after' modelsearch
            if modeltype == 'drug_metabolite':
                func = _subfunc_structsearch(
                    type=modeltype,
                    search_space=structsearch_features,
                    ctx=ctx,
                )
                run_subfuncs['structsearch'] = func
        elif section == 'iivsearch':
            if 'iivsearch' in run_subfuncs.keys():
                run_name = 'rerun_iivsearch'
                func = _subfunc_iiv(
                    iiv_strategy='no_add',
                    strictness=strictness,
                    ctx=ctx,
                    dir_name="rerun_iivsearch",
                )
            else:
                run_name = 'iivsearch'
                func = _subfunc_iiv(
                    iiv_strategy=iiv_strategy,
                    strictness=strictness,
                    ctx=ctx,
                    dir_name="iivsearch",
                )
            run_subfuncs[run_name] = func
        elif section == 'iovsearch':
            func = _subfunc_iov(
                amd_start_model=model, occasion=occasion, strictness=strictness, ctx=ctx
            )
            run_subfuncs['iovsearch'] = func
        elif section == 'residual':
            if any(k.startswith('ruvsearch') for k in run_subfuncs.keys()):
                run_name = 'rerun_ruvsearch'
            else:
                run_name = 'ruvsearch'
            if modeltype == 'drug_metabolite':
                # FIXME : Assume the dv number?
                # Perform two searches
                # One for the drug
                func = _subfunc_ruvsearch(
                    dv=1,
                    strictness=strictness,
                    ctx=ctx,
                    dir_name=f'{run_name}_drug',
                )
                run_subfuncs[f'{run_name}_drug'] = func
                # And one for the metabolite
                func = _subfunc_ruvsearch(
                    dv=2,
                    strictness=strictness,
                    ctx=ctx,
                    dir_name=f'{run_name}_metabolite',
                )
                run_subfuncs[f'{run_name}_metabolite'] = func
            elif modeltype == 'tmdd' and dv_types is not None:
                for key, value in dv_types.items():
                    func = _subfunc_ruvsearch(
                        dv=value,
                        strictness=strictness,
                        ctx=ctx,
                        dir_name=f'{run_name}_tmdd_{key}',
                    )
                    run_subfuncs[f'ruvsearch_{key}'] = func
            else:
                func = _subfunc_ruvsearch(dv=dv, strictness=strictness, ctx=ctx, dir_name=run_name)
                run_subfuncs[f'{run_name}'] = func
        elif section == 'allometry':
            func = _subfunc_allometry(
                amd_start_model=model, allometric_variable=allometric_variable, ctx=ctx
            )
            run_subfuncs['allometry'] = func
        elif section == 'covariates':
            func = _subfunc_mechanistic_exploratory_covariates(
                amd_start_model=model,
                search_space=covsearch_features,
                mechanistic_covariates=mechanistic_covariates,
                strictness=strictness,
                ctx=ctx,
            )
            run_subfuncs['covsearch'] = func
        else:
            raise ValueError(f"Unrecognized section {section} in order.")
        if retries_strategy == 'all_final':
            func = _subfunc_retires(tool=section, strictness=strictness, seed=seed, ctx=ctx)
            run_subfuncs[f'{section}_retries'] = func

    if retries_strategy == 'final':
        func = _subfunc_retires(tool="", strictness=strictness, seed=seed, ctx=ctx)
        run_subfuncs['retries'] = func

    # Filter data to only contain dvid=1
    if modeltype == "drug_metabolite":
        orig_dataset = model.dataset
        # FIXME : remove
        model = filter_dataset(model, f'{dvid_name} != 2')

    if results is None:
        results = run_tool('modelfit', model, path=ctx.path, resume=resume)
    model_entry = ModelEntry.create(model=model, modelfit_results=results)
    next_model_entry = model_entry
    sum_subtools, sum_models, sum_inds_counts, sum_amd = [], [], [], []
    sum_subtools.append(_create_sum_subtool('start', model_entry))
    for tool_name, func in run_subfuncs.items():
        next_model, next_res = next_model_entry.model, next_model_entry.modelfit_results
        if modeltype == 'drug_metabolite' and tool_name == "structsearch":
            next_model = next_model.replace(dataset=orig_dataset)
        subresults = func(next_model, next_res)

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
                    cov_differences = cov_after - cov_before
                    if cov_differences:
                        covsearch_features = covsearch_features.expand(subresults.final_model)
                        covsearch_features += cov_differences
                        func = _subfunc_mechanistic_exploratory_covariates(
                            amd_start_model=model,
                            search_space=covsearch_features,
                            strictness=strictness,
                            mechanistic_covariates=mechanistic_covariates,
                            ctx=ctx,
                        )
                        run_subfuncs['covsearch'] = func
                next_model = subresults.final_model
                next_model_entry = ModelEntry.create(
                    model=next_model, modelfit_results=subresults.final_results
                )
            sum_subtools.append(_create_sum_subtool(tool_name, next_model_entry))
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

    final_model = next_model_entry.model
    final_results = next_model_entry.modelfit_results
    summary_errors = summarize_errors_from_entries([next_model_entry])

    # run simulation for VPC plot
    sim_model = set_simulation(final_model, n=300)
    sim_res = _run_simulation(sim_model, ctx)
    simulation_data = sim_res.table

    if final_results.predictions is not None:
        dv_vs_ipred_plot = plot_dv_vs_ipred(model, final_results.predictions, dvid_name)
        dv_vs_pred_plot = plot_dv_vs_pred(model, final_results.predictions, dvid_name)
    else:
        dv_vs_pred_plot = None
        dv_vs_ipred_plot = None
    if final_results.residuals is not None:
        cwres_vs_idv_plot = plot_cwres_vs_idv(model, final_results.residuals, dvid_name)
    else:
        cwres_vs_idv_plot = None
    if final_results.predictions is not None and final_results.residuals is not None:
        abs_cwres_vs_ipred_plot = plot_abs_cwres_vs_ipred(
            model,
            predictions=final_results.predictions,
            residuals=final_results.residuals,
            stratify_on=dvid_name,
        )
    else:
        abs_cwres_vs_ipred_plot = None
    if final_results.individual_estimates is not None:
        eta_distribution_plot = plot_eta_distributions(
            final_model, final_results.individual_estimates
        )
    else:
        eta_distribution_plot = None

    final_vpc_plot = vpc_plot(final_model, simulation_data, stratify_on=dvid_name)

    res = AMDResults(
        final_model=final_model.name,
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_individuals_count=summary_individuals_count,
        summary_errors=summary_errors,
        final_model_parameter_estimates=_table_final_parameter_estimates(
            final_results.parameter_estimates_sdcorr, final_results.standard_errors_sdcorr
        ),
        final_model_dv_vs_ipred_plot=dv_vs_ipred_plot,
        final_model_dv_vs_pred_plot=dv_vs_pred_plot,
        final_model_cwres_vs_idv_plot=cwres_vs_idv_plot,
        final_model_abs_cwres_vs_ipred_plot=abs_cwres_vs_ipred_plot,
        final_model_eta_distribution_plot=eta_distribution_plot,
        final_model_eta_shrinkage=table_final_eta_shrinkage(final_model, final_results),
        final_model_vpc_plot=final_vpc_plot,
    )
    # Since we are outside of the regular tools machinery the following is needed
    results_path = ctx.path / 'results.json'
    write_results(results=res, path=results_path)
    write_results(results=res, path=ctx.path / 'results.csv', csv=True)
    rst_path = Path(__file__).parent / 'report.rst'
    target_path = ctx.path / 'results.html'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate_report(rst_path, results_path, target_path)
    return res


def _table_final_parameter_estimates(parameter_estimates, ses):
    rse = ses / parameter_estimates
    rse.name = "RSE"
    df = pd.concat([parameter_estimates, rse], axis=1)
    return df


def _create_sum_subtool(tool_name, selected_model_entry):
    model, res = selected_model_entry.model, selected_model_entry.modelfit_results
    return {
        'tool': tool_name,
        'selected_model': model.name,
        'description': model.description,
        'n_params': len(model.parameters.nonfixed),
        'ofv': res.ofv,
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


def _run_simulation(model, ctx):
    subctx = ctx.create_subcontext('simulation')
    res = run_tool('simulation', model=model, path=subctx.path)
    return res


def _subfunc_retires(tool, strictness, seed, ctx):
    subctx = ctx.create_subcontext(f'{tool}_retries')

    def _run_retries(model, modelfit_results):
        res = run_tool(
            'retries',
            model=model,
            results=modelfit_results,
            strictness=strictness,
            scale='UCP',
            prefix_name=tool,
            seed=seed,
            path=subctx.path,
        )
        assert isinstance(res, Results)
        return res

    return _run_retries


def _subfunc_modelsearch(search_space: Tuple[Statement, ...], strictness, ctx) -> SubFunc:
    subctx = ctx.create_subcontext('modelsearch')

    def _run_modelsearch(model, modelfit_results):
        res = run_tool(
            'modelsearch',
            search_space=search_space,
            algorithm='reduced_stepwise',
            model=model,
            strictness=strictness,
            rank_type='bic',
            results=modelfit_results,
            path=subctx.path,
        )
        assert isinstance(res, Results)

        return res

    return _run_modelsearch


def _subfunc_structsearch(ctx, **kwargs) -> SubFunc:
    subctx = ctx.create_subcontext("structsearch")

    def _run_structsearch(model, modelfit_results):
        res = run_tool(
            'structsearch',
            model=model,
            results=modelfit_results,
            **kwargs,
            path=subctx.path,
        )
        assert isinstance(res, Results)
        return res

    return _run_structsearch


def _subfunc_structsearch_tmdd(
    search_space, type, strictness, dv_types, orig_dataset, ctx
) -> SubFunc:
    subctx1 = ctx.create_subcontext("modelsearch")
    subctx2 = ctx.create_subcontext("structsearch")

    def _run_structsearch_tmdd(model, modelfit_results):
        res = run_tool(
            'modelsearch',
            search_space=search_space,
            algorithm='reduced_stepwise',
            model=model,
            strictness=strictness,
            rank_type='bic',
            results=modelfit_results,
            path=subctx1.path,
        )

        final_model = res.final_model
        all_models = [
            subctx1.retrieve_model_entry(model_name).model
            for model_name in subctx1.list_all_names()
            if model_name not in ['input', 'final']
        ]

        if not has_mixed_mm_fo_elimination(final_model):
            # Only select models that have mixed MM FO elimination
            # If no model with mixed MM FO then final model from modelsearch will be used
            models_mixed_mm_fo_el = [
                model.name for model in all_models if has_mixed_mm_fo_elimination(model)
            ]
            if len(models_mixed_mm_fo_el) > 0:
                rank_all = res.summary_tool.dropna(subset='bic')[['rank']]
                rank_filtered = rank_all.query('model in @models_mixed_mm_fo_el')
                if len(rank_filtered) > 0:
                    rank_filtered = rank_filtered.sort_values(by=['rank'])
                    highest_ranked = rank_filtered.index[0]
                    final_model = retrieve_models(subctx1.path, names=[highest_ranked])[0]

        res_path = (
            subctx1.path
            / "models"
            / final_model.name
            / ("model" + subctx1.model_database.file_extension)
        )
        final_res = get_modelfit_results(final_model, res_path)

        extra_model = None
        extra_model_results = None
        n_peripherals = len(final_model.statements.ode_system.find_peripheral_compartments())
        modelfeatures = ModelFeatures.create_from_mfl_string(get_model_features(final_model))
        # Model features - 1 peripheral compartment
        modelfeatures_minus = modelfeatures.replace(
            peripherals=(Peripherals((n_peripherals - 1,)),)
        )
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
                extra_model = retrieve_models(subctx1.path, names=[highest_ranked])[0]
                if dv_types is not None:
                    extra_model = extra_model.replace(dataset=orig_dataset)
                res_path = (
                    subctx1.path
                    / "models"
                    / extra_model.name
                    / ("model" + subctx1.model_database.file_extension)
                )
                extra_model_results = get_modelfit_results(extra_model, res_path)

        # Replace original dataset if multiple DVs
        if dv_types is not None:
            final_model = final_model.replace(dataset=orig_dataset)

        res = run_tool(
            'structsearch',
            type=type,
            model=final_model,
            results=final_res,
            extra_model=extra_model,
            extra_model_results=extra_model_results,
            strictness=strictness,
            dv_types=dv_types,
            path=subctx2.path,
        )
        assert isinstance(res, Results)
        return res

    return _run_structsearch_tmdd


def _subfunc_iiv(iiv_strategy, strictness, ctx, dir_name) -> SubFunc:
    subctx = ctx.create_subcontext(dir_name)

    def _run_iiv(model, modelfit_results):
        keep = [
            str(symbol)
            for symbol in get_central_volume_and_clearance(model)
            if symbol in find_clearance_parameters(model)
        ]
        res = run_tool(
            'iivsearch',
            'top_down_exhaustive',
            iiv_strategy=iiv_strategy,
            model=model,
            results=modelfit_results,
            strictness=strictness,
            rank_type='bic',
            keep=keep,
            path=subctx.path,
        )
        assert isinstance(res, Results)
        return res

    return _run_iiv


def _subfunc_ruvsearch(dv, strictness, ctx, dir_name) -> SubFunc:
    subctx = ctx.create_subcontext(dir_name)

    def _run_ruvsearch(model, modelfit_results):
        if has_blq_transformation(model):
            skip, max_iter = ['IIV_on_RUV', 'time_varying'], 1
        else:
            skip, max_iter = [], 3
        res = run_tool(
            'ruvsearch',
            model,
            results=modelfit_results,
            skip=skip,
            max_iter=max_iter,
            dv=dv,
            strictness=strictness,
            path=subctx.path,
        )
        assert isinstance(res, Results)
        return res

    return _run_ruvsearch


def _subfunc_structural_covariates(
    amd_start_model: Model,
    search_space: ModelFeatures,
    strictness,
    ctx,
) -> SubFunc:
    subctx = ctx.create_subcontext("covsearch_structural")

    def _run_structural_covariates(model, modelfit_results):
        allowed_parameters = allowed_parameters = set(get_pk_parameters(model)).union(
            str(statement.symbol) for statement in model.statements.before_odes
        )
        # Extract all forced
        mfl = search_space
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
            mfl.create_from_mfl_statement_list(structural_searchspace),
            model=model,
            strictness=strictness,
            results=modelfit_results,
            path=subctx.path,
        )
        assert isinstance(res, Results)
        return res

    return _run_structural_covariates


def _subfunc_mechanistic_exploratory_covariates(
    amd_start_model: Model,
    search_space: ModelFeatures,
    mechanistic_covariates,
    strictness,
    ctx,
) -> SubFunc:
    covariates = set(extract_covariates(amd_start_model, search_space.mfl_statement_list()))
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

    # FIXME: Will always create these
    subcontext1 = ctx.create_subcontext("covsearch_mechanistic")
    subcontext2 = ctx.create_subcontext("covsearch_exploratory")

    def _run_mechanistic_exploratory_covariates(model, modelfit_results):
        index_offset = 0  # For naming runs

        effects = search_space.convert_to_funcs()

        if not effects:
            warnings.warn(
                'Skipping COVsearch because no effect candidates could be generated.'
                ' Check search_space definition'
                ' and .datainfo usage of "covariate" type and "continuous" flag.'
            )
            return None

        if mechanistic_covariates:
            mechanistic_searchspace, filtered_searchspace = _mechanistic_cov_extraction(
                search_space, model, mechanistic_covariates
            )

            # FIXME : Move to validation
            if not mechanistic_searchspace:
                warnings.warn(
                    'No covariate effect for given mechanistic covariates found.'
                    ' Skipping mechanistic COVsearch.'
                )
            else:
                res = run_tool(
                    'covsearch',
                    mechanistic_searchspace,
                    model=model,
                    strictness=strictness,
                    results=modelfit_results,
                    path=subcontext1.path,
                )
                covsearch_model_number = [
                    re.search(r"(\d*)$").group(1)
                    for model_name in subcontext1.list_all_names()
                    if model_name.startswith('covsearch')
                ]
                if covsearch_model_number:
                    index_offset = max(covsearch_model_number)  # Get largest number of run
                if res.final_model.name != model.name:
                    model = res.final_model
                    res_path = (
                        subcontext1.path
                        / "models"
                        / model.name
                        / ("model" + subcontext1.model_database.file_extension)
                    )
                    modelfit_results = get_modelfit_results(model, res_path)
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
            filtered_searchspace,
            model=model,
            strictness=strictness,
            results=modelfit_results,
            path=subcontext2.path,
            naming_index_offset=index_offset,
        )
        assert isinstance(res, Results)
        return res

    return _run_mechanistic_exploratory_covariates


def _mechanistic_cov_extraction(search_space, model, mechanistic_covariates):
    mechanistic_covariates = [c if isinstance(c, str) else set(c) for c in mechanistic_covariates]
    # Extract them and all forced
    mfl = search_space
    mfl_covariates = mfl.expand(model).covariate
    mechanistic_searchspace = []
    for cov_statement in mfl_covariates:
        if not cov_statement.optional.option:
            # Not optional -> Add search space (was added in structural run)
            mechanistic_searchspace.append(cov_statement)
        else:
            current_cov = []
            current_param = []
            for cov in cov_statement.covariate:
                if cov in mechanistic_covariates:
                    current_cov.append(cov)
                    current_param.append(cov_statement.parameter)
                else:
                    for param in cov_statement.parameter:
                        if {cov, param} in mechanistic_covariates:
                            current_cov.append(cov)
                            current_param.append([param])
            for cc, cp in zip(current_cov, current_param):
                mechanistic_cov = Covariate(
                    tuple(cp),
                    (cc,),
                    cov_statement.fp,
                    cov_statement.op,
                    cov_statement.optional,
                )
                mechanistic_searchspace.append(mechanistic_cov)
    if mechanistic_searchspace:
        mechanistic_searchspace = ModelFeatures.create_from_mfl_statement_list(
            mechanistic_searchspace
        )
        filtered_searchspace = mfl - mechanistic_searchspace
    else:
        filtered_searchspace = mfl
    return mechanistic_searchspace, filtered_searchspace


def _subfunc_allometry(amd_start_model: Model, allometric_variable, ctx) -> SubFunc:
    if allometric_variable is None:  # Somewhat redundant with validation function
        allometric_variable = amd_start_model.datainfo.descriptorix["body weight"][0].name

    subctx = ctx.create_subcontext("allometry")

    def _run_allometry(model, modelfit_results):
        res = run_tool(
            'allometry',
            model,
            results=modelfit_results,
            allometric_variable=allometric_variable,
            path=subctx.path,
        )
        assert isinstance(res, Results)
        return res

    return _run_allometry


def _subfunc_iov(amd_start_model, occasion, strictness, ctx) -> SubFunc:
    subctx = ctx.create_subcontext("iovsearch")

    def _run_iov(model, modelfit_results):
        res = run_tool(
            'iovsearch',
            model=model,
            results=modelfit_results,
            column=occasion,
            strictness=strictness,
            rank_type='bic',
            path=subctx.path,
        )
        assert isinstance(res, Results)
        return res

    return _run_iov


@with_runtime_arguments_type_check
def validate_input(
    input: Model,
    results: Optional[ModelfitResults] = None,
    modeltype: str = 'basic_pk',
    administration: str = 'oral',
    strategy: str = "default",
    cl_init: Optional[float] = None,
    vc_init: Optional[float] = None,
    mat_init: Optional[float] = None,
    b_init: Optional[float] = None,
    emax_init: Optional[float] = None,
    ec50_init: Optional[float] = None,
    met_init: Optional[float] = None,
    search_space: Optional[str] = None,
    lloq_method: Optional[str] = None,
    lloq_limit: Optional[str] = None,
    allometric_variable: Optional[TSymbol] = None,
    occasion: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    resume: bool = False,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    dv_types: Optional[dict[Literal[DV_TYPES], int]] = None,
    mechanistic_covariates: Optional[List[Union[str, tuple]]] = None,
    retries_strategy: Literal["final", "all_final", "skip"] = "all_final",
    seed: Optional[Union[np.random.Generator, int]] = None,
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
    ignore_datainfo_fallback: bool = False,
):
    model = input
    to_be_skipped = []

    check_list("modeltype", modeltype, ALLOWED_MODELTYPE)

    check_list("administration", administration, ALLOWED_ADMINISTRATION)

    check_list("strategy", strategy, ALLOWED_STRATEGY)

    if modeltype == 'pkpd':
        if cl_init is not None or vc_init is not None or mat_init is not None:
            raise ValueError("Cannot provide pk parameter inits for pkpd model")
        if b_init is None:
            raise ValueError("Initial estimate for baseline is needed")
        if emax_init is None:
            raise ValueError("Initial estimate for E_max is needed")
        if ec50_init is None:
            raise ValueError("Initial estimate for EC_50 is needed")
        if met_init is None:
            raise ValueError("Initial estimate for MET is needed")
    else:
        if cl_init is None:
            raise ValueError("Initial estimate for CL is needed")
        if vc_init is None:
            raise ValueError("Initial estimate for VC is needed")
        if administration in ('oral', 'ivoral') and mat_init is None:
            raise ValueError("Initial estimate for MAT is needed")

    if search_space is not None:
        try:
            ss_mfl = mfl_parse(search_space, True)
        except:  # noqa E722
            raise ValueError(f'Invalid `search_space`, could not be parsed: "{search_space}"')
        if len(ss_mfl.mfl_statement_list()) == 0:
            raise ValueError(f'`search_space` evaluated to be empty : "{search_space}')

    check_list("retries_strategy", retries_strategy, RETRIES_STRATEGIES)

    # IOVSEARCH
    if occasion is None:
        warnings.warn('IOVsearch will be skipped because occasion is None.')
        to_be_skipped.append("iovsearch")
    else:
        if occasion not in model.dataset:
            raise ValueError(
                f'Invalid `occasion`: got `{occasion}`,'
                f' must be one of {sorted(model.datainfo.names)}.'
            )
        categories = get_occasion_levels(model.dataset, occasion)
        if len(categories) < 2:
            warnings.warn(
                f'Skipping IOVsearch because there are less than two '
                f'occasion categories in column "{occasion}": {categories}.'
            )
            to_be_skipped.append("iovsearch")

    # ALLOMETRY
    if allometric_variable is None:
        if not ignore_datainfo_fallback:
            try:
                model.datainfo.descriptorix["body weight"]
            except IndexError:
                warnings.warn(
                    'Allometry will be skipped because allometric_variable is None and could'
                    ' not be inferred through .datainfo via "body weight" descriptor.'
                )
                to_be_skipped.append("allometry")
        else:
            warnings.warn(
                'Allometry will be skipped because allometric_variable is None and'
                ' ignore_datainfo_fallback is True'
            )
            to_be_skipped.append("allometry")
    else:
        validate_allometric_variable(model, allometric_variable)

    # COVSEARCH
    if mechanistic_covariates:
        allowed_covariates = get_covariates_allowed_in_covariate_effect(model)
        allowed_parameters = allowed_parameters = set(get_pk_parameters(model)).union(
            str(statement.symbol) for statement in model.statements.before_odes
        )
        for c in mechanistic_covariates:
            if isinstance(c, str):
                if c not in allowed_covariates:
                    raise ValueError(
                        f'Invalid mechanistic covariate: {c}.'
                        f' Must be in {sorted(allowed_covariates)}'
                    )
            else:
                if len(c) != 2:
                    raise ValueError(
                        f'Invalid argument in `mechanistic_covariate`: {c}.'
                        f' Tuples need to be given with one parameter and one covariate.'
                    )
                cov_found = False
                par_found = False
                for a in c:
                    if a in allowed_covariates:
                        if cov_found:
                            raise ValueError(
                                f'`mechanistic_covariates` contain invalid argument: got `{c}`,'
                                f' which contain two covariates.'
                                f' Tuples need to be given with one parameter and one covariate.'
                            )
                        cov_found = True
                    elif a in allowed_parameters:
                        if par_found:
                            raise ValueError(
                                f'`mechanistic_covariates` contain invalid argument: got `{c}`,'
                                f' which contain two parameters.'
                                f' Tuples need to be given with one parameter and one covariate.'
                            )
                        par_found = True
                    else:
                        raise ValueError(
                            f'`mechanistic_covariates` contain invalid argument: got `{c}`,'
                            f' which contain {a}.'
                            f' Tuples need to be given with one parameter and one covariate.'
                        )

    if search_space is not None:
        covsearch_features = ModelFeatures.create(covariate=ss_mfl.covariate)
        covsearch_features = covsearch_features.expand(model)
        covariates = []
        if cov_attr := covsearch_features.covariate:  # Check COVARIATE()
            covariates.extend([x for cov in cov_attr for x in cov.covariate])
        if covariates:
            allowed_covariates = get_covariates_allowed_in_covariate_effect(model)
            for covariate in sorted(covariates):
                if covariate not in allowed_covariates:
                    raise ValueError(
                        f'Invalid `search_space` because of invalid covariate found in'
                        f' search_space: got `{covariate}`,'
                        f' must be in {sorted(allowed_covariates)}.'
                    )
        else:
            if ignore_datainfo_fallback:
                warnings.warn(
                    'COVsearch will be skipped because no covariates were given'
                    ' and ignore_datainfo_fallback is True.'
                )
                to_be_skipped.append("covariates")
            elif not any(column.type == 'covariate' for column in model.datainfo):
                warnings.warn(
                    'COVsearch will be skipped because no covariates were given'
                    ' or could be extracted.'
                    ' Check search_space definition'
                    ' and .datainfo usage of "covariate" type and "continuous" flag.'
                )
                to_be_skipped.append("covariates")
    else:
        if ignore_datainfo_fallback:
            warnings.warn(
                'COVsearch will be skipped because no covariates were given'
                ' and ignore_datainfo_fallback is True.'
            )
            to_be_skipped.append("covariates")
        elif not any(column.type == 'covariate' for column in model.datainfo):
            warnings.warn(
                'COVsearch will be skipped because no covariates were given'
                ' or could be extracted.'
                ' Check search_space definition'
                ' and .datainfo usage of "covariate" type and "continuous" flag.'
            )
            to_be_skipped.append("covariates")

    return to_be_skipped
