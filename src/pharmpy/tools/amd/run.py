import re
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

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
    plot_vpc,
    set_dataset,
    set_simulation,
)
from pharmpy.modeling.blq import has_blq_transformation, transform_blq
from pharmpy.modeling.common import convert_model, filter_dataset
from pharmpy.modeling.covariate_effect import get_covariates_allowed_in_covariate_effect
from pharmpy.modeling.parameter_variability import get_occasion_levels
from pharmpy.modeling.tmdd import DV_TYPES
from pharmpy.tools import retrieve_models
from pharmpy.tools.allometry.tool import validate_allometric_variable
from pharmpy.tools.common import table_final_eta_shrinkage
from pharmpy.tools.mfl.feature.covariate import covariates as extract_covariates
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.mfl.parse import parse as mfl_parse
from pharmpy.tools.mfl.statement.feature.covariate import Covariate
from pharmpy.tools.mfl.statement.feature.peripherals import Peripherals
from pharmpy.tools.mfl.statement.statement import Statement
from pharmpy.tools.run import is_strictness_fulfilled, run_subtool, summarize_errors_from_entries
from pharmpy.workflows import Context, ModelEntry, Results, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.model_database.local_directory import get_modelfit_results
from pharmpy.workflows.results import ModelfitResults

from ...internals.fs.path import path_absolute
from .results import AMDResults

ALLOWED_STRATEGY = ["default", "reevaluation", "SIR", "SRI", "RSI"]
ALLOWED_ADMINISTRATION = ["iv", "oral", "ivoral"]
ALLOWED_MODELTYPE = ['basic_pk', 'pkpd', 'drug_metabolite', 'tmdd']
RETRIES_STRATEGIES = ["final", "all_final", "skip"]
DEFAULT_STRICTNESS = "minimization_successful or (rounding_errors and sigdigs>=0.1)"


def create_workflow(
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
    lloq_limit: Optional[float] = None,
    allometric_variable: Optional[TSymbol] = None,
    occasion: Optional[str] = None,
    strictness: str = DEFAULT_STRICTNESS,
    dv_types: Optional[dict[Literal[DV_TYPES], int]] = None,
    mechanistic_covariates: Optional[list[Union[str, tuple[str]]]] = None,
    retries_strategy: Literal["final", "all_final", "skip"] = "all_final",
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
    ignore_datainfo_fallback: bool = False,
    _E: Optional[dict[str, Union[float, str]]] = None,
):
    """Run Automatic Model Development (AMD) tool

    Parameters
    ----------
    input : Model, Path or DataFrame
        Starting model or dataset
    results : ModelfitResults
        Reults of input if input is a model
    modeltype : str
        Type of model to build. Valid strings are 'basic_pk', 'pkpd', 'drug_metabolite' and 'tmdd'
    administration : str
        Route of administration. Either 'iv', 'oral' or 'ivoral'
    strategy : str
        Run algorithm for AMD procedure. Valid options are 'default', 'reevaluation', 'SIR', 'SRI', and 'RSI'.
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
        MFL for search space for structural and covariate model
    lloq_method : str
        Method for how to remove LOQ data. See `transform_blq` for list of available methods
    lloq_limit : float
        Lower limit of quantification. If None LLOQ column from dataset will be used
    allometric_variable: str or Symbol
        Variable to use for allometry. This option is deprecated.
        Please use ALLOMETRY in the mfl instead.
    occasion : str
        Name of occasion column
    strictness : str
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
    parameter_uncertainty_method: {'SANDWICH', 'SMAT', 'RMAT', 'EFIM'} or None
        Parameter uncertainty method.
    ignore_datainfo_fallback : bool
        Ignore using datainfo to get information not given by the user. Default is False
    _E: dict
        EXPERIMENTAL FEATURE. Dictionary of different E-values used in mBIC.

    Returns
    -------
    AMDResults
        Results for the run

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import run_amd, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> res = run_amd(model, results=results)      # doctest: +SKIP

    See also
    --------
    run_iiv
    run_tool

    """
    kwargs = locals()
    # Needs to be done client side
    if isinstance(input, str):
        kwargs['input'] = path_absolute(Path(input))

    wb = WorkflowBuilder(name='amd')
    start_task = Task(
        'run_amd_task',
        run_amd_task,
        *tuple(kwargs.values()),
    )
    wb.add_task(start_task)
    task_results = Task('results', _results)
    wb.add_task(task_results, predecessors=[start_task])
    return Workflow(wb)


# FIXME: refactor into separate tasks
def run_amd_task(
    context: Context,
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
    lloq_limit: Optional[float] = None,
    allometric_variable: Optional[TSymbol] = None,
    occasion: Optional[str] = None,
    strictness: str = DEFAULT_STRICTNESS,
    dv_types: Optional[dict[Literal[DV_TYPES], int]] = None,
    mechanistic_covariates: Optional[list[Union[str, tuple[str]]]] = None,
    retries_strategy: Literal["final", "all_final", "skip"] = "all_final",
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
    ignore_datainfo_fallback: bool = False,
    _E: Optional[dict[str, Union[float, str]]] = None,
):
    kwargs = locals()

    context.log_info("Starting tool amd")
    rng = context.create_rng(0)

    from pharmpy.model.external import nonmem  # FIXME: We should not depend on NONMEM

    if search_space is not None:
        try:
            ss_mfl = mfl_parse(search_space, True)
        except:  # noqa E722
            raise ValueError(f'Invalid `search_space`, could not be parsed: "{search_space}"')
    else:
        ss_mfl = ModelFeatures()

    if ss_mfl.allometry is not None:
        # Take it out and put back later
        if allometric_variable is not None:
            raise ValueError(
                "Having both allometric_variable and ALLOMETRY in the mfl is not allowed"
            )
        mfl_allometry = ss_mfl.allometry
        ss_mfl = ss_mfl.replace(allometry=None)
    else:
        mfl_allometry = None

    if modeltype == 'pkpd':
        dv = 2
        iiv_strategy = 'pd_fullblock'
    else:
        dv = None
        iiv_strategy = 'fullblock'

    if isinstance(input, str):
        input = Path(input)

    if isinstance(input, Path):
        model = create_basic_pk_model(
            administration,
            dataset_path=input,
            cl_init=cl_init,
            vc_init=vc_init,
            mat_init=mat_init,
        )
        model = convert_model(model, 'nonmem')  # FIXME: Workaround for results retrieval system
    elif isinstance(input, pd.DataFrame):
        model = create_basic_pk_model(
            administration,
            cl_init=cl_init,
            vc_init=vc_init,
            mat_init=mat_init,
        )
        model = set_dataset(model, input, datatype='nonmem')
        model = convert_model(model, 'nonmem')  # FIXME: Workaround for results retrieval system
    elif isinstance(input, nonmem.model.Model):
        model = input
        model = model.replace(name='start')
        context.store_input_model_entry(model)
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
    kwargs['input'] = model
    del kwargs['context']
    later_input_validation(**kwargs)
    to_be_skipped = check_skip(
        context, model, occasion, allometric_variable, ignore_datainfo_fallback, search_space
    )

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
        context.log_warning('Skipping allometry since modeltype is "pkpd"')
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
                    "DIRECTEFFECT([LINEAR, EMAX, SIGMOID]);"
                    "EFFECTCOMP([LINEAR, EMAX, SIGMOID]);"
                    "INDIRECTEFFECT([LINEAR, EMAX, SIGMOID], *)",
                    True,
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
    if mfl_allometry is not None:
        modelsearch_features = modelsearch_features.replace(allometry=mfl_allometry)

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
            model = model.replace(dataset=model.dataset.reset_index())

    run_subfuncs = {}

    for section in order:
        if section == 'structural':
            if modeltype != 'pkpd':
                run_subfuncs['structural_covariates'] = _subfunc_structural_covariates(
                    amd_start_model=model,
                    search_space=covsearch_features,
                    strictness=strictness,
                    ctx=context,
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
                    ctx=context,
                )
                run_subfuncs['structsearch'] = func
            elif modeltype == 'tmdd':
                func = _subfunc_structsearch_tmdd(
                    search_space=modelsearch_features,
                    type=modeltype,
                    strictness=strictness,
                    dv_types=dv_types,
                    orig_dataset=orig_dataset,
                    ctx=context,
                )
                run_subfuncs['structsearch'] = func
            else:
                func = _subfunc_modelsearch(
                    search_space=modelsearch_features, strictness=strictness, E=_E, ctx=context
                )
                run_subfuncs['modelsearch'] = func
            # Perfomed 'after' modelsearch
            if modeltype == 'drug_metabolite':
                func = _subfunc_structsearch(
                    type=modeltype,
                    search_space=structsearch_features,
                    ctx=context,
                )
                run_subfuncs['structsearch'] = func
        elif section == 'iivsearch':
            if 'iivsearch' in run_subfuncs.keys():
                run_name = 'rerun_iivsearch'
                func = _subfunc_iiv(
                    iiv_strategy='no_add',
                    strictness=strictness,
                    E=_E,
                    ctx=context,
                    dir_name="rerun_iivsearch",
                )
            else:
                run_name = 'iivsearch'
                func = _subfunc_iiv(
                    iiv_strategy=iiv_strategy,
                    strictness=strictness,
                    E=_E,
                    ctx=context,
                    dir_name="iivsearch",
                )
            run_subfuncs[run_name] = func
        elif section == 'iovsearch':
            func = _subfunc_iov(
                amd_start_model=model, occasion=occasion, strictness=strictness, E=_E, ctx=context
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
                    ctx=context,
                    dir_name=f'{run_name}_drug',
                )
                run_subfuncs[f'{run_name}_drug'] = func
                # And one for the metabolite
                func = _subfunc_ruvsearch(
                    dv=2,
                    strictness=strictness,
                    ctx=context,
                    dir_name=f'{run_name}_metabolite',
                )
                run_subfuncs[f'{run_name}_metabolite'] = func
            elif modeltype == 'tmdd' and dv_types is not None:
                for key, value in dv_types.items():
                    func = _subfunc_ruvsearch(
                        dv=value,
                        strictness=strictness,
                        ctx=context,
                        dir_name=f'{run_name}_tmdd_{key}',
                    )
                    run_subfuncs[f'ruvsearch_{key}'] = func
            else:
                func = _subfunc_ruvsearch(
                    dv=dv, strictness=strictness, ctx=context, dir_name=run_name
                )
                run_subfuncs[f'{run_name}'] = func
        elif section == 'allometry':
            if mfl_allometry is None:
                func = _subfunc_allometry(
                    amd_start_model=model, allometric_variable=allometric_variable, ctx=context
                )
                run_subfuncs['allometry'] = func
        elif section == 'covariates':
            func = _subfunc_mechanistic_exploratory_covariates(
                amd_start_model=model,
                search_space=covsearch_features,
                mechanistic_covariates=mechanistic_covariates,
                strictness=strictness,
                ctx=context,
            )
            run_subfuncs['covsearch'] = func
        else:
            raise ValueError(f"Unrecognized section {section} in order.")
        if retries_strategy == 'all_final':
            func = _subfunc_retries(
                tool=section, strictness=strictness, seed=context.spawn_seed(rng), ctx=context
            )
            run_subfuncs[f'{section}_retries'] = func

    if retries_strategy == 'final':
        func = _subfunc_retries(
            tool="", strictness=strictness, seed=context.spawn_seed(rng), ctx=context
        )
        run_subfuncs['retries'] = func

    # Filter data to only contain dvid=1
    if modeltype == "drug_metabolite":
        orig_dataset = model.dataset
        # FIXME : remove
        model = filter_dataset(model, f'{dvid_name} != 2')
        model = model.replace(dataset=model.dataset.reset_index())

    if results is None:
        context.log_info('Running base model')
        results = run_subtool('modelfit', context, model_or_models=model)
        if not is_strictness_fulfilled(model, results, DEFAULT_STRICTNESS):
            context.log_warning('Base model failed strictness')
        model = model.replace(name='base')
        context.store_model_entry(ModelEntry.create(model=model, modelfit_results=results))

    model_entry = ModelEntry.create(model=model, modelfit_results=results)
    next_model_entry = model_entry
    sum_subtools = []
    sum_models = dict()
    sum_subtools.append(_create_sum_subtool('start', model_entry))
    for tool_name, func in run_subfuncs.items():
        next_model, next_res = next_model_entry.model, next_model_entry.modelfit_results
        if modeltype == 'drug_metabolite' and tool_name == "structsearch":
            next_model = next_model.replace(dataset=orig_dataset)
        subresults = func(next_model, next_res)

        if subresults is None:
            continue
        elif subresults == "CRITICAL":
            # Substitute for abort_workflow
            return None
        else:
            final_model = subresults.final_model.replace(name=f"final_{tool_name}")
            final_model_entry = ModelEntry.create(
                model=final_model, modelfit_results=subresults.final_results, parent=next_model
            )
            context.store_model_entry(final_model_entry)
            if (mfl_allometry is not None and tool_name == 'modelsearch') or (
                tool_name == "allometry" and 'allometry' in order[: order.index('covariates')]
            ):
                cov_before = ModelFeatures.create_from_mfl_string(get_model_features(next_model))
                cov_after = ModelFeatures.create_from_mfl_string(get_model_features(final_model))
                cov_differences = cov_after - cov_before
                if cov_differences:
                    covsearch_features = covsearch_features.expand(final_model)
                    covsearch_features += cov_differences
                    func = _subfunc_mechanistic_exploratory_covariates(
                        amd_start_model=model,
                        search_space=covsearch_features,
                        strictness=strictness,
                        mechanistic_covariates=mechanistic_covariates,
                        ctx=context,
                    )
                    run_subfuncs['covsearch'] = func
            next_model = final_model
            next_model_entry = final_model_entry
            sum_subtools.append(_create_sum_subtool(tool_name, next_model_entry))
            sum_models[tool_name] = subresults.summary_models

    # FIXME: add start model
    summary_models = _create_model_summary(sum_models)
    summary_tool = _create_tool_summary(sum_subtools)

    if summary_models is None:
        context.log_warning(
            'AMDResults.summary_models is None because none of the tools yielded a summary.'
        )

    final_model = next_model_entry.model
    final_results = next_model_entry.modelfit_results
    summary_errors = summarize_errors_from_entries([next_model_entry])

    context.store_final_model_entry(final_model)

    # run simulation for VPC plot
    # NOTE: The seed is set to be in range for NONMEM
    sim_model = set_simulation(final_model, n=300, seed=context.spawn_seed(rng, n=32))
    sim_res = _run_simulation(sim_model, context)
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

    if not simulation_data.empty:
        final_vpc_plot = plot_vpc(final_model, simulation_data, stratify_on=dvid_name)
    else:
        context.log_warning("No vpc could be generated. Did the simulation fail?")
        final_vpc_plot = None

    res = AMDResults(
        final_model=final_model,
        final_results=final_results,
        summary_tool=summary_tool,
        summary_models=summary_models,
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


def _create_model_summary(summaries):
    dfs = []
    for tool_name, df in summaries.items():
        df = df.reset_index()
        df['tool'] = [tool_name] * len(df)
        df.set_index(['tool', 'step', 'model'], inplace=True)
        dfs.append(df)
    model_summary = pd.concat(dfs, axis=0)
    return model_summary


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
    res = run_subtool('simulation', ctx, model=model)
    return res


def _subfunc_retries(tool, strictness, seed, ctx):
    def _run_retries(model, modelfit_results):
        res = run_subtool(
            'retries',
            ctx,
            name=f'{tool}_retries',
            model=model,
            results=modelfit_results,
            strictness=strictness,
            scale='UCP',
            prefix_name=tool,
            seed=seed,
        )
        assert isinstance(res, Results)
        return res

    return _run_retries


def _subfunc_modelsearch(search_space: tuple[Statement, ...], strictness, E, ctx) -> SubFunc:
    def _run_modelsearch(model, modelfit_results):
        if E and 'modelsearch' in E.keys():
            rank_type = 'mbic'
            e = E['modelsearch']
        else:
            rank_type = 'bic'
            e = None

        res = run_subtool(
            'modelsearch',
            ctx,
            model=model,
            results=modelfit_results,
            search_space=search_space,
            algorithm='reduced_stepwise',
            strictness=strictness,
            rank_type=rank_type,
            E=e,
        )
        assert isinstance(res, Results)
        if res.final_model is None:
            ctx.log_message("critical", "No model passed strictness criteria in modelsearch")
            res = "CRITICAL"

        return res

    return _run_modelsearch


def _subfunc_structsearch(ctx, **kwargs) -> SubFunc:
    def _run_structsearch(model, modelfit_results):
        res = run_subtool(
            'structsearch',
            ctx,
            model=model,
            results=modelfit_results,
            **kwargs,
        )
        assert isinstance(res, Results)
        return res

    return _run_structsearch


def _subfunc_structsearch_tmdd(
    search_space, type, strictness, dv_types, orig_dataset, ctx
) -> SubFunc:
    def _run_structsearch_tmdd(model, modelfit_results):
        res = run_subtool(
            'modelsearch',
            ctx,
            model=model,
            results=modelfit_results,
            search_space=search_space,
            algorithm='reduced_stepwise',
            strictness=strictness,
            rank_type='bic',
        )

        final_model = res.final_model
        subctx1 = ctx.get_subcontext('modelsearch1')
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

        res = run_subtool(
            'structsearch',
            ctx,
            model=final_model,
            results=final_res,
            type=type,
            extra_model=extra_model,
            extra_model_results=extra_model_results,
            strictness=strictness,
            dv_types=dv_types,
        )
        assert isinstance(res, Results)
        return res

    return _run_structsearch_tmdd


def _subfunc_iiv(iiv_strategy, strictness, E, ctx, dir_name) -> SubFunc:
    def _run_iiv(model, modelfit_results):
        if E and 'iivsearch' in E.keys():
            rank_type = 'mbic'
            e_p, e_q = E['iivsearch']
        else:
            rank_type = 'bic'
            e_p, e_q = None, None

        keep = [
            str(symbol)
            for symbol in get_central_volume_and_clearance(model)
            if symbol in find_clearance_parameters(model)
        ]
        res = run_subtool(
            'iivsearch',
            ctx,
            name=dir_name,
            model=model,
            results=modelfit_results,
            algorithm='top_down_exhaustive',
            iiv_strategy=iiv_strategy,
            strictness=strictness,
            rank_type=rank_type,
            E_p=e_p,
            E_q=e_q,
            keep=keep,
        )
        assert isinstance(res, Results)
        return res

    return _run_iiv


def _subfunc_ruvsearch(dv, strictness, ctx, dir_name) -> SubFunc:
    def _run_ruvsearch(model, modelfit_results):
        if has_blq_transformation(model):
            skip, max_iter = ['IIV_on_RUV', 'time_varying'], 1
        else:
            skip, max_iter = [], 3
        res = run_subtool(
            'ruvsearch',
            ctx,
            name=dir_name,
            model=model,
            results=modelfit_results,
            skip=skip,
            max_iter=max_iter,
            dv=dv,
            strictness=strictness,
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
    def _run_structural_covariates(model, modelfit_results):
        allowed_parameters = set(get_pk_parameters(model)).union(
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
            ctx.log_warning(
                f'{skipped_parameters} missing in start model and structural covariate effect cannot be added'
                ' Might be added during a later COVsearch step if possible.'
            )
        if not structural_searchspace and not skipped_parameters:
            # Uneccessary to warn (?)
            return None
        elif not structural_searchspace:
            ctx.log_warning(
                'No applicable structural covariates found in search space. Skipping structural_COVsearch'
            )
            return None
        struct_searchspace = mfl.create_from_mfl_statement_list(structural_searchspace)
        res = run_subtool(
            'covsearch',
            ctx,
            name='covsearch_structural',
            model=model,
            results=modelfit_results,
            search_space=struct_searchspace,
            strictness=strictness,
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
        ctx.log_warning(
            'COVsearch will most likely be skipped because no covariates could be found.'
            ' Check search_space definition'
            ' and .datainfo usage of "covariate" type and "continuous" flag.'
        )

    def _run_mechanistic_exploratory_covariates(model, modelfit_results):
        index_offset = 0  # For naming runs

        effects = search_space.convert_to_funcs(model=model)

        if not effects:
            ctx.log_warning(
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
                ctx.log_warning(
                    'No covariate effect for given mechanistic covariates found.'
                    ' Skipping mechanistic COVsearch.'
                )
            else:
                res = run_subtool(
                    'covsearch',
                    ctx,
                    name='covsearch_mechanistic',
                    model=model,
                    results=modelfit_results,
                    search_space=mechanistic_searchspace,
                    strictness=strictness,
                )
                subcontext1 = ctx.get_subcontext('covsearch_mechanistic')
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

        res = run_subtool(
            'covsearch',
            ctx,
            name='covsearch_exploratory',
            model=model,
            results=modelfit_results,
            search_space=filtered_searchspace,
            strictness=strictness,
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

    def _run_allometry(model, modelfit_results):
        res = run_subtool(
            'allometry',
            ctx,
            model=model,
            results=modelfit_results,
            allometric_variable=allometric_variable,
        )
        assert isinstance(res, Results)
        return res

    return _run_allometry


def _subfunc_iov(amd_start_model, occasion, strictness, E, ctx) -> SubFunc:
    def _run_iov(model, modelfit_results):
        if E and 'iovsearch' in E.keys():
            rank_type = 'mbic'
            e = E['iovsearch']
        else:
            rank_type = 'bic'
            e = None

        res = run_subtool(
            'iovsearch',
            ctx,
            model=model,
            results=modelfit_results,
            column=occasion,
            strictness=strictness,
            rank_type=rank_type,
            E=e,
        )
        assert isinstance(res, Results)
        return res

    return _run_iov


def check_skip(
    context,
    model: Model,
    occasion: str,
    allometric_variable: str,
    ignore_datainfo_fallback: bool = False,
    search_space: Optional[str] = None,
):
    to_be_skipped = []

    # IOVSEARCH
    if occasion is None:
        context.log_warning('IOVsearch will be skipped because occasion is None.')
        to_be_skipped.append("iovsearch")
    else:
        categories = get_occasion_levels(model.dataset, occasion)
        if len(categories) < 2:
            context.log_warning(
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
                context.log_warning(
                    'Allometry will be skipped because allometric_variable is None and could'
                    ' not be inferred through .datainfo via "body weight" descriptor.'
                )
                to_be_skipped.append("allometry")
        else:
            context.log_warning(
                'Allometry will be skipped because allometric_variable is None and'
                ' ignore_datainfo_fallback is True'
            )
            to_be_skipped.append("allometry")

    if search_space is not None:
        ss_mfl = mfl_parse(search_space, True)
        covsearch_features = ModelFeatures.create(covariate=ss_mfl.covariate)
        covsearch_features = covsearch_features.expand(model)
        covariates = []
        if cov_attr := covsearch_features.covariate:
            covariates.extend([x for cov in cov_attr for x in cov.covariate])
        if not covariates:
            if ignore_datainfo_fallback:
                context.log_warning(
                    'COVsearch will be skipped because no covariates were given'
                    ' and ignore_datainfo_fallback is True.'
                )
                to_be_skipped.append("covariates")
            elif not any(column.type == 'covariate' for column in model.datainfo):
                context.log_warning(
                    'COVsearch will be skipped because no covariates were given'
                    ' or could be extracted.'
                    ' Check search_space definition'
                    ' and .datainfo usage of "covariate" type and "continuous" flag.'
                )
                to_be_skipped.append("covariates")
    else:
        if ignore_datainfo_fallback:
            context.log_warning(
                'COVsearch will be skipped because no covariates were given'
                ' and ignore_datainfo_fallback is True.'
            )
            to_be_skipped.append("covariates")
        elif not any(column.type == 'covariate' for column in model.datainfo):
            context.log_warning(
                'COVsearch will be skipped because no covariates were given'
                ' or could be extracted.'
                ' Check search_space definition'
                ' and .datainfo usage of "covariate" type and "continuous" flag.'
            )
            to_be_skipped.append("covariates")

    return to_be_skipped


def _results(context, res):
    context.log_info("Finishing tool amd")
    return res


@with_runtime_arguments_type_check
def validate_input(
    input: Union[Model, Path, str, pd.DataFrame],
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
    lloq_limit: Optional[float] = None,
    allometric_variable: Optional[TSymbol] = None,
    occasion: Optional[str] = None,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    dv_types: Optional[dict[Literal[DV_TYPES], int]] = None,
    mechanistic_covariates: Optional[list[Union[str, tuple]]] = None,
    retries_strategy: Literal["final", "all_final", "skip"] = "all_final",
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
    ignore_datainfo_fallback: bool = False,
    _E: Optional[dict[str, Union[float, str, Sequence[Union[float, str]]]]] = None,
):
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

        if (
            administration == "oral"
            and ss_mfl.absorption is not None
            and "INST" in (a.name for a in ss_mfl.absorption.modes)
        ):
            raise ValueError(
                'The given search space have instantaneous absorption (´INST´)'
                ' which is not allowed with ´oral´ administration.'
            )

    check_list("retries_strategy", retries_strategy, RETRIES_STRATEGIES)

    if _E:
        if any(value in (0.0, '0%') for value in _E.values()):
            raise ValueError('E-values in `_E` cannot be 0')


def later_input_validation(
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
    lloq_limit: Optional[float] = None,
    allometric_variable: Optional[TSymbol] = None,
    occasion: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    resume: bool = False,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs>=0.1)",
    dv_types: Optional[dict[Literal[DV_TYPES], int]] = None,
    mechanistic_covariates: Optional[list[Union[str, tuple]]] = None,
    retries_strategy: Literal["final", "all_final", "skip"] = "all_final",
    seed: Optional[Union[np.random.Generator, int]] = None,
    parameter_uncertainty_method: Optional[Literal['SANDWICH', 'SMAT', 'RMAT', 'EFIM']] = None,
    ignore_datainfo_fallback: bool = False,
    _E: Optional[dict[str, Union[float, str, Sequence[Union[float, str]]]]] = None,
):
    # FIXME: This function should be removed and refactored into validate_inputs
    # and optionally give warnings/errors during the run
    # it currently depends on a model being created if a dataset was input to AMD.
    model = input

    # IOVSEARCH
    if occasion is not None and occasion not in model.dataset:
        raise ValueError(
            f'Invalid `occasion`: got `{occasion}`,'
            f' must be one of {sorted(model.datainfo.names)}.'
        )

    # ALLOMETRY
    if allometric_variable is not None:
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
        ss_mfl = mfl_parse(search_space, True)
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
