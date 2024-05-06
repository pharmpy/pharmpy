from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence, Type, TypeVar

from pharmpy.deps import altair as alt
from pharmpy.model import Model
from pharmpy.modeling import (
    calculate_eta_shrinkage,
    plot_abs_cwres_vs_ipred,
    plot_cwres_vs_idv,
    plot_dv_vs_ipred,
    plot_dv_vs_pred,
    plot_eta_distributions,
    set_initial_estimates,
)
from pharmpy.tools.run import rank_models, summarize_errors_from_entries
from pharmpy.workflows import ModelEntry, ModelfitResults, Results
from pharmpy.workflows.hashing import ModelHash

from .funcs import summarize_individuals, summarize_individuals_count_table

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    from pharmpy.deps import numpy as np
    from pharmpy.deps import pandas as pd

DataFrame = Any  # NOTE: Should be pd.DataFrame but we want lazy loading

RANK_TYPES = frozenset(('ofv', 'lrt', 'aic', 'bic', 'mbic'))


def update_initial_estimates(model: Model, modelfit_results: Optional[ModelfitResults]):
    if modelfit_results is None:
        return model
    if not modelfit_results.minimization_successful:
        if modelfit_results.termination_cause != 'rounding_errors':
            return model

    try:
        model = set_initial_estimates(
            model, modelfit_results.parameter_estimates, move_est_close_to_bounds=True
        )
    except (ValueError, np.linalg.LinAlgError):
        pass
    return model


@dataclass(frozen=True)
class ToolResults(Results):
    summary_tool: Optional[Any] = None
    summary_models: Optional[pd.DataFrame] = None
    summary_individuals: Optional[pd.DataFrame] = None
    summary_individuals_count: Optional[pd.DataFrame] = None
    summary_errors: Optional[pd.DataFrame] = None
    final_model: Optional[Model] = None
    final_results: Optional[ModelfitResults] = None
    models: Sequence[Model] = ()
    final_model_dv_vs_ipred_plot: Optional[alt.Chart] = None
    final_model_dv_vs_pred_plot: Optional[alt.Chart] = None
    final_model_cwres_vs_idv_plot: Optional[alt.Chart] = None
    final_model_abs_cwres_vs_ipred_plot: Optional[alt.Chart] = None
    final_model_eta_distribution_plot: Optional[alt.Chart] = None
    final_model_eta_shrinkage: Optional[pd.Series] = None


T = TypeVar('T', bound=ToolResults)


def create_results(
    res_class: Type[T],
    input_model_entry: ModelEntry,
    base_model_entry: ModelEntry,
    cand_model_entries: Sequence[ModelEntry],
    rank_type: str,
    cutoff: Optional[float],
    bic_type: str = 'mixed',
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs >= 0.1)",
    n_predicted=None,
    n_expected=None,
    context=None,
    **rest,
) -> T:
    summary_tool = summarize_tool(
        cand_model_entries,
        base_model_entry,
        rank_type,
        cutoff,
        bic_type,
        strictness,
        n_predicted,
        n_expected,
    )
    if rank_type == 'lrt':
        delta_name = 'dofv'
    elif rank_type == 'mbic':
        delta_name = 'dbic'
    else:
        delta_name = f'd{rank_type}'

    base_model = base_model_entry.model
    # FIXME: Temporary until parent_model attribute has been removed, e.g. summarize_individuals fails otherwise
    base_model = base_model.replace(parent_model=base_model.name)
    base_res = base_model_entry.modelfit_results

    # FIXME: Temporary until parent_model attribute has been removed, e.g. summarize_individuals fails otherwise
    cand_models = [
        model_entry.model.replace(parent_model=model_entry.parent.name)
        for model_entry in cand_model_entries
    ]
    cand_res = [model_entry.modelfit_results for model_entry in cand_model_entries]

    summary_individuals, summary_individuals_count = summarize_tool_individuals(
        [base_model] + cand_models,
        [base_res] + cand_res,
        summary_tool['description'],
        summary_tool[delta_name],
    )
    summary_errors = summarize_errors_from_entries([base_model_entry] + cand_model_entries)

    if summary_tool['rank'].isnull().all():
        best_model = None
    else:
        best_model_name = summary_tool['rank'].idxmin()
        best_model = next(
            filter(lambda model: model.name == best_model_name, cand_models), base_model
        )

    if base_model.name != input_model_entry.model.name:
        models = [base_model] + cand_models
    else:
        # Check if any resulting models exist
        if cand_models:
            models = [base_model] + cand_models
        else:
            models = None

    if best_model is None:
        best_model = base_model

    final_results = None
    for me in cand_model_entries + [base_model_entry]:
        if me.model.name == best_model.name:
            final_results = me.modelfit_results
            break

    # Create links to input model and final model
    context.store_key("input", ModelHash(input_model_entry.model))
    context.store_key("final", ModelHash(best_model))

    plots = create_plots(best_model, final_results)

    # FIXME: Remove best_model, input_model, models when there is function to read db
    res = res_class(
        summary_tool=summary_tool,
        summary_individuals=summary_individuals,
        summary_individuals_count=summary_individuals_count,
        summary_errors=summary_errors,
        final_model=best_model,
        final_results=final_results,
        models=models,
        final_model_dv_vs_ipred_plot=plots['dv_vs_ipred'],
        final_model_dv_vs_pred_plot=plots['dv_vs_pred'],
        final_model_cwres_vs_idv_plot=plots['cwres_vs_idv'],
        final_model_abs_cwres_vs_ipred_plot=plots['abs_cwres_vs_ipred'],
        final_model_eta_distribution_plot=plots['eta_distribution'],
        final_model_eta_shrinkage=table_final_eta_shrinkage(best_model, final_results),
        **rest,
    )

    return res


def summarize_tool(
    model_entries: Sequence[ModelEntry],
    start_model_entry: ModelEntry,
    rank_type: str,
    cutoff: Optional[float],
    bic_type: str = 'mixed',
    strictness: Optional[str] = None,
    n_predicted=None,
    n_expected=None,
) -> DataFrame:
    start_model_res = start_model_entry.modelfit_results
    models_res = [model_entry.modelfit_results for model_entry in model_entries]

    if rank_type == 'mbic':
        rank_type = 'bic'
        if len(model_entries) > 0:
            multiple_testing = True
            n_predicted_models = len(model_entries) if n_predicted is None else n_predicted
            n_expected_models = 1 if n_predicted is None else n_expected
        else:  # This can happen if the search space of e.g. modelsearch only includes the base model
            multiple_testing = False
            n_predicted_models = None
            n_expected_models = None
    else:
        multiple_testing = False
        n_predicted_models = None
        n_expected_models = None

    start_model = start_model_entry.model
    models = [model_entry.model for model_entry in model_entries]

    df_rank = rank_models(
        start_model,
        start_model_res,
        models,
        models_res,
        strictness=strictness,
        rank_type=rank_type,
        cutoff=cutoff,
        bic_type=bic_type,
        multiple_testing=multiple_testing,
        mult_test_p=n_predicted_models,
        mult_test_e=n_expected_models,
    )
    if rank_type != "lrt" and df_rank.dropna(subset=rank_type).shape[0] == 0:
        raise ValueError("All models fail the strictness criteria!")

    rows = {}

    for model_entry in [start_model_entry] + model_entries:
        model = model_entry.model
        parent_model = model_entry.parent if model_entry.parent is not None else model
        description = model.description
        n_params = len(model.parameters.nonfixed)
        if model.name == start_model.name:
            d_params = 0
        else:
            d_params = n_params - len(parent_model.parameters.nonfixed)
        rows[model.name] = (description, n_params, d_params, parent_model.name)

    colnames = ['description', 'n_params', 'd_params', 'parent_model']
    index = pd.Index(rows.keys(), name='model')
    df_descr = pd.DataFrame(rows.values(), index=index, columns=colnames)

    df = pd.concat([df_descr, df_rank], axis=1)
    df['parent_model'] = df.pop('parent_model')

    df_sorted = df.reindex(df_rank.index)

    assert df_sorted is not None
    return df_sorted


def summarize_tool_individuals(
    models: Sequence[Model],
    models_res: Sequence[ModelfitResults],
    description_col: pd.Series,
    rank_type_col: pd.Series,
):
    summary_individuals = summarize_individuals(models, models_res)
    summary_individuals = summary_individuals.join(description_col, how='inner')
    col_to_move = summary_individuals.pop('description')
    summary_individuals.insert(0, 'description', col_to_move)

    suminds_count = summarize_individuals_count_table(df=summary_individuals)
    if suminds_count is None:
        summary_individuals_count = None
    else:
        suminds_count.insert(0, description_col.name, description_col)
        suminds_count.insert(1, rank_type_col.name, rank_type_col)
        suminds_count['parent_model'] = suminds_count.pop('parent_model')
        summary_individuals_count = suminds_count.sort_values(
            by=[rank_type_col.name], ascending=False
        )
    return summary_individuals, summary_individuals_count


def create_plots(model: Model, results: ModelfitResults):
    if 'dvid' in model.datainfo.types:
        dvid_name = model.datainfo.typeix['dvid'].names[0]
    else:
        dvid_name = None

    pred = results.predictions
    res = results.residuals

    if pred is not None and 'PRED' in pred.columns:
        dv_vs_pred_plot = plot_dv_vs_pred(model, results.predictions, dvid_name)
    else:
        dv_vs_pred_plot = None

    if pred is not None and ('IPRED' in pred.columns or 'CIPREDI' in pred.columns):
        dv_vs_ipred_plot = plot_dv_vs_ipred(model, results.predictions, dvid_name)
    else:
        dv_vs_ipred_plot = None

    if (
        pred is not None
        and res is not None
        and ('IPRED' in pred.columns or 'CIPREDI' in pred.columns)
        and 'CWRES' in res.columns
    ):
        cwres_vs_idv_plot = plot_cwres_vs_idv(model, results.residuals, dvid_name)
    else:
        cwres_vs_idv_plot = None

    if pred is not None and res is not None and 'CWRES' in res.columns:
        abs_cwres_vs_ipred_plot = plot_abs_cwres_vs_ipred(
            model,
            predictions=results.predictions,
            residuals=results.residuals,
            stratify_on=dvid_name,
        )
    else:
        abs_cwres_vs_ipred_plot = None

    if results.individual_estimates is not None and results.individual_estimates.any(axis=None):
        eta_distribution_plot = plot_eta_distributions(model, results.individual_estimates)
    else:
        eta_distribution_plot = None

    return {
        'dv_vs_pred': dv_vs_pred_plot,
        'dv_vs_ipred': dv_vs_ipred_plot,
        'cwres_vs_idv': cwres_vs_idv_plot,
        'abs_cwres_vs_ipred': abs_cwres_vs_ipred_plot,
        'eta_distribution': eta_distribution_plot,
    }


def table_final_parameter_estimates(parameter_estimates, ses):
    rse = ses / parameter_estimates
    rse.name = "RSE"
    df = pd.concat([parameter_estimates, rse], axis=1)
    return df


def table_final_eta_shrinkage(model, results):
    if results.parameter_estimates is not None and results.individual_estimates is not None:
        eta_shrinkage = calculate_eta_shrinkage(
            model, results.parameter_estimates, results.individual_estimates
        )
    else:
        eta_shrinkage = None
    return eta_shrinkage
