import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    from pharmpy.deps import numpy as np
    import pandas as pd

from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.internals.math import is_posdef, nearest_positive_definite
from pharmpy.model import Model
from pharmpy.modeling import (
    calculate_parameters_from_ucp,
    calculate_ucp_scale,
    create_rng,
    sample_parameters_uniformly,
    set_initial_estimates,
)
from pharmpy.tools.common import ToolResults, create_results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import summarize_modelfit_results_from_entries
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

SCALES = frozenset(('normal', 'UCP'))


@dataclass
class Retry:
    modelentry: ModelEntry
    number_of_retries: int


def create_workflow(
    model: Optional[Model] = None,
    results: Optional[ModelfitResults] = None,
    number_of_candidates: int = 5,
    fraction: float = 0.1,
    use_initial_estimates: bool = False,
    strictness: Optional[str] = "minimization_successful or (rounding_errors and sigdigs >= 0.1)",
    scale: Optional[Literal[tuple(SCALES)]] = "UCP",
    prefix_name: Optional[str] = "",  # FIXME : Remove once new database has been implemented
    seed: Optional[Union[np.random.Generator, int]] = None,
):
    """
    Run retries tool.

    Parameters
    ----------
    model : Optional[Model], optional
        Model object to run retries on. The default is None.
    results : Optional[ModelfitResults], optional
        Connected ModelfitResults object. The default is None.
    number_of_candidates : int, optional
        Number of retry candidates to run. The default is 5.
    fraction: float
        Determines allowed increase/decrease from initial parameter estimate. Default is 0.1 (10%)
    use_initial_estimates : bool
        Use initial parameter estimates instead of final estimates of input model when creating candidate models.
    strictness : Optional[str], optional
        Strictness criteria. The default is "minimization_successful or (rounding_errors and sigdigs >= 0.1)".
    scale : {'normal', 'UCP'}
        Which scale to update the initial values on. Either normal scale or UCP scale.
    prefix_name: Optional[str]
        Prefix the candidate model names with given string.
    seed: int or rng
        Random number generator or seed to be used.

    Returns
    -------
    RetriesResults
        Retries tool results object.

    """

    wb = WorkflowBuilder(name='retries')

    if model is not None:
        start_task = Task('Start_retries', _start, results, model)
    else:
        # Remove?
        start_task = Task('Start_retries', _start)

    wb.add_task(start_task)

    candidate_tasks = []
    seed = create_rng(seed)
    for i in range(1, number_of_candidates + 1):
        new_candidate_task = Task(
            f"Create_candidate_{i}",
            create_random_init_model,
            i,
            scale,
            fraction,
            use_initial_estimates,
            prefix_name,
            seed,
        )
        wb.add_task(new_candidate_task, predecessors=start_task)
        candidate_tasks.append(new_candidate_task)
    task_gather = Task('Gather', lambda *retries: retries)
    wb.add_task(task_gather, predecessors=[start_task] + candidate_tasks)

    results_task = Task('Results', task_results, strictness)
    wb.add_task(results_task, predecessors=task_gather)

    return Workflow(wb)


def _start(results, model):
    # Convert to modelentry
    return ModelEntry.create(model=model, modelfit_results=results)


def create_random_init_model(
    context, index, scale, fraction, use_initial_estimates, prefix_name, seed, modelentry
):
    original_model = modelentry.model
    # Update inits once before running
    if not use_initial_estimates and modelentry.modelfit_results:
        original_model = set_initial_estimates(
            original_model, modelentry.modelfit_results.parameter_estimates
        )

    original_model = convert_to_posdef(original_model)

    # Add any description?
    if prefix_name:
        name = f'{prefix_name}_retries_run{index}'
    else:
        name = f'retries_run{index}'
    new_candidate_model = original_model.replace(name=name)

    try_number = 1
    if scale == "normal":
        maximum_tests = 20  # TODO : Convert to argument

        for try_number in range(1, maximum_tests + 1):
            new_parameters = create_new_parameter_inits(new_candidate_model, fraction, scale, seed)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error",
                    message="Adjusting initial estimates to create positive semidefinite omega/sigma matrices",
                    category=UserWarning,
                )
                try:
                    new_candidate_model = set_initial_estimates(new_candidate_model, new_parameters)
                    break
                except UserWarning:
                    if try_number == maximum_tests:
                        raise ValueError(
                            f"{new_candidate_model.name} could not be determined"
                            f" to be positive semi-definite."
                        )
    elif scale == "UCP":
        new_parameters = create_new_parameter_inits(new_candidate_model, fraction, scale, seed)
        new_candidate_model = set_initial_estimates(new_candidate_model, new_parameters)
    else:
        # Should be caught in validate_input()
        raise ValueError(f'Scale ({scale}) is not supported')

    new_candidate_model_entry = ModelEntry.create(
        model=new_candidate_model,
        modelfit_results=None,
        parent=original_model,
    )
    new_candidate_model_fit_wf = create_fit_workflow(modelentries=[new_candidate_model_entry])
    new_modelentry = call_workflow(new_candidate_model_fit_wf, f'fit_candidate_run{index}', context)
    return Retry(
        modelentry=new_modelentry,
        number_of_retries=try_number,
    )


def create_new_parameter_inits(model, fraction, scale, seed):
    if scale == "normal":
        new_parameters = sample_parameters_uniformly(
            model,
            pd.Series(model.parameters.inits),
            fraction=fraction,
            scale=scale,
            seed=seed,
        )
        new_parameters = {p: new_parameters[p][0] for p in new_parameters}
    elif scale == "UCP":
        ucp_scale = calculate_ucp_scale(model)
        new_parameters = sample_parameters_uniformly(
            model,
            pd.Series(model.parameters.inits),
            fraction=fraction,
            scale=scale,
            seed=seed,
        )
        new_parameters = {p: new_parameters[p][0] for p in new_parameters}
        new_parameters = calculate_parameters_from_ucp(model, ucp_scale, new_parameters)
    else:
        raise ValueError(f'{scale} is not supported')
    return new_parameters


def task_results(context, strictness, retries):
    # Note : the input (modelentry) is a part of retries
    retry_runs = []
    for r in retries:
        if isinstance(r, ModelEntry):
            input_model_entry = r
        elif isinstance(r, Retry):
            retry_runs.append(r)
        else:
            raise ValueError(f'Unknown type ({type(r)}) found when summarizing results.')
    res_models = [r.modelentry for r in retry_runs]
    results_to_summarize = [input_model_entry] + [r.modelentry for r in retry_runs]
    rank_type = "ofv"
    cutoff = None

    summary_models = summarize_modelfit_results_from_entries(results_to_summarize)
    summary_models['step'] = [0] + [1] * (len(summary_models) - 1)
    summary_models = summary_models.reset_index().set_index(['step', 'model'])

    res = create_results(
        RetriesResults,
        input_model_entry,
        input_model_entry,
        res_models,
        rank_type,
        cutoff,
        strictness=strictness,
        context=context,
        summary_models=summary_models,
    )

    res = replace(
        res,
        summary_tool=_modify_summary_tool(res.summary_tool, retry_runs),
    )

    return res


def convert_to_posdef(model):
    new_parameter_estimates = {}

    omega_symbolic = model.random_variables.etas.covariance_matrix
    omega = omega_symbolic.subs(model.parameters.inits)
    omega = omega.to_numpy()
    if not is_posdef(omega):
        omega = nearest_positive_definite(omega)
        for row in range(len(omega)):
            for col in range(len(omega)):
                new_parameter_estimates[str(omega_symbolic[row, col])] = omega[row, col]

    sigma_symbolic = model.random_variables.epsilons.covariance_matrix
    sigma = sigma_symbolic.subs(model.parameters.inits)
    sigma = sigma.to_numpy()
    if not is_posdef(sigma):
        sigma = nearest_positive_definite(sigma)
        for row in range(len(sigma)):
            for col in range(len(sigma)):
                new_parameter_estimates[str(sigma_symbolic[row, col])] = sigma[row, col]
    if new_parameter_estimates:
        warnings.warn(
            f'{model.name} required adjusted value to becom'
            f'postive definite. Adjusted values for parameters : '
            f'{new_parameter_estimates.keys()}'
        )
        return set_initial_estimates(model, new_parameter_estimates)
    else:
        return model


def _modify_summary_tool(summary_tool, retry_runs):
    summary_tool = summary_tool.reset_index()
    number_of_retries_dict = {r.modelentry.model.name: r.number_of_retries for r in retry_runs}
    summary_tool['Number_of_retries'] = summary_tool['model'].map(number_of_retries_dict)

    column_to_move = summary_tool.pop('Number_of_retries')
    summary_tool.insert(1, 'Number_of_retries', column_to_move)

    return summary_tool.set_index(['model'])


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(model, results, number_of_candidates, fraction, strictness, scale, prefix_name):
    if not isinstance(model, Model):
        raise ValueError(
            f'Invalid `model` type: got `{type(model)}`, must be one of pharmpy Model object.'
        )

    if not isinstance(results, ModelfitResults):
        raise ValueError(
            f'Invalid `results` type: got `{type(results)}`, must be one of pharmpy ModelfitResults object.'
        )

    if not isinstance(number_of_candidates, int):
        raise ValueError(
            f'Invalid `number_of_candidates` type: got `{type(number_of_candidates)}`, must be an integer.'
        )
    elif number_of_candidates <= 0:
        raise ValueError(
            f'`number_of_candidates` need to be a positiv integer, not `{number_of_candidates}`'
        )

    if not isinstance(fraction, float):
        raise ValueError(
            f'Invalid `fraction` type: got `{type(fraction)}`, must be an number (float).'
        )

    # STRICTNESS?

    if scale not in SCALES:
        raise ValueError(f'Invalid `scale`: got `{scale}`, must be one of {sorted(SCALES)}.')


@dataclass(frozen=True)
class RetriesResults(ToolResults):
    rst_path = Path(__file__).resolve().parent / 'report.rst'
    pass
