from typing import Literal, Optional

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fn.type import with_runtime_arguments_type_check
from pharmpy.model import Model
from pharmpy.modeling import (
    create_joint_distribution,
    plot_transformed_eta_distributions,
    set_initial_estimates,
    transform_etas_boxcox,
    transform_etas_tdist,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import run_subtool
from pharmpy.workflows import ModelEntry, ModelfitResults, Task, Workflow, WorkflowBuilder

from .results import QAResults, calc_fullblock, calc_transformed_etas

SECTIONS = frozenset(('tdist', 'boxcox', 'fullblock'))


def create_workflow(
    model: Optional[Model] = None,
    results: Optional[ModelfitResults] = None,
    linearize: bool = False,
    skip: Optional[list[Literal[tuple(SECTIONS)]]] = None,
):
    """
    Run QA tool.

    Parameters
    ----------
    model : Model
        Pharmpy model
    results : ModelfitResults
        Results of model
    linearize : bool
        Whether or not to use linearization when running the tool
    skip : list of {'tdist', 'boxcox', 'fullblock'}
        A list of sections to skip


    Returns
    -------
    QAResults
        QA tool result object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> from pharmpy.tools import run_qa, load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> run_qa(model=model, results=results, linearize=False, skip=['fullblock'])      # doctest: +SKIP

    """
    wb = WorkflowBuilder(name='qa')

    start_task = Task('start', start, model, results)
    wb.add_task(start_task)

    if skip is None:
        skip = []

    if linearize:
        base_task = Task('linearization', run_linearization)
        wb.add_task(base_task, predecessors=start_task)
    else:
        base_task = start_task

    for section in SECTIONS:
        if section in skip:
            continue
        task = Task(section, run_candidate, section)
        wb.add_task(task, predecessors=base_task)

    result_task = Task('results', create_results)
    wb.add_task(result_task, predecessors=wb.output_tasks + [base_task])

    return Workflow(wb)


def start(context, model, results):
    context.log_info("Starting tool QA")
    context.log_info(f"Input model OFV: {results.ofv:.3f}")

    input_model = model.replace(name="input", description="")
    input_model_entry = ModelEntry.create(model=input_model, modelfit_results=results)

    context.store_input_model_entry(input_model_entry)

    return input_model_entry


def run_linearization(context, me):
    lin_res = run_subtool(
        'linearize',
        context,
        model=me.model,
        results=me.modelfit_results,
        description='linearized',
    )
    lin_me = ModelEntry.create(
        model=lin_res.final_model, modelfit_results=lin_res.final_model_results
    )
    return lin_me


def run_candidate(context, type, me_base):
    me_candidate = create_candidate(me_base, type)
    wf_fit = create_fit_workflow(me_candidate)
    me_fitted = context.call_workflow(wf_fit, f"fit-{me_candidate.model.name}")
    return me_fitted


def create_candidate(me_base, type):
    assert type in SECTIONS
    if type == 'tdist':
        model = transform_etas_tdist(me_base.model)
    elif type == 'boxcox':
        model = transform_etas_boxcox(me_base.model)
    else:
        model = create_joint_distribution(me_base.model)
    model = set_initial_estimates(model, me_base.modelfit_results.parameter_estimates)
    model = model.replace(name=type, description=type)
    return ModelEntry.create(model, parent=me_base.model)


def create_results(context, *mes):
    res_dict = dict()
    me_base, me_cands = categorize_model_entries(mes)

    dofv_table = create_dofv_table(me_base, me_cands)
    res_dict['dofv'] = dofv_table

    for i, me in enumerate(me_cands):
        if np.isnan(me.modelfit_results.ofv):
            context.log_warning(f'Model {me.model.name} has no OFV, skipping results')
            continue
        if 'boxcox' in me.model.name or 'tdist' in me.model.name:
            rng = context.create_rng(i)
            res_trans = create_eta_transformation_results(me, me_base, rng)
            res_dict.update(res_trans)
        elif 'fullblock' in me.model.name:
            table, _ = calc_fullblock(me_base, me)
            res_dict['fullblock_parameters'] = table
        else:
            raise NotImplementedError

    res = QAResults(**res_dict)

    return res


def categorize_model_entries(model_entries):
    me_base = None
    me_cands = []
    for me in model_entries:
        if any(section in me.model.name for section in SECTIONS):
            me_cands.append(me)
        else:
            if me_base is not None:
                raise ValueError(
                    f'More than one models are not candidates: {[me.model.name, me_base.model.name]}'
                )
            me_base = me
    assert me_base is not None
    return me_base, me_cands


def create_dofv_table(me_base, me_cands):
    dofv_mapping = dict()
    for me in me_cands:
        dofv = me_base.modelfit_results.ofv - me.modelfit_results.ofv
        d_params = len(me.model.parameters) - len(me_base.model.parameters)
        dofv_mapping[me.model.name] = {'dofv': dofv, 'added_params': d_params}

    table = pd.DataFrame.from_dict(dofv_mapping, orient='index')
    return table


def create_eta_transformation_results(me, me_base, rng):
    res_dict = dict()
    pes = me.modelfit_results.parameter_estimates
    ies = me.modelfit_results.individual_estimates
    pes_base = me_base.modelfit_results.parameter_estimates
    trans_type = 'boxcox' if 'boxcox' in me.model.name else 'tdist'
    param_type = 'lambda' if trans_type == 'boxcox' else 'df'
    table, _ = calc_transformed_etas(me_base, me, trans_type, param_type)
    plot = plot_transformed_eta_distributions(me.model, pes, ies, pes_base, rng)
    res_dict[f'{trans_type}_parameters'] = table
    res_dict[f'{trans_type}_plot'] = plot
    return res_dict


@with_runtime_arguments_type_check
@with_same_arguments_as(create_workflow)
def validate_input(model, results, linearize, skip):
    if set(skip) == set(SECTIONS):
        raise ValueError('Invalid `skip`: all analysis would be skipped')
