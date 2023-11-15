from typing import List

from pharmpy.model import Model
from pharmpy.tools.mfl.helpers import funcs, structsearch_metabolite_features
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.modelsearch.algorithms import exhaustive_stepwise
from pharmpy.workflows import ModelEntry, Task, WorkflowBuilder

from ..mfl.parse import parse as mfl_parse
from ..mfl.statement.feature.metabolite import Metabolite
from ..mfl.statement.feature.peripherals import Peripherals


def create_drug_metabolite_models(
    model: Model, results, search_space: str
) -> tuple[List[Model], Model]:
    # FIXME : Implement ModelFeatures when we can extract METABOLITE information

    mfl_statements = mfl_parse(search_space)
    metabolite_mfl_statements = [s for s in mfl_statements if isinstance(s, Metabolite)]
    metabolite_functions = funcs(model, metabolite_mfl_statements, structsearch_metabolite_features)
    peripheral_mfl_statements = [s for s in mfl_statements if isinstance(s, Peripherals)]
    peripheral_functions = funcs(model, peripheral_mfl_statements, structsearch_metabolite_features)

    # TODO: Update method for finding metabolite name
    if (
        model.statements.ode_system.find_compartment('METABOLITE')
        and len(metabolite_functions) != 0
    ):
        raise NotImplementedError(
            'Metabolite transformation on drug metabolite models is not yet possible.'
            ' Either remove METABOLITE transformations from search space or use another input model'
        )
    elif (
        not model.statements.ode_system.find_compartment('METABOLITE')
        and len(metabolite_functions) == 0
    ):
        raise ValueError(
            'Require at least one metabolite model type.'
            ' Try adding METABOLITE(BASIC) or METABOLITE(PSC) to search space'
        )

    if model.statements.ode_system.find_compartment('METABOLITE'):
        base_description = model.description
    else:
        base_description = determine_base_description(
            metabolite_mfl_statements, peripheral_mfl_statements
        )

    wb = WorkflowBuilder(name="drug_metabolite")

    start_task = Task("start", _start, model, results)
    wb.add_task(start_task)

    def apply_transformation(eff, f, model_entry):
        candidate_model = f(model_entry.model)
        candidate_model = candidate_model.replace(
            name="TEMP", description='_'.join(eff), modelfit_results=None
        )
        return ModelEntry.create(model=candidate_model, modelfit_results=None, parent=model)

    for eff, func in metabolite_functions.items():
        candidate_met_task = Task(str(eff), apply_transformation, eff, func)
        wb.add_task(candidate_met_task, predecessors=start_task)

    if len(peripheral_mfl_statements) == 0:
        candidate_model_tasks = []
        model_index = 1
        for out_task in wb.output_tasks:
            change_name_task = Task("number_run", change_name, model_index)
            model_index += 1
            wb.add_task(change_name_task, predecessors=[out_task])
            wf_fit = create_fit_workflow(n=1)
            wb.insert_workflow(wf_fit, predecessors=[change_name_task])

            candidate_model_tasks += wf_fit.output_tasks
    else:
        wb, candidate_model_tasks = exhaustive_stepwise(
            peripheral_functions, "no_add", wb, "structsearch"
        )

    return WorkflowBuilder(wb), candidate_model_tasks, base_description


def _start(model, results):
    return ModelEntry.create(model=model, modelfit_results=results)


def change_name(index, modelentry):
    return ModelEntry.create(
        model=modelentry.model.replace(name=f'structsearch_run{index}'),
        modelfit_results=modelentry.modelfit_results,
        parent=modelentry.parent,
    )


def determine_base_description(met_mfl, per_mfl):
    description = []
    if "BASIC" in [t.name for m in met_mfl for t in m.modes]:
        description.append("METABOLITE_BASIC")
    else:
        description.append("METABOLITE_PSC")
    if per_mfl:
        description.append(f"PERIPHERALS({min([c for p in per_mfl for c in p.counts])})")
    return ";".join(description)
