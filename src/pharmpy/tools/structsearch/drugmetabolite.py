from typing import Union

from pharmpy.model import Model
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.modelsearch.algorithms import exhaustive_stepwise
from pharmpy.workflows import ModelEntry, Task, WorkflowBuilder

from ..mfl.parse import ModelFeatures
from ..mfl.parse import parse as mfl_parse


def create_drug_metabolite_models(
    model: Model, results, search_space: Union[str, ModelFeatures]
) -> tuple[list[Model], Model]:
    # FIXME : Implement ModelFeatures when we can extract METABOLITE information

    if isinstance(search_space, str):
        mfl_statements = mfl_parse(search_space, True)
    metabolite_functions = mfl_statements.convert_to_funcs(
        attribute_type=["metabolite"], subset_features="metabolite"
    )
    peripheral_functions = mfl_statements.convert_to_funcs(
        attribute_type=["peripherals"], subset_features="metabolite"
    )
    # Filter away DRUG compartment peripherals (if any)
    peripheral_functions = {k: v for k, v in peripheral_functions.items() if len(k) == 3}

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
        base_description = determine_base_description(metabolite_functions, peripheral_functions)

    wb = WorkflowBuilder(name="drug_metabolite")

    start_task = Task("start", _start, model, results)
    wb.add_task(start_task)

    def apply_transformation(eff, f, model_entry):
        candidate_model = f(model_entry.model)
        candidate_model = candidate_model.replace(name="TEMP", description='_'.join(eff))
        return ModelEntry.create(model=candidate_model, modelfit_results=None, parent=model)

    for eff, func in metabolite_functions.items():
        candidate_met_task = Task(str(eff), apply_transformation, eff, func)
        wb.add_task(candidate_met_task, predecessors=start_task)

    if len(peripheral_functions) == 0:
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


def determine_base_description(met_mfl_func, per_mfl_func):
    description = []
    if "BASIC" in [k[1] for k in met_mfl_func.keys()]:
        description.append("METABOLITE_BASIC")
    else:
        description.append("METABOLITE_PSC")
    if per_mfl_func:
        description.append(f"PERIPHERALS({min([k[1] for k in per_mfl_func.keys()])}, METABOLITE)")
    return ";".join(description)
