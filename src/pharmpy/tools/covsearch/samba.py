from typing import Literal, Optional, Union

import pharmpy.tools.covsearch.tool as scm_tool
from pharmpy.model import Model
from pharmpy.tools.covsearch.fast_forward import fast_forward
from pharmpy.tools.covsearch.util import store_input_model, set_maxevals, SearchState, Candidate
from pharmpy.tools.mfl.feature.covariate import parse_spec, spec
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder, call_workflow
from pharmpy.workflows.results import ModelfitResults

NAME_WF = 'covsearch'


def samba_workflow(
    search_space: Union[str, ModelFeatures],
    max_steps: int = -1,
    alpha: float = 0.05,
    results: Optional[ModelfitResults] = None,
    model: Optional[Model] = None,
    max_eval: bool = False,
    statsmodels: bool = True,
    algorithm: Literal['samba'] = 'samba',
    nsamples: int = 10,
    weighted_linreg: bool = False,
    lin_filter: int = 2,
):
    """
    Workflow builder for SAMBA covariate search algorithm.
    """

    wb = WorkflowBuilder(name=NAME_WF)

    # Initiate model and search state
    store_task = Task("store_input_model", store_input_model, model, results, max_eval)
    wb.add_task(store_task)

    init_task = Task("init", init_search_state, search_space, algorithm, nsamples)
    wb.add_task(init_task, predecessors=store_task)

    # SAMBA search task
    samba_search_task = Task(
        "samba_search",
        fast_forward,
        alpha,
        max_steps,
        algorithm,
        nsamples,
        weighted_linreg,
        statsmodels,
        lin_filter,
    )
    wb.add_task(samba_search_task, predecessors=init_task)
    search_output = wb.output_tasks

    # Results task
    results_task = Task(
        "results",
        samba_task_results,
        alpha,
    )
    wb.add_task(results_task, predecessors=search_output)

    return Workflow(wb)


def store_input_model(context, model, results, max_eval):
    """Store the input model"""
    model = model.replace(name="input", description="")
    if max_eval:
        input_modelentry = set_maxevals(model, results)
    else:
        input_modelentry = ModelEntry.create(model=model, modelfit_results=results)
    context.store_input_model_entry(input_modelentry)
    return input_modelentry


def init_search_state(context, search_space, algorithm, nsamples, modelentry):
    model = modelentry.model
    effect_funcs, filtered_model = filter_search_space_and_model(search_space, model)
    search_state = init_nonlinear_search_state(
        context, modelentry, filtered_model, algorithm, nsamples
    )
    return StateAndEffect(search_state=search_state, effect_funcs=effect_funcs)


def init_nonlinear_search_state(context, input_modelentry, filtered_model, algorithm, nsamples):
    if algorithm == "samba":
        filtered_model = mu_reference_model(filtered_model)
        filtered_model = remove_estimation_step(filtered_model, idx=0)
        filtered_model = add_estimation_step(
            filtered_model,
            method="ITS",
            idx=0,
            interaction=True,
            auto=True,
            niter=5,
        )
        filtered_model = add_estimation_step(
            filtered_model,
            method="SAEM",
            idx=1,
            interaction=True,
            auto=True,
            niter=200,
            keep_every_nth_iter=50,
            tool_options={'PHITYPE': "1", 'FNLETA': "0"},
        )
        if nsamples > 0:
            filtered_model = add_estimation_step(
                filtered_model,
                method="SAEM",
                idx=2,
                niter=0,
                tool_options={
                    "EONLY": "1",
                    "NBURN": "0",
                    "MASSRESET": "0",
                    "ETASAMPLES": "1",
                    "ISAMPLE": f"{nsamples}",
                },
            )
        if nsamples == 1:
            filtered_model = add_estimation_step(
                filtered_model,
                method="IMP",
                idx=3,
                auto=True,
                interaction=True,
                niter=20,
                tool_options={
                    "EONLY": "1",
                    "MASSRESET": "1",
                    "ISAMPLE": "1000",
                },
            )
    if algorithm != "samba" and filtered_model.execution_steps[0].method != "FOCE":
        filtered_model = remove_estimation_step(filtered_model, idx=0)
        filtered_model = add_estimation_step(
            filtered_model,
            method="FOCE",
            idx=0,
            interaction=True,
            auto=True,
        )

    # nonlinear mixed effect modelentry creation and fit
    if filtered_model != input_modelentry.model:
        filtered_modelentry = ModelEntry.create(model=filtered_model)
        filtered_fit_wf = create_fit_workflow(modelentries=[filtered_modelentry])
        filtered_modelentry = call_workflow(filtered_fit_wf, 'fit_filtered_model', context)
    else:
        filtered_modelentry = input_modelentry

    candidate = Candidate(filtered_modelentry, ())
    return SearchState(input_modelentry, filtered_modelentry, candidate, [candidate])


def filter_search_space_and_model(search_space, model):
    filtered_model = model.replace(name="filtered_input_model")
    if isinstance(search_space, str):
        search_space = ModelFeatures.create_from_mfl_string(search_space)
    ss_mfl = search_space.expand(filtered_model)  # expand to remove LET / REF
    model_mfl = ModelFeatures.create_from_mfl_string(get_model_features(filtered_model))

    covariate_to_keep = model_mfl - ss_mfl
    covariate_to_remove = model_mfl - covariate_to_keep
    covariate_to_remove = covariate_to_remove.mfl_statement_list(["covariate"])
    description = ["REMOVE"]
    if len(covariate_to_remove) != 0:
        for cov_effect in parse_spec(spec(filtered_model, covariate_to_remove)):
            filtered_model = remove_covariate_effect(filtered_model, cov_effect[0], cov_effect[1])
            description.append(f'({cov_effect[0]}-{cov_effect[1]}-{cov_effect[2]})')

    covariate_to_keep = covariate_to_keep.mfl_statement_list(["covariate"])
    for cov_effect in parse_spec(spec(filtered_model, covariate_to_keep)):
        if cov_effect[2].lower() == "custom":
            filtered_model = remove_covariate_effect(filtered_model, cov_effect[0], cov_effect[1])
            description.append(f'({cov_effect[0]}-{cov_effect[1]}-{cov_effect[2]})')

    structural_cov = tuple(c for c in ss_mfl.covariate if not c.optional.option)
    structural_cov_funcs = all_funcs(Model(), structural_cov)
    if len(structural_cov_funcs) != 0:
        description.append("ADD_STRUCT")
        for cov_effect, cov_func in structural_cov_funcs.items():
            filtered_model = cov_func(filtered_model)
            description.append(f'({cov_effect[0]}-{cov_effect[1]}-{cov_effect[2]})')
    description.append("ADD_EXPLOR")
    filtered_model = filtered_model.replace(description=";".join(description))

    exploratory_cov = tuple(c for c in ss_mfl.covariate if c.optional.option)
    exploratory_cov_funcs = all_funcs(Model(), exploratory_cov)
    exploratory_cov_funcs = {
        cov_effect[1:-1]: cov_func
        for cov_effect, cov_func in exploratory_cov_funcs.items()
        if cov_effect[-1] == "ADD"
    }
    return (exploratory_cov_funcs, filtered_model)



def samba_task_results(context, p_forward, state):
    # set p_backward and strictness to None
    return scm_tool.task_results(context, p_forward, p_backward=None, strictness=None, state=state)
