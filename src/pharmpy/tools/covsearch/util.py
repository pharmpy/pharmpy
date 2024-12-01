from dataclasses import dataclass

from pharmpy.model import Model
from pharmpy.modeling import (
    add_estimation_step,
    mu_reference_model,
    remove_covariate_effect,
    remove_estimation_step,
    set_estimation_step,
)
from pharmpy.tools.mfl.feature.covariate import parse_spec, spec
from pharmpy.tools.mfl.helpers import all_funcs
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import ModelEntry


@dataclass(frozen=True)
class Effect:
    parameter: str
    covariate: str
    fp: str
    operation: str


class DummyEffect(Effect):
    pass


@dataclass(frozen=True)
class Step:
    alpha: float
    effect: Effect


class ForwardStep(Step):
    pass


@dataclass
class Candidate:
    modelentry: ModelEntry
    steps: tuple[Step, ...]


@dataclass
class SearchState:
    user_input_modelentry: ModelEntry
    start_modelentry: ModelEntry
    best_candidate_so_far: Candidate
    all_candidates_so_far: list[Candidate]


@dataclass
class StateAndEffect:
    search_state: SearchState
    effect_funcs: dict


@dataclass
class LinStateAndEffect(StateAndEffect):
    linear_models: dict
    param_cov_list: dict


def store_input_model(context, model, results, max_eval):
    """Store the input model"""
    context.log_info("Starting tool covsearch")
    model = model.replace(name="input", description="input")
    if max_eval:
        input_modelentry = set_maxevals(model, results)
    else:
        input_modelentry = ModelEntry.create(model=model, modelfit_results=results)
    context.store_input_model_entry(input_modelentry)
    return input_modelentry


def set_maxevals(model, results, max_evals=3.1):
    max_eval_number = round(max_evals * results.function_evaluations_iterations.loc[1])
    first_es = model.execution_steps[0]
    model = set_estimation_step(model, first_es.method, 0, maximum_evaluations=max_eval_number)
    return ModelEntry.create(model=model, parent=None, modelfit_results=results)


def init_search_state(context, search_space, algorithm, nsamples, modelentry):
    model = modelentry.model
    effect_funcs, filtered_model = filter_search_space_and_model(search_space, model)
    search_state = init_nonlinear_search_state(
        context, modelentry, filtered_model, algorithm, nsamples
    )
    return StateAndEffect(search_state=search_state, effect_funcs=effect_funcs)


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
    filtered_model = filtered_model.replace(description="input;" + ";".join(description))

    exploratory_cov = tuple(c for c in ss_mfl.covariate if c.optional.option)
    exploratory_cov_funcs = all_funcs(Model(), exploratory_cov)
    exploratory_cov_funcs = {
        cov_effect[1:-1]: cov_func
        for cov_effect, cov_func in exploratory_cov_funcs.items()
        if cov_effect[-1] == "ADD"
    }
    return (exploratory_cov_funcs, filtered_model)


def init_nonlinear_search_state(context, input_modelentry, filtered_model, algorithm, nsamples):

    if "samba" in algorithm:
        filtered_model = set_samba_estimation(filtered_model, algorithm, nsamples)

    elif filtered_model.execution_steps[0].method != "FOCE":
        filtered_model = remove_estimation_step(filtered_model, idx=0)
        filtered_model = add_estimation_step(
            filtered_model,
            method="FOCE",
            idx=0,
            interaction=True,
            auto=True,
            tool_options={'PHITYPE': "1", 'FNLETA': "0"},
        )

    # nonlinear mixed effect modelentry creation and fit
    if filtered_model != input_modelentry.model:
        filtered_modelentry = ModelEntry.create(model=filtered_model)
        filtered_fit_wf = create_fit_workflow(modelentries=[filtered_modelentry])
        filtered_modelentry = context.call_workflow(filtered_fit_wf, 'fit_filtered_model')
    else:
        filtered_modelentry = input_modelentry

    candidate = Candidate(filtered_modelentry, ())
    return SearchState(input_modelentry, filtered_modelentry, candidate, [candidate])


def set_samba_estimation(filtered_model, algorithm, nsamples):
    # estimation method 1 for population parameters
    if "foce" in algorithm and filtered_model.execution_steps[0] != "FOCE":
        filtered_model = remove_estimation_step(filtered_model, idx=0)
        filtered_model = add_estimation_step(
            filtered_model,
            method="FOCE",
            idx=0,
            interaction=True,
            auto=True,
            tool_options={'PHITYPE': "1", 'FNLETA': "0"},
        )
    else:
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
            isample=10,
            keep_every_nth_iter=50,
            tool_options={'PHITYPE': "1", 'FNLETA': "0"},
        )
    # estimation method 2 for individual parameters
    if nsamples > 0:
        filtered_model = add_estimation_step(
            filtered_model,
            method="SAEM",
            idx=2,
            niter=0 if 'saem' in algorithm else 10,
            isample=nsamples,
            tool_options={
                "EONLY": "1",
                "NBURN": "0",
                "MASSRESET": "0",
                "ETASAMPLES": "1",
            },
        )

    # estimation method 3 for stable OFV
    filtered_model = add_estimation_step(
        filtered_model,
        method="IMP",
        idx=3,
        auto=True,
        niter=20,
        isample=1000,
        tool_options={
            "EONLY": "1",
            "MASSRESET": "1",
            "ETASAMPLES": "0",
        },
    )

    return filtered_model
