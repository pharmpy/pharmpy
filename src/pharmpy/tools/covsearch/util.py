from dataclasses import dataclass
from typing import List, Tuple

from pharmpy.modeling import set_estimation_step
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
    steps: Tuple[Step, ...]


@dataclass
class SearchState:
    user_input_modelentry: ModelEntry
    start_modelentry: ModelEntry
    best_candidate_so_far: Candidate
    all_candidates_so_far: List[Candidate]


def set_maxevals(model, results, max_evals=3.1):
    max_eval_number = round(max_evals * results.function_evaluations_iterations.loc[1])
    first_es = model.execution_steps[0]
    model = set_estimation_step(model, first_es.method, 0, maximum_evaluations=max_eval_number)
    return ModelEntry.create(
        model=model.replace(name="input", description=""), parent=None, modelfit_results=results
    )
