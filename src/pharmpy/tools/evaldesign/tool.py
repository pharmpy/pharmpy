from dataclasses import dataclass
from typing import Any, Optional

from pharmpy.model import Results
from pharmpy.workflows import Task, Workflow


def create_workflow(model):
    wf = Workflow()
    wf.name = 'evaldesign'

    from pharmpy.plugins.nonmem.run import evaluate_design

    task = Task('run', evaluate_design, model)
    wf.add_task(task)

    task_result = Task('results', post_process_results)
    wf.add_task(task_result, predecessors=[task])
    return wf


def post_process_results(res):
    return res


@dataclass(frozen=True)
class EvalDesignResults(Results):
    ofv: Optional[float] = None
    individual_ofv: Optional[Any] = None
    information_matrix: Optional[Any] = None
