from pharmpy.results import Results
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


class EvalDesignResults(Results):
    def __init__(self, ofv=None, individual_ofv=None, information_matrix=None):
        self.ofv = ofv
        self.individual_ofv = individual_ofv
        self.information_matrix = information_matrix
