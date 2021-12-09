import pharmpy.tools
from pharmpy.model import Model
from pharmpy.plugins.nonmem.run import execute_model
from pharmpy.workflows import Task, Workflow


def create_workflow(models=None, n=None):
    """
    If models is None and n is None: fit one unknown model
    If models is model object (n must be None): fit that one model
    If models is list of models (n must be None): fit these models
    If models is None and n is an int: fit n unknown models
    """
    wf = Workflow()
    wf.name = "modelfit"
    if models is None:
        if n is None:
            task = Task('run', execute_model)
            wf.add_task(task)
        else:
            for i in range(n):
                task = Task(f'run{i}', execute_model)
                wf.add_task(task)
    elif isinstance(models, Model):
        task = Task('run', execute_model, models)
        wf.add_task(task)
    else:
        for i, model in enumerate(models):
            task = Task(f'run{i}', execute_model, model)
            wf.add_task(task)
    task_result = Task('results', post_process_results)
    wf.add_task(task_result, predecessors=wf.output_tasks)
    return wf


# FIXME: These two aren't really needed
def create_single_fit_workflow(model=None):
    wf = Workflow()
    if model is None:
        task = Task('run', execute_model)
    else:
        task = Task('run', execute_model, model)
    wf.add_task(task)
    return wf


def create_multiple_fit_workflow(models=None, n=None):
    """Either specify models or n"""
    wf = Workflow()
    if models is None:
        for i in range(n):
            task = Task(f'run{i}', execute_model)
            wf.add_task(task)
    else:
        for i, model in enumerate(models):
            task = Task(f'run{i}', execute_model, model)
            wf.add_task(task)
    return wf


class Modelfit(pharmpy.tools.Tool):
    def __init__(self, models, **kwargs):
        self.models = models
        super().__init__(**kwargs)

    def run(self):
        wf_fit = create_multiple_fit_workflow(self.models)
        task_result = Task('results', post_process_results)
        wf_fit.add_task(task_result, predecessors=wf_fit.output_tasks)
        for model in self.models:
            model.dataset
            model.database = self.database.model_database
        fit_models = self.dispatcher.run(wf_fit, self.database)
        for i in range(len(fit_models)):
            self.models[i].modelfit_results = fit_models[i].modelfit_results


def post_process_results(*models):
    return models
