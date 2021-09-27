import pharmpy.tools.modelsearch as modelsearch
import pharmpy.tools.resmod as resmod
from pharmpy.workflows import Task, Workflow


def create_workflow(model):
    wf = Workflow()
    wf.name = 'amd'

    preprocess_task = Task('preprocess', preprocess, model)

    mfl = 'ABSORPTION(ZO)\nPERIPHERALS(1)'
    wf_modelsearch = modelsearch.create_workflow('exhaustive_stepwise', mfl)
    wf.insert_workflow(wf_modelsearch, predecessors=preprocess_task)

    select_modelsearch_task = Task('select_modelsearch', select_modelsearch, model)
    wf.add_task(select_modelsearch_task, predecessors=wf_modelsearch.output_tasks)

    wf_resmod = resmod.create_workflow()
    wf.insert_workflow(wf_resmod, predecessors=wf.output_tasks)

    select_task = Task('select_resmod', select_resmod)
    wf.add_task(select_task, predecessors=wf_resmod.output_tasks)

    post_process_task = Task('results', post_process)
    wf.add_task(post_process_task, predecessors=wf.output_tasks)

    return wf


def preprocess(model):
    return model


def select_modelsearch(model, res):
    if res.best_model:
        return res.best_model
    else:
        return model


def select_resmod(res):
    return res.best_model


def post_process(*models):
    res = [model for model in models]
    return res
