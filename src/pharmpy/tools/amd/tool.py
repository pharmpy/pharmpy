import pharmpy.modeling as modeling
import pharmpy.tools.resmod as resmod
from pharmpy.tools.workflows import Task, Workflow


def create_workflow(model):
    wf = Workflow()
    wf.name = 'amd'

    preprocess_task = Task('preprocess', preprocess, model)

    wf_resmod = resmod.create_workflow()
    wf.insert_workflow(wf_resmod, predecessors=preprocess_task)

    select_task = Task('select_resmod', select_resmod)
    wf.add_task(select_task, predecessors=[preprocess_task] + wf_resmod.output_tasks)

    return wf


def preprocess(model):
    return model


def select_resmod(model, res):
    idx = res.models['dofv'].idxmax()
    name = idx[0]
    if name == 'power':
        modeling.set_power_on_ruv(model)
    else:
        modeling.set_iiv_on_ruv(model)
    model.update_source()
    return model
