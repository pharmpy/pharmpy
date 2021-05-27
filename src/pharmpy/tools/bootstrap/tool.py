import pharmpy.tools
import pharmpy.tools.modelfit as modelfit
from pharmpy.data.iterators import Resample
from pharmpy.tools.workflows import Task, Workflow


class Bootstrap(pharmpy.tools.Tool):
    def __init__(self, model, resamples=1):
        self.model = model
        self.resamples = resamples
        super().__init__()

    def run(self):
        workflow = Workflow()
        resample_tasks, task_names = [], []

        for i in range(self.resamples):
            task = Task(f'resample-{i}', resample_model, [self.model])
            resample_tasks.append(task)
            task_names.append(task.task_id)

        modelfit_run = modelfit.Modelfit(task_names, path=self.rundir.path)

        workflow.add_tasks(resample_tasks)
        workflow.merge_workflows(modelfit_run.workflow_creator(task_names))
        workflow.add_tasks(Task('results', final_models, [workflow.tasks[-1]], final_task=True))

        fit_models = self.dispatcher.run(workflow, self.database)
        return fit_models
        # run(self.models, self.rundir.path)
        # res = self.models[0].modelfit_results
        # res.to_json(path=self.rundir.path / 'results.json')
        # res.to_csv(path=self.rundir.path / 'results.csv')


def resample_model(model):
    resample = Resample(model, model.dataset.pharmpy.id_label, resamples=1, name_pattern='bs_{}')
    model, _ = next(resample)
    return model


def final_models(models):
    return models
