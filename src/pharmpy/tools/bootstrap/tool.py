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
        resample_tasks, resample_names = [], []

        for i in range(self.resamples):
            task = Task(f'resample-{i}', resample_model, [self.model])
            resample_tasks.append(task)
            resample_names.append(task.task_id)

        workflow.add_tasks(resample_tasks)

        modelfit_run = modelfit.Modelfit(resample_names, path=self.rundir.path)
        modelfit_workflow = modelfit_run.workflow_creator(resample_names)

        workflow.merge_workflows(modelfit_workflow)
        postprocessing_task = Task('results', final_models, [workflow.tasks[-1]], final_task=True)
        workflow.add_tasks(postprocessing_task)

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
