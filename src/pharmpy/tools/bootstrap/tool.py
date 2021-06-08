import pharmpy.tools
import pharmpy.tools.modelfit as modelfit
from pharmpy.data.iterators import Resample
from pharmpy.execute import NullToolDatabase
from pharmpy.tools.bootstrap.results import calculate_results
from pharmpy.tools.workflows import Task, Workflow


class Bootstrap(pharmpy.tools.Tool):
    def __init__(self, model, resamples=1):
        self.model = model
        self.resamples = resamples
        super().__init__()
        self.model.database = self.database.model_database

    def run(self):
        workflow = Workflow()
        resample_tasks, resample_names = [], []

        for i in range(self.resamples):
            task = Task(f'resample-{i}', resample_model, [i, self.model])
            resample_tasks.append(task)
            resample_names.append(task.task_id)

        workflow.add_tasks(resample_tasks)

        db = NullToolDatabase
        db.model_database = self.database.model_database
        modelfit_run = modelfit.Modelfit(resample_names, database=db)
        modelfit_workflow = modelfit_run.workflow_creator(resample_names)

        workflow.merge_workflows(modelfit_workflow)
        postprocessing_task = Task('results', final_models, [workflow.tasks[-1]], final_task=True)
        workflow.add_tasks(postprocessing_task)

        res = self.dispatcher.run(workflow, self.database)
        res.to_json(path=self.database.path / 'results.json')
        res.to_csv(path=self.database.path / 'results.csv')
        return res


def resample_model(i, model):
    resample = Resample(
        model, model.dataset.pharmpy.id_label, resamples=1, replace=True, name_pattern=f'bs_{i}'
    )
    model, _ = next(resample)
    return model


def final_models(models):
    res = calculate_results(
        models, original_model=None, included_individuals=None, dofv_results=None
    )
    return res
