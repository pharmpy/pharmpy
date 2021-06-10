import pharmpy.tools
from pharmpy.data.iterators import Resample
from pharmpy.tools.bootstrap.results import calculate_results
from pharmpy.tools.workflows import Task, Workflow


class Bootstrap(pharmpy.tools.Tool):
    def __init__(self, model, resamples=1):
        self.model = model
        self.resamples = resamples
        super().__init__()
        self.model.database = self.database.model_database

    def run(self):
        wf_bootstrap = self.create_workflow()
        task_result = Task('results', post_process_results, final_task=True)
        wf_bootstrap.add_tasks(task_result, connect=True)
        res = self.dispatcher.run(wf_bootstrap, self.database)
        res.to_json(path=self.database.path / 'results.json')
        res.to_csv(path=self.database.path / 'results.csv')
        return res

    def create_workflow(self):
        wf_bootstrap = Workflow()

        for i in range(self.resamples):
            wf_resample = Workflow()
            task_resample = Task('resample', resample_model, self.model, f'bs_{i + 1}')
            wf_resample.add_tasks(task_resample, connect=False)
            wf_fit = self.workflow_creator()
            wf_resample.add_tasks(wf_fit, connect=True)

            wf_bootstrap.add_tasks(wf_resample, connect=False)

        return wf_bootstrap


def resample_model(model, name):
    resample = Resample(model, model.dataset.pharmpy.id_label, resamples=1, name=name)
    model, _ = next(resample)
    return model


def post_process_results(models):
    # Flattening nested list
    models = [model for model_sublist in models for model in model_sublist]
    res = calculate_results(
        models, original_model=None, included_individuals=None, dofv_results=None
    )
    return res
