import pharmpy.tools
from pharmpy.data.iterators import Resample
from pharmpy.tools.bootstrap.results import calculate_results
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


class Bootstrap(pharmpy.tools.Tool):
    def __init__(self, model, resamples=1):
        self.model = model
        self.resamples = resamples
        super().__init__()
        self.model.database = self.database.model_database

    def run(self):
        wf_bootstrap = self.create_workflow()
        res = self.dispatcher.run(wf_bootstrap, self.database)
        res.to_json(path=self.database.path / 'results.json')
        res.to_csv(path=self.database.path / 'results.csv')
        return res

    def create_workflow(self):
        wf = Workflow()

        for i in range(self.resamples):
            task_resample = Task('resample', resample_model, self.model, f'bs_{i + 1}')
            wf.add_task(task_resample)

        wf_fit = create_fit_workflow(n=self.resamples)
        wf.insert_workflow(wf_fit)

        task_result = Task('results', post_process_results, self.model)
        wf.add_task(task_result, predecessors=wf.output_tasks)

        return wf


def resample_model(model, name):
    resample = Resample(model, model.datainfo.id_label, resamples=1, name=name)
    model, _ = next(resample)
    return model


def post_process_results(original_model, *models):
    res = calculate_results(
        models, original_model=original_model, included_individuals=None, dofv_results=None
    )
    return res
