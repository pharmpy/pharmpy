import pharmpy.tools
from pharmpy.tools.workflows import Task


class Modelfit(pharmpy.tools.Tool):
    def __init__(self, models, **kwargs):
        self.models = models
        super().__init__(**kwargs)

    def run(self):
        wf_fit = self.workflow_creator(self.models)
        task_result = Task('results', post_process_results, final_task=True)
        wf_fit.add_tasks(task_result, connect=True)
        for model in self.models:
            model.dataset
            model.database = self.database.model_database
        fit_models = self.dispatcher.run(wf_fit, self.database)
        for i in range(len(fit_models)):
            self.models[i].modelfit_results = fit_models[i].modelfit_results
        # res = self.models[0].modelfit_results
        # res.to_json(path=self.rundir.path / 'results.json')
        # res.to_csv(path=self.rundir.path / 'results.csv')


def post_process_results(models):
    return models
