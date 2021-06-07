import pharmpy.tools
from pharmpy.tools.workflows import Task


class Modelfit(pharmpy.tools.Tool):
    def __init__(self, models, **kwargs):
        self.models = models
        super().__init__(**kwargs)

    def run(self):
        wf = self.workflow_creator(self.models)
        leaf_tasks = wf.get_leaf_tasks()
        result_task = Task('results', post_process_results, leaf_tasks, final_task=True)
        wf.add_tasks(result_task)
        wf.connect_tasks({task: result_task for task in leaf_tasks})
        for model in self.models:
            model.dataset
            model.database = self.database.model_database
        fit_models = self.dispatcher.run(wf, self.database)
        for i in range(len(fit_models)):
            self.models[i].modelfit_results = fit_models[i].modelfit_results
        # res = self.models[0].modelfit_results
        # res.to_json(path=self.rundir.path / 'results.json')
        # res.to_csv(path=self.rundir.path / 'results.csv')


def post_process_results(models):
    return [model for model_sublist in models for model in model_sublist]
