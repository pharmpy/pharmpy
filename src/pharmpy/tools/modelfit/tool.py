import pharmpy.tools
from pharmpy.tools.workflows import Task


class Modelfit(pharmpy.tools.Tool):
    def __init__(self, models, **kwargs):
        self.models = models
        super().__init__(**kwargs)

    def run(self):
        for model in self.models:
            model.dataset
            model.database = self.database.model_database
        workflow = self.workflow_creator(self.models)
        workflow.add_tasks(Task('results', final_models, [workflow.tasks[-1]], final_task=True))
        fit_models = self.dispatcher.run(workflow, self.database)
        for i in range(len(fit_models)):
            self.models[i].modelfit_results = fit_models[i].modelfit_results
        # res = self.models[0].modelfit_results
        # res.to_json(path=self.rundir.path / 'results.json')
        # res.to_csv(path=self.rundir.path / 'results.csv')


def final_models(models):
    return models
