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
        wf = Workflow()

        for i in range(self.resamples):
            resample_task = Task('resample', resample_model, self.model, f'bs_{i + 1}')
            wf.add_tasks(resample_task)
            run_workflow = self.workflow_creator([resample_task])
            wf.merge_workflows(
                run_workflow,
                edges={resample_task: run_task for run_task in run_workflow.get_root_tasks()},
            )

        # db = NullToolDatabase
        # db.model_database = self.database.model_database
        # modelfit_run = modelfit.Modelfit(resample_names, database=db)
        # modelfit_workflow = modelfit_run.workflow_creator(resample_names)

        leaf_tasks = wf.get_leaf_tasks()
        result_task = Task('results', post_process_results, leaf_tasks, self.model, final_task=True)
        wf.add_tasks(result_task)
        wf.connect_tasks({task: result_task for task in leaf_tasks})

        res = self.dispatcher.run(wf, self.database)
        res.to_json(path=self.database.path / 'results.json')
        res.to_csv(path=self.database.path / 'results.csv')
        return res


def resample_model(model, name):
    resample = Resample(model, model.dataset.pharmpy.id_label, resamples=1, name=name)
    model, _ = next(resample)
    return model


def post_process_results(models, original_model):
    models = [model for model_sublist in models for model in model_sublist]
    res = calculate_results(
        models, original_model=original_model, included_individuals=None, dofv_results=None
    )
    return res
