import pharmpy.results
import pharmpy.tools
import pharmpy.tools.modelfit as modelfit
import pharmpy.tools.modelsearch.algorithms as algorithms
import pharmpy.tools.rankfuncs as rankfuncs
import pharmpy.workflows as workflows
from pharmpy.tools.modelfit import create_single_fit_workflow
from pharmpy.workflows import Task, Workflow


class ModelSearch(pharmpy.tools.Tool):
    def __init__(self, start_model, algorithm, mfl, rankfunc='ofv', cutoff=None, **kwargs):
        self.start_model = start_model
        self.mfl = mfl
        self.algorithm = getattr(algorithms, algorithm)
        self.rankfunc = getattr(rankfuncs, rankfunc)
        self.cutoff = cutoff
        super().__init__(**kwargs)
        self.start_model.database = self.database.model_database

    def fit(self, models):
        db = workflows.LocalDirectoryDatabase(self.rundir.path / 'models')
        modelfit_run = modelfit.Modelfit(models, database=db, path=self.rundir.path)
        modelfit_run.run()

    def run(self):
        df = self.algorithm(
            self.start_model,
            self.mfl,
            self.fit,
            self.rankfunc,
        )
        res = ModelSearchResults(summary=df)
        res.to_json(path=self.rundir.path / 'results.json')
        res.to_csv(path=self.rundir.path / 'results.csv')
        return res


def create_workflow(
    algorithm, mfl, rankfunc='ofv', cutoff=None, add_etas=False, etas_as_fullblock=False, model=None
):
    algorithm_func = getattr(algorithms, algorithm)

    wf = Workflow()
    wf.name = 'modelsearch'

    if model:
        start_task = Task('start_modelsearch', start, model)
    else:
        start_task = Task('start_modelsearch', start)

    wf.add_task(start_task)

    if model and not model.modelfit_results:
        wf_fit = create_single_fit_workflow()
        wf.insert_workflow(wf_fit, predecessors=start_task)
        start_model_task = wf_fit.output_tasks
    else:
        start_model_task = [start_task]

    wf_search, candidate_model_tasks, model_features = algorithm_func(
        mfl, add_etas, etas_as_fullblock
    )
    wf.insert_workflow(wf_search, predecessors=wf.output_tasks)

    task_result = Task(
        'results',
        post_process_results,
        rankfunc,
        cutoff,
        model_features,
    )

    wf.add_task(task_result, predecessors=start_model_task + candidate_model_tasks)

    return wf


def start(model):
    return model


def post_process_results(rankfunc, cutoff, model_features, *models):
    res_models = []
    for model in models:
        if not model.name.startswith('modelsearch_candidate'):
            start_model = model
        else:
            res_models.append(model)

    df = pharmpy.tools.common.create_summary(
        res_models, start_model, rankfunc, cutoff, model_features
    )

    best_model_name = df['rank'].idxmin()
    try:
        best_model = [model for model in res_models if model.name == best_model_name][0]
    except IndexError:
        best_model = start_model

    res = ModelSearchResults(
        summary=df, best_model=best_model, start_model=start_model, models=res_models
    )

    return res


class ModelSearchResults(pharmpy.results.Results):
    def __init__(self, summary=None, best_model=None, start_model=None, models=None):
        self.summary = summary
        self.best_model = best_model
        self.start_model = start_model
        self.models = models

    def to_json(self, path=None, lzma=False):
        s = pharmpy.results.Results.to_json(self.summary, path, lzma)
        if s:
            return s


def run_modelsearch(base_model, algorithm, mfl, **kwargs):
    ms = ModelSearch(base_model, algorithm, mfl, **kwargs)
    res = ms.run()
    return res
