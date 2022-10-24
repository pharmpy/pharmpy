from pathlib import Path

import pharmpy.tools.estmethod.algorithms as algorithms
from pharmpy.deps import pandas as pd
from pharmpy.model import Results
from pharmpy.tools import summarize_modelfit_results
from pharmpy.tools.common import summarize_tool
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.workflows import Task, Workflow


def create_workflow(algorithm, methods=None, solvers=None, model=None):
    wf = Workflow()
    wf.name = "estmethod"

    algorithm_func = getattr(algorithms, algorithm)

    if model is not None:
        start_task = Task('start_estmethod', start, model)
    else:
        start_task = Task('start_estmethod', start)

    wf.add_task(start_task)

    methods, solvers = _format_input_options(methods, solvers)

    wf_algorithm, task_base_model_fit = algorithm_func(methods, solvers)
    wf.insert_workflow(wf_algorithm, predecessors=start_task)

    wf_fit = create_fit_workflow(n=len(wf.output_tasks))
    wf.insert_workflow(wf_fit, predecessors=wf.output_tasks)

    if task_base_model_fit:
        model_tasks = wf.output_tasks + task_base_model_fit
    else:
        model_tasks = wf.output_tasks

    task_post_process = Task('post_process', post_process, model)
    wf.add_task(task_post_process, predecessors=model_tasks)

    return wf


def _format_input_options(methods, solvers):
    if not methods:
        methods = ['foce', 'fo', 'imp', 'impmap', 'its', 'saem', 'laplace', 'bayes']
    elif isinstance(methods, str):
        methods = [methods]
    elif isinstance(methods, list):
        methods = [method.lower() for method in methods]

    if solvers == 'all':
        solvers = [None, 'cvodes', 'dgear', 'dverk', 'ida', 'lsoda', 'lsodi']
    elif isinstance(solvers, str) or not solvers:
        solvers = [solvers]
    elif isinstance(solvers, list):
        solvers = [solver.lower() for solver in solvers]
    if None not in solvers:
        solvers.insert(0, None)

    return methods, solvers


def start(model):
    return model


def post_process(input_model, *models):
    res_models = []
    base_model = input_model
    for model in models:
        if model.name == 'base_model':
            base_model = model
        else:
            res_models.append(model)

    # FIXME: support other rank_type, allow None as cutoff
    rank_type = 'ofv'
    summary_tool = summarize_tool(
        res_models,
        base_model,
        rank_type,
        -1000,
    )
    summary_models = summarize_modelfit_results(
        [base_model.modelfit_results] + [model.modelfit_results for model in res_models]
    )
    summary_settings = summarize_estimation_steps([base_model] + res_models)

    if base_model.name == input_model.name:
        models = res_models
    else:
        models = [base_model] + res_models

    res = EstMethodResults(
        summary_tool=summary_tool,
        summary_models=summary_models,
        summary_settings=summary_settings,
        input_model=input_model,
        models=models,
    )

    return res


class EstMethodResults(Results):
    rst_path = Path(__file__).resolve().parent / 'report.rst'

    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        summary_settings=None,
        best_model=None,
        input_model=None,
        models=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.summary_settings = summary_settings
        self.best_model = best_model
        self.input_model = input_model
        self.models = models


def summarize_estimation_steps(models):
    dfs = {}
    for model in models:
        df = model.estimation_steps.to_dataframe()
        df.index = range(1, len(df) + 1)
        dfs[model.name] = df.drop(columns=['tool_options'])

    return pd.concat(dfs.values(), keys=dfs.keys())
