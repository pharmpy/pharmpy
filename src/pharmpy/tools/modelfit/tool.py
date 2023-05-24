from typing import Iterable, Literal, Optional, Tuple, Union

from pharmpy.model import Model
from pharmpy.workflows import Task, Workflow

SupportedPlugin = Literal['nonmem', 'nlmixr']


def create_workflow(
    model_or_models: Optional[Union[Model, Iterable[Model]]] = None,
    n: Optional[int] = None,
    tool: Optional[SupportedPlugin] = None,
) -> Workflow[Union[Model, Tuple[Model, ...]]]:
    """Run modelfit tool.

    Parameters
    ----------
    model_or_models : Model
        A list of models are one single model object
    n : int
        Number of models to fit. This is only used if the tool is going to be combined with other tools.
    tool : str
        Which tool to use for fitting. Currently 'nonmem' or 'nlmixr' can be used.

    Returns
    -------
    ModelfitResults
        Modelfit tool result object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model
    >>> model = load_example_model("pheno")
    >>> from pharmpy.tools import run_modelfit     # doctest: +SKIP
    >>> run_modelfit(model)   # doctest: +SKIP
    """

    wf = create_fit_workflow(model_or_models, n, tool)
    wf.name = "modelfit"
    if isinstance(model_or_models, Model) or (model_or_models is None and n is None):
        post_process_results = post_process_results_one
    else:
        post_process_results = post_process_results_many
    task_result: Task[Union[Model, Tuple[Model, ...]]] = Task('results', post_process_results)
    wf.add_task(task_result, predecessors=wf.output_tasks)
    return wf


def create_fit_workflow(models=None, n=None, tool=None):
    execute_model = retrieve_from_database_or_execute_model_with_tool(tool)

    wf = Workflow()
    if models is None:
        if n is None:
            task = Task('run', execute_model)
            wf.add_task(task)
        else:
            for i in range(n):
                task = Task(f'run{i}', execute_model)
                wf.add_task(task)
    elif isinstance(models, Model):
        task = Task('run', execute_model, models)
        wf.add_task(task)
    else:
        for i, model in enumerate(models):
            task = Task(f'run{i}', execute_model, model)
            wf.add_task(task)
    return wf


def post_process_results_one(context, *models: Model):
    return models[0]


def post_process_results_many(context, *models: Model):
    return models


def retrieve_from_database_or_execute_model_with_tool(tool):
    def task(context, model):
        try:
            db_results = context.model_database.retrieve_modelfit_results(model.name)
        except (KeyError, AttributeError, FileNotFoundError):
            db_results = None

        if db_results is not None:
            # NOTE We have the results
            try:
                db_model = context.model_database.retrieve_model(model.name)
            except (KeyError, AttributeError, FileNotFoundError):
                db_model = None

            # NOTE Here we could invalidate cached results if certain errors
            # happened such as a missing or outdated license. We do not do that
            # at the moment.

            # NOTE Right now we only rely on model name comparison
            # if db_model == model and model.has_same_dataset_as(db_model):
            if db_model and model.has_same_dataset_as(db_model):
                # NOTE Inputs are identical so we can reuse the results
                model = model.replace(modelfit_results=db_results)
                return model

        # NOTE Fallback to executing the model
        execute_model = get_execute_model(tool)
        return execute_model(model, context)

    return task


def get_execute_model(tool: Optional[SupportedPlugin]):
    from pharmpy.tools.modelfit import conf

    if tool is None:
        tool = conf.default_tool

    if tool == 'nonmem':
        from pharmpy.tools.external.nonmem.run import execute_model
    elif tool == 'nlmixr':
        from pharmpy.tools.external.nlmixr.run import execute_model
    else:
        raise ValueError(f"Unknown estimation tool {tool}")

    return execute_model
