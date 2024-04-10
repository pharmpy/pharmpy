from typing import Iterable, Literal, Optional, Tuple, Union

from pharmpy.model import Model
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.hashing import ModelHash

SupportedExternalTools = Literal['nonmem', 'nlmixr', 'rxode']


def create_workflow(
    model_or_models: Optional[Union[Model, Iterable[Model]]] = None,
    n: Optional[int] = None,
    tool: Optional[SupportedExternalTools] = None,
) -> Workflow[Union[Model, Tuple[Model, ...]]]:
    """Run modelfit tool.

    Parameters
    ----------
    model_or_models : Model
        A list of models are one single model object
    n : int
        Number of models to fit. This is only used if the tool is going to be combined with other tools.
    tool : str
        Which tool to use for fitting. Currently, 'nonmem', 'nlmixr', 'rxode' can be used.

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
    if model_or_models:
        if not isinstance(model_or_models, Iterable):
            model_or_models = [model_or_models]
        modelentries = [ModelEntry.create(model=model) for model in model_or_models]
    wf = create_fit_workflow(modelentries, n, tool)
    wf = wf.replace(name="modelfit")
    if len(modelentries) == 1 or (modelentries is None and n is None):
        post_process_results = post_process_results_one
    else:
        post_process_results = post_process_results_many
    task_result: Task[Union[Model, Tuple[Model, ...]]] = Task('results', post_process_results)
    wb = WorkflowBuilder(wf)
    wb.add_task(task_result, predecessors=wf.output_tasks)
    wf = Workflow(wb)
    return wf


def create_fit_workflow(modelentries=None, n=None, tool=None):
    execute_model = retrieve_from_database_or_execute_model_with_tool(tool)

    wb = WorkflowBuilder()
    if modelentries is None:
        if n is None:
            task = Task('run', execute_model)
            wb.add_task(task)
        else:
            for i in range(n):
                task = Task(f'run{i}', execute_model)
                wb.add_task(task)
    elif isinstance(modelentries, ModelEntry):
        task = Task('run', execute_model, modelentries)
        wb.add_task(task)
    else:
        assert all(isinstance(m, ModelEntry) for m in modelentries)
        for i, model_entry in enumerate(modelentries):
            task = Task(f'run{i}', execute_model, model_entry)
            wb.add_task(task)
    return Workflow(wb)


def post_process_results_one(context, *model_entry: ModelEntry):
    return model_entry[0].modelfit_results


def post_process_results_many(context, *modelentries: ModelEntry):
    return tuple([m.modelfit_results for m in modelentries])


def retrieve_from_database_or_execute_model_with_tool(tool):
    def task(context, model_entry):
        assert isinstance(model_entry, ModelEntry)
        model = model_entry.model
        try:
            db_model_entry = context.model_database.retrieve_model_entry(model)
        except (KeyError, AttributeError, FileNotFoundError):
            db_model_entry = None

        if db_model_entry and db_model_entry.modelfit_results is not None:
            if model.has_same_dataset_as(db_model_entry.model):
                me = model_entry.attach_results(db_model_entry.modelfit_results, db_model_entry.log)
                context.store_key(model.name, ModelHash(model))
                context.store_annotation(model.name, model.description)
                return me

        # NOTE: Fallback to executing the model
        execute_model = get_execute_model(tool)
        me = execute_model(model_entry, context)
        return me

    return task


def get_execute_model(tool: Optional[SupportedExternalTools]):
    from pharmpy.tools.modelfit import conf

    if tool is None:
        tool = conf.default_tool

    if tool == 'nonmem':
        from pharmpy.tools.external.nonmem.run import execute_model
    elif tool == 'nlmixr':
        from pharmpy.tools.external.nlmixr.run import execute_model
    elif tool == 'rxode':
        from pharmpy.tools.external.rxode.run import execute_model
    else:
        raise ValueError(f"Unknown estimation tool {tool}")

    return execute_model
