from pharmpy.utils import normalize_user_given_path


def execute_workflow(workflow, dispatcher=None, database=None, path=None, resume=False):
    """Execute workflow

    Parameters
    ----------
    workflow : Workflow
        Workflow to execute
    dispatcher : ExecutionDispatcher
        Dispatcher to use
    database : ToolDatabase
        Tool database to use. None for the default Tool database.
    path : Path
        Path to use for database if applicable.
    resume : bool
        Whether to allow resuming previous workflow execution.

    Returns
    -------
    Results
        Results object created by workflow
    """
    if dispatcher is None:
        from pharmpy.workflows import default_dispatcher

        dispatcher = default_dispatcher
    if database is None:
        from pharmpy.workflows import default_tool_database

        database = default_tool_database(
            toolname=workflow.name, path=path, exist_ok=resume
        )  # TODO: database -> tool_database

    # For all input models set new database and read in results
    original_input_models = []
    input_models = []
    for task in workflow.tasks:
        if task.has_input():
            new_inp = []
            from pharmpy.model import Model

            for inp in task.task_input:
                if isinstance(inp, Model):
                    original_input_models.append(inp)
                    inp.modelfit_results  # To read in the results
                    new_model = inp.copy()
                    new_model.parent_model = new_model.name
                    new_model.dataset
                    new_model.database = database.model_database
                    new_inp.append(new_model)
                    input_models.append(new_model)
                else:
                    new_inp.append(inp)
            task.task_input = new_inp

    res = dispatcher.run(workflow)

    from pharmpy.results import Results

    if isinstance(res, Results):
        database.store_results(res)
        if hasattr(res, 'rst_path'):
            from pharmpy.modeling.reporting import create_report

            create_report(res, database.path)
    elif isinstance(res, Model):
        original_input_models[0].modelfit_results = res.modelfit_results
    elif isinstance(res, list) or isinstance(res, tuple):
        for model in res:
            for original_model in original_input_models:
                if original_model.name == model.name:
                    original_model.modelfit_results = model.modelfit_results
                    break

    return res


def split_common_options(d):
    """Split the dict into common options and other options

    Parameters
    ----------
    d : dict
        Dictionary of all options

    Returns
    -------
    Tuple of common options and other option dictionaries
    """
    execute_options = ['path', 'resume']
    common_options = dict()
    other_options = dict()
    for key, value in d.items():
        if key in execute_options:
            if key == 'path':
                if value is not None:
                    value = normalize_user_given_path(value)
            common_options[key] = value
        else:
            other_options[key] = value
    return common_options, other_options


def call_workflow(wf, unique_name):
    """Dynamically call a workflow from another workflow.

    Currently only supports dask distributed

    Parameters
    ----------
    wf : Workflow
        A workflow object
    unique_name : str
        A name of the results node that is unique between parent and dynamically created workflows

    Returns
    -------
    Any
        Whatever the dynamic workflow returns
    """
    from dask.distributed import get_client, rejoin, secede

    client = get_client()
    secede()
    d = wf.as_dask_dict()
    d[unique_name] = d.pop('results')
    res = client.get(d, unique_name)
    rejoin()
    return res
