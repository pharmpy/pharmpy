from pathlib import Path

from pharmpy.utils import TemporaryDirectory


def execute_workflow(workflow, dispatcher=None, database=None, path=None):
    """Execute workflow

    Parameters
    ----------
    dispatcher : ExecutionDispatcher
        Dispatcher to use. None for the default dispatcher
    database : ToolDatabase
        Tool database to use. None for the default Tool database.
    path : Path
        Path to use for database if applicable.

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
            toolname=workflow.name, path=path
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
                    new_model.dataset
                    new_model.database = database.model_database
                    new_inp.append(new_model)
                    input_models.append(new_model)
                else:
                    new_inp.append(inp)
            task.task_input = new_inp

    res = dispatcher.run(workflow)

    # Transfer files from tool model database to default model database
    for model in original_input_models:
        with TemporaryDirectory() as temppath:
            database.model_database.retrieve_local_files(model.name, temppath)
            for f in Path(temppath).glob('*'):
                # Copies all result files, copy model file if model does not have a file
                model_file = model.name + model.filename_extension
                if f.name != model_file or not (model.database.path / model_file).exists():
                    model.database.store_local_file(model, f)
        if isinstance(res, Model):
            # Special case to handle modelfit for generic models
            model.modelfit_results = res.modelfit_results
        else:
            # Set modelfit_results for local model objects
            try:
                model.read_modelfit_results()
            except NotImplementedError:
                pass

    from pharmpy.results import Results

    if isinstance(res, Results):
        database.store_results(res)
        if hasattr(res, 'rst_path'):
            from pharmpy.modeling.reporting import create_report

            create_report(res, database.path)

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
    execute_options = ['path']
    common_options = dict()
    other_options = dict()
    for key, value in d.items():
        if key in execute_options:
            common_options[key] = value
        else:
            other_options[key] = value
    return common_options, other_options
