from pathlib import Path

from pharmpy.utils import TemporaryDirectory, normalize_user_given_path


def execute_workflow(workflow, dispatcher, database):
    """Execute workflow

    Parameters
    ----------
    workflow : Workflow
        Workflow to execute
    dispatcher : ExecutionDispatcher
        Dispatcher to use
    database : ToolDatabase
        Tool database to use

    Returns
    -------
    Results
        Results object created by workflow
    """
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
    # FIXME: add dispatcher/database
    execute_options = ['path']
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
