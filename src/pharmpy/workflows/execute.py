from pathlib import Path

from pharmpy.utils import TemporaryDirectory

from .databases import LocalDirectoryDatabase, LocalDirectoryToolDatabase
from .dispatchers import LocalDispatcher

default_model_database = LocalDirectoryDatabase
default_tool_database = LocalDirectoryToolDatabase
default_dispatcher = LocalDispatcher()


def execute_workflow(workflow, dispatcher=None, database=None, path=None, directory=None):
    if dispatcher is None:
        dispatcher = default_dispatcher
    if database is None:
        database = default_tool_database(toolname=workflow.name, path=path)
    if directory is not None:
        pass

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

    res = dispatcher.run(workflow, database)
    # Transfer files from tool model database to default model database
    for model in original_input_models:
        with TemporaryDirectory() as temppath:
            database.model_database.retrieve_local_files(model.name, temppath)
            for f in Path(temppath).glob('*'):
                # Copies all result files, copy model file if model does not have a file
                model_file = model.name + model.filename_extension
                if f.name != model_file or not (model.database.path / model_file).exists():
                    model.database.store_local_file(model, f)
        # Set modelfit_results for local model objects
        model.read_modelfit_results()

    return res


def split_common_options(d):
    """Split the dict into common options and other options"""
    execute_options = ['directory']
    common_options = dict()
    other_options = dict()
    for key, value in d.items():
        if key in execute_options:
            common_options[key] = value
        else:
            other_options[key] = value
    return common_options, other_options
