from pathlib import Path

from pharmpy.model import Model
from pharmpy.utils import TemporaryDirectory

from .databases import LocalDirectoryDatabase, LocalDirectoryToolDatabase
from .dispatchers import LocalDispatcher

default_model_database = LocalDirectoryDatabase()
default_tool_database = LocalDirectoryToolDatabase
default_dispatcher = LocalDispatcher()


def execute_workflow(workflow, dispatcher=None, database=None, path=None):
    if dispatcher is None:
        dispatcher = default_dispatcher
    if database is None:
        database = default_tool_database(toolname=workflow.name, path=path)

    # For all input models set new database and read in results
    input_models = []
    for task in workflow.tasks:
        if task.has_input():
            new_inp = []
            for inp in task.task_input:
                if isinstance(inp, Model):
                    new_model = inp.copy()
                    new_model.database = database.model_database
                    try:
                        new_model.modelfit_results.predictions
                    except (AttributeError, KeyError):
                        pass
                    new_model.dataset
                    new_inp.append(new_model)
                    input_models.append(new_model)
                else:
                    new_inp.append(inp)
            task.task_input = new_inp

    res = dispatcher.run(workflow, database)
    # Transfer files from tool model database to default model database
    for model in input_models:
        with TemporaryDirectory() as temppath:
            database.model_database.retrieve_local_files(model.name, temppath)
            for f in Path(temppath).glob('*'):
                # Do not copy the model file.
                if f.name != model.source.path.name:
                    default_model_database.store_local_file(model, f)
        # Set modelfit_results for local model objects
        new_model = default_model_database.get_model(model.name)
        model.modelfit_results = new_model.modelfit_results

    return res
