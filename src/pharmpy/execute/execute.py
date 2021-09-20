from pharmpy.model import Model

from .databases import LocalDirectoryToolDatabase
from .dispatchers import LocalDispatcher

default_tool_database = LocalDirectoryToolDatabase
default_dispatcher = LocalDispatcher()


def execute_workflow(workflow, dispatcher=None, database=None, path=None):
    if dispatcher is None:
        dispatcher = default_dispatcher
    if database is None:
        database = default_tool_database(toolname=workflow.name, path=path)

    # For all input models set new database and read in results
    for task in workflow.tasks:
        if task.has_input():
            new_inp = []
            for inp in task.task_input:
                if isinstance(inp, Model):
                    new_model = inp.copy()
                    new_model.database = database.model_database
                    try:
                        new_model.modelfit_results.residuals
                        new_model.modelfit_results.predictions
                    except (AttributeError, KeyError):
                        pass
                    new_inp.append(new_model)
                else:
                    new_inp.append(inp)
            task.task_input = new_inp

    res = dispatcher.run(workflow, database)
    return res
