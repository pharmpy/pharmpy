import inspect

from .workflow import Workflow


def insert_context(workflow: Workflow, context):
    """Insert tool context (database) for all tasks in a workflow needing it

    having context as first argument of function
    """
    for task in workflow.tasks:
        parameters = tuple(inspect.signature(task.function).parameters)
        if parameters and parameters[0] == 'context':
            task.task_input = (context, *task.task_input)
