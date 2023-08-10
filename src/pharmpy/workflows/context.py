import inspect

from .workflow import WorkflowBuilder


def insert_context(wb: WorkflowBuilder, context):
    """Insert tool context (database) for all tasks in a workflow needing it

    having context as first argument of function
    """
    for task in wb.tasks:
        parameters = tuple(inspect.signature(task.function).parameters)
        if parameters and parameters[0] == 'context':
            new_task = task.replace(task_input=(context, *task.task_input))
            wb.replace_task(task, new_task)
