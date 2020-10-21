class Workflow:
    """Representation of an entire workflow
    The first set of tasks of a workflow is start
    """

    pass


class Task:
    """Representation of one task in a workflow

    attributes: successors (set of tasks to run after this task has been completed)
    run: what to be done for the task.
    """

    pass


class LocalNONMEMUnix:
    def run(arg):
        """The argument is a model object"""
