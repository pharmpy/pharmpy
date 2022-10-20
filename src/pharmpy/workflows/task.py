from typing import Callable, Generic, TypeVar

T = TypeVar('T')


class Task(Generic[T]):
    """One task

    Parameters
    ----------
    name : str
        Name of task
    function : func
        Task function
    task_input : any
        Input arguments to func
    """

    def __init__(self, name: str, function: Callable[..., T], *task_input):
        self.name = name
        self.function = function
        self.task_input: tuple = task_input

    def __repr__(self):
        return self.name
