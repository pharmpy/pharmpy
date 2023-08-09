from typing import Callable, Generic, TypeVar

from pharmpy.internals.immutable import Immutable

T = TypeVar('T')


class Task(Generic[T], Immutable):
    """One task in a workflow

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
        self._name = name
        self._function = function
        self._task_input: tuple = task_input

    @classmethod
    def create(cls, name: str, function: Callable[..., T], *task_input):
        return cls(name, function, *task_input)

    def replace(self, **kwargs):
        name = kwargs.get("name", self._name)
        function = kwargs.get("function", self._function)
        task_input = kwargs.get("task_input", self._task_input)
        return Task.create(
            name,
            function,
            *task_input,
        )

    @property
    def name(self):
        """Task name"""
        return self._name

    @property
    def function(self):
        """Task function"""
        return self._function

    @property
    def task_input(self):
        """Tuple of static input to function"""
        return self._task_input

    def __repr__(self):
        return self._name
