from __future__ import annotations

from typing import Callable, Self

from pharmpy.internals.immutable import Immutable


class Task(Immutable):
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

    def __init__(self, name: str, function: Callable, *task_input):
        self._name = name
        self._function = function
        self._task_input: tuple = task_input

    @classmethod
    def create(cls, name: str, function: Callable, *task_input) -> Self:
        return cls(name, function, *task_input)

    def replace(self, **kwargs) -> Task:
        name = kwargs.get("name", self._name)
        function = kwargs.get("function", self._function)
        task_input = kwargs.get("task_input", self._task_input)
        return Task.create(
            name,
            function,
            *task_input,
        )

    @property
    def name(self) -> str:
        """Task name"""
        return self._name

    @property
    def function(self) -> Callable:
        """Task function"""
        return self._function

    @property
    def task_input(self) -> tuple:
        """Tuple of static input to function"""
        return self._task_input

    def __repr__(self) -> str:
        return self._name
