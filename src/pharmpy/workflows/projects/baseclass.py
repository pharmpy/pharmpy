from abc import ABC, abstractmethod
from typing import Optional


class Project(ABC):
    def __init__(
        self,
        name: str,
        ref: Optional[str] = None,
    ):
        # If the project already exists it will be opened, otherwise created
        # An implementation needs to create the model database here
        # If ref is None an implementation specific default ref will be used
        self._name = name
        if ref is None:
            ref = ""
        self._ref = ref

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def ref(self) -> str:
        return self._ref
