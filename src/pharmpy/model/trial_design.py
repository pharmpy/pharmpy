from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Optional, Union, overload

from pharmpy.internals.immutable import Immutable
from pharmpy.model.datainfo import DataVariable
from pharmpy.model.statements import Dose


class Activity(Immutable):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Activity:
        if d['class'] == 'Observations':
            act = Observations.from_dict(d)
        else:
            act = Administration.from_dict(d)
        return act


class Observations(Activity):
    """Observation activity"""

    def __init__(self, variable: DataVariable, start_time: float, time_points: tuple[float, ...]):
        self._variable = variable
        self._start_time = start_time
        self._time_points = time_points

    @classmethod
    def create(
        cls, variable: DataVariable, start_time: float, time_points: Sequence[float]
    ) -> Observations:
        return cls(variable, start_time, tuple(time_points))

    def replace(
        self,
        variable: Optional[DataVariable] = None,
        start_time: Optional[float] = None,
        time_points: Optional[Sequence[float]] = None,
    ) -> Observations:
        if variable is None:
            variable = self._variable
        if start_time is None:
            start_time = self._start_time
        if time_points is None:
            time_points = self._time_points
        return Observations.create(variable, start_time, time_points)

    @property
    def variable(self) -> DataVariable:
        """Observed variable"""
        return self._variable

    @property
    def start_time(self) -> float:
        """Start time of observation sequence"""
        return self._start_time

    @property
    def time_points(self) -> tuple[float, ...]:
        """Observation times relative to start_time"""
        return self._time_points

    def to_dict(self) -> dict[str, Any]:
        return {
            'class': 'Observations',
            'variable': self._variable.to_dict(),
            'start_time': self._start_time,
            'time_points': self._time_points,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Observations:
        return cls.create(DataVariable.from_dict(d['variable']), d['start_time'], d['time_points'])

    def __eq__(self, other: Any):
        if self is other:
            return True
        if not isinstance(other, Observations):
            return NotImplemented
        return (
            self._variable == other._variable
            and self._start_time == other._start_time
            and self._time_points == other._time_points
        )

    def __hash__(self):
        return hash((self._variable, self._start_time, self._time_points))

    def __repr__(self):
        return f"Observations({self._variable.name}, {self._start_time}, {self._time_points})"


class Administration(Activity):
    """Administration activity"""

    def __init__(
        self, variable: DataVariable, dose: Dose, start_time: float, time_points: tuple[float, ...]
    ):
        self._variable = variable
        self._dose = dose
        self._start_time = start_time
        self._time_points = time_points

    @classmethod
    def create(
        cls, variable: DataVariable, dose: Dose, start_time: float, time_points: Sequence[float]
    ) -> Administration:
        return cls(variable, dose, start_time, tuple(time_points))

    def replace(
        self,
        variable: Optional[DataVariable] = None,
        dose: Optional[Dose] = None,
        start_time: Optional[float] = None,
        time_points: Optional[Sequence[float]] = None,
    ) -> Administration:
        if variable is None:
            variable = self._variable
        if dose is None:
            dose = self._dose
        if start_time is None:
            start_time = self._start_time
        if time_points is None:
            time_points = self._time_points
        return Administration.create(variable, dose, start_time, time_points)

    @property
    def variable(self) -> DataVariable:
        """The dose data variable"""
        return self._variable

    @property
    def dose(self) -> Dose:
        """The dose"""
        return self._dose

    @property
    def start_time(self) -> float:
        """Start time of observation sequence"""
        return self._start_time

    @property
    def time_points(self) -> tuple[float, ...]:
        """Observation times relative to start_time"""
        return self._time_points

    def to_dict(self) -> dict[str, Any]:
        return {
            'class': 'Administration',
            'variable': self._variable.to_dict(),
            'dose': self._dose.to_dict(),
            'start_time': self._start_time,
            'time_points': self._time_points,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Administration:
        return cls.create(
            DataVariable.from_dict(d['variable']),
            Dose.from_dict(d['dose']),
            d['start_time'],
            d['time_points'],
        )

    def __eq__(self, other: Any):
        if self is other:
            return True
        if not isinstance(other, Administration):
            return NotImplemented
        return (
            self._variable == other._variable
            and self._dose == other._dose
            and self._start_time == other._start_time
            and self._time_points == other._time_points
        )

    def __hash__(self):
        return hash((self._variable, self._dose, self._start_time, self._time_points))

    def __repr__(self):
        return f"Administration({self._variable.name}, {self._dose}, {self._start_time}, {self._time_points})"


class Arm:
    """Arm definition"""

    def __init__(self, size: int, activities: tuple[Activity, ...]):
        self._size = size
        self._activities = activities

    @classmethod
    def create(cls, size: int, activities: Sequence[Activity]) -> Arm:
        for act in activities:
            if not isinstance(act, Activity):
                raise TypeError("Activities in Arm must be of type Activity")
        return cls(size, tuple(activities))

    def replace(
        self,
        size: Optional[int] = None,
        activities: Optional[Sequence[Activity]] = None,
    ) -> Arm:
        if size is None:
            size = self._size
        if activities is None:
            activities = self._activities
        return Arm.create(size=size, activities=activities)

    @property
    def size(self) -> int:
        """Size of arm"""
        return self._size

    @property
    def activities(self) -> tuple[Activity, ...]:
        """All activities in the arm"""
        return self._activities

    def to_dict(self) -> dict[str, Any]:
        acts = tuple(a.to_dict() for a in self)
        return {
            'size': self._size,
            'activities': acts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Arm:
        acts = []
        for sdict in d['activities']:
            acts.append(Activity.from_dict(sdict))
        return cls.create(size=d['size'], activities=acts)

    def __len__(self):
        return len(self._activities)

    @overload
    def __getitem__(self, ind: int) -> Activity: ...

    @overload
    def __getitem__(self, ind: slice) -> Arm: ...

    def __getitem__(self, ind: Union[int, slice]) -> Union[Activity, Arm]:
        if isinstance(ind, slice):
            return self.replace(activities=self._activities[ind])
        else:
            return self._activities[ind]

    def __add__(self, other: Union[Activity, Iterable[Activity]]) -> Arm:
        if isinstance(other, Activity):
            return self.replace(activities=self._activities + (other,))
        elif isinstance(other, Iterable):
            return self.replace(activities=self._activities + tuple(other))
        else:
            return NotImplemented

    def __radd__(self, other: Union[Activity, Iterable[Activity]]) -> Arm:
        if isinstance(other, Activity):
            return self.replace(activities=(other,) + self._activities)
        elif isinstance(other, Iterable):
            return self.replace(activities=tuple(other) + self._activities)
        else:
            return NotImplemented

    def __eq__(self, other: Any):
        if self is other:
            return True
        if not isinstance(other, Arm):
            return NotImplemented
        return self._size == other._size and self._activities == other._activities

    def __hash__(self):
        return hash((self._size, self._activities))

    def __repr__(self):
        return f"Arm(size={self._size}, {self._activities})"
