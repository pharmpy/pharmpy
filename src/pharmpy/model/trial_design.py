from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Optional, Union, overload

from pharmpy.deps import pandas as pd
from pharmpy.deps.rich import box
from pharmpy.deps.rich import console as rich_console
from pharmpy.deps.rich import panel as rich_panel
from pharmpy.internals.immutable import Immutable
from pharmpy.internals.math import round_and_keep_sum
from pharmpy.model.datainfo import DataVariable
from pharmpy.model.statements import Dose


class Activity(Immutable):
    _start_time: float
    _time_points: tuple[float, ...]

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Activity:
        if d['class'] == 'Observations':
            act = Observations.from_dict(d)
        else:
            act = Administration.from_dict(d)
        return act

    @property
    def start_time(self) -> float:
        """Start time of activity"""
        return self._start_time

    @property
    def end_time(self) -> float:
        """End time of activity"""
        return self._start_time + self._time_points[-1]


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


class Arm(Sequence, Immutable):
    """Arm definition"""

    def __init__(self, name: str, size: int, activities: tuple[Activity, ...]):
        self._name = name
        self._size = size
        self._activities = activities

    @classmethod
    def create(cls, name: str, size: int, activities: Sequence[Activity]) -> Arm:
        for act in activities:
            if not isinstance(act, Activity):
                raise TypeError("Activities in Arm must be of type Activity")
        if not isinstance(name, str):
            raise TypeError("name of Arm must be str")
        if not isinstance(size, int):
            raise TypeError("size of Arm must be int")
        return cls(name, size, tuple(activities))

    def replace(
        self,
        name: Optional[str] = None,
        size: Optional[int] = None,
        activities: Optional[Sequence[Activity]] = None,
    ) -> Arm:
        if name is None:
            name = self._name
        if size is None:
            size = self._size
        if activities is None:
            activities = self._activities
        return Arm.create(name=name, size=size, activities=activities)

    @property
    def name(self) -> str:
        """Name of arm"""
        return self._name

    @property
    def size(self) -> int:
        """Size of arm"""
        return self._size

    @property
    def start_time(self) -> float:
        """Start time for Arm activities

        Will default to 0.0 for an Arm with no activities
        """
        if self._activities:
            start_time = min(act.start_time for act in self._activities)
        else:
            start_time = 0.0
        return start_time

    @property
    def end_time(self) -> float:
        """End time for Arm activites

        Will default to 0.0 for an Arm with no activites
        """
        if self._activities:
            end_time = max(act.end_time for act in self._activities)
        else:
            end_time = 0.0
        return end_time

    @property
    def activities(self) -> tuple[Activity, ...]:
        """All activities in the arm"""
        return self._activities

    def to_dict(self) -> dict[str, Any]:
        acts = tuple(a.to_dict() for a in self)
        return {
            'name': self._name,
            'size': self._size,
            'activities': acts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Arm:
        acts = []
        for sdict in d['activities']:
            acts.append(Activity.from_dict(sdict))
        return cls.create(name=d['name'], size=d['size'], activities=acts)

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
        return (
            self._name == other._name
            and self._size == other._size
            and self._activities == other._activities
        )

    def __hash__(self):
        return hash((self._name, self._size, self._activities))

    def __repr__(self):
        return f"Arm(name={self._name}, size={self._size}, {self._activities})"

    def is_placebo(self) -> bool:
        """Is this arm a placebo arm?"""
        for act in self._activities:
            if isinstance(act, Administration) and act.dose.amount != 0:
                return False
        return True


class TrialDesign(Sequence, Immutable):
    """TrialDesign"""

    def __init__(self, arms: tuple[Arm, ...], independent_variable: DataVariable):
        self._arms = arms
        self._independent_variable = independent_variable

    @classmethod
    def create(
        cls, arms: Sequence[Arm], independent_variable: Optional[DataVariable]
    ) -> TrialDesign:
        seen_names = set()
        for arm in arms:
            if not isinstance(arm, Arm):
                raise TypeError("Arms in TrialDesign must be of type Arm")
            if arm.name in seen_names:
                raise ValueError(f"The Arm name {arm.name} is not unique")
            else:
                seen_names.add(arm.name)
        if not isinstance(independent_variable, DataVariable):
            raise TypeError("The independent_variable of TrialDesign must be of type DataVariable")
        return cls(tuple(arms), independent_variable)

    def replace(
        self,
        arms: Optional[Sequence[Arm]] = None,
        independent_variable: Optional[DataVariable] = None,
    ) -> TrialDesign:
        if arms is None:
            arms = self._arms
        if independent_variable is None:
            independent_variable = self._independent_variable
        return TrialDesign.create(arms=arms, independent_variable=independent_variable)

    @property
    def arms(self) -> tuple[Arm, ...]:
        """The arms"""
        return self._arms

    @property
    def independent_variable(self) -> DataVariable:
        """Independent variable for the entire trial"""
        return self._independent_variable

    def to_dict(self) -> dict[str, Any]:
        arms = tuple(arm.to_dict() for arm in self)
        return {
            'arms': arms,
            'independent_variable': self._independent_variable.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrialDesign:
        arms = []
        for sdict in d['arms']:
            arms.append(Arm.from_dict(sdict))
        return cls.create(
            arms=arms, independent_variable=DataVariable.from_dict(d['independent_variable'])
        )

    def __len__(self):
        return len(self._arms)

    @overload
    def __getitem__(self, ind: int) -> Arm: ...

    @overload
    def __getitem__(self, ind: str) -> Arm: ...

    @overload
    def __getitem__(self, ind: slice) -> TrialDesign: ...

    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, ind: Union[int, slice, str]
    ) -> Union[Arm, TrialDesign]:
        if isinstance(ind, slice):
            return TrialDesign(
                arms=self._arms[ind], independent_variable=self._independent_variable
            )
        elif isinstance(ind, str):
            for arm in self._arms:
                if arm.name == ind:
                    return arm
            raise KeyError(f"Cannot find arm named {ind}")
        else:
            return self._arms[ind]

    def __add__(self, other: Union[Arm, Iterable[Arm]]) -> TrialDesign:
        if isinstance(other, Arm):
            return self.replace(arms=self._arms + (other,))
        elif isinstance(other, Iterable):
            return self.replace(arms=self._arms + tuple(other))
        else:
            return NotImplemented

    def __radd__(self, other: Union[Arm, Iterable[Arm]]) -> TrialDesign:
        if isinstance(other, Arm):
            return self.replace(arms=(other,) + self._arms)
        elif isinstance(other, Iterable):
            return self.replace(arms=tuple(other) + self._arms)
        else:
            return NotImplemented

    def __eq__(self, other: Any):
        if self is other:
            return True
        if not isinstance(other, TrialDesign):
            return NotImplemented
        return (
            self._arms == other._arms and self._independent_variable == other._independent_variable
        )

    def __hash__(self):
        return hash((self._arms, self._independent_variable))

    def __repr__(self):
        if self.is_empty:
            return "Empty trial design object"
        else:
            return render_trial_design(self)

    @property
    def is_empty(self):
        if not self._arms:
            return True
        for arm in self._arms:
            if len(arm) > 0:
                return False
        return True

    def replace_arm(self, arm: Arm) -> TrialDesign:
        """Replace an Arm with an existing name

        Parameters
        ----------
        arm : Arm
            New Arm

        Returns
        -------
        TrialDesign
            Updated TrialDesign
        """

        newarms = []
        for cur in self:
            if cur.name != arm.name:
                newarms.append(cur)
            else:
                newarms.append(arm)
        new_design = self.replace(arms=newarms)
        return new_design


def render_rich_object(obj):
    buffer = StringIO()
    console = rich_console.Console(file=buffer)
    console.print(obj)
    return buffer.getvalue().split('\n')[:-1]


@dataclass(frozen=True)
class TimeSlot:
    start: float
    end: float
    title: str
    footer: str
    content: str


@dataclass
class ScheduleGrid:
    lanes: list[list[TimeSlot]] = field(default_factory=lambda: [[]])

    def pack_activities(self):
        slots = sorted(self.lanes[0], key=lambda slot: slot.start)
        lanes = []
        end_times = []
        for slot in slots:
            for i, lane in enumerate(lanes):
                if slot.start >= end_times[i]:
                    lanes[i].append(slot)
                    end_times[i] = slot.end
                    break
            else:
                lanes.append([slot])
                end_times.append(slot.end)
        return ScheduleGrid(lanes=lanes)


@dataclass
class Timeline:
    grids: dict[str, ScheduleGrid] = field(default_factory=dict)

    def pack_activities(self):
        grids = {}
        for key, value in self.grids.items():
            grids[key] = value.pack_activities()
        return Timeline(grids=grids)

    @property
    def start(self) -> float:
        return min(lane[0].start for grid in self.grids.values() for lane in grid.lanes)

    @property
    def end(self) -> float:
        return max(lane[-1].end for grid in self.grids.values() for lane in grid.lanes)


def get_time_points(activity):
    # Make into method?
    adjusted_time_points = [activity.start_time + time for time in activity.time_points]
    return adjusted_time_points


def get_start_and_end_time(activity):
    start_time = activity.start_time
    end_time = activity.end_time
    return start_time, end_time


def get_unit_string(activity):
    unit = activity.variable.properties.get('unit', None)
    unit_str = "" if unit is None else " " + str(unit)
    return unit_str


def list_with_unit(x, unit=None):
    s = ", ".join(map(str, x))
    if unit is not None:
        s += f" {unit}"
    return s


def create_administration_slot(admin, idv_unit, end) -> TimeSlot:
    start, _ = get_start_and_end_time(admin)
    unit = get_unit_string(admin)
    title = "Administration"
    footer = f"{float(admin.dose.amount)}{unit} {admin.dose.__class__.__name__}"
    content = list_with_unit(admin.time_points, idv_unit)
    slot = TimeSlot(start, end, title, footer, content)
    return slot


def create_observation_slot(obs, idv_unit) -> TimeSlot:
    start, end = get_start_and_end_time(obs)
    title = "Observations"
    footer = obs.variable.name
    content = list_with_unit(get_time_points(obs), idv_unit)
    slot = TimeSlot(start, end, title, footer, content)
    return slot


def build_timeline(td: TrialDesign) -> Timeline:
    time_line = Timeline()
    idv_unit = td.independent_variable.properties.get("unit", None)

    for arm in td:
        # FIXME: Make these into properties
        admin_starts = [act.start_time for act in arm if isinstance(act, Administration)]
        arm_end = max(get_start_and_end_time(act)[1] for act in arm)
        admin_end_times = admin_starts[1:] + [arm_end]

        grid = ScheduleGrid()
        next_admin = 0
        for act in arm:
            if isinstance(act, Administration):
                slot = create_administration_slot(act, idv_unit, admin_end_times[next_admin])
                next_admin += 1
            else:  # Observations
                slot = create_observation_slot(act, idv_unit)
            grid.lanes[0].append(slot)  # Everything in first lane before packing
        time_line.grids[arm.name] = grid
    return time_line


class Block(ABC):
    @property
    @abstractmethod
    def min_char_size(self) -> int: ...

    @property
    @abstractmethod
    def min_chars_per_time(self) -> float: ...

    @abstractmethod
    def render(self, width: int, height: int) -> Tile: ...


@dataclass(frozen=True)
class EmptyBlock(Block):
    time_length: float

    @property
    def min_char_size(self) -> int:
        return 0

    @property
    def min_chars_per_time(self) -> float:
        return 0.0

    def render(self, width: int, height: int) -> Tile:
        return Tile([" " * width for _ in range(height)])


@dataclass(frozen=True)
class FramedBlock(Block):
    title: str
    footer: str
    content: str
    time_length: float

    @property
    def min_char_size(self) -> int:
        return max(len(self.title) + 2, len(self.footer) + 2, len(self.content) + 4)

    @property
    def min_chars_per_time(self) -> float:
        tl = self.time_length if self.time_length > 0 else 1.0
        return self.min_char_size / tl

    def render(self, width: int, height: int) -> Tile:
        panel = rich_panel.Panel(
            self.content,
            title=f"[cyan]{self.title}",
            subtitle=f"[dim]{self.footer}",
            border_style="green",
            width=width,
            height=height,
        )
        return Tile(render_rich_object(panel))


@dataclass(frozen=True)
class PlainBlock(Block):
    text: str

    @property
    def min_char_size(self):
        return len(self.text) + 4

    @property
    def min_chars_per_time(self) -> float:
        return 0.0

    def render(self, width: int, height: int) -> Tile:
        panel = rich_panel.Panel(self.text, box=box.SIMPLE, width=width, height=height)
        return Tile(render_rich_object(panel))


@dataclass(frozen=True)
class EmptyAlignedBlock(Block):
    @property
    def min_char_size(self) -> int:
        return 0

    @property
    def min_chars_per_time(self) -> float:
        return 0.0

    def render(self, width: int, height: int) -> Tile:
        return Tile([" " * width for _ in range(height)])


@dataclass
class BlockGrid:
    lanes: list[list[Block]] = field(default_factory=lambda: [[]])


@dataclass
class Tile:
    rows: list[str]

    def __add__(self, other: Tile) -> Tile:
        new_rows = [first + second for first, second in zip(self.rows, other.rows)]
        return Tile(new_rows)

    def __repr__(self):
        return '\n'.join(self.rows)


def timeline_to_block_grid(tl: Timeline) -> BlockGrid:
    all_blocks = []
    for name, schedule in tl.grids.items():
        for i, lane in enumerate(schedule.lanes):
            if i == 0:
                row_title = PlainBlock(name)
            else:
                row_title = EmptyAlignedBlock()
            lane_blocks: list[Block] = [row_title]
            prev_end = lane[0].start
            for slot in lane:
                if slot.start != prev_end:
                    empty = EmptyBlock(time_length=slot.start - prev_end)
                    lane_blocks.append(empty)
                block = FramedBlock(
                    title=slot.title,
                    footer=slot.footer,
                    content=slot.content,
                    time_length=slot.end - slot.start,
                )
                lane_blocks.append(block)
                prev_end = slot.end
            all_blocks.append(lane_blocks)
    grid = BlockGrid(lanes=all_blocks)
    return grid


def calculate_row_header_width(grid: BlockGrid) -> int:
    min_char_size_first_column = max(lane[0].min_char_size for lane in grid.lanes)
    return min_char_size_first_column


def calculate_needed_chars_per_time(grid: BlockGrid) -> float:
    min_chars_per_time = max(block.min_chars_per_time for lane in grid.lanes for block in lane)
    return min_chars_per_time


def text_axis(points, size):
    STARTCH = "├"
    ENDCH = "┤"
    TICKCH = "┬"
    BARCH = "─"

    chars_for_bars = size - len(points)
    interval_per_char = (points[-1] - points[0]) / chars_for_bars

    distances = [j - i for i, j in zip(points, points[1:])]
    char_distances = [dist / interval_per_char for dist in distances]
    char_distances = list(round_and_keep_sum(pd.Series(char_distances), chars_for_bars))

    bars = [BARCH * n for n in char_distances]
    ticked_bars = STARTCH + TICKCH.join(bars) + ENDCH

    point_strings = list(map(str, points))

    chars_after_tick = [len(s) - (len(s) // 2) - 1 for s in point_strings]
    chars_before_tick = [len(s) - n - 1 for s, n in zip(point_strings, chars_after_tick)]

    spaces = [
        " " * (dist - before - after)
        for before, after, dist in zip(chars_before_tick[1:], chars_after_tick, char_distances)
    ]
    interleaved = [point_strings[0]] + [x for pair in zip(spaces, point_strings[1:]) for x in pair]
    points_line = "".join(interleaved)
    padding = " " * chars_before_tick[0]
    return Tile([padding + ticked_bars, points_line])


def print_grid(grid: BlockGrid, start: float, end: float) -> str:
    s = ""
    height = 3
    row_header_width = calculate_row_header_width(grid)
    chars_per_time = calculate_needed_chars_per_time(grid)
    total_target_width = round((end - start) * chars_per_time)

    for lane in grid.lanes:
        row_tile = lane[0].render(width=row_header_width, height=height)
        current_time = start
        final_block_index = len(lane) - 2
        for i, block in enumerate(lane[1:]):
            col_start = round((current_time - start) * chars_per_time)
            duration = block.time_length  # pyright: ignore
            if i == final_block_index:
                col_end = total_target_width
            else:
                col_end = round((current_time + duration - start) * chars_per_time)
            calculated_width = max(1, col_end - col_start)
            tile = block.render(width=calculated_width, height=height)
            row_tile += tile
            current_time += duration
        s += str(row_tile) + "\n"

    axis = text_axis([start, end], total_target_width)
    axis_padding = Tile([" " * (row_header_width - 1)] * 2)
    s += str(axis_padding + axis)
    return s


def render_trial_design(td: TrialDesign) -> str:
    timeline = build_timeline(td)
    timeline = timeline.pack_activities()
    grid = timeline_to_block_grid(timeline)
    s = print_grid(grid, start=timeline.start, end=timeline.end)
    return s
