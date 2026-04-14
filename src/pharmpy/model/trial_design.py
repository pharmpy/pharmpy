from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from io import StringIO
from typing import Any, Optional, Union, overload

from pharmpy.deps import pandas as pd
from pharmpy.deps.rich import box
from pharmpy.deps.rich import columns as rich_columns
from pharmpy.deps.rich import console as rich_console
from pharmpy.deps.rich import panel as rich_panel
from pharmpy.internals.immutable import Immutable
from pharmpy.internals.math import round_and_keep_sum
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


class Arm(Immutable):
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

    def is_placebo(self) -> bool:
        """Is this arm a placebo arm?"""
        for act in self._activities:
            if isinstance(act, Administration) and act.dose.amount != 0:
                return False
        return True


class TrialDesign(Immutable):
    """TrialDesign"""

    def __init__(self, arms: tuple[Arm, ...]):
        self._arms = arms

    @classmethod
    def create(cls, arms: Sequence[Arm]) -> TrialDesign:
        for arm in arms:
            if not isinstance(arm, Arm):
                raise TypeError("Arms in TrialDesign must be of type Arm")
        return cls(tuple(arms))

    def replace(
        self,
        arms: Sequence[Arm],
    ) -> TrialDesign:
        return TrialDesign.create(arms=arms)

    @property
    def arms(self) -> tuple[Arm, ...]:
        """The arms"""
        return self._arms

    def to_dict(self) -> dict[str, Any]:
        arms = tuple(arm.to_dict() for arm in self)
        return {
            'arms': arms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrialDesign:
        arms = []
        for sdict in d['arms']:
            arms.append(Arm.from_dict(sdict))
        return cls.create(arms=arms)

    def __len__(self):
        return len(self._arms)

    @overload
    def __getitem__(self, ind: int) -> Arm: ...

    @overload
    def __getitem__(self, ind: slice) -> TrialDesign: ...

    def __getitem__(self, ind: Union[int, slice]) -> Union[Arm, TrialDesign]:
        if isinstance(ind, slice):
            return TrialDesign(arms=self._arms[ind])
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
        return self._arms == other._arms

    def __hash__(self):
        return hash(self._arms)

    def __repr__(self):
        return render_trial_design(self)


def get_time_points(activity):
    # Make into method?
    adjusted_time_points = [activity.start_time + time for time in activity.time_points]
    return adjusted_time_points


@dataclass
class Frame:
    start_time: float
    end_time: float
    activity: Optional[Activity]
    chars_per_scale: float = -float("inf")
    width: int = 0


def create_frames(arm):
    frames = []
    for act in arm:
        time_points = get_time_points(act)
        start_time = time_points[0]
        end_time = time_points[-1]
        frame = Frame(start_time, end_time, act)
        frames.append(frame)
    return frames


def sort_activity_frames(frames):
    # acts is a list of Frames
    return sorted(frames, key=lambda frame: frame.start_time)


def split_into_lanes(frames):
    # Splits into lanes if necessary and adds gaps between activities
    lanes = []
    end_times = []
    for frame in frames:
        for i, lane in enumerate(lanes):
            if frame.start_time > end_times[i]:
                gap = Frame(end_times[i], frame.start_time, None)
                lanes[i].append(gap)
            if frame.start_time >= end_times[i]:
                lanes[i].append(frame)
                end_times[i] = frame.end_time
                break
        else:
            # Add lane
            lanes.append([frame])
            end_times.append(frame.end_time)
    return lanes


def get_global_start_end_times(arm_lanes):
    max_end_time = 0
    min_start_time = 1e18
    for arm in arm_lanes:
        for lane in arm:
            for frame in lane:
                if frame.end_time > max_end_time:
                    max_end_time = frame.end_time
                if frame.start_time < min_start_time:
                    min_start_time = frame.start_time
    return min_start_time, max_end_time


def add_start_and_end_gaps(arm_lanes, min_start_time, max_end_time):
    for arm in arm_lanes:
        for lane in arm:
            last_frame = lane[-1]
            if last_frame.end_time < max_end_time:
                gap = Frame(last_frame.end_time, max_end_time, None)
                lane.append(gap)
            first_frame = lane[0]
            if first_frame.start_time > min_start_time:
                gap = Frame(min_start_time, first_frame.start_time, None)
                lane.insert(0, gap)


def preliminary_rendering(lane):
    for frame in lane:
        act = frame.activity
        if isinstance(act, Observations):
            panel = observations_panel(act)
        elif isinstance(act, Administration):
            panel = administration_panel(act)
        else:
            panel = None
        if panel is not None:
            tmp = rich_console.Console(file=StringIO(), record=True, width=1000)
            tmp.print(panel)
            rendered = tmp.export_text()
            actual_width = max(len(line) for line in rendered.splitlines())
            frame.chars_per_scale = actual_width / (frame.end_time - frame.start_time)


def calculate_widths(arm_lanes, chars_per_scale, total_width):
    for arm in arm_lanes:
        for lane in arm:
            widths = []
            for frame in lane:
                width = (frame.end_time - frame.start_time) * chars_per_scale
                widths.append(width)
            widths = list(round_and_keep_sum(pd.Series(widths), total_width))
            for frame, width in zip(lane, widths):
                frame.width = width


def render_lanes(arm_lanes, padding=0):
    s = ""
    for n_arm, arm in enumerate(arm_lanes, start=1):
        for n_lane, lane in enumerate(arm):
            if n_lane == 0:
                arm_panel = rich_panel.Panel(f"Arm {n_arm}", box=box.SIMPLE, width=9)
            else:
                arm_panel = rich_panel.Panel("", box=box.SIMPLE, width=9)
            columns = [arm_panel]
            if padding > 0:
                pad_panel = rich_panel.Panel("", box=box.SIMPLE, width=padding)
                columns.append(pad_panel)
            for frame in lane:
                act = frame.activity
                if act is None:
                    panel = rich_panel.Panel("", box=box.SIMPLE, width=frame.width)
                elif isinstance(act, Observations):
                    panel = observations_panel(act, width=frame.width)
                else:  # isinstance(act, Administration):
                    panel = administration_panel(act, width=frame.width)
                columns.append(panel)
            cols = rich_columns.Columns(columns, padding=0)
            console = rich_console.Console()
            with console.capture() as capture:
                console.print(cols)
            s += capture.get()
    return s


def render_trial_design(td):
    arm_lanes = []
    for arm in td:
        frames = create_frames(arm)
        frames = sort_activity_frames(frames)
        lanes = split_into_lanes(frames)
        for lane in lanes:
            preliminary_rendering(lane)
        arm_lanes.append(lanes)

    min_start_time, max_end_time = get_global_start_end_times(arm_lanes)
    add_start_and_end_gaps(arm_lanes, min_start_time, max_end_time)

    max_chars_per_scale = max(
        [frame.chars_per_scale for arm in arm_lanes for lane in arm for frame in lane]
    )
    total_width = math.ceil((max_end_time - min_start_time) * max_chars_per_scale)
    calculate_widths(arm_lanes, max_chars_per_scale, total_width)

    axis = text_axis([min_start_time, max_end_time], total_width)
    axis_padding = len(axis) - len(axis.lstrip(" "))
    s = render_lanes(arm_lanes, axis_padding)

    axis = text_axis([min_start_time, max_end_time], total_width)
    for line in axis.split("\n"):
        s += " " * 9 + line + "\n"
    return s


def observations_panel(obs, width=None):
    if width is None:
        expand = False
    else:
        expand = True
    panel = rich_panel.Panel(
        list_with_unit(get_time_points(obs), obs.variable.properties.get('unit', None)),
        title="[cyan]Observations",
        subtitle=f"[dim]{obs.variable.name}",
        border_style="green",
        expand=expand,
        width=width,
    )
    return panel


def administration_panel(admin, width=None):
    unit = admin.variable.properties.get('unit', None)
    unit_str = "" if unit is None else " " + str(unit)
    panel = rich_panel.Panel(
        list_with_unit(admin.time_points, unit),
        title="[cyan]Administration",
        subtitle=f"[dim]{admin.dose.amount}{unit_str} {admin.dose.__class__.__name__}",
        border_style="green",
        expand=False,
        width=width,
    )
    return panel


def list_with_unit(x, unit=None):
    s = ", ".join(map(str, x))
    if unit is not None:
        s += f" {unit}"
    return s


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
    return padding + ticked_bars + '\n' + points_line
