from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

from pharmpy.internals.immutable import Immutable
from pharmpy.internals.parse import AttrTree

from .parsers import RecordParser


@dataclass(frozen=True)
class Record(Immutable):
    """
    Top level class for records.

    Create objects only by using the factory function create_record.
    """

    name: str
    raw_name: str

    @property
    @abstractmethod
    def root(self) -> AttrTree:
        ...

    def __post_init__(self):
        self.root  # FIXME remove this, this is only for testing

    def __str__(self):
        return self.raw_name + str(self.root)


@dataclass(frozen=True)
class ParsedRecord(Record):
    parser: RecordParser
    content: str

    @cached_property
    def root(self) -> AttrTree:
        return self.parser.parse(self.content)

    def __str__(self):
        # NOTE This is faster than recreating the string from the tree
        return self.raw_name + self.content


@dataclass(frozen=True)
class GeneratedRecord(Record):
    root: AttrTree
