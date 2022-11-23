from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Type, TypeVar

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
    def buffer(self) -> str:
        # NOTE The sequence of characters corresponding to this record
        ...

    @property
    @abstractmethod
    def tree(self) -> AttrTree:
        # NOTE The parse tree corresponding to this record
        ...

    def __post_init__(self):
        self.tree  # FIXME remove this, this is only for testing
        self.buffer  # FIXME remove this, this is only for testing

    def __str__(self):
        return self.raw_name + self.buffer


@dataclass(frozen=True)
class ReplaceableRecord(Record):
    @property
    @classmethod
    @abstractmethod
    def generated(cls: Type[R]) -> Type[GeneratedRecord]:
        # NOTE The type that should be used to build a generated record
        ...

    @property
    @classmethod
    @abstractmethod
    def parsed(cls: Type[R]) -> Type[ParsedRecord]:
        # NOTE The type that should be used to build a parsed record
        ...


R = TypeVar('R', bound=ReplaceableRecord)


def replace_tree(record: R, tree: AttrTree) -> R:
    cls = record.__class__.generated
    return cls(name=record.name, raw_name=record.raw_name, tree=tree)


@dataclass(frozen=True)
class ParsedRecord(Record):
    parser: RecordParser
    buffer: str

    @cached_property
    def tree(self) -> AttrTree:
        return self.parser.parse(self.buffer)


@dataclass(frozen=True)
class GeneratedRecord(Record):
    tree: AttrTree

    @cached_property
    def buffer(self) -> str:
        return str(self.tree)


def with_parsed_and_generated(cls):
    for name in ('parsed', 'generated'):
        if getattr(cls, name, None) is not None:
            raise ValueError(
                'Cannot make a record class out of this becaused property {name} already exists.'
            )

    p = dataclass(frozen=True)(type(f'Parsed{cls.__qualname__}', (cls, ParsedRecord), {}))
    g = dataclass(frozen=True)(type(f'Generated{cls.__qualname__}', (cls, GeneratedRecord), {}))

    setattr(cls, 'parsed', p)
    setattr(cls, 'generated', g)

    return cls
