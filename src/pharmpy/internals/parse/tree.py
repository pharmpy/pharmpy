from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Generic, List, Tuple, TypeVar, Union

from pharmpy.internals.immutable import Immutable

T = TypeVar('T', bound='Tree')
L = TypeVar('L', bound='Leaf')


@dataclass(frozen=True)
class Leaf(Immutable):
    rule: str
    value: str


@dataclass(frozen=True)
class Tree(Generic[T, L], Immutable):
    rule: str
    children: Tuple[Union[T, L], ...]


R = TypeVar('R')


class Interpreter(Generic[T, L, R], ABC):
    def visit(self, tree: Tree[T, L]) -> R:
        return self.visit_tree(tree)

    def visit_tree(self, tree: Tree[T, L]):
        f = getattr(self, tree.rule)
        return f(tree)

    def visit_subtrees(self, tree: Tree[T, L]) -> List:
        """Does not visit and discards tokens"""
        return [self.visit_tree(child) for child in tree.children if isinstance(child, Tree)]

    def __getattr__(self, name):
        return self.__default__

    def __default__(self, tree):
        return self.visit_subtrees(tree)
