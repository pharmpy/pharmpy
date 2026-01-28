import sys
from itertools import islice, tee
from typing import Generic, Iterable, Iterator, TypeVar

T = TypeVar('T')

_version = sys.version_info[:3]
_minor = _version[:2]

if _minor == (3, 12) and _version >= (3, 12, 8):
    # SEE: https://github.com/python/cpython/commit/cf2532b39d099e004d1c07b2d0fcc46567b68e75
    _tee = tee

elif _minor == (3, 13) and _version >= (3, 13, 1):
    # SEE: https://github.com/python/cpython/commit/7bc99dd49ed4cebe4795cc7914c4231209b2aa4b
    _tee = tee

elif _minor >= (3, 14):
    # SEE: https://github.com/python/cpython/pull/124490
    _tee = tee

else:
    # SEE: https://github.com/python/cpython/issues/137597#issuecomment-3186240062
    def _tee(iterable: Iterable[T], n: int = 2, /):
        if hasattr(iterable, "__copy__"):
            return tee(iterable, n + 1)[1:]
        else:
            return tee(iterable, n)


def _fork(iterable: Iterable[T]):
    return _tee(iterable, 1)[0]


class PeekableIterator(Generic[T]):
    def __init__(self, iterator: Iterator[T]):
        # NOTE: Adapted from https://docs.python.org/3/library/itertools.html#itertools.tee
        # NOTE: Extra copy required for Python 3.11
        self._it = _fork(iterator)

    def __next__(self):
        return next(self._it)

    def __iter__(self):
        return self._it  # NOTE: This bypasses `__next__` in `for ... in ...` constructs.

    def lookahead(self, n: int):
        # NOTE: Extra copy required for Python 3.11
        # SEE: https://github.com/python/cpython/issues/137597#issuecomment-3186240062
        return islice(_fork(self._it), n)

    def peek(self):
        return next(self.lookahead(1))
