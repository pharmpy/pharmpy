import sys
from itertools import islice, tee
from typing import Iterable, Iterator, TypeVar

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


def make_peekable(iterator: Iterator[T]):
    # NOTE: Adapted from https://docs.python.org/3/library/itertools.html#itertools.tee
    # NOTE: Extra copy required for Python 3.11
    (tee_iterator,) = _tee(iterator, 1)

    def lookahead(n: int) -> Iterator[T]:
        # NOTE: Extra copy required for Python 3.11
        # SEE: https://github.com/python/cpython/issues/137597#issuecomment-3186240062
        (forked_iterator,) = _tee(tee_iterator, 1)
        return islice(forked_iterator, n)

    return tee_iterator, lookahead
