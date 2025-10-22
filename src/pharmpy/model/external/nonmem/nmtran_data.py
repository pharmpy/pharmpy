import re
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, TextIO, Union

from pharmpy.deps import pandas as pd
from pharmpy.model.external.nonmem.dataset import DatasetError

_ignore_alpha = re.compile(r"[ \t]*[A-Za-z#@]")
_ignore_heading = re.compile(r"TABLE[ \t]NO[.][ \t]")

SEP_INPUT = re.compile(r"[,\t] *|  *(?:(?=[^,\t ])|, *)")

PAD = " "
SEP = ","


def _ignore(ignore_character: str) -> re.Pattern[str]:
    if ignore_character == "@":
        return _ignore_alpha
    else:
        return re.compile(f"[{re.escape(ignore_character)}]")


def NMTRANStreamIterator(stream: TextIO, sep: re.Pattern[str], ignore: re.Pattern[str]):
    assert SEP == ','

    for line in stream:
        if ignore.match(line) or _ignore_heading.match(line):
            continue
        break
    else:
        return

    while True:
        if " \t" in line:
            raise DatasetError(
                "The dataset contains a TAB preceeded by a space, "
                "which is not allowed by NM-TRAN"
            )

        _line = line.lstrip(PAD).rstrip()
        if not _line:
            raise DatasetError(
                "The dataset contains one or more blank lines. This is not "
                "allowed by NM-TRAN without the BLANKOK option"
            )

        if ' ' in _line:
            yield SEP.join(sep.split(_line)) + '\n'
        elif '\t' in line:
            # NOTE: ~6x speedup for large TSVish files.
            yield _line.replace('\t', SEP) + '\n'
        else:
            # NOTE: ~6x speedup for large CSVish files.
            yield _line + '\n'

        try:
            line = next(stream)
            while ignore.match(line):
                line = next(stream)
        except StopIteration:
            return


class NMTRANReader:
    def __init__(self, stream: TextIO, sep: re.Pattern[str], ignore: re.Pattern[str]):
        self._lines: Iterator[bytes] = map(str.encode, NMTRANStreamIterator(stream, sep, ignore))
        self._buffer: deque[bytes] = deque()
        self._n: int = 0

    def read(self, n: int = -1) -> bytes:
        assert n >= 1

        lines = self._lines
        buffer = self._buffer

        while self._n < n:
            try:
                chunk = next(lines)
            except StopIteration:
                break

            buffer.append(chunk)
            self._n += len(chunk)

        if self._n == 0:
            return b""

        last = buffer.pop()
        remainder = max(0, self._n - n)
        value = b"".join(buffer) + last[: len(last) - remainder]
        assert len(value) == n or remainder == 0
        buffer.clear()
        if remainder != 0:
            buffer.append(last[len(last) - remainder :])
        self._n = remainder
        return value


def open_NMTRAN(path: Union[str, Path]):
    return open(str(path), "r", encoding="latin-1")


@contextmanager
def NMTRANDataIO(
    path_or_io: Union[str, Path, TextIO],
    sep: re.Pattern[str],
    ignore_character: Optional[str] = None,
):
    if ignore_character is None:
        ignore_character = "#"

    ignore = _ignore(ignore_character)

    if isinstance(path_or_io, (str, Path)):
        with open_NMTRAN(path_or_io) as stream:
            yield NMTRANReader(stream, sep, ignore)

    else:
        yield NMTRANReader(path_or_io, sep, ignore)


def read_NMTRAN_data(io: NMTRANReader, **kwargs: Any) -> pd.DataFrame:
    return pd.read_table(  # type: ignore
        io,  # type: ignore
        **kwargs,
        sep=SEP,
        na_filter=False,
        engine="c",
        quoting=3,
        dtype=object,
        index_col=False,
    )
