import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO, Union

from pharmpy.deps import pandas as pd
from pharmpy.model import DatasetError

_ignore_alpha = re.compile(r"[ \t]*[A-Za-z#@]")
_ignore_heading = re.compile(r"TABLE[ \t]NO[.][ \t]")

SEP_INPUT = re.compile(r"[,\t] *|  *(?:(?=[^,\t ])|, *)")

PAD = " "


def _ignore(ignore_character: str) -> re.Pattern[str]:
    if ignore_character == "@":
        return _ignore_alpha
    else:
        return re.compile(f"[{re.escape(ignore_character)}]")


def NMTRANStreamIterator(stream: TextIO, sep: re.Pattern[str], ignore: re.Pattern[str]):
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
            yield sep.split(_line)
        elif '\t' in line:
            # NOTE: ~6x speedup for large TSVish files.
            yield _line.replace('\t', ',').split(',')
        else:
            # NOTE: ~6x speedup for large CSVish files.
            yield _line.split(',')

        try:
            line = next(stream)
            while ignore.match(line):
                line = next(stream)
        except StopIteration:
            return


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
            yield NMTRANStreamIterator(stream, sep, ignore)

    else:
        yield NMTRANStreamIterator(path_or_io, sep, ignore)


def read_NMTRAN_data(io: Iterable[list[str]], **kwargs: Any) -> pd.DataFrame:
    assert 'header' in kwargs
    assert kwargs['header'] is None
    assert len(kwargs) == 1

    return pd.DataFrame(io)
