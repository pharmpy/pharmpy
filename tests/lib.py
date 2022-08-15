from difflib import unified_diff
from itertools import islice
from typing import Any, List


def _diff_string_lists(a: List[str], b: List[str]):
    return islice(
        unified_diff(a, b, n=0, lineterm=''),
        2,
        None,
    )


def _diff_texts(a: str, b: str):
    return _diff_string_lists(a.splitlines(), b.splitlines())


def diff(a: Any, b: Any):
    lines = _diff_texts(str(a), str(b))
    joined = '\n'.join(lines)
    return '' if joined == '' else joined + '\n'
