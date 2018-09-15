# -*- encoding: utf-8 -*-
"""Contains the exported API definitions.

Definitions
===========
"""

from pathlib import Path

from .api_utils import detectAPI
from .api_utils import getAPI


def Model(path=None, **kwargs):
    """Factory for creating a Model object from a path.

    Arguments:
        path: A `path-like object`_ pointing to the model file.

    Generic :class:`~pharmpy.generic.Model` if path is None, otherwise appropriate implementation is
    invoked (e.g. NONMEM7 :class:`~pharmpy.api_nonmem.model.Model`).

    .. _path-like object: https://docs.python.org/3/glossary.html#term-path-like-object
    """
    _Model = getAPI('generic')
    if path:
        path = Path(path).resolve()
        _Model = detectAPI(path)
    obj = _Model(path, **kwargs)
    return obj
