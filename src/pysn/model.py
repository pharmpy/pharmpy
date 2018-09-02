# -*- encoding: utf-8 -*-

from pathlib import Path

from .api_utils import detectAPI
from .api_utils import getAPI


def Model(path=None, **kwargs):
    """
    Creates Model object from path.

    Generic API if path is None, otherwise appropriate API will be used.
    """
    _Model = getAPI('generic')
    if path:
        path = Path(path).resolve()
        _Model = detectAPI(path)
    obj = _Model(path, **kwargs)
    return obj
