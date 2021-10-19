"""Factory for creating a pharmpy model from a "native" object representation.

Definitions
===========
"""

import io
import pathlib
from pathlib import Path

from pharmpy.plugins.utils import detect_model


def Model(obj, **kwargs):
    """Factory for creating a :class:`pharmpy.model` object from an object representing the model
    (i.e. path).

    .. _path-like object: https://docs.python.org/3/glossary.html#term-path-like-object

    Parameters
    ----------
    obj
        Currently a `path-like object`_ pointing to the model file.

    Returns
    -------
    - Generic :class:`~pharmpy.generic.Model` if path is None, otherwise appropriate implementation
      is invoked (e.g. NONMEM7 :class:`~pharmpy.api_nonmem.model.Model`).
    """
    if isinstance(obj, str):
        path = Path(obj)
    elif isinstance(obj, pathlib.Path):
        path = obj
    elif isinstance(obj, io.IOBase):
        path = None
    else:
        raise ValueError("Unknown input type to Model constructor")
    if path is not None:
        with open(path, 'r', encoding='latin-1') as fp:
            code = fp.read()
    else:
        code = obj.read()
    model_class = detect_model(code)
    model = model_class(code, path, **kwargs)
    # Setup model database here
    # Read in model results here?
    # Set filename extension?
    return model
