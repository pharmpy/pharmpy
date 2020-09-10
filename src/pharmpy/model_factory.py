"""Factory for creating a pharmpy model from a "native" object representation.

Definitions
===========
"""
import pharmpy.source as source
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
    src = source.Source(obj)
    model_class = detect_model(src)
    model = model_class(src, **kwargs)
    return model
