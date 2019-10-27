# -*- encoding: utf-8 -*-
"""Factory for creating a pharmpy model from a "native" object representation.

Definitions
===========
"""
from pharmpy.plugins.utils import detect_model


def Model(obj, from_string=False, **kwargs):
    """Factory for creating a Model object from an object representing the model (i.e. path).

    Arguments:
        obj: Currently a `path-like object`_ pointing to the model file.
        from_string: True if the obj contains the model code

    Generic :class:`~pharmpy.generic.Model` if path is None, otherwise appropriate implementation is
    invoked (e.g. NONMEM7 :class:`~pharmpy.api_nonmem.model.Model`).

    .. _path-like object: https://docs.python.org/3/glossary.html#term-path-like-object
    """
    #_Model = getAPI('generic').Model       # Don't support generic model for now
    if obj:
        model_class = detect_model(obj, from_string=from_string)
        model = model_class(obj, from_string=from_string, **kwargs)
        return model
