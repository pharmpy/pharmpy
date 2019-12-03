"""
=============================
Generic Sources/Resources I/O
=============================

Source code/resource manager for a :class:`~pharmpy.generic.Model` class implementation.

Noteworthy
----------

Module is target for *all* generalizable and non-agnostic code that concerns:

    1. Reading *in* streams/file-like resources.
    2. Formatting/generating source code from current state of :class:`~pharmpy.generic.Model`.
    3. Writing *out* source code to streams/file-like resources.
"""

import io
from pathlib import Path


def Source(obj):
    """Factory to create source object
    """
    if isinstance(obj, str) or isinstance(obj, Path) or isinstance(obj, io.IOBase):
        src = FileSource(obj)
        return src



class SourceBase:
    """API to manage the original source object of a model.

    Is a model API attached to attribute :attr:`Model.source <pharmpy.generic.Model.source>`.

    The source format can be a text file, but it could also be any kind of object, i.e. an R object.
    """

    def __init__(self, obj):
        self.obj = obj
        self.code = self.read(obj)

    def update(self):
        """Update the source string from the model

        This needs to be overridden to handle model specific updates (i.e. relative $DATA)
        """
        self.code = str(self.model)

    def write(self, path):
        """Write source to file. Do automatic update
        """
        self.update()
        with open(path, 'w') as fp:
            fp.write(self.code)



class FileSource(SourceBase):
    """Source formats for files
    """
    def read(self, path_or_io):
        """Read source from io object or from str path or path object
        """
        if isinstance(path_or_io, str) or isinstance(path_or_io, Path):
            with open(path_or_io, 'r') as fp:
                return fp.read()
        else:
            return path_or_io.read()
