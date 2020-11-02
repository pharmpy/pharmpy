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
    """Factory to create source object"""
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

    def write(self, path, force=False):
        """Write source to file."""
        if not force and path.exists():
            raise FileExistsError(f'Cannot overwrite model at {path} with "force" not set')
        with open(path, 'w', encoding='latin-1') as fp:
            fp.write(self.code)
        self.path = path


class FileSource(SourceBase):
    """Source formats for files
    property: filename_extension    (includes the dot)
    """

    def __init__(self, obj):
        if isinstance(obj, str) or isinstance(obj, Path):
            path = Path(obj)
            self.filename_extension = path.suffix
        else:
            self.filename_extension = ''
        super().__init__(obj)

    def read(self, path_or_io):
        """Read source from io object or from str path or path object"""
        if isinstance(path_or_io, str) or isinstance(path_or_io, Path):
            self.path = Path(path_or_io)
            with open(path_or_io, 'r', encoding='latin-1') as fp:
                return fp.read()
        else:
            self.path = Path('.')
            return path_or_io.read()
