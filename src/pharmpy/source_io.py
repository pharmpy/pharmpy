# -*- encoding: utf-8 -*-
"""
=============================
Generic Sources/Resources I/O
=============================

Source code/resource manager for a :class:`~pharmpy.generic.Model` class implementation.

Cached input
------------

The input is cached as much as possible, with automatic invalidation. Yes that's right. *Cached*.
Why? Because :attr:`SourceResource.input` must be trusted to be **the exact input** that was/is used
for all the other API's. If not, it's *useless*.

The desire for some great input immutability is what motivated this module.

Noteworthy
----------

Module is target for *all* generalizable and non-agnostic code that concerns:

    1. Reading *in* streams/file-like resources.
    2. Formatting/generating source code from current state of :class:`~pharmpy.generic.Model`.
    3. Writing *out* source code to streams/file-like resources.
    4. Detecting changes and managing concurrency/version control of such I/O resources.

.. todo:: Graft over :attr:`generic.Model.path` and related utils.
.. todo:: Implement input source cache (read `Cached input`_ for why).

Definitions
-----------
"""

from pathlib import Path
from io import StringIO


class SourceIO(StringIO):
    """An IO class for reading/writing :class:`~pharmpy.generic.Model` source code.

    Arguments:
        file: Path of the file containing the initializing source.
        source: The source code if no *file*.
    """

    def __init__(self, file=None, source=''):
        if file:
            with open(str(file), 'r') as source_file:
                source = source_file.read()
        super().__init__(source)


class SourceResource:
    """A manager of (source code) resource input/output of a :class:`~pharmpy.generic.Model`.

    Is a model API attached to attribute :attr:`Model.source <pharmpy.generic.Model.source>`.

    .. note:: Implementation should only need to override :func:`generate_source`
    """

    model = None
    """:class:`~pharmpy.generic.Model` owner of API."""

    SourceIO = SourceIO
    """:class:`~pharmpy.source_io.SourceIO` implementation."""

    def __init__(self, path):
        self.path = path

    @property
    def input(self):
        """Input source code."""
        if self.on_disk:
            if 'input' not in self._cache:
                self._cache['input'] = self.SourceIO(file=self.path)
            return self._cache['input']
        else:
            return self.SourceIO(source='')

    @property
    def output(self):
        """Output source code."""
        return self.SourceIO(source=self.source_generator(self.model))

    @property
    def path(self):
        """Source on disk as a (resolved) `path-like object`_ (or None).

        .. _path-like object: https://docs.python.org/3/glossary.html#term-path-like-object"""
        if self._path:
            try:
                return self._path.resolve()
            except FileNotFoundError:
                pass
        return self._path

    @path.setter
    def path(self, path):
        self._cache = dict()
        if path:
            self._path = Path(path)
        else:
            self._path = None

    @property
    def on_disk(self):
        """Returns True if readable."""
        return self.path and self.path.exists() and self.path.is_file()

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.path)

    @classmethod
    def source_generator(cls, model):
        """Generator of source code.

        Generated source must be:

            1. Current, i.e. :class:`~pharmpy.generic.Model` state-dependent (limit side-effects).
            2. *Fully* agnostic of existing files and other resources. Again, no side-effects!
        """
        return str(model)
