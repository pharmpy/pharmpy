# -*- encoding: utf-8 -*-
"""
===================
Generic Model class
===================

**Parent to all implementations.**

Inherit to *implement*, i.e. to define support for a specific model type. Duck typing is utilized,
but an implementation is expected to implement **all** methods/attributes.

Definitions
-----------
"""

from copy import deepcopy
from pathlib import Path

from pharmpy.execute import Engine
from pharmpy.input import ModelInput
from pharmpy.output import ModelEstimation  # noqa (only to bring into generic namespace)
from pharmpy.output import ModelOutput
from pharmpy.parameters import ParameterModel
from pharmpy.source_io import SourceResource


def detect(lines):
    return False


class ModelException(Exception):
    pass


class ModelParsingError(ModelException):
    def __init__(self, msg='model parse error'):
        super().__init__(msg)


class ModelLookupError(LookupError):
    def __init__(self, lookup):
        self.lookup = lookup
        try:
            reason = 'index %d out of range' % (self.lookup,)
        except TypeError:
            reason = 'name %s not found' % (repr(str(self.lookup)),)
        msg = 'submodel does not exist (%s)' % (reason,)
        super().__init__(msg)


class Model(object):
    """(Generic) Model class.

    Represents a model file object, that may or may not exist on disk too.

    Attributes:

    .. note:: Attribute :attr:`path` always :class:`~pathlib.Path` object, but only resolved (set to
        absolute) by :attr:`exists`, which should be checked before any IO (read/write) on disk.
        Thus, :attr:`path` needn't exist until needed!
    """

    SourceResource = SourceResource

    Engine = Engine
    """:class:`~pharmpy.execute.Engine` API.
    Evaluation, estimation & simulation tasks."""

    ModelInput = ModelInput
    """:class:`~pharmpy.input.ModelInput` API.
    E.g. data."""

    ModelOutput = ModelOutput
    """:class:`~pharmpy.output.ModelOutput` API.
    Results of evaluations, estimations & simulations."""

    ParameterModel = ParameterModel
    """:class:`~pharmpy.parameters.ParameterModel` API.
    E.g. parameter estimates & initial values."""

    _path = None
    _index = 0

    def __init__(self, path):
        self.source = self.SourceResource(path)
        self.input = self.ModelInput(self)
        self.output = self.ModelOutput(self)
        self.parameters = self.ParameterModel(self)
        self.execute = self.Engine(self)
        self.source.model = self

    @property
    def index(self):
        """Current model (subproblem) index.

        The context for everything else changes if changed. Implementation might accept name lookup.
        """
        return self._index

    @index.setter
    def index(self, new):
        if new != 0:
            raise ModelLookupError(new)
        self._index = new

    @property
    def content(self):
        """Raw content stream of model."""
        if self.source.on_disk:
            return self.source.input.getvalue()
        else:
            return None

    def validate(self):
        """Test if model is syntactically valid (raises if not)."""
        raise NotImplementedError

    def write(self, path=None, exist_ok=True):
        """Writes model to disk.

        Will also update model to link that file.

        Arguments:
            path: A `path-like object`_ to write.
            exist_ok: If False, :exc:`FileExistsError` is raised if the file already exists.

        If no *path* given (default), model :attr:`path` attribute will be used. If not changed and
        *exist_ok* (default), the model will be overwritten.

        .. _path-like object: https://docs.python.org/3/glossary.html#term-path-like-object
        """
        path = path or self.path
        if not path:
            raise ValueError("No filesystem path set (can't write model)")
        path = Path(path)
        if path.exists and not exist_ok:
            raise FileExistsError("Expected creating new file but path exists: %r" % str(path))
        with open(str(path), 'w') as f:
            f.write(str(self.source.output))
        self.path = path.resolve()

    @property
    def path(self):
        """Source filesystem path (if any)."""
        return self.source.path

    @path.setter
    def path(self, path):
        self.source.path = path

    @property
    def has_results(self):
        """True *if and only if* model has results.

        Must be True for accessing :class:`~pharmpy.output.ModelOutput`.

        .. todo::
            Implement model execution/results status checker.
            **Should** contain a call to :class:`.engine` class. An implementation of *that* should
            then know how to check on current platform/cluster system (also *without* initializing a
            run directory).
            **Shouldn't** need to override this (by implementation).
        """
        return True

    def copy(self, dest=None, write=None):
        """Returns a copy of this model.

        Arguments:
            dest: New filesystem path. If None, new :class:`Model` object retains previous
                :attr:`~pharmpy.generic.Model.path`.
            write: Write copy to disk (*dest* or previous :attr:`~pharmpy.generic.Model.path`).

        By default, *write* is True if *dest* given and False otherwise.
        """
        model = deepcopy(self)
        if dest:
            model.path = dest
            write = True if (write is None) else write
        else:
            write = False if (write is None) else write
        if write:
            model.write()
        return model

    def __repr__(self):
        path = None if self.path is None else str(self.path)
        return "%s(%r)" % (self.__class__.__name__, path)

    def __str__(self):
        return str(self.source.output)

    def __deepcopy__(self, memo):
        """Copy model completely.

        Utilized by e.g. :class:`pharmpy.execute.run_directory.RunDirectory` to "take" the model in
        a dissociated state from the original.

        .. note::
            Lazy solution with re-parsing path for now. Can't deepcopy down without implementing
            close to Lark tree's, since compiled regexes must be re-compiled.

            .. todo:: Deepcopy Model objects "correctly"."""
        if self.source.on_disk:
            return type(self)(self.path)
        elif self.content is not None:
            raise NotImplementedError("Tried to (deeply) copy %r content without source; "
                                      "Not yet supported" % (self,))
