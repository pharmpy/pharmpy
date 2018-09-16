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

    execute = None
    """:class:`~pharmpy.execute.Engine` API.
    Evaluation, estimation & simulation tasks."""

    input = None
    """:class:`~pharmpy.input.ModelInput` API.
    E.g. data."""

    output = None
    """:class:`~pharmpy.output.ModelOutput` API.
    Results of evaluations, estimations & simulations."""

    parameters = None
    """:class:`~pharmpy.parameters.ParameterModel` API.
    E.g. parameter estimates & initial values."""

    _path = None
    _index = 0

    def __init__(self, path):
        self.path = path
        self.source = SourceResource(self)
        if self.exists:
            self.read()

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
    def exists(self):
        """Resolves path and returns True if model exists on disk."""
        if self.path and self.path.is_file():
            self.path = self.path.resolve()
            return True

    @property
    def content(self):
        """Raw content stream of model."""
        if not self.exists:
            return None
        with open(str(self.path), 'r') as f:
            content = f.read()
        return content

    def validate(self):
        """Test if model is syntactically valid (raises if not)."""
        raise NotImplementedError

    def read(self):
        """Read model from disk.

        Initiates all the API:s of the Model, e.g. :class:`~pharmpy.input.ModelInput`,
        :class:`~pharmpy.input.ModelOutput` and :class:`~pharmpy.parameters.ParameterModel`.
        """
        self.input = ModelInput(self)
        self.output = ModelOutput(self)
        self.parameters = ParameterModel(self)
        self.execute = Engine(self)
        self.validate()

    def write(self, path=None, exist_ok=True):
        """Writes model to disk.

        Will also update model to link that file.

        Arguments:
            path: A `path-like object`_ to write.
            exist_ok: If False, :exc:`FileExistsError` is raised if the file already exists.

        If no *path* given (default), model :attr:`path` attribute will be used. If not changed and
        *exist_ok* (default), the model will be overwritten.

        .. todo::
            Implement true model write (just copies read buffer now). Will require thoughts on how
            to "bootstrap" up a rendering of the low-level objects (e.g. every ThetaRecord, etc.).

        .. _path-like object: https://docs.python.org/3/glossary.html#term-path-like-object
        """
        path = path or self.path
        if not path:
            raise ValueError("No filesystem path set (can't write model)")
        path = Path(path)
        if path.exists and not exist_ok:
            raise FileExistsError("Expected creating new file but path exists: %r" % str(path))
        with open(str(path), 'w') as f:
            f.write(self.content)
        self.path = path.resolve()

    @property
    def path(self):
        """File path of the model."""
        return self._path

    @path.setter
    def path(self, path):
        if path:
            self._path = Path(path)
        else:
            self._path = None

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

    def copy(self, path=None, write=False):
        """Returns a copy of this model.

        Arguments:
            path: If None, only deepcopy :class:`Model` object. Otherwise, set filesystem path on
                new model and write to disk.
        """
        model = deepcopy(self)
        if path:
            model.path = path
            if write:
                model.write()
        return model

    def __repr__(self):
        path = None if self.path is None else str(self.path)
        return "%s(%r)" % (self.__class__.__name__, path)

    def __str__(self):
        if self.exists:
            return self.content

    def __deepcopy__(self, memo):
        """Copy model completely.

        Utilized by e.g. :class:`pharmpy.execute.run_directory.RunDirectory` to "take" the model in
        a dissociated state from the original.

        .. note::
            Lazy solution with re-parsing path for now. Can't deepcopy down without implementing
            close to Lark tree's, since compiled regexes must be re-compiled.

            .. todo:: Deepcopy Model objects "correctly"."""
        if self.exists:
            return type(self)(self.path)
        elif self.content is not None:
            raise NotImplementedError("Tried to (deeply) copy %r without path but content; "
                                      "Not yet supported" % (self,))
