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

from enum import Enum
from pathlib import Path

from pysn import output  # TODO: ModelEstimation uses 'import generic; generic.output.XXX'
from pysn.output import ModelOutput
from pysn.parameters import ParameterModel
from pysn.execute import Engine


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
        self.input: Instance of (model API) :class:`~pysn.input.ModelInput` (e.g. data).
        self.output: Instance of (model API) :class:`~pysn.output.ModelOutput` (results of
            evaluation, estimation or simulations).
        self.parameters: Instance of (model API) :class:`~pysn.parameters.ParameterModel` (e.g.
            parameter estimates or initial values).
        self.execute: Instance of (model API) :class:`~pysn.execute.Engine` (executing evaluation,
            estimation or simulation).
    """

    _path = None
    _index = 0

    def __init__(self, path, **kwargs):
        self._path = Path(path).resolve() if path else None
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
        """True *if and only if* model exists on disk."""
        if self.path and self.path.is_file():
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

        Initiates all the API:s of the Model, e.g. :class:`~pysn.input.ModelInput`,
        :class:`~pysn.input.ModelOutput` and :class:`~pysn.parameters.ParameterModel`.
        """
        self.input = ModelInput(self)
        self.output = ModelOutput(self)
        self.parameters = ParameterModel(self)
        self.execute = Engine(self)

    def write(self, path):
        """Write model to disk.

        .. todo:: Start implementing Model write. Will require thoughts on how to "bootstrap" up a
            rendering of the low-level objects (e.g. every ThetaRecord, etc.).
        """
        raise NotImplementedError

    @property
    def path(self):
        """File path of the model."""
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def has_results(self):
        """True *if and only if* model has results.

        Must be True for accessing :class:`~pysn.output.ModelOutput`.

        .. todo::
            Implement model execution/results status checker.
            **Should** contain a call to :class:`.engine` class. An implementation of *that* should
            then know how to check on current platform/cluster system (also *without* initializing a
            run directory).
            **Shouldn't** need to override this (by implementation).
        """
        return True

    def __str__(self):
        if self.exists:
            return self.content


class ModelInput(object):
    """Implements API for :attr:`Model.input`, the model dataset"""
    def __init__(self, model):
        self.model = model

    @property
    def path(self):
        """Gets the path of the dataset"""
        raise NotImplementedError

    @path.setter
    def path(self):
        """Sets the path of the dataset"""
        raise NotImplementedError

    @property
    def data_frame(self, p):
        """Gets the pandas DataFrame object representing the dataset"""
        raise NotImplementedError

    @property
    def filters(self):
        """Gets an InputFilters object representing
        all data filters of the model
        """
        raise NotImplementedError

    @filters.setter
    def filters(self, new):
        """Sets all data filters
        """
        raise NotImplementedError


# FIXME: Put the different apis and helpers in separate files
class InputFilterOperator(Enum):
    EQUAL = 1
    NOT_EQUAL = 2
    LESS_THAN = 3
    LESS_THAN_OR_EQUAL = 4
    GREATER_THAN = 5
    GREATER_THAN_OR_EQUAL = 6
    STRING_EQUAL = 7
    STRING_NOT_EQUAL = 8


class InputFilter:
    def __init__(self, symbol, operator, value):
        self.symbol = symbol
        self.operator = operator
        self.value = value


class InputFilters(list):
    pass
