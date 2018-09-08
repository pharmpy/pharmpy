# -*- encoding: utf-8 -*-

from enum import Enum
from pathlib import Path

from pysn.parameter_model import ParameterModel
from pysn.output import ModelOutput


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
    """Generic model of any type

    Agnostic of model type. Subclass me to implement non-agnostic API."""

    def __init__(self, path, **kwargs):
        self.path = Path(path).resolve() if path else None
        if self.exists:
            self.load()

    @property
    def exists(self):
        if self.path and self.path.is_file():
            return True

    @property
    def content(self):
        if not self.exists:
            return None
        with open(str(self.path), 'r') as f:
            content = f.read()
        return content

    def write(self, *args, **kwargs):
        pass

    def validate(self):
        raise NotImplementedError

    def load(self):
        self.index = 0
        self.input = ModelInput(self)
        self.output = ModelOutput(self)
        self.parameters = ParameterModel(self)

    @property
    def index(self):
        """Submodel unique index (setter might accept name lookup)"""
        return self._index

    @index.setter
    def index(self, new):
        if new != 0:
            raise ModelLookupError(new)
        self._index = new

    @property
    def executed(self):
        """True iff model has been previously executed (i.e. has results).

        Is callback for :class:`ModelOutput`."""
        # SHOULD contain a call to execution class. An implementation of THAT should then know how
        # to check on current platform/cluster system (also WITHOUT initializing a run directory).
        # SHOULDN'T need to overload this by implementation.
        return True  # TODO: implement me

    def __str__(self):
        if self.exists:
            return self.read


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
class Operator(Enum):
    EQUAL = 1
    NOT_EQUAL = 2
    LESS_THAN = 3
    LESS_THAN_OR_EQUAL = 4
    GREATER_THAN = 5
    GREATER_THAN_OR_EQUAL = 6


class InputFilter:
    def __init__(self, symbol, operator, value):
        self.symbol = symbol
        self.operator = operator
        self.value = value


class InputFilters(list):
    pass
