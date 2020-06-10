"""
===================
Generic Model class
===================

**Base class of all implementations.**

Inherit to *implement*, i.e. to define support for a specific model type. Duck typing is utilized,
but an implementation is expected to implement **all** methods/attributes.

Definitions
-----------
"""

import copy
from pathlib import Path

import sympy


class ModelException(Exception):
    pass


class ModelSyntaxError(ModelException):
    def __init__(self, msg='model syntax error'):
        super().__init__(msg)


class Model:
    """
     Property: name
    """
    @property
    def modelfit_results(self):
        return None

    def update_source(self):
        """Update the source"""
        self.source.code = str(self)

    def write(self, path='', force=False):
        """Write model to file using its source format
           If no path is supplied or does not contain a filename a name is created
           from the name property of the model
           Will not overwrite in case force is True.
           return path written to
        """
        path = Path(path)
        if not path or path.is_dir():
            try:
                filename = f'{self.name}{self.source.filename_extension}'
            except AttributeError:
                raise ValueError('Cannot name model file as no path argument was supplied and the'
                                 'model has no name.')
            path = path / filename
        if not force and path.exists():
            raise FileExistsError(f'File {path} already exists.')
        self.update_source(path=path, force=force)
        self.source.write(path, force=force)
        return path

    def update_inits(self):
        """Update inital estimates of model from its own ModelfitResults
        """
        if self.modelfit_results:
            self.parameters = self.modelfit_results.parameter_estimates
        else:
            # FIXME: Other exception here. ModelfitError?
            raise ModelException("Cannot update initial parameter estimates "
                                 "since parameters were not estimated")

    def copy(self):
        """Create a deepcopy of the model object"""
        return copy.deepcopy(self)

    def update_individual_estimates(self, source):
        self.initial_individual_estimates = self.modelfit_results.individual_estimates

    @property
    def dataset(self):
        raise NotImplementedError()

    def read_raw_dataset(self, parse_columns=tuple()):
        raise NotImplementedError()

    def create_symbol(self, stem):
        """Create a new unique variable symbol

           stem - First part of the new variable name
        """
        # TODO: Also check parameter and rv names and dataset columns
        symbols = self.statements.free_symbols
        i = 1
        while True:
            candidate = sympy.Symbol(f'{stem}{i}')
            if candidate not in symbols:
                return candidate
            i += 1
