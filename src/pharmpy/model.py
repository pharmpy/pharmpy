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
import re
from pathlib import Path

import sympy

import pharmpy.symbols
from pharmpy import Parameters, RandomVariables
from pharmpy.datainfo import ColumnInfo, DataInfo
from pharmpy.workflows import default_model_database


def canonicalize_data_transformation(model, value):
    if value is None:
        value = model.dependent_variable
    else:
        value = sympy.sympify(value)
        if value.free_symbols != {model.dependent_variable}:
            raise ValueError(
                f"Expression for data transformation must contain the dependent variable "
                f"{model.dependent_variable} and no other variables"
            )
    return value


class ModelException(Exception):
    pass


class ModelSyntaxError(ModelException):
    def __init__(self, msg='model syntax error'):
        super().__init__(msg)


class Model:
    """
    Attribute: name
       dependent_variable
       parameters
       random_variables
       statements
       dataset
    """

    def __init__(self):
        self.modelfit_results = None

    def to_generic_model(self):
        """Convert a model into the base model class"""
        model = Model()
        model.parameters = self.parameters.copy()
        model.random_variables = self.random_variables.copy()
        model.statements = self.statements.copy()
        model.dataset = self.dataset.copy()
        model.name = self.name
        model.dependent_variable = self.dependent_variable
        model.estimation_steps = self.estimation_steps
        try:
            model.database = self.database
        except AttributeError:
            pass
        return model

    def __repr__(self):
        return f'<Pharmpy model object {self.name}>'

    def _repr_html_(self):
        stat = self.statements._repr_html_()
        rvs = self.random_variables._repr_latex_()
        return f'<hr>{stat}<hr>${rvs}$<hr>{self.parameters._repr_html_()}<hr>'

    @property
    def modelfit_results(self):
        return self._modelfit_results

    @modelfit_results.setter
    def modelfit_results(self, value):
        self._modelfit_results = value

    @property
    def data_transformation(self):
        """Transformation used for DV in dataset"""
        try:
            return self._data_transformation
        except AttributeError:
            return self.dependent_variable

    @data_transformation.setter
    def data_transformation(self, value):
        value = canonicalize_data_transformation(self, value)
        self._data_transformation = value

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
                filename = f'{self.name}{self.filename_extension}'
            except AttributeError:
                raise ValueError(
                    'Cannot name model file as no path argument was supplied and the'
                    'model has no name.'
                )
            path = path / filename
            new_name = None
        else:
            # Set new name given filename, but after we've checked for existence
            new_name = path.stem
        if not force and path.exists():
            raise FileExistsError(f'File {path} already exists.')
        if new_name:
            self.name = new_name
        self.update_source(path=path, force=force)
        if not force and path.exists():
            raise FileExistsError(f'Cannot overwrite model at {path} with "force" not set')
        with open(path, 'w', encoding='latin-1') as fp:
            fp.write(self.model_code)
        self.database = default_model_database(path=path.parent)
        return path

    def update_inits(self):
        """Update inital estimates of model from its own ModelfitResults"""
        if self.modelfit_results:
            self.parameters = self.modelfit_results.parameter_estimates
        else:
            # FIXME: Other exception here. ModelfitError?
            raise ModelException(
                "Cannot update initial parameter estimates " "since parameters were not estimated"
            )

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self.update_datainfo()

    def update_datainfo(self):
        """Update model.datainfo for a new dataset"""
        colnames = self.dataset.columns
        try:
            curdi = self.datainfo
        except AttributeError:
            curdi = DataInfo()
        newdi = DataInfo()
        for colname in colnames:
            try:
                col = curdi[colname]
            except IndexError:
                col = ColumnInfo(colname)
            newdi.append(col)
        if curdi != newdi:
            # Remove path if dataset has been updated
            newdi.path = None
        self.datainfo = newdi

    def copy(self):
        """Create a deepcopy of the model object"""
        return copy.deepcopy(self)

    def update_individual_estimates(self, source):
        self.initial_individual_estimates = source.modelfit_results.individual_estimates

    def read_raw_dataset(self, parse_columns=tuple()):
        raise NotImplementedError()

    def bump_model_number(self, path='.'):
        """If the model name ends in a number increase it to next available file
        else do nothing.
        """
        path = Path(path)
        name = self.name
        m = re.search(r'(.*?)(\d+)$', name)
        if m:
            stem = m.group(1)
            n = int(m.group(2))
            while True:
                n += 1
                new_name = f'{stem}{n}'
                new_path = (path / new_name).with_suffix(self.filename_extension)
                if not new_path.exists():
                    break
            self.name = new_name

    def create_symbol(self, stem, force_numbering=False):
        """Create a new unique variable symbol

        Parameters
        ----------
        stem : str
            First part of the new variable name
        force_numbering : bool
            Forces addition of number to name even if variable does not exist, e.g.
            COVEFF --> COVEFF1
        """
        symbols = [str(symbol) for symbol in self.statements.free_symbols]
        params = [param.name for param in self.parameters]
        rvs = [rv.name for rv in self.random_variables]
        dataset_col = self.datainfo.names
        misc = [self.dependent_variable]

        all_names = symbols + params + rvs + dataset_col + misc

        if str(stem) not in all_names and not force_numbering:
            return pharmpy.symbols.symbol(str(stem))

        i = 1
        while True:
            candidate = f'{stem}{i}'
            if candidate not in all_names:
                return pharmpy.symbols.symbol(candidate)
            i += 1

    def remove_unused_parameters_and_rvs(self):
        """Remove any parameters and rvs that are not used in the model statements"""
        symbols = self.statements.free_symbols

        new_rvs = RandomVariables()
        for rv in self.random_variables:
            # FIXME: change if rvs are random symbols in expressions
            if rv.symbol in symbols or not symbols.isdisjoint(rv.sympy_rv.pspace.free_symbols):
                new_rvs.append(rv)
        self.random_variables = new_rvs

        new_params = Parameters()
        for p in self.parameters:
            symb = p.symbol
            if symb in symbols or symb in new_rvs.free_symbols or (p.fix and p.init == 0):
                new_params.append(p)
        self.parameters = new_params
