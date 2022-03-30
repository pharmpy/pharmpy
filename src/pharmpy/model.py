"""
===================
Generic Model class
===================

**Base class of all implementations.**

Inherit to *implement*, i.e. to define support for a specific model type.

Definitions
-----------
"""

import copy
import io
import pathlib
import warnings
from pathlib import Path

import pandas as pd
import sympy

from pharmpy.datainfo import ColumnInfo, DataInfo
from pharmpy.estimation import EstimationSteps
from pharmpy.parameter import Parameters
from pharmpy.plugins.utils import detect_model
from pharmpy.random_variables import RandomVariables
from pharmpy.statements import ModelStatements


class ModelError(Exception):
    """Exception for errors in model object"""

    pass


class ModelSyntaxError(ModelError):
    """Exception for Syntax errors in model code"""

    def __init__(self, msg='model syntax error'):
        super().__init__(msg)


class ModelfitResultsError(ModelError):
    """Exception for issues with ModelfitResults"""

    pass


class Model:
    """The Pharmpy model class"""

    def __init__(self):
        self.parameters = Parameters([])
        self.random_variables = RandomVariables([])
        self.statements = ModelStatements([])
        self.dependent_variable = sympy.Symbol('y')
        self.observation_transformation = self.dependent_variable
        self.modelfit_results = None
        self.parent_model = None

    def __repr__(self):
        return f'<Pharmpy model object {self.name}>'

    def _repr_html_(self):
        stat = self.statements._repr_html_()
        rvs = self.random_variables._repr_latex_()
        return f'<hr>{stat}<hr>${rvs}$<hr>{self.parameters._repr_html_()}<hr>'

    @property
    def name(self):
        """Name of the model"""
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("Name of a model has to be of string type")
        self._name = value

    @property
    def filename_extension(self):
        """Filename extension of model file"""
        return self._filename_extension

    @filename_extension.setter
    def filename_extension(self, value):
        if not isinstance(value, str):
            raise TypeError("Filename extension has to be of string type")
        self._filename_extension = value

    @property
    def dependent_variable(self):
        """The model dependent variable, i.e. y"""
        return self._dependent_variable

    @dependent_variable.setter
    def dependent_variable(self, value):
        self._dependent_variable = value

    @property
    def observation_transformation(self):
        """Transformation to be applied to the observation data"""
        return self._observation_transformation

    @observation_transformation.setter
    def observation_transformation(self, value):
        self._observation_transformation = value

    @property
    def parameters(self):
        """Definitions of population parameters

        See :class:`pharmpy.Parameters`
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        if isinstance(value, Parameters):
            inits = value.inits
        else:
            inits = value
        if isinstance(inits, pd.Series):
            inits = inits.to_dict()
        if inits and not self.random_variables.validate_parameters(inits):
            nearest = self.random_variables.nearest_valid_parameters(inits)
            if nearest != inits:
                before, after = self._compare_before_after_params(inits, nearest)
                warnings.warn(
                    f"Adjusting initial estimates to create positive semidefinite "
                    f"omega/sigma matrices.\nBefore adjusting:  {before}.\n"
                    f"After adjusting: {after}"
                )
                if isinstance(value, Parameters):
                    value.inits = nearest
                else:
                    value = nearest
            else:
                raise ValueError("New parameter inits are not valid")

        if not isinstance(value, Parameters):
            self._parameters.inits = value
        else:
            self._parameters = value

    @staticmethod
    def _compare_before_after_params(old, new):
        before = dict()
        after = dict()
        for key, value in old.items():
            if new[key] != value:
                before[key] = value
                after[key] = new[key]
        return before, after

    @property
    def random_variables(self):
        """Definitions of random variables

        See :class:`pharmpy.RandomVariables`
        """
        return self._random_variables

    @random_variables.setter
    def random_variables(self, value):
        if not isinstance(value, RandomVariables):
            raise TypeError("model.random_variables must be of RandomVariables type")
        self._random_variables = value

    @property
    def statements(self):
        """Definitions of model statements

        See :class:`pharmpy.ModelStatements`
        """
        return self._statements

    @statements.setter
    def statements(self, value):
        if not isinstance(value, ModelStatements):
            raise TypeError("model.statements must be of ModelStatements type")
        self._statements = value

    @property
    def estimation_steps(self):
        """Definitions of estimation steps

        See :class:`pharmpy.EstimationSteps`
        """
        return self._estimation_steps

    @estimation_steps.setter
    def estimation_steps(self, value):
        if not isinstance(value, EstimationSteps):
            raise TypeError("model.estimation_steps must be of EstimationSteps type")
        self._estimation_steps = value

    @property
    def datainfo(self):
        """Definitions of model statements

        See :class:`pharmpy.ModelStatements`
        """
        return self._datainfo

    @datainfo.setter
    def datainfo(self, value):
        if not isinstance(value, DataInfo):
            raise TypeError("model.datainfo must be of DataInfo type")
        self._datainfo = value

    @property
    def dataset(self):
        """Dataset connected to model"""
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self.update_datainfo()

    @property
    def initial_individual_estimates(self):
        """Initial estimates for individual parameters"""
        return self._initial_individual_estimates

    @initial_individual_estimates.setter
    def initial_individual_estimates(self, value):
        self._initial_individual_estimates = value

    @property
    def modelfit_results(self):
        """Modelfit results for this model"""
        return self._modelfit_results

    @modelfit_results.setter
    def modelfit_results(self, value):
        self._modelfit_results = value

    @property
    def model_code(self):
        """Model type specific code"""
        raise NotImplementedError("Generic model does not implement the model_code property")

    @property
    def parent_model(self):
        """Name of parent model"""
        return self._parent_model

    @parent_model.setter
    def parent_model(self, value):
        self._parent_model = value

    def read_modelfit_results(self):
        """Read in modelfit results"""
        raise NotImplementedError("Read modelfit results not implemented for generic models")

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
        model_copy = copy.deepcopy(self)
        model_copy.parent_model = self.name
        return model_copy

    @staticmethod
    def create_model(obj=None, **kwargs):
        """Factory for creating a :class:`pharmpy.model` object from an object representing the model

        .. _path-like object: https://docs.python.org/3/glossary.html#term-path-like-object

        Parameters
        ----------
        obj
            `path-like object`_ pointing to the model file or an IO object.

        Returns
        -------
        Model
            Generic :class:`~pharmpy.generic.Model` if obj is None, otherwise appropriate
            implementation is invoked (e.g. NONMEM7 :class:`~pharmpy.plugins.nonmem.Model`).
        """
        if isinstance(obj, str):
            path = Path(obj)
        elif isinstance(obj, pathlib.Path):  # DO NOT change to Path! Will fail fakefs tests
            path = obj
        elif isinstance(obj, io.IOBase):
            path = None
        elif obj is None:
            return Model()
        else:
            raise ValueError("Unknown input type to Model constructor")
        if path is not None:
            with open(path, 'r', encoding='latin-1') as fp:
                code = fp.read()
        else:
            code = obj.read()
        model_class = detect_model(code)
        model = model_class(code, path, **kwargs)
        # Setup model database here
        # Read in model results here?
        # Set filename extension?
        return model
