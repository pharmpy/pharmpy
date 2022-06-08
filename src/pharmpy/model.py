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
        self.initial_individual_estimates = None
        self.value_type = 'PREDICTION'
        self.description = ''

    def __eq__(self, other):
        """Compare two models for equality

        Tests whether a model is equal to another model. This ignores
        implementation-specific details such as NONMEM $DATA and FILE
        pointers, or certain $TABLE printing options.

        Parameters
        ----------
        other : Model
            Other model to compare this one with

        Examples
        --------
        >>> from pharmpy import Model
        >>> a = Model()
        >>> a == a
        True
        >>> a == 0
        Traceback (most recent call last):
         ...
        NotImplementedError: Cannot compare Model with <class 'int'>
        >>> a == None
        Traceback (most recent call last):
         ...
        NotImplementedError: Cannot compare Model with <class 'NoneType'>
        >>> b = Model()
        >>> b == a
        True
        >>> a.name = 'a'
        >>> b.name = 'b'
        >>> a == b
        True
        """
        if self is other:
            return True
        if not isinstance(other, Model):
            raise NotImplementedError(f'Cannot compare Model with {type(other)}')

        if self.parameters != other.parameters:
            return False
        if self.random_variables != other.random_variables:
            return False
        if self.statements != other.statements:
            return False
        if self.dependent_variable != other.dependent_variable:
            return False
        if self.observation_transformation != other.observation_transformation:
            return False
        if self.estimation_steps != other.estimation_steps:
            return False
        if self.initial_individual_estimates != other.initial_individual_estimates:
            return False
        if self.datainfo != other.datainfo:
            return False
        if self.value_type != other.value_type:
            return False

        return True

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
    def value_type(self):
        """The type of the model value (dependent variable)

        By default this is set to 'PREDICTION' to mean that the model outputs a prediction.
        It could optionally be set to 'LIKELIHOOD' or '-2LL' to let the model output the likelihood
        or -2*log(likelihood) of the prediction. If set to a symbol this variable can be used to
        change the type for different records. The model would then set this symbol to 0 for
        a prediction value, 1 for likelihood and 2 for -2ll.
        """
        return self._value_type

    @value_type.setter
    def value_type(self, value):
        allowed_strings = ['PREDICTION', 'LIKELIHOOD', '-2LL']
        if isinstance(value, str):
            if value.upper() not in allowed_strings:
                raise ValueError(
                    f"Cannot set value_type to {value}. Must be one of {allowed_strings} "
                    f"or a symbol"
                )
            value = value.upper()
        elif not isinstance(value, sympy.Symbol):
            raise ValueError("Can only set value_type to one of {allowed_strings} or a symbol")
        self._value_type = value

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
        inits = value.inits
        if inits and not self.random_variables.validate_parameters(inits):
            nearest = self.random_variables.nearest_valid_parameters(inits)
            if nearest != inits:
                before, after = self._compare_before_after_params(inits, nearest)
                warnings.warn(
                    f"Adjusting initial estimates to create positive semidefinite "
                    f"omega/sigma matrices.\nBefore adjusting:  {before}.\n"
                    f"After adjusting: {after}"
                )
                value = value.set_initial_estimates(nearest)
            else:
                raise ValueError("New parameter inits are not valid")

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

    def has_same_dataset_as(self, other):
        """Check if this model has the same dataset as another model

        Parameters
        ----------
        other : Model
            Another model

        Returns
        -------
        bool
            True if both models have the same dataset
        """
        if self.dataset is None:
            if other.dataset is None:
                return True
            else:
                return False

        if other.dataset is None:
            return False

        # NOTE rely on duck-typing here (?)
        return self.dataset.equals(other.dataset)

    @property
    def description(self):
        """A free text discription of the model"""
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    def read_modelfit_results(self, path: Path):
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
                datatype = ColumnInfo.convert_pd_dtype_to_datatype(
                    self.dataset.dtypes[colname].name
                )
                col = ColumnInfo(colname, datatype=datatype)
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
        elif isinstance(obj, pathlib.Path):
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
