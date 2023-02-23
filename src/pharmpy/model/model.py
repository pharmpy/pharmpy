"""
===================
Generic Model class
===================

**Base class of all implementations.**

Inherit to *implement*, i.e. to define support for a specific model type.

Definitions
-----------
"""
from __future__ import annotations

import warnings
from io import IOBase
from pathlib import Path

from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.immutable import Immutable
from pharmpy.plugins.utils import detect_model

from .datainfo import ColumnInfo, DataInfo
from .estimation import EstimationSteps
from .parameters import Parameters
from .random_variables import RandomVariables
from .statements import Statements


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


class Model(Immutable):
    """The Pharmpy model class"""

    def __init__(
        self,
        name='',
        parameters=Parameters(),
        random_variables=RandomVariables.create(()),
        statements=Statements(),
        dataset=None,
        datainfo=DataInfo(),
        dependent_variable=None,
        observation_transformation=None,
        estimation_steps=EstimationSteps(),
        modelfit_results=None,
        parent_model=None,
        initial_individual_estimates=None,
        filename_extension='',
        value_type='PREDICTION',
        description='',
        internals=None,
    ):
        actual_dependent_variable = (
            sympy.Symbol('y') if dependent_variable is None else dependent_variable
        )
        self._name = name
        self._datainfo = datainfo
        self._dataset = dataset
        self._random_variables = random_variables
        self._parameters = parameters
        self._statements = statements
        self._dependent_variable = actual_dependent_variable
        if observation_transformation is None:
            self._observation_transformation = actual_dependent_variable
        else:
            self._observation_transformation = observation_transformation
        self._estimation_steps = estimation_steps
        self._modelfit_results = modelfit_results
        self._parent_model = parent_model
        self._initial_individual_estimates = initial_individual_estimates
        self._filename_extension = filename_extension
        self._value_type = value_type
        self._description = description
        self._internals = internals

    def _canonicalize_value_type(self, value):
        allowed_strings = ('PREDICTION', 'LIKELIHOOD', '-2LL')
        if isinstance(value, str):
            if value.upper() not in allowed_strings:
                raise ValueError(
                    f"Cannot set value_type to {value}. Must be one of {allowed_strings} "
                    f"or a symbol"
                )
            value = value.upper()
        elif not isinstance(value, sympy.Symbol):
            raise ValueError("Can only set value_type to one of {allowed_strings} or a symbol")
        return value

    @staticmethod
    def _canonicalize_parameter_estimates(params, rvs):
        inits = params.inits
        if not rvs.validate_parameters(inits):
            nearest = rvs.nearest_valid_parameters(inits)
            before, after = compare_before_after_params(inits, nearest)
            warnings.warn(
                f"Adjusting initial estimates to create positive semidefinite "
                f"omega/sigma matrices.\nBefore adjusting:  {before}.\n"
                f"After adjusting: {after}"
            )
            params = params.set_initial_estimates(nearest)
        return params

    @staticmethod
    def _canonicalize_random_variables(rvs):
        if not isinstance(rvs, RandomVariables):
            raise TypeError("model.random_variables must be of RandomVariables type")

    @staticmethod
    def _canonicalize_statements(statements):
        if not isinstance(statements, Statements):
            raise TypeError("model.statements must be of Statements type")

    def replace(self, **kwargs):
        name = kwargs.get('name', self.name)
        if not isinstance(name, str):
            raise TypeError("Name of a model has to be of string type")

        dependent_variable = kwargs.get('dependent_variable', self.dependent_variable)
        parameters = kwargs.get('parameters', self.parameters)

        if 'random_variables' in kwargs:
            random_variables = kwargs['random_variables']
            Model._canonicalize_random_variables(random_variables)
        else:
            random_variables = self.random_variables

        parameters = Model._canonicalize_parameter_estimates(parameters, random_variables)

        if 'statements' in kwargs:
            statements = kwargs['statements']
            Model._canonicalize_statements(statements)
        else:
            statements = self.statements

        if 'dataset' in kwargs:
            dataset = kwargs['dataset']
            new_dataset = True
        else:
            dataset = self._dataset
            new_dataset = False

        if 'datainfo' in kwargs:
            datainfo = kwargs['datainfo']
            if not isinstance(datainfo, DataInfo):
                raise TypeError("model.datainfo must be of DataInfo type")
        else:
            datainfo = self._datainfo

        if new_dataset:
            datainfo = update_datainfo(datainfo, dataset)

        estimation_steps = kwargs.get('estimation_steps', self.estimation_steps)
        if not isinstance(estimation_steps, EstimationSteps):
            raise TypeError("model.estimation_steps must be of EstimationSteps type")
        modelfit_results = kwargs.get('modelfit_results', self.modelfit_results)
        parent_model = kwargs.get('parent_model', self.parent_model)
        initial_individual_estimates = kwargs.get(
            'initial_individual_estimates', self.initial_individual_estimates
        )
        filename_extension = kwargs.get('filename_extension', self.filename_extension)
        if not isinstance(filename_extension, str):
            raise TypeError("Filename extension has to be of string type")
        if 'value_type' in kwargs:
            value_type = self._canonicalize_value_type(kwargs['value_type'])
        else:
            value_type = self.value_type
        description = kwargs.get('description', self.description)
        observation_transformation = kwargs.get(
            'observation_transformation', self.observation_transformation
        )
        internals = kwargs.get('internals', self._internals)
        return self.__class__(
            name=name,
            dependent_variable=dependent_variable,
            parameters=parameters,
            random_variables=random_variables,
            statements=statements,
            dataset=dataset,
            datainfo=datainfo,
            estimation_steps=estimation_steps,
            modelfit_results=modelfit_results,
            parent_model=parent_model,
            initial_individual_estimates=initial_individual_estimates,
            filename_extension=filename_extension,
            value_type=value_type,
            description=description,
            internals=internals,
            observation_transformation=observation_transformation,
        )

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
        >>> from pharmpy.model import Model
        >>> from pharmpy.modeling import load_example_model
        >>> a = load_example_model("pheno")
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
        >>> b = load_example_model("pheno")
        >>> b == a
        True
        >>> a = a.replace(name='a')
        >>> b = b.replace(name='b')
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

    @property
    def filename_extension(self):
        """Filename extension of model file"""
        return self._filename_extension

    @property
    def dependent_variable(self):
        """The model dependent variable, i.e. y"""
        return self._dependent_variable

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

    @property
    def observation_transformation(self):
        """Transformation to be applied to the observation data"""
        return self._observation_transformation

    @property
    def parameters(self):
        """Definitions of population parameters

        See :class:`pharmpy.Parameters`
        """
        return self._parameters

    @property
    def random_variables(self):
        """Definitions of random variables

        See :class:`pharmpy.RandomVariables`
        """
        return self._random_variables

    @property
    def statements(self):
        """Definitions of model statements

        See :class:`pharmpy.Statements`
        """
        return self._statements

    @property
    def estimation_steps(self):
        """Definitions of estimation steps

        See :class:`pharmpy.EstimationSteps`
        """
        return self._estimation_steps

    @property
    def datainfo(self):
        """Definitions of model statements

        See :class:`pharmpy.Statements`
        """
        return self._datainfo

    @property
    def dataset(self):
        """Dataset connected to model"""
        return self._dataset

    @property
    def initial_individual_estimates(self):
        """Initial estimates for individual parameters"""
        return self._initial_individual_estimates

    @property
    def internals(self):
        """Internal data for tool specific part of model"""
        return self._internals

    @property
    def modelfit_results(self):
        """Modelfit results for this model"""
        return self._modelfit_results

    @property
    def model_code(self):
        """Model type specific code"""
        raise NotImplementedError("Generic model does not implement the model_code property")

    @property
    def parent_model(self):
        """Name of parent model"""
        return self._parent_model

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
        if obj is None:
            return Model()
        elif isinstance(obj, IOBase):
            path = None
            code = obj.read()
        elif isinstance(obj, (str, Path)):
            path = Path(obj)
            with open(path, 'r', encoding='latin-1') as fp:
                code = fp.read()
        else:
            raise ValueError("Unknown input type to Model constructor")

        model_module = detect_model(code)
        model = model_module.parse_code(code, path, **kwargs)
        # Setup model database here
        # Read in model results here?
        # Set filename extension?
        return model

    def update_source(self):
        """Update source code of the model. If any paths need to be changed or added (e.g. for a
        NONMEM model with an updated dataset) they will be replaced with DUMMYPATH"""
        return self

    def write_files(self, path=None, force=False):
        """Write necessary files of the model. If any paths need to be changed or added (e.g. for a
        NONMEM model with an updated dataset) they will be replaced with the correct path. update_source()
        will be called first."""
        return self


def compare_before_after_params(old, new):
    # FIXME Move this to the right module
    before = {}
    after = {}
    for key, value in old.items():
        if new[key] != value:
            before[key] = value
            after[key] = new[key]
    return before, after


def update_datainfo(curdi: DataInfo, dataset: pd.DataFrame):
    colnames = dataset.columns
    columns = []
    for colname in colnames:
        try:
            col = curdi[colname]
        except IndexError:
            datatype = ColumnInfo.convert_pd_dtype_to_datatype(dataset.dtypes[colname].name)
            col = ColumnInfo.create(colname, datatype=datatype)
        columns.append(col)
    newdi = curdi.replace(columns=columns)

    # NOTE Remove path if dataset has been updated
    return curdi if newdi == curdi else newdi.replace(path=None)
