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

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Self, Union

import pharmpy
from pharmpy.basic import Expr, TExpr, TSymbol
from pharmpy.deps import pandas as pd
from pharmpy.internals.df import hash_df_runtime
from pharmpy.internals.immutable import Immutable, cache_method, frozenmapping
from pharmpy.model.external import detect_model

from .datainfo import ColumnInfo, DataInfo
from .execution_steps import ExecutionSteps
from .parameters import Parameters
from .random_variables import RandomVariables
from .statements import CompartmentalSystem, Statements


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


@dataclass(frozen=True)
class ModelInternals:
    def __init__(self):
        pass

    def replace(self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)


class Model(Immutable):
    """The Pharmpy model class"""

    filename_extension = '.ppmod'

    def __init__(
        self,
        name: str = '',
        parameters: Parameters = Parameters(),
        random_variables: RandomVariables = RandomVariables.create(()),
        statements: Statements = Statements(),
        dataset: Optional[pd.DataFrame] = None,
        datainfo: DataInfo = DataInfo(),
        dependent_variables: frozenmapping[Expr, int] = frozenmapping({Expr.symbol('y'): 1}),
        observation_transformation: Optional[frozenmapping[Expr, Expr]] = None,
        execution_steps: ExecutionSteps = ExecutionSteps(),
        initial_individual_estimates: Optional[pd.DataFrame] = None,
        value_type: Union[str, Expr] = 'PREDICTION',
        description: str = '',
        internals: Optional[ModelInternals] = None,
    ):
        self._name = name
        self._datainfo = datainfo
        self._dataset = dataset
        self._random_variables = random_variables
        self._parameters = parameters
        self._statements = statements
        self._dependent_variables = dependent_variables
        if observation_transformation is None:
            observation_transformation = frozenmapping(
                {dv: dv for dv in dependent_variables.keys()}
            )
        self._observation_transformation = observation_transformation
        self._execution_steps = execution_steps
        self._initial_individual_estimates = initial_individual_estimates
        self._value_type = value_type
        self._description = description
        self._internals = internals

    @classmethod
    def create(
        cls,
        name: str,
        parameters: Optional[Parameters] = None,
        random_variables: Optional[RandomVariables] = None,
        statements: Optional[Statements] = None,
        dataset: Optional[pd.DataFrame] = None,
        datainfo: DataInfo = DataInfo(),
        dependent_variables: Optional[Mapping[TSymbol, int]] = None,
        observation_transformation: Optional[Mapping[TSymbol, TExpr]] = None,
        execution_steps: Optional[ExecutionSteps] = None,
        initial_individual_estimates: Optional[pd.DataFrame] = None,
        value_type: Union[str, Expr] = 'PREDICTION',
        description: str = '',
        internals: Optional[ModelInternals] = None,
    ):
        Model._canonicalize_name(name)
        dvs = Model._canonicalize_dependent_variables(dependent_variables)
        obs_transformation = Model._canonicalize_observation_transformation(
            observation_transformation, dvs
        )
        parameters = Model._canonicalize_parameters(parameters)
        random_variables = Model._canonicalize_random_variables(random_variables)
        parameters = Model._canonicalize_parameter_estimates(parameters, random_variables)
        execution_steps = Model._canonicalize_execution_steps(execution_steps)
        value_type = Model._canonicalize_value_type(value_type)
        if not isinstance(datainfo, DataInfo):
            raise TypeError("model.datainfo must be of DataInfo type")

        if dataset is not None:
            datainfo = update_datainfo(datainfo, dataset)

        statements = Model._canonicalize_statements(
            statements, parameters, random_variables, datainfo
        )
        Model._check_symbol_names(datainfo, statements)
        return cls(
            name=name,
            dependent_variables=dvs,
            observation_transformation=obs_transformation,
            parameters=parameters,
            random_variables=random_variables,
            execution_steps=execution_steps,
            statements=statements,
            description=description,
            value_type=value_type,
            internals=internals,
            initial_individual_estimates=initial_individual_estimates,
            dataset=dataset,
            datainfo=datainfo,
        )

    ALLOWED_VALUE_TYPE_STRINGS = ('PREDICTION', 'LIKELIHOOD', '-2LL')

    @staticmethod
    def _canonicalize_value_type(value: Any) -> Union[str, Expr]:
        allowed_strings = ('PREDICTION', 'LIKELIHOOD', '-2LL')
        if isinstance(value, str):
            if value.upper() not in Model.ALLOWED_VALUE_TYPE_STRINGS:
                raise ValueError(
                    f"Cannot set value_type to {value}. Must be one of {allowed_strings} "
                    f"or a symbol"
                )
            value = value.upper()
        elif not (isinstance(value, Expr) and value.is_symbol()):
            raise ValueError(f"Can only set value_type to one of {allowed_strings} or a symbol")
        return value

    @staticmethod
    def _canonicalize_parameter_estimates(params, rvs) -> Parameters:
        inits = params.inits
        if not rvs.validate_parameters(inits):
            nearest = rvs.nearest_valid_parameters(inits)
            params = params.set_initial_estimates(nearest)
        return params

    @staticmethod
    def _canonicalize_random_variables(rvs: Optional[RandomVariables]) -> RandomVariables:
        if isinstance(rvs, RandomVariables):
            return rvs
        elif rvs is None:
            return RandomVariables.create()
        else:
            raise TypeError("model.random_variables must be of RandomVariables type")

    @staticmethod
    def _canonicalize_statements(
        statements: Any,
        params: Parameters,
        rvs: RandomVariables,
        datainfo: DataInfo,
    ) -> Statements:
        if statements is None:
            return Statements()
        if not isinstance(statements, Statements):
            raise TypeError("model.statements must be of Statements type")

        colnames = {Expr.symbol(colname) for colname in datainfo.names}
        symbs_all = rvs.free_symbols.union(params.symbols).union(colnames)
        if statements.ode_system is not None:
            symbs_all = symbs_all.union({statements.ode_system.t})

        for i, statement in enumerate(statements):
            if isinstance(statement, CompartmentalSystem):
                continue

            symbs = statement.expression.free_symbols
            if not symbs.issubset(symbs_all):
                # E.g. after solve_ode_system
                if statement.symbol.is_function():
                    symbs_all.add(Expr.symbol(statement.symbol.name))
                    for arg in statement.symbol.args:
                        if arg.is_symbol():
                            symbs_all.add(arg)

                for symb in symbs:
                    if symb in symbs_all:
                        continue
                    if str(symb) == 'NaN':
                        continue
                    if statements.find_assignment(symb) is None:
                        raise ValueError(f'Symbol {symb} is not defined')
                    if statements[:i].find_assignment_index(symb) is None:
                        raise ValueError(f'Symbol {symb} defined after being used')

        return statements

    @staticmethod
    def _canonicalize_name(name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("Name of a model has to be of string type")

    @staticmethod
    def _canonicalize_dependent_variables(
        dvs: Optional[Mapping[TSymbol, int]],
    ) -> frozenmapping[Expr, int]:
        if dvs is None:
            dvs = {Expr.symbol('y'): 1}
        for key, value in dvs.items():
            if isinstance(key, str):
                key = Expr.symbol(key)
            if not (isinstance(key, Expr) and key.is_symbol()):
                raise TypeError("Dependent variable keys must be a string or a symbol")
            if not isinstance(value, int):
                raise TypeError("Dependent variable values must be of int type")
        return frozenmapping(dvs)

    @staticmethod
    def _canonicalize_observation_transformation(
        obs: Optional[Mapping[TSymbol, TExpr]],
        dvs: frozenmapping[Expr, int],
    ) -> frozenmapping[Expr, Expr]:
        if obs is None:
            obs = {dv: dv for dv in dvs.keys()}
        for key, value in obs.items():
            key = Expr(key)
            value = Expr(value)
        return frozenmapping(obs)

    @staticmethod
    def _canonicalize_parameters(params: Optional[Parameters]) -> Parameters:
        if params is None:
            return Parameters()
        else:
            if not isinstance(params, Parameters):
                raise TypeError("parameters must be of Parameters type")
            return params

    @staticmethod
    def _canonicalize_execution_steps(steps: Optional[ExecutionSteps]) -> ExecutionSteps:
        if steps is None:
            return ExecutionSteps()
        else:
            if not isinstance(steps, ExecutionSteps):
                raise TypeError("model.execution_steps must be of ExecutionSteps type")
            return steps

    @staticmethod
    def _check_symbol_names(datainfo: DataInfo, statements: Statements) -> None:
        # Currently that column names do not overlap with lhs in statements
        colnames = {Expr.symbol(colname) for colname in datainfo.names}
        for name in colnames:
            if not str(name).isidentifier():
                raise ValueError(f"A column name is not a valid variable identifier: {name}")
        col_lhs = colnames.intersection(statements.lhs_symbols)
        if col_lhs:
            raise ValueError(
                f"The following symbols are defined both in the dataset "
                f"and in the model statements: {col_lhs}"
            )

    def replace(self, **kwargs) -> Self:
        name = kwargs.get('name', self.name)
        Model._canonicalize_name(name)

        description = kwargs.get('description', self.description)
        internals = kwargs.get('internals', self._internals)
        initial_individual_estimates = kwargs.get(
            'initial_individual_estimates', self.initial_individual_estimates
        )
        for key_name in (
            'name',
            'description',
            'internals',
            'initial_individual_estimates',
        ):
            try:
                kwargs.pop(key_name)
            except KeyError:
                pass

        if 'dependent_variables' in kwargs:
            dependent_variables = Model._canonicalize_dependent_variables(
                kwargs['dependent_variables']
            )
            kwargs.pop('dependent_variables')
        else:
            dependent_variables = self.dependent_variables

        if 'observation_transformation' in kwargs:
            observation_transformation = Model._canonicalize_observation_transformation(
                kwargs['observation_transformation'], dependent_variables
            )
            kwargs.pop('observation_transformation')
        else:
            observation_transformation = self.observation_transformation

        if 'parameters' in kwargs:
            parameters = Model._canonicalize_parameters(kwargs['parameters'])
            kwargs.pop('parameters')
        else:
            parameters = self.parameters

        if 'random_variables' in kwargs:
            random_variables = Model._canonicalize_random_variables(kwargs['random_variables'])
            kwargs.pop('random_variables')
        else:
            random_variables = self.random_variables

        parameters = Model._canonicalize_parameter_estimates(parameters, random_variables)

        if 'dataset' in kwargs:
            dataset = kwargs['dataset']
            new_dataset = True
            kwargs.pop('dataset')
        else:
            dataset = self._dataset
            new_dataset = False

        if 'datainfo' in kwargs:
            datainfo = kwargs['datainfo']
            if not isinstance(datainfo, DataInfo):
                raise TypeError("model.datainfo must be of DataInfo type")
            new_datainfo = True
            kwargs.pop('datainfo')
        else:
            datainfo = self._datainfo
            new_datainfo = False

        if new_dataset and dataset is not None:
            datainfo = update_datainfo(datainfo, dataset)
            if not new_datainfo:
                datainfo = datainfo.replace(path=None)

        # Has to be checked after datainfo is updated since it looks for symbols in datainfo as well
        if 'statements' in kwargs:
            statements = Model._canonicalize_statements(
                kwargs['statements'], parameters, random_variables, datainfo
            )
            kwargs.pop('statements')
        else:
            statements = self.statements

        if 'execution_steps' in kwargs:
            execution_steps = Model._canonicalize_execution_steps(kwargs['execution_steps'])
            kwargs.pop('execution_steps')
        else:
            execution_steps = self.execution_steps

        if 'value_type' in kwargs:
            value_type = Model._canonicalize_value_type(kwargs['value_type'])
            kwargs.pop('value_type')
        else:
            value_type = self.value_type

        if len(kwargs) != 0:
            raise ValueError(f'Invalid keywords given : {[key for key in kwargs.keys()]}')

        if new_dataset or 'datainfo' in kwargs or 'statements' in kwargs:
            Model._check_symbol_names(datainfo, statements)

        return self.__class__(
            name=name,
            dependent_variables=dependent_variables,
            parameters=parameters,
            random_variables=random_variables,
            statements=statements,
            dataset=dataset,
            datainfo=datainfo,
            execution_steps=execution_steps,
            initial_individual_estimates=initial_individual_estimates,
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
        >>> a == None
        False
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
            return NotImplemented

        if self.parameters != other.parameters:
            return False
        if self.random_variables != other.random_variables:
            return False
        if self.statements != other.statements:
            return False
        if self.dependent_variables != other.dependent_variables:
            return False
        if self.observation_transformation != other.observation_transformation:
            return False
        if self.execution_steps != other.execution_steps:
            return False
        if self.initial_individual_estimates is None:
            if other.initial_individual_estimates is not None:
                return False
        else:
            if other.initial_individual_estimates is None:
                return False
            elif not self.initial_individual_estimates.equals(other.initial_individual_estimates):
                return False
        if self.datainfo != other.datainfo:
            return False
        if self.value_type != other.value_type:
            return False

        return True

    @cache_method
    def __hash__(self):
        dataset_hash = hash_df_runtime(self._dataset) if self._dataset is not None else None
        return hash(
            (
                self._parameters,
                self._random_variables,
                self._statements,
                self._dependent_variables,
                self._observation_transformation,
                self._execution_steps,
                self._initial_individual_estimates,
                self._datainfo,
                dataset_hash,
                self._value_type,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        if self._initial_individual_estimates is not None:
            ie = self._initial_individual_estimates.to_dict()
        else:
            ie = None
        depvars = {str(key): val for key, val in self._dependent_variables.items()}
        obstrans = {
            key.serialize(): val.serialize()
            for key, val in self._observation_transformation.items()
        }
        value_type = (
            self._value_type if isinstance(self._value_type, str) else self._value_type.serialize()
        )
        return {
            'parameters': self._parameters.to_dict(),
            'random_variables': self._random_variables.to_dict(),
            'statements': self._statements.to_dict(),
            'execution_steps': self._execution_steps.to_dict(),
            'datainfo': self._datainfo.to_dict(),
            'value_type': value_type,
            'dependent_variables': depvars,
            'observation_transformation': obstrans,
            'initial_individual_estimates': ie,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Model:
        ie_dict = d['initial_individual_estimates']
        if ie_dict is None:
            ie = None
        else:
            ie = pd.DataFrame.from_dict(ie_dict)
        depvars = {Expr.symbol(key): value for key, value in d['dependent_variables'].items()}
        obstrans = {
            Expr.deserialize(key): Expr.deserialize(val)
            for key, val in d['observation_transformation'].items()
        }
        value_type = (
            d['value_type']
            if d['value_type'] in Model.ALLOWED_VALUE_TYPE_STRINGS
            else Expr.deserialize(d['value_type'])
        )
        return cls(
            parameters=Parameters.from_dict(d['parameters']),
            random_variables=RandomVariables.from_dict(d['random_variables']),
            statements=Statements.from_dict(d['statements']),
            execution_steps=ExecutionSteps.from_dict(d['execution_steps']),
            datainfo=DataInfo.from_dict(d['datainfo']),
            value_type=value_type,
            dependent_variables=frozenmapping(depvars),
            observation_transformation=frozenmapping(obstrans),
            initial_individual_estimates=ie,
        )

    def __repr__(self):
        return f'<Pharmpy model object {self.name}>'

    def _repr_html_(self) -> str:
        stat = self.statements._repr_html_()
        rvs = self.random_variables._repr_latex_()
        params = self.parameters._repr_html_()
        return f'<hr>{stat}<hr>${rvs}$<hr>{params}<hr>'

    @property
    def name(self) -> str:
        """Name of the model"""
        return self._name

    @property
    def dependent_variables(self) -> frozenmapping[Expr, int]:
        """The dependent variables of the model mapped to the corresponding DVIDs"""
        return self._dependent_variables

    @property
    def value_type(self) -> Union[str, Expr]:
        """The type of the model value (dependent variables)

        By default this is set to 'PREDICTION' to mean that the model outputs a prediction.
        It could optionally be set to 'LIKELIHOOD' or '-2LL' to let the model output the likelihood
        or -2*log(likelihood) of the prediction. If set to a symbol this variable can be used to
        change the type for different records. The model would then set this symbol to 0 for
        a prediction value, 1 for likelihood and 2 for -2ll.
        """
        return self._value_type

    @property
    def observation_transformation(self) -> frozenmapping[Expr, Expr]:
        """Transformation to be applied to the observation data"""
        return self._observation_transformation

    @property
    def parameters(self) -> Parameters:
        """Definitions of population parameters

        See :class:`pharmpy.Parameters`
        """
        return self._parameters

    @property
    def random_variables(self) -> RandomVariables:
        """Definitions of random variables

        See :class:`pharmpy.RandomVariables`
        """
        return self._random_variables

    @property
    def statements(self) -> Statements:
        """Definitions of model statements

        See :class:`pharmpy.Statements`
        """
        return self._statements

    @property
    def execution_steps(self) -> ExecutionSteps:
        """Definitions of estimation steps

        See :class:`pharmpy.ExecutionSteps`
        """
        return self._execution_steps

    @property
    def datainfo(self) -> DataInfo:
        """Definitions of model statements

        See :class:`pharmpy.Statements`
        """
        return self._datainfo

    @property
    def dataset(self) -> Optional[pd.DataFrame]:
        """Dataset connected to model"""
        return self._dataset

    @property
    def initial_individual_estimates(self) -> Optional[pd.DataFrame]:
        """Initial estimates for individual parameters"""
        return self._initial_individual_estimates

    @property
    def internals(self) -> Optional[ModelInternals]:
        """Internal data for tool specific part of model"""
        return self._internals

    @property
    def code(self) -> str:
        """Model type specific code"""
        d = self.to_dict()
        d['__magic__'] = "Pharmpy Model"
        d['__version__'] = pharmpy.__version__
        return json.dumps(d)

    def has_same_dataset_as(self, other: Model) -> bool:
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

        # NOTE: Rely on duck-typing here (?)
        return self.dataset.equals(other.dataset)

    @property
    def description(self) -> str:
        """A free text description of the model"""
        return self._description

    @staticmethod
    def parse_model(path: Union[Path, str], missing_data_token: Optional[str] = None):
        """Create a model object by parsing a model file of any supported type

        Parameters
        ----------
        path : Path or str
            Path to a model file
        missing_data_token : str
            Set to override the configuration

        Returns
        -------
        Model
            A model object
        """
        path = Path(path)
        with open(path, 'r', encoding='latin-1') as fp:
            code = fp.read()

        model_module = detect_model(code)
        model = model_module.parse_model(code, path, missing_data_token=missing_data_token)
        return model

    @staticmethod
    def parse_model_from_string(code: str) -> Model:
        """Create a model object by parsing a string with model code of any supported type

        Parameters
        ----------
        code : str
            Model code

        Returns
        -------
        Model
            A model object
        """
        model_module = detect_model(code)
        model = model_module.parse_model(code, None)
        return model

    def update_source(self) -> Model:
        """Update source code of the model. If any paths need to be changed or added (e.g. for a
        NONMEM model with an updated dataset) they will be replaced with DUMMYPATH"""
        return self

    def write_files(self, path: Optional[Union[Path, str]] = None, force: bool = False) -> Model:
        """Write all extra files needed for a specific external format."""
        return self


def update_datainfo(curdi: DataInfo, dataset: pd.DataFrame) -> DataInfo:
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

    # NOTE: Remove path if dataset has been updated
    return curdi if newdi == curdi else newdi.replace(path=None)


def get_and_check_odes(model: Model) -> CompartmentalSystem:
    """Get the ode_system from model and raise if not a CompartmentalSystem

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    CompartmentalSystem
        The compartmental system if it exists
    """

    odes = model.statements.ode_system
    if not isinstance(odes, CompartmentalSystem):
        raise ValueError(f'Model {model.name} has no ODE system')
    return odes


def get_and_check_dataset(model: Model) -> pd.DataFrame:
    """Get the ode_system from model and raise if not a CompartmentalSystem

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    pd.DataFrame
        The dataset if it exists
    """

    dataset = model.dataset
    if dataset is None:
        raise ValueError(f"Model {model.name} has no dataset")
    return dataset
