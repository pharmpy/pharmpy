from __future__ import annotations

from collections.abc import Sequence as CollectionsSequence
from typing import Any, Optional, Sequence, Union, overload

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.immutable import Immutable


class Parameter(Immutable):
    """A single parameter

    Example
    -------

    >>> from pharmpy.model import Parameter
    >>> param = Parameter("TVCL", 0.005, lower=0.0)
    >>> param.init
    0.005

    Parameters
    ----------
    name : str
        Name of the parameter
    init : number
        Initial estimate or simply the value of parameter.
    fix : bool
        A boolean to indicate whether the parameter is fixed or not. Note that fixing a parameter
        will keep its bounds even if a fixed parameter is actually constrained to one single
        value. This is so that unfixing will take back the previous bounds.
    lower : float
        The lower bound of the parameter. Default no bound. Must be less than the init.
    upper : float
        The upper bound of the parameter. Default no bound. Must be greater than the init.
    """

    def __init__(
        self,
        name: str,
        init: float,
        lower: float = -float("inf"),
        upper: float = float("inf"),
        fix: bool = False,
    ):
        self._name = name
        self._init = init
        self._lower = lower
        self._upper = upper
        self._fix = fix

    @classmethod
    def create(
        cls,
        name: str,
        init: Union[float, sympy.Float],
        lower: Optional[Union[float, sympy.Float]] = None,
        upper: Optional[Union[float, sympy.Float]] = None,
        fix: Any = False,
    ):
        """Alternative constructor for Parameter with error checking"""
        if not isinstance(name, str):
            raise ValueError("Name of parameter must be of type string")
        if init is sympy.nan or np.isnan(init):
            raise ValueError('Initial estimate cannot be NaN')
        if lower is None:
            lower = -float('inf')
        else:
            lower = float(lower)
        if upper is None:
            upper = float('inf')
        else:
            upper = float(upper)
        if init < lower:
            raise ValueError(f'Lower bound {lower} cannot be greater than init {init}')
        if init > upper:
            raise ValueError(f'Upper bound {upper} cannot be less than init {init}')
        return cls(name, init, lower, upper, bool(fix))

    def replace(self, **kwargs):
        """Replace properties and create a new Parameter"""
        name = kwargs.get('name', self._name)
        init = kwargs.get('init', self._init)
        lower = kwargs.get('lower', self._lower)
        upper = kwargs.get('upper', self._upper)
        fix = kwargs.get('fix', self._fix)
        new = Parameter.create(name, init, lower=lower, upper=upper, fix=fix)
        return new

    @property
    def name(self):
        """Parameter name"""
        return self._name

    @property
    def fix(self):
        """Should parameter be fixed or not"""
        return self._fix

    @property
    def symbol(self):
        """Symbol representing the parameter"""
        return sympy.Symbol(self._name)

    @property
    def lower(self):
        """Lower bound of the parameter"""
        return self._lower

    @property
    def upper(self):
        """Upper bound of the parameter"""
        return self._upper

    @property
    def init(self):
        """Initial parameter estimate or value"""
        return self._init

    def __hash__(self):
        return hash((self.name, self.init, self.lower, self.upper, self.fix))

    def to_dict(self):
        return {
            'name': self.name,
            'init': self.init,
            'lower': self.lower,
            'upper': self.upper,
            'fix': self.fix,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def __eq__(self, other):
        """Two parameters are equal if they have the same name, init and constraints"""
        return (
            isinstance(other, Parameter)
            and self._init == other._init
            and self._lower == other._lower
            and self._upper == other._upper
            and self._name == other._name
            and self._fix == other._fix
        )

    def __repr__(self):
        if self._lower == -float("inf"):
            lower = "-∞"
        else:
            lower = self._lower
        if self._upper == float("inf"):
            upper = "∞"
        else:
            upper = self._upper
        return (
            f'Parameter("{self._name}", {self._init}, lower={lower}, upper={upper}, '
            f'fix={self._fix})'
        )


class Parameters(CollectionsSequence, Immutable):
    """An immutable collection of parameters

    Class representing a group of parameters. Usually all parameters in a model.
    This class give a ways of displaying, summarizing and manipulating
    more than one parameter at a time.

    Specific parameters can be found using indexing on the parameter name

    Example
    -------

    >>> from pharmpy.model import Parameters, Parameter
    >>> par1 = Parameter("x", 0)
    >>> par2 = Parameter("y", 1)
    >>> pset = Parameters((par1, par2))
    >>> pset["x"]
    Parameter("x", 0, lower=-∞, upper=∞, fix=False)

    >>> "x" in pset
    True

    """

    def __init__(self, parameters=()):
        self._params = parameters

    @classmethod
    def create(cls, parameters=None):
        if isinstance(parameters, Parameters):
            pass
        elif parameters is None:
            parameters = ()
        else:
            parameters = tuple(parameters)
        names = set()
        for p in parameters:
            if not isinstance(p, Parameter):
                raise ValueError(f'Can not add variable of type {type(p)} to Parameters')
            if p.name in names:
                raise ValueError(
                    f'Parameter names must be unique. Parameter "{p.name}" '
                    'was added more than once to Parameters'
                )
            names.add(p.name)
        return cls(parameters)

    def replace(self, **kwargs):
        """Replace properties and create a new Parameters object"""
        parameters = kwargs.get('parameters', self._params)
        new = Parameters.create(parameters)
        return new

    def __len__(self):
        return len(self._params)

    def _lookup_param(self, ind: Union[int, str, sympy.Symbol, Parameter]):
        if isinstance(ind, sympy.Symbol):
            ind = ind.name  # pyright: ignore [reportGeneralTypeIssues]
        if isinstance(ind, str):
            for i, param in enumerate(self._params):
                if ind == param.name:
                    return i, param
            raise KeyError(f'Could not find {ind} in Parameters')
        elif isinstance(ind, Parameter):
            try:
                i = self._params.index(ind)
            except ValueError:
                raise KeyError(f'Could not find {ind.name} in Parameters')
            return i, ind
        return ind, self._params[ind]

    @overload
    def __getitem__(self, ind: Union[int, str, sympy.Symbol, Parameter]) -> Parameter:
        ...

    @overload
    def __getitem__(
        self, ind: Union[slice, Sequence[Union[int, str, sympy.Symbol, Parameter]]]
    ) -> Parameters:
        ...

    def __getitem__(self, ind):
        if isinstance(ind, slice):
            return Parameters.create(self._params[ind])
        elif not isinstance(ind, str) and isinstance(ind, CollectionsSequence):
            return Parameters.create((self._lookup_param(i)[1] for i in ind))
        else:
            _, param = self._lookup_param(ind)
            return param

    def __contains__(self, ind):
        try:
            self._lookup_param(ind)
            return True
        except KeyError:
            return False

    def to_dataframe(self):
        """Create a dataframe with a summary of all Parameters

        Returns
        -------
        DataFrame
            A dataframe with one row per parameter.
            The columns are value, lower, upper and fix
            Row Index is the names

        Example
        -------
        >>> from pharmpy.model import Parameters, Parameter
        >>> par1 = Parameter("CL", 1, lower=0, upper=10)
        >>> par2 = Parameter("V", 10, lower=0, upper=100)
        >>> pset = Parameters((par1, par2))
        >>> pset.to_dataframe()
            value  lower  upper    fix
        CL      1      0     10  False
        V      10      0    100  False

        """
        symbols = [param.name for param in self._params]
        values = [param.init for param in self._params]
        lower = [param.lower for param in self._params]
        upper = [param.upper for param in self._params]
        fix = [param.fix for param in self._params]
        return pd.DataFrame(
            {'value': values, 'lower': lower, 'upper': upper, 'fix': fix}, index=symbols
        )

    @property
    def names(self):
        """List of all parameter names"""
        return [p.name for p in self._params]

    @property
    def symbols(self):
        """List of all parameter symbols"""
        return [p.symbol for p in self._params]

    @property
    def lower(self):
        """Lower bounds of all parameters as a dictionary"""
        return {p.name: p.lower for p in self._params}

    @property
    def upper(self):
        """Upper bounds of all parameters as a dictionary"""
        return {p.name: p.upper for p in self._params}

    @property
    def inits(self):
        """Initial estimates of parameters as dict"""
        return {p.name: p.init for p in self._params}

    def set_initial_estimates(self, inits):
        """Create a new Parameters with changed initial estimates

        Parameters
        ----------
        inits : dict
            A dictionary of parameter names to initial estimates

        Return
        ------
        Parameters
            An update Parameters object
        """
        new = []
        for p in self:
            if p.name in inits:
                newparam = p.replace(init=inits[p.name])
            else:
                newparam = p
            new.append(newparam)
        return Parameters(tuple(new))

    @property
    def fix(self):
        """Fixedness of parameters as dict"""
        return {p.name: p.fix for p in self._params}

    def set_fix(self, fix):
        """Create a new Parameters with changed fix state

        Parameters
        ----------
        fix : dict
            A dictionary of parameter names to boolean fix state

        Return
        ------
        Parameters
            An update Parameters object
        """
        new = []
        for p in self:
            if p.name in fix:
                newparam = p.replace(fix=fix[p.name])
            else:
                newparam = p
            new.append(newparam)
        return Parameters(tuple(new))

    @property
    def fixed(self):
        """All fixed parameters"""
        fixed = [p for p in self._params if p.fix]
        return Parameters(tuple(fixed))

    @property
    def nonfixed(self):
        """All non-fixed parameters"""
        nonfixed = [p for p in self._params if not p.fix]
        return Parameters(tuple(nonfixed))

    def __add__(self, other):
        if isinstance(other, Parameter):
            return Parameters.create(self._params + (other,))
        elif isinstance(other, Parameters):
            return Parameters.create(self._params + other._params)
        elif isinstance(other, Sequence):
            return Parameters.create(self._params + tuple(other))
        else:
            raise ValueError(f"Cannot add {other} to Parameters")

    def __radd__(self, other):
        if isinstance(other, Parameter):
            return Parameters.create((other,) + self._params)
        elif isinstance(other, Sequence):
            return Parameters.create(tuple(other) + self._params)
        else:
            raise ValueError(f"Cannot add {other} to Parameters")

    def __eq__(self, other):
        if not isinstance(other, Parameters):
            return False
        if len(self) != len(other):
            return False
        for p1, p2 in zip(self, other):
            if p1 != p2:
                return False
        return True

    def __hash__(self):
        return hash(self._params)

    def to_dict(self):
        params = tuple(p.to_dict() for p in self._params)
        return {'parameters': params}

    @classmethod
    def from_dict(cls, d):
        params = tuple(Parameter.from_dict(p) for p in d['parameters'])
        return cls(parameters=params)

    def __repr__(self):
        if len(self) == 0:
            return "Parameters()"
        return (
            self.to_dataframe().replace(float("inf"), "∞").replace(-float("inf"), "-∞").to_string()
        )

    def _repr_html_(self):
        if len(self) == 0:
            return "Parameters()"
        else:
            return (
                self.to_dataframe()
                .replace(float("inf"), "∞")
                .replace(-float("inf"), "-∞")
                .to_html()
            )
