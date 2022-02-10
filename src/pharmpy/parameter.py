import copy
from collections.abc import MutableSequence

import numpy as np
import pandas as pd
import sympy

import pharmpy.symbols as symbols


class Parameters(MutableSequence):
    """A collection of parameters

    Class representing a group of parameters. Usually all parameters in a model.
    This class give a ways of displaying, summarizing and manipulating
    more than one parameter at a time.

    Specific parameters can be found using indexing on the parameter name

    Example
    -------

    >>> from pharmpy import Parameters, Parameter
    >>> par1 = Parameter("x", 0)
    >>> par2 = Parameter("y", 1)
    >>> pset = Parameters([par1, par2])
    >>> pset["x"]
    Parameter("x", 0, lower=-oo, upper=oo, fix=False)

    >>> "x" in pset
    True

    >>> pset.append(Parameter("z", 0.1))
    >>> len(pset)
    3

    >>> del pset["x"]
    >>> len(pset)
    2
    """

    def __init__(self, params=None):
        if isinstance(params, Parameters):
            self._params = copy.deepcopy(params._params)
        elif params is None:
            self._params = []
        else:
            self._params = list(params)
        names = set()
        for p in self._params:
            if not isinstance(p, Parameter):
                raise ValueError(f'Can not add variable of type {type(p)} to Parameters')
            if p.name in names:
                raise ValueError(
                    f'Parameter names must be unique. Parameter "{p.name}" '
                    'was added more than once to Parameters'
                )
            names.add(p.name)

    def __len__(self):
        return len(self._params)

    def _lookup_param(self, ind, insert=False):
        if isinstance(ind, sympy.Symbol):
            ind = ind.name
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
        if insert:
            # Must allow for inserting after last element.
            return ind, None
        else:
            return ind, self._params[ind]

    def __getitem__(self, ind):
        if isinstance(ind, list):
            params = []
            for i in ind:
                index, param = self._lookup_param(i)
                params.append(self[index])
            return Parameters(params)
        else:
            _, param = self._lookup_param(ind)
            return param

    def __setitem__(self, ind, value):
        if not isinstance(value, Parameter):
            raise ValueError(f"Can not set variable of type {type(value)} to Parameters.")
        i, rv = self._lookup_param(ind)
        if rv.name != value.name and value.name in self.names:
            raise ValueError(
                f"Cannot set parameter with already existing name {value.name} " "into Parameters."
            )
        self._params[i] = value

    def __delitem__(self, ind):
        i, _ = self._lookup_param(ind)
        del self._params[i]

    def insert(self, ind, value):
        if not isinstance(value, Parameter):
            raise ValueError(
                f'Trying to insert {type(value)} into Parameters. Must be of type Parameter.'
            )
        i, _ = self._lookup_param(ind, insert=True)
        if value.name in self.names:
            raise ValueError(
                f"Cannot insert parameter with already existing name {value.name} "
                "into Parameters."
            )
        self._params.insert(i, value)

    def __contains__(self, ind):
        try:
            _, _ = self._lookup_param(ind)
        except KeyError:
            return False
        return True

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
        >>> from pharmpy import Parameters, Parameter
        >>> par1 = Parameter("CL", 1, lower=0, upper=10)
        >>> par2 = Parameter("V", 10, lower=0, upper=100)
        >>> pset = Parameters([par1, par2])
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

    @property
    def nonfixed_inits(self):
        """Dict of initial estimates for all non-fixed parameters"""
        return {p.name: p.init for p in self._params if not p.fix}

    @inits.setter
    def inits(self, init_dict):
        for key, _ in init_dict.items():
            if key not in self:
                raise KeyError(f'Parameter {key} not in Parameters')
        [self[name].verify_init(value) for name, value in init_dict.items()]
        for name, value in init_dict.items():
            self[name].init = value

    @property
    def fix(self):
        """Fixedness of parameters as dict"""
        return {p.name: p.fix for p in self._params}

    @fix.setter
    def fix(self, fix_dict):
        for key in fix_dict:
            if key not in self:
                raise KeyError(f'Parameter {key} not in Parameters')
        for name, value in fix_dict.items():
            self[name].fix = value

    def remove_fixed(self):
        """Remove all fixed parameters"""
        nonfixed = [p for p in self._params if not p.fix]
        self._params = nonfixed

    def copy(self):
        """Create a deep copy of this Parameters"""
        return copy.deepcopy(self)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for p1, p2 in zip(self, other):
            if p1 != p2:
                return False
        return True

    def __repr__(self):
        if len(self) == 0:
            return "Parameters()"
        return self.to_dataframe().to_string()

    def _repr_html_(self):
        if len(self) == 0:
            return "Parameters()"
        else:
            return self.to_dataframe().to_html()


class Parameter:
    """A single parameter

    Example
    -------

    >>> from pharmpy import Parameter
    >>> param = Parameter("TVCL", 0.005, lower=0)
    >>> param.fix = True

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
    lower : number
        The lower bound of the parameter. Default no bound. Must be less than the init.
    upper : number
        The upper bound of the parameter. Default no bound. Must be greater than the init.
    """

    def __init__(self, name, init, lower=None, upper=None, fix=False):
        self._init = init
        self.name = name
        self.fix = bool(fix)
        self._lower = -sympy.oo
        self._upper = sympy.oo
        if lower is not None:
            self.lower = lower
        if upper is not None:
            self.upper = upper
        self.verify_init(self._init)

    @property
    def name(self):
        """Parameter name"""
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError("Name of parameter must be of type string")
        self._name = value

    @property
    def fix(self):
        """Should parameter be fixed or not"""
        return self._fix

    @fix.setter
    def fix(self, value):
        self._fix = bool(value)

    @property
    def symbol(self):
        """Symbol representing the parameter"""
        return symbols.symbol(self.name)

    @property
    def lower(self):
        """Lower bound of the parameter"""
        return self._lower

    @lower.setter
    def lower(self, new_lower):
        if new_lower > self.init:
            raise ValueError(f'Lower bound {new_lower} cannot be greater than init {self.init}')
        self._lower = new_lower

    @property
    def upper(self):
        """Upper bound of the parameter"""
        return self._upper

    @upper.setter
    def upper(self, new_upper):
        if new_upper < self.init:
            raise ValueError(f'Upper bound {new_upper} cannot be less than init {self.init}')
        self._upper = new_upper

    @property
    def init(self):
        """Initial parameter estimate or value"""
        return self._init

    @init.setter
    def init(self, new_init):
        self.verify_init(new_init)
        self._init = new_init

    def verify_init(self, new_init):
        """Verify initial estimate

        Verifies that an initial estimate is valid (i.e. is within bounds and is not NaN).

        Parameters
        ----------
        new_init : number
            value to to set as new initial estimate

        Examples
        --------
        >>> import numpy as np
        >>> from pharmpy import Parameter
        >>> par = Parameter("x", 1, lower=0, upper=10)
        >>> par.verify_init(5)
        >>> par.verify_init(11)
        Traceback (most recent call last):
        ...
        ValueError: Initial estimate must be within the constraints of the parameter: 11 ∉ [0, 10]
        Unconstrain the parameter before setting an initial estimate.

        >>> par.verify_init(np.nan)
        Traceback (most recent call last):
        ...
        ValueError: Initial estimate cannot be set to NaN
        """
        if new_init is sympy.nan or np.isnan(new_init):
            raise ValueError('Initial estimate cannot be set to NaN')
        if new_init < self.lower or new_init > self.upper:
            raise ValueError(
                f'Initial estimate must be within the constraints of the parameter: '
                f'{new_init} ∉ {sympy.pretty(sympy.Interval(self.lower, self.upper))}'
                f'\nUnconstrain the parameter before setting an initial estimate.'
            )

    def unconstrain(self):
        """Remove all constraints from this parameter

        Example
        -------
        >>> from pharmpy import Parameter
        >>> par = Parameter("x", 1, lower=0, upper=2)
        >>> par.unconstrain()
        >>> par
        Parameter("x", 1, lower=-oo, upper=oo, fix=False)
        """
        self._lower = -sympy.oo
        self._upper = sympy.oo
        self.fix = False

    @property
    def parameter_space(self):
        """The parameter space set

        A fixed parameter will be constrained to one single value
        and non-fixed parameters will be constrained to an interval
        possibly open in one or both ends.

        Examples
        --------

        >>> import sympy
        >>> from pharmpy import Parameter
        >>> par = Parameter("x", 1, lower=0, upper=10)
        >>> sympy.pprint(par.parameter_space)
        [0, 10]

        >>> par.fix = True
        >>> sympy.pprint(par.parameter_space)
        {1}
        """
        if self.fix:
            return sympy.FiniteSet(self.init)
        else:
            return sympy.Interval(self.lower, self.upper)

    def copy(self):
        """Create a copy of this Parameter"""
        return copy.deepcopy(self)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        """Two parameters are equal if they have the same name, init and constraints"""
        return (
            self.init == other.init
            and self.lower == other.lower
            and self.upper == other.upper
            and self.name == other.name
            and self.fix == other.fix
        )

    def __repr__(self):
        return (
            f'Parameter("{self.name}", {self.init}, lower={self.lower}, upper={self.upper}, '
            f'fix={self.fix})'
        )
