import copy

import pandas as pd
import sympy

import pharmpy.symbols as symbols
from pharmpy.data_structures import OrderedSet
from pharmpy.math import is_near_target


class ParameterSet(OrderedSet):
    """A set of parameters

    Class representing a group of parameters. Usually all parameters in a model.
    This class give a ways of displaying, summarizing and manipulating
    more than one parameter at a time.

    Specific parameters can be found using indexing on the parameter name

    Example
    -------

    >>> from pharmpy import ParameterSet, Parameter
    >>> par1 = Parameter("x", 0)
    >>> par2 = Parameter("y", 1)
    >>> pset = ParameterSet([par1, par2])
    >>> pset["x"]
    Parameter("x", 0, lower=-oo, upper=oo, fix=False)

    >>> "x" in pset
    True

    >>> pset.add(Parameter("z", 0.1))
    >>> len(pset)
    3

    >>> pset.remove("x")
    >>> len(pset)
    2
    """

    def add(self, key):
        if not isinstance(key, Parameter):
            raise ValueError(f"Can not add variable of type {type(key)} to ParameterSet.")
        super().add(key)

    def discard(self, key):
        if isinstance(key, str) or isinstance(key, sympy.Symbol):
            if key in self:
                key = self[key]
        super().discard(key)

    def __getitem__(self, index):
        for e in self:
            if e.name == index or e.symbol == index:
                return e
        raise KeyError(f'Parameter "{index}" does not exist')

    def __contains__(self, item):
        for e in self:
            if (
                isinstance(item, str)
                and e.name == item
                or isinstance(item, sympy.Symbol)
                and e.name == item.name
                or isinstance(item, Parameter)
                and e == item
            ):
                return True
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
        >>> from pharmpy import ParameterSet, Parameter
        >>> par1 = Parameter("CL", 1, lower=0, upper=10)
        >>> par2 = Parameter("V", 10, lower=0, upper=100)
        >>> pset = ParameterSet([par1, par2])
        >>> pset.to_dataframe()

            value  lower  upper    fix
        CL      1      0     10  False
        V      10      0    100  False
        """
        symbols = [param.name for param in self]
        values = [param.init for param in self]
        lower = [param.lower for param in self]
        upper = [param.upper for param in self]
        fix = [param.fix for param in self]
        return pd.DataFrame(
            {'value': values, 'lower': lower, 'upper': upper, 'fix': fix}, index=symbols
        )

    def is_close_to_bound(self, values=None, zero_limit=0.01, significant_digits=2):
        """Logical Series of whether values are close to the respective bounds

        Parameters
        ----------
        values : pd.Series
            Series of values with index a subset of parameter names.
            Default is to use all parameter inits

        Returns
        -------
        pd.Series
            Logical Series with same index as values

        Example
        -------
        >>> from pharmpy import ParameterSet, Parameter
        >>> par1 = Parameter("CL", 1, lower=0, upper=10)
        >>> par2 = Parameter("V", 10, lower=0, upper=100)
        >>> pset = ParameterSet([par1, par2])
        >>> pset.is_close_to_bound()
        CL    False
        V     False
        dtype: bool
        """
        if values is None:
            values = pd.Series(self.inits)
        ser = pd.Series(
            [
                self[p].is_close_to_bound(values.loc[p], zero_limit, significant_digits)
                for p in values.index
            ],
            index=values.index,
            dtype=bool,
        )
        return ser

    @property
    def names(self):
        """List of all parameter names"""
        return [p.name for p in self]

    @property
    def symbols(self):
        """List of all parameter symbols"""
        return [p.symbol for p in self]

    @property
    def lower(self):
        """Lower bounds of all parameters as a dictionary"""
        return {p.name: p.lower for p in self}

    @property
    def upper(self):
        """Upper bounds of all parameters as a dictionary"""
        return {p.name: p.upper for p in self}

    @property
    def inits(self):
        """Initial estimates of parameters as dict"""
        return {p.name: p.init for p in self}

    @property
    def nonfixed_inits(self):
        """Dict of initial estimates for all non-fixed parameters"""
        return {p.name: p.init for p in self if not p.fix}

    @inits.setter
    def inits(self, init_dict):
        for key, _ in init_dict.items():
            if key not in self:
                raise KeyError(f'Parameter {key} not in ParameterSet')
        for name, value in init_dict.items():
            self[name].init = value

    @property
    def fix(self):
        """Fixedness of parameters as dict"""
        return {p.name: p.fix for p in self}

    @fix.setter
    def fix(self, fix_dict):
        for key in fix_dict:
            if key not in self:
                raise KeyError(f'Parameter {key} not in ParameterSet')
        for name, value in fix_dict.items():
            self[name].fix = value

    def remove_fixed(self):
        """Remove all fixed parameters"""
        fixed = [p for p in self if p.fix]
        self -= fixed

    def copy(self):
        """Create a deep copy of this ParameterSet"""
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
            return "ParameterSet()"
        return self.to_dataframe().to_string()

    def _repr_html_(self):
        if len(self) == 0:
            return "ParameterSet()"
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
        if new_init < self.lower or new_init > self.upper:
            raise ValueError(
                f'Initial estimate must be within the constraints of the parameter: '
                f'{new_init} ∉ {sympy.pretty(sympy.Interval(self.lower, self.upper))}'
                f'\nUnconstrain the parameter before setting an initial estimate.'
            )
        self._init = new_init

    def is_close_to_bound(self, value=None, zero_limit=0.01, significant_digits=2):
        """Check if parameter value is close to any bound

        Parameters
        ----------
        value : number
            value to check against parameter bounds. Defaults to checking against the parameter init
        zero_limit : number
            maximum distance to 0 bounds
        significant_digits : int
            maximum distance to non-zero bounds in number of significant digits

        Examples
        --------

        >>> from pharmpy import Parameter
        >>> par = Parameter("x", 1, lower=0, upper=10)
        >>> par.is_close_to_bound()
        False

        >>> par.is_close_to_bound(0.005)
        True

        >>> par.is_close_to_bound(0.005, zero_limit=0.0001)
        False

        >>> par.is_close_to_bound(9.99)
        True

        >>> par.is_close_to_bound(9.99, significant_digits=3)
        False
        """
        if value is None:
            value = self.init
        return (
            self.lower > -sympy.oo
            and is_near_target(value, self.lower, zero_limit, significant_digits)
        ) or (
            self.upper < sympy.oo
            and is_near_target(value, self.upper, zero_limit, significant_digits)
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
