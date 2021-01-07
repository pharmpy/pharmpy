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

    .. code-block:: python

        params['THETA(1)']
    """

    def __getitem__(self, index):
        for e in self:
            if e.name == index:
                return e
        raise KeyError(f'Parameter "{index}" does not exist')

    def __contains__(self, item):
        for e in self:
            if e.name == item:
                return True
        return False

    def summary(self):
        """Give a dataframe with a summary of all Parameters

        :returns: A dataframe with one row per parameter.
                  The columns are value, lower, upper and fix
                  Row Index is the names
        """
        symbols = [param.name for param in self]
        values = [param.init for param in self]
        lower = [param.lower for param in self]
        upper = [param.upper for param in self]
        fix = [param.fix for param in self]
        return pd.DataFrame(
            {'value': values, 'lower': lower, 'upper': upper, 'fix': fix}, index=symbols
        )

    def values_near_bounds(self, values, zero_limit, significant_digits):
        """Logical Series of whether values are near the respective boundaries
        values : Series of floats with index a subset of parameter names
        :returns: Logical Series with same index as values
        """
        return pd.Series(
            [
                self[p].any_boundary_near_value(values.loc[p], zero_limit, significant_digits)
                for p in values.index
            ],
            index=values.index,
        )

    @property
    def names(self):
        """Names of all parameters"""
        return [p.name for p in self]

    @property
    def symbols(self):
        """Symbols of all parameters"""
        return [p.symbol for p in self]

    @property
    def lower(self):
        """Lower bounds of all parameters"""
        return {p.name: p.lower for p in self}

    @property
    def upper(self):
        """Upper bounds of all parameters"""
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
        for name, value in init_dict.items():
            self[name].init = value

    @property
    def fix(self):
        """Fixedness of parameters as dict"""
        return {p.name: p.fix for p in self}

    @fix.setter
    def fix(self, fix_dict):
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
        return self.summary().to_string()

    def _repr_html_(self):
        if len(self) == 0:
            return "ParameterSet()"
        else:
            return self.summary().to_html()


class Parameter:
    """A single parameter

    .. code-block::

        param = Parameter("TVCL", 0.005, lower=0)
        param.fix = True

    Parameters
    ----------
    name : str
        Name of the parameter
    fix : bool
        A boolean to indicate whether the parameter is fixed or not. Note that fixing a parameter
        will keep its bounds even if a fixed parameter is actually constrained to one single
        value. This is so that unfixing will take back the previous bounds.
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
        """Initial parameter estimate"""
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

    def any_boundary_near_value(self, value, zero_limit, significant_digits):
        """Is any boundary near this value"""
        return (
            bool(self.lower > -sympy.oo)
            and is_near_target(value, self.lower, zero_limit, significant_digits)
        ) or (
            bool(self.upper < sympy.oo)
            and is_near_target(value, self.upper, zero_limit, significant_digits)
        )

    def unconstrain(self):
        """Remove all constraints of a parameter"""
        self._lower = -sympy.oo
        self._upper = sympy.oo
        self.fix = False

    @property
    def parameter_space(self):
        """The parameter space set

        A fixed parameter will be constrained to one single value
        and non-fixed parameters will be constrained to an interval
        possibly open in one or both ends.
        """
        if self.fix:
            return sympy.FiniteSet(self.init)
        else:
            return sympy.Interval(self.lower, self.upper)

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
