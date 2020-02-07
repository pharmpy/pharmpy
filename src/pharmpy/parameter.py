import pandas as pd
import sympy

from pharmpy.data_structures import OrderedSet


class ParameterSet(OrderedSet):
    """ ParameterSet

    Representing a group of parameters usually all parameters in a model.
    Even though parameters can directly be a part of the definitions of a
    model this class give a ways of displaying, summarizing and manipulating
    more than one parameter at a time.
    """
    def __getitem__(self, index):
        for e in self:
            if e.name == index:
                return e
        raise KeyError(f'Parameter "{index}" does not exist')

    def summary(self):
        """Construct a dataframe to summarize the Parameters
        """
        symbols = [param.name for param in self]
        values = [param.init for param in self]
        lower = [param.lower for param in self]
        upper = [param.upper for param in self]
        fix = [param.fix for param in self]
        return pd.DataFrame({'name': symbols, 'value': values,
                             'lower': lower, 'upper': upper, 'fix': fix})

    def __repr__(self):
        """View the parameters as a table.
        """
        if len(self) == 0:
            return "ParameterSet()"
        return self.summary().to_string(index=False)

    def _repr_html_(self):
        """For viewing in html capable environments
        """
        if len(self) == 0:
            return "ParameterSet()"
        else:
            return self.summary().to_html(index=False)


class Parameter:
    def __init__(self, name, init, lower=None, upper=None, fix=False):
        """A parameter

        Constraints are currently supported as lower and upper bounds and fix.
        Fix is regarded as constraining the parameter to one single value

        Properties: name, init, lower, upper and fix
        """
        self._init = init
        self.name = name
        self.lower = -sympy.oo
        self.upper = sympy.oo
        if fix:
            if lower is not None or upper is not None:
                raise ValueError('Cannot fix a parameter that has lower and upper bounds')
            self.lower = init
            self.upper = init
        if lower is not None:
            if lower > init:
                raise ValueError(f'Lower bound {lower} is greater than init {init}')
            self.lower = lower
        if upper is not None:
            if upper < init:
                raise ValueError(f'Upper bound {upper} is greater than init {init}')
            self.upper = upper

    @property
    def fix(self):
        """Is the parameter fixed?
        """
        return self.lower == self.upper

    @fix.setter
    def fix(self, value):
        if value:
            self.lower = self.init
            self.upper = self.init
        else:
            if self.fix:
                self.unconstrain()

    @property
    def init(self):
        """Initial parameter estimate
        """
        return self._init

    @init.setter
    def init(self, new_init):
        if new_init < self.lower or new_init > self.upper:
            raise ValueError(f'Initial estimate must be within the constraints of the parameter: '
                             f'{new_init} ∉ {sympy.pretty(sympy.Interval(self.lower, self.upper))}'
                             f'\nUnconstrain the parameter before setting an initial estimate.')
        self._init = new_init
        if self.fix:
            self.lower = new_init
            self.upper = new_init

    def unconstrain(self):
        """Remove all constraints of a parameter
        """
        self.lower = -sympy.oo
        self.upper = sympy.oo

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        """Two parameters are equal if they have the same symbol, init and constraints
        """
        return self.init == other.init and self.lower == other.lower and \
            self.upper == other.upper and self.name == other.name
