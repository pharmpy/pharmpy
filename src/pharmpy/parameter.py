import pandas as pd
import sympy

from pharmpy.data_structures import OrderedSet


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
        return pd.DataFrame({'value': values, 'lower': lower, 'upper': upper, 'fix': fix},
                            index=symbols)

    def remove_fixed(self):
        """Remove all fixed parameters
        """
        fixed = [p for p in self if p.fix]
        self -= fixed

    def update_inits(self, inits):
        """Update the initial estimates of some or all parameters

           All parameters of inits must already be in the set

           :param inits: A dict of parameter names to new initial estimates
        """
        for name, value in inits.items():
            self[name].init = value

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
       Constraints are currently supported as lower and upper bounds and fix.
       Fix is regarded as constraining the parameter to one single value

        .. code-block::

            param = Parameter("TVCL", 0.005, lower=0)

       .. attribute:: name

           Name of the parameter

    """
    def __init__(self, name, init, lower=None, upper=None, fix=False):
        self._init = init
        self.name = name
        self._lower = -sympy.oo
        self._upper = sympy.oo
        if fix:
            if lower is not None or upper is not None:
                raise ValueError('Cannot fix a parameter that has lower and upper bounds')
            self._lower = init
            self._upper = init
        if lower is not None:
            self.lower = lower
        if upper is not None:
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
    def lower(self):
        """Lower bound of the parameter
        """
        return self._lower

    @lower.setter
    def lower(self, new_lower):
        if new_lower > self.init:
            raise ValueError(f'Lower bound {new_lower} cannot be greater than init {self.init}')
        self._lower = new_lower

    @property
    def upper(self):
        """Upper bound of the parameter
        """
        return self._upper

    @upper.setter
    def upper(self, new_upper):
        if new_upper < self.init:
            raise ValueError(f'Upper bound {new_upper} cannot be less than init {self.init}')
        self._upper = new_upper

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
        self._lower = -sympy.oo
        self._upper = sympy.oo

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        """Two parameters are equal if they have the same name, init and constraints
        """
        return self.init == other.init and self.lower == other.lower and \
            self.upper == other.upper and self.name == other.name
