import sympy

from pharmpy.data_structures import OrderedSet


class ParameterSet(OrderedSet):
    def __getitem__(self, index):
        for e in self:
            if e.symbol == index or e.symbol.name == index:
                return e
        raise KeyError(f'Parameter "{index}" does not exist')


class Parameter:
    def __init__(self, name, init, lower=None, upper=None, fix=False):
        """A parameter

        constraints are currently supported as lower and upper bounds and fix.
        Fix is regarded as constraining the parameter to one single value
        i.e. the domain is a finite set with one member

        General constraints could easily be added whenever needed.
        Properties: symbol, constraints (appart from those with setters and getters).
            FIXME: These should be read only.
        """
        self._symbol = sympy.Symbol(name, real=True)
        self._init = init
        if fix:
            if lower is not None or upper is not None:
                raise ValueError('Cannot fix a parameter that has lower and upper bounds')
            lower = init
            upper = init
        if lower is not None:
            if lower > init:
                raise ValueError(f'Lower bound {lower} is greater than init {init}')
            lower_expr = self._symbol >= lower
        else:
            lower_expr = self._symbol >= -sympy.oo
        if upper is not None:
            if upper < init:
                raise ValueError(f'Upper bound {upper} is greater than init {init}')
            upper_expr = self._symbol <= upper
        else:
            upper_expr = self._symbol <= sympy.oo
        self._constraints = sympy.And(lower_expr, upper_expr)

    @property
    def symbol(self):
        return self._symbol

    @property
    def constraints(self):
        return self._constraints

    @property
    def lower(self):
        """The lower bound of the parameter
        Actually the infimum of the domain of the parameter.
        """
        return self._constraints.as_set().inf

    @property
    def upper(self):
        """The upper bound of the parameter
        Actually the supremum of the domain of the parameter.
        """
        return self._constraints.as_set().sup

    @property
    def fix(self):
        """Is the parameter fixed?
        Actually checking if the domain of the parameter is finite with length 1. 
        """
        domain = self._constraints.as_set()
        return domain.is_FiniteSet and len(domain) == 1

    @fix.setter
    def fix(self, value):
        if value:
            self._constraints = sympy.And(self.symbol <= self._init, self.symbol >= self._init)
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
        if new_init not in self._constraints.as_set():
            raise ValueError(f'Initial estimate must be within the constraints of the parameter: {new_init} ∉ {sympy.pretty(self._constraints.as_set())}\nUnconstrain the parameter before setting an initial estimate.')
        self._init = new_init
        if self.fix:
            self._constraints = sympy.And(self.symbol <= new_init, self.symbol >= new_init)

    def unconstrain(self):
        """Remove all constraints of a parameter
        """
        self._constraints = sympy.And(self.symbol <= sympy.oo, self.symbol >= -sympy.oo)

    def __hash__(self):
        return hash(self._symbol.name)

    def __eq__(self, other):
        """Two parameters are equal if they have the same symbol.name, init and domain
        """
        return self.symbol.name == other.symbol.name and self.init == other.init and self.constraints.as_set() == other.constraints.as_set()

    def __str__(self):
        return self._symbol.name

    def __repr__(self):
        return f"Parameter('{self.symbol.name}', {self.init}, lower={self.lower}, upper={self.upper}, fix={self.fix})"
