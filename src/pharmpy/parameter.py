import sympy


class Parameter:
    def __init__(self, name, init, lower=None, upper=None, fix=False):
        """A parameter

        constraints are supported as lower and upper bounds and fix.

        General constraints could easily be added whenever needed.
        """
        self.symbol = sympy.Symbol(name, real=True)
        self.init = init
        if fix:
            lower = init
            upper = init
        if lower:
            lower_expr = self.symbol => lower
        else:
            lower_expr = True
        if upper:
            upper_expr = self.symbol <= upper
        else:
            upper_expr = True
        self.constraints = sympy.And(lower_expr, upper_expr)

    @property
    def lower(self):
        """The lower bound of the parameter
        Actually the infimum of the domain of the parameter.
        """
        return self.constraints.as_set().inf

    @property
    def upper(self):
        """The upper bound of the parameter
        Actually the supremum of the domain of the parameter.
        """
        return self.constraints.as_set().sup

    @property
    def fix(self):
        """Is the parameter fixed?
        Actually checking if the domain of the parameter is finite with length 1. 
        """
        domain = self.constraints.as_set()
        return domain.is_FiniteSet and len(domain) == 1

    @property
    def init(self):
        """Initial parameter estimate
        """
        return self.init

    @init.setter
    def init(self, new_init):
        # Set init, upper and lower at same time? moving anyone the others need to move 
        self.init = new_init
        if self.fix:
            self.constraints = sympy.And(self.symbol <= new_init, self.symbol >= new_init)
