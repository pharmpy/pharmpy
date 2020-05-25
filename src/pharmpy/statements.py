import sympy


class Assignment:
    """Representation of variable assignment
       similar to sympy.codegen.Assignment
    """
    def __init__(self, symbol, expression):
        """ symbol can be either string or sympy symbol
            symbol can be a string and a real symbol will be created
        """
        try:
            symbol.is_Symbol
            self.symbol = symbol
        except AttributeError:
            self.symbol = sympy.Symbol(symbol, real=True)
        self.expression = expression

    def __str__(self):
        return f'{self.symbol} := {self.expression}'


class ModelStatements(list):
    """A list of sympy statements describing the model
    """
    @property
    def symbols(self):
        """Get a set of all used symbols"""
        symbols = set()
        for assignment in self:
            symbols |= {assignment.symbol}
            symbols |= assignment.expression.free_symbols
        return symbols

    def subs(self, old, new):
        """Substitute old expression for new in all rhs of assignments"""
        for assignment in self:
            assignment.expression = assignment.expression.subs(old, new)

    def __str__(self):
        s = [str(assignment) for assignment in self]
        return '\n'.join(s)
