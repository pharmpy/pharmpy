import copy

import networkx as nx
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

    def subs(self, old, new):
        """Substitute old into new of rhs. Inplace
        """
        self.expression = self.expression.subs(old, new)

    @property
    def free_symbols(self):
        symbols = {self.symbol}
        symbols |= self.expression.free_symbols
        return symbols

    def __eq__(self, other):
        return self.symbol == other.symbol and self.expression == other.expression

    def pretty(self):
        expression = sympy.pretty(self.expression)
        lines = expression.split('\n')
        definition = f'{self.symbol} := '
        s = ''
        for line in lines:
            if line == lines[-1]:
                s += definition + line + '\n'
            else:
                s += len(definition) * ' ' + line + '\n'
        return s

    def __str__(self):
        return f'{self.symbol} := {self.expression}'


class ODESystem:
    """Base class and placeholder for ODE systems of different forms
    """
    @property
    def free_symbols(self):
        return set()

    def subs(self, old, new):
        pass

    def __eq__(self, other):
        return isinstance(other, ODESystem)

    def pretty(self):
        return str(self) + '\n'

    def __str__(self):
        return 'ODE-system-placeholder'


class ExplicitODESystem(ODESystem):
    """System of ODEs described explicitly
    """
    pass


class CompartmentalSystem(ODESystem):
    """System of ODEs descibed as a compartmental system
    """
    def __init__(self):
        self._g = nx.DiGraph()

    def __eq__(self, other):
        return nx.to_dict_of_dicts(self._g) == nx.to_dict_of_dicts(other._g)

    def __deepcopy__(self, memo):
        newone = type(self)()
        newone._g = copy.deepcopy(self._g, memo)
        return newone

    def add_compartment(self, name):
        comp = Compartment(name)
        self._g.add_node(comp)
        return comp

    def add_flow(self, source, destination, rate):
        self._g.add_edge(source, destination, rate=rate)

    def find_output(self):
        zeroout = [node for node, out_degree in self._g.out_degree_iter() if out_degree == 0]
        if len(zeroout) == 1:
            return zeroout[0]
        else:
            raise ValueError('More than one or zero output compartments')

    @property
    def compartmental_matrix(self):
        dod = nx.to_dict_of_dicts(self._g)
        size = len(dod) - 1
        f = sympy.zeros(size)
        for i in range(0, size):
            from_comp = list(self._g.nodes)[i]
            diagsum = 0
            for j in range(0, size + 1):
                to_comp = list(self._g.nodes)[j]
                try:
                    rate = dod[from_comp][to_comp]['rate']
                except KeyError:
                    rate = 0
                if to_comp is not None:
                    f[j, i] = rate
                    diagsum += f[j, i]
                else:
                    f[i, i] = -rate
            f[i, i] -= diagsum
        return f

    @property
    def amounts(self):
        amts = [node.amount for node in self._g.nodes if node is not None]
        return sympy.Matrix(amts)

    def to_explicit_odes(self):
        t = sympy.Symbol('t')
        amount_funcs = sympy.Matrix([sympy.Function(amt.name)('t') for amt in self.amounts])
        derivatives = sympy.Matrix([sympy.Derivative(fn, t) for fn in amount_funcs])
        a = self.compartmental_matrix @ amount_funcs
        eqs = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(derivatives, a)]
        ics = {}
        for node in self._g.nodes:
            if node is not None:
                if node.dose is None:
                    ics[sympy.Function(node.amount.name)(0)] = 0
                else:
                    ics[sympy.Function(node.amount.name)(0)] = node.dose.symbol
        return eqs, ics


class Compartment:
    def __init__(self, name):
        self.name = name
        self.dose = None

    def __eq__(self, other):
        return isinstance(other, Compartment) and self.name == other.name and \
            self.dose == other.dose

    def __hash__(self):
        return hash(self.name)

    @property
    def amount(self):
        return sympy.Symbol(f'A_{self.name}', real=True)


class IVBolus:
    def __init__(self, symbol):
        self.symbol = sympy.Symbol(symbol)

    def __eq__(self, other):
        return isinstance(other, IVBolus) and self.symbol == other.symbol


class ModelStatements(list):
    """A list of sympy statements describing the model
    """
    @property
    def free_symbols(self):
        """Get a set of all free symbols"""
        symbols = set()
        for assignment in self:
            symbols |= assignment.free_symbols
        return symbols

    def subs(self, old, new):
        """Substitute old expression for new in all rhs of assignments"""
        for assignment in self:
            assignment.subs(old, new)

    def find_assignment(self, symbol):
        """Returns full last statement given the symbol of an assignment"""
        statement = None
        for s in self:
            if isinstance(s, Assignment) and str(s.symbol) == symbol:
                statement = s
        return statement

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        else:
            for i in range(len(self)):
                if self[i] != other[i]:
                    return False
        return True

    def pretty(self):
        s = ''
        for statement in self:
            s += statement.pretty()
        return s

    def __str__(self):
        s = [str(assignment) for assignment in self]
        return '\n'.join(s)
