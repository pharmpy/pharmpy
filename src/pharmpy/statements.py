import copy
import math

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
        return isinstance(other, Assignment) and self.symbol == other.symbol and \
            self.expression == other.expression

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
        return f'{self.symbol} := {self.expression}'.upper()


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


def _bracket(a):
    """Append a left bracket for an array of lines
    """
    if len(a) == 1:
        return '{' + a[0]
    if len(a) == 2:
        a.append('')
    if (len(a) % 2) == 0:
        upper = len(a) // 2 - 1
    else:
        upper = len(a) // 2
    a[0] = '⎧' + a[0]
    for i in range(1, upper):
        a[i] = '⎪' + a[i]
    a[upper] = '⎨' + a[upper]
    for i in range(upper + 1, len(a) - 1):
        a[i] = '⎪' + a[i]
    a[-1] = '⎩' + a[-1]
    return '\n'.join(a) + '\n'


class ExplicitODESystem(ODESystem):
    """System of ODEs described explicitly
    """
    def __init__(self, odes, ics):
        self.odes = odes
        self.ics = ics

    def pretty(self):
        a = []
        for ode in self.odes:
            ode_str = sympy.pretty(ode)
            a += ode_str.split('\n')
        for key, value in self.ics.items():
            ics_str = sympy.pretty(sympy.Eq(key, value))
            a += ics_str.split('\n')
        return _bracket(a)

    def __eq__(self, other):
        return isinstance(other, ExplicitODESystem) and self.odes == other.odes and \
            self.ics == other.ics


class CompartmentalSystem(ODESystem):
    """System of ODEs descibed as a compartmental system
    """
    def __init__(self):
        self._g = nx.DiGraph()

    def __eq__(self, other):
        return isinstance(other, CompartmentalSystem) and \
            nx.to_dict_of_dicts(self._g) == nx.to_dict_of_dicts(other._g)

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

    def get_flow(self, source, destination):
        return self._g.edges[source, destination]['rate']

    def find_output(self):
        zeroout = [node for node, out_degree in self._g.out_degree() if out_degree == 0]
        if len(zeroout) == 1:
            return zeroout[0]
        else:
            raise ValueError('More than one or zero output compartments')

    def find_central(self):
        output = self.find_output()
        central = next(self._g.predecessors(output))
        return central

    def find_peripherals(self):
        central = self.find_central()
        oneout = {node for node, out_degree in self._g.out_degree() if out_degree == 1}
        onein = {node for node, in_degree in self._g.in_degree() if in_degree == 1}
        peripherals = (oneout & onein) - {central}
        return list(peripherals)

    def find_depot(self):
        central = self.find_central()
        zeroin = [node for node, in_degree in self._g.in_degree() if in_degree == 0]
        if len(zeroin) == 1 and zeroin[0] != central:
            return zeroin[0]
        else:
            return None

    @property
    def compartmental_matrix(self):
        dod = nx.to_dict_of_dicts(self._g)
        size = len(self._g.nodes)
        f = sympy.zeros(size)
        for i in range(0, size):
            from_comp = list(self._g.nodes)[i]
            diagsum = 0
            for j in range(0, size):
                to_comp = list(self._g.nodes)[j]
                try:
                    rate = dod[from_comp][to_comp]['rate']
                except KeyError:
                    rate = 0
                f[j, i] = rate
                diagsum += f[j, i]
            f[i, i] -= diagsum
        return f

    @property
    def amounts(self):
        amts = [node.amount for node in self._g.nodes]
        return sympy.Matrix(amts)

    def to_explicit_odes(self):
        t = sympy.Symbol('t')
        amount_funcs = sympy.Matrix([sympy.Function(amt.name)('t') for amt in self.amounts])
        derivatives = sympy.Matrix([sympy.Derivative(fn, t) for fn in amount_funcs])
        a = self.compartmental_matrix @ amount_funcs
        eqs = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(derivatives, a)]
        ics = {}
        for node in self._g.nodes:
            if node.dose is None:
                ics[sympy.Function(node.amount.name)(0)] = 0
            else:
                ics[sympy.Function(node.amount.name)(0)] = node.dose.symbol
        return eqs, ics

    def pretty(self):
        output = self.find_output()
        output_box = box(output.name)
        central = self.find_central()
        central_box = box(central.name)
        depot = self.find_depot()
        if depot:
            depot_box = box(depot.name)
            depot_central_arrow = arrow(str(self.get_flow(depot, central)))
        periphs = self.find_peripherals()
        periph_box = []
        for p in periphs:
            periph_box.append(box(p.name))

        upper = []
        if periphs:
            upper += box(periphs[0].name)
            up_arrow = vertical_arrow(str(self.get_flow(central, periphs[0])), down=False)
            down_arrow = vertical_arrow(str(self.get_flow(periphs[0], central)))
            for i in range(0, len(up_arrow)):
                upper.append(up_arrow[i] + '  ' + down_arrow[i])

        bottom = []
        central_output_arrow = arrow(str(self.get_flow(central, output)))
        for i in range(0, len(output_box)):
            if i == 1:
                flow = central_output_arrow
            else:
                flow = ' ' * len(central_output_arrow)
            if depot:
                if i == 1:
                    ab = depot_central_arrow
                else:
                    ab = ' ' * len(depot_central_arrow)
                curdepot = depot_box[i] + ab
            else:
                curdepot = ''

            bottom.append(curdepot + central_box[i] + flow + output_box[i])

        upper_str = ''
        if upper:
            if depot:
                pad = ' ' * (len(depot_box[0]) + len(depot_central_arrow))
            else:
                pad = ''
            for line in upper:
                upper_str += pad + line + '\n'
        return upper_str + '\n'.join(bottom) + '\n'


def box(s):
    """Draw unicode box around string and return new string
    """
    upper = '┌' + '─' * len(s) + '┐'
    mid = '│' + s + '│'
    lower = '└' + '─' * len(s) + '┘'
    return [upper, mid, lower]


def arrow(flow, right=True):
    if right:
        return '─' * 2 + flow + '→'
    else:
        return '←' + flow + '─' * 2


def vertical_arrow(flow, down=True):
    n = len(flow) / 2
    before = ' ' * math.floor(n)
    after = ' ' * math.ceil(n)
    if down:
        return [before + '│' + after, flow, before + '↓' + after]
    else:
        return [before + '↑' + after,  flow, before + '│' + after]


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


class Bolus:
    def __init__(self, symbol):
        self.symbol = sympy.Symbol(symbol)

    def __eq__(self, other):
        return isinstance(other, Bolus) and self.symbol == other.symbol


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

    @property
    def ode_system(self):
        """Returns the ODE system of the model or None if the model doesn't have an ODE system
        """
        for s in self:
            if isinstance(s, ODESystem):
                return s
        return None

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
