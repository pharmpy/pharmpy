import copy
import math

import networkx as nx
import sympy

import pharmpy.symbols as symbols


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
            self.symbol = symbols.real(symbol)
        self.expression = expression

    def subs(self, substitutions):
        """Substitute old into new of rhs. Inplace
        """
        self.symbol = symbols.subs(self.symbol, substitutions)
        self.expression = symbols.subs(self.expression, substitutions)

    @property
    def free_symbols(self):
        symbols = {self.symbol}
        symbols |= self.expression.free_symbols
        return symbols

    @property
    def rhs_symbols(self):
        return self.expression.free_symbols

    def __eq__(self, other):
        return isinstance(other, Assignment) and self.symbol == other.symbol and \
            self.expression == other.expression

    def __str__(self):
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

    def __deepcopy__(self, memo):
        return type(self)(self.symbol, self.expression)

    def __repr__(self):
        return f'{self.symbol} := {self.expression}'.upper()


class ODESystem:
    """Base class and placeholder for ODE systems of different forms
    """
    @property
    def free_symbols(self):
        return set()

    @property
    def rhs_symbols(self):
        return set()

    def subs(self, substitutions):
        pass

    def __eq__(self, other):
        return isinstance(other, ODESystem)

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

    @property
    def free_symbols(self):
        free = set()
        for ode in self.odes:
            free |= ode.free_symbols
        for key, value in self.ics.items():
            free |= key.free_symbols
            try:        # To allow for regular python classes as values for ics
                free |= value.free_symbols
            except AttributeError:
                pass
        return free

    @property
    def rhs_symbols(self):
        return self.free_symbols        # This works currently

    def __str__(self):
        a = []
        for ode in self.odes:
            ode_str = sympy.pretty(ode)
            a += ode_str.split('\n')
        for key, value in self.ics.items():
            ics_str = sympy.pretty(sympy.Eq(key, value))
            a += ics_str.split('\n')
        return _bracket(a)

    def __deepcopy__(self, memo):
        newone = type(self)(copy.copy(self.odes), copy.copy(self.ics))
        return newone

    def __eq__(self, other):
        return isinstance(other, ExplicitODESystem) and self.odes == other.odes and \
            self.ics == other.ics


class CompartmentalSystem(ODESystem):
    """System of ODEs descibed as a compartmental system
    """
    def __init__(self):
        self._g = nx.DiGraph()

    def rename_symbol(self, substitutions):
        pass

    @property
    def free_symbols(self):
        free = {symbols.real('t')}
        for (_, _, rate) in self._g.edges.data('rate'):
            free |= rate.free_symbols
        for node in self._g.nodes:
            free |= node.free_symbols
        return free

    @property
    def rhs_symbols(self):
        return self.free_symbols        # This works currently

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

    def remove_compartment(self, compartment):
        self._g.remove_node(compartment)

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
        t = symbols.real('t')
        amount_funcs = sympy.Matrix([sympy.Function(amt.name)(t) for amt in self.amounts])
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

    def __len__(self):
        """Get the number of compartments including output
        """
        return len(self._g.nodes)

    def __str__(self):
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

        lower = []
        if periphs:
            down_arrow = vertical_arrow(str(self.get_flow(central, periphs[1])))
            up_arrow = vertical_arrow(str(self.get_flow(periphs[1], central)), down=False)
            for i in range(0, len(up_arrow)):
                lower.append(up_arrow[i] + '  ' + down_arrow[i])
            lower += box(periphs[1].name)

        upper_str = ''
        if upper:
            if depot:
                pad = ' ' * (len(depot_box[0]) + len(depot_central_arrow))
            else:
                pad = ''
            for line in upper:
                upper_str += pad + line + '\n'

        lower_str = ''
        if lower:
            if depot:
                pad = ' ' * (len(depot_box[0]) + len(depot_central_arrow))
            else:
                pad = ''
            for line in lower:
                lower_str += pad + line + '\n'

        return upper_str + '\n'.join(bottom) + '\n' + lower_str


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

    @property
    def free_symbols(self):
        if self.dose is not None:
            return self.dose.free_symbols
        else:
            return set()

    def __eq__(self, other):
        return isinstance(other, Compartment) and self.name == other.name and \
            self.dose == other.dose

    def __hash__(self):
        return hash(self.name)

    @property
    def amount(self):
        return symbols.real(f'A_{self.name}')


class Bolus:
    def __init__(self, symbol):
        self.symbol = symbols.real(str(symbol))

    @property
    def free_symbols(self):
        return {self.symbol}

    def __deepcopy__(self, memo):
        newone = type(self)(self.symbol)
        return newone

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

    def subs(self, substitutions):
        """Substitute old expression for new in all rhs of assignments"""
        for assignment in self:
            assignment.subs(substitutions)

    def find_assignment(self, symbol):
        """Returns full last statement given the symbol of an assignment"""
        statement = None
        for s in self:
            if isinstance(s, Assignment) and str(s.symbol) == symbol:
                statement = s
        return statement

    def remove_symbol_definition(self, symbol, statement):
        """Remove symbol definition and dependencies not used elsewhere

            statement is the statement from which the symbol was removed
        """
        removed_ind = self.index(statement)
        depinds = self._find_statement_and_deps(symbol, removed_ind)
        depsymbs = [self[i].symbol for i in depinds]
        keep = []
        for ind, symb in zip(depinds, depsymbs):
            for s in self[ind + 1:]:
                if ind not in depinds and symb in s.rhs_symbols:
                    keep.append(ind)
                    break
        for i in reversed(depinds):
            if i not in keep:
                del self[i]

    def _find_statement_and_deps(self, symbol, ind):
        """Find all statements and their dependencies before a certain statement
        """
        # Find index of final assignment of symbol before before
        statement = None
        for i in reversed(range(0, ind)):       # Might want to include the before for generality
            statement = self[i]
            if symbol == statement.symbol:
                break
        if statement is None:
            return ModelStatements([])
        found = [i]
        remaining = statement.rhs_symbols
        for j in reversed(range(0, i)):
            statement = self[j]
            if statement.symbol in remaining:
                found = [j] + found
                remaining.remove(statement.symbol)
                remaining |= statement.rhs_symbols
        return found

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

    def __repr__(self):
        s = ''
        for statement in self:
            s += repr(statement) + '\n'
        return s

    def __str__(self):
        s = ''
        for statement in self:
            s += str(statement)
        return s
