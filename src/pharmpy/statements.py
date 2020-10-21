import copy
import math

import networkx as nx
import sympy

import pharmpy.symbols as symbols


class Assignment:
    """Representation of variable assignment, similar to :class:`sympy.codegen.Assignment`

    Attributes
    ----------
    symbol : sympy.Symbol
        Symbol of statement
    expression
        Expression of statement
    """

    def __init__(self, symbol, expression):
        """symbol can be either string or sympy symbol
        symbol can be a string and a real symbol will be created
        """
        try:
            symbol.is_Symbol
            self.symbol = symbol
        except AttributeError:
            self.symbol = symbols.symbol(symbol)
        self.expression = expression

    def subs(self, substitutions):
        """Substitute symbols in assignment.

        substitutions - dictionary with old-new pair (can be type str or
                        sympy symbol)
        """
        self.symbol = self.symbol.subs(substitutions, simultaneous=True)
        self.expression = self.expression.subs(substitutions, simultaneous=True)

    @property
    def free_symbols(self):
        symbols = {self.symbol}
        symbols |= self.expression.free_symbols
        return symbols

    @property
    def rhs_symbols(self):
        return self.expression.free_symbols

    def __eq__(self, other):
        return (
            isinstance(other, Assignment)
            and self.symbol == other.symbol
            and self.expression == other.expression
        )

    def __str__(self):
        expression = sympy.pretty(self.expression)
        lines = expression.split('\n')
        definition = f'{sympy.pretty(self.symbol)} := '
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
    """Base class and placeholder for ODE systems of different forms"""

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
    """Append a left bracket for an array of lines"""
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
    """System of ODEs described explicitly"""

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
            try:  # To allow for regular python classes as values for ics
                free |= value.free_symbols
            except AttributeError:
                pass
        return free

    @property
    def rhs_symbols(self):
        return self.free_symbols  # This works currently

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
        return (
            isinstance(other, ExplicitODESystem)
            and self.odes == other.odes
            and self.ics == other.ics
        )


class CompartmentalSystem(ODESystem):
    """System of ODEs descibed as a compartmental system"""

    t = symbols.symbol('t')

    def __init__(self):
        self._g = nx.DiGraph()

    def subs(self, substitutions):
        for (u, v, rate) in self._g.edges.data('rate'):
            rate_sub = rate.subs(substitutions, simultaneous=True)
            self._g.edges[u, v]['rate'] = rate_sub
        for comp in self._g.nodes:
            comp.subs(substitutions)

    @property
    def free_symbols(self):
        free = {symbols.symbol('t')}
        for (_, _, rate) in self._g.edges.data('rate'):
            free |= rate.free_symbols
        for node in self._g.nodes:
            free |= node.free_symbols
        return free

    @property
    def rhs_symbols(self):
        return self.free_symbols  # This works currently

    def __eq__(self, other):
        return (
            isinstance(other, CompartmentalSystem)
            and nx.to_dict_of_dicts(self._g) == nx.to_dict_of_dicts(other._g)
            and self.find_dosing().dose == other.find_dosing().dose
        )

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
        try:
            rate = self._g.edges[source, destination]['rate']
        except KeyError:
            rate = None
        return rate

    def get_compartment_outflows(self, compartment):
        """Generate all flows going out of a compartment"""
        flows = []
        for node in self._g.successors(compartment):
            flow = self.get_flow(compartment, node)
            flows.append((node, flow))
        return flows

    def get_compartment_inflows(self, compartment):
        """Generate all flows going in to a compartment"""
        flows = []
        for node in self._g.predecessors(compartment):
            flow = self.get_flow(node, compartment)
            flows.append((node, flow))
        return flows

    def find_compartment(self, name):
        for comp in self._g.nodes:
            if comp.name == name:
                return comp
        else:
            return None

    def find_output(self):
        """Find the output compartment

        An output compartment is defined to be a compartment that does not have any outward
        flow. A model has to have one and only one output compartment.
        """
        zeroout = [node for node, out_degree in self._g.out_degree() if out_degree == 0]
        if len(zeroout) == 1:
            return zeroout[0]
        else:
            raise ValueError('More than one or zero output compartments')

    def find_dosing(self):
        """Find the dosing compartment

        A dosing compartment is a compartment that receives an input dose. Only one dose
        compartment is supported.
        """
        for node in self._g.nodes:
            if node.dose is not None:
                return node

    def find_central(self):
        """Find the central compartment

        The central compartment is defined to be the compartment that has an outward flow
        to the output compartment. Only one central compartment is supported.
        """
        output = self.find_output()
        central = next(self._g.predecessors(output))
        return central

    def find_peripherals(self):
        central = self.find_central()
        oneout = {node for node, out_degree in self._g.out_degree() if out_degree == 1}
        onein = {node for node, in_degree in self._g.in_degree() if in_degree == 1}
        peripherals = (oneout & onein) - {central}
        return list(peripherals)

    def find_transit_compartments(self, statements):
        """Find all transit compartments

        Transit compartments are a chain of compartments with the same out rate starting from
        the dose compartment. Because one single transit compartment cannot be distinguished
        from one depot compartment such compartment will be defined to be a depot and not
        a transit compartment.
        """
        transits = []
        comp = self.find_dosing()
        if len(self.get_compartment_inflows(comp)) != 0:
            return transits
        outflows = self.get_compartment_outflows(comp)
        if len(outflows) != 1:
            return transits
        transits.append(comp)
        comp, rate = outflows[0]
        rate = statements.full_expression_from_odes(rate)
        while True:
            if len(self.get_compartment_inflows(comp)) != 1:
                break
            outflows = self.get_compartment_outflows(comp)
            if len(outflows) != 1:
                break
            next_comp, next_rate = outflows[0]
            next_rate = statements.full_expression_from_odes(next_rate)
            if rate != next_rate:
                break
            transits.append(comp)
            comp = next_comp
        # Special case of one transit directly into central is not defined as a transit
        # Also not central itself
        central = self.find_central()
        if len(transits) == 1 and (
            self.get_flow(transits[0], central) is not None or transits[0] == central
        ):
            return []
        else:
            return transits

    def find_depot(self):
        """Find the depot compartment

        The depot compartment is defined to be the compartment that only has out flow to the
        central compartment, but no flow from the central compartment.
        """
        central = self.find_central()
        depot = None
        for to_central, _ in self.get_compartment_inflows(central):
            outflows = self.get_compartment_outflows(to_central)
            if len(outflows) == 1:
                inflows = self.get_compartment_inflows(to_central)
                for in_comp, _ in inflows:
                    if in_comp == central:
                        break
                else:
                    depot = to_central
                    break
        return depot

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

    @property
    def names(self):
        """A list of the names of all compartments"""
        return [node.name for node in self._g.nodes]

    @property
    def zero_order_inputs(self):
        """A vector of all zero order inputs to each compartment"""
        inputs = []
        for node in self._g.nodes:
            if node.dose is not None and isinstance(node.dose, Infusion):
                if node.dose.rate is not None:
                    expr = node.dose.rate
                    cond = node.dose.amount / node.dose.rate
                else:
                    expr = node.dose.amount / node.dose.duration
                    cond = node.dose.duration
                infusion_func = sympy.Piecewise((expr, self.t < cond), (0, True))
                inputs.append(infusion_func)
            else:
                inputs.append(0)
        return sympy.Matrix(inputs)

    def to_explicit_odes(self):
        amount_funcs = sympy.Matrix([sympy.Function(amt.name)(self.t) for amt in self.amounts])
        derivatives = sympy.Matrix([sympy.Derivative(fn, self.t) for fn in amount_funcs])
        inputs = self.zero_order_inputs
        a = self.compartmental_matrix @ amount_funcs + inputs
        eqs = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(derivatives, a)]
        ics = {}
        for node in self._g.nodes:
            if node.dose is not None and isinstance(node.dose, Bolus):
                if node.lag_time:
                    time = node.lag_time
                else:
                    time = 0
                ics[sympy.Function(node.amount.name)(time)] = node.dose.amount
            else:
                ics[sympy.Function(node.amount.name)(0)] = 0
        return eqs, ics

    def __len__(self):
        """Get the number of compartments including output"""
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

        dose = self.find_dosing().dose
        return str(dose) + '\n' + upper_str + '\n'.join(bottom) + '\n' + lower_str


def box(s):
    """Draw unicode box around string and return new string"""
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
        return [before + '↑' + after, flow, before + '│' + after]


class Compartment:
    def __init__(self, name, lag_time=0):
        self.name = name
        self.dose = None
        self.lag_time = lag_time

    @property
    def lag_time(self):
        return self._lag_time

    @lag_time.setter
    def lag_time(self, value):
        self._lag_time = sympy.sympify(value)

    @property
    def free_symbols(self):
        symbs = set()
        if self.dose is not None:
            symbs |= self.dose.free_symbols
        symbs |= self.lag_time.free_symbols
        return symbs

    def subs(self, substitutions):
        if self.dose is not None:
            self.dose.subs(substitutions)
        self.lag_time.subs(substitutions)

    def __eq__(self, other):
        return (
            isinstance(other, Compartment)
            and self.name == other.name
            and self.dose == other.dose
            and self.lag_time == other.lag_time
        )

    def __hash__(self):
        return hash(self.name)

    @property
    def amount(self):
        return symbols.symbol(f'A_{self.name}')


class Bolus:
    def __init__(self, amount):
        self.amount = symbols.symbol(str(amount))

    @property
    def free_symbols(self):
        return {self.amount}

    def subs(self, substitutions):
        self.amount = self.amount.subs(substitutions, simultaneous=True)

    def __deepcopy__(self, memo):
        newone = type(self)(self.amount)
        return newone

    def __eq__(self, other):
        return isinstance(other, Bolus) and self.amount == other.amount

    def __repr__(self):
        return f'Bolus({self.amount})'


class Infusion:
    def __init__(self, amount, rate=None, duration=None):
        if rate is None and duration is None:
            raise ValueError('Need rate or duration for Infusion')
        self.rate = sympy.sympify(rate)
        self.duration = sympy.sympify(duration)
        self.amount = sympy.sympify(amount)

    @property
    def free_symbols(self):
        if self.rate is not None:
            symbs = self.rate.free_symbols
        else:
            symbs = self.duration.free_symbols
        return symbs | self.amount.free_symbols

    def subs(self, substitutions):
        self.amount = self.amount.subs(substitutions, simultaneous=True)
        if self.rate is not None:
            self.rate = self.rate.subs(substitutions, simultaneous=True)
        else:
            self.duration = self.duration.subs(substitutions, simultaneous=True)

    def __deepcopy__(self, memo):
        new = type(self)(self.amount, rate=self.rate, duration=self.duration)
        return new

    def __eq__(self, other):
        return (
            isinstance(other, Infusion)
            and self.rate == other.rate
            and self.duration == other.duration
            and self.amount == other.amount
        )

    def __repr__(self):
        if self.rate is not None:
            arg = f'rate={self.rate}'
        else:
            arg = f'duration={self.duration}'
        return f'Infusion({self.amount}, {arg})'


class ModelStatements(list):
    """A list of sympy statements describing the model"""

    @property
    def free_symbols(self):
        """Get a set of all free symbols"""
        symbols = set()
        for assignment in self:
            symbols |= assignment.free_symbols
        return symbols

    def subs(self, substitutions):
        """Substitute symbols in all statements.

        substitutions - dictionary with old-new pair (can be type str or
                        sympy symbol)
        """
        for assignment in self:
            assignment.subs(substitutions)

    def find_assignment(self, symbol):
        """Returns full last statement given the symbol of an assignment"""
        statement = None
        for s in self:
            if isinstance(s, Assignment) and str(s.symbol) == symbol:
                statement = s
        return statement

    def reassign(self, symbol, expression):
        """Reassign symbol to expression"""
        last = True
        for i, stat in zip(range(len(self) - 1, -1, -1), reversed(self)):
            if isinstance(stat, Assignment) and stat.symbol == symbol:
                if last:
                    stat.expression = expression
                    last = False
                else:
                    del self[i]

    def remove_symbol_definitions(self, symbols, statement):
        """Remove symbol remove_symbol_definitions and dependencies not used elsewhere. statement
        is the statement from which the symbol was removed
        """
        removed_ind = self.index(statement)
        depinds = set()
        for symbol in symbols:
            depinds |= self._find_statement_and_deps(symbol, removed_ind)
        depsymbs = {self[i].symbol for i in depinds}
        keep = []
        for candidate in depsymbs:
            for i in depinds:
                if self[i].symbol == candidate:
                    final = i
            for stat_ind in range(final + 1, len(self)):
                stat = self[stat_ind]
                if stat_ind not in depinds and candidate in stat.rhs_symbols:
                    indices = [i for i in depinds if self[i].symbol == candidate]
                    keep += indices
                    break
        for i in reversed(sorted(depinds)):
            if i not in keep:
                del self[i]

    def _find_statement_and_deps(self, symbol, ind):
        """Find indices of the last symbol definition and its dependenceis before a
        certain statement
        """
        # Find index of final assignment of symbol before before
        for i in reversed(range(0, ind)):
            statement = self[i]
            if symbol == statement.symbol:
                break
        else:
            return set()
        found = {i}
        remaining = statement.free_symbols
        for j in reversed(range(0, i)):
            statement = self[j]
            if statement.symbol in remaining:
                found |= {j}
                remaining.remove(statement.symbol)
                remaining |= statement.rhs_symbols
        return found

    @property
    def ode_system(self):
        """Returns the ODE system of the model or None if the model doesn't have an ODE system"""
        for s in self:
            if isinstance(s, ODESystem):
                return s
        return None

    def _ode_index(self):
        for i, s in enumerate(self):
            if isinstance(s, ODESystem):
                return i
        return None

    def full_expression_from_odes(self, expression):
        """Expand an expression into its full definition

        Before ODE system
        """
        i = self._ode_index()
        for j in range(i - 1, -1, -1):
            expression = expression.subs({self[j].symbol: self[j].expression})
        return expression

    def add_before_odes(self, statement):
        """Add a statement just before the ODE system"""
        for i, s in enumerate(self):
            if isinstance(s, ODESystem):
                break
        else:
            i += 1
        self.insert(i, statement)

    def copy(self):
        return copy.deepcopy(self)

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
