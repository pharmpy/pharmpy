import copy

import networkx as nx
import sympy
from sympy.printing.str import StrPrinter

import pharmpy.symbols as symbols
import pharmpy.unicode as unicode
from pharmpy.random_variables import VariabilityLevel


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
        return f'{self.symbol} := {self.expression}'

    def _repr_latex_(self):
        sym = self.symbol._repr_latex_()[1:-1]
        expr = self.expression._repr_latex_()[1:-1]
        return f'${sym} := {expr}$'

    def print_custom(self, rvs, trans):
        expr_ordered = self.order_terms(rvs, trans)
        return f'{self.symbol} = {expr_ordered}'

    def order_terms(self, rvs, trans):
        """Order terms such that random variables are placed last. Currently only supports
        additions."""
        if not isinstance(self.expression, sympy.Add) or rvs is None:
            return self.expression

        rvs_names = [rv.name for rv in rvs]

        if trans:
            trans_rvs = {v.name: k.name for k, v in trans.items() if str(k) in rvs_names}

        expr_args = self.expression.args
        terms_iiv_iov, terms_ruv, terms = [], [], []

        for arg in expr_args:
            arg_symbs = [s.name for s in arg.free_symbols]
            rvs_intersect = set(rvs_names).intersection(arg_symbs)

            if trans:
                trans_intersect = set(trans_rvs.keys()).intersection(arg_symbs)
                rvs_intersect.update({trans_rvs[rv] for rv in trans_intersect})

            if rvs_intersect:
                if len(rvs_intersect) == 1:
                    rv_name = list(rvs_intersect)[0]
                    variability_level = rvs[rv_name].variability_level
                    if variability_level == VariabilityLevel.RUV:
                        terms_ruv.append(arg)
                        continue
                terms_iiv_iov.append(arg)
            else:
                terms.append(arg)

        if not terms_iiv_iov and not terms_ruv:
            return self.expression

        def arg_len(symb):
            return len([s for s in symb.args])

        terms_iiv_iov.sort(reverse=True, key=arg_len)
        terms_ruv.sort(reverse=True, key=arg_len)
        terms += terms_iiv_iov + terms_ruv

        new_order = sympy.Add(*terms, evaluate=False)
        expr_ordered = sympy.UnevaluatedExpr(new_order)

        return StrPrinter(dict(order="none")).doprint(expr_ordered)


class ODESystem:
    """Base class and placeholder for ODE systems of different forms

    Attributes
    ----------
    solver : str
        Solver to use when numerically solving the ode system
        Currently supports NONMEM ADVANs
    """

    def __init__(self):
        self.solver = None

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
        return 'ODESystem()'

    def _repr_html(self):
        return str(self)


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
        super().__init__()

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

    def subs(self, substitutions):
        d = {
            sympy.Function(key.name)(symbols.symbol('t')): value
            for key, value in substitutions.items()
        }
        self.odes = [ode.subs(d) for ode in self.odes]
        self.ics = {key.subs(d): value.subs(d) for key, value in self.ics.items()}

    @property
    def rhs_symbols(self):
        return self.free_symbols

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
            and self.solver == other.solver
        )

    def _repr_latex_(self):
        rows = []
        for ode in self.odes:
            ode_repr = ode._repr_latex_()[1:-1]
            rows.append(ode_repr)
        for k, v in self.ics.items():
            ics_eq = sympy.Eq(k, v)
            ics_repr = ics_eq._repr_latex_()[1:-1]
            rows.append(ics_repr)
        return r'\begin{cases} ' + r' \\ '.join(rows) + r' \end{cases}'


class CompartmentalSystem(ODESystem):
    """System of ODEs descibed as a compartmental system"""

    t = symbols.symbol('t')

    def __init__(self):
        self._g = nx.DiGraph()
        super().__init__()

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

    def atoms(self, cls):
        atoms = set()
        for (_, _, rate) in self._g.edges.data('rate'):
            atoms |= rate.atoms(cls)
        return atoms

    def __eq__(self, other):
        return (
            isinstance(other, CompartmentalSystem)
            and nx.to_dict_of_dicts(self._g) == nx.to_dict_of_dicts(other._g)
            and self.find_dosing().dose == other.find_dosing().dose
            and self.solver == other.solver
        )

    def __deepcopy__(self, memo):
        newone = type(self)()
        newone._g = copy.deepcopy(self._g, memo)
        return newone

    def add_compartment(self, name):
        comp = Compartment(name, len(self._g) + 1)
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

    def n_connected(self, comp):
        """Get the number of compartments connected to comp"""
        out_comps = {c for c, _ in self.get_compartment_outflows(comp)}
        in_comps = {c for c, _ in self.get_compartment_inflows(comp)}
        return len(out_comps | in_comps)

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
        raise ValueError('No dosing compartment exists')

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
        cout = {comp for comp in oneout if self.get_flow(comp, central) is not None}
        cin = {comp for comp in onein if self.get_flow(central, comp) is not None}
        peripherals = list(cout & cin)
        # Return in deterministic indexed order
        peripherals = sorted(peripherals, key=lambda comp: comp.index)
        return peripherals

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

    def find_depot(self, statements):
        """Find the depot compartment

        The depot compartment is defined to be the compartment that only has out flow to the
        central compartment, but no flow from the central compartment.
        """
        transits = self.find_transit_compartments(statements)
        depot = self._find_depot()
        if depot in transits:
            depot = None
        return depot

    def _find_depot(self):
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
        nodes = list(self._g.nodes)
        output = self.find_output()
        # Put output last
        nodes.remove(output)
        nodes.append(output)
        for i in range(0, size):
            from_comp = nodes[i]
            diagsum = 0
            for j in range(0, size):
                to_comp = nodes[j]
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

    def to_explicit_odes(self, skip_output=False):
        amount_funcs = sympy.Matrix([sympy.Function(amt.name)(self.t) for amt in self.amounts])
        derivatives = sympy.Matrix([sympy.Derivative(fn, self.t) for fn in amount_funcs])
        inputs = self.zero_order_inputs
        a = self.compartmental_matrix @ amount_funcs + inputs
        eqs = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(derivatives, a)]
        ics = {}
        output = self.find_output()
        for node in self._g.nodes:
            if skip_output and node == output:
                continue
            if node.dose is not None and isinstance(node.dose, Bolus):
                if node.lag_time:
                    time = node.lag_time
                else:
                    time = 0
                ics[sympy.Function(node.amount.name)(time)] = node.dose.amount
            else:
                ics[sympy.Function(node.amount.name)(0)] = sympy.Integer(0)
        if skip_output:
            eqs = eqs[:-1]
        return eqs, ics

    def __len__(self):
        """Get the number of compartments including output"""
        return len(self._g.nodes)

    def _repr_html_(self):
        # Use Unicode art for now. There should be ways of drawing networkx
        s = str(self)
        return f'<pre>{s}</pre>'

    def __str__(self):
        output = self.find_output()
        output_box = unicode.Box(output.name)
        central = self.find_central()
        central_box = unicode.Box(central.name)
        depot = self._find_depot()
        current = self.find_dosing()
        if depot:
            comp = depot
        else:
            comp = central
        transits = []
        while current != comp:
            transits.append(current)
            current = self.get_compartment_outflows(current)[0][0]
        periphs = self.find_peripherals()
        nrows = 1 + 2 * len(periphs)
        ncols = 2 * len(transits) + (2 if depot else 0) + 3
        grid = unicode.Grid(nrows, ncols)
        if nrows == 1:
            main_row = 0
        else:
            main_row = 2
        col = 0
        for transit in transits:
            grid.set(main_row, col, unicode.Box(transit.name))
            col += 1
            grid.set(
                main_row, col, unicode.Arrow(str(self.get_compartment_outflows(transit)[0][1]))
            )
            col += 1
        if depot:
            grid.set(main_row, col, unicode.Box(depot.name))
            col += 1
            grid.set(main_row, col, unicode.Arrow(str(self.get_compartment_outflows(depot)[0][1])))
            col += 1
        central_col = col
        grid.set(main_row, col, central_box)
        col += 1
        grid.set(main_row, col, unicode.Arrow(str(self.get_flow(central, output))))
        col += 1
        grid.set(main_row, col, output_box)
        if periphs:
            grid.set(0, central_col, unicode.Box(periphs[0].name))
            grid.set(
                1,
                central_col,
                unicode.DualVerticalArrows(
                    str(self.get_flow(central, periphs[0])), str(self.get_flow(periphs[0], central))
                ),
            )
        if len(periphs) > 1:
            grid.set(4, central_col, unicode.Box(periphs[1].name))
            grid.set(
                3,
                central_col,
                unicode.DualVerticalArrows(
                    str(self.get_flow(periphs[1], central)), str(self.get_flow(central, periphs[1]))
                ),
            )

        dose = self.find_dosing().dose
        return str(dose) + '\n' + str(grid)


class Compartment:
    def __init__(self, name, index, lag_time=0):
        self.name = name
        self.index = index
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

    def find_assignment(self, variable, is_symbol=True, last=True):
        """Returns full last statement or all assignments that contains the symbol or
        variable of interest of an assignment"""
        assignments = []
        for statement in self:
            if isinstance(statement, Assignment):
                if is_symbol and str(statement.symbol) == variable:
                    assignments.append(statement)
                elif not is_symbol and variable in [s.name for s in statement.free_symbols]:
                    assignments.append(statement)

        if last:
            try:
                return assignments[-1]
            except IndexError:
                return None
        else:
            return assignments

    def extract_params_from_symb(self, symbol_name, pset):
        terms = {symb.name for symb in self.find_assignment(symbol_name).free_symbols}
        theta_name = terms.intersection(pset.names).pop()
        return pset[theta_name]

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

    def _create_dependency_graph(self):
        """Create a graph of dependencies between assignments
        Does not support dependencies of ODESystem
        """
        graph = nx.DiGraph()
        for i in range(len(self) - 1, -1, -1):
            rhs = self[i].rhs_symbols
            for s in rhs:
                for j in range(i - 1, -1, -1):
                    if isinstance(self[j], Assignment) and self[j].symbol == s:
                        graph.add_edge(i, j)
                        break
        return graph

    def remove_symbol_definitions(self, symbols, statement):
        """Remove symbols and dependencies not used elsewhere. statement
        is the statement from which the symbol was removed
        """
        graph = self._create_dependency_graph()
        removed_ind = self.index(statement)
        # Statements defining symbols and dependencies
        candidates = set()
        for s in symbols:
            for i in range(removed_ind - 1, -1, -1):
                stat = self[i]
                if isinstance(stat, Assignment) and stat.symbol == s:
                    candidates.add(i)
                    break
        for i in candidates.copy():
            if i in graph:
                candidates |= set(nx.dfs_preorder_nodes(graph, i))
        # All statements needed for removed_ind
        if removed_ind in graph:
            keep = {down for _, down in nx.dfs_edges(graph, removed_ind)}
        else:
            keep = set()
        candidates -= keep
        # Other dependencies after removed_ind
        additional = {down for up, down in graph.edges if up > removed_ind and down in candidates}
        for add in additional.copy():
            if add in graph:
                additional |= set(nx.dfs_preorder_nodes(graph, add))
        remove = candidates - additional
        for i in reversed(sorted(remove)):
            del self[i]

    @property
    def ode_system(self):
        """Returns the ODE system of the model or None if the model doesn't have an ODE system"""
        for s in self:
            if isinstance(s, ODESystem):
                return s
        return None

    def before_ode(self):
        sset = ModelStatements()
        for s in self:
            if isinstance(s, ODESystem):
                break
            sset.append(s)

        return sset

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
        """Add a statement just before the ODE system or at the end of the model"""
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

    def _repr_html_(self):
        html = r'\begin{align*}'
        for statement in self:
            if hasattr(statement, '_repr_html_'):
                html += '\\end{align*}'
                s = statement._repr_html_()
                html += s + '\\begin{align*}'
            else:
                s = f'${statement._repr_latex_()}$'
                s = s.replace(':=', '&:=')
                s = s.replace('$', '')
                s = s + r'\\'
                html += s
        return html + '\\end{align*}'
