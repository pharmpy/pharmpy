from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Iterable, List, Optional, Set, Tuple, Union, overload

import pharmpy.internals.unicode as unicode
from pharmpy.deps import networkx as nx
from pharmpy.deps import sympy
from pharmpy.internals.expr.assumptions import assume_all
from pharmpy.internals.expr.leaves import free_images, free_images_and_symbols
from pharmpy.internals.expr.ode import canonical_ode_rhs
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.immutable import Immutable


def is_zero_matrix(A):
    for e in A:
        if not e.is_zero:
            return False
    return True


class Statement(Immutable):
    """Abstract base class for all types of statements"""

    def __add__(self, other):
        if isinstance(other, Statements):
            return Statements((self,) + other._statements)
        elif isinstance(other, Statement):
            return Statements((self, other))
        else:
            return Statements((self,) + tuple(other))

    def __radd__(self, other):
        if isinstance(other, Statement):
            return Statements((other, self))
        else:
            return Statements(tuple(other) + (self,))

    @abstractmethod
    def subs(self, substitutions):
        pass

    @property
    @abstractmethod
    def free_symbols(self) -> Set[sympy.Symbol]:
        pass

    @property
    @abstractmethod
    def rhs_symbols(self) -> Set[sympy.Symbol]:
        pass


class Assignment(Statement):
    """Representation of variable assignment

    This class represents an assignment of an expression to a variable. Multiple assignments
    are combined together into a Statements object.

    Parameters
    ----------
    symbol : sympy.Symbol or str
        Symbol of statement
    expression : sympy.Expr
        Expression of assignment
    """

    def __init__(self, symbol, expression):
        self._symbol = symbol
        self._expression = expression

    @classmethod
    def create(cls, symbol, expression):
        if isinstance(symbol, str):
            symbol = sympy.Symbol(symbol)
        if not (symbol.is_Symbol or symbol.is_Derivative or symbol.is_Function):
            raise TypeError("symbol of Assignment must be a Symbol or str representing a symbol")
        if isinstance(expression, str):
            expression = parse_expr(expression)
        # Needed for expression.free_symbols to work
        if isinstance(expression, float):
            expression = sympy.Float(expression)
        if isinstance(expression, int):
            expression = sympy.Integer(expression)
        return cls(symbol, expression)

    def replace(self, **kwargs):
        symbol = kwargs.get('symbol', self._symbol)
        expression = kwargs.get('expression', self._expression)
        return Assignment.create(symbol, expression)

    @property
    def symbol(self):
        """Symbol of statement"""
        return self._symbol

    @property
    def expression(self):
        """Expression of assignment"""
        return self._expression

    def subs(self, substitutions):
        """Substitute expressions or symbols in assignment

        Parameters
        ----------
        substitutions : dict
            old-new pairs

        Return
        ------
        Assignment
            Updated assignment object

        Examples
        --------
        >>> from pharmpy.model import Assignment
        >>> a = Assignment.create('CL', 'POP_CL + ETA_CL')
        >>> a
        CL = ETA_CL + POP_CL
        >>> b = a.subs({'ETA_CL' : 'ETA_CL * WGT'})
        >>> b
        CL = ETA_CL⋅WGT + POP_CL

        """
        symbol = subs(self.symbol, substitutions, simultaneous=True)
        expression = subs(self.expression, substitutions, simultaneous=True)
        return Assignment(symbol, expression)

    @property
    def free_symbols(self):
        """Get set of all free symbols in the assignment

        Note that the left hand side symbol will be in the set

        Examples
        --------
        >>> from pharmpy.model import Assignment
        >>> a = Assignment.create('CL', 'POP_CL + ETA_CL')
        >>> a.free_symbols      # doctest: +SKIP
        {CL, ETA_CL, POP_CL}

        """
        symbols = {self.symbol}
        symbols |= self.expression.free_symbols
        return symbols

    @property
    def rhs_symbols(self):
        """Get set of all free symbols in the right hand side expression

        Examples
        --------
        >>> from pharmpy.model import Assignment
        >>> a = Assignment.create('CL', 'POP_CL + ETA_CL')
        >>> a.rhs_symbols      # doctest: +SKIP
        {ETA_CL, POP_CL}

        """
        return self.expression.free_symbols

    def __eq__(self, other):
        return (
            isinstance(other, Assignment)
            and self.symbol == other.symbol
            and self.expression == other.expression
        )

    def __hash__(self):
        return hash((self.symbol, self.expression))

    def to_dict(self):
        return {
            'class': 'Assignment',
            'symbol': sympy.srepr(self.symbol),
            'expression': sympy.srepr(self.expression),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            symbol=sympy.parse_expr(d['symbol']), expression=sympy.parse_expr(d['expression'])
        )

    def __repr__(self):
        expression = sympy.pretty(self.expression)
        lines = [line.rstrip() for line in expression.split('\n')]
        definition = f'{sympy.pretty(self.symbol)} = '
        s = ''
        for line in lines:
            if line == lines[-1]:
                s += definition + line + '\n'
            else:
                s += len(definition) * ' ' + line + '\n'
        return s.rstrip()

    def _repr_latex_(self):
        sym = sympy.latex(self.symbol)
        expr = sympy.latex(self.expression, mul_symbol='dot')
        return f'${sym} = {expr}$'


class ODESystem(Statement):
    """Abstract base class for ODE systems of different forms"""

    pass


class Output(Immutable):
    def __new__(cls):
        # Singleton class
        if not hasattr(cls, 'instance'):
            cls.instance = super(Output, cls).__new__(cls)
        return cls.instance

    def __repr__(self):
        return "Output()"

    def __hash__(self):
        return 5267

    def to_dict(self):
        return {'class': 'Output'}

    @classmethod
    def from_dict(cls, d):
        return cls.instance

    def __eq__(self, other):
        return self is other


output = Output()


class CompartmentalSystemBuilder:
    """Builder for CompartmentalSystem"""

    def __init__(self, cs=None):
        if cs:
            self._g = cs._g.copy()
        else:
            self._g = nx.DiGraph()
            self._g.add_node(output)  # Single output "compartment"

    def add_compartment(self, compartment):
        """Add compartment to system

        The compartment will be added without any flows to other compartments.
        Use the add_flow method to add flows to and from the newly added compartment.

        Parameters
        ----------
        compartment : Compartment
            Compartment to add

        Examples
        --------
        >>> from pharmpy.model import CompartmentalSystemBuilder
        >>> cb = CompartmentalSystemBuilder()
        >>> central = cb.add_compartment("CENTRAL")
        """
        self._g.add_node(compartment)

    def remove_compartment(self, compartment):
        """Remove compartment from system

        Parameters
        ----------
        compartment : Compartment
            Compartment object to remove from system

        Examples
        --------
        >>> from pharmpy.model import CompartmentalSystemBuilder
        >>> cb = CompartmentalSystemBuilder()
        >>> central = Compartment("CENTRAL")
        >>> cb.add_compartment(central)
        >>> cb.remove_compartment(central)
        """
        self._g.remove_node(compartment)

    def add_flow(self, source, destination, rate):
        """Add flow between two compartments

        Parameters
        ----------
        source : Compartment
            Source compartment
        destination : Compartment
            Destination compartment
        rate : Expression
            Symbolic rate of flow

        Examples
        --------
        >>> from pharmpy.model import Compartment, CompartmentalSystemBuilder
        >>> cb = CompartmentalSystemBuilder()
        >>> depot = Compartment("DEPOT")
        >>> cb.add_compartment(depot)
        >>> central = Compartment("CENTRAL")
        >>> cb.add_compartment(central)
        >>> cb.add_flow(depot, central, "KA")
        """
        self._g.add_edge(source, destination, rate=parse_expr(rate))

    def remove_flow(self, source, destination):
        """Remove flow between two compartments

        Parameters
        ----------
        source : Compartment
            Source compartment
        destination : Compartment
            Destination compartment

        Examples
        --------
        >>> from pharmpy.model import CompartmentalSystemBuilder
        >>> cb = CompartmentalSystemBuilder()
        >>> depot = Compartment("DEPOT")
        >>> cb.add_compartment(depot)
        >>> central = Compartment("CENTRAL")
        >>> cb.add_compartment(central)
        >>> cb.add_flow(depot, central, "KA")
        >>> cb.remove_flow(depot, central)
        """
        self._g.remove_edge(source, destination)

    def move_dose(self, source, destination):
        """Move a dose input from one compartment to another

        Parameters
        ----------
        source : Compartment
            Source compartment
        destination : Compartment
            Destination compartment

        """
        new_source = source.replace(dose=None)
        new_dest = destination.replace(dose=source.dose)
        mapping = {source: new_source, destination: new_dest}
        nx.relabel_nodes(self._g, mapping, copy=False)

    def set_dose(self, compartment, dose):
        """Set dose of compartment

        Parameters
        ----------
        compartment : Compartment
            Compartment for which to change dose
        dose : Dose
            New dose

        Returns
        -------
        Compartment
            The new updated compartment
        """
        new_comp = compartment.replace(dose=dose)
        mapping = {compartment: new_comp}
        nx.relabel_nodes(self._g, mapping, copy=False)
        return new_comp

    def set_lag_time(self, compartment, lag_time):
        """Set lag time of compartment

        Parameters
        ----------
        compartment : Compartment
            Compartment for which to change lag time
        lag_time : expr
            New lag time

        Returns
        -------
        Compartment
            The new updated compartment
        """
        new_comp = compartment.replace(lag_time=lag_time)
        mapping = {compartment: new_comp}
        nx.relabel_nodes(self._g, mapping, copy=False)
        return new_comp

    def set_bioavailability(self, compartment, bioavailability):
        """Set bioavailability of compartment

        Parameters
        ----------
        compartment : Compartment
            Compartment for which to change bioavailability
        bioavailability : expr
            New bioavailability

        Returns
        -------
        Compartment
            The new updated compartment
        """
        new_comp = compartment.replace(bioavailability=bioavailability)
        mapping = {compartment: new_comp}
        nx.relabel_nodes(self._g, mapping, copy=False)
        return new_comp

    def set_input(self, compartment, input):
        """Set zero order input of compartment

        Parameters
        ----------
        compartment : Compartment
            Compartment for which to change zero order input
        input : expr
            New input

        Returns
        -------
        Compartment
            The new updated compartment
        """
        new_comp = compartment.replace(input=input)
        mapping = {compartment: new_comp}
        nx.relabel_nodes(self._g, mapping, copy=False)
        return new_comp

    def find_compartment(self, name):
        for comp in self._g.nodes:
            if not isinstance(comp, Output) and comp.name == name:
                return comp


def _is_positive(expr: sympy.Expr) -> bool:
    return sympy.ask(
        sympy.Q.positive(expr), assume_all(sympy.Q.positive, free_images_and_symbols(expr))
    )


def _comps(graph):
    return {comp for comp in graph.nodes if not isinstance(comp, Output)}


def to_compartmental_system(names, eqs):
    """Convert an list of odes to a compartmental system

    names : func to compartment name map
    """

    cb = CompartmentalSystemBuilder()
    compartments = {}
    concentrations = set()
    for eq in eqs:
        A = eq.lhs.args[0]
        concentrations.add(A)
        name = names[A.func]
        comp = Compartment.create(name)
        cb.add_compartment(comp)
        compartments[name] = comp

    neweqs = list(eqs)  # Remaining flows

    for eq in eqs:
        checked_terms = set()
        for comp_func in concentrations.intersection(free_images(eq.rhs)):
            dep = eq.rhs.as_independent(comp_func, as_Add=True)[1]
            terms = sympy.Add.make_args(dep.expand())
            for term in terms:
                if term not in checked_terms:
                    checked_terms.add(term)
                    from_comp = None
                    to_comp = None
                    if len(concentrations.intersection(free_images(term))) >= 2:
                        # This means second order absorption -> find matching term
                        # to determine flow
                        if _is_positive(term):
                            for second_comp in concentrations.intersection(free_images(term)):
                                for eq_2 in eqs:
                                    if eq_2.lhs.args[0].name == second_comp.name:
                                        if -term in sympy.Add.make_args(eq_2.rhs.expand()):
                                            from_comp = compartments[names[second_comp.func]]
                                            to_comp = compartments[names[eq.lhs.args[0].func]]
                    else:
                        # Find matching term to determine if flow is between
                        # compartments or not
                        if _is_positive(term):
                            for eq_2 in eqs:
                                if -term in sympy.Add.make_args(eq_2.rhs.expand()):
                                    from_comp = compartments[names[eq_2.lhs.args[0].func]]
                                    to_comp = compartments[names[eq.lhs.args[0].func]]

                    if from_comp is not None and to_comp is not None:
                        # FIXME : get current flow from builder instead?
                        cs = CompartmentalSystem(cb)
                        current_flow = cs.get_flow(from_comp, to_comp)

                        if isinstance(current_flow, sympy.core.numbers.Zero):
                            cb.add_flow(from_comp, to_comp, term / comp_func)
                        else:
                            new_flow = term / comp_func + current_flow
                            cb.add_flow(from_comp, to_comp, new_flow)

                        for i, neweq in enumerate(neweqs):
                            if neweq.lhs.args[0].name == eq.lhs.args[0].name:
                                neweqs[i] = sympy.Eq(neweq.lhs, sympy.expand(neweq.rhs - term))
                            elif neweq.lhs.args[0].name == comp_func.name:
                                neweqs[i] = sympy.Eq(neweq.lhs, sympy.expand(neweq.rhs + term))
    for eq in neweqs:
        if eq.rhs != 0:
            i = sympy.Integer(0)
            o = sympy.Integer(0)
            for term in sympy.Add.make_args(eq.rhs):
                if _is_positive(term):
                    i = i + term
                else:
                    o = o + term
            comp_func = eq.lhs.args[0]
            from_comp = compartments[names[comp_func.func]]
            if o != 0:
                cb.add_flow(from_comp, output, -o / comp_func)
            if i != 0:
                cb.set_input(from_comp, i)
    cs = CompartmentalSystem(cb)
    return cs


class CompartmentalSystem(ODESystem):
    """System of ODEs descibed as a compartmental system

    Examples
    --------
    >>> from pharmpy.model import Bolus, Compartment, output
    >>> from pharmpy.model import CompartmentalSystemBuilder, CompartmentalSystem
    >>> cb = CompartmentalSystemBuilder()
    >>> dose = Bolus.create("AMT")
    >>> central = Compartment("CENTRAL", dose)
    >>> cb.add_compartment(central)
    >>> peripheral = Compartment("PERIPHERAL")
    >>> cb.add_compartment(peripheral)
    >>> cb.add_flow(central, peripheral, "K12")
    >>> cb.add_flow(peripheral, central, "K21")
    >>> cb.add_flow(central, output, "CL / V")
    >>> CompartmentalSystem(cb)        # doctest: +SKIP
    Bolus(AMT)
                     ┌──────────┐
                     │PERIPHERAL│
                     └──────────┘
                      ↑        │
                     K12      K21
                      │        ↓
    ┌───────┐      ┌──────────┐      ┌──────────┐
    │CENTRAL│──K12→│PERIPHERAL│──K21→│ CENTRAL  │──CL/V→
    └───────┘      └──────────┘      └──────────┘
    """

    def __init__(self, builder=None, graph=None, t=sympy.Symbol('t')):
        if builder is not None:
            self._g = nx.freeze(builder._g.copy())
        elif graph is not None:
            self._g = graph
        self._t = t

    @classmethod
    def create(cls, builder=None, graph=None, eqs=None, t=sympy.Symbol('t')):
        numargs = (
            (0 if builder is None else 1) + (0 if graph is None else 1) + (0 if eqs is None else 1)
        )
        if numargs != 1:
            raise ValueError(
                "Need exactly one of builder, graph or eqs to create a CompartmentalSystem"
            )
        if eqs is not None:
            pass
        t = sympy.sympify(t)
        return cls(builder=builder, graph=graph, t=t)

    def replace(self, **kwargs):
        t = kwargs.get('t', self._t)
        builder = kwargs.get('builder', None)
        if builder is None:
            g = kwargs.get('graph', self._g)
        else:
            g = None
        return CompartmentalSystem.create(builder=builder, graph=g, t=t)

    @property
    def t(self):
        """Independent variable of ODESystem"""
        return self._t

    @property
    def eqs(self):
        """Tuple of equations"""
        amount_funcs = sympy.Matrix([sympy.Function(amt.name)(self.t) for amt in self.amounts])
        derivatives = sympy.Matrix([sympy.Derivative(fn, self.t) for fn in amount_funcs])
        inputs = self.zero_order_inputs
        a = self.compartmental_matrix @ amount_funcs + inputs
        eqs = [sympy.Eq(lhs, canonical_ode_rhs(rhs)) for lhs, rhs in zip(derivatives, a)]
        return tuple(eqs)

    @property
    def free_symbols(self):
        """Get set of all free symbols in the compartmental system

        Returns
        -------
        set
            Set of symbols

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.free_symbols  # doctest: +SKIP
        {AMT, CL, V, t}
        """
        free = {sympy.Symbol('t')}
        for _, _, rate in self._g.edges.data('rate'):
            free |= rate.free_symbols
        for node in _comps(self._g):
            free |= node.free_symbols
        return free

    @property
    def rhs_symbols(self):
        """Get set of all free symbols in the right hand side expressions

        Returns
        -------
        set
            Set of symbols

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.rhs_symbols   # doctest: +SKIP
        {AMT, CL, V, t}
        """
        return self.free_symbols  # This works currently

    def subs(self, substitutions):
        """Substitute expressions or symbols in ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.subs({'AMT': 'DOSE'})
        Bolus(DOSE, admid=1)
        ┌───────┐
        │CENTRAL│──CL/V→
        └───────┘
        """
        cb = CompartmentalSystemBuilder(self)
        for u, v, rate in cb._g.edges.data('rate'):
            rate_sub = subs(rate, substitutions, simultaneous=True)
            cb._g.edges[u, v]['rate'] = rate_sub
        mapping = {comp: comp.subs(substitutions) for comp in _comps(self._g)}
        nx.relabel_nodes(cb._g, mapping, copy=False)
        return CompartmentalSystem(cb)

    def atoms(self, cls):
        """Get set of all symbolic atoms of some kind

        For more information see
        https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.atoms

        Parameters
        ----------
        cls : type
            Type of atoms to find

        Returns
        -------
        set
            Set of symbolic atoms
        """
        atoms = set()
        for _, _, rate in self._g.edges.data('rate'):
            atoms |= rate.atoms(cls)
        return atoms

    def __eq__(self, other):
        return (
            isinstance(other, CompartmentalSystem)
            and self._t == other._t
            and nx.to_dict_of_dicts(self._g) == nx.to_dict_of_dicts(other._g)
            and self.dosing_compartment == other.dosing_compartment
        )

    def __hash__(self):
        return hash((self._t, self._g))

    def to_dict(self):
        comps = [comp for comp in self._g.nodes]
        comps_dicts = tuple(comp.to_dict() for comp in comps)

        edges = []
        for from_comp, to_comp, rate in self._g.edges.data('rate'):
            from_n = comps.index(from_comp)
            to_n = comps.index(to_comp)
            edge = (from_n, to_n, sympy.srepr(rate))
            edges.append(edge)

        d = {
            'class': 'CompartmentalSystem',
            'compartments': comps_dicts,
            'rates': edges,
            't': sympy.srepr(self._t),
        }
        return d

    @classmethod
    def from_dict(cls, d):
        cb = CompartmentalSystemBuilder()
        comps = []
        for comp_dict in d['compartments']:
            if comp_dict['class'] == 'Output':
                compartment = output
            else:
                compartment = Compartment.from_dict(comp_dict)
                cb.add_compartment(compartment)
            comps.append(compartment)

        for from_n, to_n, rate in d['rates']:
            from_comp = comps[from_n]
            to_comp = comps[to_n]
            cb.add_flow(from_comp, to_comp, sympy.parse_expr(rate))

        return cls(cb, t=sympy.parse_expr(d['t']))

    def get_flow(self, source, destination):
        """Get the rate of flow between two compartments

        Parameters
        ----------
        source : Compartment
            Source compartment
        destination : Compartment
            Destination compartment

        Returns
        -------
        Expression
            Symbolic rate

        Examples
        --------
        >>> from pharmpy.model import CompartmentalSystem, Compartment
        >>> cb = CompartmentalSystemBuilder()
        >>> depot = Compartment("DEPOT")
        >>> cb.add_compartment(depot)
        >>> central = Compartment("CENTRAL")
        >>> cb.add_compartment(central)
        >>> cb.add_flow(depot, central, "KA")
        >>> odes = CompartmentalSystem(cb)
        >>> odes.get_flow(depot, central)
        KA
        >>> odes.get_flow(central, depot)
        0
        """
        try:
            rate = self._g.edges[source, destination]['rate']
        except KeyError:
            rate = sympy.Integer(0)
        return rate

    def get_compartment_outflows(self, compartment):
        """Get list of all flows going out from a compartment

        Parameters
        ----------
        compartment : Compartment or str
            Get outflows for this compartment

        Returns
        -------
        list
            Pairs of compartments and symbolic rates

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.get_compartment_outflows("CENTRAL")
        [(Output(), CL/V)]
        """
        if isinstance(compartment, str):
            compartment = self.find_compartment(compartment)
        flows = []
        for node in self._g.successors(compartment):
            flow = self.get_flow(compartment, node)
            flows.append((node, flow))
        return flows

    def get_compartment_inflows(self, compartment):
        """Get list of all flows going in to a compartment

        Parameters
        ----------
        compartment : Compartment or str
            Get inflows to this compartment

        Returns
        -------
        list
            Pairs of compartments and symbolic rates

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.get_compartment_inflows(output)
        [(Compartment(CENTRAL, amount=A_CENTRAL, dose=Bolus(AMT, admid=1)), CL/V)]
        """
        if isinstance(compartment, str):
            compartment = self.find_compartment(compartment)
        flows = []
        for node in self._g.predecessors(compartment):
            flow = self.get_flow(node, compartment)
            flows.append((node, flow))
        return flows

    def get_bidirectionals(self, compartment):
        """Get list of all compartments with bidirectional flow from/to a compartment

        Parameters
        ----------
        compartment : Compartment or str
            Compartment of interest

        Returns
        -------
        list
            Compartments with bidirectional flow

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> central = model.statements.ode_system.central_compartment
        >>> model.statements.ode_system.get_bidirectionals(central)
        []
        """
        if isinstance(compartment, str):
            compartment = self.find_compartment(compartment)
        comps = []
        for node in self._g.predecessors(compartment):
            if self._g.has_edge(compartment, node):
                comps.append(node)
        return comps

    def find_compartment(self, name):
        """Find a compartment using its name

        Parameters
        ----------
        name : str
            Name of compartment to find

        Returns
        -------
        Compartment
            Compartment named name or None if not found

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> central = model.statements.ode_system.find_compartment("CENTRAL")
        >>> central
        Compartment(CENTRAL, amount=A_CENTRAL, dose=Bolus(AMT, admid=1))
        """
        for comp in _comps(self._g):
            if comp.name == name:
                return comp
        else:
            return None

    def get_n_connected(self, comp):
        """Get the number of compartments connected to a compartment

        Parameters
        ----------
        comp : Compartment or str
            The compartment

        Returns
        -------
        int
            Number of compartments connected to comp

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.get_n_connected("CENTRAL")
        0
        """
        out_comps = {c for c, _ in self.get_compartment_outflows(comp)}
        in_comps = {c for c, _ in self.get_compartment_inflows(comp)}
        return len((out_comps | in_comps) - {output})

    @property
    def dosing_compartment(self):
        """The dosing compartment

        A dosing compartment is a compartment that receives an input dose. Multiple
        dose compartments are supported. The order of dose compartments is defined
        to put the central compartment last.

        Returns
        -------
        tuple
            A tuple of dose compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.dosing_compartment[0]
        Compartment(CENTRAL, amount=A_CENTRAL, dose=Bolus(AMT, admid=1))
        """
        dosing_comps = tuple()
        for node in _comps(self._g):
            if node.dose is not None:
                if node.name != self.central_compartment.name:
                    dosing_comps = (node,) + dosing_comps
                else:
                    dosing_comps = dosing_comps + (node,)

        if len(dosing_comps) != 0:
            return dosing_comps

        raise ValueError('No dosing compartment exists')

    @property
    def central_compartment(self):
        """The central compartment

        The central compartment is defined to be the compartment that has an outward flow
        to the output compartment. Only one central compartment is supported.

        Returns
        -------
        Compartment
            Central compartment

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.central_compartment
        Compartment(CENTRAL, amount=A_CENTRAL, dose=Bolus(AMT, admid=1))
        """
        try:
            # E.g. TMDD models have more than one input
            central = list(self._g.predecessors(output))[-1]
        except IndexError:
            raise ValueError('Cannot find central compartment')
        return central

    @property
    def peripheral_compartments(self):
        """Find perihperal compartments

        A peripheral compartment is defined as having one flow to the central compartment and
        one flow from the central compartment.

        Returns
        -------
        list of compartments
            Peripheral compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.peripheral_compartments
        []
        """
        central = self.central_compartment
        oneout = {node for node, out_degree in self._g.out_degree() if out_degree == 1}
        onein = {node for node, in_degree in self._g.in_degree() if in_degree == 1}
        cout = {comp for comp in oneout if self.get_flow(comp, central) != 0}
        cin = {comp for comp in onein if self.get_flow(central, comp) != 0}
        peripherals = list(cout & cin)
        # Return in deterministic order
        peripherals = sorted(peripherals, key=lambda comp: comp.name)
        return peripherals

    def find_transit_compartments(self, statements: Statements) -> List[Compartment]:
        """Find all transit compartments

        Transit compartments are a chain of compartments with the same out rate starting from
        the dose compartment. Because one single transit compartment cannot be distinguished
        from one depot compartment such compartment will be defined to be a depot and not
        a transit compartment.

        Returns
        -------
        list of compartments
            Transit compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.find_transit_compartments(model.statements)
        []
        """
        transits = []
        comp = self.dosing_compartment[0]
        if len(self.get_compartment_inflows(comp)) != 0:
            return transits
        outflows = self.get_compartment_outflows(comp)
        if len(outflows) != 1:
            return transits
        transits.append(comp)
        comp, rate = outflows[0]
        rate = statements.before_odes.full_expression(rate)
        while True:
            if len(self.get_compartment_inflows(comp)) != 1:
                break
            outflows = self.get_compartment_outflows(comp)
            if len(outflows) != 1:
                break
            next_comp, next_rate = outflows[0]
            next_rate = statements.before_odes.full_expression(next_rate)
            if rate != next_rate:
                break
            transits.append(comp)
            comp = next_comp
        # Special case of one transit directly into central is not defined as a transit
        # Also not central itself
        central = self.central_compartment
        if len(transits) == 1 and (
            self.get_flow(transits[0], central) != 0 or transits[0] == central
        ):
            return []
        else:
            return transits

    def find_depot(self, statements):
        """Find the depot compartment

        The depot compartment is defined to be the compartment that only has out flow to the
        central compartment, but no flow from the central compartment.

        Returns
        -------
        Compartment
            Depot compartment

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model, set_first_order_absorption
        >>> model = load_example_model("pheno")
        >>> model = set_first_order_absorption(model)
        >>> model.statements.ode_system.find_depot(model.statements)
        Compartment(DEPOT, amount=A_DEPOT, dose=Bolus(AMT, admid=1))
        """
        transits = self.find_transit_compartments(statements)
        depot = self._find_depot()
        if depot in transits:
            depot = None
        return depot

    def _find_depot(self):
        central = self.central_compartment
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
        """Compartmental matrix of the compartmental system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model, set_first_order_absorption
        >>> import sympy
        >>> model = load_example_model("pheno")
        >>> sympy.pprint(model.statements.ode_system.compartmental_matrix)
        ⎡-CL ⎤
        ⎢────⎥
        ⎣ V  ⎦
        """
        nodes = self._order_compartments()
        size = len(nodes)
        f = sympy.zeros(size)
        for i in range(0, size):
            from_comp = nodes[i]
            diagsum = 0
            for j in range(0, size):
                to_comp = nodes[j]
                rate = self.get_flow(from_comp, to_comp)
                if i != j:
                    f[j, i] = rate
                diagsum += rate
            outrate = self.get_flow(from_comp, output)
            f[i, i] = -outrate - diagsum
        return f

    @property
    def amounts(self):
        """Column vector of amounts for all compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> import sympy
        >>> model = load_example_model("pheno")
        >>> sympy.pprint(model.statements.ode_system.amounts)
        [A_CENTRAL]
        """
        ordered_cmts = self._order_compartments()
        amts = [cmt.amount for cmt in ordered_cmts]
        return sympy.Matrix(amts)

    @property
    def compartment_names(self) -> List[str]:
        """Names of all compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.compartment_names
        ['CENTRAL']
        """
        ordered_cmts = self._order_compartments()
        names = [cmt.name for cmt in ordered_cmts]
        return names

    def _order_compartments(self):
        """Return list of all compartments in canonical order"""
        try:
            dosecmt = self.dosing_compartment[0]
        except ValueError:
            # Fallback for cases where no dose is available (yet)
            return list(_comps(self._g))
        # Order compartments

        def sortfunc(x):
            a = list(x)
            if output in a:
                a.remove(output)
            a = sorted(a, key=lambda x: x.name)
            return iter(a)

        nodes = list(nx.bfs_tree(self._g, dosecmt, sort_neighbors=sortfunc))
        remaining = set(self._g.nodes) - {output} - set(nodes)
        while remaining:  # Disjoint graph
            comp = sorted(remaining, key=lambda x: x.name)[0]
            # Start with a compartment having a zero order input
            for c in remaining:
                if comp is None and c.input != 0:
                    comp = c
            ordered_remaining = list(nx.bfs_tree(self._g, comp, sort_neighbors=sortfunc))
            cleaned_ordered_remaining = []
            for c in ordered_remaining:
                if c not in set(nodes):
                    cleaned_ordered_remaining.append(c)
            nodes += cleaned_ordered_remaining
            remaining = set(self._g.nodes) - {output} - set(nodes)
        return nodes

    @property
    def zero_order_inputs(self):
        """Vector of all zero order inputs to each compartment

        Example
        -------
        >>> from pharmpy.modeling import load_example_model, set_zero_order_absorption
        >>> import sympy
        >>> model = load_example_model("pheno")
        >>> sympy.pprint(model.statements.ode_system.zero_order_inputs)
        [0]

        """
        inputs = [node.input for node in self._order_compartments()]
        return sympy.Matrix(inputs)

    def __len__(self):
        """The number of compartments"""
        return len(self._g.nodes) - 1

    def _repr_html_(self):
        # Use Unicode art for now. There should be ways of drawing networkx
        s = str(self)
        return f'<pre>{s}</pre>'

    def __repr__(self):
        def lag_string(comp):
            return f"\ntlag={comp.lag_time}" if comp.lag_time != 0 else ""

        def f_string(comp):
            return f"\nF={comp.bioavailability}" if comp.bioavailability != 1 else ""

        def comp_string(comp):
            return comp.name + lag_string(comp) + f_string(comp)

        current = self.dosing_compartment[0]
        comp_height = 0
        comp_width = 0

        while True:
            bidirects = self.get_bidirectionals(current)
            outflows = self.get_compartment_outflows(current)
            comp_height = max(comp_height, len(bidirects) + 1)
            comp_width += 1
            for comp, rate in outflows:
                if comp not in bidirects and comp != output:
                    current = comp
                    break
            else:
                break

        noutput = len(self.get_compartment_inflows(output))
        have_zo_input = int(not is_zero_matrix(self.zero_order_inputs))
        comp_nrows = comp_height * 2 - 1 + (1 if noutput > 1 else 0)
        nrows = comp_nrows + have_zo_input
        ncols = comp_width * 2
        grid = unicode.Grid(nrows, ncols)

        current = self.dosing_compartment[0]
        col = 0
        if comp_nrows == 1 or comp_nrows == 2:
            main_row = 0 + have_zo_input
        else:
            main_row = 2 + have_zo_input

        if have_zo_input:
            # Assuming a fully linear layout
            for i, zo in enumerate(self.zero_order_inputs):
                if zo != 0:
                    try:
                        grid.set(0, i * 2, unicode.VerticalArrow(str(zo)))
                    except IndexError:
                        warnings.warn(
                            """The ODE system cannot be printed. Try statements.before_odes
                            and statements.after_odes instead."""
                        )
        while True:
            bidirects = self.get_bidirectionals(current)
            outflows = self.get_compartment_outflows(current)
            comp_box = unicode.Box(comp_string(current))
            grid.set(main_row, col, comp_box)

            if bidirects:
                grid.set(0, col, unicode.Box(comp_string(bidirects[0])))
                grid.set(
                    1,
                    col,
                    unicode.DualVerticalArrows(
                        str(self.get_flow(current, bidirects[0])),
                        str(self.get_flow(bidirects[0], current)),
                    ),
                )
            if len(bidirects) > 1:
                grid.set(4, col, unicode.Box(comp_string(bidirects[1])))
                grid.set(
                    3,
                    col,
                    unicode.DualVerticalArrows(
                        str(self.get_flow(bidirects[1], current)),
                        str(self.get_flow(current, bidirects[1])),
                    ),
                )

            for comp, rate in outflows:
                if comp not in bidirects and comp != output:
                    next_comp = comp
                    break
            else:
                next_comp = None

            if next_comp:
                rate = self.get_flow(current, next_comp)
            else:
                rate = self.get_flow(current, output)
            arrow = unicode.Arrow(str(rate))
            grid.set(main_row, col + 1, arrow)
            if next_comp and noutput > 1:
                rate = self.get_flow(current, output)
                arrow = unicode.VerticalArrow(str(rate))
                grid.set(main_row + 1, col, arrow)
            col += 2
            current = next_comp
            if not current:
                break

        dose = self.dosing_compartment[0].dose
        s = str(dose) + '\n' + str(grid).rstrip()
        return s


class Compartment:
    """Compartment for a compartmental system

    Parameters
    ----------
    name : str
        Compartment name
    dose : Dose
        Dose object for dose into this compartment. Default None for no dose.
    input : Expression
        Expression for other inputs to the compartment
    lag_time : Expression
        Lag time for doses entering this compartment. Default 0
    bioavailability : Expression
        Bioavailability fraction for doses entering this compartment. Default 1

    Examples
    --------
    >>> from pharmpy.model import Bolus, Compartment
    >>> comp = Compartment.create("CENTRAL")
    >>> comp
    Compartment(CENTRAL, amount=A_CENTRAL)
    >>> comp = Compartment.create("DEPOT", lag_time="ALAG")
    >>> comp
    Compartment(DEPOT, amount=A_DEPOT, lag_time=ALAG)
    >>> dose = Bolus.create("AMT")
    >>> comp = Compartment.create("DEPOT", dose=dose)
    >>> comp
    Compartment(DEPOT, amount=A_DEPOT, dose=Bolus(AMT, admid=1))
    """

    def __init__(
        self,
        name,
        amount=None,
        dose=None,
        input=sympy.Integer(0),
        lag_time=sympy.Integer(0),
        bioavailability=sympy.Integer(1),
    ):
        self._name = name
        self._amount = amount
        self._dose = dose
        self._input = input
        self._lag_time = lag_time
        self._bioavailability = bioavailability

    @classmethod
    def create(
        cls,
        name,
        amount=None,
        dose=None,
        input=sympy.Integer(0),
        lag_time=sympy.Integer(0),
        bioavailability=sympy.Integer(1),
    ):
        if not isinstance(name, str):
            raise TypeError("Name of a Compartment must be of string type")
        if amount is not None:
            amount = parse_expr(amount)
        else:
            amount = sympy.Symbol(f'A_{name}')
        if dose is not None and not isinstance(dose, Dose):
            raise TypeError("dose must be of Dose type (or None)")
        input = parse_expr(input)
        lag_time = parse_expr(lag_time)
        bioavailability = parse_expr(bioavailability)
        return cls(
            name=name,
            amount=amount,
            dose=dose,
            input=input,
            lag_time=lag_time,
            bioavailability=bioavailability,
        )

    def replace(self, **kwargs):
        name = kwargs.get("name", self._name)
        amount = kwargs.get("amount", self._amount)
        dose = kwargs.get("dose", self._dose)
        input = kwargs.get("input", self._input)
        lag_time = kwargs.get("lag_time", self._lag_time)
        bioavailability = kwargs.get("bioavailability", self._bioavailability)
        return Compartment.create(
            name=name,
            amount=amount,
            dose=dose,
            input=input,
            lag_time=lag_time,
            bioavailability=bioavailability,
        )

    @property
    def name(self):
        """Compartment name"""
        return self._name

    @property
    def amount(self):
        """Compartment amount symbol"""
        return self._amount

    @property
    def dose(self):
        return self._dose

    @property
    def input(self):
        return self._input

    @property
    def lag_time(self):
        """Lag time for doses into compartment"""
        return self._lag_time

    @property
    def bioavailability(self):
        """Bioavailability fraction for doses into compartment"""
        return self._bioavailability

    @property
    def free_symbols(self):
        """Get set of all free symbols in the compartment

        Examples
        --------
        >>> from pharmpy.model import Bolus, Compartment
        >>> dose = Bolus.create("AMT")
        >>> comp = Compartment.create("CENTRAL", dose=dose, lag_time="ALAG")
        >>> comp.free_symbols  # doctest: +SKIP
        {A_CENTRAL, ALAG, AMT}
        """
        symbs = set()
        if self.dose is not None:
            symbs |= self.dose.free_symbols
        symbs |= self.input.free_symbols
        symbs |= self.lag_time.free_symbols
        symbs |= self.bioavailability.free_symbols
        return symbs

    def subs(self, substitutions):
        """Substitute expressions or symbols in compartment

        Examples
        --------
        >>> from pharmpy.model import Bolus, Compartment
        >>> dose = Bolus.create("AMT")
        >>> comp = Compartment.create("CENTRAL", dose=dose)
        >>> comp.subs({"AMT": "DOSE"})
        Compartment(CENTRAL, amount=A_CENTRAL, dose=Bolus(DOSE, admid=1))
        """
        if self.dose is not None:
            dose = self.dose.subs(substitutions)
        else:
            dose = None
        return Compartment(
            self.name,
            amount=subs(self._amount, substitutions),
            dose=dose,
            input=subs(self._input, substitutions),
            lag_time=subs(self._lag_time, substitutions),
            bioavailability=subs(self._bioavailability, substitutions),
        )

    def __eq__(self, other):
        return (
            isinstance(other, Compartment)
            and self._name == other._name
            and self._amount == other._amount
            and self._dose == other._dose
            and self._input == other._input
            and self._lag_time == other._lag_time
            and self._bioavailability == other._bioavailability
        )

    def __hash__(self):
        return hash(
            (
                self._name,
                self._amount,
                self._dose,
                self._input,
                self._lag_time,
                self._bioavailability,
            )
        )

    def to_dict(self):
        if self._dose is None:
            dose = None
        else:
            dose = self._dose.to_dict()
        return {
            'class': 'Compartment',
            'name': self._name,
            'amount': sympy.srepr(self._amount),
            'dose': dose,
            'input': sympy.srepr(self._input),
            'lag_time': sympy.srepr(self._lag_time),
            'bioavailability': sympy.srepr(self._bioavailability),
        }

    @classmethod
    def from_dict(cls, d):
        if d['dose'] is None:
            dose = None
        elif d['dose']['class'] == 'Bolus':
            dose = Bolus.from_dict(d['dose'])
        else:
            dose = Infusion.from_dict(d['dose'])
        return cls(
            name=d['name'],
            amount=sympy.parse_expr(d['amount']),
            dose=dose,
            input=sympy.parse_expr(d['input']),
            lag_time=sympy.parse_expr(d['lag_time']),
            bioavailability=sympy.parse_expr(d['bioavailability']),
        )

    def __repr__(self):
        lag = '' if self.lag_time == 0 else f', lag_time={self._lag_time}'
        dose = '' if self.dose is None else f', dose={self._dose}'
        input = '' if self.input == 0 else f', input={self._input}'
        bioavailability = (
            '' if self.bioavailability == 1 else f', bioavailability={self._bioavailability}'
        )
        return f'Compartment({self.name}, amount={self._amount}{dose}{input}{lag}{bioavailability})'


class Dose(ABC):
    """Abstract base class for different types of doses"""

    @abstractmethod
    def subs(self, substitutions):
        ...

    @property
    @abstractmethod
    def free_symbols(self):
        ...


class Bolus(Dose):
    """A Bolus dose

    Parameters
    ----------
    amount : symbol
        Symbolic amount of dose
    admid : int
        Administration ID

    Examples
    --------
    >>> from pharmpy.model import Bolus
    >>> dose = Bolus.create("AMT")
    >>> dose
    Bolus(AMT, admid=1)
    """

    def __init__(self, amount, admid=1):
        self._amount = amount
        self._admid = admid

    @classmethod
    def create(cls, amount, admid=1):
        return cls(parse_expr(amount), admid=admid)

    def replace(self, **kwargs):
        amount = kwargs.get("amount", self._amount)
        admid = kwargs.get("admid", self._admid)
        return Bolus.create(amount=amount, admid=admid)

    @property
    def amount(self):
        """Symbolic amount of dose"""
        return self._amount

    @property
    def admid(self):
        """Administration ID of dose"""
        return self._admid

    @property
    def free_symbols(self):
        """Get set of all free symbols in the dose

        Examples
        --------
        >>> from pharmpy.model import Bolus
        >>> dose = Bolus.create("AMT")
        >>> dose.free_symbols
        {AMT}
        """
        return {self.amount}

    def subs(self, substitutions):
        """Substitute expressions or symbols in dose

        Parameters
        ----------
        substitutions : dict
            Dictionary of from, to pairs

        Examples
        --------
        >>> from pharmpy.model import Bolus
        >>> dose = Bolus.create("AMT")
        >>> dose.subs({'AMT': 'DOSE'})
        Bolus(DOSE, admid=1)
        """
        return Bolus(subs(self.amount, substitutions, simultaneous=True), admid=self._admid)

    def __eq__(self, other):
        return (
            isinstance(other, Bolus)
            and self._amount == other._amount
            and self._admid == other._admid
        )

    def __hash__(self):
        return hash((self._amount, self._admid))

    def to_dict(self):
        return {'class': 'Bolus', 'amount': sympy.srepr(self._amount), 'admid': self._admid}

    @classmethod
    def from_dict(cls, d):
        return cls(amount=sympy.parse_expr(d['amount']), admid=d['admid'])

    def __repr__(self):
        return f'Bolus({self.amount}, admid={self._admid})'


class Infusion(Dose):
    """An infusion dose

    Parameters
    ----------
    amount : expression
        Symbolic amount of dose
    admid : int
        Administration ID
    rate : expression
        Symbolic rate. Mutually exclusive with duration
    duration : expression
        Symbolic duration. Mutually excluseive with rate

    Examples
    --------
    >>> from pharmpy.model import Infusion
    >>> dose = Infusion("AMT", duration="D1")
    >>> dose
    Infusion(AMT, admid=1, duration=D1)
    >>> dose = Infusion("AMT", rate="R1")
    >>> dose
    Infusion(AMT, admid=1, rate=R1)
    """

    def __init__(self, amount, admid=1, rate=None, duration=None):
        self._amount = amount
        self._admid = admid
        self._rate = rate
        self._duration = duration

    @classmethod
    def create(cls, amount, admid=1, rate=None, duration=None):
        if rate is None and duration is None:
            raise ValueError('Need rate or duration for Infusion')
        if rate is not None and duration is not None:
            raise ValueError('Cannot have both rate and duration for Infusion')
        if rate is not None:
            rate = parse_expr(rate)
        else:
            duration = parse_expr(duration)
        return cls(parse_expr(amount), admid=admid, rate=rate, duration=duration)

    def replace(self, **kwargs):
        amount = kwargs.get("amount", self._amount)
        admid = kwargs.get("admid", self._admid)
        rate = kwargs.get("rate", self._rate)
        duration = kwargs.get("duration", self._duration)
        return Infusion.create(amount=amount, admid=admid, rate=rate, duration=duration)

    @property
    def amount(self):
        """Symbolic amount of dose"""
        return self._amount

    @property
    def admid(self):
        """Administration ID"""
        return self._admid

    @property
    def rate(self):
        """Symbolic rate

        Mutually exclusive with duration.
        """
        return self._rate

    @property
    def duration(self):
        """Symbolc duration

        Mutually exclusive with rate.
        """
        return self._duration

    @property
    def free_symbols(self):
        """Get set of all free symbols in the dose

        Examples
        --------
        >>> from pharmpy.model import Infusion
        >>> dose = Infusion.create("AMT", rate="RATE")
        >>> dose.free_symbols   # doctest: +SKIP
        {AMT, RATE}
        """
        if self.rate is not None:
            symbs = self.rate.free_symbols
        else:
            assert self.duration is not None
            symbs = self.duration.free_symbols
        return symbs | self.amount.free_symbols

    def subs(self, substitutions):
        """Substitute expressions or symbols in dose

        Parameters
        ----------
        substitutions : dict
            Dictionary of from, to pairs

        Examples
        --------
        >>> from pharmpy.model import Infusion
        >>> dose = Infusion.create("AMT", duration="DUR")
        >>> dose.subs({'DUR': 'D1'})
        Infusion(AMT, admid=1, duration=D1)
        """
        amount = subs(self.amount, substitutions, simultaneous=True)
        if self.rate is not None:
            rate = subs(self.rate, substitutions, simultaneous=True)
            duration = None
        else:
            rate = None
            duration = subs(self.duration, substitutions, simultaneous=True)
        return Infusion(amount, admid=self._admid, rate=rate, duration=duration)

    def __eq__(self, other):
        return (
            isinstance(other, Infusion)
            and self._admid == other._admid
            and self._rate == other._rate
            and self._duration == other._duration
            and self._amount == other._amount
        )

    def __hash__(self):
        return hash((self._admid, self._rate, self._duration, self._amount))

    def to_dict(self):
        return {
            'class': 'Infusion',
            'amount': sympy.srepr(self._amount),
            'rate': sympy.srepr(self._rate),
            'duration': sympy.srepr(self._duration),
            'admid': self._admid,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            amount=sympy.parse_expr(d['amount']),
            admid=d['admid'],
            rate=sympy.parse_expr(d['rate']),
            duration=sympy.parse_expr(d['duration']),
        )

    def __repr__(self):
        if self.rate is not None:
            arg = f'rate={self.rate}'
        else:
            arg = f'duration={self.duration}'
        return f'Infusion({self.amount}, admid={self._admid}, {arg})'


class Statements(Sequence, Immutable):
    """A sequence of symbolic statements describing the model

    Two types of statements are supported: Assignment and ODESystem.
    A Statements object can have 0 or 1 ODESystem. The order of
    the statements is significant and the same symbol can be assigned
    to multiple times.

    Parameters
    ----------
    statements : list or Statements
        A list of Statement or another Statements to populate this object
    """

    def __init__(self, statements: Union[None, Statements, Iterable[Statement]] = None):
        if isinstance(statements, Statements):
            self._statements = statements._statements
        elif statements is None:
            self._statements = ()
        else:
            self._statements = tuple(statements)

    @overload
    def __getitem__(self, ind: slice) -> Statements:
        ...

    @overload
    def __getitem__(self, ind: int) -> Statement:
        ...

    def __getitem__(self, ind):
        if isinstance(ind, slice):
            return Statements(self._statements[ind])
        else:
            return self._statements[ind]

    def __len__(self):
        return len(self._statements)

    def __add__(self, other):
        if isinstance(other, Statements):
            return Statements(self._statements + other._statements)
        elif isinstance(other, Statement):
            return Statements(self._statements + (other,))
        else:
            return Statements(self._statements + tuple(other))

    def __radd__(self, other):
        if isinstance(other, Statement):
            return Statements((other,) + self._statements)
        else:
            return Statements(tuple(other) + self._statements)

    @property
    def free_symbols(self):
        """Get a set of all free symbols

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.free_symbols   # doctest: +SKIP
        {AMT, APGR, A_CENTRAL, BTIME, CL, DV, EPS(1), ETA(1), ETA(2), F, IPRED, IRES, IWRES, S1,
        TAD, THETA(1), THETA(2), THETA(3), TIME, TVCL, TVV, V, W, WGT, Y, t}

        """
        symbols = set()
        for assignment in self:
            symbols |= assignment.free_symbols
        return symbols

    def _get_ode_system_index(self):
        return next(
            map(lambda t: t[0], filter(lambda t: isinstance(t[1], ODESystem), enumerate(self))), -1
        )

    @property
    def ode_system(self) -> Optional[ODESystem]:
        """Returns the ODE system of the model or None if the model doesn't have an ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system
        Bolus(AMT, admid=1)
        ┌───────┐
        │CENTRAL│──CL/V→
        └───────┘
        """
        i = self._get_ode_system_index()
        ret = None if i == -1 else self[i]
        assert ret is None or isinstance(ret, ODESystem)
        return ret

    @property
    def before_odes(self):
        """All statements before the ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.before_odes
                ⎧TIME  for AMT > 0
                ⎨
        BTIME = ⎩ 0     otherwise
        TAD = -BTIME + TIME
        TVCL = PTVCL⋅WGT
        TVV = PTVV⋅WGT
              ⎧TVV⋅(THETA₃ + 1)  for APGR < 5
              ⎨
        TVV = ⎩       TVV           otherwise
                   ETA₁
        CL = TVCL⋅ℯ
                 ETA₂
        V = TVV⋅ℯ
        S₁ = V
        """
        i = self._get_ode_system_index()
        return self if i == -1 else self[:i]

    @property
    def after_odes(self):
        """All statements after the ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.after_odes
            A_CENTRAL
            ─────────
        F =     S₁
        W = F
        Y = EPS₁⋅W + F
        IPRED = F
        IRES = DV - IPRED
                 IRES
                 ────
        IWRES =   W
        """
        i = self._get_ode_system_index()
        return Statements() if i == -1 else self[i + 1 :]

    @property
    def error(self):
        """All statements after the ODE system or the whole model if no ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.error
            A_CENTRAL
            ─────────
        F =     S₁
        W = F
        Y = EPS₁⋅W + F
        IPRED = F
        IRES = DV - IPRED
                 IRES
                 ────
        IWRES =   W
        """
        i = self._get_ode_system_index()
        return self if i == -1 else self[i + 1 :]

    def subs(self, substitutions):
        """Substitute symbols in all statements.

        Parameters
        ----------
        substitutions : dict
            Old-new pairs(can be type str or sympy symbol)

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> stats = model.statements.subs({'WGT': 'WT'})
        >>> stats.before_odes
                ⎧TIME  for AMT > 0
                ⎨
        BTIME = ⎩ 0     otherwise
        TAD = -BTIME + TIME
        TVCL = PTVCL⋅WT
        TVV = PTVV⋅WT
              ⎧TVV⋅(THETA₃ + 1)  for APGR < 5
              ⎨
        TVV = ⎩       TVV           otherwise
                   ETA₁
        CL = TVCL⋅ℯ
                 ETA₂
        V = TVV⋅ℯ
        S₁ = V
        """
        return Statements(s.subs(substitutions) for s in self)

    def _lookup_last_assignment(
        self, symbol: Union[str, sympy.Symbol]
    ) -> Tuple[Optional[int], Optional[Assignment]]:
        if isinstance(symbol, str):
            symbol = sympy.Symbol(symbol)
        ind = None
        assignment = None
        for i, statement in enumerate(self):
            if isinstance(statement, Assignment):
                if statement.symbol.name == symbol.name:  # Threadsafe and faster to compare names
                    ind = i
                    assignment = statement
        return ind, assignment

    def find_assignment(self, symbol: Union[str, sympy.Symbol]) -> Optional[Assignment]:
        """Returns last assignment of symbol

        Parameters
        ----------
        symbol : sympy.Symbol or str
            Symbol to look for

        Returns
        -------
        Assignment or None
            An Assignment or None if no assignment to symbol exists

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.find_assignment("CL")
                   ETA₁
        CL = TVCL⋅ℯ
        """
        return self._lookup_last_assignment(symbol)[1]

    def find_assignment_index(self, symbol: Union[str, sympy.Symbol]) -> Optional[int]:
        """Returns index of last assignment of symbol

        Parameters
        ----------
        symbol : sympy.Symbol or str
            Symbol to look for

        Returns
        -------
        int or None
            Index of Assignment or None if no assignment to symbol exists

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.find_assignment_index("CL")
        5
        """
        return self._lookup_last_assignment(symbol)[0]

    def reassign(self, symbol: Union[str, sympy.Symbol], expression: Union[str, sympy.Expr]):
        """Reassign symbol to expression

        Set symbol to be expression and remove all previous assignments of symbol

        Parameters
        ----------
        symbol : sypmpy.Symbol or str
            Symbol to reassign
        expression : sympy.Expr or str
            The new expression to assign to symbol

        Return
        ------
        Statements
            Updated statements

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.reassign("CL", "TVCL + eta")   # doctest: +SKIP
        """
        if isinstance(symbol, str):
            symbol = sympy.Symbol(symbol)
        if isinstance(expression, str):
            expression = parse_expr(expression)

        last = True
        new = list(self._statements)
        for i, stat in zip(range(len(new) - 1, -1, -1), reversed(new)):
            if isinstance(stat, Assignment) and stat.symbol == symbol:
                if last:
                    new[i] = Assignment(symbol, expression)
                    last = False
                else:
                    del new[i]
        return Statements(new)

    def _create_dependency_graph(self):
        """Create a graph of dependencies between statements"""
        graph = nx.DiGraph()
        for i in range(len(self) - 1, -1, -1):
            rhs = self[i].rhs_symbols
            for j in range(i - 1, -1, -1):
                statement = self[j]
                if (
                    isinstance(statement, Assignment)
                    and statement.symbol in rhs
                    or isinstance(statement, ODESystem)
                    and not rhs.isdisjoint(statement.amounts)
                ):
                    graph.add_edge(i, j)
        return graph

    def direct_dependencies(self, statement):
        """Find all direct dependencies of a statement

        Parameters
        ----------
        statement : Statement
            Input statement

        Returns
        -------
        Statements
            Direct dependency statements

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> odes = model.statements.ode_system
        >>> model.statements.direct_dependencies(odes)
                   ETA₁
        CL = TVCL⋅ℯ
                 ETA₂
        V = TVV⋅ℯ
        """
        g = self._create_dependency_graph()
        index = self.index(statement)
        succ = sorted(list(g.successors(index)))
        stats = Statements()
        stats._statements = [self[i] for i in succ]
        return stats

    def dependencies(self, symbol_or_statement):
        """Find all dependencies of a symbol or statement

        Parameters
        ----------
        symbol : Symbol, str or Statement
            Input symbol or statement

        Returns
        -------
        set
            Set of symbols

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.dependencies("CL")   # doctest: +SKIP
        {ETA(1), THETA(1), WGT}
        """
        if isinstance(symbol_or_statement, Statement):
            i = self.index(symbol_or_statement)
        else:
            symbol = (
                sympy.Symbol(symbol_or_statement)
                if isinstance(symbol_or_statement, str)
                else symbol_or_statement
            )
            for i in range(len(self) - 1, -1, -1):
                statement = self[i]
                if (
                    isinstance(statement, Assignment)
                    and statement.symbol == symbol
                    or isinstance(statement, ODESystem)
                    and symbol in statement.amounts
                ):
                    break
            else:
                raise KeyError(f"Could not find symbol {symbol}")
        g = self._create_dependency_graph()
        symbs = self[i].rhs_symbols
        if i == 0 or not g:
            # Special case for models with only one statement or no dependent statements
            return symbs
        for j, _ in nx.bfs_predecessors(g, i, sort_neighbors=lambda x: reversed(sorted(x))):
            statement = self[j]
            if isinstance(statement, Assignment):
                symbs -= {statement.symbol}
            else:
                assert isinstance(statement, ODESystem)
                symbs -= set(statement.amounts)
            symbs |= statement.rhs_symbols
        return symbs

    def remove_symbol_definitions(self, symbols, statement):
        """Remove symbols and dependencies not used elsewhere

        If the statement no longer depends on the specified
        symbols, this method will make sure that the definitions
        of these symbols will be removed unless they are dependencies
        of other statements.

        Parameters
        ----------
        symbols : iterable
            Iterable of symbols no longer used in the statement
        statement : Statement
            Statement from which the symbols were removed
        """
        graph = self._create_dependency_graph()
        removed_ind = self._statements.index(statement)
        # Statements defining symbols and dependencies
        candidates = set()
        symbols_set = set(symbols)
        for i in range(removed_ind - 1, -1, -1):
            stat = self[i]
            if isinstance(stat, Assignment) and stat.symbol in symbols_set:
                candidates.add(i)
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
        return Statements(tuple(self[i] for i in range(len(self)) if i not in remove))

    def full_expression(self, expression):
        """Expand an expression into its full definition

        Parameters
        ----------
        expression : expression or str
            Expression to expand. A string will be converted to an expression.

        Return
        ------
        expression
            Expanded expression

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.before_odes.full_expression("CL")
        PTVCL*WGT*exp(ETA_1)
        """
        if isinstance(expression, str):
            expression = parse_expr(expression)
        for statement in reversed(self):
            if isinstance(statement, ODESystem):
                raise ValueError(
                    "ODESystem not supported by full_expression. Use the properties before_odes "
                    "or after_odes."
                )
            expression = subs(
                expression, {statement.symbol: statement.expression}, simultaneous=True
            )
        return expression

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        else:
            for first, second in zip(self, other):
                if first != second:
                    return False
        return True

    def __hash__(self):
        return hash(self._statements)

    def to_dict(self):
        stats = tuple(s.to_dict() for s in self)
        return {'statements': stats}

    @classmethod
    def from_dict(cls, d):
        statements = []
        for sdict in d['statements']:
            if sdict['class'] == 'Assignment':
                s = Assignment.from_dict(sdict)
            else:
                s = CompartmentalSystem.from_dict(sdict)
            statements.append(s)
        return cls(tuple(statements))

    def __repr__(self):
        return '\n'.join([repr(statement) for statement in self])

    def _repr_html_(self):
        html = r'\begin{align*}'
        for statement in self:
            if hasattr(statement, '_repr_html_'):
                html += '\\end{align*}'
                s = statement._repr_html_()
                html += s + '\\begin{align*}'
            else:
                s = f'${statement._repr_latex_()}$'
                s = s.replace('=', '&=')
                s = s.replace('$', '')
                s = s + r'\\'
                html += s
        return html + '\\end{align*}'
