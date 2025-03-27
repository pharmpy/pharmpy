from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, Self, Union, overload

import pharmpy.internals.unicode as unicode
from pharmpy.basic import BooleanExpr, Expr, Matrix, TExpr, TSymbol
from pharmpy.deps import networkx as nx
from pharmpy.deps import symengine, sympy
from pharmpy.internals.expr.assumptions import assume_all
from pharmpy.internals.expr.leaves import free_images, free_images_and_symbols
from pharmpy.internals.expr.ode import canonical_ode_rhs
from pharmpy.internals.immutable import Immutable, cache_method


class Statement(Immutable):
    """Abstract base class for all types of statements"""

    def __add__(self, other: Union[Statement, Statements, Iterable[Statement]]) -> Statements:
        if isinstance(other, Statements):
            return Statements((self,) + other._statements)
        elif isinstance(other, Statement):
            return Statements((self, other))
        elif isinstance(other, Iterable):
            return Statements.create((self,) + tuple(other))
        else:
            return NotImplemented

    def __radd__(self, other: Union[Statement, Iterable[Statement]]) -> Statements:
        if isinstance(other, Iterable):
            return Statements.create(tuple(other) + (self,))
        else:
            return NotImplemented

    @abstractmethod
    def subs(self, substitutions: Mapping[Expr, Expr]) -> Statement:
        pass

    @property
    @abstractmethod
    def free_symbols(self) -> set[Expr]:
        pass

    @property
    @abstractmethod
    def rhs_symbols(self) -> set[Expr]:
        pass


class Assignment(Statement):
    """Representation of variable assignment

    This class represents an assignment of an expression to a variable. Multiple assignments
    are combined together into a Statements object.

    Parameters
    ----------
    symbol : Expr
        Symbol of assignment
    expression : Expr
        Expression of assignment
    """

    def __init__(self, symbol: Expr, expression: Expr):
        self._symbol = symbol
        self._expression = expression

    @classmethod
    def create(cls, symbol: TExpr, expression: TExpr) -> Assignment:
        symbol = Expr(symbol)
        if not symbol.is_symbol():
            raise TypeError("symbol of Assignment must be a Symbol or str representing a symbol")
        expression = Expr(expression)
        # To avoid nested piecewises
        expression = expression.piecewise_fold()
        return cls(symbol, expression)

    def replace(self, **kwargs) -> Assignment:
        symbol = kwargs.get('symbol', self._symbol)
        expression = kwargs.get('expression', self._expression)
        return Assignment.create(symbol, expression)

    @property
    def symbol(self) -> Expr:
        """Symbol of statement"""
        return self._symbol

    @property
    def expression(self) -> Expr:
        """Expression of assignment"""
        return self._expression

    def subs(self, substitutions: Mapping[Expr, Expr]) -> Assignment:
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
        symbol = self.symbol.subs(substitutions)
        expression = self.expression.subs(substitutions)
        expression = expression.piecewise_fold()
        return Assignment(symbol, expression)

    @property
    def free_symbols(self) -> set[Expr]:
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
    def rhs_symbols(self) -> set[Expr]:
        """Get set of all free symbols in the right hand side expression

        Examples
        --------
        >>> from pharmpy.model import Assignment
        >>> a = Assignment.create('CL', 'POP_CL + ETA_CL')
        >>> a.rhs_symbols      # doctest: +SKIP
        {ETA_CL, POP_CL}

        """
        prefuncs = self._expression._sympy_().atoms(sympy.Function)
        from sympy.core.function import AppliedUndef

        # Allow applied undefined functions
        funcs = {Expr(f) for f in prefuncs if isinstance(f, AppliedUndef)}
        symbols = self._expression.free_symbols
        return funcs | symbols

    def __eq__(self, other: Any):
        if not isinstance(other, Assignment):
            return NotImplemented
        if hash(self) != hash(other):
            return False
        return self.symbol == other.symbol and self.expression == other.expression

    @cache_method
    def __hash__(self):
        return hash((self._symbol, self._expression))

    def to_dict(self) -> dict[str, Any]:
        return {
            'class': 'Assignment',
            'symbol': self._symbol.serialize(),
            'expression': self._expression.serialize(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Assignment:
        return cls(
            symbol=Expr.deserialize(d['symbol']), expression=Expr.deserialize(d['expression'])
        )

    def __repr__(self):
        expression = self._expression.unicode()
        lines = [line.rstrip() for line in expression.split('\n')]
        definition = f'{self._symbol.unicode()} = '
        s = ''
        for line in lines:
            if line == lines[-1]:
                s += definition + line + '\n'
            else:
                s += len(definition) * ' ' + line + '\n'
        return s.rstrip()

    def _repr_latex_(self) -> str:
        sym = self._symbol.latex()
        expr = self._expression.latex()
        return f'${sym} = {expr}$'


class CompartmentBase(Immutable):
    pass


class Output(CompartmentBase):
    def __new__(cls):
        # Singleton class
        if not hasattr(cls, 'instance'):
            cls.instance = super(Output, cls).__new__(cls)
        return cls.instance

    def __repr__(self):
        return "Output()"

    def __hash__(self):
        return 5267

    def to_dict(self) -> dict[str, str]:
        return {'class': 'Output'}

    @classmethod
    def from_dict(cls, d) -> Output:
        return cls.instance

    def __eq__(self, other: Any):
        if not isinstance(other, CompartmentBase):
            return NotImplemented
        return self is other


output = Output()


class CompartmentalSystemBuilder:
    """Builder for CompartmentalSystem"""

    def __init__(self, cs: Optional[CompartmentalSystem] = None):
        if cs:
            self._g = cs._g.copy()
        else:
            self._g = nx.DiGraph()
            self._g.add_node(output)  # Single output "compartment"

    def add_compartment(self, compartment: Compartment) -> None:
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

    def remove_compartment(self, compartment: Compartment) -> None:
        """Remove compartment from system

        Parameters
        ----------
        compartment : Compartment
            Compartment object to remove from system

        Examples
        --------
        >>> from pharmpy.model import Compartment, CompartmentalSystemBuilder
        >>> cb = CompartmentalSystemBuilder()
        >>> central = Compartment.create("CENTRAL")
        >>> cb.add_compartment(central)
        >>> cb.remove_compartment(central)
        """
        self._g.remove_node(compartment)

    def add_flow(self, source: Compartment, destination: CompartmentBase, rate: TExpr) -> None:
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
        >>> depot = Compartment.create("DEPOT")
        >>> cb.add_compartment(depot)
        >>> central = Compartment.create("CENTRAL")
        >>> cb.add_compartment(central)
        >>> cb.add_flow(depot, central, "KA")
        """
        self._g.add_edge(source, destination, rate=Expr(rate))

    def remove_flow(self, source: Compartment, destination: CompartmentBase) -> None:
        """Remove flow between two compartments

        Parameters
        ----------
        source : Compartment
            Source compartment
        destination : Compartment
            Destination compartment

        Examples
        --------
        >>> from pharmpy.model import Compartment, CompartmentalSystemBuilder
        >>> cb = CompartmentalSystemBuilder()
        >>> depot = Compartment.create("DEPOT")
        >>> cb.add_compartment(depot)
        >>> central = Compartment.create("CENTRAL")
        >>> cb.add_compartment(central)
        >>> cb.add_flow(depot, central, "KA")
        >>> cb.remove_flow(depot, central)
        """
        self._g.remove_edge(source, destination)

    def move_dose(
        self,
        source: Compartment,
        destination: Compartment,
        admid: Optional[int] = None,
    ) -> tuple[Compartment, Compartment]:
        """Move a dose input from one compartment to another

        Parameters
        ----------
        source : Compartment
            Source compartment
        destination : Compartment
            Destination compartment
        admid : int
            Move dose with specified admid, move all if None. Default is None.

        """
        if source is None:
            raise ValueError('Option `source` cannot be None')
        if destination is None:
            raise ValueError('Option `destination` cannot be None')
        if not source.doses:
            raise ValueError(f'No doses to move in compartment: `{source.name}`')

        if destination.doses:
            new_dest_dose = destination.doses
        else:
            new_dest_dose = tuple()
        if admid:
            new_source_dose = tuple([d for d in source.doses if d.admid != admid])
            new_dest_dose += tuple([d for d in source.doses if d.admid == admid])
        else:
            new_source_dose = tuple()
            new_dest_dose += source.doses
        new_source = source.replace(doses=new_source_dose)
        new_dest = destination.replace(doses=new_dest_dose)
        mapping = {source: new_source, destination: new_dest}
        nx.relabel_nodes(self._g, mapping, copy=False)
        return new_source, new_dest

    def set_dose(
        self,
        compartment: Compartment,
        dose: Optional[Union[Dose, tuple[Dose, ...]]],
    ) -> Compartment:
        """Set dose of compartment, replacing the previous.

        Parameters
        ----------
        compartment : Compartment
            Compartment for which to set dose
        dose : Dose
            New dose

        Returns
        -------
        Compartment
            The new updated compartment
        """
        if compartment is None:
            raise ValueError('Option `compartment` cannot be None')
        if dose is None:
            dose = tuple()
        elif isinstance(dose, Dose):
            dose = (dose,)

        new_comp = compartment.replace(doses=dose)
        mapping = {compartment: new_comp}
        nx.relabel_nodes(self._g, mapping, copy=False)
        return new_comp

    def add_dose(self, compartment: Compartment, dose: Union[Dose, tuple[Dose, ...]]):
        """Add dose to compartment.

        Parameters
        ----------
        compartment : Compartment
            Compartment for which to add dose
        dose : Dose
            New dose

        Returns
        -------
        Compartment
            The new updated compartment
        """
        if compartment is None:
            raise ValueError('Option `compartment` cannot be None')
        if dose is None:
            raise ValueError('Option `dose` cannot be None')
        elif isinstance(dose, Dose):
            dose = (dose,)

        new_comp = compartment.replace(doses=compartment.doses + dose)
        mapping = {compartment: new_comp}
        nx.relabel_nodes(self._g, mapping, copy=False)
        return new_comp

    def remove_dose(self, compartment: Compartment, admid: Optional[int] = None):
        """Remove dose of compartment.

        Removes dose(s) of compartment. If admid is specified, only doses of that
        admid will be removed.

        Parameters
        ----------
        compartment : Compartment
            Compartment for which to remove dose
        admid : int
            Remove dose of compartment with specified admid, remove all if None.
            Default is None.

        Returns
        -------
        Compartment
            The new updated compartment
        """
        if compartment is None:
            raise ValueError('Option `compartment` cannot be None')

        mapping = dict()
        if admid:
            doses = tuple(dose for dose in compartment.doses if dose.admid != admid)
        else:
            doses = tuple()
        new_comp = compartment.replace(doses=doses)
        mapping[compartment] = new_comp

        nx.relabel_nodes(self._g, mapping, copy=False)
        return new_comp

    def set_lag_time(self, compartment: Compartment, lag_time: TExpr) -> Compartment:
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

    def set_bioavailability(self, compartment: Compartment, bioavailability: TExpr) -> Compartment:
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

    def set_input(self, compartment: Compartment, input: TExpr) -> Compartment:
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

    def find_compartment(self, name: str) -> Optional[Compartment]:
        for comp in self._g.nodes:
            if not isinstance(comp, Output) and comp.name == name:
                return comp
        return None


def _is_positive(expr: sympy.Expr) -> bool:
    return (
        sympy.ask(
            sympy.Q.positive(expr), assume_all(sympy.Q.positive, free_images_and_symbols(expr))
        )
        is True
    )


def _comps(graph):
    return {comp for comp in graph.nodes if not isinstance(comp, Output)}


def to_compartmental_system(names, eqs: Sequence[sympy.Eq]) -> CompartmentalSystem:
    """Convert an list of odes to a compartmental system

    names : func to compartment name map
    """

    eqs = [sympy.sympify(eq) for eq in eqs]
    cb = CompartmentalSystemBuilder()
    compartments = {}
    concentrations = set()
    for eq in eqs:
        A = eq.lhs.args[0]
        concentrations.add(A)
        name = names[Expr(A)]
        comp = Compartment.create(name)
        cb.add_compartment(comp)
        compartments[name] = comp

    neweqs = list(eqs)  # Remaining flows

    for eq in eqs:
        rhs = eq.rhs
        assert isinstance(rhs, sympy.Expr)
        for comp_func in concentrations.intersection(free_images(rhs)):
            dep = rhs.as_independent(comp_func, as_Add=True)[1]
            terms = sympy.Add.make_args(dep.expand())
            for term in terms:
                assert isinstance(term, sympy.Expr)
                from_comp = None
                to_comp = None
                if len(concentrations.intersection(free_images(term))) >= 2:
                    # This means second order absorption -> find matching term
                    # to determine flow
                    if _is_positive(term):
                        for second_comp in concentrations.intersection(free_images(term)):
                            for eq_2 in eqs:
                                if eq_2.lhs.args[0].name == second_comp.name:
                                    # If this is False, then input to compartment is of second order
                                    if -term in sympy.Add.make_args(eq_2.rhs.expand()):
                                        from_comp = compartments[names[Expr(second_comp)]]
                                        to_comp = compartments[names[Expr(eq.lhs.args[0])]]
                else:
                    # Find matching term to determine if flow is between
                    # compartments or not
                    if _is_positive(term):
                        for eq_2 in eqs:
                            if -term in sympy.Add.make_args(eq_2.rhs.expand()):
                                from_comp = compartments[names[Expr(eq_2.lhs.args[0])]]
                                to_comp = compartments[names[Expr(eq.lhs.args[0])]]

                if from_comp is not None and to_comp is not None:
                    # FIXME: Get current flow from builder instead?
                    cs = CompartmentalSystem(cb)
                    current_flow = cs.get_flow(from_comp, to_comp)
                    if current_flow == 0:
                        cb.add_flow(from_comp, to_comp, term / comp_func)
                    else:
                        new_flow = term / comp_func + current_flow
                        cb.add_flow(from_comp, to_comp, new_flow)
                    for i, neweq in enumerate(neweqs):
                        xrhs = neweq.rhs
                        assert isinstance(xrhs, sympy.Expr)
                        if neweq.lhs.args[0].name == eq.lhs.args[0].name:
                            neweqs[i] = sympy.Eq(
                                neweq.lhs, sympy.expand(xrhs - term)  # pyright: ignore
                            )
                        elif neweq.lhs.args[0].name == comp_func.name:
                            neweqs[i] = sympy.Eq(
                                neweq.lhs, sympy.expand(xrhs + term)  # pyright: ignore
                            )
    for eq in neweqs:
        if eq.rhs != 0:
            i = sympy.Integer(0)
            o = sympy.Integer(0)
            for term in sympy.Add.make_args(eq.rhs):
                assert isinstance(term, sympy.Expr)
                if _is_positive(term):
                    i = i + term  # pyright: ignore
                else:
                    o = o + term  # pyright: ignore
            comp_func = eq.lhs.args[0]
            from_comp = compartments[names[Expr(comp_func)]]
            if o != 0:
                cb.add_flow(from_comp, output, -o / comp_func)
            if i != 0:
                cb.set_input(from_comp, i)
    cs = CompartmentalSystem(cb)
    return cs


class CompartmentalSystem(Statement):
    """System of ODEs descibed as a compartmental system

    Examples
    --------
    >>> from pharmpy.model import Bolus, Compartment, output
    >>> from pharmpy.model import CompartmentalSystemBuilder, CompartmentalSystem
    >>> cb = CompartmentalSystemBuilder()
    >>> dose = Bolus.create("AMT")
    >>> central = Compartment.create("CENTRAL", doses=(dose,))
    >>> cb.add_compartment(central)
    >>> peripheral = Compartment.create("PERIPHERAL")
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

    def __init__(
        self,
        builder: CompartmentalSystemBuilder,
        t: Expr = Expr.symbol('t'),
    ):
        self._g = nx.freeze(builder._g.copy())
        self._t = t

    @classmethod
    def create(
        cls,
        builder: CompartmentalSystemBuilder,
        t: Optional[Union[Expr, str]] = Expr.symbol('t'),
    ) -> CompartmentalSystem:
        if builder is None:
            raise TypeError('Argument `builder` cannot be None`')
        elif not isinstance(builder, CompartmentalSystemBuilder):
            raise TypeError(
                f'Argument `builder` must be of type CompartmentalSystemBuilder: got `{type(builder)}`'
            )
        if not isinstance(t, Expr) or isinstance(t, str):
            raise TypeError(f'Argument `t` must be of type str or Expr: got `{type(t)}`')
        t = Expr(t)
        return cls(builder=builder, t=t)

    def replace(self, **kwargs) -> CompartmentalSystem:
        t = kwargs.get('t', self._t)
        builder = kwargs.get('builder', None)
        if builder is None:
            builder = CompartmentalSystemBuilder(self)
        return CompartmentalSystem.create(builder=builder, t=t)

    @property
    def t(self) -> Expr:
        """Independent variable of CompartmentalSystem"""
        return self._t

    @property
    def eqs(self) -> tuple[BooleanExpr, ...]:
        """Tuple of equations"""
        amount_funcs = Matrix(list(self.amounts))
        derivatives = Matrix([Expr.derivative(fn, self.t) for fn in amount_funcs])
        inputs = self.zero_order_inputs
        a = self.compartmental_matrix @ amount_funcs + inputs
        eqs = [
            BooleanExpr(sympy.Eq(lhs, canonical_ode_rhs(rhs._sympy_())))
            for lhs, rhs in zip(derivatives, a)
        ]
        return tuple(eqs)

    @property
    def free_symbols(self) -> set[Expr]:
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
        free = {Expr.symbol('t')}
        for _, _, rate in self._g.edges.data('rate'):
            free |= rate.free_symbols
        for node in _comps(self._g):
            free |= node.free_symbols
        return free

    @property
    def rhs_symbols(self) -> set[Expr]:
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

    def subs(self, substitutions: Mapping[Expr, Expr]) -> CompartmentalSystem:
        """Substitute expressions or symbols in ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.subs({'AMT': 'DOSE'})
        Bolus(DOSE, admid=1) → CENTRAL
        ┌───────┐
        │CENTRAL│──CL/V→
        └───────┘
        """
        cb = CompartmentalSystemBuilder(self)
        for u, v, rate in cb._g.edges.data('rate'):
            rate_sub = rate.subs(substitutions)
            cb._g.edges[u, v]['rate'] = rate_sub
        mapping = {comp: comp.subs(substitutions) for comp in _comps(self._g)}
        nx.relabel_nodes(cb._g, mapping, copy=False)
        return CompartmentalSystem(cb)

    def __eq__(self, other):
        if other is self:
            return True
        if not isinstance(other, CompartmentalSystem):
            return NotImplemented
        return self._t == other._t and nx.to_dict_of_dicts(self._g) == nx.to_dict_of_dicts(other._g)

    def __hash__(self):
        return hash((self._t, self._g))

    def to_dict(self) -> dict[str, Any]:
        comps = [comp for comp in self._g.nodes]
        comps_dicts = tuple(comp.to_dict() for comp in comps)

        edges = []
        for from_comp, to_comp, rate in self._g.edges.data('rate'):
            from_n = comps.index(from_comp)
            to_n = comps.index(to_comp)
            edge = (from_n, to_n, rate.serialize())
            edges.append(edge)

        d = {
            'class': 'CompartmentalSystem',
            'compartments': comps_dicts,
            'rates': edges,
            't': self._t.serialize(),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompartmentalSystem:
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
            cb.add_flow(from_comp, to_comp, Expr.deserialize(rate))

        return cls(cb, t=Expr.deserialize(d['t']))

    def get_flow(
        self, source: Optional[CompartmentBase], destination: Optional[CompartmentBase]
    ) -> Expr:
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
        >>> from pharmpy.model import Compartment, CompartmentalSystemBuilder
        >>> from pharmpy.model import CompartmentalSystem
        >>> cb = CompartmentalSystemBuilder()
        >>> depot = Compartment.create("DEPOT")
        >>> cb.add_compartment(depot)
        >>> central = Compartment.create("CENTRAL")
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
            rate = Expr.integer(0)
        return rate

    def get_compartment_outflows(
        self, compartment: Union[str, CompartmentBase]
    ) -> list[tuple[CompartmentBase, Expr]]:
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
        compartment = self.find_compartment_or_raise(compartment)
        flows = []
        for node in self._g.successors(compartment):
            flow = self.get_flow(compartment, node)
            flows.append((node, flow))
        return flows

    def get_compartment_inflows(
        self, compartment: Union[CompartmentBase, str]
    ) -> list[tuple[Compartment, Expr]]:
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
        >>> from pharmpy.model import output
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.get_compartment_inflows(output)
        [(Compartment(CENTRAL, amount=A_CENTRAL(t), doses=Bolus(AMT, admid=1)), CL/V)]
        """
        if isinstance(compartment, str):
            destination = self.find_compartment(compartment)
            if destination is None:
                raise ValueError(f"Cannot find compartment {compartment}")
        else:
            destination = compartment
        flows = []
        for node in self._g.predecessors(destination):
            flow = self.get_flow(node, destination)
            flows.append((node, flow))
        return flows

    def get_bidirectionals(self, compartment: Union[CompartmentBase, str]) -> list[Compartment]:
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
        compartment = self.find_compartment_or_raise(compartment)
        comps = []
        for node in self._g.predecessors(compartment):
            if self._g.has_edge(compartment, node):
                comps.append(node)
        return comps

    def find_compartment(self, name: str) -> Optional[Compartment]:
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
        Compartment(CENTRAL, amount=A_CENTRAL(t), doses=Bolus(AMT, admid=1))
        """
        for comp in _comps(self._g):
            if comp.name == name:
                return comp
        else:
            return None

    def find_compartment_or_raise(self, comp: Union[str, CompartmentBase]) -> Compartment:
        if isinstance(comp, CompartmentBase):
            return comp  # pyright: ignore
        found_comp = self.find_compartment(comp)
        if found_comp is None:
            raise ValueError(f"No compartment named {comp}")
        return found_comp

    def get_n_connected(self, comp: Compartment) -> int:
        """Get the number of compartments connected to a compartment

        Parameters
        ----------
        comp : Compartment
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
    def dosing_compartments(self) -> tuple[Compartment, ...]:
        """The dosing compartment(s)

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
        >>> model.statements.ode_system.dosing_compartments
        (Compartment(CENTRAL, amount=A_CENTRAL(t), doses=Bolus(AMT, admid=1)),)
        """
        dosing_comps = tuple()
        comps = sorted(list(_comps(self._g)), key=lambda comp: comp.name)
        for node in comps:
            if node.doses:
                if node.name != self.central_compartment.name:
                    if len(dosing_comps) >= 2:
                        dosing_comps = dosing_comps[:-1] + (node,) + dosing_comps[-1:]
                    else:
                        dosing_comps = (node,) + dosing_comps
                else:
                    dosing_comps = dosing_comps + (node,)

        if len(dosing_comps) != 0:
            return dosing_comps

        raise ValueError('No dosing compartment exists')

    @property
    def central_compartment(self) -> Compartment:
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
        Compartment(CENTRAL, amount=A_CENTRAL(t), doses=Bolus(AMT, admid=1))
        """
        try:
            # E.g. TMDD models have more than one output
            # FIXME: Relying heavily on compartment naming for drug metabolite
            central = list(self._g.predecessors(output))[-1]
            if central.name in ["METABOLITE", "EFFECT", "COMPLEX", "RESPONSE"]:
                central = self.find_compartment("CENTRAL")
                if central is None:
                    raise ValueError('Cannot find central compartment')
        except IndexError:
            raise ValueError('Cannot find central compartment')
        return central

    def find_peripheral_compartments(self, name: Optional[str] = None) -> list[Compartment]:
        """Find perihperal compartments

        A peripheral compartment is defined as having one flow to the central compartment and
        one flow from the central compartment.

        If name is set, peripheral compartments connected to the compartment
        with the associated name is returned.

        Returns
        -------
        list of compartments
            Peripheral compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.find_peripheral_compartments()
        []
        """
        if name is not None:
            central = self.find_compartment(name)
            if central is None:
                raise ValueError(f"{name} is not a name of an existing compartment")
        else:
            central = self.central_compartment
        oneout = {node for node, out_degree in self._g.out_degree() if out_degree == 1}
        onein = {node for node, in_degree in self._g.in_degree() if in_degree == 1}
        cout = {comp for comp in oneout if self.get_flow(comp, central) != 0}
        cin = {comp for comp in onein if self.get_flow(central, comp) != 0}
        peripherals = list(cout & cin)
        # Return in deterministic order
        peripherals = sorted(peripherals, key=lambda comp: comp.name)
        return peripherals

    def find_transit_compartments(self, statements: Statements) -> list[Compartment]:
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
        comp = self.dosing_compartments[0]
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

    def find_depot(self, statements: Statements) -> Optional[Compartment]:
        """Find the depot compartment

        The depot compartment is defined to be the compartment that only has out flow to the
        central compartment, but no flow from the central compartment.

        For drug metabolite models however, it is possible to have outflow with
        unidirectional flow to both the central and metabolite compartment. In this case,
        the central compartment is found based on name.

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
        Compartment(DEPOT, amount=A_DEPOT(t), doses=Bolus(AMT, admid=1))
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
            assert len(outflows) == 1 or len(outflows) == 2
            if len(outflows) == 2:
                # FIXME: Use another method than compartment name
                metabolite = self.find_compartment("METABOLITE")
                if metabolite is None or not self.get_flow(to_central, metabolite):
                    break
            inflows = self.get_compartment_inflows(to_central)
            for in_comp, _ in inflows:
                if in_comp == central:
                    break
            else:
                depot = to_central
                break
        return depot

    @property
    def compartmental_matrix(self) -> Matrix:
        """Compartmental matrix of the compartmental system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model, set_first_order_absorption
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.compartmental_matrix
        ⎡-CL ⎤
        ⎢────⎥
        ⎣ V  ⎦
        """
        nodes = self._order_compartments()
        size = len(nodes)
        f = symengine.zeros(size)
        for i in range(0, size):
            from_comp = nodes[i]
            diagsum = 0
            for j in range(0, size):
                to_comp = nodes[j]
                rate = self.get_flow(from_comp, to_comp)
                if i != j:
                    f[j, i] = rate
                diagsum -= rate
            outrate = self.get_flow(from_comp, output)
            f[i, i] = diagsum - outrate
        return Matrix(f)

    @property
    def amounts(self) -> Matrix:
        """Column vector of amounts for all compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> import sympy
        >>> model = load_example_model("pheno")
        >>> sympy.pprint(model.statements.ode_system.amounts)
        [A_CENTRAL(t)]
        """
        ordered_cmts = self._order_compartments()
        amts = [cmt.amount for cmt in ordered_cmts]
        return Matrix(amts)

    @property
    def compartment_names(self) -> list[str]:
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
            dosecmt = self.dosing_compartments[0]
        except ValueError:
            # Fallback for cases where no dose is available (yet)
            comps = list(_comps(self._g))
            return sorted(comps, key=lambda comp: comp.name)
        # Order compartments

        def sortfunc(x):
            a = list(x)
            if output in a:
                a.remove(output)
            a = sorted(a, key=lambda x: x.name)
            return iter(a)

        nodes = list(nx.bfs_tree(self._g, dosecmt, sort_neighbors=sortfunc))
        remaining_unsorted = set(self._g.nodes) - {output} - set(nodes)
        remaining_with_input = {comp for comp in remaining_unsorted if comp.input != 0}
        remaining_without_input = remaining_unsorted - remaining_with_input
        remaining = sorted(remaining_with_input, key=lambda x: x.name) + sorted(
            remaining_without_input, key=lambda x: x.name
        )

        while remaining:  # Disjoint or upstream of dosing
            comp = remaining.pop(0)
            connected = list(nx.bfs_tree(self._g, comp, sort_neighbors=sortfunc))
            for c in connected:
                if c not in nodes:
                    nodes.append(c)
                    if c != comp:
                        remaining.remove(c)
        return nodes

    @property
    def zero_order_inputs(self) -> Matrix:
        """Vector of all zero order inputs to each compartment

        Example
        -------
        >>> from pharmpy.modeling import load_example_model, set_zero_order_absorption
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.zero_order_inputs
        [0]

        """
        inputs = [node.input for node in self._order_compartments()]
        return Matrix(inputs)

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

        current = self.dosing_compartments[0]
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
        have_zo_input = int(not self.zero_order_inputs.is_zero_matrix())
        comp_nrows = comp_height * 2 - 1 + (1 if noutput > 1 else 0)
        nrows = comp_nrows + have_zo_input
        ncols = comp_width * 2
        grid = unicode.Grid(nrows, ncols)

        current = self.dosing_compartments[0]
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

        all_doses = ""
        for dose_comp in self.dosing_compartments:
            for dose in dose_comp.doses:
                all_doses += f'{str(dose)} → {dose_comp.name} \n'
        s = all_doses + str(grid).rstrip()
        return s


class Dose(ABC):
    """Abstract base class for different types of doses"""

    def __init__(self, amount: Expr, admid: int):
        self._admid = admid
        self._amount = amount

    @property
    def admid(self) -> int:
        """Administration ID of dose"""
        return self._admid

    @property
    def amount(self) -> Expr:
        """Symbolic amount of dose"""
        return self._amount

    @abstractmethod
    def subs(self, substitutions) -> Dose: ...

    @property
    @abstractmethod
    def free_symbols(self) -> set[Expr]: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def replace(self, **kwargs) -> Self: ...


class Bolus(Dose, Immutable):
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

    def __init__(self, amount: Expr, admid: int = 1):
        super().__init__(amount, admid)

    @classmethod
    def create(cls, amount: TExpr, admid: int = 1) -> Bolus:
        return cls(Expr(amount), admid=admid)

    def replace(self, **kwargs) -> Bolus:
        amount = kwargs.get("amount", self._amount)
        admid = kwargs.get("admid", self._admid)
        return Bolus.create(amount=amount, admid=admid)

    @property
    def amount(self) -> Expr:
        """Symbolic amount of dose"""
        return self._amount

    @property
    def free_symbols(self) -> set[Expr]:
        """Get set of all free symbols in the dose

        Examples
        --------
        >>> from pharmpy.model import Bolus
        >>> dose = Bolus.create("AMT")
        >>> dose.free_symbols
        {AMT}
        """
        return {self._amount}

    def subs(self, substitutions: Mapping[Expr, Expr]) -> Bolus:
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
        return Bolus(self._amount.subs(substitutions), admid=self._admid)

    def __eq__(self, other):
        if not isinstance(other, Bolus):
            return NotImplemented
        return self._amount == other._amount and self._admid == other._admid

    def __hash__(self):
        return hash((self._amount, self._admid))

    def to_dict(self) -> dict[str, Any]:
        return {'class': 'Bolus', 'amount': self._amount.serialize(), 'admid': self._admid}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Bolus:
        return cls(amount=Expr.deserialize(d['amount']), admid=d['admid'])

    def __repr__(self):
        return f'Bolus({self._amount}, admid={self._admid})'


class Infusion(Dose, Immutable):
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
    >>> dose = Infusion.create("AMT", duration="D1")
    >>> dose
    Infusion(AMT, admid=1, duration=D1)
    >>> dose = Infusion.create("AMT", rate="R1")
    >>> dose
    Infusion(AMT, admid=1, rate=R1)
    """

    def __init__(
        self,
        amount: Expr,
        admid: int = 1,
        rate: Optional[Expr] = None,
        duration: Optional[Expr] = None,
    ):
        super().__init__(amount, admid)
        self._rate = rate
        self._duration = duration

    @classmethod
    def create(
        cls,
        amount: TExpr,
        admid: int = 1,
        rate: Optional[TExpr] = None,
        duration: Optional[TExpr] = None,
    ) -> Infusion:
        if rate is None and duration is None:
            raise ValueError('Need rate or duration for Infusion')
        if rate is not None and duration is not None:
            raise ValueError('Cannot have both rate and duration for Infusion')
        if rate is not None:
            rate = Expr(rate)
        if duration is not None:
            duration = Expr(duration)
        return cls(Expr(amount), admid=admid, rate=rate, duration=duration)

    def replace(self, **kwargs) -> Infusion:
        amount = kwargs.get("amount", self._amount)
        admid = kwargs.get("admid", self._admid)
        rate = kwargs.get("rate", self._rate)
        duration = kwargs.get("duration", self._duration)
        return Infusion.create(amount=amount, admid=admid, rate=rate, duration=duration)

    @property
    def rate(self) -> Optional[Expr]:
        """Symbolic rate

        Mutually exclusive with duration.
        """
        return self._rate

    @property
    def duration(self) -> Optional[Expr]:
        """Symbolc duration

        Mutually exclusive with rate.
        """
        return self._duration

    @property
    def free_symbols(self) -> set[Expr]:
        """Get set of all free symbols in the dose

        Examples
        --------
        >>> from pharmpy.model import Infusion
        >>> dose = Infusion.create("AMT", rate="RATE")
        >>> dose.free_symbols   # doctest: +SKIP
        {AMT, RATE}
        """
        if self._rate is not None:
            symbs = self._rate.free_symbols
        else:
            assert self._duration is not None
            symbs = self._duration.free_symbols
        return symbs | self._amount.free_symbols

    def subs(self, substitutions: Mapping[Expr, Expr]) -> Infusion:
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
        amount = self._amount.subs(substitutions)
        if self._rate is not None:
            rate = self._rate.subs(substitutions)
            duration = None
        else:
            rate = None
            assert self._duration is not None
            duration = self._duration.subs(substitutions)
        return Infusion(amount, admid=self._admid, rate=rate, duration=duration)

    def __eq__(self, other):
        if not isinstance(other, Infusion):
            return NotImplemented
        return (
            self._admid == other._admid
            and self._rate == other._rate
            and self._duration == other._duration
            and self._amount == other._amount
        )

    def __hash__(self):
        return hash((self._admid, self._rate, self._duration, self._amount))

    def to_dict(self) -> dict[str, Any]:
        return {
            'class': 'Infusion',
            'amount': self._amount.serialize(),
            'rate': self._rate.serialize() if self._rate is not None else None,
            'duration': self._duration.serialize() if self._duration is not None else None,
            'admid': self._admid,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Infusion:
        return cls(
            amount=Expr.deserialize(d['amount']),
            admid=d['admid'],
            rate=None if d['rate'] is None else Expr.deserialize(d['rate']),
            duration=None if d['duration'] is None else Expr.deserialize(d['duration']),
        )

    def __repr__(self):
        if self.rate is not None:
            arg = f'rate={self._rate}'
        else:
            arg = f'duration={self._duration}'
        return f'Infusion({self._amount}, admid={self._admid}, {arg})'


class Compartment(CompartmentBase):
    """Compartment for a compartmental system

    Parameters
    ----------
    name : str
        Compartment name
    doses : tuple(Dose)
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
    Compartment(CENTRAL, amount=A_CENTRAL(t))
    >>> comp = Compartment.create("DEPOT", lag_time="ALAG")
    >>> comp
    Compartment(DEPOT, amount=A_DEPOT(t), lag_time=ALAG)
    >>> dose = Bolus.create("AMT")
    >>> comp = Compartment.create("DEPOT", doses=(dose,))
    >>> comp
    Compartment(DEPOT, amount=A_DEPOT(t), doses=Bolus(AMT, admid=1))
    """

    def __init__(
        self,
        name: str,
        amount: Expr,
        doses: tuple[Dose, ...] = tuple(),
        input: Expr = Expr.integer(0),
        lag_time: Expr = Expr.integer(0),
        bioavailability: Expr = Expr.integer(1),
    ):
        self._name = name
        self._amount = amount
        self._doses = doses
        self._input = input
        self._lag_time = lag_time
        self._bioavailability = bioavailability

    @classmethod
    def create(
        cls,
        name: str,
        amount: Optional[TExpr] = None,
        doses: tuple[Dose, ...] = tuple(),
        input: TExpr = Expr.integer(0),
        lag_time: TExpr = Expr.integer(0),
        bioavailability: TExpr = Expr.integer(1),
    ) -> Compartment:
        if not isinstance(name, str):
            raise TypeError("Name of a Compartment must be of string type")
        if amount is not None:
            amount_expr = Expr(amount)
        else:
            # NOTE: Uses a default idv
            amount_expr = Expr.function(f'A_{name}', 't')
        if not isinstance(doses, tuple):
            try:
                doses = tuple(doses)
            except TypeError:
                raise TypeError("dose(s) need to be given as a sequence")
        for d in doses:
            if not isinstance(d, Dose):
                raise TypeError("All doses need to be of type Dose")
        input = Expr(input)
        lag_time = Expr(lag_time)
        bioavailability = Expr(bioavailability)
        return cls(
            name=name,
            amount=amount_expr,
            doses=doses,
            input=input,
            lag_time=lag_time,
            bioavailability=bioavailability,
        )

    def replace(self, **kwargs):
        name = kwargs.get("name", self._name)
        amount = kwargs.get("amount", self._amount)
        doses = kwargs.get("doses", self._doses)
        input = kwargs.get("input", self._input)
        lag_time = kwargs.get("lag_time", self._lag_time)
        bioavailability = kwargs.get("bioavailability", self._bioavailability)
        return Compartment.create(
            name=name,
            amount=amount,
            doses=doses,
            input=input,
            lag_time=lag_time,
            bioavailability=bioavailability,
        )

    @property
    def name(self) -> str:
        """Compartment name"""
        return self._name

    @property
    def amount(self) -> Expr:
        """Compartment amount symbol"""
        return self._amount

    @property
    def doses(self) -> tuple[Dose, ...]:
        # FIXME: Return in defined order with oral doses coming first!
        # Only do this for multiple doses.
        if len(self._doses) > 1:
            return tuple(sorted(self._doses, key=lambda d: isinstance(d, Infusion), reverse=True))
        else:
            return self._doses

    @property
    def input(self) -> Expr:
        return self._input

    @property
    def lag_time(self) -> Expr:
        """Lag time for doses into compartment"""
        return self._lag_time

    @property
    def bioavailability(self) -> Expr:
        """Bioavailability fraction for doses into compartment"""
        return self._bioavailability

    @property
    def free_symbols(self) -> set[Expr]:
        """Get set of all free symbols in the compartment

        Examples
        --------
        >>> from pharmpy.model import Bolus, Compartment
        >>> dose = Bolus.create("AMT")
        >>> comp = Compartment.create("CENTRAL", doses=(dose,), lag_time="ALAG")
        >>> comp.free_symbols  # doctest: +SKIP
        {A_CENTRAL, ALAG, AMT}
        """
        symbs = set()
        for d in self.doses:
            symbs |= d.free_symbols
        symbs |= self.input.free_symbols
        symbs |= self.lag_time.free_symbols
        symbs |= self.bioavailability.free_symbols
        return symbs

    def subs(self, substitutions: Mapping[Expr, Expr]) -> Compartment:
        """Substitute expressions or symbols in compartment

        Examples
        --------
        >>> from pharmpy.model import Bolus, Compartment
        >>> dose = Bolus.create("AMT")
        >>> comp = Compartment.create("CENTRAL", doses=(dose,))
        >>> comp.subs({"AMT": "DOSE"})
        Compartment(CENTRAL, amount=A_CENTRAL(t), doses=Bolus(DOSE, admid=1))
        """
        if self.doses:
            new_doses = tuple()
            for d in self.doses:
                new_doses = new_doses + (d.subs(substitutions),)
        else:
            new_doses = tuple()
        return Compartment(
            self.name,
            amount=self._amount.subs(substitutions),
            doses=new_doses,
            input=self._input.subs(substitutions),
            lag_time=self._lag_time.subs(substitutions),
            bioavailability=self._bioavailability.subs(substitutions),
        )

    def __eq__(self, other):
        if not isinstance(other, Compartment):
            return NotImplemented
        return (
            self._name == other._name
            and self._amount == other._amount
            and self._doses == other._doses
            and self._input == other._input
            and self._lag_time == other._lag_time
            and self._bioavailability == other._bioavailability
        )

    def __hash__(self):
        return hash(
            (
                self._name,
                self._amount,
                self._doses,
                self._input,
                self._lag_time,
                self._bioavailability,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        if not self._doses:
            all_doses = None
        else:
            all_doses = tuple(dose.to_dict() for dose in self._doses)
        return {
            'class': 'Compartment',
            'name': self._name,
            'amount': self._amount.serialize(),
            'doses': all_doses,
            'input': self._input.serialize(),
            'lag_time': self._lag_time.serialize(),
            'bioavailability': self._bioavailability.serialize(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Compartment:
        if d['doses'] is None:
            all_doses = tuple()
        else:
            all_doses = tuple()
            for dose in d['doses']:
                if dose['class'] == 'Bolus':
                    all_doses += (Bolus.from_dict(dose),)
                else:
                    all_doses += (Infusion.from_dict(dose),)
        return cls(
            name=d['name'],
            amount=Expr.deserialize(d['amount']),
            doses=all_doses,
            input=Expr.deserialize(d['input']),
            lag_time=Expr.deserialize(d['lag_time']),
            bioavailability=Expr.deserialize(d['bioavailability']),
        )

    def __repr__(self):
        lag = '' if self._lag_time == 0 else f', lag_time={self._lag_time}'
        doses = (
            ''
            if self._doses is tuple()
            else f', doses={self._doses[0] if len(self._doses) == 1 else self._doses}'
        )
        input = '' if self._input == 0 else f', input={self._input}'
        bioavailability = (
            '' if self._bioavailability == 1 else f', bioavailability={self._bioavailability}'
        )
        return (
            f'Compartment({self._name}, amount={self._amount}{doses}{input}{lag}{bioavailability})'
        )


class Statements(Sequence, Immutable):
    """A sequence of symbolic statements describing the model

    Two types of statements are supported: Assignment and CompartmentalSystem.
    A Statements object can have 0 or 1 CompartmentalSystem. The order of
    the statements is significant and the same symbol can be assigned
    to multiple times.

    Parameters
    ----------
    statements : list or Statements
        A list of Statement or another Statements to populate this object
    """

    def __init__(self, statements: Union[Statements, Iterable[Statement]] = ()):
        if not isinstance(statements, tuple):
            statements = tuple(statements)
        self._statements = statements

    @classmethod
    def create(cls, statements: Optional[Union[Statements, Iterable[Statement]]] = None):
        if isinstance(statements, Statements):
            statements = statements
        elif statements is None:
            statements = ()
        elif isinstance(statements, Iterable):
            if any(not isinstance(s, Statement) for s in statements):
                raise TypeError('`statements` must consist of only type Statement')
            statements = tuple(statements)
        else:
            raise TypeError(
                f'`statements` must be of type Statements or an iterable of Statement: got `{type(statements)}`'
            )
        return cls(statements)

    @overload
    def __getitem__(self, ind: slice) -> Statements: ...

    @overload
    def __getitem__(self, ind: int) -> Statement: ...

    def __getitem__(self, ind):  # pyright: ignore
        if isinstance(ind, slice):
            return Statements(self._statements[ind])
        else:
            return self._statements[ind]

    def __len__(self):
        return len(self._statements)

    def __add__(self, other: Union[Statements, Statement, Iterable[Statement]]) -> Statements:
        if isinstance(other, Statements):
            return Statements(self._statements + other._statements)
        elif isinstance(other, Statement):
            return Statements(self._statements + (other,))
        elif isinstance(other, Iterable):
            return Statements.create(self._statements + tuple(other))
        else:
            return NotImplemented

    def __radd__(self, other: Union[Statement, Iterable[Statement]]) -> Statements:
        if isinstance(other, Iterable):
            return Statements.create(tuple(other) + self._statements)
        else:
            return NotImplemented

    @property
    def free_symbols(self) -> set[Expr]:
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

    @property
    def lhs_symbols(self) -> set[Expr]:
        """Get set of all symbols defined in this Statements

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.lhs_symbols   # doctest: +SKIP
        {F, A_CENTRAL(t), TVV, CL, Y, VC, V, TVCL, S1}
        """
        symbs = set()
        for s in self:
            if isinstance(s, Assignment):
                symbs.add(s.symbol)
            else:
                assert isinstance(s, CompartmentalSystem)
                symbs |= set(s.amounts)
        return symbs

    def _get_ode_system_index(self):
        return next(
            map(
                lambda t: t[0],
                filter(lambda t: isinstance(t[1], CompartmentalSystem), enumerate(self)),
            ),
            -1,
        )

    @property
    def ode_system(self) -> Optional[CompartmentalSystem]:
        """Returns the ODE system of the model or None if the model doesn't have an ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system
        Bolus(AMT, admid=1) → CENTRAL
        ┌───────┐
        │CENTRAL│──CL/V→
        └───────┘
        """
        i = self._get_ode_system_index()
        if i == -1:
            return None
        else:
            cs = self[i]
            assert isinstance(cs, CompartmentalSystem)
            return cs

    @property
    def before_odes(self) -> Statements:
        """All statements before the ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.before_odes
                TVCL = POP_CL⋅WGT
                TVV = POP_VC⋅WGT
                          ⎧TVV⋅(COVAPGR + 1)  for APGR < 5
                          ⎨
                TVV = ⎩       TVV          otherwise
                                   ETA_CL
                CL = TVCL⋅ℯ
                                  ETA_VC
                VC = TVV⋅ℯ
                V = VC
                S₁ = VC

        """
        i = self._get_ode_system_index()
        return self if i == -1 else self[:i]

    @property
    def after_odes(self) -> Statements:
        """All statements after the ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.after_odes
                        A_CENTRAL(t)
                        ────────────
                F =      S₁
                Y = EPS₁⋅F + F

        """
        i = self._get_ode_system_index()
        return Statements() if i == -1 else self[i + 1 :]

    @property
    def error(self) -> Statements:
        """All statements after the ODE system or the whole model if no ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.error
                        A_CENTRAL(t)
                        ────────────
                F =      S₁
                Y = EPS₁⋅F + F
        """
        i = self._get_ode_system_index()
        return self if i == -1 else self[i + 1 :]

    def subs(self, substitutions: Mapping[Expr, Expr]) -> Statements:
        """Substitute symbols in all statements.

        Parameters
        ----------
        substitutions : dict
            Old-new pairs(can be type str or symbol)

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> stats = model.statements.subs({'WGT': 'WT'})
        >>> stats.before_odes
                TVCL = POP_CL⋅WT
                TVV = POP_VC⋅WT
                          ⎧TVV⋅(COVAPGR + 1)  for APGR < 5
              ⎨
                TVV = ⎩       TVV          otherwise
                           ETA_CL
        CL = TVCL⋅ℯ
                                  ETA_VC
                VC = TVV⋅ℯ
                V = VC
                S₁ = VC
        """
        return Statements(s.subs(substitutions) for s in self)

    def _lookup_last_assignment(
        self, symbol: TSymbol
    ) -> tuple[Optional[int], Optional[Assignment]]:
        if isinstance(symbol, str):
            symbol = Expr.symbol(symbol)
        ind = None
        assignment = None
        for i, statement in enumerate(self):
            if isinstance(statement, Assignment):
                if statement.symbol == symbol:
                    ind = i
                    assignment = statement
        return ind, assignment

    def find_assignment(self, symbol: TSymbol) -> Optional[Assignment]:
        """Returns last assignment of symbol

        Parameters
        ----------
        symbol : Symbol or str
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
                   ETA_CL
        CL = TVCL⋅ℯ
        """
        return self._lookup_last_assignment(symbol)[1]

    def get_assignment(self, symbol: TSymbol) -> Assignment:
        """Returns last assignment of symbol

        This method assumes that the assignment exists

        Parameters
        ----------
        symbol : Symbol or str
            Symbol to get

        Returns
        -------
        Assignment
            The Assignment

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.get_assignment("CL")
                   ETA_CL
        CL = TVCL⋅ℯ

        See Also
        --------
        find_assignment : Find assignment (allowing for the assigment to not exist)
        """

        assignment = self.find_assignment(symbol)
        if assignment is None:
            raise ValueError(f"Assignment of {symbol} not found")
        return assignment

    def find_assignment_index(self, symbol: TSymbol) -> Optional[int]:
        """Returns index of last assignment of symbol

        Parameters
        ----------
        symbol : Symbol or str
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
        3
        """
        return self._lookup_last_assignment(symbol)[0]

    def reassign(self, symbol: TSymbol, expression: TExpr) -> Statements:
        """Reassign symbol to expression

        Set symbol to be expression and remove all previous assignments of symbol

        Parameters
        ----------
        symbol : sympy.Symbol or str
            Symbol to reassign
        expression : Expr or str
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
        symbol = Expr(symbol)
        expression = Expr(expression)

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
                if isinstance(statement, Assignment):
                    if statement.symbol in rhs:
                        graph.add_edge(i, j)
                else:
                    assert isinstance(statement, CompartmentalSystem)
                    amts = set(statement.amounts)
                    if not rhs.isdisjoint(amts):
                        graph.add_edge(i, j)
        return graph

    def direct_dependencies(self, statement: Statement) -> Statements:
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
                   ETA_CL
        CL = TVCL⋅ℯ
        V = VC
        """
        g = self._create_dependency_graph()
        index = self.index(statement)
        succ = sorted(list(g.successors(index)))
        stats = Statements()
        stats._statements = [self[i] for i in succ]
        return stats

    def dependencies(self, symbol_or_statement: Union[TSymbol, Statement]) -> set[Expr]:
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
            symbol = Expr(symbol_or_statement)
            for i in range(len(self) - 1, -1, -1):
                statement = self[i]
                if (
                    isinstance(statement, Assignment)
                    and statement.symbol == symbol
                    or isinstance(statement, CompartmentalSystem)
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
                assert isinstance(statement, CompartmentalSystem)
                symbs -= set(statement.amounts)
            symbs |= statement.rhs_symbols
        return symbs

    def remove_symbol_definitions(
        self, symbols: Iterable[Expr], statement: Statement
    ) -> Statements:
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
            additional |= set(nx.dfs_preorder_nodes(graph, add))
        remove = candidates - additional
        return Statements(tuple(self[i] for i in range(len(self)) if i not in remove))

    def full_expression(self, expression: TExpr) -> Expr:
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
        POP_CL*WGT*exp(ETA_CL)
        """
        expression = Expr(expression)
        for statement in reversed(self):
            if isinstance(statement, CompartmentalSystem):
                raise ValueError(
                    "CompartmentalSystem not supported by full_expression. Use the properties before_odes "
                    "or after_odes."
                )
            expression = expression.subs({statement.symbol: statement.expression})
        return expression

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Statements):
            return NotImplemented
        if len(self) != len(other):
            return False
        else:
            for first, second in zip(self, other):
                if first != second:
                    return False
        return True

    def __hash__(self):
        return hash(self._statements)

    def to_dict(self) -> dict[str, Any]:
        stats = tuple(s.to_dict() for s in self)
        return {'statements': stats}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Statements:
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
