from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Iterable, List, Mapping, Optional, Set, Tuple, Union, overload

import pharmpy.internals.unicode as unicode
from pharmpy.deps import networkx as nx
from pharmpy.deps import sympy
from pharmpy.internals.expr.assumptions import assume_all
from pharmpy.internals.expr.leaves import free_images, free_images_and_symbols
from pharmpy.internals.expr.ode import canonical_ode_rhs
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.immutable import Immutable, frozenmapping


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
        return cls(symbol, expression)

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


class ODESystemMetaclass(ABCMeta):
    def __getattr__(cls, key):
        # NOTE see https://stackoverflow.com/a/3155493
        if key == 't':
            return sympy.Symbol('t')
        raise AttributeError(key)


class ODESystem(Statement, ABC, metaclass=ODESystemMetaclass):
    """Abstract base class for ODE systems of different forms"""

    def __getattr__(self, key):
        if key == 't':
            return ODESystem.t
        raise AttributeError(key)

    @abstractmethod
    def to_compartmental_system(self) -> CompartmentalSystem:
        pass

    @abstractmethod
    def to_explicit_system(self) -> ExplicitODESystem:
        pass


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
    """System of ODEs described explicitly

    Parameters
    ----------
    odes : list
        Symbolic differential equations
    ics : dict
        Symbolic initial conditions

    Examples
    --------
    >>> from pharmpy.model import ExplicitODESystem
    >>> import sympy
    >>> A_DEPOT = sympy.Function('A_DEPOT')
    >>> A_CENTRAL = sympy.Function('A_CENTRAL')
    >>> t = ExplicitODESystem.t
    >>> AMT, KA, K = sympy.symbols('AMT KA K')
    >>> eq1 = sympy.Eq(sympy.Derivative(A_DEPOT(t), t), -KA * A_DEPOT(t))
    >>> eq2 = sympy.Eq(sympy.Derivative(A_CENTRAL(t)), -K*A_CENTRAL(t) + KA*A_DEPOT(t))
    >>> ics = {A_DEPOT(0): AMT, A_CENTRAL(0): 0}
    >>> odes = ExplicitODESystem([eq1, eq2], ics)
    >>> odes
    ⎧d
    ⎪──(A_DEPOT(t)) = -KA⋅A_DEPOT(t)
    ⎪dt
    ⎨d
    ⎪──(A_CENTRAL(t)) = -K⋅A_CENTRAL(t) + KA⋅A_DEPOT(t)
    ⎪dt
    ⎪A_DEPOT(0) = AMT
    ⎩A_CENTRAL(0) = 0
    """

    def __init__(self, odes: Tuple[sympy.Eq, ...], ics: Mapping[sympy.Expr, sympy.Expr]):
        self._odes = odes
        self._ics = frozenmapping(ics)

    @property
    def odes(self) -> Tuple[sympy.Eq, ...]:
        """List of ordinary differential equations"""
        return self._odes

    @property
    def ics(self) -> Mapping[sympy.Expr, sympy.Expr]:
        """Initial conditions"""
        return self._ics

    @property
    def free_symbols(self):
        """Get set of all free symbols in the ODE system

        Returns
        -------
        set
            Set of symbols

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> odes = model.statements.ode_system.to_explicit_system()
        >>> odes.free_symbols  # doctest: +SKIP
        {AMT, CL, V, t}
        """

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
        """Substitute expressions or symbols in ODE system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> odes = model.statements.ode_system.to_explicit_system()
        >>> odes.subs({'AMT': 'DOSE'})
        ⎧d                  -CL⋅A_CENTRAL(t)
        ⎪──(A_CENTRAL(t)) = ─────────────────
        ⎪dt                         V
        ⎨d                 CL⋅A_CENTRAL(t)
        ⎪──(A_OUTPUT(t)) = ───────────────
        ⎪dt                       V
        ⎪A_CENTRAL(0) = DOSE
        ⎩A_OUTPUT(0) = 0
        <BLANKLINE>
        """
        d = {
            sympy.Function(str(key))(sympy.Symbol('t')): value
            for key, value in substitutions.items()
        }
        d.update(substitutions)
        odes = tuple(subs(ode, d) for ode in self.odes)
        ics = {subs(key, d): subs(value, d) for key, value in self.ics.items()}
        return ExplicitODESystem(odes, ics)

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
        >>> odes = model.statements.ode_system.to_explicit_system()
        >>> odes.rhs_symbols   # doctest: +SKIP
        {AMT, CL, V, t}
        """
        return self.free_symbols

    @property
    def amounts(self):
        """Column vector of all amount functions

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> import sympy
        >>> model = load_example_model("pheno")
        >>> odes = model.statements.ode_system.to_explicit_system()
        >>> sympy.pprint(odes.amounts)
        ⎡A_CENTRAL⎤
        ⎢         ⎥
        ⎣A_OUTPUT ⎦
        """
        amounts = [ode.lhs.args[0].name for ode in self.odes]
        return sympy.Matrix(amounts)

    @property
    def compartment_names(self) -> List[str]:
        """Names of all compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> odes = model.statements.ode_system.to_explicit_system()
        >>> odes.compartment_names
        ['CENTRAL', 'OUTPUT']
        """
        return [ode.lhs.args[0].name[2:] for ode in self.odes]

    def __repr__(self):
        a = []
        for ode in self.odes:
            ode_str = sympy.pretty(ode)
            a += ode_str.split('\n')
        for key, value in self.ics.items():
            ics_str = sympy.pretty(sympy.Eq(key, value))
            a += ics_str.split('\n')
        return _bracket(a)

    def __eq__(self, other):
        return (
            isinstance(other, ExplicitODESystem)
            and self.odes == other.odes
            and self.ics == other.ics
        )

    def _repr_latex_(self):
        rows = []
        for ode in self.odes:
            ode_repr = sympy.latex(ode, mul_symbol='dot')
            rows.append(ode_repr)
        for k, v in self.ics.items():
            ics_eq = sympy.Eq(k, v)
            ics_repr = sympy.latex(ics_eq, mul_symbol='dot')
            rows.append(ics_repr)
        return r'\begin{cases} ' + r' \\ '.join(rows) + r' \end{cases}'

    def to_explicit_system(self, skip_output=False):
        return self

    def to_compartmental_system(self):
        """Get the explicit system as a compartmental ODE system

        Returns
        -------
        CompartmentalSystem
            The same ODE system in compartmental representation

        Example
        -------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> statements = model.statements.to_explicit_system()
        >>> statements.to_compartmental_system()    # doctest: +SKIP
        """

        def convert_name(name: str):
            return name[2:] if name[:2] == 'A_' else name

        cb = CompartmentalSystemBuilder()
        compartments = {}
        concentrations = set()
        for i, eq in enumerate(self.odes):
            A = eq.lhs.args[0]
            concentrations.add(A)
            name = convert_name(A.name)
            # FIXME The following is not true in general!
            dose = Bolus(sympy.Symbol("AMT")) if i == 0 else None
            comp = Compartment(convert_name(A.name), dose)
            cb.add_compartment(comp)
            compartments[name] = comp

        for eq in self.odes:
            for comp_func in concentrations.intersection(free_images(eq.rhs)):
                dep = eq.rhs.as_independent(comp_func, as_Add=True)[1]
                terms = sympy.Add.make_args(dep)
                for term in terms:
                    if _is_positive(term):
                        from_comp = compartments[convert_name(comp_func.name)]
                        to_comp = compartments[convert_name(eq.lhs.args[0].name)]
                        cb.add_flow(from_comp, to_comp, term / comp_func)

        return CompartmentalSystem(cb)


class CompartmentalSystemBuilder:
    """Builder for CompartmentalSystem"""

    def __init__(self, cs=None):
        if cs:
            self._g = cs._g.copy()
        else:
            self._g = nx.DiGraph()

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
        >>> from pharmpy.model import CompartmentalSystemBuilder
        >>> cb = CompartmentalSystemBuilder()
        >>> depot = Compartment("DEPOT")
        >>> cb.add_compartment(depot)
        >>> central = Compartment("CENTRAL")
        >>> cb.add_compartment("CENTRAL")
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
        new_source = Compartment(source.name, None, source.lag_time, source.bioavailability)
        new_dest = Compartment(
            destination.name, source.dose, destination.lag_time, destination.bioavailability
        )
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
        new_comp = Compartment(
            compartment.name, dose, compartment.lag_time, compartment.bioavailability
        )
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
        new_comp = Compartment(
            compartment.name, compartment.dose, lag_time, compartment.bioavailability
        )
        mapping = {compartment: new_comp}
        nx.relabel_nodes(self._g, mapping, copy=False)
        return new_comp


def _is_positive(expr: sympy.Expr) -> bool:
    return sympy.ask(
        sympy.Q.positive(expr), assume_all(sympy.Q.positive, free_images_and_symbols(expr))
    )


class CompartmentalSystem(ODESystem):
    """System of ODEs descibed as a compartmental system

    Examples
    --------
    >>> from pharmpy.model import Bolus, CompartmentalSystem
    >>> cb = CompartmentalSystemBuilder()
    >>> dose = Bolus.create("AMT")
    >>> central = Compartment("CENTRAL", dose)
    >>> cb.add_compartment(central)
    >>> peripheral = Compartment("PERIPHERAL")
    >>> cb.add_compartment(peripheral)
    >>> output = Compartment("OUTPUT")
    >>> cb.add_compartment(output)
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
    ┌───────┐      ┌──────────┐      ┌──────────┐       ┌──────┐
    │CENTRAL│──K12→│PERIPHERAL│──K21→│ CENTRAL  │──CL/V→│OUTPUT│
    └───────┘      └──────────┘      └──────────┘       └──────┘
    """

    def __init__(self, builder):
        self._g = nx.freeze(builder._g.copy())

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
        for (_, _, rate) in self._g.edges.data('rate'):
            free |= rate.free_symbols
        for node in self._g.nodes:
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
        Bolus(DOSE)
        ┌───────┐       ┌──────┐
        │CENTRAL│──CL/V→│OUTPUT│
        └───────┘       └──────┘
        """
        cb = CompartmentalSystemBuilder(self)
        for (u, v, rate) in cb._g.edges.data('rate'):
            rate_sub = subs(rate, substitutions, simultaneous=True)
            cb._g.edges[u, v]['rate'] = rate_sub
        mapping = {comp: comp.subs(substitutions) for comp in self._g.nodes}
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
        for (_, _, rate) in self._g.edges.data('rate'):
            atoms |= rate.atoms(cls)
        return atoms

    def __eq__(self, other):
        return (
            isinstance(other, CompartmentalSystem)
            and nx.to_dict_of_dicts(self._g) == nx.to_dict_of_dicts(other._g)
            and self.dosing_compartment.dose == other.dosing_compartment.dose
        )

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
        """
        try:
            rate = self._g.edges[source, destination]['rate']
        except KeyError:
            rate = None
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
        [(Compartment(OUTPUT), CL/V)]
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
        >>> model.statements.ode_system.get_compartment_inflows("OUTPUT")
        [(Compartment(CENTRAL, dose=Bolus(AMT)), CL/V)]
        """
        if isinstance(compartment, str):
            compartment = self.find_compartment(compartment)
        flows = []
        for node in self._g.predecessors(compartment):
            flow = self.get_flow(node, compartment)
            flows.append((node, flow))
        return flows

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
        Compartment(CENTRAL, dose=Bolus(AMT))
        """
        for comp in self._g.nodes:
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
        1
        """
        out_comps = {c for c, _ in self.get_compartment_outflows(comp)}
        in_comps = {c for c, _ in self.get_compartment_inflows(comp)}
        return len(out_comps | in_comps)

    @property
    def output_compartment(self):
        """Get the output compartment

        An output compartment is defined to be a compartment that does not have any outward
        flow. A model has to have one and only one output compartment.

        Returns
        -------
        Compartment
            Output compartment of compartmental system

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.output_compartment
        Compartment(OUTPUT)
        """
        zeroout = [node for node, out_degree in self._g.out_degree() if out_degree == 0]
        if len(zeroout) == 1:
            return zeroout[0]
        else:
            raise ValueError('More than one or zero output compartments')

    @property
    def dosing_compartment(self):
        """The dosing compartment

        A dosing compartment is a compartment that receives an input dose. Only one dose
        compartment is supported.

        Returns
        -------
        Compartment
            Dosing compartment

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.dosing_compartment
        Compartment(CENTRAL, dose=Bolus(AMT))
        """
        for node in self._g.nodes:
            if node.dose is not None:
                return node
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
        Compartment(CENTRAL, dose=Bolus(AMT))
        """
        output = self.output_compartment
        try:
            central = next(self._g.predecessors(output))
        except StopIteration:
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
        cout = {comp for comp in oneout if self.get_flow(comp, central) is not None}
        cin = {comp for comp in onein if self.get_flow(central, comp) is not None}
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
        comp = self.dosing_compartment
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
            self.get_flow(transits[0], central) is not None or transits[0] == central
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
        >>> set_first_order_absorption(model)       # doctest: +ELLIPSIS
        <...>
        >>> model.statements.ode_system.find_depot(model.statements)
        Compartment(DEPOT, dose=Bolus(AMT))
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
        ⎡-CL    ⎤
        ⎢────  0⎥
        ⎢ V     ⎥
        ⎢       ⎥
        ⎢ CL    ⎥
        ⎢ ──   0⎥
        ⎣ V     ⎦
        """
        dod = nx.to_dict_of_dicts(self._g)
        size = len(self._g.nodes)
        f = sympy.zeros(size)
        nodes = self._order_compartments()
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
        """Column vector of amounts for all compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> import sympy
        >>> model = load_example_model("pheno")
        >>> sympy.pprint(model.statements.ode_system.amounts)
        ⎡A_CENTRAL⎤
        ⎢         ⎥
        ⎣A_OUTPUT ⎦
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
        ['CENTRAL', 'OUTPUT']
        """
        ordered_cmts = self._order_compartments()
        names = [cmt.name for cmt in ordered_cmts]
        return names

    def _order_compartments(self):
        """Return list of all compartments in canonical order"""
        try:
            dosecmt = self.dosing_compartment
        except ValueError:
            # Fallback for cases where no dose is available (yet)
            return list(self._g.nodes)
        # Order compartments
        output = self.output_compartment

        def sortfunc(x):
            a = list(x)
            if output in a:
                a.remove(output)
            a = sorted(a, key=lambda x: x.name)
            return iter(a)

        nodes = list(nx.bfs_tree(self._g, dosecmt, sort_neighbors=sortfunc))
        nodes.append(output)
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
        ⎡0⎤
        ⎢ ⎥
        ⎣0⎦
        >>> set_zero_order_absorption(model)    # doctest: +ELLIPSIS
        <...>
        >>> sympy.pprint(model.statements.ode_system.zero_order_inputs)
        ⎡⎧ AMT                ⎤
        ⎢⎪─────  for t < 2⋅MAT⎥
        ⎢⎨2⋅MAT               ⎥
        ⎢⎪                    ⎥
        ⎢⎩  0      otherwise  ⎥
        ⎢                     ⎥
        ⎣          0          ⎦
        """
        inputs = []
        for node in self._order_compartments():  # self._g.nodes:
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

    def to_compartmental_system(self):
        return self

    def to_explicit_system(self, skip_output=False):
        """Get the compartmental system as an explicit ODE system

        Parameters
        ----------
        skip_output : boolean
            Set to true to leave the output compartment out

        Results
        -------
        ExplicitODESystem
            The same ODE system with explicit equations and initial conditions

        Example
        -------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> odes = model.statements.ode_system.to_explicit_system()
        >>> odes
        ⎧d                  -CL⋅A_CENTRAL(t)
        ⎪──(A_CENTRAL(t)) = ─────────────────
        ⎪dt                         V
        ⎨d                 CL⋅A_CENTRAL(t)
        ⎪──(A_OUTPUT(t)) = ───────────────
        ⎪dt                       V
        ⎪A_CENTRAL(0) = AMT
        ⎩A_OUTPUT(0) = 0
        """
        amount_funcs = sympy.Matrix([sympy.Function(amt.name)(self.t) for amt in self.amounts])
        derivatives = sympy.Matrix([sympy.Derivative(fn, self.t) for fn in amount_funcs])
        inputs = self.zero_order_inputs
        a = self.compartmental_matrix @ amount_funcs + inputs
        eqs = [sympy.Eq(lhs, canonical_ode_rhs(rhs)) for lhs, rhs in zip(derivatives, a)]
        if skip_output:
            eqs = eqs[:-1]
        ics = {}
        output = self.output_compartment
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
        return ExplicitODESystem(tuple(eqs), ics)

    def __len__(self):
        """The number of compartments including output"""
        return len(self._g.nodes)

    def _repr_html_(self):
        # Use Unicode art for now. There should be ways of drawing networkx
        s = str(self)
        return f'<pre>{s}</pre>'

    def __repr__(self):
        output = self.output_compartment
        output_box = unicode.Box(output.name)
        central = self.central_compartment
        central_box = unicode.Box(central.name)
        depot = self._find_depot()
        current = self.dosing_compartment
        if depot:
            comp = depot
        else:
            comp = central
        transits = []
        while current != comp:
            transits.append(current)
            current = self.get_compartment_outflows(current)[0][0]
        periphs = self.peripheral_compartments
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

        dose = self.dosing_compartment.dose
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
    lag_time : Expression
        Lag time for doses entering this compartment. Default 0
    bioavailability : Expression
        Bioavailability fraction for doses entering this compartment. Default 1

    Examples
    --------
    >>> from pharmpy.model import Bolus, Compartment
    >>> comp = Compartment("CENTRAL")
    >>> comp
    Compartment(CENTRAL)
    >>> comp = Compartment("DEPOT", lag_time="ALAG")
    >>> comp
    Compartment(DEPOT, lag_time=ALAG)
    >>> dose = Bolus.create("AMT")
    >>> comp = Compartment("DEPOT", dose=dose)
    >>> comp
    Compartment(DEPOT, dose=Bolus(AMT))
    """

    def __init__(self, name, dose=None, lag_time=None, bioavailability=None):
        self._name = name
        self._dose = dose
        if lag_time is None:
            self._lag_time = sympy.Integer(0)
        else:
            self._lag_time = lag_time
        if bioavailability is None:
            self._bioavailability = sympy.Integer(1)
        else:
            self._bioavailability = bioavailability

    @classmethod
    def create(cls, name, dose=None, lag_time=None, bioavailability=None):
        if not isinstance(name, str):
            raise TypeError("Name of a Compartment must be of string type")
        if dose is not None and not isinstance(dose, Dose):
            raise TypeError("dose must be of Dose type (or None)")
        if lag_time is not None:
            lag_time = parse_expr(lag_time)
        if bioavailability is not None:
            bioavailability = parse_expr(bioavailability)
        return cls(name, dose, lag_time, bioavailability)

    @property
    def name(self):
        """Compartment name"""
        return self._name

    @property
    def dose(self):
        return self._dose

    @property
    def lag_time(self):
        """Lag time for doses into compartment"""
        return self._lag_time

    @property
    def bioavailability(self):
        """Bioavailability fraction for doses into compartment"""
        return self._bioavailability

    @property
    def amount(self):
        """Symbol for the amount in the compartment

        Examples
        --------
        >>> from pharmpy.model import Compartment
        >>> comp = Compartment("CENTRAL")
        >>> comp.amount
        A_CENTRAL
        """
        return sympy.Symbol(f'A_{self.name}')

    @property
    def free_symbols(self):
        """Get set of all free symbols in the compartment

        Examples
        --------
        >>> from pharmpy.model import Bolus, Compartment
        >>> dose = Bolus.create("AMT")
        >>> comp = Compartment("CENTRAL", dose=dose, lag_time="ALAG")
        >>> comp.free_symbols  # doctest: +SKIP
        {ALAG, AMT}
        """
        symbs = set()
        if self.dose is not None:
            symbs |= self.dose.free_symbols
        symbs |= self.lag_time.free_symbols
        symbs |= self.bioavailability.free_symbols
        return symbs

    def subs(self, substitutions):
        """Substitute expressions or symbols in compartment

        Examples
        --------
        >>> from pharmpy.model import Bolus, Compartment
        >>> dose = Bolus.create("AMT")
        >>> comp = Compartment("CENTRAL", dose=dose)
        >>> comp.subs({"AMT": "DOSE"})
        Compartment(CENTRAL, dose=Bolus(DOSE))
        """
        if self.dose is not None:
            dose = self.dose.subs(substitutions)
        else:
            dose = None
        return Compartment(
            self.name,
            dose,
            subs(self.lag_time, substitutions),
            subs(self.bioavailability, substitutions),
        )

    def __eq__(self, other):
        return (
            isinstance(other, Compartment)
            and self.name == other.name
            and self.dose == other.dose
            and self.lag_time == other.lag_time
        )

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        lag = '' if self.lag_time == 0 else f', lag_time={self.lag_time}'
        dose = '' if self.dose is None else f', dose={self.dose}'
        return f'Compartment({self.name}{dose}{lag})'


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

    Examples
    --------
    >>> from pharmpy.model import Bolus
    >>> dose = Bolus.create("AMT")
    >>> dose
    Bolus(AMT)
    """

    def __init__(self, amount):
        self._amount = amount

    @classmethod
    def create(cls, amount):
        return cls(parse_expr(amount))

    @property
    def amount(self):
        """Symbolic amount of dose"""
        return self._amount

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
        Bolus(DOSE)
        """
        return Bolus(subs(self.amount, substitutions, simultaneous=True))

    def __eq__(self, other):
        return isinstance(other, Bolus) and self.amount == other.amount

    def __repr__(self):
        return f'Bolus({self.amount})'


class Infusion(Dose):
    """An infusion dose

    Parameters
    ----------
    amount : expression
        Symbolic amount of dose
    rate : expression
        Symbolic rate. Mutually exclusive with duration
    duration : expression
        Symbolic duration. Mutually excluseive with rate

    Examples
    --------
    >>> from pharmpy.model import Infusion
    >>> dose = Infusion("AMT", duration="D1")
    >>> dose
    Infusion(AMT, duration=D1)
    >>> dose = Infusion("AMT", rate="R1")
    >>> dose
    Infusion(AMT, rate=R1)
    """

    def __init__(self, amount, rate=None, duration=None):
        self._amount = amount
        self._rate = rate
        self._duration = duration

    @classmethod
    def create(cls, amount, rate=None, duration=None):
        if rate is None and duration is None:
            raise ValueError('Need rate or duration for Infusion')
        if rate is not None and duration is not None:
            raise ValueError('Cannot have both rate and duration for Infusion')
        if rate is not None:
            rate = parse_expr(rate)
        else:
            duration = parse_expr(duration)
        return cls(parse_expr(amount), rate, duration)

    @property
    def amount(self):
        """Symbolic amount of dose"""
        return self._amount

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
        Infusion(AMT, duration=D1)
        """
        amount = subs(self.amount, substitutions, simultaneous=True)
        if self.rate is not None:
            rate = subs(self.rate, substitutions, simultaneous=True)
            duration = None
        else:
            rate = None
            duration = subs(self.duration, substitutions, simultaneous=True)
        return Infusion(amount, rate, duration)

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


class Statements(Sequence):
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
        Bolus(AMT)
        ┌───────┐       ┌──────┐
        │CENTRAL│──CL/V→│OUTPUT│
        └───────┘       └──────┘
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
        TVCL = THETA(1)⋅WGT
        TVV = THETA(2)⋅WGT
              ⎧TVV⋅(THETA(3) + 1)  for APGR < 5
              ⎨
        TVV = ⎩       TVV           otherwise
                   ETA(1)
        CL = TVCL⋅ℯ
                 ETA(2)
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
        Y = EPS(1)⋅W + F
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
        Y = EPS(1)⋅W + F
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
        TVCL = THETA(1)⋅WT
        TVV = THETA(2)⋅WT
              ⎧TVV⋅(THETA(3) + 1)  for APGR < 5
              ⎨
        TVV = ⎩       TVV           otherwise
               ETA(1)
        CL = TVCL⋅ℯ
             ETA(2)
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
                if statement.symbol == symbol:
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
                   ETA(1)
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
                   ETA(1)
        CL = TVCL⋅ℯ
                 ETA(2)
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
        THETA(1)*WGT*exp(ETA(1))
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

    def to_compartmental_system(self):
        """Convert ODE system to a compartmental system

        raise if not possible
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> statements = model.statements.to_compartmental_system()
        """
        i = self._get_ode_system_index()
        if i == -1:
            return self

        odes = self[i]
        assert isinstance(odes, ODESystem)

        if isinstance(odes, CompartmentalSystem):
            return self

        return self[:i] + odes.to_compartmental_system() + self[i + 1 :]

    def to_explicit_system(self):
        """Convert ODE system to an explicit ODE system

        Example
        -------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system
        Bolus(AMT)
        ┌───────┐       ┌──────┐
        │CENTRAL│──CL/V→│OUTPUT│
        └───────┘       └──────┘
        >>> model.statements.ode_system.to_explicit_system()
        ⎧d                  -CL⋅A_CENTRAL(t)
        ⎪──(A_CENTRAL(t)) = ─────────────────
        ⎪dt                         V
        ⎨d                 CL⋅A_CENTRAL(t)
        ⎪──(A_OUTPUT(t)) = ───────────────
        ⎪dt                       V
        ⎪A_CENTRAL(0) = AMT
        ⎩A_OUTPUT(0) = 0
        """
        i = self._get_ode_system_index()
        if i == -1:
            return self

        odes = self[i]
        assert isinstance(odes, ODESystem)

        if isinstance(odes, ExplicitODESystem):
            return self

        return self[:i] + odes.to_explicit_system() + self[i + 1 :]

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        else:
            for first, second in zip(self, other):
                if first != second:
                    return False
        return True

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
