import copy
from abc import ABC, abstractmethod
from collections.abc import MutableSequence

import networkx as nx
import sympy

import pharmpy.symbols as symbols
import pharmpy.unicode as unicode


class Statement(ABC):
    """Abstract base class for all types of statements"""

    @abstractmethod
    def subs(self, substitutions):
        pass

    @property
    @abstractmethod
    def free_symbols(self):
        pass

    @property
    @abstractmethod
    def rhs_symbols(self):
        pass


class Assignment(Statement):
    """Representation of variable assignment

    This class is similar to :class:`sympy.codegen.Assignment` and are
    combined together into a ModelStatements object.

    Attributes
    ----------
    symbol : sympy.Symbol
        Symbol of statement
    expression : sympy.Expr
        Expression of statement
    """

    def __init__(self, symbol, expression):
        try:
            symbol.is_Symbol
            self.symbol = symbol
        except AttributeError:
            self.symbol = sympy.Symbol(symbol)
        self.expression = sympy.sympify(expression)

    def subs(self, substitutions):
        """Substitute expressions or symbols in assignment

        Parameters
        ----------
        substitutions : dict
            old-new pairs

        Examples
        --------
        >>> from pharmpy import Assignment
        >>> a = Assignment('CL', 'POP_CL + ETA_CL')
        >>> a
        CL := ETA_CL + POP_CL
        >>> a.subs({'ETA_CL' : 'ETA_CL * WGT'})
        >>> a
        CL := ETA_CL⋅WGT + POP_CL

        """
        self.symbol = self.symbol.subs(substitutions, simultaneous=True)
        self.expression = self.expression.subs(substitutions, simultaneous=True)

    @property
    def free_symbols(self):
        """Get set of all free symbols in the assignment

        Note that the left hand side symbol will be in the set

        Examples
        --------
        >>> from pharmpy import Assignment
        >>> a = Assignment('CL', 'POP_CL + ETA_CL')
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
        >>> from pharmpy import Assignment
        >>> a = Assignment('CL', 'POP_CL + ETA_CL')
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
        definition = f'{sympy.pretty(self.symbol)} := '
        s = ''
        for line in lines:
            if line == lines[-1]:
                s += definition + line + '\n'
            else:
                s += len(definition) * ' ' + line + '\n'
        return s.rstrip()

    def __deepcopy__(self, memo):
        return type(self)(self.symbol, self.expression)

    def copy(self):
        """Create a copy of the Assignment object"""
        return copy.deepcopy(self)

    def _repr_latex_(self):
        sym = self.symbol._repr_latex_()[1:-1]
        expr = self.expression._repr_latex_()[1:-1]
        return f'${sym} := {expr}$'


class ODESystem(Statement):
    """Base class and placeholder for ODE systems of different forms

    Attributes
    ----------
    solver : str
        Solver to use when numerically solving the ode system
        Supported solvers and their NONMEM ADVAN

        +------------------------+------------------+
        | Solver                 | NONMEM ADVAN     |
        +========================+==================+
        | CVODES                 | ADVAN14          |
        +------------------------+------------------+
        | DGEAR                  | ADVAN8           |
        +------------------------+------------------+
        | DVERK                  | ADVAN6           |
        +------------------------+------------------+
        | IDA                    | ADVAN15          |
        +------------------------+------------------+
        | LSODA                  | ADVAN13          |
        +------------------------+------------------+
        | LSODI                  | ADVAN9           |
        +------------------------+------------------+
    """

    t = symbols.symbol('t')

    def __init__(self):
        self._solver = None

    @property
    def free_symbols(self):
        return set()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, value):
        supported = ['LSODA']
        if not (value is None or value.upper() in supported):
            raise ValueError(f"Unknown solver {value}. Recognized solvers are {supported}.")
        self._solver = value

    @property
    def rhs_symbols(self):
        return set()

    def subs(self, substitutions):
        pass

    def __eq__(self, other):
        return isinstance(other, ODESystem)

    def __repr__(self):
        return 'ODESystem()'

    def _repr_html_(self):
        return str(self)

    def _repr_latex_(self):
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
    """System of ODEs described explicitly

    Attributes
    ----------
    odes : list
        Symbolic differential equations
    ics : dict
        Symbolic initial conditions

    Examples
    --------
    >>> from pharmpy import ExplicitODESystem
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

    def __init__(self, odes, ics):
        self.odes = odes
        self.ics = ics
        super().__init__()

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
        >>> model.statements.to_explicit_system()
        >>> model.statements.ode_system.free_symbols  # doctest: +SKIP
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
        >>> model.statements.to_explicit_system()
        >>> model.statements.ode_system.subs({'AMT': 'DOSE'})
        >>> model.statements.ode_system
        ⎧d                  -CL⋅A_CENTRAL(t)
        ⎪──(A_CENTRAL(t)) = ─────────────────
        ⎪dt                         V
        ⎨d                 CL⋅A_CENTRAL(t)
        ⎪──(A_OUTPUT(t)) = ───────────────
        ⎪dt                       V
        ⎪A_CENTRAL(0) = DOSE
        ⎩A_OUTPUT(0) = 0
        """
        d = {
            sympy.Function(str(key))(symbols.symbol('t')): value
            for key, value in substitutions.items()
        }
        d.update(substitutions)
        self.odes = [ode.subs(d) for ode in self.odes]
        self.ics = {key.subs(d): value.subs(d) for key, value in self.ics.items()}

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
        >>> model.statements.to_explicit_system()
        >>> model.statements.ode_system.rhs_symbols   # doctest: +SKIP
        {AMT, CL, V, t}
        """
        return self.free_symbols

    def __repr__(self):
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

    def to_compartmental_system(self):
        """Get the explicit system as a compartmental ODE system

        Results
        -------
        CompartmentalSystem
            The same ODE system in compartmental representation

        Example
        -------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.to_explicit_system()
        >>> odes = model.statements.ode_system.to_compartmental_system()
        >>> odes
        Bolus(AMT)
        ┌───────┐       ┌──────┐
        │CENTRAL│──CL/V→│OUTPUT│
        └───────┘       └──────┘
        """

        def convert_name(name):
            if name.startswith('A_'):
                return name[2:]
            else:
                return name

        funcs = [eq.lhs.args[0] for eq in self.odes]
        cs = CompartmentalSystem()
        for f in funcs:
            cs.add_compartment(convert_name(f.name))

        for eq in self.odes:
            for comp_func in funcs:
                dep = eq.rhs.as_independent(comp_func, as_Add=True)[1]
                if dep == 0:
                    continue
                terms = sympy.Add.make_args(dep)
                for term in terms:
                    expr = term / comp_func
                    if term.args[0] != -1:
                        from_comp = cs.find_compartment(convert_name(comp_func.name))
                        to_comp = cs.find_compartment(convert_name(eq.lhs.args[0].name))
                        cs.add_flow(from_comp, to_comp, expr)

        dose = Bolus("AMT")  # FIXME: not true in general!
        cs.find_compartment(convert_name(funcs[0].name)).dose = dose
        return cs


class CompartmentalSystem(ODESystem):
    """System of ODEs descibed as a compartmental system

    Examples
    --------
    >>> from pharmpy import Bolus, CompartmentalSystem
    >>> cs = CompartmentalSystem()
    >>> dose = Bolus("AMT")
    >>> central = cs.add_compartment("CENTRAL")
    >>> central.dose = dose
    >>> peripheral = cs.add_compartment("PERIPHERAL")
    >>> output = cs.add_compartment("OUTPUT")
    >>> cs.add_flow(central, peripheral, "K12")
    >>> cs.add_flow(peripheral, central, "K21")
    >>> cs.add_flow(central, output, "CL / V")
    >>> cs        # doctest: +SKIP
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

    def __init__(self):
        self._g = nx.DiGraph()
        super().__init__()

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
        free = {symbols.symbol('t')}
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
        >>> model.statements.ode_system
        Bolus(DOSE)
        ┌───────┐       ┌──────┐
        │CENTRAL│──CL/V→│OUTPUT│
        └───────┘       └──────┘
        """
        for (u, v, rate) in self._g.edges.data('rate'):
            rate_sub = rate.subs(substitutions, simultaneous=True)
            self._g.edges[u, v]['rate'] = rate_sub
        for comp in self._g.nodes:
            comp.subs(substitutions)

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
            and self.solver == other.solver
        )

    def __deepcopy__(self, memo):
        newone = type(self)()
        newone._g = copy.deepcopy(self._g, memo)
        return newone

    def add_compartment(self, name):
        """Create and add compartment to system

        The compartment will be added without any flows to other compartments.
        Use the add_flow method to add flows to and from the newly added compartment.

        Parameters
        ----------
        name : str
            Name for created compartment

        Returns
        -------
        Compartment
            Newly created compartment

        Examples
        --------
        >>> from pharmpy import CompartmentalSystem
        >>> odes = CompartmentalSystem()
        >>> central = odes.add_compartment("CENTRAL")
        """
        comp = Compartment(name)
        self._g.add_node(comp)
        return comp

    def remove_compartment(self, compartment):
        """Remove compartment from system

        Parameters
        ----------
        compartment : Compartment
            Compartment object to remove from system

        Examples
        --------
        >>> from pharmpy import CompartmentalSystem
        >>> odes = CompartmentalSystem()
        >>> central = odes.add_compartment("CENTRAL")
        >>> odes.remove_compartment(central)
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
        >>> from pharmpy import CompartmentalSystem
        >>> odes = CompartmentalSystem()
        >>> depot = odes.add_compartment("DEPOT")
        >>> central = odes.add_compartment("CENTRAL")
        >>> odes.add_flow(depot, central, "KA")
        """
        self._g.add_edge(source, destination, rate=sympy.sympify(rate))

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
        >>> from pharmpy import CompartmentalSystem
        >>> odes = CompartmentalSystem()
        >>> depot = odes.add_compartment("DEPOT")
        >>> central = odes.add_compartment("CENTRAL")
        >>> odes.add_flow(depot, central, "KA")
        >>> odes.remove_flow(depot, central)
        """
        self._g.remove_edge(source, destination)

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
        list
            Symbolic rates

        Examples
        --------
        >>> from pharmpy import CompartmentalSystem
        >>> odes = CompartmentalSystem()
        >>> depot = odes.add_compartment("DEPOT")
        >>> central = odes.add_compartment("CENTRAL")
        >>> odes.add_flow(depot, central, "KA")
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
        central = next(self._g.predecessors(output))
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

    def find_transit_compartments(self, statements):
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
        nodes = list(self._g.nodes)
        output = self.output_compartment
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
        """Column vector of amounts for all compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model, set_first_order_absorption
        >>> import sympy
        >>> model = load_example_model("pheno")
        >>> sympy.pprint(model.statements.ode_system.amounts)
        ⎡A_CENTRAL⎤
        ⎢         ⎥
        ⎣A_OUTPUT ⎦
        """
        amts = [node.amount for node in self._g.nodes]
        return sympy.Matrix(amts)

    @property
    def names(self):
        """List of the names of all compartments

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model, set_first_order_absorption
        >>> model = load_example_model("pheno")
        >>> model.statements.ode_system.names
        ['CENTRAL', 'OUTPUT']
        """
        return [node.name for node in self._g.nodes]

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
        eqs = [sympy.Eq(lhs, rhs) for lhs, rhs in zip(derivatives, a)]
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
        if skip_output:
            eqs = eqs[:-1]
        return ExplicitODESystem(eqs, ics)

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

    Attributes
    ----------
    name : str
        Compartment name
    dose : Dose
        Dose object for dose into this compartment. Default None for no dose.
    lag_time : Expression
        Lag time for doses entering this compartment. Default 0

    Examples
    --------
    >>> from pharmpy import Bolus, Compartment
    >>> comp = Compartment("CENTRAL")
    >>> comp
    Compartment(CENTRAL)
    >>> comp = Compartment("DEPOT", lag_time="ALAG")
    >>> comp
    Compartment(DEPOT, lag_time=ALAG)
    >>> dose = Bolus("AMT")
    >>> comp = Compartment("DEPOT", dose=dose)
    >>> comp
    Compartment(DEPOT, dose=Bolus(AMT))
    """

    def __init__(self, name, dose=None, lag_time=0):
        self.name = name
        self.dose = dose
        self.lag_time = lag_time

    @property
    def amount(self):
        """Symbol for the amount in the compartment

        Examples
        --------
        >>> from pharmpy import Compartment
        >>> comp = Compartment("CENTRAL")
        >>> comp.amount
        A_CENTRAL
        """
        return symbols.symbol(f'A_{self.name}')

    @property
    def lag_time(self):
        return self._lag_time

    @lag_time.setter
    def lag_time(self, value):
        self._lag_time = sympy.sympify(value)

    @property
    def free_symbols(self):
        """Get set of all free symbols in the compartment

        Examples
        --------
        >>> from pharmpy import Bolus, Compartment
        >>> dose = Bolus("AMT")
        >>> comp = Compartment("CENTRAL", dose=dose, lag_time="ALAG")
        >>> comp.free_symbols  # doctest: +SKIP
        {ALAG, AMT}
        """
        symbs = set()
        if self.dose is not None:
            symbs |= self.dose.free_symbols
        symbs |= self.lag_time.free_symbols
        return symbs

    def subs(self, substitutions):
        """Substitute expressions or symbols in compartment

        Examples
        --------
        >>> from pharmpy import Bolus, Compartment
        >>> dose = Bolus("AMT")
        >>> comp = Compartment("CENTRAL", dose=dose)
        >>> comp.subs({"AMT": "DOSE"})
        >>> comp
        Compartment(CENTRAL, dose=Bolus(DOSE))
        """
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

    def __repr__(self):
        lag = '' if self.lag_time == 0 else f', lag_time={self.lag_time}'
        dose = '' if self.dose is None else f', dose={self.dose}'
        return f'Compartment({self.name}{dose}{lag})'


class Bolus:
    """A Bolus dose

    Attributes
    ----------
    amount : symbol
        Symbolic amount of dose

    Examples
    --------
    >>> from pharmpy import Bolus
    >>> dose = Bolus("AMT")
    >>> dose
    Bolus(AMT)
    """

    def __init__(self, amount):
        self.amount = sympy.sympify(amount)

    @property
    def free_symbols(self):
        """Get set of all free symbols in the dose

        Examples
        --------
        >>> from pharmpy import Bolus
        >>> dose = Bolus("AMT")
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
        >>> from pharmpy import Bolus
        >>> dose = Bolus("AMT")
        >>> dose.subs({'AMT': 'DOSE'})
        >>> dose
        Bolus(DOSE)
        """
        self.amount = self.amount.subs(substitutions, simultaneous=True)

    def __deepcopy__(self, memo):
        newone = type(self)(self.amount)
        return newone

    def __eq__(self, other):
        return isinstance(other, Bolus) and self.amount == other.amount

    def __repr__(self):
        return f'Bolus({self.amount})'


class Infusion:
    """An infusion dose

    Attributes
    ----------
    amount : expression
        Symbolic amount of dose
    rate : expression
        Symbolic rate. Mutually excluseive with duration
    duration : expression
        Symbolic duration. Mutually excluseive with rate

    Examples
    --------
    >>> from pharmpy import Infusion
    >>> dose = Infusion("AMT", duration="D1")
    >>> dose
    Infusion(AMT, duration=D1)
    >>> dose = Infusion("AMT", rate="R1")
    >>> dose
    Infusion(AMT, rate=R1)
    """

    def __init__(self, amount, rate=None, duration=None):
        if rate is None and duration is None:
            raise ValueError('Need rate or duration for Infusion')
        self.rate = sympy.sympify(rate)
        self.duration = sympy.sympify(duration)
        self.amount = sympy.sympify(amount)

    @property
    def free_symbols(self):
        """Get set of all free symbols in the dose

        Examples
        --------
        >>> from pharmpy import Infusion
        >>> dose = Infusion("AMT", rate="RATE")
        >>> dose.free_symbols   # doctest: +SKIP
        {AMT, RATE}
        """
        if self.rate is not None:
            symbs = self.rate.free_symbols
        else:
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
        >>> from pharmpy import Infusion
        >>> dose = Infusion("AMT", duration="DUR")
        >>> dose.subs({'DUR': 'D1'})
        >>> dose
        Infusion(AMT, duration=D1)
        """
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


class ModelStatements(MutableSequence):
    """A sequence of symbolic statements describing the model

    Two types of statements are supported: Assignment and ODESystem.
    A ModelStatements object can have 0 or 1 ODESystem. The order of
    the statements is significant and the same symbol can be assigned
    to multiple times.
    """

    def __init__(self, statements=None):
        if isinstance(statements, ModelStatements):
            self._statements = copy.deepcopy(statements._statements)
        elif statements is None:
            self._statements = []
        else:
            self._statements = list(statements)

    def __getitem__(self, ind):
        return self._statements[ind]

    def __setitem__(self, ind, value):
        self._statements[ind] = value

    def __delitem__(self, ind):
        del self._statements[ind]

    def __len__(self):
        return len(self._statements)

    def insert(self, ind, value):
        self._statements.insert(ind, value)

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

    @property
    def ode_system(self):
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
        for s in self:
            if isinstance(s, ODESystem):
                return s
        return None

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
        BTIME := ⎩ 0     otherwise
        TAD := -BTIME + TIME
        TVCL := THETA(1)⋅WGT
        TVV := THETA(2)⋅WGT
               ⎧TVV⋅(THETA(3) + 1)  for APGR < 5
               ⎨
        TVV := ⎩       TVV           otherwise
                    ETA(1)
        CL := TVCL⋅ℯ
                  ETA(2)
        V := TVV⋅ℯ
        S₁ := V
        """
        sset = ModelStatements()
        for s in self:
            if isinstance(s, ODESystem):
                break
            sset.append(s)
        return sset

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
        F :=     S₁
        W := F
        Y := EPS(1)⋅W + F
        IPRED := F
        IRES := DV - IPRED
                  IRES
                  ────
        IWRES :=   W
        """
        sset = ModelStatements()
        found = False
        if self.ode_system is None:
            return self
        for s in self:
            if isinstance(s, ODESystem):
                found = True
            elif found:
                sset.append(s)
        return sset

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
        >>> model.statements.subs({'WGT': 'WT'})

        """
        for statement in self:
            statement.subs(substitutions)

    def find_assignment(self, symbol):
        """Returns last assignment of symbol

        Parameters
        ----------
        symbol : Symbol or str
            Symbol to look for

        Returns
        -------
        Assignment
            An Assignment or None if no assignment to symbol exists

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.find_assignment("CL")
                    ETA(1)
        CL := TVCL⋅ℯ
        """
        symbol = sympy.sympify(symbol)
        assignment = None
        for statement in self:
            if isinstance(statement, Assignment):
                if statement.symbol == symbol:
                    assignment = statement
        return assignment

    def reassign(self, symbol, expression):
        """Reassign symbol to expression

        Set symbol to be expression and remove all previous assignments of symbol

        Parameters
        ----------
        symbol : Symbol or str
            Symbol to reassign
        expression : Expression or str
            The new expression to assign to symbol

        Examples
        --------
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.reassign("CL", "TVCL + eta")
        """
        if isinstance(symbol, str):
            symbol = sympy.sympify(symbol)
        if isinstance(expression, str):
            expression = sympy.sympify(expression)

        last = True
        for i, stat in zip(range(len(self) - 1, -1, -1), reversed(self)):
            if isinstance(stat, Assignment) and stat.symbol == symbol:
                if last:
                    stat.expression = expression
                    last = False
                else:
                    del self[i]

    def _create_dependency_graph(self):
        """Create a graph of dependencies between statements"""
        graph = nx.DiGraph()
        for i in range(len(self) - 1, -1, -1):
            rhs = self[i].rhs_symbols
            for s in rhs:
                for j in range(i - 1, -1, -1):
                    if (
                        isinstance(self[j], Assignment)
                        and self[j].symbol == s
                        or isinstance(self[j], ODESystem)
                        and s in self[j].amounts
                    ):
                        graph.add_edge(i, j)
        return graph

    def dependencies(self, symbol):
        """Find all dependencies of a symbol

        Parameters
        ----------
        symbol : Symbol or str
            Input symbol

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
        symbol = sympy.sympify(symbol)
        for i in range(len(self) - 1, -1, -1):
            if (
                isinstance(self[i], Assignment)
                and self[i].symbol == symbol
                or isinstance(self[i], ODESystem)
                and symbol in self[i].amounts
            ):
                break
        else:
            raise KeyError(f"Could not find symbol {symbol}")
        g = self._create_dependency_graph()
        symbs = self[i].rhs_symbols
        if i == 0:
            # Special case for models with only one statement
            return symbs
        for j, _ in nx.bfs_predecessors(g, i, sort_neighbors=lambda x: reversed(sorted(x))):
            if isinstance(self[j], Assignment):
                symbs -= {self[j].symbol}
            elif isinstance(self[j], ODESystem):
                symbs -= set(self[j].amounts)
            symbs |= self[j].rhs_symbols
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
            expression = sympy.sympify(expression)
        for statement in reversed(self):
            if isinstance(statement, ODESystem):
                raise ValueError(
                    "ODESystem not supported by full_expression. Use the properties before_odes "
                    "or after_odes."
                )
            expression = expression.subs({statement.symbol: statement.expression})
        return expression

    def insert_before(self, statement, new_statement):
        """Insert a statement just before another statement

        Parameters
        ----------
        statement : Statement
            Insert before this statement
        new_statement : Statement
            Statement to insert

        Examples
        --------
        >>> from pharmpy import Assignment
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> a = Assignment("WGT_G", "WGT*1000")
        >>> b = model.statements.find_assignment("CL")
        >>> model.statements.insert_before(b, a)

        """
        i = self.index(statement)
        self.insert(i, new_statement)

    def insert_before_odes(self, statement):
        """Insert a statement just before the ODE system or at the end of the model

        Parameters
        ----------
        statement : Statement
            Statement to insert

        Examples
        --------
        >>> from pharmpy import Assignment
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> a = Assignment("WGT_G", "WGT*1000")
        >>> model.statements.insert_before_odes(a)
        """
        for i, s in enumerate(self):
            if isinstance(s, ODESystem):
                break
        else:
            i += 1
        self.insert(i, statement)

    def to_compartmental_system(self):
        """Convert ODE system to a compartmental system

        raise if not possible
        >>> from pharmpy.modeling import load_example_model
        >>> model = load_example_model("pheno")
        >>> model.statements.to_explicit_system()
        >>> model.statements.to_compartmental_system()
        >>> model.statements.ode_system
        Bolus(AMT)
        ┌───────┐       ┌──────┐
        │CENTRAL│──CL/V→│OUTPUT│
        └───────┘       └──────┘
        """
        for i, s in enumerate(self):
            if isinstance(s, ExplicitODESystem):
                new = s.to_compartmental_system()
                self[i] = new
                return

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
        >>> model.statements.to_explicit_system()
        >>> model.statements.ode_system
        ⎧d                  -CL⋅A_CENTRAL(t)
        ⎪──(A_CENTRAL(t)) = ─────────────────
        ⎪dt                         V
        ⎨d                 CL⋅A_CENTRAL(t)
        ⎪──(A_OUTPUT(t)) = ───────────────
        ⎪dt                       V
        ⎪A_CENTRAL(0) = AMT
        ⎩A_OUTPUT(0) = 0
        """
        for i, s in enumerate(self):
            if isinstance(s, CompartmentalSystem):
                new = s.to_explicit_system()
                self[i] = new
                return

    def copy(self):
        """Create a copy of the ModelStatements object"""
        return copy.deepcopy(self)

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
                s = s.replace(':=', '&:=')
                s = s.replace('$', '')
                s = s + r'\\'
                html += s
        return html + '\\end{align*}'
