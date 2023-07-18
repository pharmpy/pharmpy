from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import filterfalse
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar, Union

from pharmpy.deps import sympy
from pharmpy.internals.expr.assumptions import assume_all
from pharmpy.internals.expr.leaves import free_images_and_symbols
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.graph.directed.connected_components import strongly_connected_component_of
from pharmpy.internals.graph.directed.inverse import inverse as graph_inverse
from pharmpy.internals.graph.directed.reachability import reachable_from
from pharmpy.model import (
    Assignment,
    Compartment,
    CompartmentalSystem,
    Model,
    Statement,
    Statements,
    output,
)

from .parameters import get_thetas

T = TypeVar('T')
U = TypeVar('U')


def get_observation_expression(model: Model):
    """Get the full symbolic expression for the observation according to the model

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Expression
        Symbolic expression

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_observation_expression
    >>> import sympy
    >>> model = load_example_model("pheno_linear")
    >>> expr = get_observation_expression(model)
    >>> sympy.pprint(expr)
    D_EPSETA1_2⋅EPS₁⋅(ETA₂ - OETA₂) + D_ETA1⋅(ETA₁ - OETA₁) + D_ETA2⋅(ETA₂ - OETA₂
    ) + EPS₁⋅(D_EPS1 + D_EPSETA1_1⋅(ETA₁ - OETA₁)) + OPRED
    """
    stats = model.statements
    # FIXME: handle other DVs
    dv = list(model.dependent_variables.keys())[0]
    for i, s in enumerate(stats):
        if s.symbol == dv:
            y = s.expression
            break
    else:
        raise ValueError('Could not locate dependent variable expression')

    for j in range(i, -1, -1):
        y = subs(y, {stats[j].symbol: stats[j].expression}, simultaneous=True)

    return y


def get_individual_prediction_expression(model: Model):
    """Get the full symbolic expression for the modelled individual prediction

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Expression
        Symbolic expression

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_individual_prediction_expression
    >>> model = load_example_model("pheno_linear")
    >>> get_individual_prediction_expression(model)
    D_ETA1*(ETA_1 - OETA1) + D_ETA2*(ETA_2 - OETA2) + OPRED

    See Also
    --------
    get_population_prediction_expression : Get full symbolic epression for the population prediction
    """
    return subs(
        get_observation_expression(model),
        {sympy.Symbol(eps): 0 for eps in model.random_variables.epsilons.names},
        simultaneous=True,
    )


def get_population_prediction_expression(model: Model):
    """Get the full symbolic expression for the modelled population prediction

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Expression
        Symbolic expression

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_population_prediction_expression
    >>> model = load_example_model("pheno_linear")
    >>> get_population_prediction_expression(model)
    -D_ETA1*OETA1 - D_ETA2*OETA2 + OPRED

    See also
    --------
    get_individual_prediction_expression : Get full symbolic epression for the individual prediction
    """

    return subs(
        get_individual_prediction_expression(model),
        {sympy.Symbol(eta): 0 for eta in model.random_variables.etas.names},
        simultaneous=True,
    )


def calculate_eta_gradient_expression(model: Model):
    """Calculate the symbolic expression for the eta gradient

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Expression
        Symbolic expression

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, calculate_eta_gradient_expression
    >>> model = load_example_model("pheno_linear")
    >>> calculate_eta_gradient_expression(model)
    [D_ETA1, D_ETA2]

    See also
    --------
    calculate_epsilon_gradient_expression : Epsilon gradient
    """
    y = get_individual_prediction_expression(model)
    d = [y.diff(sympy.Symbol(x)) for x in model.random_variables.etas.names]
    return d


def calculate_epsilon_gradient_expression(model: Model):
    """Calculate the symbolic expression for the epsilon gradient

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Expression
        Symbolic expression

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, calculate_epsilon_gradient_expression
    >>> model = load_example_model("pheno_linear")
    >>> calculate_epsilon_gradient_expression(model)
    [D_EPS1 + D_EPSETA1_1*(ETA_1 - OETA1) + D_EPSETA1_2*(ETA_2 - OETA2)]

    See also
    --------
    calculate_eta_gradient_expression : Eta gradient
    """

    y = get_observation_expression(model)
    d = [y.diff(sympy.Symbol(x)) for x in model.random_variables.epsilons.names]
    return d


def create_symbol(model: Model, stem: str, force_numbering: bool = False):
    """Create a new unique variable symbol given a model

    Parameters
    ----------
    model : Model
        Pharmpy model object
    stem : str
        First part of the new variable name
    force_numbering : bool
        Forces addition of number to name even if variable does not exist, e.g.
        COVEFF --> COVEFF1

    Returns
    -------
    Symbol
        Created symbol with unique name

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, create_symbol
    >>> model = load_example_model("pheno")
    >>> create_symbol(model, "TEMP")
    TEMP
    >>> create_symbol(model, "TEMP", force_numbering=True)
    TEMP1
    >>> create_symbol(model, "CL")
    CL1
    """
    return _create_symbol(
        model.statements,
        model.parameters,
        model.random_variables,
        model.datainfo,
        stem,
        force_numbering,
    )


def _create_symbol(statements, parameters, random_variables, datainfo, stem, force_numbering):
    symbols = [str(symbol) for symbol in statements.free_symbols]
    params = [param.name for param in parameters]
    rvs = random_variables.names
    dataset_col = datainfo.names

    all_names = symbols + params + rvs + dataset_col

    if str(stem) not in all_names and not force_numbering:
        return sympy.Symbol(str(stem))

    i = 1
    while True:
        candidate = f'{stem}{i}'
        if candidate not in all_names:
            return sympy.Symbol(candidate)
        i += 1


def _find_eta_assignments(model):
    # NOTE This locates all assignment to ETAs of symbols that do not depend on
    # any other ETA
    statements = model.statements.before_odes
    etas = {sympy.Symbol(eta) for eta in model.random_variables.etas.names}
    found = set()
    leafs = []

    for i, s in reversed(list(enumerate(statements))):
        if (
            s.symbol not in found
            and not etas.isdisjoint(s.free_symbols)
            and len(etas & statements[:i].full_expression(s.expression).free_symbols) == 1
        ):
            leafs.append((i, s))
            found.update(s.free_symbols)

    return reversed(leafs)


def mu_reference_model(model: Model):
    r"""Convert model to use mu-referencing

    Mu-referencing an eta is to separately define its actual mu (mean) parameter.
    For example: :math:`CL = \theta_1 e^{\eta_1}` with :math:`\eta_1` following a zero-mean
    normal distribution would give :math:`\mu_1 = \log{\theta_1}` and
    :math:`CL = e^{\mu_1 + \eta_1}`

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, mu_reference_model
    >>> model = load_example_model("pheno")
    >>> model = mu_reference_model(model)
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
    μ₁ = log(TVCL)
          ETA₁ + μ₁
    CL = ℯ
    μ₂ = log(TVV)
         ETA₂ + μ₂
    V = ℯ
    S₁ = V
    """
    index = {sympy.Symbol(eta): i for i, eta in enumerate(model.random_variables.etas.names, 1)}
    etas = set(index)

    offset = 0

    statements = model.statements
    for old_ind, assignment in _find_eta_assignments(model):
        # NOTE The sequence of old_ind must be increasing
        eta = next(iter(etas.intersection(assignment.expression.free_symbols)))
        old_def = assignment.expression
        dep = old_def.as_independent(eta)[1]
        mu = sympy.Symbol(f'mu_{index[eta]}')
        new_def = subs(dep, {eta: mu + eta})
        mu_expr = sympy.solve(old_def - new_def, mu)[0]
        insertion_ind = offset + old_ind
        statements = (
            statements[0:insertion_ind]
            + Assignment(mu, mu_expr)
            + Assignment(assignment.symbol, new_def)
            + statements[insertion_ind + 1 :]
        )
        offset += 1  # NOTE We need this offset because we replace one
        # statement by two statements
    model = model.replace(statements=statements).update_source()
    return model


def simplify_expression(model: Model, expr: Union[int, float, str, sympy.Expr]):
    """Simplify expression given constraints in model

    Parameters
    ----------
    model : Model
        Pharmpy model object
    expr : Expression
        Expression to simplify

    Returns
    -------
    Expression
        Simplified expression

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, simplify_expression
    >>> model = load_example_model("pheno")
    >>> simplify_expression(model, "Abs(PTVCL)")
    PTVCL
    """
    return _simplify_expression_from_parameters(expr, model.parameters)


def _simplify_expression_from_parameters(expr, parameters):
    expr = parse_expr(expr)
    d = {}
    for p in parameters:
        if p.fix:
            s = sympy.Float(p.init)
        elif p.upper < 0:
            s = sympy.Symbol(p.name, real=True, negative=True)
            d[s] = p.symbol
        elif p.upper <= 0:
            s = sympy.Symbol(p.name, real=True, nonpositive=True)
            d[s] = p.symbol
        elif p.lower > 0:
            s = sympy.Symbol(p.name, real=True, positive=True)
            d[s] = p.symbol
        elif p.lower >= 0:
            s = sympy.Symbol(p.name, real=True, nonnegative=True)
            d[s] = p.symbol
        else:
            s = sympy.Symbol(p.name, real=True)
            d[s] = p.symbol
        expr = subs(expr, {p.symbol: s})
    # Remaining symbols should all be real
    for s in expr.free_symbols:
        if s.is_real is not True:
            new = sympy.Symbol(s.name, real=True)
            expr = subs(expr, {s: new})
            d[new] = s
    simp = subs(sympy.simplify(expr), d)  # Subs symbols back to non-constrained
    return simp


def make_declarative(model: Model):
    """Make the model statments declarative

    Each symbol will only be declared once.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
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
    >>> model = make_declarative(model)
    >>> model.statements.before_odes
            ⎧TIME  for AMT > 0
            ⎨
    BTIME = ⎩ 0     otherwise
    TAD = -BTIME + TIME
    TVCL = PTVCL⋅WGT
          ⎧PTVV⋅WGT⋅(THETA₃ + 1)  for APGR < 5
          ⎨
    TVV = ⎩      PTVV⋅WGT          otherwise
               ETA₁
    CL = TVCL⋅ℯ
             ETA₂
    V = TVV⋅ℯ
    S₁ = V
    """
    assigned_symbols = set()
    duplicated_symbols = {}  # symbol to last index
    for i, s in enumerate(model.statements):
        if not isinstance(s, Assignment):
            continue
        symb = s.symbol
        if symb in assigned_symbols:
            if symb not in duplicated_symbols:
                duplicated_symbols[symb] = []
            duplicated_symbols[symb].append(i)
        else:
            assigned_symbols.add(symb)

    current = {}
    newstats = []
    for i, s in enumerate(model.statements):
        if not isinstance(s, Assignment):
            s = s.subs(current)
            newstats.append(s)
        elif s.symbol in duplicated_symbols:
            if i not in duplicated_symbols[s.symbol]:
                current[s.symbol] = s.expression
            else:
                duplicated_symbols[s.symbol] = duplicated_symbols[s.symbol][1:]
                if duplicated_symbols[s.symbol]:
                    current[s.symbol] = subs(s.expression, current)
                else:
                    ass = Assignment(s.symbol, subs(s.expression, current))
                    newstats.append(ass)
                    del current[s.symbol]
        else:
            ass = Assignment(s.symbol, subs(s.expression, current))
            newstats.append(ass)

    model = model.replace(statements=Statements(newstats))
    return model.update_source()


def cleanup_model(model: Model):
    """Perform various cleanups of a model

    This is what is currently done

    * Make model statements declarative, i.e. only one assignment per symbol
    * Inline all assignments of one symbol, e.g. X = Y

    Notes
    -----
    When creating NONMEM code from the cleaned model Pharmpy might need to
    add certain assignments to make it in line with what NONMEM requires.

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model.statements
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
    Bolus(AMT, admid=1)
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL
        ─────────
    F =     S₁
    W = F
    Y = EPS₁⋅W + F
    IPRED = F
    IRES = DV - IPRED
            IRES
            ────
    IWRES =  W
    >>> model = cleanup_model(model)
    >>> model.statements
            ⎧TIME  for AMT > 0
            ⎨
    BTIME = ⎩ 0     otherwise
    TAD = -BTIME + TIME
    TVCL = PTVCL⋅WGT
          ⎧PTVV⋅WGT⋅(THETA₃ + 1)  for APGR < 5
          ⎨
    TVV = ⎩      PTVV⋅WGT           otherwise
               ETA₁
    CL = TVCL⋅ℯ
             ETA₂
    V = TVV⋅ℯ
    Bolus(AMT, admid=1)
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL
        ─────────
    F =     V
    Y = EPS₁⋅F + F
    IRES = DV - F
            IRES
            ────
    IWRES =  F
    """
    model = make_declarative(model)

    current = {}
    newstats = []
    for s in model.statements:
        if isinstance(s, Assignment) and s.expression.is_Symbol:
            current[s.symbol] = s.expression
        else:
            n = s.subs(current)
            newstats.append(n)

    model = model.replace(statements=Statements(newstats))
    return model.update_source()


def greekify_model(model: Model, named_subscripts: bool = False):
    """Convert to using greek letters for all population parameters

    Parameters
    ----------
    model : Model
        Pharmpy model
    named_subscripts : bool
        Use previous parameter names as subscripts. Default is to use integer subscripts

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model.statements
            ⎧TIME  for AMT > 0
            ⎨
    BTIME = ⎩ 0     otherwise
    TAD = -BTIME + TIME
    TVCL = PTVCL⋅WGT
    TVV = PTVV⋅WGT
          ⎧TVV⋅(THETA₃ + 1)  for APGR < 5
          ⎨
    TVV = ⎩      TVV           otherwise
               ETA₁
    CL = TVCL⋅ℯ
             ETA₂
    V = TVV⋅ℯ
    S₁ = V
    Bolus(AMT, admid=1)
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL
        ─────────
    F =     S₁
    W = F
    Y = EPS₁⋅W + F
    IPRED = F
    IRES = DV - IPRED
            IRES
            ────
    IWRES =  W

    >>> model = greekify_model(cleanup_model(model))
    >>> model.statements
            ⎧TIME  for AMT > 0
            ⎨
    BTIME = ⎩ 0     otherwise
    TAD = -BTIME + TIME
    TVCL = WGT⋅θ₁
          ⎧WGT⋅θ₂⋅(θ₃ + 1)  for APGR < 5
          ⎨
    TVV = ⎩    WGT⋅θ₂        otherwise
               η₁
    CL = TVCL⋅ℯ
             η₂
    V = TVV⋅ℯ
    Bolus(AMT, admid=1)
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL
        ─────────
    F =     V
    Y = F⋅ε₁ + F
    IRES = DV - F
            IRES
            ────
    IWRES =  F

    """

    def get_subscript(param, i, named_subscripts):
        if named_subscripts:
            if isinstance(param, str):
                subscript = param
            else:
                subscript = param.name
        else:
            subscript = i
        return subscript

    def get_2d_subscript(param, row, col, named_subscripts):
        if named_subscripts:
            subscript = param.name
        else:
            subscript = f'{row}{col}'
        return subscript

    subs = {}
    for i, theta in enumerate(get_thetas(model), start=1):
        subscript = get_subscript(theta, i, named_subscripts)
        subs[theta.symbol] = sympy.Symbol(f"theta_{subscript}")
    omega = model.random_variables.covariance_matrix
    for row in range(omega.rows):
        for col in range(omega.cols):
            if col > row:
                break
            elt = omega[row, col]
            if elt == 0:
                continue
            subscript = get_2d_subscript(elt, row + 1, col + 1, named_subscripts)
            subs[elt] = sympy.Symbol(f"omega_{subscript}")
    sigma = model.random_variables.covariance_matrix
    for row in range(sigma.rows):
        for col in range(sigma.cols):
            if col > row:
                break
            elt = sigma[row, col]
            if elt == 0:
                continue
            subscript = get_2d_subscript(elt, row + 1, col + 1, named_subscripts)
            subs[elt] = sympy.Symbol(f"sigma_{subscript}")
    for i, eta in enumerate(model.random_variables.etas.names, start=1):
        subscript = get_subscript(eta, i, named_subscripts)
        subs[sympy.Symbol(eta)] = sympy.Symbol(f"eta_{subscript}")
    for i, epsilon in enumerate(model.random_variables.epsilons.names, start=1):
        subscript = get_subscript(epsilon, i, named_subscripts)
        subs[sympy.Symbol(epsilon)] = sympy.Symbol(f"epsilon_{subscript}")
    from pharmpy.modeling import rename_symbols

    model = rename_symbols(model, subs)
    return model


def get_individual_parameters(model: Model, level: str = 'all') -> List[str]:
    """Retrieves all parameters with IIV or IOV in :class:`pharmpy.model`.

    Parameters
    ----------
    model : Model
        Pharmpy model to retrieve the individuals parameters from
    level : str
        The variability level to look for: 'iiv', 'iov', or 'all' (default)

    Return
    ------
    list[str]
        A list of the parameter names as strings

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_individual_parameters(model)
    ['CL', 'V']
    >>> get_individual_parameters(model, 'iiv')
    ['CL', 'V']
    >>> get_individual_parameters(model, 'iov')
    []

    See also
    --------
    get_pk_parameters
    get_rv_parameters
    has_random_effect

    """

    rvs = _rvs(model, level)

    assignments = _get_natural_assignments(model.statements.before_odes)

    free_symbols = {assignment.symbol for assignment in assignments}

    dependency_graph = _dependency_graph(assignments)

    return sorted(
        map(
            str,
            _filter_symbols(
                dependency_graph,
                free_symbols,
                set().union(
                    *(rvs[rv].free_symbols for rv in rvs.names if rvs[rv].get_variance(rv) != 0)
                ),
            ),
        )
    )


def _rvs(model: Model, level: str):
    if level == 'iiv':
        return model.random_variables.iiv
    if level == 'iov':
        return model.random_variables.iov
    if level == 'all':
        return model.random_variables.etas

    raise ValueError(f'Cannot handle level `{level}`')


def depends_on(model: Model, symbol: str, other: str):
    return _depends_on_any_of(
        model.statements.before_odes, sympy.Symbol(symbol), [sympy.Symbol(other)]
    )


def _depends_on_any_of(
    assignments: Statements, symbol: sympy.Symbol, symbols: Iterable[sympy.Symbol]
):
    dependency_graph = _dependency_graph(assignments)
    if symbol not in dependency_graph:
        raise KeyError(symbol)

    # NOTE Could be faster by returning immediately once found
    return not reachable_from({symbol}, lambda x: dependency_graph.get(x, [])).isdisjoint(symbols)


def has_random_effect(model: Model, parameter: str, level: str = 'all') -> bool:
    """Decides whether the given parameter of a :class:`pharmpy.model` has a
    random effect.

    Parameters
    ----------
    model : Model
        Input Pharmpy model
    parameter: str
        Input parameter
    level : str
        The variability level to look for: 'iiv', 'iov', or 'all' (default)

    Return
    ------
    bool
        Whether the given parameter has a random effect

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> has_random_effect(model, 'S1')
    True
    >>> has_random_effect(model, 'CL', 'iiv')
    True
    >>> has_random_effect(model, 'CL', 'iov')
    False

    See also
    --------
    get_individual_parameters
    get_rv_parameters

    """

    rvs = _rvs(model, level)
    symbol = sympy.Symbol(parameter)
    return _depends_on_any_of(model.statements.before_odes, symbol, map(sympy.Symbol, rvs.names))


def get_rv_parameters(model: Model, rv: str) -> List[str]:
    """Retrieves parameters in :class:`pharmpy.model` given a random variable.

    Parameters
    ----------
    model : Model
        Pharmpy model to retrieve parameters from
    rv : str
        Name of random variable to retrieve

    Return
    ------
    list[str]
        A list of parameter names for the given random variable

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_rv_parameters(model, 'ETA_1')
    ['CL']

    See also
    --------
    has_random_effect
    get_pk_parameters
    get_individual_parameters

    """
    if rv not in model.random_variables.names:
        raise ValueError(f'Could not find random variable: {rv}')

    natural_assignments = _get_natural_assignments(model.statements.before_odes)

    free_symbols = model.statements.free_symbols
    dependency_graph = _dependency_graph(natural_assignments)
    return sorted(map(str, _filter_symbols(dependency_graph, free_symbols, {sympy.Symbol(rv)})))


@dataclass(frozen=True)
class AssignmentGraphNode:
    expression: sympy.Expr
    index: int
    previous: Dict[sympy.Symbol, AssignmentGraphNode]


def _make_assignments_graph(statements: Statements) -> Dict[sympy.Symbol, AssignmentGraphNode]:
    last_assignments: Dict[sympy.Symbol, AssignmentGraphNode] = {}

    for i, statement in enumerate(statements):
        if not isinstance(statement, Assignment):
            continue

        node = AssignmentGraphNode(
            statement.expression,
            i,
            {
                symbol: last_assignments[symbol]
                for symbol in statement.expression.free_symbols
                if symbol in last_assignments
            },
        )

        last_assignments[statement.symbol] = node

    return last_assignments


def remove_covariate_effect_from_statements(
    model: Model, before_odes: Statements, parameter: str, covariate: str
) -> Iterable[Statement]:
    assignments = _make_assignments_graph(before_odes)

    thetas = _theta_symbols(model)

    new_before_odes = list(before_odes)

    symbol = sympy.Symbol(parameter)
    graph_node = assignments[symbol]

    tree_node = _remove_covariate_effect_from_statements_recursive(
        thetas,
        graph_node.previous,
        new_before_odes,
        symbol,
        graph_node.expression,
        sympy.Symbol(covariate),
        None,
    )

    assert tree_node.changed
    assert tree_node.contains_theta

    if tree_node.changed:
        new_before_odes[graph_node.index] = Assignment(
            sympy.Symbol(parameter), tree_node.expression
        )

    return new_before_odes


def _neutral(expr: sympy.Expr) -> sympy.Integer:
    if isinstance(expr, sympy.Add):
        return sympy.Integer(0)
    if isinstance(expr, sympy.Mul):
        return sympy.Integer(1)
    if isinstance(expr, sympy.Pow):
        return sympy.Integer(1)

    raise ValueError(f'{type(expr)}: {repr(expr)} ({expr.free_symbols})')


def _theta_symbols(model: Model) -> Set[sympy.Symbol]:
    rvs_fs = model.random_variables.free_symbols
    return {p.symbol for p in model.parameters if p.symbol not in rvs_fs}


def _depends_on_any(symbols: Set[sympy.Symbol], expr: sympy.Expr) -> bool:
    return any(map(lambda s: s in symbols, expr.free_symbols))


def _is_constant(thetas: Set[sympy.Symbol], expr: sympy.Expr) -> bool:
    return all(map(lambda s: s in thetas, expr.free_symbols))


def _is_univariate(thetas: Set[sympy.Symbol], expr: sympy.Expr, variable: sympy.Symbol) -> bool:
    return all(map(lambda s: s in thetas, expr.free_symbols - {variable}))


def simplify_model(
    model: Model, old_statements: Iterable[Statement], statements: Iterable[Statement]
):
    odes = model.statements.ode_system
    fs = odes.free_symbols if odes is not None else set()
    old_fs = fs.copy()

    kept_statements_reversed = []

    for old_statement, statement in reversed(list(zip(old_statements, statements))):
        if not isinstance(statement, Assignment):
            kept_statements_reversed.append(statement)
            continue

        assert isinstance(old_statement, Assignment)

        if (old_statement == statement and statement.symbol not in old_fs) or (
            statement.symbol in fs and statement.symbol != statement.expression
        ):
            kept_statements_reversed.append(statement)
            fs.discard(statement.symbol)
            fs.update(statement.expression.free_symbols)

        old_fs.discard(old_statement.symbol)
        old_fs.update(old_statement.expression.free_symbols)

    kept_thetas = fs.intersection(_theta_symbols(model))
    kept_statements = list(reversed(kept_statements_reversed))

    return kept_thetas, kept_statements


@dataclass(frozen=True)
class ExpressionTreeNode:
    expression: sympy.Expr
    changed: bool
    constant: bool
    contains_theta: bool


def _full_expression(assignments: Dict[sympy.Symbol, AssignmentGraphNode], expr: sympy.Expr):
    return expr.xreplace(
        {
            symbol: _full_expression(node.previous, node.expression)
            for symbol, node in assignments.items()
        }
    )


def _remove_covariate_effect_from_statements_recursive(
    thetas: Set[sympy.Symbol],
    assignments: Dict[sympy.Symbol, AssignmentGraphNode],
    statements: List[Assignment],
    symbol: sympy.Symbol,
    expression: sympy.Expr,
    covariate: sympy.Symbol,
    parent: Union[None, sympy.Expr],
) -> ExpressionTreeNode:
    if not expression.args:
        if expression in assignments:
            # NOTE expression is a symbol and is defined in a previous assignment
            graph_node = assignments[expression]
            tree_node = _remove_covariate_effect_from_statements_recursive(
                thetas,
                graph_node.previous,
                statements,
                expression,
                graph_node.expression,
                covariate,
                parent,
            )
            if tree_node.changed:
                statements[graph_node.index] = Assignment(expression, tree_node.expression)

            return ExpressionTreeNode(
                expression, tree_node.changed, tree_node.constant, tree_node.contains_theta
            )

        if expression == covariate:
            # NOTE expression is the covariate symbol for which we want to
            # remove all effects
            return ExpressionTreeNode(_neutral(parent), True, True, False)

        # NOTE other atom
        return ExpressionTreeNode(
            expression, False, _is_constant(thetas, expression), _depends_on_any(thetas, expression)
        )

    if isinstance(expression, sympy.Piecewise):
        if any(map(lambda t: covariate in t[1].free_symbols, expression.args)):
            # NOTE At least one condition depends on the covariate
            if all(
                map(
                    lambda t: _is_univariate(
                        thetas, _full_expression(assignments, t[1]), covariate
                    ),
                    expression.args,
                )
            ):
                # NOTE If expression is piecewise univariate and condition depends on
                # covariate, return simplest expression from cases
                expr = min(
                    (t[0] for t in expression.args),
                    key=sympy.count_ops,
                )
                tree_node = _remove_covariate_effect_from_statements_recursive(
                    thetas, assignments, statements, symbol, expr, covariate, parent
                )
                return ExpressionTreeNode(
                    tree_node.expression, True, tree_node.constant, tree_node.contains_theta
                )
            else:
                raise NotImplementedError(
                    'Cannot handle multivariate Piecewise where condition depends on covariate.'
                )

    children = list(
        map(
            lambda expr: _remove_covariate_effect_from_statements_recursive(
                thetas, assignments, statements, symbol, expr, covariate, expression
            ),
            expression.args,
        )
    )

    # TODO Take THETA limits into account. Currently we assume any
    # offset/factor can be compensated but this is not true in general.
    can_be_scaled_or_offset = any(map(lambda n: not n.changed and n.contains_theta, children))

    changed = any(map(lambda n: n.changed, children))
    is_constant = all(map(lambda n: n.constant, children))
    contains_theta = any(map(lambda n: n.contains_theta, children))

    if not changed:
        return ExpressionTreeNode(expression, False, is_constant, contains_theta)

    if not can_be_scaled_or_offset:
        return ExpressionTreeNode(
            expression.func(*map(lambda n: n.expression, children)),
            True,
            is_constant,
            contains_theta,
        )

    return ExpressionTreeNode(
        expression.func(
            *map(
                lambda n: _neutral(expression)
                if n.changed and n.constant and n.expression != symbol
                else n.expression,
                children,
            )
        ),
        True,
        is_constant,
        contains_theta,
    )


def get_pk_parameters(model: Model, kind: str = 'all') -> List[str]:
    """Retrieves PK parameters in :class:`pharmpy.model.Model`.

    Parameters
    ----------
    model : Model
        Pharmpy model to retrieve the PK parameters from
    kind : str
        The type of parameter to retrieve: 'absorption', 'distribution',
        'elimination', or 'all' (default).

    Return
    ------
    list[str]
        A list of the PK parameter names of the given model

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_pk_parameters(model)
    ['CL', 'V']
    >>> get_pk_parameters(model, 'absorption')
    []
    >>> get_pk_parameters(model, 'distribution')
    ['V']
    >>> get_pk_parameters(model, 'elimination')
    ['CL']

    See also
    --------
    get_individual_parameters
    get_rv_parameters

    """
    natural_assignments = _get_natural_assignments(model.statements.before_odes)
    cs_remapped = _remap_compartmental_system(model.statements, natural_assignments)

    free_symbols = set(_pk_free_symbols(cs_remapped, kind))

    dependency_graph = _dependency_graph(natural_assignments)

    pk_symbols = _filter_symbols(dependency_graph, free_symbols)
    if model.statements.ode_system.find_compartment("EFFECT") is not None:
        pd_symbols = model.statements.ode_system.find_compartment("EFFECT").input.free_symbols
        pk_symbols = pk_symbols.difference(pd_symbols)
    return sorted(map(str, pk_symbols))


def get_pd_parameters(model: Model) -> List[str]:
    """Retrieves PD parameters in :class:`pharmpy.model.Model`.

    Parameters
    ----------
    model : Model
        Pharmpy model to retrieve the PD parameters from

    Return
    ------
    list[str]
        A list of the PD parameter names of the given model

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_direct_effect(model, "linear")
    >>> get_pd_parameters(model)
    ['E0', 'S']

    See also
    --------
    get_pk_parameters

    """
    # FIXME: this function needs to be updated. Currently uses fixed parameter names.

    pd_symbols = []
    sset = model.statements
    if sympy.Symbol('Y_2') in model.dependent_variables:
        parameters_in_e = sset.find_assignment('E').free_symbols
        pk_parameters = get_pk_parameters(model)
        parameters_in_e = {x for x in parameters_in_e if str(x) not in pk_parameters}
        pd_symbols = {statement.symbol for statement in sset.before_odes}
        pd_symbols = pd_symbols.intersection(parameters_in_e)
        if sset.ode_system.find_compartment("EFFECT") is not None:
            pd_symbols.add("KE0")
    return sorted(map(str, pd_symbols))


def _get_natural_assignments(before_odes):
    # Return assignments where assignments that are constants (e.g. X=1),
    # single length expressions (e.g. S1=V), and divisions between parameters
    # (e.g. K=CL/V) have been filtered out
    classified_assignments = list(_classify_assignments(list(_assignments(before_odes))))
    natural_assignments = list(_remove_synthetic_assignments(classified_assignments))
    return natural_assignments


def _remap_compartmental_system(sset, natural_assignments):
    # Return compartmental system where rates that are synthetic assignments
    # have been substituted with their full definition (e.g K -> CL/V)
    cs = sset.ode_system

    assignments = list(_assignments(sset.before_odes))
    for assignment in reversed(assignments):
        # FIXME can be made more general, doesn't cover cases with recursively defined symbols (e.g. V=V/2)
        if assignment not in natural_assignments:
            # NOTE Substitution must be made in this order
            cs = cs.subs({assignment.symbol: assignment.expression})
    return cs


def _pk_free_symbols(cs: CompartmentalSystem, kind: str) -> Iterable[sympy.Symbol]:
    if kind == 'all':
        return cs.free_symbols

    if kind == 'absorption':
        return (
            []
            if cs.dosing_compartment[0] == cs.central_compartment
            else _pk_free_symbols_from_compartment(cs, cs.dosing_compartment[0])
        )

    if kind == 'distribution':
        return _pk_free_symbols_from_compartment(cs, cs.central_compartment)

    if kind == 'elimination':
        free_symbols = _pk_free_symbols_from_compartment(cs, output)
        return free_symbols

    raise ValueError(f'Cannot handle kind `{kind}`')


def _pk_free_symbols_from_compartment(
    cs: CompartmentalSystem, compartment: Compartment
) -> Iterable[sympy.Symbol]:
    vertices = _get_component(cs, compartment)
    edges = _get_component_edges(cs, vertices)
    is_central = compartment == cs.central_compartment
    return _get_component_free_symbols(is_central, vertices, edges)


def _get_component(cs: CompartmentalSystem, compartment: Compartment) -> Set[Compartment]:
    central_component_vertices = strongly_connected_component_of(
        cs.central_compartment,
        lambda u: map(lambda flow: flow[0], cs.get_compartment_outflows(u)),
        lambda u: map(lambda flow: flow[0], cs.get_compartment_inflows(u)),
    )

    if compartment == cs.central_compartment:
        return central_component_vertices

    flows = cs.get_compartment_inflows if compartment == output else cs.get_compartment_outflows

    return reachable_from(
        {compartment},
        lambda u: filterfalse(
            central_component_vertices.__contains__,
            map(lambda flow: flow[0], flows(u)),
        ),
    )


def _get_component_edges(cs: CompartmentalSystem, vertices: Set[Compartment]):
    return (
        ((u, v, rate) for v in vertices for u, rate in cs.get_compartment_inflows(v))
        if output in vertices
        else ((u, v, rate) for u in vertices for v, rate in cs.get_compartment_outflows(u))
    )


def _get_component_free_symbols(
    is_central: bool,
    vertices: Set[Compartment],
    edges: Iterable[Tuple[Compartment, Compartment, sympy.Expr]],
) -> Iterable[sympy.Symbol]:
    for u, v, rate in edges:
        # NOTE These must not necessarily be outgoing edges
        assert u in vertices or v in vertices

        if u not in vertices or v not in vertices:
            # NOTE This handles splitting the rate K = CL / V
            if len(rate.free_symbols) == 2:
                a, b = rate.free_symbols
                if rate == a / b:
                    yield a if v in vertices else b
                    continue
                elif rate == b / a:
                    yield b if v in vertices else a
                    continue

        if (u in vertices and v in vertices) or not is_central:
            # NOTE This handles all internal edges, and in/out rates (KA, CL/V)
            yield from rate.free_symbols

    for node in vertices:
        if node == output:
            yield from set()
        else:
            yield from node.free_symbols


def _assignments(sset: Statements):
    return filter(lambda statement: isinstance(statement, Assignment), sset)


def _filter_symbols(
    dependency_graph: Dict[sympy.Symbol, Set[sympy.Symbol]],
    roots: Set[sympy.Symbol],
    leaves: Union[Set[sympy.Symbol], None] = None,
) -> Set[sympy.Symbol]:
    dependents = graph_inverse(dependency_graph)

    free_symbols = reachable_from(roots, lambda x: dependency_graph.get(x, []))

    reachable = (
        free_symbols
        if leaves is None
        else (
            reachable_from(
                leaves,
                lambda x: dependents.get(x, []),
            ).intersection(free_symbols)
        )
    )

    return reachable.difference(dependents.keys()).intersection(dependency_graph.keys())


def _classify_assignments(assignments: Sequence[Assignment]):
    dependencies = _dependency_graph(assignments)

    # Keep all symbols that have dependencies (e.g. remove constants X=1)
    symbols = set(filter(dependencies.__getitem__, dependencies.keys()))

    for assignment in assignments:
        symbol = assignment.symbol
        expression = assignment.expression
        fs = expression.free_symbols

        if symbol not in fs:  # NOTE We skip redefinitions (e.g. CL=CL+1)
            if sympy.Symbol('t') in fs:  # FIXME: Should use ode.t here at some point
                yield 'synthetic', assignment
                continue
            elif len(fs) == 1:
                a = next(iter(fs))
                if a in symbols:
                    yield 'synthetic', assignment  # E.g. S1=V
                    continue
            elif len(fs) == 2:
                it = iter(fs)
                a = next(it)
                b = next(it)
                if (
                    a in symbols
                    and b in symbols
                    and (
                        expression == a / b
                        or expression == b / a
                        or expression == a * b
                        or expression == b * a
                    )
                ):
                    yield 'synthetic', assignment  # E.g. K=CL/V
                    continue

        yield 'natural', assignment


def _remove_synthetic_assignments(classified_assignments: List[Tuple[str, Assignment]]):
    assignments = []
    last_defined = {}

    for t, assignment in reversed(classified_assignments):
        if t == 'synthetic':
            substitution_starts_at_index = last_defined.get(assignment.symbol, 0)
            assignments = [
                succeeding
                if i < substitution_starts_at_index
                else Assignment(
                    succeeding.symbol,
                    subs(
                        succeeding.expression,
                        {assignment.symbol: assignment.expression},
                        simultaneous=True,
                    ),
                )
                for i, succeeding in enumerate(assignments)
            ]
        else:
            last_defined[assignment.symbol] = len(assignments)
            assignments.append(assignment)

    return reversed(assignments)


def _dependency_graph(assignments: Sequence[Assignment]):
    dependencies = {}

    for assignment in assignments:
        symbol = assignment.symbol
        fs = assignment.expression.free_symbols

        previous_def = dependencies.get(symbol)
        dependencies[symbol] = fs

        if previous_def is not None:
            # NOTE This handles redefinition of symbols by expanding
            # the previous definition of symbol into existing definitions
            for key, value in dependencies.items():
                if symbol in value:
                    dependencies[key] = (value - {symbol}) | previous_def

    return dependencies


def is_real(model: Model, expr: Union[str, sympy.Expr]) -> Optional[bool]:
    """Determine if an expression is real valued given constraints of a model

    Parameters
    ----------
    model : Model
        Pharmpy model
    expr : str or expression
        Expression to test

    Return
    ------
    bool or None
        True if expression is real, False if not and None if unknown

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, is_real
    >>> import sympy
    >>> model = load_example_model("pheno")
    >>> is_real(model, "CL")
    True

    """
    expr = sympy.sympify(expr)
    return sympy.ask(sympy.Q.real(expr), assume_all(sympy.Q.real, free_images_and_symbols(expr)))


def is_linearized(model: Model):
    """Determine if a model is linearized

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    bool
        True if model has been linearized and False otherwise

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, is_linearized
    >>> model1 = load_example_model("pheno")
    >>> is_linearized(model1)
    False
    >>> model2 = load_example_model("pheno_linear")
    >>> is_linearized(model2)
    True

    """
    statements = model.statements
    if statements.ode_system is not None:
        return False
    lhs = set()
    rhs = set()
    for s in statements:
        name = s.symbol.name
        if name == 'Y':  # To support linearized frem models
            break
        lhs.add(name)
        rhs_names = {symb.name for symb in s.expression.free_symbols}
        rhs.update(rhs_names)

    for name in lhs:
        m = re.match(r'BASE|BSUM|BASE_TERMS|IPRED|ERR|ESUM|ERROR_TERMS', name)
        if not m:
            return False

    for name in rhs:
        m = re.match(
            r'D_ETA|ETA|OETA|BASE|BSUM|BASE_TERMS|OPRED|EPS|D_EPS|D_EPSETA|ERR|ESUM|ERROR_TERMS|IPRED',
            name,
        )
        if not m:
            return False

    return True


def get_dv_symbol(model: Model, dv: Union[sympy.Symbol, str, int, None] = None) -> sympy.Symbol:
    """Get the symbol for a certain dvid or dv and check that it is valid

    Parameters
    ----------
    model : Model
        Pharmpy model
    dv : Union[sympy.Symbol, str, int]
        Either a dv symbol, str or dvid. If None (default) return the
        only or first dv.

    Return
    ------
    sympy.Symbol
        DV symbol

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, get_dv_symbol
    >>> model = load_example_model("pheno")
    >>> get_dv_symbol(model, "Y")
    Y
    >>> get_dv_symbol(model, 1)
    Y

    """
    if dv is None:
        dv = next(iter(model.dependent_variables))
    elif isinstance(dv, str):
        dv = sympy.Symbol(dv)
    elif isinstance(dv, int):
        for key, val in model.dependent_variables.items():
            if dv == val:
                dv = key
                break
        else:
            raise ValueError(f"DVID {dv} not defined in model")
    elif isinstance(dv, sympy.Symbol):
        pass
    else:
        raise TypeError(
            f"dv is of type {type(dv)} has to be one of Symbol or str representing "
            "a dv or int representing a dvid"
        )
    if dv not in model.dependent_variables:
        raise ValueError(f"DV {dv} not defined in model")
    return dv
