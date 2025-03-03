from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import filterfalse
from typing import Iterable, Literal, Optional, Sequence, TypeVar, Union

from pharmpy.basic import Expr, TExpr, TSymbol
from pharmpy.deps import networkx as nx
from pharmpy.deps import sympy
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

from .parameters import get_omegas, get_sigmas, get_thetas, replace_fixed_thetas
from .random_variables import replace_non_random_rvs

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
    >>> model = load_example_model("pheno_linear")
    >>> expr = get_observation_expression(model)
    >>> print(expr.unicode())
    D_EPSETA1_2⋅EPS₁⋅(ETA₂ - OETA₂) + D_ETA1⋅(ETA₁ - OETA₁) + D_ETA2⋅(ETA₂ - OETA₂) + EPS₁⋅(D_EPS1 + D_EPSETA1_1⋅(ETA₁ - OETA₁)) + OPRED
    """  # noqa E501
    stats = model.statements
    # FIXME: Handle other DVs
    dv = list(model.dependent_variables.keys())[0]
    for i, s in enumerate(stats):
        if s.symbol == dv:
            y = s.expression
            break
    else:
        raise ValueError('Could not locate dependent variable expression')

    for j in range(i, -1, -1):
        expr = stats[j]
        assert isinstance(expr, Assignment)
        y = y.subs({expr.symbol: expr.expression})

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
    return get_observation_expression(model).subs(
        {Expr.symbol(eps): 0 for eps in model.random_variables.epsilons.names}
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

    return get_individual_prediction_expression(model).subs(
        {Expr.symbol(eta): 0 for eta in model.random_variables.etas.names}
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
    d = [y.diff(Expr.symbol(x)) for x in model.random_variables.etas.names]
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
    d = [y.diff(Expr.symbol(x)) for x in model.random_variables.epsilons.names]
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
        COVEFF → COVEFF1

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
        return Expr(str(stem))

    i = 1
    while True:
        candidate = f'{stem}{i}'
        if candidate not in all_names:
            return Expr(candidate)
        i += 1


def _find_eta_assignments(model):
    # NOTE: This locates all assignment to ETAs of symbols that do not depend on
    # any other ETA
    statements = model.statements.before_odes
    etas = {Expr.symbol(eta) for eta in model.random_variables.etas.names}
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
    TVCL = POP_CL⋅WGT
    TVV = POP_VC⋅WGT
          ⎧TVV⋅(COVAPGR + 1)  for APGR < 5
          ⎨
    TVV = ⎩       TVV          otherwise
    μ₁ = log(TVCL)
          ETA_CL + μ₁
    CL = ℯ
    μ₂ = log(TVV)
          ETA_VC + μ₂
    VC = ℯ
    V = VC
    S₁ = VC

    """
    if has_mu_reference(model):
        return model

    index = {Expr.symbol(eta): i for i, eta in enumerate(model.random_variables.etas.names, 1)}
    etas = set(index)

    offset = 0

    statements = model.statements
    for old_ind, assignment in _find_eta_assignments(model):
        # NOTE: The sequence of old_ind must be increasing
        eta = next(iter(etas.intersection(assignment.expression.free_symbols)))
        old_def = assignment.expression._sympy_()
        dep = old_def.as_independent(eta)[1]
        mu = Expr.symbol(f'mu_{index[eta]}')
        if mu in old_def.free_symbols:
            # If mu reference is already used, ignore
            pass
        elif old_ind == model.statements.find_assignment_index(assignment.symbol):
            new_def = subs(dep, {eta: mu + eta})
            mu_expr = sympy.solve(old_def - new_def, mu)[0]
            insertion_ind = offset + old_ind
            statements = (
                statements[0:insertion_ind]
                + Assignment.create(mu, mu_expr)
                + Assignment.create(assignment.symbol, new_def)
                + statements[insertion_ind + 1 :]
            )
            offset += 1  # NOTE: We need this offset because we replace one
            # statement by two statements
        else:
            # Parameter is manipulated 'after' adding IIV
            old_def = assignment.expression._sympy_()
            # Remove IIV from first definition
            remove_iiv_def = old_def.as_independent(eta)[0]
            insertion_ind = offset + old_ind
            statements = (
                statements[0:insertion_ind]
                + Assignment.create(assignment.symbol, remove_iiv_def)
                + statements[insertion_ind + 1 :]
            )
            # Add mu referencing to last definition of parameter instead
            last_ind = model.statements.find_assignment_index(assignment.symbol)
            last_assignment = model.statements.find_assignment(assignment.symbol)
            assert last_assignment is not None
            last_def = last_assignment.expression
            new_def = subs(dep, {eta: mu + eta})
            mu_expr = sympy.solve(subs(old_def, {remove_iiv_def: last_def}) - new_def, mu)[0]
            insertion_ind = offset + last_ind
            statements = (
                statements[0:insertion_ind]
                + Assignment.create(mu, mu_expr)
                + Assignment.create(assignment.symbol, new_def)
                + statements[insertion_ind + 1 :]
            )
            offset += 1  # NOTE: We need this offset because we replace one
            # statement by two statements

    model = model.replace(statements=statements).update_source()
    return model


def has_mu_reference(model: Model) -> bool:
    """Check if model is Mu-reference or not.

    Will return True if each parameter with an ETA is dependent on a Mu parameter.

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    bool
        Whether the model is mu referenced

    """
    ind_index_assignments = list(_find_eta_assignments(model))
    if not ind_index_assignments:
        return False
    ind_parameters = [a[1].symbol for a in ind_index_assignments]
    mu_regex = re.compile(r'^mu_\d*$', re.IGNORECASE)
    for ind_param in ind_parameters:
        ind_statement = model.statements.get_assignment(ind_param)
        if not any(re.match(mu_regex, str(p)) for p in ind_statement.free_symbols):
            return False

    return True


def get_mu_connected_to_parameter(model: Model, parameter: str) -> Optional[str]:
    """Return Mu name connected to parameter

    If the given parameter is not dependent on any Mu, None is returned

    Parameters
    ----------
    model : Model
        Pharmpy model object.
    parameter : str
        Name of parameter which to find Mu parameter for.

    Returns
    -------
    str
        Name of Mu parameter or None

    """
    mu_regex = r'^mu_\d*$'
    for p in model.statements.find_assignment(parameter).free_symbols:
        if match := re.match(mu_regex, str(p)):
            return match[0]
    return None


def simplify_expression(model: Model, expr: Union[str, TExpr]):
    """Simplify expression given constraints in model

    Parameters
    ----------
    model : Model
        Pharmpy model object
    expr : TExpr or str
        Expression to simplify

    Returns
    -------
    Expression
        Simplified expression

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, simplify_expression
    >>> model = load_example_model("pheno")
    >>> simplify_expression(model, "Abs(POP_CL)")
    POP_CL
    """
    return _simplify_expression_from_parameters(expr, model.parameters)


def _simplify_expression_from_parameters(expr, parameters) -> Expr:
    expr = sympy.sympify(expr)
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
    simp = sympy.simplify(expr).subs(d)  # Subs symbols back to non-constrained
    return Expr(simp)


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
    >>> model = make_declarative(model)
    >>> model.statements.before_odes
    TVCL = POP_CL⋅WGT
          ⎧POP_VC⋅WGT⋅(COVAPGR + 1)  for APGR < 5
          ⎨
    TVV = ⎩       POP_VC⋅WGT          otherwise
               ETA_CL
    CL = TVCL⋅ℯ
              ETA_VC
    VC = TVV⋅ℯ
    V = VC
    S₁ = VC

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
                    current[s.symbol] = s.expression.subs(current)
                else:
                    ass = Assignment.create(s.symbol, s.expression.subs(current))
                    newstats.append(ass)
                    del current[s.symbol]
        else:
            ass = Assignment.create(s.symbol, s.expression.subs(current))
            newstats.append(ass)

    model = model.replace(statements=Statements(newstats))
    return model.update_source()


def cleanup_model(model: Model):
    """Perform various cleanups of a model

    This is what is currently done

    * Make model statements declarative, i.e. only one assignment per symbol
    * Inline all assignments of one symbol, e.g. X = Y
    * Remove all random variables with no variability (i.e. with omegas fixed to zero)
    * Put fixed thetas directly in the model statements

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
        Updated model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model.statements
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
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL(t)
        ────────────
    F =      S₁
    Y = EPS₁⋅F + F
    >>> model = cleanup_model(model)
    >>> model.statements
    TVCL = POP_CL⋅WGT
          ⎧POP_VC⋅WGT⋅(COVAPGR + 1)  for APGR < 5
          ⎨
    TVV = ⎩       POP_VC⋅WGT          otherwise
               ETA_CL
    CL = TVCL⋅ℯ
              ETA_VC
    VC = TVV⋅ℯ
    V = VC
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL(t)
        ────────────
    F =      VC
    Y = EPS₁⋅F + F

    """
    model = make_declarative(model)

    current = {}
    newstats = []
    for s in model.statements:
        if (
            isinstance(s, Assignment)
            and s.expression.is_symbol()
            and not s.expression.is_function()
        ):
            current[s.symbol] = s.expression
        else:
            n = s.subs(current)
            newstats.append(n)

    model = model.replace(statements=Statements(newstats))
    model = replace_non_random_rvs(model)
    model = replace_fixed_thetas(model)
    return model


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
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL(t)
        ────────────
    F =      S₁
    Y = EPS₁⋅F + F

    >>> model = greekify_model(cleanup_model(model))
    >>> model.statements
    TVCL = WGT⋅θ₁
          ⎧WGT⋅θ₂⋅(θ₃ + 1)  for APGR < 5
          ⎨
    TVV = ⎩    WGT⋅θ₂        otherwise
               η₁
    CL = TVCL⋅ℯ
              η₂
    VC = TVV⋅ℯ
    V = VC
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
        A_CENTRAL(t)
        ────────────
    F =      VC
    Y = F⋅ε₁ + F

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
        subs[theta.symbol] = Expr.symbol(f"theta_{subscript}")
    omega = model.random_variables.covariance_matrix
    for row in range(omega.rows):
        for col in range(omega.cols):
            if col > row:
                break
            elt = omega[row, col]
            if elt == 0:
                continue
            subscript = get_2d_subscript(elt, row + 1, col + 1, named_subscripts)
            subs[elt] = Expr.symbol(f"omega_{subscript}")
    sigma = model.random_variables.covariance_matrix
    for row in range(sigma.rows):
        for col in range(sigma.cols):
            if col > row:
                break
            elt = sigma[row, col]
            if elt == 0:
                continue
            subscript = get_2d_subscript(elt, row + 1, col + 1, named_subscripts)
            subs[elt] = Expr.symbol(f"sigma_{subscript}")
    for i, eta in enumerate(model.random_variables.etas.names, start=1):
        subscript = get_subscript(eta, i, named_subscripts)
        subs[Expr.symbol(eta)] = Expr.symbol(f"eta_{subscript}")
    for i, epsilon in enumerate(model.random_variables.epsilons.names, start=1):
        subscript = get_subscript(epsilon, i, named_subscripts)
        subs[Expr.symbol(epsilon)] = Expr.symbol(f"epsilon_{subscript}")
    from pharmpy.modeling import rename_symbols

    model = rename_symbols(model, subs)
    return model


def get_individual_parameters(
    model: Model,
    level: Literal['iiv', 'iov', 'random', 'all'] = 'all',
    dv: Union[TSymbol, int, None] = None,
) -> list[str]:
    """Retrieves all individual parameters in a :class:`pharmpy.model`.

    By default all individual parameters will be found even ones having no random effect. The level
    arguments makes it possible to find only those having any random effect or only those having a certain
    random effect. Using the dv option will give all individual parameters affecting a certain dv. Note that
    the DV for PD in a PKPD model often also is affected by the PK parameters.

    Parameters
    ----------
    model : Model
        Pharmpy model to retrieve the individuals parameters from
    level : {'iiv', 'iov', 'random', 'all'}
        The variability level to look for: 'iiv', 'iov', 'random' or 'all' (default)
    dv : Union[Expr, str, int, None]
        Name or DVID of dependent variable. None for all (default)

    Return
    ------
    list[str]
        A list of the parameter names as strings

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_individual_parameters(model)
    ['CL', 'VC']
    >>> get_individual_parameters(model, 'iiv')
    ['CL', 'VC']
    >>> get_individual_parameters(model, 'iov')
    []

    See also
    --------
    get_pd_parameters
    get_pk_parameters
    get_rv_parameters
    has_random_effect

    """
    # FIXME: Support multiple DVs
    model = make_declarative(model)
    model = _replace_trivial_redefinitions(model)
    statements = model.statements

    # Arrow means is depending on
    full_graph = statements._create_dependency_graph()
    all_statements = set(full_graph.nodes)
    theta_eta_deps = _find_theta_eta_dependents(model, full_graph)
    eps_deps = _find_eps_dependents(model, full_graph)
    separable_statements = _find_separable_statements(model)
    ode_statement = _find_ode_statement(model)
    dv_statements = _find_dv_statements(model)
    time_deps = _find_time_dependents(model, full_graph)
    all_candidates = (
        all_statements.intersection(theta_eta_deps)
        - eps_deps
        - separable_statements
        - ode_statement
        - dv_statements
        - time_deps
    )

    parameter_symbs = set()
    if dv is None:
        dvs = model.dependent_variables.keys()
    else:
        dvs = {get_dv_symbol(model, dv)}

    for y in dvs:
        ind = statements.find_assignment_index(y)
        gsub = _subgraph_of(full_graph, ind)
        gsub = _cut_partial_odes(model, gsub, ind)
        candidates = set(gsub.nodes).intersection(all_candidates)

        parameter_indices = _find_individual_parameters(gsub, candidates)

        parameter_symbs |= {statements[i].symbol for i in parameter_indices}

    if level == 'random':
        wanted_levels = {'iov', 'iiv'}
    elif level == 'iiv':
        wanted_levels = {'iiv'}
    elif level == 'iov':
        wanted_levels = {'iov'}
    else:
        wanted_levels = None

    if wanted_levels is not None:
        filtered = set()
        for param in parameter_symbs:
            levels = _random_levels_of_parameter(model, full_graph, param)
            if not wanted_levels.isdisjoint(levels):
                filtered.add(param)
    else:
        filtered = parameter_symbs

    return sorted([s.name for s in filtered])


def _subgraph_of(g, ind):
    gsub = g.subgraph(nx.descendants(g, ind) | {ind})
    return gsub


def _cut_partial_odes(model, g, dv):
    # Cut not needed deps of the ode-system for this dv
    # This prevents the entire ode-system to be needed in case
    # of a disjoint system.

    odes = model.statements.ode_system

    if odes is None:
        return g

    def dep_funcs(eq):
        from sympy.core.function import AppliedUndef

        funcs = eq._sympy_().atoms(AppliedUndef)
        funcs = {Expr(func) for func in funcs}
        return funcs

    def dep_amounts(ode_deps, amounts):
        deps = set(amounts)
        for amount in amounts:
            deps |= set(nx.descendants(ode_deps, amount))
        return deps

    ode_deps = nx.DiGraph()
    for eq in odes.eqs:
        fn = dep_funcs(eq.lhs).pop()
        ode_deps.add_node(fn)

    for eq in odes.eqs:
        lhs_func = dep_funcs(eq.lhs).pop()
        rhs_funcs = dep_funcs(eq.rhs) - {lhs_func}
        for fn in rhs_funcs:
            ode_deps.add_edge(lhs_func, fn)

    all_dep_funcs = set()
    for i in g.nodes():
        s = model.statements[i]
        if isinstance(s, Assignment):
            all_dep_funcs |= dep_funcs(s.expression)

    deps = dep_amounts(ode_deps, all_dep_funcs)

    dep_symbs = set()
    nondep_symbs = set()

    for eq in odes.eqs:
        lhs_func = dep_funcs(eq.lhs).pop()
        if lhs_func in deps:
            dep_symbs |= eq.free_symbols
        else:
            nondep_symbs |= eq.free_symbols

    not_needed_symbs = nondep_symbs - dep_symbs

    for i, s in enumerate(model.statements):
        if isinstance(s, CompartmentalSystem):
            ode_index = i
            break
    else:
        assert False  # This should never happen

    g = g.copy()
    for i, s in enumerate(model.statements):
        if i in g and isinstance(s, Assignment) and s.symbol in not_needed_symbs:
            try:
                g.remove_edge(ode_index, i)
            except nx.NetworkXError:
                pass
            try:
                g.remove_edge(i, ode_index)
            except nx.NetworkXError:
                pass
    g = _subgraph_of(g, dv)  # Remove all nodes no longer connected
    return g


def _random_levels_of_parameter(model, g, param):
    ind = model.statements.find_assignment_index(param)
    iiv_symbs = model.random_variables.iiv.symbols
    iov_symbs = model.random_variables.iov.symbols
    levels = set()
    for node in nx.dfs_preorder_nodes(g, ind):
        if not model.statements[node].rhs_symbols.isdisjoint(iiv_symbs):
            levels.add('iiv')
        if not model.statements[node].rhs_symbols.isdisjoint(iov_symbs):
            levels.add('iov')
    return sorted(levels)


def _find_ode_statement(model):
    for i, s in enumerate(model.statements):
        if isinstance(s, CompartmentalSystem):
            return {i}
    return set()


def _find_dv_statements(model):
    inds = set()
    for i, s in enumerate(model.statements):
        if isinstance(s, Assignment) and s.symbol in model.dependent_variables.keys():
            inds.add(i)
    return inds


def _find_separable_statements(model):
    # Statements with only assignment dependencies
    # where these are dependencies of other assignments
    statements = model.statements
    all_assigned_symbols = _get_all_assigned_symbols(model)
    candidates = {
        i
        for i, s in enumerate(statements)
        if isinstance(s, Assignment) and s.rhs_symbols.issubset(all_assigned_symbols)
    }
    separable = set()
    for i in candidates:
        rhs = statements[i].rhs_symbols
        for j, s in enumerate(statements):
            if i != j and not rhs.isdisjoint(s.rhs_symbols):
                separable.add(i)
                break
    return separable


def _get_all_assigned_symbols(model):
    statements = model.statements
    all_assigned_symbols = {
        statement.symbol for statement in statements if isinstance(statement, Assignment)
    }
    return all_assigned_symbols


def _replace_trivial_redefinitions(model):
    # Remove and replace X = Y and X = 1/Y
    statements = model.statements
    all_assigned_symbols = _get_all_assigned_symbols(model)

    d = {}
    keep = []
    for s in statements:
        if isinstance(s, Assignment) and (
            s.expression in all_assigned_symbols or 1 / s.expression in all_assigned_symbols
        ):
            if s.expression in d.keys():
                d[s.symbol] = d[s.expression]
            else:
                d[s.symbol] = s.expression
        else:
            keep.append(s)

    new = Statements(tuple(keep)).subs(d)
    return model.replace(statements=new)


def _find_individual_parameters(g, candidates):
    # Starting from all sinks of g find all deepest candidate nodes
    keep = set()
    current_step = _sinks(g)

    while current_step:
        next_step = set()
        for node in current_step:
            pred = set(g.predecessors(node))
            if node in candidates:
                if pred.isdisjoint(candidates):
                    keep.add(node)
                else:
                    next_step.update(pred)
            else:
                next_step.update(pred)
        current_step = next_step
    return keep


def _sinks(g):
    return (node for node, out_degree in g.out_degree() if out_degree == 0)


def _find_time_dependents(model, g):
    direct = _find_direct_time_dependents(model)
    return _find_dependents(g, direct)


def _find_dependents(g, nodes):
    deps = nodes.copy()
    for i in nodes:
        ancestors = nx.ancestors(g, i)
        deps.update(ancestors)
    return deps


def _find_direct_param_dependents(model, g, eps=False):
    # Look for statements that are directly depending on
    # either (thetas, omegas or etas) or (sigmas or epsilons)
    deps = set()
    if not eps:
        thetas_and_omegas = get_thetas(model) + get_omegas(model)
        pop_params = set(thetas_and_omegas.symbols)
        rvs = set(model.random_variables.etas.symbols)
    else:
        sigmas = get_sigmas(model)
        pop_params = set(sigmas.symbols)
        rvs = set(model.random_variables.epsilons.symbols)
    all_symbs = pop_params | rvs
    for i in g.nodes:
        s = model.statements[i]
        if not all_symbs.isdisjoint(s.rhs_symbols):
            deps.add(i)
    return deps


def _find_theta_eta_dependents(model, g):
    direct = _find_direct_param_dependents(model, g)
    return _find_dependents(g, direct)


def _find_eps_dependents(model, g):
    direct = _find_direct_param_dependents(model, g, eps=True)
    return _find_dependents(g, direct)


def _find_direct_time_dependents(model):
    statements = model.statements
    odes = statements.ode_system
    if odes is None:
        return set()
    return {i for i, s in enumerate(statements) if odes.t in s.rhs_symbols}


def depends_on(model: Model, symbol: str, other: str):
    return _depends_on_any_of(
        model.statements.before_odes, Expr.symbol(symbol), [Expr.symbol(other)]
    )


def _depends_on_any_of(
    assignments: Statements, symbol: sympy.Symbol, symbols: Iterable[sympy.Symbol]
):
    dependency_graph = _dependency_graph(assignments)
    if symbol not in dependency_graph:
        raise KeyError(symbol)

    # NOTE: Could be faster by returning immediately once found
    return not reachable_from({symbol}, lambda x: dependency_graph.get(x, [])).isdisjoint(symbols)


def has_random_effect(
    model: Model, parameter: str, level: Literal['iiv', 'iov', 'all'] = 'all'
) -> bool:
    """Decides whether the given parameter of a :class:`pharmpy.model` has a
    random effect.

    Parameters
    ----------
    model : Model
        Input Pharmpy model
    parameter: str
        Input parameter
    level : {'iiv', 'iov', 'all'}
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
    symbol = Expr.symbol(parameter)
    return _depends_on_any_of(model.statements.before_odes, symbol, rvs.symbols)


def get_rv_parameters(model: Model, rv: str) -> list[str]:
    """Retrieves parameters in :class:`pharmpy.model.Model` given a random variable.

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
    >>> get_rv_parameters(model, 'ETA_CL')
    ['CL']

    See also
    --------
    has_random_effect
    get_pk_parameters
    get_individual_parameters

    """
    if rv not in model.random_variables.names:
        raise ValueError(f'Could not find random variable: {rv}')

    ind_param = get_individual_parameters(model, level="random")

    rv_parameters = []
    for param in ind_param:
        if Expr.symbol(rv) in model.statements.dependencies(param):
            rv_parameters.append(param)
    if not rv_parameters:
        raise ValueError(f"Cannot find any parameter connected to '{rv}'.")
    else:
        return sorted(map(str, rv_parameters))


def get_parameter_rv(
    model: Model, parameter: str, var_type: Literal['iiv', 'iov'] = 'iiv'
) -> list[str]:
    """Retrieves name of random variable in :class:`pharmpy.model.Model` given a parameter.

    Parameters
    ----------
    model : Model
        Pharmpy model to retrieve parameters from
    parameter : str
        Name of parameter to retrieve random variable from
    var_type: {'iiv', 'iov'}
        Variability type: iiv (default) or iov

    Return
    ------
    list[str]
        A list of random variable names for the given parameter

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_parameter_rv(model, 'CL')
    ['ETA_CL']

    See also
    --------
    get_rv_parameters
    has_random_effect
    get_pk_parameters
    get_individual_parameters

    """
    if parameter not in list(map(str, model.statements.free_symbols)):
        raise ValueError(f'Could not find parameter {parameter}')
    if parameter in model.random_variables.names:
        raise ValueError(f"{parameter} is a random variable. Only parameters are accepted as input")

    natural_assignments = _get_natural_assignments(model.statements.before_odes)

    rv = list(
        map(
            lambda rv_string: Expr.symbol(rv_string),
            getattr(model.random_variables, var_type).names,
        )
    )

    dependency_graph = graph_inverse(_dependency_graph(natural_assignments))
    return sorted(map(str, _filter_symbols(dependency_graph, rv, {Expr.symbol(parameter)})))


@dataclass(frozen=True)
class AssignmentGraphNode:
    expression: Expr
    index: int
    previous: dict[Expr, AssignmentGraphNode]


def _make_assignments_graph(statements: Statements) -> dict[sympy.Symbol, AssignmentGraphNode]:
    last_assignments: dict[Expr, AssignmentGraphNode] = {}

    for i, statement in enumerate(statements):
        if not isinstance(statement, Assignment):
            continue

        expr_as_sympy = sympy.sympify(statement.expression)
        node = AssignmentGraphNode(
            expr_as_sympy,
            i,
            {
                symbol: last_assignments[symbol]
                for symbol in expr_as_sympy.free_symbols
                if symbol in last_assignments
            },
        )

        last_assignments[sympy.sympify(statement.symbol)] = node

    return last_assignments


def remove_covariate_effect_from_statements(
    model: Model, before_odes: Statements, parameter: str, covariate: str
) -> Iterable[Statement]:
    assignments = _make_assignments_graph(before_odes)
    thetas = {sympy.sympify(symb) for symb in _theta_symbols(model)}

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
        new_before_odes[graph_node.index] = Assignment.create(
            Expr.symbol(parameter), tree_node.expression
        )

    return new_before_odes


def _neutral(expr: Expr) -> sympy.Integer:
    expr = Expr(expr)
    if expr.is_add():
        return sympy.Integer(0)
    if expr.is_mul():
        return sympy.Integer(1)
    if expr.is_pow():
        return sympy.Integer(1)

    raise ValueError(f'{type(expr)}: {repr(expr)} ({expr.free_symbols})')


def _theta_symbols(model: Model) -> set[sympy.Symbol]:
    rvs_fs = model.random_variables.free_symbols
    return {p.symbol for p in model.parameters if p.symbol not in rvs_fs}


def _depends_on_any(symbols: set[sympy.Symbol], expr: sympy.Expr) -> bool:
    return any(map(lambda s: s in symbols, expr.free_symbols))


def _is_constant(thetas: set[sympy.Symbol], expr: sympy.Expr) -> bool:
    return all(map(lambda s: s in thetas, expr.free_symbols))


def _is_univariate(thetas: set[sympy.Symbol], expr: sympy.Expr, variable: sympy.Symbol) -> bool:
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
    expression: Expr
    changed: bool
    constant: bool
    contains_theta: bool


def _full_expression(assignments: dict[sympy.Symbol, AssignmentGraphNode], expr: sympy.Expr):
    return expr.xreplace(
        {
            symbol: _full_expression(node.previous, node.expression)
            for symbol, node in assignments.items()
        }
    )


def _remove_covariate_effect_from_statements_recursive(
    thetas: set[Expr],
    assignments: dict[Expr, AssignmentGraphNode],
    statements: list[Assignment],
    symbol: Expr,
    expression: Expr,
    covariate: Expr,
    parent: Union[None, Expr],
) -> ExpressionTreeNode:
    if not expression.args:
        if expression in assignments:
            # NOTE: expression is a symbol and is defined in a previous assignment
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
                statements[graph_node.index] = Assignment.create(expression, tree_node.expression)

            return ExpressionTreeNode(
                expression, tree_node.changed, tree_node.constant, tree_node.contains_theta
            )

        if expression == covariate:
            # NOTE: expression is the covariate symbol for which we want to
            # remove all effects
            return ExpressionTreeNode(_neutral(parent), True, True, False)

        # NOTE: Other atom
        return ExpressionTreeNode(
            expression, False, _is_constant(thetas, expression), _depends_on_any(thetas, expression)
        )

    if isinstance(expression, sympy.Piecewise):
        if any(map(lambda t: covariate in t[1].free_symbols, expression.args)):
            # NOTE: At least one condition depends on the covariate
            if all(
                map(
                    lambda t: _is_univariate(
                        thetas, _full_expression(assignments, t[1]), covariate
                    ),
                    expression.args,
                )
            ):
                # NOTE: If expression is piecewise univariate and condition depends on
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

    # TODO: Take THETA limits into account. Currently we assume any
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
                lambda n: (
                    _neutral(expression)
                    if n.changed and n.constant and n.expression != symbol
                    else n.expression
                ),
                children,
            )
        ),
        True,
        is_constant,
        contains_theta,
    )


def get_pk_parameters(
    model: Model, kind: Literal['absorption', 'distribution', 'elimination', 'all'] = 'all'
) -> list[str]:
    """Retrieves PK parameters in :class:`pharmpy.model.Model`.

    Parameters
    ----------
    model : Model
        Pharmpy model to retrieve the PK parameters from
    kind : {'absorption', 'distribution', 'elimination', 'all'}
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
    ['CL', 'VC']
    >>> get_pk_parameters(model, 'absorption')
    []
    >>> get_pk_parameters(model, 'distribution')
    ['VC']
    >>> get_pk_parameters(model, 'elimination')
    ['CL']

    See also
    --------
    get_individual_parameters
    get_rv_parameters

    """
    pkparams = get_individual_parameters(model, dv=1)
    if len(model.dependent_variables) == 2:
        dmparams = _get_drug_metabolite_parameters(model, dv=2)
        if dmparams:
            pkparams = list(set(pkparams).union(dmparams))
    model = make_declarative(model)
    model = _replace_trivial_redefinitions(model)
    if kind != 'all':
        symbs = {str(symb) for symb in _pk_free_symbols(model, kind)}
        pkparams = set(pkparams).intersection(symbs)
    return sorted(pkparams)


def _get_drug_metabolite_parameters(model, dv=2):
    statements = model.statements
    ode = statements.ode_system

    dv1_deps = [
        Expr(s)
        for s in sympy.sympify(
            model.statements.after_odes.full_expression(get_dv_symbol(model, dv=1))
        ).atoms(sympy.Function)
    ]
    dv1_comp = None
    dv2_deps = [
        Expr(s)
        for s in sympy.sympify(
            model.statements.after_odes.full_expression(get_dv_symbol(model, dv=dv))
        ).atoms(sympy.Function)
    ]
    dv2_comp = None

    comp_names = ode.compartment_names
    for comp in [ode.find_compartment(name) for name in comp_names]:
        amounts = comp.amount
        if amounts in dv1_deps:
            dv1_comp = comp
        if amounts in dv2_deps:
            dv2_comp = comp

    if dv2_comp is not None:
        rate = ode.get_flow(dv1_comp, dv2_comp)
        if rate != 0:
            return get_individual_parameters(model, dv=2)
    else:
        None


def get_pd_parameters(model: Model) -> list[str]:
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
    ['B', 'SLOPE']

    See also
    --------
    get_pk_parameters

    """
    pkparams = get_individual_parameters(model, dv=1)
    pkpdparams = get_individual_parameters(model, dv=2)
    pdparams = sorted(set(pkpdparams) - set(pkparams))
    return pdparams


def _rvs(model: Model, level: str):
    if level == 'iiv':
        return model.random_variables.iiv
    if level == 'iov':
        return model.random_variables.iov
    if level == 'all':
        return model.random_variables.etas

    raise ValueError(f'Cannot handle level `{level}`')


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
        # FIXME: Can be made more general, doesn't cover cases with recursively defined symbols (e.g. V=V/2)
        if assignment not in natural_assignments:
            # NOTE: Substitution must be made in this order
            cs = cs.subs({assignment.symbol: assignment.expression})
    return cs


def _pk_free_symbols(model, kind: str) -> Iterable[sympy.Symbol]:
    cs = model.statements.ode_system
    symbols = set()
    if kind == 'absorption':
        central = cs.central_compartment
        for dosing in cs.dosing_compartments:
            comp = dosing
            while comp != central:
                symbols |= comp.free_symbols
                comp, rate = cs.get_compartment_outflows(comp)[0]  # Assumes only one flow
                symbols |= rate.free_symbols
    elif kind == 'distribution':
        central = cs.central_compartment
        periphs = cs.find_peripheral_compartments()
        for p in periphs:
            symbols |= cs.get_flow(central, p).free_symbols
            symbols |= cs.get_flow(p, central).free_symbols
        outflow = cs.get_flow(central, output)
        cl, v = outflow.as_numer_denom()
        if v != 1:
            symbols |= {v}
        else:
            ass = model.statements.find_assignment(cl)
            cl2, v2 = ass.expression.as_numer_denom()
            if v2 != 1:
                symbols |= {v2}
    elif kind == 'elimination':
        central = cs.central_compartment
        outflow = cs.get_flow(central, output)
        cl, v = outflow.as_numer_denom()
        if v != 1:
            symbols |= {cl}
        else:
            ass = model.statements.find_assignment(cl)
            cl2, v2 = ass.expression.as_numer_denom()
            if v2 != 1:
                symbols |= {cl2}
            else:
                symbols |= {cl}
    else:
        raise ValueError(f'Cannot handle kind `{kind}`')
    return symbols


def _pk_free_symbols_from_compartment(
    cs: CompartmentalSystem, compartment: Compartment
) -> Iterable[sympy.Symbol]:
    vertices = _get_component(cs, compartment)
    edges = _get_component_edges(cs, vertices)
    is_central = compartment == cs.central_compartment
    return _get_component_free_symbols(is_central, vertices, edges)


def _get_component(cs: CompartmentalSystem, compartment: Compartment) -> set[Compartment]:
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


def _get_component_edges(cs: CompartmentalSystem, vertices: set[Compartment]):
    return (
        ((u, v, rate) for v in vertices for u, rate in cs.get_compartment_inflows(v))
        if output in vertices
        else ((u, v, rate) for u in vertices for v, rate in cs.get_compartment_outflows(u))
    )


def _get_component_free_symbols(
    is_central: bool,
    vertices: set[Compartment],
    edges: Iterable[tuple[Compartment, Compartment, sympy.Expr]],
) -> Iterable[sympy.Symbol]:
    for u, v, rate in edges:
        # NOTE: These must not necessarily be outgoing edges
        assert u in vertices or v in vertices

        if u not in vertices or v not in vertices:
            # NOTE: This handles splitting the rate K = CL / V
            if len(rate.free_symbols) == 2:
                a, b = rate.free_symbols
                if rate == a / b:
                    yield a if v in vertices else b
                    continue
                elif rate == b / a:
                    yield b if v in vertices else a
                    continue

        if (u in vertices and v in vertices) or not is_central:
            # NOTE: This handles all internal edges, and in/out rates (KA, CL/V)
            yield from rate.free_symbols

    for node in vertices:
        if node == output:
            yield from set()
        else:
            yield from node.free_symbols


def _assignments(sset: Statements):
    return filter(lambda statement: isinstance(statement, Assignment), sset)


def _filter_symbols(
    dependency_graph: dict[sympy.Symbol, set[sympy.Symbol]],
    roots: set[sympy.Symbol],
    leaves: Union[set[sympy.Symbol], None] = None,
) -> set[sympy.Symbol]:
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

        if symbol not in fs:  # NOTE: We skip redefinitions (e.g. CL=CL+1)
            if Expr.symbol('t') in fs:  # FIXME: Should use ode.t here at some point
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


def _remove_synthetic_assignments(classified_assignments: list[tuple[str, Assignment]]):
    assignments = []
    last_defined = {}

    for t, assignment in reversed(classified_assignments):
        if t == 'synthetic':
            substitution_starts_at_index = last_defined.get(assignment.symbol, 0)
            assignments = [
                (
                    succeeding
                    if i < substitution_starts_at_index
                    else Assignment.create(
                        succeeding.symbol,
                        succeeding.expression.subs({assignment.symbol: assignment.expression}),
                    )
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
            # NOTE: This handles redefinition of symbols by expanding
            # the previous definition of symbol into existing definitions
            for key, value in dependencies.items():
                if symbol in value:
                    dependencies[key] = (value - {symbol}) | previous_def

    return dependencies


def is_real(model: Model, expr: TExpr) -> Optional[bool]:
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
    return Expr(expr).is_real()


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


def get_dv_symbol(model: Model, dv: Union[Expr, str, int, None] = None) -> Expr:
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
    Expr
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
    elif isinstance(dv, int):
        for key, val in model.dependent_variables.items():
            if dv == val:
                dv = key
                break
        else:
            raise ValueError(f"DVID {dv} not defined in model")
    else:
        try:
            dv = Expr(dv)
        except Exception:
            raise TypeError(
                f"dv is of type {type(dv)} has to be one of Symbol or str representing "
                "a dv or int representing a dvid"
            )
    if dv not in model.dependent_variables:
        raise ValueError(f"DV {dv} not defined in model")
    return dv
