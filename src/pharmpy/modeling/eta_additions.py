"""
:meta private:
"""

from collections import Counter
from functools import reduce
from itertools import chain, combinations
from operator import add, mul

import sympy
from sympy import Eq, Piecewise
from sympy.stats.crv_types import NormalDistribution
from sympy.stats.joint_rv_types import MultivariateNormalDistribution

from pharmpy.modeling.expressions import create_symbol
from pharmpy.modeling.help_functions import _format_input_list, _format_options, _get_etas
from pharmpy.parameter import Parameter, Parameters
from pharmpy.random_variables import RandomVariable
from pharmpy.statements import Assignment, ModelStatements, sympify
from pharmpy.symbols import symbol as S


def add_iiv(
    model, list_of_parameters, expression, operation='*', initial_estimate=0.09, eta_names=None
):
    """Adds IIVs to :class:`pharmpy.model`.

    Effects that currently have templates are:

    - Additive (*add*)
    - Proportional (*prop*)
    - Exponential (*exp*)
    - Logit (*log*)

    For all except exponential the operation input is not needed. Otherwise user specified
    input is supported. Initial estimates for new etas are 0.09.

    Parameters
    ----------
    model : Model
        Pharmpy model to add new IIVs to.
    list_of_parameters : str, list
        Name/names of parameter to add new IIVs to.
    expression : str, list
        Effect/effects on eta. Either abbreviated (see above) or custom.
    operation : str, list, optional
        Whether the new IIV should be added or multiplied (default).
    initial_estimate : float
        Value of initial estimate of parameter. Default is 0.09
    eta_names : str, list, optional
        Custom name/names of new eta

    Return
    ------
    Model
        Reference to the same model

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> remove_iiv(model, "CL") # doctest: +ELLIPSIS
    <...>
    >>> add_iiv(model, "CL", "add")  # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("CL")
    CL = ETA_CL + TVCL

    See also
    --------
    add_pk_iiv
    add_iov
    remove_iiv
    remove_iov

    """
    rvs, pset, sset = (
        model.random_variables.copy(),
        [p for p in model.parameters],
        model.statements.copy(),
    )

    list_of_parameters = _format_input_list(list_of_parameters)
    list_of_options = _format_options([expression, operation, eta_names], len(list_of_parameters))
    expression, operation, eta_names = list_of_options

    if all(eta_name is None for eta_name in eta_names):
        eta_names = None

    if not all(len(opt) == len(list_of_parameters) for opt in list_of_options if opt):
        raise ValueError(
            'The number of provided expressions and operations must either be equal '
            'to the number of parameters or equal to 1'
        )

    for i in range(len(list_of_parameters)):
        omega = S(f'IIV_{list_of_parameters[i]}')
        if not eta_names:
            eta_name = f'ETA_{list_of_parameters[i]}'
        else:
            eta_name = eta_names[i]

        eta = RandomVariable.normal(eta_name, 'iiv', 0, omega)

        rvs.append(eta)
        pset.append(Parameter(str(omega), init=initial_estimate))

        statement = sset.find_assignment(list_of_parameters[i])

        if statement is None:
            raise ValueError(f'Could not find parameter: {list_of_parameters[i]}')

        eta_addition = _create_template(expression[i], operation[i])
        eta_addition.apply(statement.expression, eta.name)

        statement.expression = eta_addition.template

    model.random_variables = rvs
    model.parameters = Parameters(pset)
    model.statements = sset

    return model


def add_iov(model, occ, list_of_parameters=None, eta_names=None, distribution='disjoint'):
    """Adds IOVs to :class:`pharmpy.model`.

    Initial estimate of new IOVs are 10% of the IIV eta it is based on.

    Parameters
    ----------
    model : Model
        Pharmpy model to add new IOVs to.
    occ : str
        Name of occasion column.
    list_of_parameters : str, list
        List of names of parameters and random variables. Accepts random variable names, parameter
        names, or a mix of both.
    eta_names : str, list
        Custom names of new etas. Must be equal to the number of input etas times the number of
        categories for occasion.
    distribution : str
        The distribution that should be used for the new etas. Options are
        'disjoint' for disjoint normal distributions, 'joint' for joint normal
        distribution, 'explicit' for an explicit mix of joint and disjoint
        distributions, and 'same-as-iiv' for copying the distribution of IIV etas.

    Return
    ------
    Model
        Reference to the same model

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> add_iov(model, "TIME", "CL")  # doctest: +SKIP
    <...>
    >>> model.statements.find_assignment("CL")  # doctest: +SKIP
    CL = ETA_CL + TVCL

    See also
    --------
    add_iiv
    add_pk_iiv
    remove_iiv
    remove_iov

    """

    if distribution not in ['disjoint', 'joint', 'explicit', 'same-as-iiv']:
        raise ValueError(f'"{distribution}" is not a valid value for distribution')

    list_of_parameters = _format_input_list(list_of_parameters)

    if distribution == 'explicit':
        if list_of_parameters is None or not all(
            map(lambda x: isinstance(x, list), list_of_parameters)
        ):
            raise ValueError(
                'distribution == "explicit" requires parameters to be given as lists of lists'
            )
    else:
        if list_of_parameters is not None and not all(
            map(lambda x: isinstance(x, str), list_of_parameters)
        ):
            raise ValueError(
                'distribution != "explicit" requires parameters to be given as lists of strings'
            )

    if list_of_parameters is None:
        if distribution == 'disjoint':
            etas = list(map(lambda x: [x], _get_etas(model, None, include_symbols=True)))
        else:
            etas = [_get_etas(model, None, include_symbols=True)]

    else:
        if distribution == 'disjoint':
            list_of_parameters = list(map(lambda x: [x], list_of_parameters))
        elif distribution == 'joint' or distribution == 'same-as-iiv':
            list_of_parameters = [list_of_parameters]

        if not all(
            map(
                lambda x: isinstance(x, list) and all(map(lambda y: isinstance(y, str), x)),
                list_of_parameters,
            )
        ):
            raise ValueError('not all parameters are strings')

        etas = [_get_etas(model, grp, include_symbols=True) for grp in list_of_parameters]

        for dist, grp in zip(etas, list_of_parameters):
            assert len(dist) <= len(grp)

    categories = _get_occ_levels(model.dataset, occ)

    if eta_names and len(eta_names) != sum(map(len, etas)) * len(categories):
        raise ValueError(
            'Number of given eta names is incorrect, '
            f'need {sum(map(len,etas)) * len(categories)} names.'
        )

    if len(categories) == 1:
        raise ValueError(f'Only one value in {occ} column.')

    # NOTE This declares the ETAS and their corresponding OMEGAs
    if distribution == 'same-as-iiv':
        # NOTE We filter existing IIV distributions for selected ETAs and then
        # let the explicit distribution logic handle the rest
        assert len(etas) == 1
        etas_set = set(etas[0])
        etas = []
        for variables, dist in model.random_variables.distributions():

            intersection = list(filter(etas_set.__contains__, variables))

            if not intersection:
                continue

            if len(variables) == 1:
                assert isinstance(dist, NormalDistribution)
            else:
                assert isinstance(dist, MultivariateNormalDistribution)

            etas.append(intersection)

    first_iov_name = create_symbol(model, 'IOV_', force_numbering=True).name
    first_iov_number = int(first_iov_name.split('_')[-1])

    def eta_name(i, k):
        return (
            eta_names[(i - 1) * len(categories) + k - 1] if eta_names else f'ETA_{iov_name(i)}_{k}'
        )

    def omega_iov_name(i, j):
        return f'OMEGA_{iov_name(i)}' if i == j else f'OMEGA_{iov_name(i)}_{j}'

    def iov_name(i):
        return f'IOV_{first_iov_number + i - 1}'

    def etai_name(i):
        return f'ETAI{first_iov_number + i - 1}'

    rvs, pset, iovs = _add_iov_explicit(
        model, occ, etas, categories, iov_name, etai_name, eta_name, omega_iov_name
    )

    model.random_variables, model.parameters, model.statements = rvs, Parameters(pset), iovs

    return model


def _add_iov_explicit(model, occ, etas, categories, iov_name, etai_name, eta_name, omega_iov_name):
    assert all(map(bool, etas))

    ordered_etas = list(chain.from_iterable(etas))

    eta, count = next(iter(Counter(ordered_etas).most_common()))

    if count >= 2:
        raise ValueError(f'{eta} was given twice.')

    distributions = [
        range(i, i + len(grp))
        for i, grp in zip(reduce(lambda acc, x: acc + [acc[-1] + x], map(len, etas), [1]), etas)
    ]

    rvs, pset, sset = (
        model.random_variables.copy(),
        [p for p in model.parameters],
        model.statements.copy(),
    )

    iovs, etais = _add_iov_declare_etas(
        sset,
        occ,
        ordered_etas,
        range(1, len(ordered_etas) + 1),
        categories,
        eta_name,
        iov_name,
        etai_name,
    )

    for dist in distributions:
        assert dist
        _add_iov_etas = _add_iov_etas_disjoint if len(dist) == 1 else _add_iov_etas_joint
        rvs.extend(
            _add_iov_etas(
                rvs,
                pset,
                ordered_etas,
                dist,
                categories,
                omega_iov_name,
                eta_name,
            )
        )

    iovs.extend(etais)
    iovs.extend(sset)

    return rvs, pset, iovs


def _add_iov_declare_etas(sset, occ, etas, indices, categories, eta_name, iov_name, etai_name):
    iovs, etais = ModelStatements(), ModelStatements()

    for i in indices:
        eta = etas[i - 1]
        # NOTE This declares IOV-ETA case assignments and replaces the existing
        # ETA with its sum with the new IOV ETA

        iov = S(iov_name(i))

        expression = Piecewise(
            *((S(eta_name(i, k)), Eq(cat, S(occ))) for k, cat in enumerate(categories, 1))
        )

        iovs.append(Assignment(iov, sympify(0)))
        iovs.append(Assignment(iov, expression))

        etai = S(etai_name(i))
        etais.append(Assignment(etai, eta.symbol + iov))
        sset.subs({eta.name: etai})

    return iovs, etais


def _add_iov_etas_disjoint(rvs, pset, etas, indices, categories, omega_iov_name, eta_name):

    _add_iov_declare_diagonal_omegas(rvs, pset, etas, indices, omega_iov_name)

    for i in indices:
        omega_iov = S(omega_iov_name(i, i))
        for k in range(1, len(categories) + 1):
            yield RandomVariable.normal(eta_name(i, k), 'iov', 0, omega_iov)


def _add_iov_etas_joint(rvs, pset, etas, indices, categories, omega_iov_name, eta_name):

    _add_iov_declare_diagonal_omegas(rvs, pset, etas, indices, omega_iov_name)

    # NOTE Declare off-diagonal OMEGAs
    for i, j in combinations(indices, r=2):
        omega_iov = S(omega_iov_name(i, j))
        omega_iiv = rvs.get_covariance(etas[i - 1], etas[j - 1])
        paramset = Parameters(pset)  # FIXME!
        init = paramset[omega_iiv].init * 0.1 if omega_iiv != 0 and omega_iiv in paramset else 0.001
        pset.append(Parameter(str(omega_iov), init=init))

    mu = [0] * len(indices)
    sigma = [[S(omega_iov_name(min(i, j), max(i, j))) for i in indices] for j in indices]

    for k in range(1, len(categories) + 1):
        names = list(map(lambda i: eta_name(i, k), indices))
        yield from RandomVariable.joint_normal(names, 'iov', mu, sigma)


def _add_iov_declare_diagonal_omegas(rvs, pset, etas, indices, omega_iov_name):
    for i in indices:
        eta = etas[i - 1]
        omega_iiv = rvs.get_variance(eta)
        omega_iov = S(omega_iov_name(i, i))
        paramset = Parameters(pset)  # FIXME!
        init = paramset[omega_iiv].init * 0.1 if omega_iiv in paramset else 0.01
        pset.append(Parameter(str(omega_iov), init=init))


def add_pk_iiv(model, initial_estimate=0.09):
    """Adds IIVs to all PK parameters in :class:`pharmpy.model`.

    Will add exponential IIVs to all parameters that are included in the ODE.

    Parameters
    ----------
    model : Model
        Pharmpy model to add new IIVs to.
    initial_estimate : float
        Value of initial estimate of parameter. Default is 0.09

    Return
    ------
    Model
        Reference to the same model

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_first_order_absorption(model) # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("MAT")
    MAT = POP_MAT
    >>> add_pk_iiv(model) # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("MAT")
                   ETA_MAT
    MAT = POP_MAT⋅ℯ

    See also
    --------
    add_iiv
    add_iov
    remove_iiv
    remove_iov

    """
    sset, rvs = model.statements, model.random_variables
    odes = sset.ode_system

    params_to_add_etas = []

    for param in odes.free_symbols:
        assign = sset.find_assignment(param)
        if assign:
            if _has_iiv(sset, rvs, assign):
                continue
            dep_assignments = _get_dependent_assignments(sset, assign)
            if dep_assignments:
                for dep_assign in dep_assignments:
                    param_name = dep_assign.symbol.name
                    if not _has_iiv(sset, rvs, dep_assign) and param_name not in params_to_add_etas:
                        params_to_add_etas.append(param_name)
            else:
                if param.name not in params_to_add_etas:
                    params_to_add_etas.append(param.name)

    if params_to_add_etas:
        add_iiv(model, params_to_add_etas, 'exp', initial_estimate=initial_estimate)

    return model


def _create_template(expression, operation):
    operation_func = _get_operation_func(operation)
    if expression == 'add':
        return EtaAddition.additive()
    elif expression == 'prop':
        return EtaAddition.proportional()
    elif expression == 'exp':
        return EtaAddition.exponential(operation_func)
    elif expression == 'log':
        return EtaAddition.logit()
    else:
        expression = sympify(f'original {operation} {expression}')
        return EtaAddition(expression)


def _get_operation_func(operation):
    """Gets sympy operation based on string"""
    if operation == '*':
        return mul
    elif operation == '+':
        return add


def _get_occ_levels(df, occ):
    levels = df[occ].unique()
    return _round_categories(levels)


def _round_categories(categories):
    categories_rounded = []
    for c in categories:
        if not isinstance(c, int) or c.is_integer():
            categories_rounded.append(int(c))
        else:
            categories_rounded.append(c)
    categories_rounded.sort()
    return categories_rounded


def _get_dependent_assignments(sset, assignment):
    # Finds dependant assignments one layer deep
    dep_assignments = [sset.find_assignment(symb) for symb in assignment.expression.free_symbols]
    return list(filter(None, dep_assignments))


def _has_iiv(sset, rvs, assignment):
    full_expression = sset.before_odes.full_expression(assignment.symbol)
    symb_names = {symb.name for symb in full_expression.free_symbols}
    if symb_names.intersection(rvs.iiv.names):
        return True
    return False


class EtaAddition:
    """
    Eta addition consisting of expression.

    Attributes
    ----------
    template
        Expression consisting of original statement +/* new eta with effect.

    :meta private:

    """

    def __init__(self, template):
        self.template = template

    def apply(self, original, eta):
        self.template = self.template.subs({'original': original, 'eta_new': eta})

    @classmethod
    def additive(cls):
        template = S('original') + S('eta_new')

        return cls(template)

    @classmethod
    def proportional(cls):
        template = S('original') * S('eta_new')

        return cls(template)

    @classmethod
    def exponential(cls, operation):
        template = operation(S('original'), sympy.exp(S('eta_new')))

        return cls(template)

    @classmethod
    def logit(cls):
        template = S('original') * (sympy.exp(S('eta_new')) / (1 + sympy.exp(S('eta_new'))))

        return cls(template)
