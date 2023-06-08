"""
:meta private:
"""

import re
import warnings
from collections import Counter, defaultdict
from functools import reduce
from itertools import chain, combinations
from operator import add, mul
from typing import List, Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.math import corr2cov, nearest_postive_semidefinite
from pharmpy.model import (
    Assignment,
    JointNormalDistribution,
    Model,
    NormalDistribution,
    Parameter,
    Parameters,
)

from .common import remove_unused_parameters_and_rvs
from .expressions import create_symbol, get_pk_parameters, has_random_effect
from .help_functions import _format_input_list, _format_options, _get_etas

ADD_IOV_DISTRIBUTION = frozenset(('disjoint', 'joint', 'explicit', 'same-as-iiv'))


def add_iiv(
    model: Model,
    list_of_parameters: Union[List[str], str],
    expression: Union[List[str], str],
    operation: str = '*',
    initial_estimate: float = 0.09,
    eta_names: Optional[List[str]] = None,
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
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_iiv(model, "CL")
    >>> model = add_iiv(model, "CL", "add")
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
        model.random_variables,
        list(model.parameters),
        model.statements,
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
        omega = sympy.Symbol(f'IIV_{list_of_parameters[i]}')
        if not eta_names:
            eta_name = f'ETA_{list_of_parameters[i]}'
        else:
            eta_name = eta_names[i]

        eta = NormalDistribution.create(eta_name, 'iiv', 0, omega)

        rvs = rvs + eta
        pset.append(Parameter(str(omega), init=initial_estimate))

        index = sset.find_assignment_index(list_of_parameters[i])

        if index is None:
            raise ValueError(f'Could not find parameter: {list_of_parameters[i]}')
        statement = sset[index]

        eta_addition = _create_template(expression[i], operation[i])
        eta_addition.apply(statement.expression, eta.names[0])

        sset = (
            sset[0:index] + Assignment(statement.symbol, eta_addition.template) + sset[index + 1 :]
        )

    model = model.replace(random_variables=rvs, parameters=Parameters.create(pset), statements=sset)

    return model.update_source()


def add_iov(
    model: Model,
    occ: str,
    list_of_parameters: Optional[Union[List[str], str]] = None,
    eta_names: Optional[Union[List[str], str]] = None,
    distribution: str = 'disjoint',
):
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
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_iov(model, "TIME", "CL")
    >>> model.statements.find_assignment("CL")  # doctest: +SKIP
    CL = ETA_CL + TVCL

    See also
    --------
    add_iiv
    add_pk_iiv
    remove_iiv
    remove_iov

    """

    if distribution not in ADD_IOV_DISTRIBUTION:
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

    categories = get_occasion_levels(model.dataset, occ)

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
        for dist in model.random_variables:
            intersection = list(filter(etas_set.__contains__, dist.names))

            if not intersection:
                continue

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

    model = model.replace(random_variables=rvs, parameters=Parameters.create(pset), statements=iovs)
    return model.update_source()


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
        model.random_variables,
        list(model.parameters),
        model.statements,
    )

    iovs, etais, sset = _add_iov_declare_etas(
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
        to_add = _add_iov_etas(
            rvs,
            pset,
            ordered_etas,
            dist,
            categories,
            omega_iov_name,
            eta_name,
        )
        rvs = rvs + list(to_add)

    return rvs, pset, iovs + etais + sset


def _add_iov_declare_etas(sset, occ, etas, indices, categories, eta_name, iov_name, etai_name):
    iovs, etais = [], []

    for i in indices:
        eta = etas[i - 1]
        # NOTE This declares IOV-ETA case assignments and replaces the existing
        # ETA with its sum with the new IOV ETA

        iov = sympy.Symbol(iov_name(i))

        expression = sympy.Piecewise(
            *(
                (sympy.Symbol(eta_name(i, k)), sympy.Eq(cat, sympy.Symbol(occ)))
                for k, cat in enumerate(categories, 1)
            )
        )

        iovs.append(Assignment(iov, parse_expr(0)))
        iovs.append(Assignment(iov, expression))

        etai = sympy.Symbol(etai_name(i))
        etais.append(Assignment(etai, sympy.Symbol(eta) + iov))
        sset = sset.subs({eta: etai})

    return iovs, etais, sset


def _add_iov_etas_disjoint(rvs, pset, etas, indices, categories, omega_iov_name, eta_name):
    _add_iov_declare_diagonal_omegas(rvs, pset, etas, indices, omega_iov_name)

    for i in indices:
        omega_iov = sympy.Symbol(omega_iov_name(i, i))
        for k in range(1, len(categories) + 1):
            yield NormalDistribution.create(eta_name(i, k), 'iov', 0, omega_iov)


def _add_iov_etas_joint(rvs, pset, etas, indices, categories, omega_iov_name, eta_name):
    _add_iov_declare_diagonal_omegas(rvs, pset, etas, indices, omega_iov_name)

    # NOTE Declare off-diagonal OMEGAs
    for i, j in combinations(indices, r=2):
        omega_iov = sympy.Symbol(omega_iov_name(i, j))
        omega_iiv = rvs.get_covariance(etas[i - 1], etas[j - 1])
        paramset = Parameters.create(pset)  # FIXME!
        init = paramset[omega_iiv].init * 0.1 if omega_iiv != 0 and omega_iiv in paramset else 0.001
        pset.append(Parameter(str(omega_iov), init=init))

    mu = [0] * len(indices)
    sigma = [[sympy.Symbol(omega_iov_name(min(i, j), max(i, j))) for i in indices] for j in indices]

    for k in range(1, len(categories) + 1):
        names = list(map(lambda i: eta_name(i, k), indices))
        yield JointNormalDistribution.create(names, 'iov', mu, sigma)


def _add_iov_declare_diagonal_omegas(rvs, pset, etas, indices, omega_iov_name):
    for i in indices:
        eta = etas[i - 1]
        omega_iiv = rvs[eta].get_variance(eta)
        omega_iov = sympy.Symbol(omega_iov_name(i, i))
        paramset = Parameters.create(pset)  # FIXME!
        init = paramset[omega_iiv].init * 0.1 if omega_iiv in paramset else 0.01
        pset.append(Parameter(str(omega_iov), init=init))


def add_pk_iiv(model: Model, initial_estimate: float = 0.09):
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
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_first_order_absorption(model)
    >>> model.statements.find_assignment("MAT")
    MAT = POP_MAT
    >>> model = add_pk_iiv(model)
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
    params_to_add_etas = [
        param for param in get_pk_parameters(model) if not has_random_effect(model, param, 'iiv')
    ]

    if params_to_add_etas:
        model = add_iiv(model, params_to_add_etas, 'exp', initial_estimate=initial_estimate)

    return model.update_source()


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
        expression = parse_expr(f'original {operation} {expression}')
        return EtaAddition(expression)


def _get_operation_func(operation):
    """Gets sympy operation based on string"""
    if operation == '*':
        return mul
    elif operation == '+':
        return add


def get_occasion_levels(df, occ):
    levels = df[occ].unique()
    return _canonicalize_categories(levels)


def _canonicalize_categories(categories):
    return sorted(map(_canonicalize_category, categories))


def _canonicalize_category(c: Union[int, float, str]):
    if isinstance(c, int):
        return c

    if isinstance(c, float):
        return int(c)

    if isinstance(c, (np.int32, np.int64)):
        return int(c)

    if isinstance(c, str):
        return _canonicalize_category(float(c) if '.' in c else int(c))

    raise ValueError(f'Cannot canonicalize category "{type(c)}({c})"')


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
        self.template = subs(self.template, {'original': original, 'eta_new': eta})

    @classmethod
    def additive(cls):
        template = sympy.Symbol('original') + sympy.Symbol('eta_new')

        return cls(template)

    @classmethod
    def proportional(cls):
        template = sympy.Symbol('original') * sympy.Symbol('eta_new')

        return cls(template)

    @classmethod
    def exponential(cls, operation):
        template = operation(sympy.Symbol('original'), sympy.exp(sympy.Symbol('eta_new')))

        return cls(template)

    @classmethod
    def logit(cls):
        template = sympy.Symbol('original') * (
            sympy.exp(sympy.Symbol('eta_new')) / (1 + sympy.exp(sympy.Symbol('eta_new')))
        )

        return cls(template)


def remove_iiv(model: Model, to_remove: Optional[Union[List[str], str]] = None):
    """
    Removes all IIV etas given a list with eta names and/or parameter names.

    Parameters
    ----------
    model : Model
        Pharmpy model to create block effect on.
    to_remove : str, list
        Name/names of etas and/or name/names of individual parameters to remove.
        If None, all etas that are IIVs will be removed. None is default.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_iiv(model)
    >>> model.statements.find_assignment("CL")
    CL = TVCL

    >>> model = load_example_model("pheno")
    >>> model = remove_iiv(model, "V")
    >>> model.statements.find_assignment("V")
    V = TVV

    See also
    --------
    remove_iov
    add_iiv
    add_iov
    add_pk_iiv

    """
    rvs, sset = model.random_variables, model.statements
    to_remove = _format_input_list(to_remove)
    etas = _get_etas(model, to_remove, include_symbols=True)

    for eta in etas:
        sset = sset.subs({sympy.Symbol(eta): 0})

    keep = [name for name in model.random_variables.names if name not in etas]
    model = model.replace(random_variables=rvs[keep], statements=sset)

    model = remove_unused_parameters_and_rvs(model)
    model = model.update_source()
    return model


def remove_iov(model: Model, to_remove: Optional[Union[List[str], str]] = None):
    """Removes all IOV etas given a list with eta names.

    Parameters
    ----------
    model : Model
        Pharmpy model to remove IOV from.
    to_remove : str, list
        Name/names of IOV etas to remove, e.g. 'ETA_IOV_1_1'.
        If None, all etas that are IOVs will be removed. None is default.

    Return
    ------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_iov(model)

    See also
    --------
    add_iiv
    add_iov
    remove_iiv
    add_pk_iiv

    """
    rvs, sset = model.random_variables, model.statements
    etas = _get_iov_etas(model, to_remove)
    if not etas:
        warnings.warn('No IOVs present')
        return model

    keep = [name for name in rvs.names if name not in etas]
    d = {sympy.Symbol(name): 0 for name in etas}

    model = model.replace(statements=sset.subs(d), random_variables=rvs[keep])
    model = remove_unused_parameters_and_rvs(model)
    return model.update_source()


def _get_iov_etas(model: Model, list_of_etas):
    list_of_etas = _format_input_list(list_of_etas)
    rvs = model.random_variables
    if list_of_etas is None:
        return set(rvs.iov.names)

    # NOTE Include all directly referenced ETAs
    direct_etas = set(list_of_etas)

    # NOTE Include all IOV ETAs that are identically distributed to the ones
    # directly referenced
    indirect_etas = set()
    for group in _get_iov_groups(model):
        if not direct_etas.isdisjoint(group):
            indirect_etas.update(group)

    return direct_etas | indirect_etas


def _get_iov_groups(model: Model):
    iovs = model.random_variables.iov
    same = defaultdict(set)
    for dist in iovs:
        for i, name in enumerate(dist.names):
            key = (i, dist.variance)
            same[key].add(name)

    return same.values()


def transform_etas_boxcox(model: Model, list_of_etas: Optional[Union[List[str], str]] = None):
    """Applies a boxcox transformation to selected etas

    Initial estimate for lambda is 0.1 with bounds (-3, 3).

    Parameters
    ----------
    model : Model
        Pharmpy model to apply boxcox transformation to.
    list_of_etas : str, list
        Name/names of etas to transform. If None, all etas will be transformed (default).

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = transform_etas_boxcox(model, ["ETA_1"])
    >>> model.statements.before_odes.full_expression("CL")
    PTVCL*WGT*exp((exp(ETA_1)**lambda1 - 1)/lambda1)

    See also
    --------
    transform_etas_tdist
    transform_etas_john_draper

    """
    list_of_etas = _format_input_list(list_of_etas)
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.boxcox(len(etas))
    model = _transform_etas(model, eta_transformation, etas)
    return model.update_source()


def transform_etas_tdist(model: Model, list_of_etas: Optional[Union[List[str], str]] = None):
    """Applies a t-distribution transformation to selected etas

    Initial estimate for degrees of freedom is 80 with bounds (3, 100).

    Parameters
    ----------
    model : Model
        Pharmpy model to apply t distribution transformation to.
    list_of_etas : str, list
        Name/names of etas to transform. If None, all etas will be transformed (default).

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = transform_etas_tdist(model, ["ETA_1"])
    >>> model.statements.before_odes.full_expression("CL")    # doctest: +ELLIPSIS
    PTVCL*WGT*exp(ETA_1*(1 + (ETA_1**2 + 1)/(4*df1) + (5*ETA_1**4 + 16*ETA_1**2 + 3)/(96*...

    See also
    --------
    transform_etas_boxcox
    transform_etas_john_draper

    """
    list_of_etas = _format_input_list(list_of_etas)
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.tdist(len(etas))
    model = _transform_etas(model, eta_transformation, etas)
    return model.update_source()


def transform_etas_john_draper(model: Model, list_of_etas: Optional[Union[List[str], str]] = None):
    """Applies a John Draper transformation [1]_ to spelected etas

    Initial estimate for lambda is 0.1 with bounds (-3, 3).

    .. [1] John, J., Draper, N. (1980). An Alternative Family of Transformations.
       Journal of the Royal Statistical Society. Series C (Applied Statistics),
       29(2), 190-197. doi:10.2307/2986305

    Parameters
    ----------
    model : Model
        Pharmpy model to apply John Draper transformation to.
    list_of_etas : str, list
        Name/names of etas to transform. If None, all etas will be transformed (default).

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = transform_etas_john_draper(model, ["ETA_1"])
    >>> model.statements.before_odes.full_expression("CL")
    PTVCL*WGT*exp(((Abs(ETA_1) + 1)**lambda1 - 1)*sign(ETA_1)/lambda1)

    See also
    --------
    transform_etas_boxcox
    transform_etas_tdist

    """
    list_of_etas = _format_input_list(list_of_etas)
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.john_draper(len(etas))
    model = _transform_etas(model, eta_transformation, etas)
    return model.update_source()


def _transform_etas(model, transformation, etas):
    etas_assignment, etas_subs = _create_new_etas(etas, transformation.name)
    parameters, thetas = _create_new_thetas(model, transformation.theta_type, len(etas))
    transformation.apply(etas_assignment, thetas)
    statements_new = transformation.assignments
    sset = model.statements.subs(etas_subs)
    model = model.replace(parameters=parameters, statements=statements_new + sset)
    return model


def _create_new_etas(etas_original, transformation):
    etas_subs = {}
    etas_assignment = {}
    if transformation == 'boxcox':
        eta_new = 'etab'
    elif transformation == 'tdist':
        eta_new = 'etat'
    elif transformation == 'johndraper':
        eta_new = 'etad'
    else:
        eta_new = 'etan'
    for i, eta in enumerate(etas_original, 1):
        etas_subs[sympy.Symbol(eta)] = sympy.Symbol(f'{eta_new.upper()}{i}')
        etas_assignment[sympy.Symbol(f'{eta_new}{i}')] = sympy.Symbol(f'{eta_new.upper()}{i}')
        etas_assignment[sympy.Symbol(f'eta{i}')] = sympy.Symbol(eta)

    return etas_assignment, etas_subs


def _create_new_thetas(model, transformation, no_of_thetas):
    pset = list(model.parameters)
    thetas = {}
    theta_name = str(create_symbol(model, stem=transformation, force_numbering=True))

    param_settings = (0.01, -3, 3) if transformation == 'lambda' else (80, 3, 100)

    if no_of_thetas == 1:
        pset.append(Parameter(theta_name, *param_settings))
        thetas['theta1'] = theta_name
    else:
        theta_no = int(re.findall(r'\d', theta_name)[0])

        for i in range(1, no_of_thetas + 1):
            pset.append(Parameter(theta_name, 0.01, -3, 3))
            thetas[f'theta{i}'] = theta_name
            theta_name = f'{transformation}{theta_no + i}'

    return Parameters.create(pset), thetas


class EtaTransformation:
    def __init__(self, name, assignments, theta_type):
        self.name = name
        self.assignments = assignments
        self.theta_type = theta_type

    def apply(self, etas, thetas):
        for i, assignment in enumerate(self.assignments):
            self.assignments[i] = assignment.subs(etas).subs(thetas)

    @classmethod
    def boxcox(cls, no_of_etas):
        assignments = []
        for i in range(1, no_of_etas + 1):
            symbol = sympy.Symbol(f'etab{i}')
            expression = (sympy.exp(sympy.Symbol(f'eta{i}')) ** sympy.Symbol(f'theta{i}') - 1) / (
                sympy.Symbol(f'theta{i}')
            )

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('boxcox', assignments, 'lambda')

    @classmethod
    def tdist(cls, no_of_etas):
        assignments = []
        for i in range(1, no_of_etas + 1):
            symbol = sympy.Symbol(f'etat{i}')

            eta = sympy.Symbol(f'eta{i}')
            theta = sympy.Symbol(f'theta{i}')

            num_1 = eta**2 + 1
            denom_1 = 4 * theta

            num_2 = (5 * eta**4) + (16 * eta**2 + 3)
            denom_2 = 96 * theta**2

            num_3 = (3 * eta**6) + (19 * eta**4) + (17 * eta**2) - 15
            denom_3 = 384 * theta**3

            expression = eta * (1 + (num_1 / denom_1) + (num_2 / denom_2) + (num_3 / denom_3))

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('tdist', assignments, 'df')

    @classmethod
    def john_draper(cls, no_of_etas):
        assignments = []
        for i in range(1, no_of_etas + 1):
            symbol = sympy.Symbol(f'etad{i}')

            eta = sympy.Symbol(f'eta{i}')
            theta = sympy.Symbol(f'theta{i}')

            expression = sympy.sign(eta) * (((abs(eta) + 1) ** theta - 1) / theta)

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('johndraper', assignments, 'lambda')

    def __str__(self):
        return str(self.assignments)


def create_joint_distribution(
    model: Model,
    rvs: Optional[List[str]] = None,
    individual_estimates: Optional[pd.DataFrame] = None,
):
    """
    Combines some or all etas into a joint distribution.

    The etas must be IIVs and cannot
    be fixed. Initial estimates for covariance between the etas is dependent on whether
    the model has results from a previous run. In that case, the correlation will
    be calculated from individual estimates, otherwise correlation will be set to 10%.

    Parameters
    ----------
    model : Model
        Pharmpy model
    rvs : list
        Sequence of etas or names of etas to combine. If None, all etas that are IIVs and
        non-fixed will be used (full block). None is default.
    individual_estimates : pd.DataFrame
        Optional individual estimates to use for calculation of initial estimates

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, create_joint_distribution
    >>> model = load_example_model("pheno")
    >>> model.random_variables.etas
    ETA₁ ~ N(0, IVCL)
    ETA₂ ~ N(0, IVV)
    >>> model = create_joint_distribution(model, ['ETA_1', 'ETA_2'])
    >>> model.random_variables.etas
    ⎡ETA₁⎤    ⎧⎡0⎤  ⎡    IVCL      IIV_CL_IIV_V⎤⎫
    ⎢    ⎥ ~ N⎪⎢ ⎥, ⎢                          ⎥⎪
    ⎣ETA₂⎦    ⎩⎣0⎦  ⎣IIV_CL_IIV_V      IVV     ⎦⎭

    See also
    --------
    split_joint_distribution : split etas into separate distributions
    """
    all_rvs = model.random_variables
    if rvs is None:
        rvs = []
        iiv_rvs = model.random_variables.iiv
        for rv in iiv_rvs:
            for name in rv.parameter_names:
                if model.parameters[name].fix:
                    break
            else:
                rvs.extend(rv.names)
    else:
        for name in rvs:
            if name in all_rvs and all_rvs[name].level == 'IOV':
                raise ValueError(
                    f'{name} describes IOV: Joining IOV random variables is currently not supported'
                )
    if len(rvs) == 1:
        raise ValueError('At least two random variables are needed')

    sset = model.statements
    paramnames = []
    for rv in rvs:
        parameter_names = '_'.join(
            [s.symbol.name for s in sset if sympy.Symbol(rv) in s.rhs_symbols]
        )
        paramnames.append(parameter_names)

    all_rvs, cov_to_params = all_rvs.join(
        rvs, name_template='IIV_{}_IIV_{}', param_names=paramnames
    )
    pset_new = model.parameters
    for cov_name, param_names in cov_to_params.items():
        parent1, parent2 = model.parameters[param_names[0]], model.parameters[param_names[1]]
        covariance_init = _choose_cov_param_init(
            model, individual_estimates, all_rvs, parent1, parent2
        )
        param_new = Parameter(cov_name, covariance_init)
        pset_new += param_new
    model = model.replace(parameters=Parameters.create(pset_new), random_variables=all_rvs)

    return model.update_source()


def split_joint_distribution(model: Model, rvs: Optional[Union[List[str], str]] = None):
    """
    Splits etas following a joint distribution into separate distributions.

    Parameters
    ----------
    model : Model
        Pharmpy model
    rvs : str, list
        Name/names of etas to separate. If None, all etas that are IIVs and
        non-fixed will become single. None is default.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = create_joint_distribution(model, ['ETA_1', 'ETA_2'])
    >>> model.random_variables.etas
    ⎡ETA₁⎤    ⎧⎡0⎤  ⎡    IVCL      IIV_CL_IIV_V⎤⎫
    ⎢    ⎥ ~ N⎪⎢ ⎥, ⎢                          ⎥⎪
    ⎣ETA₂⎦    ⎩⎣0⎦  ⎣IIV_CL_IIV_V      IVV     ⎦⎭
    >>> model = split_joint_distribution(model, ['ETA_1', 'ETA_2'])
    >>> model.random_variables.etas
    ETA₁ ~ N(0, IVCL)
    ETA₂ ~ N(0, IVV)

    See also
    --------
    create_joint_distribution : combine etas into a join distribution
    """
    all_rvs = model.random_variables
    names = _get_etas(model, rvs)

    new_rvs = all_rvs.unjoin(names)

    parameters_before = all_rvs.parameter_names
    parameters_after = new_rvs.parameter_names

    removed_parameters = set(parameters_before) - set(parameters_after)
    new_params = Parameters(
        tuple([p for p in model.parameters if p.name not in removed_parameters])
    )
    model = model.replace(random_variables=new_rvs, parameters=new_params).update_source()
    return model


def _choose_cov_param_init(model, individual_estimates, rvs, parent1, parent2):
    etas = []

    for name in rvs.names:
        if rvs[name].get_variance(name).name in (parent1.name, parent2.name):
            etas.append(name)

    sd = np.array([np.sqrt(parent1.init), np.sqrt(parent2.init)])
    init_default = round(0.1 * sd[0] * sd[1], 7)

    last_estimation_step = [est for est in model.estimation_steps if not est.evaluation][-1]
    if last_estimation_step.method == 'FO':
        return init_default
    elif individual_estimates is not None:
        try:
            ie = individual_estimates
            if not all(eta in ie.columns for eta in etas):
                return init_default
        except KeyError:
            return init_default
        # NOTE Use pd.corr() and not pd.cov(). SD is chosen from the final estimates, if cov is used
        # it will be calculated from the EBEs.
        eta_corr = ie[etas].corr()
        if eta_corr.isnull().values.any():
            warnings.warn(
                f'Correlation of individual estimates between {parent1.name} and '
                f'{parent2.name} is NaN, returning default initial estimate'
            )
            return init_default
        cov = corr2cov(eta_corr.to_numpy(), sd)
        cov[cov == 0] = 0.0001
        cov = nearest_postive_semidefinite(cov)
        init_cov = cov[1][0]
        return round(init_cov, 7)
    else:
        return init_default


def update_initial_individual_estimates(
    model: Model, individual_estimates: pd.Series, force: bool = True
):
    """Update initial individual estimates for a model

    Updates initial individual estimates for a model.

    Parameters
    ----------
    model : Model
        Pharmpy model to update initial estimates
    individual_estimates : pd.DataFrame
        Individual estimates to use
    force : bool
        Set to False to only update if the model had initial individual estimates before

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, update_initial_individual_estimates
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> ie = results.individual_estimates
    >>> model = update_initial_individual_estimates(model, ie)
    """
    if not force and model.initial_individual_estimates is None:
        return model

    model = model.replace(initial_individual_estimates=individual_estimates)
    return model.update_source()
