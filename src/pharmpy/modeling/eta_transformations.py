"""
:meta private:
"""

import re
import warnings

from sympy import exp, sign

from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.symbols import symbol as S


def boxcox(model, list_of_etas=None):
    """
    Applies a boxcox transformation to specified etas from a :class:`pharmpy.model`. Initial
    estimate for lambda is 0.1 with bounds (-3, 3).

    Parameters
    ----------
    model : Model
        Pharmpy model to apply boxcox transformation to.
    list_of_etas : list
        List of etas to transform. If None, all etas will be transformed (default).
    """
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.boxcox(len(etas))
    _transform_etas(model, eta_transformation, etas)
    return model


def tdist(model, list_of_etas=None):
    """
    Applies a t-distribution transformation to specified etas from a :class:`pharmpy.model`. Initial
    estimate for degrees of freedom is 80 with bounds (3, 100).

    Parameters
    ----------
    model : Model
        Pharmpy model to apply t distribution transformation to.
    list_of_etas : list
        List of etas to transform. If None, all etas will be transformed (default).
    """
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.tdist(len(etas))
    _transform_etas(model, eta_transformation, etas)
    return model


def john_draper(model, list_of_etas=None):
    """
    Applies a John Draper transformation [1]_ to specified etas from a
    :class:`pharmpy.model`. Initial estimate for lambda is 0.1 with bounds (-3, 3).

    .. [1] John, J., Draper, N. (1980). An Alternative Family of Transformations.
       Journal of the Royal Statistical Society. Series C (Applied Statistics),
       29(2), 190-197. doi:10.2307/2986305

    Parameters
    ----------
    model : Model
        Pharmpy model to apply John Draper transformation to.
    list_of_etas : list
        List of etas to transform. If None, all etas will be transformed (default).
    """
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.john_draper(len(etas))
    _transform_etas(model, eta_transformation, etas)
    return model


def _get_etas(model, list_of_etas):
    rvs = model.random_variables

    if list_of_etas is None:
        return rvs.etas
    else:
        etas = []
        for eta in list_of_etas:
            try:
                etas.append(rvs[eta.upper()])
            except KeyError:
                warnings.warn(f'Random variable "{eta}" does not exist')
        return etas


def _transform_etas(model, transformation, etas):
    etas_assignment, etas_subs = _create_new_etas(etas, transformation.name)
    thetas = _create_new_thetas(model, transformation.theta_type, len(etas))
    transformation.apply(etas_assignment, thetas)
    statements_new = transformation.assignments
    sset = model.statements
    sset.subs(etas_subs)

    statements_new.extend(sset)

    model.statements = statements_new


def _create_new_etas(etas_original, transformation):
    etas_subs = dict()
    etas_assignment = dict()
    if transformation == 'boxcox':
        eta_new = 'etab'
    elif transformation == 'tdist':
        eta_new = 'etat'
    elif transformation == 'johndraper':
        eta_new = 'etad'
    else:
        eta_new = 'etan'

    for i, eta in enumerate(etas_original, 1):
        etas_subs[eta.name] = f'{eta_new.upper()}{i}'
        etas_assignment[f'{eta_new}{i}'] = f'{eta_new.upper()}{i}'
        etas_assignment[f'eta{i}'] = eta.name

    return etas_assignment, etas_subs


def _create_new_thetas(model, transformation, no_of_thetas):
    pset = model.parameters
    thetas = dict()
    theta_name = str(model.create_symbol(stem=transformation, force_numbering=True))

    if transformation == 'lambda':
        param_settings = [0.01, -3, 3]
    else:
        param_settings = [80, 3, 100]

    if no_of_thetas == 1:
        pset.add(Parameter(theta_name, *param_settings))
        thetas['theta1'] = theta_name
    else:
        theta_no = int(re.findall(r'\d', theta_name)[0])

        for i in range(1, no_of_thetas + 1):
            pset.add(Parameter(theta_name, 0.01, -3, 3))
            thetas[f'theta{i}'] = theta_name
            theta_name = f'{transformation}{theta_no + i}'

    model.parameters = pset

    return thetas


class EtaTransformation:
    def __init__(self, name, assignments, theta_type):
        self.name = name
        self.assignments = assignments
        self.theta_type = theta_type

    def apply(self, etas, thetas):
        for assignment in self.assignments:
            assignment.subs(etas)
            assignment.subs(thetas)

    @classmethod
    def boxcox(cls, no_of_etas):
        assignments = ModelStatements()
        for i in range(1, no_of_etas + 1):
            symbol = S(f'etab{i}')
            expression = (exp(S(f'eta{i}')) ** S(f'theta{i}') - 1) / (S(f'theta{i}'))

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('boxcox', assignments, 'lambda')

    @classmethod
    def tdist(cls, no_of_etas):
        assignments = ModelStatements()
        for i in range(1, no_of_etas + 1):
            symbol = S(f'etat{i}')

            eta = S(f'eta{i}')
            theta = S(f'theta{i}')

            num_1 = eta ** 2 + 1
            denom_1 = 4 * theta

            num_2 = (5 * eta ** 4) + (16 * eta ** 2 + 3)
            denom_2 = 96 * theta ** 2

            num_3 = (3 * eta ** 6) + (19 * eta ** 4) + (17 * eta ** 2) - 15
            denom_3 = 384 * theta ** 3

            expression = eta * (1 + (num_1 / denom_1) + (num_2 / denom_2) + (num_3 / denom_3))

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('tdist', assignments, 'df')

    @classmethod
    def john_draper(cls, no_of_etas):
        assignments = ModelStatements()
        for i in range(1, no_of_etas + 1):
            symbol = S(f'etad{i}')

            eta = S(f'eta{i}')
            theta = S(f'theta{i}')

            expression = sign(eta) * (((abs(eta) + 1) ** theta - 1) / theta)

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('johndraper', assignments, 'lambda')

    def __str__(self):
        return str(self.assignments)
