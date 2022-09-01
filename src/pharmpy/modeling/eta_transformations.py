"""
:meta private:
"""

import re

import sympy
from sympy import Symbol as S
from sympy import exp, sign

from pharmpy.modeling.help_functions import _format_input_list, _get_etas
from pharmpy.parameters import Parameter, Parameters
from pharmpy.statements import Assignment

from .expressions import create_symbol


def transform_etas_boxcox(model, list_of_etas=None):
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
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> transform_etas_boxcox(model, ["ETA(1)"])    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.before_odes.full_expression("CL")
    THETA(1)*WGT*exp((exp(ETA(1))**lambda1 - 1)/lambda1)

    See also
    --------
    transform_etas_tdist
    transform_etas_john_draper

    """
    list_of_etas = _format_input_list(list_of_etas)
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.boxcox(len(etas))
    _transform_etas(model, eta_transformation, etas)
    return model


def transform_etas_tdist(model, list_of_etas=None):
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
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> transform_etas_tdist(model, ["ETA(1)"])    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.before_odes.full_expression("CL")    # doctest: +ELLIPSIS
    THETA(1)*WGT*exp(ETA(1)*(1 + (ETA(1)**2 + 1)/(4*df1) + (5*ETA(1)**4 + 16*ETA(1)**2 + 3)/(96*...

    See also
    --------
    transform_etas_boxcox
    transform_etas_john_draper

    """
    list_of_etas = _format_input_list(list_of_etas)
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.tdist(len(etas))
    _transform_etas(model, eta_transformation, etas)
    return model


def transform_etas_john_draper(model, list_of_etas=None):
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
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> transform_etas_john_draper(model, ["ETA(1)"])    # doctest: +ELLIPSIS
    <...>
    >>> model.statements.before_odes.full_expression("CL")
    THETA(1)*WGT*exp(((Abs(ETA(1)) + 1)**lambda1 - 1)*sign(ETA(1))/lambda1)

    See also
    --------
    transform_etas_boxcox
    transform_etas_tdist

    """
    list_of_etas = _format_input_list(list_of_etas)
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.john_draper(len(etas))
    _transform_etas(model, eta_transformation, etas)
    return model


def _transform_etas(model, transformation, etas):
    etas_assignment, etas_subs = _create_new_etas(etas, transformation.name)
    thetas = _create_new_thetas(model, transformation.theta_type, len(etas))
    transformation.apply(etas_assignment, thetas)
    statements_new = transformation.assignments
    sset = model.statements
    sset = sset.subs(etas_subs)

    model.statements = statements_new + sset


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
        etas_subs[eta.symbol] = sympy.Symbol(f'{eta_new.upper()}{i}')
        etas_assignment[sympy.Symbol(f'{eta_new}{i}')] = sympy.Symbol(f'{eta_new.upper()}{i}')
        etas_assignment[sympy.Symbol(f'eta{i}')] = eta.symbol

    return etas_assignment, etas_subs


def _create_new_thetas(model, transformation, no_of_thetas):
    pset = [p for p in model.parameters]
    thetas = dict()
    theta_name = str(create_symbol(model, stem=transformation, force_numbering=True))

    if transformation == 'lambda':
        param_settings = [0.01, -3, 3]
    else:
        param_settings = [80, 3, 100]

    if no_of_thetas == 1:
        pset.append(Parameter(theta_name, *param_settings))
        thetas['theta1'] = theta_name
    else:
        theta_no = int(re.findall(r'\d', theta_name)[0])

        for i in range(1, no_of_thetas + 1):
            pset.append(Parameter(theta_name, 0.01, -3, 3))
            thetas[f'theta{i}'] = theta_name
            theta_name = f'{transformation}{theta_no + i}'

    model.parameters = Parameters(pset)

    return thetas


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
            symbol = S(f'etab{i}')
            expression = (exp(S(f'eta{i}')) ** S(f'theta{i}') - 1) / (S(f'theta{i}'))

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('boxcox', assignments, 'lambda')

    @classmethod
    def tdist(cls, no_of_etas):
        assignments = []
        for i in range(1, no_of_etas + 1):
            symbol = S(f'etat{i}')

            eta = S(f'eta{i}')
            theta = S(f'theta{i}')

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
            symbol = S(f'etad{i}')

            eta = S(f'eta{i}')
            theta = S(f'theta{i}')

            expression = sign(eta) * (((abs(eta) + 1) ** theta - 1) / theta)

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('johndraper', assignments, 'lambda')

    def __str__(self):
        return str(self.assignments)
