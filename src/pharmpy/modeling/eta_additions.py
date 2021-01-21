"""
:meta private:
"""

from operator import add, mul

import sympy
import sympy.stats as stats
from sympy import Eq, Piecewise

from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.symbols import symbol as S


def add_iiv(model, parameter, expression, operation='*'):
    """
    Adds etas to :class:`pharmpy.model`. Effects that currently have templates are:

    - Additive (*add*)
    - Proportional (*prop*)
    - Exponential (*exp*)
    - Logit (*logit*)

    For all except exponential the operation input is not needed. Otherwise user specified
    input is supported. Initial estimates for new etas are 0.09.

    Parameters
    ----------
    model : Model
        Pharmpy model to add new etas to.
    parameter : str
        Name of parameter to add new etas to.
    expression : str
        Effect on eta. Either abbreviated (see above) or custom.
    operation : str, optional
        Whether the new eta should be added or multiplied (default).
    """
    rvs, pset, sset = model.random_variables, model.parameters, model.statements

    omega = S(f'IIV_{parameter}')
    eta = stats.Normal(f'ETA_{parameter}', 0, sympy.sqrt(omega))
    eta.variability_level = VariabilityLevel.IIV

    rvs.add(eta)
    pset.add(Parameter(str(omega), init=0.09))

    statement = sset.find_assignment(parameter)
    eta_addition = _create_template(expression, operation)
    eta_addition.apply(statement.expression, eta.name)

    statement.expression = eta_addition.template

    model.random_variables = rvs
    model.parameters = pset
    model.statements = sset

    return model


def add_iov(model, occ, list_of_etas=None):
    rvs, pset, sset = model.random_variables, model.parameters, model.statements
    etas = _get_etas(rvs, list_of_etas)
    iovs, etais = ModelStatements(), ModelStatements()

    categories = _get_occ_levels(model.dataset, occ)

    for i, eta in enumerate(etas, 1):
        omega_name = str(next(iter(eta.pspace.distribution.free_symbols)))
        omega = S(f'OMEGA_IOV_{i}')  # TODO: better name
        pset.add(Parameter(str(omega), init=pset[omega_name].init * 0.1))

        iov = S(f'IOV_{i}')

        values, conditions = [], []

        for j, cat in enumerate(categories, 1):
            eta_new = stats.Normal(f'ETA_IOV_{i}{j}', 0, sympy.sqrt(omega))
            eta_new.variability_level = VariabilityLevel.IOV

            rvs.add(eta_new)

            values += [S(eta_new.name)]
            conditions += [Eq(cat, S(occ))]

        expression = Piecewise(*zip(values, conditions))

        iovs.append(Assignment(iov, sympy.sympify(0)))
        iovs.append(Assignment(iov, expression))
        etais.append(Assignment(S(f'ETAI{i}'), eta + iov))

        sset.subs({eta.name: S(f'ETAI{i}')})

    iovs.extend(etais)
    iovs.extend(sset)

    model.random_variables, model.parameters, model.statements = rvs, pset, iovs

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
        expression = sympy.sympify(f'original {operation} {expression}')
        return EtaAddition(expression)


def _get_operation_func(operation):
    """Gets sympy operation based on string"""
    if operation == '*':
        return mul
    elif operation == '+':
        return add


def _get_etas(rvs, list_of_etas):
    if list_of_etas is None:
        return rvs.etas
    else:
        etas = []
        for eta in list_of_etas:
            etas.append(rvs[eta.upper()])
        return etas


def _get_occ_levels(df, occ):
    levels = df[occ].unique()
    return _round_categories(levels)


def _round_categories(categories):
    categories_rounded = []
    for c in categories:
        if c.is_integer():
            categories_rounded.append(int(c))
        else:
            categories_rounded.append(c)
    categories_rounded.sort()
    return categories_rounded


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
