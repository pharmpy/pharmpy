from operator import add, mul

import sympy
import sympy.stats as stats

from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel
from pharmpy.statements import Assignment
from pharmpy.symbols import symbol as S


def add_etas(model, parameter, expression, operation='*'):
    """
    Adds etas to :class:`pharmpy.model`. Cuurently only exponential effect on eta is
    available as template, otherwise user specified input is supported.

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
    omega = S(f'IIV_{parameter}')
    eta = stats.Normal(f'ETA_{parameter}', 0, sympy.sqrt(omega))
    eta.variability_level = VariabilityLevel.IIV

    rvs = model.random_variables
    rvs.add(eta)
    model.random_variables = rvs

    params = model.parameters
    params.add(Parameter(str(omega), 0.1))
    model.parameters = params

    sset = model.get_pred_pk_record().statements
    statement = sset.find_assignment(parameter)

    eta_addition = _create_template(expression, operation)
    eta_addition.apply(statement.expression, eta)

    statement.expression = eta_addition.template.expression

    model.get_pred_pk_record().statements = sset


def _create_template(expression, operation):
    operation_func = _get_operation_func(operation)
    if expression == 'exp':
        return EtaAddition.exponential(operation_func)
    else:
        symbol = S('expression_new')
        expression = sympy.sympify(f'original {operation} {expression}')
        return EtaAddition(Assignment(symbol, expression))


def _get_operation_func(operation):
    """Gets sympy operation based on string"""
    if operation == '*':
        return mul
    elif operation == '+':
        return add


class EtaAddition:
    def __init__(self, template):
        self.template = template

    def apply(self, original, eta):
        self.template.subs({'original': original})
        self.template.subs({'eta_new': eta})

    @classmethod
    def exponential(cls, operation):
        expression = operation(S('original'), sympy.exp(S('eta_new')))

        template = Assignment(S('expression_new'), expression)

        return cls(template)
