from operator import add, mul

import sympy
import sympy.stats as stats

from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel
from pharmpy.symbols import symbol as S


def add_etas(model, parameter, expression, operation='*'):
    """
    Adds etas to :class:`pharmpy.model`. Effects that currently have templates are:

    - Additive (*add*)
    - Proportional (*prop*)
    - Exponential (*exp*)
    - Logit (*logit*)

    For all except exponential the operation input is not needed. Otherwise user specified
    input is supported.

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

    statement.expression = eta_addition.template

    model.get_pred_pk_record().statements = sset

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
