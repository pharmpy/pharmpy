import sympy
import sympy.stats as stats

from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel
from pharmpy.symbols import symbol as S


def add_etas(model, parameter, expression, operation):
    rvs = model.random_variables

    omega = S('omega')
    new_eta = stats.Normal('eta', 0, sympy.sqrt(omega))
    new_eta.variability_level = VariabilityLevel.IIV

    rvs.add(new_eta)
    model.random_variables = rvs

    new_eta_param = Parameter(new_eta.name, 0.1)
    params = model.parameters
    params.add(new_eta_param)

    model.parameters = params

    sset = model.get_pred_pk_record().statements
    statement = sset.find_assignment(parameter)
    expression_new = statement.expression * S(new_eta.name)

    statement.expression = expression_new

    model.get_pred_pk_record().statements = sset
