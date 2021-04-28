import numpy as np
import sympy


def evaluate_expression(model, expression):
    expression = sympy.sympify(expression)
    full_expr = model.statements.full_expression_from_odes(expression)
    pe = model.modelfit_results.parameter_estimates
    inits = model.parameters.inits
    expr = full_expr.subs(dict(pe)).subs(inits)
    data = model.dataset

    def func(row):
        return np.float64(expr.evalf(subs=dict(row)))

    df = data.apply(func, axis=1)
    return df
