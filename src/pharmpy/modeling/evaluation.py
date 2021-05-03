import numpy as np
import symengine
import sympy


def evaluate_expression(model, expression):
    """Evaluate expression using model

    Calculate the value of expression for each data record.
    The expression can contain dataset columns, variables in model and
    population parameters. If the model has parameter estimates these
    will be used. Initial estimates will be used for non-estimated parameters.

    Parameters
    ----------
    expression : str or sympy expression
        Expression to evaluate

    Returns
    -------
    pd.Series
        A series of one evaluated value for each data record
    """
    expression = sympy.sympify(expression)
    full_expr = model.statements.full_expression_from_odes(expression)
    pe = model.modelfit_results.parameter_estimates
    inits = model.parameters.inits
    expr = full_expr.subs(dict(pe)).subs(inits)
    data = model.dataset
    expr = symengine.sympify(expr)

    def func(row):
        subs = expr.subs(dict(row))
        return np.float64(subs.evalf())

    df = data.apply(func, axis=1)
    return df
