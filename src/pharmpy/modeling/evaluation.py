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
    model : Model
        Pharmpy model
    expression : str or sympy expression
        Expression to evaluate

    Returns
    -------
    pd.Series
        A series of one evaluated value for each data record

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, evaluate_expression
    >>> model = load_example_model("pheno")
    >>> evaluate_expression(model, "TVCL*1000")
    0      6.573770
    1      6.573770
    2      6.573770
    3      6.573770
    4      6.573770
             ...
    739    5.165105
    740    5.165105
    741    5.165105
    742    5.165105
    743    5.165105
    Length: 744, dtype: float64

    """
    expression = sympy.sympify(expression)
    full_expr = model.statements.before_odes.full_expression(expression)
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
