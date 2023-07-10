"""
:meta private:
"""
from typing import Optional

from pharmpy.deps import sympy
from pharmpy.internals.expr.funcs import PHI
from pharmpy.model import Assignment, EstimationSteps, JointNormalDistribution, Model

from .data import remove_loq_data
from .expressions import _simplify_expression_from_parameters, create_symbol

SUPPORTED_METHODS = frozenset(['m1', 'm3', 'm4'])


def transform_blq(model: Model, method: str = 'm4', lloq: Optional[float] = None):
    """Transform for BLQ data

    Transform a given model, methods available are m1, m3, and m4 [1]_. Current limits of the
    m3 and m4 method:

    * Does not support covariance between epsilons
    * Supports additive, proportional, combined, and power error model

    .. [1] Beal SL. Ways to fit a PK model with some data below the quantification
    limit. J Pharmacokinet Pharmacodyn. 2001 Oct;28(5):481-504. doi: 10.1023/a:1012299115260.
    Erratum in: J Pharmacokinet Pharmacodyn 2002 Jun;29(3):309. PMID: 11768292.

    Parameters
    ----------
    model : Model
        Pharmpy model
    method : str
        Which BLQ method to use
    lloq : float, optional
        LLOQ limit to use, if None Pharmpy will use the BLQ/LLOQ column in the dataset

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = transform_blq(model, method='m4', lloq=0.1)
    >>> model.statements.find_assignment("Y")
        ⎧ EPS₁⋅W + F   for DV ≥ LLOQ
        ⎪
        ⎨CUMD - CUMDZ
        ⎪────────────   otherwise
    Y = ⎩ 1 - CUMDZ

    See also
    --------
    remove_loq_data

    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f'Invalid `method`: got `{method}`,' f' must be one of {sorted(SUPPORTED_METHODS)}.'
        )
    if method == 'm1' and not isinstance(lloq, float):
        raise ValueError('Invalid type of `lloq` when combined with m1 method, must be float')

    if method == 'm1':
        model = _m1_method(model, lloq)
    if method in ('m3', 'm4'):
        _verify_model(model, method)
        model = _m3_m4_method(model, lloq, method)

    return model


def _m1_method(model, lloq):
    return remove_loq_data(model, lloq)


def _m3_m4_method(model, lloq, method):
    sset = model.statements

    est_steps = model.estimation_steps
    est_steps_new = EstimationSteps([est_step.replace(laplace=True) for est_step in est_steps])
    model = model.replace(estimation_steps=est_steps_new)

    # FIXME: handle other DVs?
    y_symb = list(model.dependent_variables.keys())[0]
    y = sset.find_assignment(y_symb)
    ipred = y.expression.subs({rv: 0 for rv in model.random_variables.epsilons.names})

    if isinstance(lloq, float):
        blq_symb = create_symbol(model, 'LLOQ')
        blq_type = 'lloq'
    else:
        blq_symb, blq_type = get_blq_symb_and_type(model)

    sd = _get_sd(model, y)
    symb_dv = sympy.Symbol(model.datainfo.dv_column.name)
    symb_fflag = create_symbol(model, 'F_FLAG')
    symb_cumd = create_symbol(model, 'CUMD')

    if blq_type == 'lloq':
        is_above_lloq = sympy.GreaterThan(symb_dv, blq_symb)
    else:
        is_above_lloq = sympy.Equality(blq_symb, 0)

    assignments = [sd]

    if isinstance(lloq, float):
        lloq = Assignment(blq_symb, sympy.Float(lloq))
        assignments.append(lloq)

    assignments += Assignment(symb_fflag, sympy.Piecewise((0, is_above_lloq), (1, True)))

    cumd = Assignment(symb_cumd, PHI((blq_symb - ipred) / sd.symbol))
    if method == 'm3':
        assignments += Assignment(
            y.symbol, sympy.Piecewise((y.expression, is_above_lloq), (cumd.expression, True))
        )
    else:
        assignments += cumd
        symb_cumdz = create_symbol(model, 'CUMDZ')
        assignments += Assignment(symb_cumdz, PHI(-ipred / sd.symbol))

        y_below_lloq = (symb_cumd - symb_cumdz) / (1 - symb_cumdz)
        assignments += Assignment(
            y.symbol, sympy.Piecewise((y.expression, is_above_lloq), (y_below_lloq, True))
        )

    y_idx = sset.find_assignment_index(y.symbol)
    sset_new = sset[:y_idx] + assignments + sset[y_idx + 1 :]
    model = model.replace(statements=sset_new)

    return model.update_source()


def has_blq_transformation(model: Model):
    # FIXME: make more general
    y = list(model.dependent_variables.keys())[0]
    y_expr = model.statements.error.find_assignment(y).expression
    if not isinstance(y_expr, sympy.Piecewise):
        return False
    for statement, cond in y_expr.args:
        blq_symb, _ = get_blq_symb_and_type(model)
        if blq_symb in cond.free_symbols:
            break
    else:
        return False

    expected_m3 = ['SD', 'F_FLAG']
    expected_m4 = ['SD', 'F_FLAG', 'CUMD', 'CUMDZ']
    return _has_all_expected_symbs(model.statements.error, expected_m3) or _has_all_expected_symbs(
        model.statements.error, expected_m4
    )


def get_blq_symb_and_type(model: Model):
    try:
        blq_datainfo = model.datainfo.typeix['lloq']
        return sympy.Symbol(blq_datainfo[0].name), 'lloq'
    except IndexError:
        try:
            blq_datainfo = model.datainfo.typeix['blqdv']
            return sympy.Symbol(blq_datainfo[0].name), 'blqdv'
        except IndexError:
            return sympy.Symbol('LLOQ'), 'lloq'


def _has_all_expected_symbs(sset, expected_symbs):
    symb_names = [s.symbol.name for s in sset]
    return all(symb in symb_names for symb in expected_symbs)


def _verify_model(model, method):
    rvs = model.random_variables.epsilons
    if any(isinstance(rv, JointNormalDistribution) for rv in rvs):
        raise ValueError(
            f'Invalid input model: covariance between epsilons not supported in `method` {method}'
        )


def _get_sd(model, y):
    y_expr = model.statements.find_assignment(y.symbol).expression
    sd_expr = get_sd_expr(y_expr, model.random_variables, model.parameters)
    symb_sd = create_symbol(model, 'SD')
    return Assignment(symb_sd, sd_expr)


def get_sd_expr(y_expr, rvs, params):
    rv_terms = [arg for arg in y_expr.args if arg.free_symbols.intersection(rvs.free_symbols)]
    sd_expr = []
    for i, term in enumerate(rv_terms, 1):
        rvs_in_term = rvs.free_symbols.intersection(term.free_symbols)
        if len(rvs_in_term) > 1:
            raise ValueError(
                'Invalid input model: error model not supported, terms in error model cannot contain '
                'more than one random variable'
            )
        expr = rvs.replace_with_sympy_rvs(term)
        sd_expr.append(sympy.stats.std(expr))

    return _simplify_expression_from_parameters(
        sympy.sqrt(sympy.Add(*[expr**2 for expr in sd_expr])), params
    )
