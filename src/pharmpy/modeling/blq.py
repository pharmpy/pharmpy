"""
:meta private:
"""

from __future__ import annotations

from typing import Literal, Optional

from pharmpy.basic import BooleanExpr, Expr
from pharmpy.deps import sympy, sympy_stats
from pharmpy.internals.expr.funcs import PHI
from pharmpy.internals.fn.type import check_list
from pharmpy.model import Assignment, ExecutionSteps, JointNormalDistribution, Model

from .data import remove_loq_data, set_lloq_data
from .expressions import _simplify_expression_from_parameters, create_symbol

SUPPORTED_METHODS = frozenset(['m1', 'm3', 'm4', 'm5', 'm6', 'm7'])


def transform_blq(
    model: Model,
    method: Literal['m1', 'm3', 'm4', 'm5', 'm6', 'm7'] = 'm4',
    lloq: Optional[float] = None,
):
    """Transform for BLQ data

    Transform a given model, methods available are m1, m3, m4, m5, m6 and m7 [1]_.
    The blq information can come from the dataset, the lloq option or a combination. Both LLOQ and BLQ
    columns are supported. The table below explains which columns are used for the various cases:

    +-------------+-------------+------------+-------------------+---------------+-------------------+
    | lloq option | LLOQ column | BLQ column | Used as indicator | Used as level | Note              |
    +=============+=============+============+===================+===============+===================+
    | Available   | NA          | NA         | DV < lloq         | lloq          |                   |
    +-------------+-------------+------------+-------------------+---------------+-------------------+
    | NA          | Available   | NA         | DV < LLOQ         | LLOQ          |                   |
    +-------------+-------------+------------+-------------------+---------------+-------------------+
    | NA          | NA          | Available  | BLQ               | nothing       | Only for M1 and M7|
    +-------------+-------------+------------+-------------------+---------------+-------------------+
    | NA          | NA          | NA         | NA                | NA            | No BLQ handling   |
    +-------------+-------------+------------+-------------------+---------------+-------------------+
    | NA          | Available   | Available  | BLQ               | LLOQ          | DV column not used|
    +-------------+-------------+------------+-------------------+---------------+-------------------+
    | Available   | NA          | Available  | BLQ               | lloq          |                   |
    +-------------+-------------+------------+-------------------+---------------+-------------------+
    | Available   | Available   | NA         | DV < lloq         | lloq          | Column overridden |
    +-------------+-------------+------------+-------------------+---------------+-------------------+
    | Available   | Available   | Available  | DV < lloq         | lloq          | Columns overridden|
    +-------------+-------------+------------+-------------------+---------------+-------------------+

    BLQ observations are defined as shown in the table above.
    If both a BLQ and an LLOQ column exist in the dataset and no lloq is specified then all dv values in
    rows with BLQ = 1 are counted as BLQ observations. If instead an lloq value is specified then all rows with
    dv values below the lloq value are counted as BLQ observations.
    If no lloq is specified and no BLQ column exists in the dataset then all rows with dv values below the value
    specified in the DV column are counted as BLQ observations.


    M1 method:
        All BLQ observations are discarded.
        This may affect the size of the dataset.
    M3 method:
        Including the probability that the BLQ observations are below the LLOQ
        as part of the maximum likelihood estimation.
        For more details see :ref:`[1]<ref_article>`.
        This method modifies the Y statement of the model (see examples below).
    M4 method:
        Including the probability that the BLQ observations are below the LLOQ and positive
        as part of the maximum likelihood estimation.
        For more details see :ref:`[1]<ref_article>`.
        This method modifies the Y statement of the model (see examples below).
    M5 method:
        All BLQ observations are replaced by level/2, where level = lloq if lloq is specified.
        Else level = value specified in LLOQ column (see table above).
        This method may change entries in the dataset.
    M6 method:
        Every BLQ observation in a consecutive series of BLQ observations is discarded except for the first one.
        The remaining BLQ observations are replaced by level/2, where level = lloq if lloq is specified.
        Else level = value specified in LLOQ column (see table above).
        This method may change entries in the dataset as well as the size of the dataset.
    M7 method:
        All BLQ observations are replaced by 0.
        This method may change entries in the dataset.



    Current limitations of the m3 and m4 method:

    * Does not support covariance between epsilons
    * Supports additive, proportional, combined, and power error model

    .. _ref_article:

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
        ⎧ EPS₁⋅F + F   for DV ≥ LLOQ
        ⎪
        ⎨CUMD - CUMDZ
        ⎪────────────    otherwise
    Y = ⎩ 1 - CUMDZ

    See also
    --------
    remove_loq_data
    set_lloq_data

    """
    lometh = method.lower()
    check_list("method", lometh, SUPPORTED_METHODS)

    blq_col, lloq_col = _get_blq_and_lloq_columns(model)
    indicator, indicator_type, level, level_type = _which_indicator_and_level(
        lloq, lloq_col, blq_col, lometh
    )

    if lometh == 'm1':
        model = _m1_method(model, indicator, indicator_type)
    elif lometh in ('m3', 'm4'):
        _verify_model(model, method)
        model = _m3_m4_method(model, indicator, indicator_type, level, level_type, lometh)
    elif lometh == 'm5':
        model = _m5_method(model, indicator, indicator_type, level, level_type)
    elif lometh == 'm6':
        model = _m6_method(model, indicator, indicator_type, level, level_type)
    elif lometh == 'm7':
        model = _m7_method(model, indicator, indicator_type)

    return model


def _m1_method(model, indicator, tp):
    if tp in ('lloq', 'LLOQ'):
        return remove_loq_data(model, lloq=indicator)
    else:  # tp == 'blq'
        return remove_loq_data(model, blq=indicator)


def _m5_method(model, indicator, tp, level, level_type):
    if level_type == 'LLOQ':
        half = f'{level}/2'
    else:
        half = level / 2
    if tp in ('lloq', 'LLOQ'):
        return set_lloq_data(model, half, lloq=indicator)
    else:  # tp == 'blq'
        return set_lloq_data(model, half, blq=indicator)


def _m6_method(model, indicator, tp, level, level_type):
    if level_type == 'LLOQ':
        half = f'{level}/2'
    else:
        half = level / 2
    if tp in ('lloq', 'LLOQ'):
        model = remove_loq_data(model, lloq=indicator, keep=1)
        return set_lloq_data(model, half, lloq=indicator)
    else:  # tp == 'blq'
        model = remove_loq_data(model, blq=indicator, keep=1)
        return set_lloq_data(model, half, blq=indicator)


def _m7_method(model, indicator, tp):
    if tp in ('lloq', 'LLOQ'):
        return set_lloq_data(model, 0, lloq=indicator)
    else:  # tp == 'blq':
        return set_lloq_data(model, 0, blq=indicator)


def _m3_m4_method(model, indicator, indicator_type, level, level_type, method):
    sset = model.statements

    est_steps = model.execution_steps
    est_steps_new = ExecutionSteps(tuple(est_step.replace(laplace=True) for est_step in est_steps))
    model = model.replace(execution_steps=est_steps_new)

    # FIXME: Handle other DVs?
    y_symb = list(model.dependent_variables.keys())[0]
    y = sset.find_assignment(y_symb)
    ipred = y.expression.subs({rv: 0 for rv in model.random_variables.epsilons.names})

    lloq_symbol = create_symbol(model, 'LLOQ')
    if indicator_type == 'lloq':
        indicator_symb = lloq_symbol
    else:
        indicator_symb = Expr.symbol(indicator)
    if level_type == 'lloq':
        level_symb = lloq_symbol
    else:
        level_symb = Expr.symbol(level)

    sd = _get_sd(model, y)
    symb_dv = Expr.symbol(model.datainfo.dv_column.name)
    symb_fflag = create_symbol(model, 'F_FLAG')
    symb_cumd = create_symbol(model, 'CUMD')

    if indicator_type in ('lloq', 'LLOQ'):
        is_above_lloq = BooleanExpr.ge(symb_dv, indicator_symb)
    else:
        is_above_lloq = BooleanExpr.eq(indicator_symb, 0)

    assignments = [sd]
    if indicator_type == 'lloq' or level_type == 'lloq':
        if indicator_type == 'lloq':
            symbol = indicator_symb
            value = indicator
        else:
            symbol = level_symb
            value = level
        lloq = Assignment(symbol, Expr.float(value))
        assignments.append(lloq)

    assignments += Assignment.create(symb_fflag, Expr.piecewise((0, is_above_lloq), (1, True)))

    cumd = Assignment.create(symb_cumd, PHI((level_symb - ipred) / sd.symbol))
    if method == 'm3':
        assignments += Assignment.create(
            y.symbol, Expr.piecewise((y.expression, is_above_lloq), (cumd.expression, True))
        )
    else:
        assignments += cumd
        symb_cumdz = create_symbol(model, 'CUMDZ')
        assignments += Assignment.create(symb_cumdz, PHI(-ipred / sd.symbol))

        y_below_lloq = (symb_cumd - symb_cumdz) / (1 - symb_cumdz)
        assignments += Assignment.create(
            y.symbol, Expr.piecewise((y.expression, is_above_lloq), (y_below_lloq, True))
        )

    y_idx = sset.find_assignment_index(y.symbol)
    sset_new = sset[:y_idx] + assignments + sset[y_idx + 1 :]
    model = model.replace(statements=sset_new, value_type=symb_fflag)

    return model.update_source()


def has_blq_transformation(model: Model):
    # FIXME: Make more general
    y_symb = list(model.dependent_variables.keys())[0]
    y = model.statements.error.find_assignment(y_symb)
    if not y:
        raise ValueError(f'Could not find assignment for \'{y_symb}\'')
    y_expr = y.expression
    if not y_expr.is_piecewise():
        return False
    for statement, cond in y_expr.args:  # pyright: ignore [reportGeneralTypeIssues]
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


def _get_blq_and_lloq_columns(model: Model):
    try:
        blq_datainfo = model.datainfo.typeix['blq']
    except IndexError:
        blq = None
    else:
        blq = blq_datainfo[0].name
    try:
        lloq_datainfo = model.datainfo.typeix['lloq']
    except IndexError:
        lloq = None
    else:
        lloq = lloq_datainfo[0].name
    return blq, lloq


def _which_indicator_and_level(lloq, lloq_col, blq_col, method):
    # Returns indicator, indicator type, level and level_type
    # indicator type can be 'lloq' for a value, 'LLOQ' for a column or 'blq'
    # level_type can be 'lloq' or 'LLOQ'
    if lloq is not None:
        if lloq_col is None and blq_col is not None:
            return blq_col, 'blq', lloq, 'lloq'
        else:
            return lloq, 'lloq', lloq, 'lloq'
    elif blq_col is not None:
        if lloq is None and lloq_col is None and method not in ('m1', 'm7'):
            raise ValueError(
                "Can only find a BLQ column. Only supported for the M1 and M7 methods."
            )
        return blq_col, 'blq', lloq_col, 'LLOQ'
    elif lloq_col is not None:
        return lloq_col, 'LLOQ', lloq_col, 'LLOQ'
    else:
        raise ValueError("No BLQ or LLOQ information available")


def _get_blq_name_and_type(model: Model):
    try:
        blq_datainfo = model.datainfo.typeix['lloq']
        return blq_datainfo[0].name, 'lloq'
    except IndexError:
        blq_datainfo = model.datainfo.typeix['blq']
        return blq_datainfo[0].name, 'blq'


def get_blq_symb_and_type(model: Model):
    try:
        name, tp = _get_blq_name_and_type(model)
        return Expr.symbol(name), tp
    except IndexError:
        return Expr.symbol('LLOQ'), 'lloq'


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
    return Assignment.create(symb_sd, sd_expr)


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
        sd_expr.append(sympy_stats.std(expr))

    return _simplify_expression_from_parameters(
        sympy.sqrt(sympy.Add(*[expr**2 for expr in sd_expr])), params
    )
