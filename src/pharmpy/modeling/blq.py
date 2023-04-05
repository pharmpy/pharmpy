from typing import Optional

from pharmpy.deps import sympy
from pharmpy.internals.expr.funcs import PHI
from pharmpy.model import Assignment, EstimationSteps, Model

from .data import remove_loq_data
from .error import (
    get_weighted_error_model_weight,
    has_weighted_error_model,
    set_weighted_error_model,
    use_thetas_for_error_stdev,
)
from .expressions import create_symbol

SUPPORTED_METHODS = frozenset(['m1', 'm4'])


def transform_blq(model: Model, lloq: Optional[float] = None, method: str = 'm4'):
    """Transform for BLQ data

    Transform a given model, methods available are m1 and m4 [1]_.

    .. [1] Beal SL. Ways to fit a PK model with some data below the quantification
    limit. J Pharmacokinet Pharmacodyn. 2001 Oct;28(5):481-504. doi: 10.1023/a:1012299115260.
    Erratum in: J Pharmacokinet Pharmacodyn 2002 Jun;29(3):309. PMID: 11768292.

    Parameters
    ----------
    model : Model
        Pharmpy model
    lloq : float, optional
        LLOQ limit to use, if None Pharmpy will use the BLQ/LLOQ column in the dataset
    method : str
        Which BLQ method to use

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = transform_blq(model, method='m4')
    >>> model.statements.find_assignment("Y")
        ⎧ EPS₁⋅W + F   for BLQ = 1
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
    if method == 'm4':
        model = _m4_method(model, lloq)

    return model


def _m1_method(model, lloq):
    return remove_loq_data(model, lloq)


def _m4_method(model, lloq):
    if not has_weighted_error_model(model):
        model = set_weighted_error_model(model)
    model = use_thetas_for_error_stdev(model)

    sset = model.statements

    est_steps = model.estimation_steps
    est_steps_new = EstimationSteps([est_step.replace(laplace=True) for est_step in est_steps])
    model = model.replace(estimation_steps=est_steps_new)

    # FIXME: handle other DVs?
    y_symb = list(model.dependent_variables.keys())[0]
    y = sset.find_assignment(y_symb)
    ipred = y.expression.subs({rv: 0 for rv in model.random_variables.epsilons.names})
    w = get_weighted_error_model_weight(model)
    assert w is not None

    if isinstance(lloq, float):
        symb_lloq = create_symbol(model, 'LLOQ')
        lloq_type = 'lloq'
    else:
        try:
            lloq_datainfo = model.datainfo.typeix['blq']
            lloq_type = 'blq'
        except IndexError:
            lloq_datainfo = model.datainfo.typeix['lloq']
            lloq_type = 'lloq'
        if len(lloq_datainfo.names) > 1:
            raise ValueError(f'Can only have one of column type: {lloq_datainfo}')
        symb_lloq = sympy.Symbol(lloq_datainfo[0].name)

    symb_dv = sympy.Symbol(model.datainfo.dv_column.name)
    symb_fflag = create_symbol(model, 'F_FLAG')
    symb_cumd = create_symbol(model, 'CUMD')
    symb_cumdz = create_symbol(model, 'CUMDZ')

    if lloq_type == 'lloq':
        is_above_lloq = sympy.GreaterThan(symb_dv, symb_lloq)
    else:
        is_above_lloq = sympy.Equality(symb_lloq, 1)

    assignments = []

    if isinstance(lloq, float):
        lloq = Assignment(symb_lloq, sympy.Float(lloq))
        assignments.append(lloq)

    cumd = Assignment(symb_cumd, PHI((symb_lloq - ipred) / w))
    cumdz = Assignment(symb_cumdz, PHI(-ipred / w))
    fflag = Assignment(symb_fflag, sympy.Piecewise((0, is_above_lloq), (1, True)))
    y_below_lloq = (symb_cumd - symb_cumdz) / (1 - symb_cumdz)
    y_new = Assignment(
        y.symbol, sympy.Piecewise((y.expression, is_above_lloq), (y_below_lloq, True))
    )

    assignments.extend([cumd, cumdz, fflag, y_new])

    y_idx = sset.find_assignment_index(y.symbol)
    sset_new = sset[:y_idx] + assignments + sset[y_idx + 1 :]
    model = model.replace(statements=sset_new)

    return model.update_source()
