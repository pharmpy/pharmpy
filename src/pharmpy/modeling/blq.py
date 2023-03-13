from typing import Optional

from pharmpy.deps import sympy
from pharmpy.internals.expr.funcs import PHI
from pharmpy.model import Assignment, EstimationSteps, Model
from pharmpy.modeling import (
    has_weighted_error_model,
    set_weighted_error_model,
    use_thetas_for_error_stdev,
)

from .error import get_weighted_error_model_weight
from .expressions import create_symbol


def transform_blq(model: Model, lloq: Optional[float] = None):
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
