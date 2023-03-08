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


def transform_blq(model: Model, lloq: float):
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

    symb_dv = sympy.Symbol(model.datainfo.dv_column.name)
    symb_lloq = create_symbol(model, 'LLOQ')
    symb_fflag = create_symbol(model, 'F_FLAG')
    symb_cumd = create_symbol(model, 'CUMD')
    symb_cumdz = create_symbol(model, 'CUMDZ')

    is_above_lloq = sympy.GreaterThan(symb_dv, symb_lloq)

    fflag = Assignment(symb_fflag, sympy.Piecewise((0, is_above_lloq), (1, True)))
    cumd = Assignment(
        symb_cumd, sympy.Piecewise((0, is_above_lloq), (PHI((symb_lloq - ipred) / w), True))
    )
    cumdz = Assignment(symb_cumdz, sympy.Piecewise((0, is_above_lloq), (PHI(-ipred / w), True)))
    y_below_lloq = (symb_cumd - symb_cumdz) / (1 - symb_cumdz)
    y_new = Assignment(
        y.symbol, sympy.Piecewise((y.expression, is_above_lloq), (y_below_lloq, True))
    )
    lloq = Assignment(symb_lloq, sympy.Float(lloq))

    y_idx = sset.find_assignment_index(y.symbol)
    sset_new = sset[:y_idx] + [lloq, fflag, cumd, cumdz, y_new] + sset[y_idx + 1 :]
    model = model.replace(statements=sset_new)

    return model.update_source()
