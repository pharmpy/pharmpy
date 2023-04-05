from typing import Optional

from pharmpy.deps import sympy
from pharmpy.internals.expr.funcs import PHI
from pharmpy.model import Assignment, EstimationSteps, JointNormalDistribution, Model

from .data import remove_loq_data
from .error import has_additive_error_model, has_combined_error_model, has_proportional_error_model
from .expressions import create_symbol

SUPPORTED_METHODS = frozenset(['m1', 'm3', 'm4'])


def transform_blq(model: Model, method: str = 'm4', lloq: Optional[float] = None):
    """Transform for BLQ data

    Transform a given model, methods available are m1, m3, and m4 [1]_. Current limits of the
    m3 and m4 method:

    * Does not support covariance between epsilons
    * Only supports additive, proportional, and combined error model

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

    sd_assignments, symb_sd = _weight_as_sd(model, y, ipred)
    symb_dv = sympy.Symbol(model.datainfo.dv_column.name)
    symb_fflag = create_symbol(model, 'F_FLAG')
    symb_cumd = create_symbol(model, 'CUMD')

    if lloq_type == 'lloq':
        is_above_lloq = sympy.GreaterThan(symb_dv, symb_lloq)
    else:
        is_above_lloq = sympy.Equality(symb_lloq, 1)

    assignments = sd_assignments

    if isinstance(lloq, float):
        lloq = Assignment(symb_lloq, sympy.Float(lloq))
        assignments.append(lloq)

    assignments += Assignment(symb_fflag, sympy.Piecewise((0, is_above_lloq), (1, True)))

    cumd = Assignment(symb_cumd, PHI((symb_lloq - ipred) / symb_sd))
    if method == 'm3':
        assignments += Assignment(
            y.symbol, sympy.Piecewise((y.expression, is_above_lloq), (cumd.expression, True))
        )
    else:
        assignments += cumd
        symb_cumdz = create_symbol(model, 'CUMDZ')
        assignments += Assignment(symb_cumdz, PHI(-ipred / symb_sd))

        y_below_lloq = (symb_cumd - symb_cumdz) / (1 - symb_cumdz)
        assignments += Assignment(
            y.symbol, sympy.Piecewise((y.expression, is_above_lloq), (y_below_lloq, True))
        )

    y_idx = sset.find_assignment_index(y.symbol)
    sset_new = sset[:y_idx] + assignments + sset[y_idx + 1 :]
    model = model.replace(statements=sset_new)

    return model.update_source()


def _verify_model(model, method):
    rvs = model.random_variables.epsilons
    if any(isinstance(rv, JointNormalDistribution) for rv in rvs):
        raise ValueError(
            f'Invalid input model: covariance between epsilons not supported in `method` {method}'
        )
    if not (
        has_additive_error_model(model)
        or has_proportional_error_model(model)
        or has_combined_error_model(model)
    ):
        raise ValueError('Invalid input model: error model not supported')


def _weight_as_sd(model, y, ipred):
    # FIXME: make more general
    sd_assignments = []

    expr = model.statements.find_assignment(y.symbol).expression
    rvs = model.random_variables.epsilons
    rvs_in_y = {sympy.Symbol(name) for name in rvs.names if sympy.Symbol(name) in expr.free_symbols}

    for arg in expr.args:
        if not rvs_in_y.intersection(arg.free_symbols):
            continue
        if isinstance(arg, sympy.Symbol):
            eps = model.random_variables[arg]
            sigma = eps.variance
            symb_add = create_symbol(model, 'ADD')
            add = Assignment(symb_add, sympy.sqrt(sigma))
            sd_assignments.append(add)
        else:
            rv = rvs_in_y.intersection(arg.free_symbols).pop()
            eps = model.random_variables[rv]
            sigma = eps.variance
            symb_prop = create_symbol(model, 'PROP')
            prop = Assignment(symb_prop, sympy.sqrt(sigma) * ipred)
            sd_assignments.append(prop)

    symb_sd = create_symbol(model, 'SD')

    sd = Assignment(symb_sd, sympy.sqrt(sympy.Add(*[ass.symbol**2 for ass in sd_assignments])))
    sd_assignments.append(sd)
    return sd_assignments, symb_sd
