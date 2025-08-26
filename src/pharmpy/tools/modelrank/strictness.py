import re
from typing import Optional

from pharmpy.basic.expr import BooleanExpr
from pharmpy.deps import numpy as np
from pharmpy.modeling import check_parameters_near_bounds, get_omegas, get_sigmas, get_thetas
from pharmpy.workflows import ModelEntry

ALLOWED_ARGS = (
    'minimization_successful',
    'rounding_errors',
    'sigdigs',
    'maxevals_exceeded',
    'rse',
    'rse_theta',
    'rse_omega',
    'rse_sigma',
    'condition_number',
    'final_zero_gradient',
    'final_zero_gradient_theta',
    'final_zero_gradient_omega',
    'final_zero_gradient_sigma',
    'estimate_near_boundary',
    'estimate_near_boundary_theta',
    'estimate_near_boundary_omega',
    'estimate_near_boundary_sigma',
)

LOGICAL_ARGS = ('and', 'or', 'not')


def get_strictness_expr(strictness: str) -> BooleanExpr:
    validate_string(strictness)
    strictness = preprocess_string(strictness)
    expr = BooleanExpr(strictness)
    return expr


def get_strictness_args(strictness: str) -> list[str]:
    find_all_words = re.findall(r'[^\d\W]+', strictness)
    return [w for w in find_all_words if w not in LOGICAL_ARGS]


def validate_string(strictness: str):
    strictness = strictness.lower()
    args = get_strictness_args(strictness)
    find_all_non_allowed_operators = re.findall(r"[^\w\s\.\<\>\=\!\(\)]", strictness)
    if len(find_all_non_allowed_operators) > 0:
        raise ValueError(f"Unallowed operators found: {', '.join(find_all_non_allowed_operators)}")

    # Check that only allowed arguments are in the statement
    if not all(map(lambda x: x in ALLOWED_ARGS, args)):
        raise ValueError(f'Some expressions were not correct. Valid arguments are: {ALLOWED_ARGS}')


def preprocess_string(strictness: str) -> str:
    if strictness == '':
        return 'True'
    strictness = strictness.lower()
    strictness = strictness.replace(' and ', ' & ').replace(' or ', ' | ').replace('not ', '~')
    comparison_pattern = r'(\w+\s*[<>=]+\s*\d+\.\d+)'
    strictness = re.sub(comparison_pattern, r'(\1)', strictness)
    return strictness


def evaluate_strictness(expr: BooleanExpr, predicates: dict[str, Optional[bool]]) -> Optional[bool]:
    sub_dict = dict()
    for key, value in predicates.items():
        # NaNs will raise in subs
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        sub_dict[key] = value
    expr_subs = expr.subs(sub_dict)
    if expr_subs.is_indeterminate():
        # Subs with expressions like sigdigs >= 3.8 will not replace, only symbols
        if str(expr_subs) in predicates.keys():
            return predicates[str(expr_subs)]
        return None
    else:
        return True if expr_subs.is_true() else False


def get_strictness_predicates(
    model_entries: list[ModelEntry], expr: BooleanExpr
) -> dict[ModelEntry, dict[str, Optional[bool]]]:
    return {me: get_strictness_predicates_me(me, expr) for me in model_entries}


def get_strictness_predicates_me(me: ModelEntry, expr: BooleanExpr) -> dict[str, Optional[bool]]:
    predicates = dict()

    def _eval_tree(sub_expr):
        for arg in sub_expr.args:
            _eval_tree(arg)
        expr_key = str(sub_expr)
        expr_eval = _eval_strictness_arg(me, expr_key)
        if expr_eval is None:
            if len(sub_expr.free_symbols) != 1:
                return
            symb = str(sub_expr.free_symbols.pop())
            if symb in predicates.keys() and (symb_expr := predicates[symb]) is not None:
                if np.isnan(symb_expr):
                    # All comparisons with NaN should be False
                    predicates[expr_key] = False
                else:
                    sub_expr = sub_expr.subs({symb: predicates[symb]})
                    if not sub_expr.is_indeterminate():
                        sub_expr = True if sub_expr.is_true() else False
                    predicates[expr_key] = sub_expr
            else:
                predicates[expr_key] = None
        else:
            predicates[expr_key] = expr_eval

    if expr.is_indeterminate():
        _eval_tree(expr)
        strictness_fulfilled = evaluate_strictness(expr, predicates)
        predicates['strictness_fulfilled'] = strictness_fulfilled
    else:
        predicates['strictness_fulfilled'] = True if expr.is_true() else False

    return predicates


def _eval_strictness_arg(me: ModelEntry, arg: str):
    model, res = me.model, me.modelfit_results
    synonyms = {'sigdigs': 'significant_digits'}
    termination_causes = ('rounding_errors', 'maxevals_exceeded')
    if arg in termination_causes:
        return arg in res.termination_cause if res.termination_cause else False
    elif arg in synonyms.keys():
        return getattr(res, synonyms[arg])
    elif arg.startswith('estimate_near_boundary'):
        ests = _get_subset_on_param_type(res.parameter_estimates, arg, model)
        near_bounds = check_parameters_near_bounds(model, ests).any()
        return bool(near_bounds)
    elif arg.startswith('rse') and not arg.startswith('rse '):
        if model.execution_steps[-1].parameter_uncertainty_method is None:
            return None
        if (rse := res.relative_standard_errors) is None or rse.isna().any():
            return np.nan
        rse = _get_subset_on_param_type(rse, arg, model)
        return max(rse)
    elif arg.startswith('final_zero_gradient'):
        if res.gradients is None:
            return None
        grd = _get_subset_on_param_type(res.gradients, arg, model)
        return bool((grd == 0).any()) or bool(grd.isnull().any())
    elif arg == 'condition_number':
        if res.correlation_matrix is None:
            return None
        return np.linalg.cond(res.correlation_matrix)
    else:
        try:
            return getattr(res, arg)
        except AttributeError:
            return None


def _get_subset_on_param_type(df, arg, model):
    if 'theta' in arg:
        return df[df.index.isin(get_thetas(model).names)]
    elif 'omega' in arg:
        return df[df.index.isin(get_omegas(model).names)]
    elif 'sigma' in arg:
        return df[df.index.isin(get_sigmas(model).names)]
    else:
        return df
