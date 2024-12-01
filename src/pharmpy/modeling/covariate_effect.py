"""
:meta private:
"""

from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from operator import add, mul
from typing import Literal, Union

from pharmpy.basic.expr import BooleanExpr, Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import sympy
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs
from pharmpy.model import Assignment, Model, Parameter, Parameters, Statement, Statements

from .common import get_model_covariates
from .data import get_baselines
from .expressions import (
    depends_on,
    get_individual_parameters,
    get_mu_connected_to_parameter,
    has_mu_reference,
    remove_covariate_effect_from_statements,
    simplify_model,
)
from .parameters import get_thetas

EffectType = Union[Literal['lin', 'cat', 'cat2', 'piece_lin', 'exp', 'pow'], str]
OperationType = Literal['*', '+']


def get_covariate_effects(model: Model) -> dict[list]:
    """Return a dictionary of all used covariates within a model

    The dictionary will have parameter name as key with a connected value as
    a list of tuple(s) with (covariate, effect type, operator)

    Parameters
    ----------
    model : Model
        Model to extract covariates from.

    Returns
    -------
        Dictionary : Dictionary of parameters and connected covariate(s)

    """
    parameters = get_individual_parameters(model)
    covariates = get_model_covariates(model)

    param_w_cov = defaultdict(list)

    for p in parameters:
        for c in covariates:
            if has_covariate_effect(model, str(p), str(c)):
                param_w_cov[p].append(c)
    res = defaultdict(list)
    for param, covariates in param_w_cov.items():
        for cov in covariates:
            coveffect, op = _get_covariate_effect(model, param, cov)
            if coveffect and op:
                res[(param, cov)].append((coveffect, op))
    return res


def _get_covariate_effect(model: Model, symbol, covariate):
    param_expr = model.statements.before_odes.full_expression(symbol)
    param_expr = sympy.sympify(param_expr)
    covariate = sympy.sympify(covariate)

    etas = tuple(sympy.sympify(e) for e in model.random_variables.etas.symbols)
    thetas = tuple(sympy.sympify(t) for t in get_thetas(model).symbols)

    if isinstance(param_expr, sympy.Mul):
        op = "*"
    elif isinstance(param_expr, sympy.Add):
        op = "+"
    else:
        # Do nothing ?
        pass

    cov_expression = None
    perform_matching = False
    for arg in param_expr.args:
        check_covariate = False
        free_symbols = arg.free_symbols
        if any(eta in free_symbols for eta in etas):
            if Expr(arg).is_exp() and covariate in free_symbols:
                skip = False
                expression = arg
                exp_terms = arg.args[0]
                for a in exp_terms.args:
                    if covariate not in a.free_symbols:
                        expression = expression.subs({a: Expr(0)})
                    else:
                        # If both covariate and ETA in same term -> SKIP
                        if any(eta in a.free_symbols for eta in etas):
                            skip = True
                if not skip:
                    check_covariate = True
        else:
            check_covariate = True
            expression = arg

        if check_covariate:
            if covariate in free_symbols:
                cov_expression = expression
                # Need at least one theta to perform matching
                cov_effect = "CUSTOM"
                if any(theta in free_symbols for theta in thetas):
                    perform_matching = True
    if perform_matching:
        for effect in ['lin', 'cat', 'cat2', 'piece_lin', 'exp', 'pow']:
            template = _create_template(effect, model, str(covariate))
            template = template.template.expression
            template = sympy.sympify(template)
            wild_dict = defaultdict(list)
            for s in template.free_symbols:
                wild_symbol = sympy.Wild(str(s))
                template = template.subs({s: wild_symbol})
                if str(s).startswith("theta"):
                    wild_dict["theta"].append(wild_symbol)
                elif str(s).startswith("cov"):
                    wild_dict["cov"].append(wild_symbol)
                elif str(s).startswith("median"):
                    wild_dict["median"].append(wild_symbol)

            cov_expression = sympy.sympify(cov_expression)
            match = cov_expression.match(template)
            if match:
                if _assert_cov_effect_match(wild_dict, match, model, str(covariate), effect):
                    return effect, op

    if cov_expression:
        return cov_effect, op

    return None, None


def _assert_cov_effect_match(symbols, match, model, covariate, effect):
    if effect == "pow":
        if (
            sympy.Wild("cov") in match.keys()
            and match[sympy.Wild("cov")].is_number
            and sympy.Wild("median") in match.keys()
            and match[sympy.Wild("median")].is_Pow
        ):
            temp = match[sympy.Wild("cov")]
            match[sympy.Wild("cov")] = match[sympy.Wild("median")]
            match[sympy.Wild("median")] = temp

    for key, values in symbols.items():
        if key == "theta":
            thetas = get_thetas(model).symbols
            if all(value in match.keys() for value in values):
                if any(match[value] not in thetas for value in values):
                    return False
        if key == "cov":
            covariate = sympy.Symbol(covariate)
            if all(value in match.keys() for value in values):
                if all(match[value] not in [covariate, 1 / covariate] for value in values):
                    return False
        if key == "median":
            if all(value in match.keys() for value in values):
                if not all(match[value].is_number for value in values):
                    return False
    return True


def has_covariate_effect(model: Model, parameter: str, covariate: str):
    """Tests if an instance of :class:`pharmpy.model` has a given covariate
    effect.

    Parameters
    ----------
    model : Model
        Pharmpy model to check for covariate effect.
    parameter : str
        Name of parameter.
    covariate : str
        Name of covariate.

    Return
    ------
    bool
        Whether input model has a covariate effect of the input covariate on
        the input parameter.

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> has_covariate_effect(model, "CL", "APGR")
    False

    """
    return depends_on(model, parameter, covariate)


def remove_covariate_effect(model: Model, parameter: str, covariate: str):
    """Remove a covariate effect from an instance of :class:`pharmpy.model`.

    Parameters
    ----------
    model : Model
        Pharmpy model from which to remove the covariate effect.
    parameter : str
        Name of parameter.
    covariate : str
        Name of covariate.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> has_covariate_effect(model, "CL", "WGT")
    True
    >>> model = remove_covariate_effect(model, "CL", "WGT")
    >>> has_covariate_effect(model, "CL", "WGT")
    False

    """
    if not has_covariate_effect(model, parameter, covariate):
        return model
    kept_thetas, before_odes = simplify_model(
        model,
        model.statements.before_odes,
        remove_covariate_effect_from_statements(
            model, model.statements.before_odes, parameter, covariate
        ),
    )
    ode_system: list[Statement] = (
        [] if model.statements.ode_system is None else [model.statements.ode_system]
    )
    after_odes = list(model.statements.after_odes)
    statements = Statements(before_odes + ode_system + after_odes)
    kept_parameters = model.random_variables.free_symbols.union(
        kept_thetas, model.statements.after_odes.free_symbols
    )
    parameters = Parameters.create((p for p in model.parameters if p.symbol in kept_parameters))
    model = model.replace(statements=statements, parameters=parameters)

    return model.update_source()


def add_covariate_effect(
    model: Model,
    parameter: str,
    covariate: str,
    effect: EffectType,
    operation: OperationType = '*',
    allow_nested: bool = False,
):
    """Adds covariate effect to :class:`pharmpy.model`.

    The following effects have templates:

    - Linear function for continuous covariates (*lin*)
        - Function:

        .. math::

            \\text{coveff} = 1 + \\text{theta} * (\\text{cov} - \\text{median})

        - Init:  0.001
        - Upper:
            - If median of covariate equals minimum: 100,000
            - Otherwise: :math:`\\frac{1}{\\text{median} - \\text{min}}`
        - Lower:
            - If median of covariate equals maximum: -100,000
            - Otherwise: :math:`\\frac{1}{\\text{median} - \\text{max}}`
    - Linear function for categorical covariates (*cat*)
        - Function:
            - If covariate is the most common category:

            .. math::

                \\text{coveff} = 1

            - For each additional category:

            .. math::

                \\text{coveff} = 1 + \\text{theta}

        - Init: 0.001
        - Upper: 5
        - Lower: -1
    - (alternative) Linear function for categorical covariates (*cat2*)
        - Function:
            - If covariate is the most common category:

            .. math::

                \\text{coveff} = 1

            - For each additional category:

            .. math::

                \\text{coveff} = \\text{theta}

        - Init: 0.001
        - Upper: 6
        - Lower: 0
    - Piecewise linear function/"hockey-stick", continuous covariates only (*piece_lin*)
        - Function:
            - If cov <= median:

            .. math::

                \\text{coveff} = 1 + \\text{theta1} * (\\text{cov} - \\text{median})

            - If cov > median:

            .. math::

                \\text{coveff} = 1 + \\text{theta2} * (\\text{cov} - \\text{median})


        - Init: 0.001
        - Upper:
            - For first state: :math:`\\frac{1}{\\text{median} - \\text{min}}`
            - Otherwise: 100,000
        - Lower:
            - For first state: -100,000
            - Otherwise: :math:`\\frac{1}{\\text{median} - \\text{max}}`
    - Exponential function, continuous covariates only (*exp*)
        - Function:

        .. math::

            \\text{coveff} = \\exp(\\text{theta} * (\\text{cov} - \\text{median}))

        - Init:
            - If lower > 0.001 or upper < 0.001: :math:`\\frac{\\text{upper} - \\text{lower}}{2}`
            - If estimated init is 0: :math:`\\frac{\\text{upper}}{2}`
            - Otherwise: 0.001
        - Upper:
            - If min - median = 0 or max - median = 0: 100
            - Otherwise:

            .. math::

                \\min(\\frac{\\log(0.01)}{\\text{min} - \\text{median}},
                \\frac{\\log(100)}{\\text{max} - \\text{median}})
        - Lower:
            - If min - median = 0 or max - median = 0: 0.01
            - Otherwise:

            .. math::

                \\max(\\frac{\\log(0.01)}{\\text{max} - \\text{median}},
                \\frac{\\log(100)}{\\text{min} - \\text{median}})

    - Power function, continuous covariates only (*pow*)
        - Function:

        .. math::

            \\text{coveff} = (\\frac{\\text{cov}}{\\text{median}})^\\text{theta}

        - Init: 0.001
        - Upper: 100,000
        - Lower: -100


    Parameters
    ----------
    model : Model
        Pharmpy model to add covariate effect to.
    parameter : str
        Name of parameter to add covariate effect to.
    covariate : str
        Name of covariate.
    effect : str
        Type of covariate effect. May be abbreviated covariate effect (see above) or custom.
    operation : str, optional
        Whether the covariate effect should be added or multiplied (default).
    allow_nested : bool, optional
        Whether to allow adding a covariate effect when one already exists for
        the input parameter-covariate pair.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_covariate_effect(model, "CL", "APGR", "exp")
    >>> model.statements.before_odes.full_expression("CL")
    POP_CL*WGT*exp(ETA_CL + POP_CLAPGR*(APGR - 7.0))

    """
    sset = model.statements

    if not allow_nested and depends_on(model, parameter, covariate):
        warnings.warn(f'Covariate effect of {covariate} on {parameter} already exists')
        return model

    statistics = {}
    statistics['mean'] = _calculate_mean(model.dataset, covariate)
    statistics['median'] = _calculate_median(model, covariate)
    statistics['std'] = _calculate_std(model, covariate)

    covariate_effect = _create_template(effect, model, covariate)
    pset, thetas = _create_thetas(model, parameter, effect, covariate, covariate_effect.template)
    covariate_effect.apply(parameter, covariate, thetas, statistics)
    # NOTE: We hoist the statistic statements to avoid referencing variables
    # before declaring them. We also avoid duplicate statements.
    sset = [s for s in covariate_effect.statistic_statements if s not in sset] + sset

    if has_mu_reference(model):
        mu_symbol = get_mu_connected_to_parameter(model, parameter)
        last_existing_parameter_assignment = sset.find_assignment(mu_symbol)
    else:
        last_existing_parameter_assignment = sset.find_assignment(parameter)
    assert last_existing_parameter_assignment is not None
    insertion_index = sset.index(last_existing_parameter_assignment) + 1

    # NOTE: We can use any assignment to the parameter since we currently only
    # use its symbol to create the new effect statement.
    effect_statement = covariate_effect.create_effect_statement(
        operation, last_existing_parameter_assignment
    )

    statements = []

    statements.append(covariate_effect.template)
    statements.append(effect_statement)
    cov_possible = {Expr.symbol(parameter)} | {
        Expr.symbol(f'{parameter}{col_name}') for col_name in model.datainfo.names
    }

    if has_mu_reference(model):
        mu_assignment = sset.find_assignment(mu_symbol)
        parameter_assignment = sset.find_assignment(parameter)

        index = {Expr.symbol(eta): i for i, eta in enumerate(model.random_variables.etas.names, 1)}
        etas = set(index)
        eta = next(iter(etas.intersection(parameter_assignment.expression.free_symbols)))

        old_def = parameter_assignment.subs(
            {mu_assignment.symbol: mu_assignment.expression}
        ).expression._sympy_()
        remove_iiv_def = old_def.as_independent(eta)[0]
        new_mu_expression = sympy.solve(
            subs(old_def, {remove_iiv_def: remove_iiv_def * statements[0].symbol})
            - parameter_assignment.expression,
            mu_assignment.symbol,
        )[0]

        statements[-1] = Assignment.create(effect_statement.symbol, new_mu_expression)
        sset = sset[0 : insertion_index - 1] + sset[insertion_index:]
        insertion_index -= 1

    # NOTE: This is a heuristic that simplifies the NONMEM statements by
    # grouping multiple effect statements in a single statement.
    if last_existing_parameter_assignment.expression.args and all(
        map(cov_possible.__contains__, last_existing_parameter_assignment.expression.args)
    ):
        statements[-1] = Assignment.create(
            effect_statement.symbol,
            effect_statement.expression.subs(
                {parameter: last_existing_parameter_assignment.expression},
            ),
        )
        sset = sset[0 : insertion_index - 1] + sset[insertion_index:]
        insertion_index -= 1
    sset = sset[0:insertion_index] + statements + sset[insertion_index:]
    model = model.replace(parameters=pset, statements=sset)
    return model.update_source()


def natural_order(string, _nsre=re.compile(r'([0-9]+)')):
    # From https://stackoverflow.com/a/16090640
    return [int(key) if key.isdigit() else (key.lower(), key) for key in _nsre.split(string)]


def _create_thetas(model, parameter, effect, covariate, template, _ctre=re.compile(r'theta\d*')):
    """Creates theta parameters and adds to parameter set of model.

    Number of parameters depends on how many thetas have been declared."""
    new_thetas = {sym for sym in map(str, template.expression.free_symbols) if _ctre.match(sym)}
    no_of_thetas = len(new_thetas)

    pset = model.parameters

    theta_names = {}

    if no_of_thetas == 1:
        inits = _choose_param_inits(effect, model, covariate)

        theta_name = f'POP_{parameter}{covariate}'
        pset = Parameters.create(
            list(pset) + [Parameter(theta_name, inits['init'], inits['lower'], inits['upper'])]
        )
        theta_names['theta'] = theta_name
    else:
        for i, new_theta in enumerate(sorted(new_thetas, key=natural_order), 1):
            inits = _choose_param_inits(effect, model, covariate, i)

            theta_name = f'POP_{parameter}{covariate}_{i}'
            pset = Parameters.create(
                list(pset) + [Parameter(theta_name, inits['init'], inits['lower'], inits['upper'])]
            )
            theta_names[new_theta] = theta_name

    return pset, theta_names


def _count_categorical(model, covariate):
    """Gets the number of individuals that has a level of categorical covariate."""
    idcol = model.datainfo.id_column.name
    df = model.dataset.set_index(idcol)
    allcounts = df[covariate].groupby('ID').value_counts()
    allcounts.name = None  # To avoid collisions when resetting index
    counts = allcounts.reset_index().iloc[:, 1].value_counts()
    counts.sort_index(inplace=True)  # To make deterministic in case of multiple modes
    if model.dataset[covariate].isna().any():
        counts[np.nan] = 0
    return counts


def _calculate_mean(df, covariate, baselines=False):
    """Calculate mean. Can be set to use baselines, otherwise it is
    calculated first per individual, then for the group."""
    if baselines:
        return df[str(covariate)].mean()
    else:
        return df.groupby('ID')[str(covariate)].mean().mean()


def _calculate_median(model, covariate, baselines=False):
    """Calculate median. Can be set to use baselines, otherwise it is
    calculated first per individual, then for the group."""
    if baselines:
        return get_baselines(model)[str(covariate)].median()
    else:
        df = model.dataset
        return df.groupby('ID')[str(covariate)].median().median()


def _calculate_std(model, covariate, baselines=False):
    """Calculate median. Can be set to use baselines, otherwise it is
    calculated first per individual, then for the group."""
    if baselines:
        return get_baselines(model)[str(covariate)].std()
    else:
        df = model.dataset
        return df.groupby('ID')[str(covariate)].mean().std()


def _choose_param_inits(effect, model, covariate, index=None):
    """Chooses inits for parameters. If the effect is exponential, the
    bounds need to be dynamic."""
    df = model.dataset
    init_default = 0.001

    inits = {}

    cov_median = _calculate_median(model, covariate)
    cov_min = df[str(covariate)].min()
    cov_max = df[str(covariate)].max()

    lower, upper = _choose_bounds(effect, cov_median, cov_min, cov_max, index)

    if effect == 'exp':
        if lower > init_default or init_default > upper:
            init = (upper + lower) / 2
            if init == 0:
                init = upper / 5
        else:
            init = init_default
    elif effect == 'pow':
        init = init_default
    elif effect == "cat2":
        init = 1.01
    else:
        init = init_default

    inits['init'] = init
    inits['lower'] = lower
    inits['upper'] = upper

    return inits


def _choose_bounds(effect, cov_median, cov_min, cov_max, index=None):
    if effect == 'exp':
        min_diff = cov_min - cov_median
        max_diff = cov_max - cov_median

        lower_expected = 0.01
        upper_expected = 100

        if min_diff == 0 or max_diff == 0:
            return lower_expected, upper_expected
        else:
            log_base = 10
            lower = max(
                math.log(lower_expected, log_base) / max_diff,
                math.log(upper_expected, log_base) / min_diff,
            )
            upper = min(
                math.log(lower_expected, log_base) / min_diff,
                math.log(upper_expected, log_base) / max_diff,
            )
    elif effect == 'lin':
        if cov_median == cov_min:
            upper = 100000
        else:
            upper = 1 / (cov_median - cov_min)
        if cov_median == cov_max:
            lower = -100000
        else:
            lower = 1 / (cov_median - cov_max)
    elif effect == 'piece_lin':
        if cov_median == cov_min or cov_median == cov_max:
            raise Exception(
                'Median cannot be same as min or max, cannot use '
                'piecewise-linear parameterization.'
            )
        if index == 0:
            lower = -100000
            upper = 1 / (cov_median - cov_min)
        else:
            lower = 1 / (cov_median - cov_max)
            upper = 100000
    elif effect == 'pow':
        lower = -100
        upper = 100000
    elif effect == 'cat':
        lower = -1
        upper = 5
    elif effect == 'cat2':
        lower = 0
        upper = 6
    else:
        lower = -100000
        upper = 100000
    return round(lower, 4), round(upper, 4)


def _create_template(effect, model, covariate):
    """Creates Covariate class objects with effect template."""
    if effect == 'lin':
        return CovariateEffect.linear()
    elif effect == 'cat':
        counts = _count_categorical(model, covariate)
        return CovariateEffect.categorical(counts)
    elif effect == 'cat2':
        counts = _count_categorical(model, covariate)
        return CovariateEffect.categorical(counts, alternative=True)
    elif effect == 'piece_lin':
        return CovariateEffect.piecewise_linear()
    elif effect == 'exp':
        return CovariateEffect.exponential()
    elif effect == 'pow':
        return CovariateEffect.power()
    else:
        symbol = Expr.symbol('symbol')
        expression = parse_expr(effect)
        return CovariateEffect(Assignment.create(symbol, expression))


class CovariateEffect:
    """
    Covariate effect consisting of new assignments.

    Attributes
    ----------
    template
        Assignment based on covariate effect
    statistic_statements
        Dict with mean, median and standard deviation

    :meta private:

    """

    def __init__(self, template):
        self.template = template
        self.statistic_statements = []

    def apply(self, parameter, covariate, thetas, statistics):
        effect_name = f'{parameter}{covariate}'
        theta_subs = self.template.expression.subs(thetas)
        cov_subs = theta_subs.subs({'cov': covariate})

        self.template = Assignment.create(
            Expr(effect_name),
            cov_subs,
        )

        template_str = [str(symbol) for symbol in self.template.free_symbols]

        if 'mean' in template_str:
            self.template = self.template.subs({'mean': f'{covariate}_MEAN'})
            s = Assignment.create(Expr.symbol(f'{covariate}_MEAN'), statistics['mean'])
            self.statistic_statements.append(s)
        if 'median' in template_str:
            self.template = self.template.subs({'median': f'{covariate}_MEDIAN'})
            s = Assignment.create(Expr.symbol(f'{covariate}_MEDIAN'), statistics['median'])
            self.statistic_statements.append(s)
        if 'std' in template_str:
            self.template = self.template.subs({'std': f'{covariate}_STD'})
            s = Assignment.create(Expr.symbol(f'{covariate}_STD'), statistics['std'])
            self.statistic_statements.append(s)

    def create_effect_statement(self, operation_str, statement_original):
        """Creates statement for addition or multiplication of covariate
        to parameter, e.g. (if parameter is CL and covariate is WGT):

            CL = CLWGT + TVCL*EXP(ETA(1))"""
        operation = self._get_operation(operation_str)

        symbol = statement_original.symbol
        expression = statement_original.symbol

        statement_new = Assignment.create(symbol, operation(expression, self.template.symbol))

        return statement_new

    @staticmethod
    def _get_operation(operation_str):
        """Gets sympy operation based on string"""
        if operation_str == '*':
            return mul
        elif operation_str == '+':
            return add

        raise NotImplementedError(f'Can only handle + or *, got {operation_str}.')

    @classmethod
    def linear(cls):
        """Linear continuous template (for continuous covariates)."""
        symbol = Expr.symbol('symbol')
        expression = 1 + Expr.symbol('theta') * (Expr.symbol('cov') - Expr.symbol('median'))
        template = Assignment.create(symbol, expression)

        return cls(template)

    @classmethod
    def categorical(cls, counts, alternative=False):
        """Linear categorical template (for categorical covariates)."""
        symbol = Expr.symbol('symbol')
        most_common = counts.idxmax()
        categories = list(counts.index)

        values = [1]
        conditions = [BooleanExpr.eq(Expr.symbol('cov'), most_common)]

        for i, cat in enumerate(categories, 1):
            if cat != most_common:
                if np.isnan(cat):
                    conditions += [BooleanExpr.eq(Expr.symbol('cov'), Expr.symbol('NaN'))]
                    values += [1]
                else:
                    conditions += [BooleanExpr.eq(Expr.symbol('cov'), cat)]
                    if len(categories) == 2:
                        if alternative:
                            values += [Expr.symbol('theta')]
                        else:
                            values += [1 + Expr.symbol('theta')]
                    else:
                        if alternative:
                            values += [Expr.symbol(f'theta{i}')]
                        else:
                            values += [1 + Expr.symbol(f'theta{i}')]

        expression = Expr.piecewise(*zip(values, conditions))

        template = Assignment.create(symbol, expression)

        return cls(template)

    @classmethod
    def piecewise_linear(cls):
        """Piecewise linear ("hockey-stick") template (for continuous
        covariates)."""
        symbol = Expr.symbol('symbol')
        values = [
            1 + Expr.symbol('theta1') * (Expr.symbol('cov') - Expr.symbol('median')),
            1 + Expr.symbol('theta2') * (Expr.symbol('cov') - Expr.symbol('median')),
        ]
        conditions = [
            BooleanExpr.le(Expr.symbol('cov'), Expr.symbol('median')),
            BooleanExpr.gt(Expr.symbol('cov'), Expr.symbol('median')),
        ]
        expression = Expr.piecewise((values[0], conditions[0]), (values[1], conditions[1]))

        template = Assignment.create(symbol, expression)

        return cls(template)

    @classmethod
    def exponential(cls):
        """Exponential template (for continuous covariates)."""
        symbol = Expr.symbol('symbol')
        expression = Expr.exp(Expr.symbol('theta') * (Expr.symbol('cov') - Expr.symbol('median')))
        template = Assignment.create(symbol, expression)

        return cls(template)

    @classmethod
    def power(cls):
        """Power template (for continuous covariates)."""
        symbol = Expr.symbol('symbol')
        expression = (Expr.symbol('cov') / Expr.symbol('median')) ** Expr.symbol('theta')
        template = Assignment.create(symbol, expression)

        return cls(template)

    def __str__(self):
        """String representation of class."""
        return str(self.template)


def get_covariates_allowed_in_covariate_effect(model: Model) -> set[str]:
    try:
        di_covariate = model.datainfo.typeix['covariate'].names
    except IndexError:
        di_covariate = []

    try:
        di_admid = model.datainfo.typeix['admid'].names
    except IndexError:
        di_admid = []

    try:
        di_unknown = model.datainfo.typeix['unknown'].names
    except IndexError:
        di_unknown = []

    return set(di_covariate).union(di_unknown, di_admid)
