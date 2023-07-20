"""
:meta private:
"""

import math
import re
import warnings
from operator import add, mul
from typing import List, Literal, Set, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import sympy
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs
from pharmpy.model import Assignment, Model, Parameter, Parameters, Statement, Statements

from .data import get_baselines
from .expressions import depends_on, remove_covariate_effect_from_statements, simplify_model

EffectType = Union[Literal['lin', 'cat', 'piece_lin', 'exp', 'pow'], str]
OperationType = Literal['*', '+']


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
    kept_thetas, before_odes = simplify_model(
        model,
        model.statements.before_odes,
        remove_covariate_effect_from_statements(
            model, model.statements.before_odes, parameter, covariate
        ),
    )
    ode_system: List[Statement] = (
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
            - If median of covariate equals minimum: :math:`100,000`
            - Otherwise: :math:`\\frac{1}{\\text{median} - \\text{min}}`
        - Lower:
            - If median of covariate equals maximum: :math:`-100,000`
            - Otherwise: :math:`\\frac{1}{\\text{median} - \\text{max}}`
    - Linear function for categorical covariates (*cat*)
        - Function:

            - If covariate is most common category:

            .. math::

                \\text{coveff} = 1

            - For each additional category:

            .. math::

                \\text{coveff} = 1 + \\text{theta}

        - Init: :math:`0.001`
        - Upper: :math:`5`
        - Lower: :math:`-1`
    - Piecewise linear function/"hockey-stick", continuous covariates only (*piece_lin*)
        - Function:
            - If cov <= median:

            .. math::

                \\text{coveff} = 1 + \\text{theta1} * (\\text{cov} - \\text{median})

            - If cov > median:

            .. math::

                \\text{coveff} = 1 + \\text{theta2} * (\\text{cov} - \\text{median})


        - Init: :math:`0.001`
        - Upper:
            - For first state: :math:`\\frac{1}{\\text{median} - \\text{min}}`
            - Otherwise: :math:`100,000`
        - Lower:
            - For first state: :math:`-100,000`
            - Otherwise: :math:`\\frac{1}{\\text{median} - \\text{max}}`
    - Exponential function, continuous covariates only (*exp*)
        - Function:

        .. math::

            \\text{coveff} = \\exp(\\text{theta} * (\\text{cov} - \\text{median}))

        - Init:
            - If lower > 0.001 or upper < 0.001: :math:`\\frac{\\text{upper} - \\text{lower}}{2}`
            - If estimated init is 0: :math:`\\frac{\\text{upper}}{2}`
            - Otherwise: :math:`0.001`
        - Upper:
            - If min - median = 0 or max - median = 0: :math:`100`
            - Otherwise:

            .. math::

                \\min(\\frac{\\log(0.01)}{\\text{min} - \\text{median}},
                \\frac{\\log(100)}{\\text{max} - \\text{median}})
        - Lower:
            - If min - median = 0 or max - median = 0: :math:`0.01`
            - Otherwise:

            .. math::

                \\max(\\frac{\\log(0.01)}{\\text{max} - \\text{median}},
                \\frac{\\log(100)}{\\text{min} - \\text{median}})

    - Power function, continuous covariates only (*pow*)
        - Function:

        .. math::

            \\text{coveff} = (\\frac{\\text{cov}}{\\text{median}})^\\text{theta}

        - Init: :math:`0.001`
        - Upper: :math:`100,000`
        - Lower: :math:`-100`


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
    allow_nested: bool, optional
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
    PTVCL*WGT*exp(ETA_1)*exp(POP_CLAPGR*(APGR - 7.0))

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
    # NOTE We hoist the statistic statements to avoid referencing variables
    # before declaring them. We also avoid duplicate statements.
    sset = [s for s in covariate_effect.statistic_statements if s not in sset] + sset

    last_existing_parameter_assignment = sset.find_assignment(parameter)
    assert last_existing_parameter_assignment is not None
    insertion_index = sset.index(last_existing_parameter_assignment) + 1

    # NOTE We can use any assignment to the parameter since we currently only
    # use its symbol to create the new effect statement.
    effect_statement = covariate_effect.create_effect_statement(
        operation, last_existing_parameter_assignment
    )

    statements = []

    statements.append(covariate_effect.template)
    statements.append(effect_statement)

    cov_possible = {sympy.Symbol(parameter)} | {
        sympy.Symbol(f'{parameter}{col_name}') for col_name in model.datainfo.names
    }

    # NOTE This is a heuristic that simplifies the NONMEM statements by
    # grouping multiple effect statements in a single statement.
    if last_existing_parameter_assignment.expression.args and all(
        map(cov_possible.__contains__, last_existing_parameter_assignment.expression.args)
    ):
        statements[-1] = Assignment(
            effect_statement.symbol,
            subs(
                effect_statement.expression,
                {parameter: last_existing_parameter_assignment.expression},
                simultaneous=True,
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
    elif effect == 'piece_lin':
        return CovariateEffect.piecewise_linear()
    elif effect == 'exp':
        return CovariateEffect.exponential()
    elif effect == 'pow':
        return CovariateEffect.power()
    else:
        symbol = sympy.Symbol('symbol')
        expression = parse_expr(effect)
        return CovariateEffect(Assignment(symbol, expression))


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
        self.template = Assignment(
            sympy.Symbol(effect_name),
            subs(
                subs(
                    self.template.expression,
                    thetas,
                ),
                {'cov': covariate},
                simultaneous=True,
            ),
        )

        template_str = [str(symbol) for symbol in self.template.free_symbols]

        if 'mean' in template_str:
            self.template = self.template.subs({'mean': f'{covariate}_MEAN'})
            s = Assignment(sympy.Symbol(f'{covariate}_MEAN'), sympy.Float(statistics['mean'], 6))
            self.statistic_statements.append(s)
        if 'median' in template_str:
            self.template = self.template.subs({'median': f'{covariate}_MEDIAN'})
            s = Assignment(
                sympy.Symbol(f'{covariate}_MEDIAN'), sympy.Float(statistics['median'], 6)
            )
            self.statistic_statements.append(s)
        if 'std' in template_str:
            self.template = self.template.subs({'std': f'{covariate}_STD'})
            s = Assignment(sympy.Symbol(f'{covariate}_STD'), sympy.Float(statistics['std'], 6))
            self.statistic_statements.append(s)

    def create_effect_statement(self, operation_str, statement_original):
        """Creates statement for addition or multiplication of covariate
        to parameter, e.g. (if parameter is CL and covariate is WGT):

            CL = CLWGT + TVCL*EXP(ETA(1))"""
        operation = self._get_operation(operation_str)

        symbol = statement_original.symbol
        expression = statement_original.symbol

        statement_new = Assignment(symbol, operation(expression, self.template.symbol))

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
        symbol = sympy.Symbol('symbol')
        expression = 1 + sympy.Symbol('theta') * (sympy.Symbol('cov') - sympy.Symbol('median'))
        template = Assignment(symbol, expression)

        return cls(template)

    @classmethod
    def categorical(cls, counts):
        """Linear categorical template (for categorical covariates)."""
        symbol = sympy.Symbol('symbol')
        most_common = counts.idxmax()
        categories = list(counts.index)

        values = [1]
        conditions = [sympy.Eq(sympy.Symbol('cov'), most_common)]

        for i, cat in enumerate(categories, 1):
            if cat != most_common:
                if np.isnan(cat):
                    conditions += [sympy.Eq(sympy.Symbol('cov'), sympy.Symbol('NaN'))]
                    values += [1]
                else:
                    conditions += [sympy.Eq(sympy.Symbol('cov'), cat)]
                    if len(categories) == 2:
                        values += [1 + sympy.Symbol('theta')]
                    else:
                        values += [1 + sympy.Symbol(f'theta{i}')]

        expression = sympy.Piecewise(*zip(values, conditions))

        template = Assignment(symbol, expression)

        return cls(template)

    @classmethod
    def piecewise_linear(cls):
        """Piecewise linear ("hockey-stick") template (for continuous
        covariates)."""
        symbol = sympy.Symbol('symbol')
        values = [
            1 + sympy.Symbol('theta1') * (sympy.Symbol('cov') - sympy.Symbol('median')),
            1 + sympy.Symbol('theta2') * (sympy.Symbol('cov') - sympy.Symbol('median')),
        ]
        conditions = [
            sympy.Le(sympy.Symbol('cov'), sympy.Symbol('median')),
            sympy.Gt(sympy.Symbol('cov'), sympy.Symbol('median')),
        ]
        expression = sympy.Piecewise((values[0], conditions[0]), (values[1], conditions[1]))

        template = Assignment(symbol, expression)

        return cls(template)

    @classmethod
    def exponential(cls):
        """Exponential template (for continuous covariates)."""
        symbol = sympy.Symbol('symbol')
        expression = sympy.exp(
            sympy.Symbol('theta') * (sympy.Symbol('cov') - sympy.Symbol('median'))
        )
        template = Assignment(symbol, expression)

        return cls(template)

    @classmethod
    def power(cls):
        """Power template (for continuous covariates)."""
        symbol = sympy.Symbol('symbol')
        expression = (sympy.Symbol('cov') / sympy.Symbol('median')) ** sympy.Symbol('theta')
        template = Assignment(symbol, expression)

        return cls(template)

    def __str__(self):
        """String representation of class."""
        return str(self.template)


def get_covariates_allowed_in_covariate_effect(model: Model) -> Set[str]:
    try:
        di_covariate = model.datainfo.typeix['covariate'].names
    except IndexError:
        di_covariate = []

    try:
        di_unknown = model.datainfo.typeix['unknown'].names
    except IndexError:
        di_unknown = []

    return set(di_covariate).union(di_unknown)
