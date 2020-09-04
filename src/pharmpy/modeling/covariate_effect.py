import math
import re
from operator import add, mul

import numpy as np
from sympy import Eq, Float, Gt, Le, Piecewise, exp

from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment
from pharmpy.symbols import real, subs, sympify


def add_covariate_effect(model, parameter, covariate, effect, operation='*'):
    """Adds covariate effect to pharmpy model.

       effect - Supports linear (continuous and categorical covariates),
                piecewise linear, exponential, and power function. Custom
                effect may be used, where thetas are denoted as 'theta'
                (if multiple: 'theta1', 'theta2' etc), covariate as 'cov',
                and mean and median are written as they are.
    """
    mean = calculate_mean(model.dataset, covariate)
    median = calculate_median(model.dataset, covariate)

    thetas = create_thetas(model, effect, covariate)
    covariate_effect = create_template(effect, model, covariate)

    sset = model.statements
    param_statement = sset.find_assignment(parameter)

    covariate_effect.apply(parameter, covariate, thetas)
    statistic_statement = covariate_effect.create_statistics_statement(parameter, mean, median)
    effect_statement = covariate_effect.create_effect_statement(operation, param_statement)

    param_index = sset.index(param_statement)
    sset.insert(param_index + 1, covariate_effect.template)
    sset.insert(param_index + 2, statistic_statement)
    sset.insert(param_index + 3, effect_statement)

    model.statements = sset

    return model


def create_thetas(model, effect, covariate):
    if effect == 'piece_lin':
        no_of_thetas = 2
    elif effect == 'lin_cat':
        no_of_thetas = count_categorical(model, covariate).nunique()
    else:
        no_of_thetas = 1

    pset = model.parameters

    theta_names = dict()
    theta_name = str(model.create_symbol(stem='COVEFF', force_numbering=True))
    theta_lower, theta_upper = choose_param_inits(effect, model.dataset, covariate)

    if no_of_thetas == 1:
        pset.add(Parameter(theta_name, theta_upper, theta_lower))
        theta_names['theta'] = theta_name
    else:
        cov_eff_number = int(re.findall(r'\d', theta_name)[0])

        for i in range(1, no_of_thetas+1):
            pset.add(Parameter(theta_name, theta_upper, theta_lower))
            theta_names[f'theta{i}'] = theta_name
            theta_name = f'COVEFF{cov_eff_number + i}'

    model.parameters = pset

    return theta_names


def count_categorical(model, covariate):
    data = model.dataset.groupby('ID')[covariate]
    counts = data.agg(lambda ids: ids.value_counts(dropna=False).index[0])

    return counts


def calculate_mean(df, covariate, baselines=False):
    if baselines:
        return df[str(covariate)].mean()
    else:
        return df.groupby('ID')[str(covariate)].mean().mean()


def calculate_median(df, covariate, baselines=False):
    if baselines:
        return df.pharmpy.baselines[str(covariate)].median()
    else:
        return df.groupby('ID')[str(covariate)].median().median()


def choose_param_inits(effect, df, covariate):
    lower_expected = 0.1
    upper_expected = 100
    if effect == 'exp':
        min_diff = df[str(covariate)].min() - calculate_median(df, covariate)
        max_diff = df[str(covariate)].max() - calculate_median(df, covariate)
        if min_diff == 0 or max_diff == 0:
            return lower_expected, upper_expected
        else:
            log_base = 10
            lower = max(math.log(lower_expected, log_base)/max_diff,
                        math.log(upper_expected, log_base)/min_diff)
            upper = min(math.log(lower_expected, log_base)/min_diff,
                        math.log(upper_expected, log_base)/max_diff)
            return lower, upper
    else:
        return lower_expected, upper_expected


def create_template(effect, model, covariate):
    if effect == 'lin_cont':
        return CovariateEffect.linear_continuous()
    elif effect == 'lin_cat':
        counts = count_categorical(model, covariate)
        return CovariateEffect.linear_categorical(counts)
    elif effect == 'piece_lin':
        return CovariateEffect.piecewise_linear()
    elif effect == 'exp':
        return CovariateEffect.exponential()
    elif effect == 'pow':
        return CovariateEffect.power()
    else:
        symbol = S('symbol')
        expression = sympify(effect)
        return CovariateEffect(Assignment(symbol, expression))


def S(x):
    return real(x)


class CovariateEffect:
    def __init__(self, template):
        self.template = template
        self.statistic_type = None

    def apply(self, parameter, covariate, thetas):
        effect_name = f'{parameter}{covariate}'
        self.template.symbol = S(effect_name)

        self.template.expression = subs(self.template.expression, thetas)
        self.template.subs({'cov': covariate})

        template_str = [str(symbol) for symbol in self.template.free_symbols]
        if 'mean' in template_str:
            self.template.subs({'mean': f'{parameter}_MEAN'})
            self.statistic_type = 'mean'
        elif 'median' in template_str:
            self.template.subs({'median': f'{parameter}_MEDIAN'})
            self.statistic_type = 'median'

    def create_effect_statement(self, operation_str, statement_original):
        operation = self._get_operation(operation_str)

        symbol = statement_original.symbol
        expression = statement_original.expression

        statement_new = Assignment(symbol, operation(expression, self.template.symbol))

        return statement_new

    def create_statistics_statement(self, parameter, mean, median):
        if self.statistic_type == 'mean':
            return Assignment(S(f'{parameter}_MEAN'), Float(mean, 6))
        else:
            return Assignment(S(f'{parameter}_MEDIAN'), Float(median, 6))

    @staticmethod
    def _get_operation(operation_str):
        if operation_str == '*':
            return mul
        elif operation_str == '+':
            return add

    @classmethod
    def linear_continuous(cls):
        symbol = S('symbol')
        expression = 1 + S('theta') * (S('cov') - S('median'))
        template = Assignment(symbol, expression)

        return cls(template)

    @classmethod
    def linear_categorical(cls, counts):
        symbol = S('symbol')
        most_common = counts[counts.idxmax()]
        categories = counts.unique()

        values = [1]

        if np.isnan(most_common):
            most_common = S('NaN')

        conditions = [Eq(S('cov'), most_common)]

        for i, cat in enumerate(categories):
            if cat != most_common:
                values += [1 + S(f'theta{i}')]
                if np.isnan(cat):
                    conditions += [Eq(S('cov'), S('NaN'))]
                else:
                    conditions += [Eq(S('cov'), cat)]

        expression = Piecewise(*zip(values, conditions))

        template = Assignment(symbol, expression)

        return cls(template)

    @classmethod
    def piecewise_linear(cls):
        symbol = S('symbol')
        values = [1 + S('theta1') * (S('cov') - S('median')),
                  1 + S('theta2') * (S('cov') - S('median'))]
        conditions = [Le(S('cov'), S('median')),
                      Gt(S('cov'), S('median'))]
        expression = Piecewise((values[0], conditions[0]),
                               (values[1], conditions[1]))

        template = Assignment(symbol, expression)

        return cls(template)

    @classmethod
    def exponential(cls):
        symbol = S('symbol')
        expression = exp(S('theta') * (S('cov') - S('median')))
        template = Assignment(symbol, expression)

        return cls(template)

    @classmethod
    def power(cls):
        symbol = S('symbol')
        expression = (S('cov')/S('median'))**S('theta')
        template = Assignment(symbol, expression)

        return cls(template)

    def __str__(self):
        return str(self.template)
