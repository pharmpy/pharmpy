import math
from operator import add, mul

from sympy import Eq, Piecewise, Symbol, exp

from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment


def add_covariate_effect(model, parameter, covariate, effect, operation='*'):
    mean = calculate_mean(model.dataset, covariate)
    median = calculate_median(model.dataset, covariate)

    theta_name = str(model.create_symbol(stem='COVEFF', force_numbering=True))
    theta_lower, theta_upper = choose_param_inits(effect, model.dataset, covariate)

    pset = model.parameters
    pset.add(Parameter(theta_name, theta_upper, theta_lower))
    model.parameters = pset

    sset = model.statements
    param_statement = sset.find_assignment(parameter)

    covariate_effect = create_template(effect)
    covariate_effect.apply(parameter, covariate, theta_name)
    statistic_statement = covariate_effect.create_statistics_statement(parameter, mean, median)
    effect_statement = covariate_effect.create_effect_statement(operation, param_statement)

    param_index = sset.index(param_statement)
    sset.insert(param_index + 1, covariate_effect.template)
    sset.insert(param_index + 2, statistic_statement)
    sset.insert(param_index + 3, effect_statement)

    model.statements = sset

    return model


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


def create_template(effect):
    if effect == 'lin_cont':
        return CovariateEffect.linear_continuous()
    elif effect == 'lin_cat':
        return CovariateEffect.linear_categorical()
    elif effect == 'exp':
        return CovariateEffect.exponential()
    elif effect == 'pow':
        return CovariateEffect.power()


def S(x):
    return Symbol(x, real=True)


class CovariateEffect:
    def __init__(self, template):
        self.template = template
        self.statistic_type = None

    def apply(self, parameter, covariate, theta_name):
        effect_name = f'{parameter}{covariate}'
        self.template.symbol = S(effect_name)

        self.template.subs(S('theta'), S(theta_name))
        self.template.subs(S('cov'), S(covariate))

        template_str = [str(symbol) for symbol in self.template.free_symbols]
        if 'mean' in template_str:
            self.template.subs(S('mean'), S(f'{parameter}_MEAN'))
            self.statistic_type = 'mean'
        elif 'median' in template_str:
            self.template.subs(S('median'), S(f'{parameter}_MEDIAN'))
            self.statistic_type = 'median'

    def create_effect_statement(self, operation_str, statement_original):
        operation = self._get_operation(operation_str)

        symbol = statement_original.symbol
        expression = statement_original.expression

        statement_new = Assignment(symbol, operation(expression, self.template.symbol))

        return statement_new

    def create_statistics_statement(self, parameter, mean, median):
        if self.statistic_type == 'mean':
            return Assignment(S(f'{parameter}_MEAN'), mean)
        else:
            return Assignment(S(f'{parameter}_MEDIAN'), median)

    @staticmethod
    def _get_operation(operation_str):
        if operation_str == '*':
            return mul
        elif operation_str == '+':
            return add

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

    @classmethod
    def linear_continuous(cls):
        symbol = S('symbol')
        expression = 1 + S('theta') * (S('cov') - S('median'))
        template = Assignment(symbol, expression)

        return cls(template)

    @classmethod
    def linear_categorical(cls):
        symbol = S('symbol')
        expression = Piecewise((1, Eq(S('cov'), 1)),
                               (1 + S('theta'), Eq(S('cov'), 0)), evaluate=False)
        template = Assignment(symbol, expression)

        return cls(template)

    def __str__(self):
        return str(self.template)
