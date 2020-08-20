from operator import add, mul

from sympy import Eq, Piecewise, Symbol, exp

from pharmpy.statements import Assignment


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
        else:
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
