import math

from pharmpy.model_transformation.covariate_effect import CovariateEffect
from pharmpy.parameter import Parameter


def add_covariate_effect(model, parameter, covariate, effect, operation='*'):
    mean = calculate_mean(model.dataset, covariate)
    median = calculate_median(model.dataset, covariate)

    theta_name = str(model.create_symbol(stem='COVEFF', force_numbering=True))
    theta_upper, theta_lower = choose_param_inits(effect, model.dataset, covariate)

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
            return upper_expected, lower_expected
        else:
            upper = min(math.log(lower_expected)/min_diff,
                        math.log(upper_expected)/max_diff)
            lower = max(math.log(lower_expected)/max_diff,
                        math.log(upper_expected)/min_diff)
            return upper, lower
    else:
        return upper_expected, lower_expected


def create_template(effect):
    if effect == 'lin_cont':
        return CovariateEffect.linear_continuous()
    elif effect == 'lin_cat':
        return CovariateEffect.linear_categorical()
    elif effect == 'exp':
        return CovariateEffect.exponential()
    elif effect == 'pow':
        return CovariateEffect.power()
