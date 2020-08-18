from pharmpy.covariate_effect import CovariateEffect
from pharmpy.parameter import Parameter


def add_covariate_effect(model, parameter, covariate, effect, operation='*'):
    baselines = model.dataset.pharmpy.baselines[str(covariate)]

    mean = baselines.mean()
    median = baselines.median()

    theta_name = str(model.create_symbol(stem='COVEFF', force_numbering=True))

    pset = model.parameters
    pset.add(Parameter(theta_name, 0.1))
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


def create_template(effect):
    if effect == 'exp':
        return CovariateEffect.exponential()
    elif effect == 'pow':
        return CovariateEffect.power()
