from pharmpy.covariate_effect import CovariateEffect
from pharmpy.parameter import Parameter


def add_covariate_effect(model, parameter, covariate, effect, operation='*'):
    covariate_effect = create_template(effect)

    baselines = model.dataset.pharmpy.baselines[str(covariate)]

    mean = baselines.mean()
    median = baselines.median()

    theta_name = f'THETA({model.get_next_theta()})'

    pset = model.parameters
    pset.add(Parameter(theta_name, 0.1))
    model.parameters = pset

    sset = model.statements
    p_statement = sset.find_assignment(parameter)

    covariate_effect.apply(parameter, covariate, theta_name, mean, median)
    effect_statement = covariate_effect.create_effect_statement(operation, p_statement)

    p_index = sset.index(p_statement)

    sset.insert(p_index + 1, covariate_effect.template)
    sset.insert(p_index + 2, effect_statement)

    model.statements = sset

    return model


def create_template(effect):
    if effect == 'exp':
        return CovariateEffect.exponential()
    elif effect == 'pow':
        return CovariateEffect.power()
