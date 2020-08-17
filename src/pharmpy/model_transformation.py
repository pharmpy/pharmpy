from pharmpy.covariate_effect import CovariateEffect
from pharmpy.parameter import Parameter


def add_covariate_effect(model, parameter, covariate, effect, operation='*'):
    covariate_effect = create_template(effect)

    mean = get_baselines(model, str(covariate)).mean()
    median = get_baselines(model, str(covariate)).median()

    theta_name = f'THETA({model.get_next_theta()})'

    pset = model.parameters
    pset.add(Parameter(theta_name, 0.1))
    model.parameters = pset

    sset = model.get_pred_pk_record().statements
    p_statement = sset.get_statement(parameter)

    covariate_effect.apply(parameter, covariate, theta_name, mean, median)
    effect_statement = covariate_effect.create_effect_statement(operation, p_statement)

    sset.append(covariate_effect.template)
    sset.append(effect_statement)
    model.statements = sset

    return model


def create_template(effect):
    if effect == 'exp':
        return CovariateEffect.exponential()
    elif effect == 'pow':
        return CovariateEffect.power()


def get_baselines(model, column_name):           # TODO: Remove
    return model.dataset.pharmpy.baselines[column_name]
