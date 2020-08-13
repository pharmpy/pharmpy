import copy

from pharmpy.parameter import Parameter


class ModelTransformation:
    def __init__(self, model):
        self.model = copy.deepcopy(model)

    def add_covariate_effect(self, parameter, covariate, effect):
        covariate_effect = effect()

        mean = self.get_baselines(str(covariate)).mean()
        median = self.get_baselines(str(covariate)).median()

        pset = self.model.parameters
        p_name = f'THETA({self.model.get_next_theta()})'
        pset.add(Parameter(p_name, 0.1))
        self.model.parameters = pset

        resulting_statement = covariate_effect.apply(parameter, covariate, p_name, mean, median)

        sset = self.model.get_pred_pk_record().statements

        sset.append(covariate_effect.template)
        sset.append(resulting_statement)

        self.model.statements = sset

        self.model.update_source()

    def get_baselines(self, column_name):
        return self.model.dataset.pharmpy.baselines[column_name]

    def __str__(self):
        return str(self.model)
