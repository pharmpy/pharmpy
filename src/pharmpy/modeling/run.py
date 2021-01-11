import pharmpy.model
from pharmpy.methods.modelfit import Modelfit


def fit(models):
    if isinstance(models, pharmpy.model.Model):
        models = [models]
        single = True
    else:
        single = False
    tool = Modelfit(models)
    tool.run()
    if single:
        return models[0]
    else:
        return models
