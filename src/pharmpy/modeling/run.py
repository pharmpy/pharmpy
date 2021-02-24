import pharmpy.methods.modelfit
import pharmpy.model


def fit(models):
    if isinstance(models, pharmpy.model.Model):
        models = [models]
        single = True
    else:
        single = False
    tool = pharmpy.methods.modelfit.Modelfit(models)
    tool.run()
    if single:
        return models[0]
    else:
        return models
