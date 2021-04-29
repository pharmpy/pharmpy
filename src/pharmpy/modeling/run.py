import pharmpy.model
import pharmpy.results
import pharmpy.tools.common
import pharmpy.tools.modelfit


def fit(models):
    if isinstance(models, pharmpy.model.Model):
        models = [models]
        single = True
    else:
        single = False
    tool = pharmpy.tools.modelfit.Modelfit(models)
    tool.run()
    if single:
        return models[0]
    else:
        return models


def create_results(path, **kwargs):
    res = pharmpy.tools.common.create_results(path, **kwargs)
    return res


def read_results(path):
    res = pharmpy.results.read_results(path)
    return res
