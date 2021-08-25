import pharmpy.model
import pharmpy.results
import pharmpy.tools.common
import pharmpy.tools.modelfit


# TODO: elaborate documentation
def fit(models):
    """Fit models.

    Parameters
    ----------
    models : list
        List of models
    """
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


# TODO: elaborate documentation
def create_results(path, **kwargs):
    """Create results object

    Parameters
    ----------
    path : str, Path
        Path to run directory
    """
    res = pharmpy.tools.common.create_results(path, **kwargs)
    return res


# TODO: elaborate documentation
def read_results(path):
    """Read results object

    Parameters
    ----------
    path : str, Path
        Path to results file
    """
    res = pharmpy.results.read_results(path)
    return res
