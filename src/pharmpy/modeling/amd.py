import pandas as pd

from pharmpy.modeling import copy_model, remove_iiv
from pharmpy.results import Results
from pharmpy.tools.iiv.tool import IIVResults
from pharmpy.workflows import default_tool_database

from .run import run_tool


class AMDResults(Results):
    def __init__(self, final_model=None):
        self.final_model = final_model


def run_amd(model):
    """Run Automatic Model Development (AMD) tool

    Runs structural modelsearch, IIV building, and resmod

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> run_amd(model)      # doctest: +SKIP

    See also
    --------
    run_iiv
    run_tool

    """
    db = default_tool_database(toolname='amd')
    run_tool('modelfit', model, path=db.path / 'modelfit')

    mfl = 'LAGTIME()\nPERIPHERALS(1)'
    res_modelsearch = run_tool(
        'modelsearch', 'exhaustive_stepwise', mfl=mfl, rankfunc='ofv', cutoff=3.84, model=model
    )
    selected_model = res_modelsearch.best_model

    res_iiv = run_tool('iiv', 'brute_force', rankfunc='ofv', cutoff=3.84, model=selected_model)
    selected_iiv_model = res_iiv.best_model

    res_resmod = run_tool('resmod', selected_iiv_model)
    final_model = res_resmod.best_model

    res = AMDResults(final_model=final_model)

    return res


def run_iiv(model):
    """Run IIV tool

    Runs two IIV workflows: testing the number of etas and testing which block structure

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> run_iiv(model)      # doctest: +SKIP

    See also
    --------
    run_amd
    run_tool

    """
    res_no_of_etas = run_tool('iiv', 'brute_force_no_of_etas', model=model)

    if res_no_of_etas.best_model != model:
        best_model = res_no_of_etas.best_model
        best_model_eta_removed = copy_model(res_no_of_etas.best_model, 'iiv_no_of_etas_best_model')
        features = res_no_of_etas.summary.loc[best_model.name]['features']
        remove_iiv(best_model_eta_removed, features)
    else:
        best_model = model

    res_block_structure = run_tool('iiv', 'brute_force_block_structure', model=best_model)

    best_model = res_block_structure.best_model
    summary = pd.concat([res_no_of_etas.summary, res_block_structure.summary])
    res = IIVResults(
        summary=summary,
        best_model=best_model,
        models=res_no_of_etas.models + res_block_structure.models,
        start_model=model,
    )

    return res
