from .run import run_tool
from pharmpy.workflows import default_tool_database
from pharmpy.results import Results


class AMDResults(Results):
    def __init__(self, final_model=None):
        self.final_model = final_model


def run_amd(model):
    db = default_tool_database(toolname='amd')
    run_tool('modelfit', model, path=db.path / 'modelfit')

    mfl = 'LAGTIME()\nPERIPHERALS(1)'
    res_modelsearch = run_tool('modelsearch', 'exhaustive_stepwise', mfl=mfl, rankfunc='ofv', cutoff=3.84, model=model)
    selected_model = res_modelsearch.best_model

    res_iiv = run_tool('iiv', 'brute_force', rankfunc='ofv', cutoff=3.84, model=selected_model)
    selected_iiv_model = res_iiv.best_model

    res_resmod = run_tool('resmod', selected_iiv_model)
    final_model = res_resmod.best_model

    res = AMDResults(final_model=final_model)

    return res
