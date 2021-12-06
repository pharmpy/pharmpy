import numpy as np
import pandas as pd

import pharmpy.tools.rankfuncs as rankfuncs
import pharmpy.workflows as workflows

from .psn_helpers import tool_name


class Tool:
    def __init__(self, dispatcher=None, database=None, path=None):
        toolname = type(self).__name__.lower()
        if dispatcher is None:
            self.dispatcher = workflows.default_dispatcher
        else:
            self.dispatcher = dispatcher
        if database is None:
            self.database = workflows.default_tool_database(toolname=toolname, path=path)
        else:
            self.database = database


def create_results(path, **kwargs):
    name = tool_name(path)
    # FIXME: Do something automatic here
    if name == 'qa':
        from pharmpy.tools.qa.results import psn_qa_results

        res = psn_qa_results(path, **kwargs)
    elif name == 'bootstrap':
        from pharmpy.tools.bootstrap.results import psn_bootstrap_results

        res = psn_bootstrap_results(path, **kwargs)
    elif name == 'cdd':
        from pharmpy.tools.cdd.results import psn_cdd_results

        res = psn_cdd_results(path, **kwargs)
    elif name == 'frem':
        from pharmpy.tools.frem.results import psn_frem_results

        res = psn_frem_results(path, **kwargs)
    elif name == 'linearize':
        from pharmpy.tools.linearize.results import psn_linearize_results

        res = psn_linearize_results(path, **kwargs)
    elif name == 'resmod':
        from pharmpy.tools.resmod.results import psn_resmod_results

        res = psn_resmod_results(path, **kwargs)

    elif name == 'scm':
        from pharmpy.tools.scm.results import psn_scm_results

        res = psn_scm_results(path, **kwargs)
    elif name == 'simeval':
        from pharmpy.tools.simeval.results import psn_simeval_results

        res = psn_simeval_results(path, **kwargs)
    elif name == 'crossval':
        from pharmpy.tools.crossval.results import psn_crossval_results

        res = psn_crossval_results(path, **kwargs)
    else:
        raise ValueError("Not a valid run directory")
    return res


def create_summary(models, start_model, rankfunc, cutoff, model_features):
    rankfunc = getattr(rankfuncs, rankfunc)

    res_data = {'dofv': [], 'features': [], 'rank': []}
    model_names = []

    if cutoff is not None:
        ranks = rankfunc(start_model, models, cutoff=cutoff)
    else:
        ranks = rankfunc(start_model, models)

    for model in models:
        model_names.append(model.name)
        try:
            res_data['dofv'].append(start_model.modelfit_results.ofv - model.modelfit_results.ofv)
        except AttributeError:
            res_data['dofv'].append(np.nan)
        res_data['features'].append(model_features[model.name])
        if model in ranks:
            res_data['rank'].append(ranks.index(model) + 1)
        else:
            res_data['rank'].append(np.nan)

    # FIXME: in ranks, if any row has NaN the rank converts to float
    return pd.DataFrame(res_data, index=model_names)
