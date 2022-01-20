import numpy as np
import pandas as pd

import pharmpy.tools.rankfuncs as rankfuncs
import pharmpy.workflows as workflows


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
