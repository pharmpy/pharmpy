import numpy as np
import pandas as pd

from pharmpy.modeling import copy_model, update_inits
from pharmpy.tools.modelfit import create_single_fit_workflow
from pharmpy.tools.workflows import Task, Workflow

from .mfl import ModelFeatures


def exhaustive(base_model, mfl, run_func, rank_func):
    features = ModelFeatures(mfl)
    torun = []
    combinations = list(features.all_combinations())
    df = pd.DataFrame(
        index=pd.RangeIndex(stop=len(combinations)),
        columns=['features', 'dofv', 'rank'],
    )
    funcs = features.all_funcs()
    for i, combo in enumerate(combinations):
        model = base_model.copy()
        model.name = f'candidate{i}'
        for feat in combo:
            funcs[feat](model)
        df.loc[i]['features'] = tuple(combo)
        torun.append(model)
    run_func(torun)
    for i, model in enumerate(torun):
        df.loc[i]['dofv'] = base_model.modelfit_results.ofv - model.modelfit_results.ofv
    ranks = rank_func(base_model, torun)
    for i, ranked_model in enumerate(ranks):
        idx = torun.index(ranked_model)
        df.loc[idx]['rank'] = i + 1
    df = df.astype({'rank': 'Int64'})
    return df


def stepwise(base_model, mfl, run_func, rank_func):
    features = ModelFeatures(mfl)
    remaining = features.all_funcs()
    start_model = base_model
    current_features = []
    features_col = []
    dofv_col = []
    step = 1
    while True:
        torun = []
        for feat, func in remaining.items():
            model = start_model.copy()
            model.name = f'step_{step}_{feat.replace("(", "_").replace(")", "")}'
            func(model)
            torun.append(model)
            features_col.append(tuple(current_features + [feat]))
        print(f'Running step {step}')
        run_func(torun)
        for model in torun:
            dofv = start_model.modelfit_results.ofv - model.modelfit_results.ofv
            dofv_col.append(dofv)
        ranks = rank_func(start_model, torun)
        if not ranks:
            break
        start_model = ranks[0]
        start_model.update_inits()
        idx = torun.index(start_model)
        selected_feature = list(remaining.keys())[idx]
        current_features.append(selected_feature)
        remaining = features.next_funcs(current_features)
        step += 1
    df = pd.DataFrame({'features': features_col, 'dofv': dofv_col, 'rank': np.nan})
    best_features = tuple(current_features)
    best_df_index = features_col.index(best_features)
    df.at[best_df_index, 'rank'] = 1
    df = df.astype({'rank': 'Int64'})
    return df


def exhaustive_stepwise(base_model, mfl):
    features = ModelFeatures(mfl)
    # TODO: Base condition/warning for input model?
    wf_search = Workflow()
    models_transformed = []

    if not base_model.modelfit_results:
        wf_fit = create_single_fit_workflow(base_model)
        wf_search.insert_workflow(wf_fit)
        models_transformed += wf_search.output_tasks

    while True:
        no_of_trans = 0
        for task in wf_search.output_tasks:
            previous_funcs = [task.function for task in wf_search.get_upstream_tasks(task)]
            possible_funcs = {
                feat: func
                for feat, func in features.all_funcs().items()
                if func not in previous_funcs
            }
            if len(possible_funcs) > 0:
                no_of_trans += 1
                for feat, func in possible_funcs.items():
                    # Create tasks
                    task_copy = Task('copy', copy, feat)
                    task_update_inits = Task('update_inits', update_inits)
                    task_function = Task(
                        feat, func
                    )  # TODO: check how partial functions work w.r.t. dask
                    wf_fit = create_single_fit_workflow()

                    # Add tasks
                    wf_search.add_task(task_copy, predecessors=[task])
                    wf_search.add_task(task_update_inits, predecessors=[task_copy])
                    wf_search.add_task(task_function, predecessors=[task_update_inits])
                    wf_search.insert_workflow(wf_fit, predecessors=[task_function])
                    models_transformed += wf_search.output_tasks
        if no_of_trans == 0:
            break

    return wf_search, models_transformed


def copy(feat, model):
    model_copy = copy_model(model, f'{model.name}_{feat}')
    return model_copy
