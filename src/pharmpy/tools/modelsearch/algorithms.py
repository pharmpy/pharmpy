import numpy as np
import pandas as pd

from pharmpy.modeling import update_inits
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


def exhaustive_stepwise(base_model, mfl, wf_run):
    # TODO: Base condition/warning for input model?
    wf_search = Workflow(Task('start_model', return_model, base_model))
    # TODO: Consider using a __bool__ in modelfit_results
    if not (base_model.modelfit_results.ofv and base_model.modelfit_results.parameter_estimates):
        wf_search.add_tasks(wf_run.copy(new_ids=True), connect=True)

    while True:
        no_of_trans = 0
        for task in wf_search.get_output():
            previous_funcs = [task.function for task in wf_search.get_upstream_tasks(task)]
            possible_funcs = {
                feat: func for feat, func in mfl.all_funcs().items() if func not in previous_funcs
            }
            if len(possible_funcs) > 0:
                no_of_trans += 1
                task_update_inits = Task('update_inits', update_inits, task)
                wf_search.add_tasks(task_update_inits, connect=True, output_nodes=[task])
                for feat, func in possible_funcs.items():
                    task_copy = Task('copy', copy_model, feat)
                    wf_search.add_tasks(task_copy, connect=True, output_nodes=[task_update_inits])
                    wf_trans = create_workflow_transform(feat, func, wf_run.copy(new_ids=True))
                    wf_search.add_tasks(wf_trans, connect=True, output_nodes=[task_copy])
        if no_of_trans == 0:
            break

    task_result = Task('results', post_process_results, final_task=True)
    wf_search.add_tasks(task_result, connect=True)

    return wf_search


def post_process_results(models):
    return models


def return_model(model):
    return model


def create_workflow_transform(feat, func, wf_run):
    # TODO: add feature tracking
    wf_trans = Workflow()
    task_function = Task(feat, func)  # TODO: check how partial functions work w.r.t. dask
    wf_trans.add_tasks(task_function, connect=False)
    wf_trans.add_tasks(wf_run, connect=True)
    return wf_trans


def copy_model(model, feat):
    model_copy = model.copy()
    model_copy.name = f'model_{feat}'
    return model_copy
