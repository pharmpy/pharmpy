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


def exhaustive_stepwise(mfl):
    features = ModelFeatures(mfl)
    wf_search = Workflow()

    model_tasks = []
    model_features = dict()

    candidate_count = 1

    while True:
        no_of_trans = 0

        if wf_search.output_tasks:
            actions = {
                task: _find_possible_trans(wf_search, task, features)
                for task in wf_search.output_tasks
            }
        else:
            actions = {'': (dict(), features.all_funcs())}

        for task, trans in actions.items():
            trans_previous, trans_possible = trans

            for feat, func in trans_possible.items():
                model_name = f'candidate{candidate_count}'

                task_copy = Task('copy', copy, model_name)
                if task:
                    wf_search.add_task(task_copy, predecessors=[task])
                else:
                    wf_search.add_task(task_copy)

                task_update_inits = Task('update_inits', update_inits)
                wf_search.add_task(task_update_inits, predecessors=task_copy)

                task_function = Task(feat, func)
                wf_search.add_task(task_function, predecessors=task_update_inits)

                task_update_source = Task('update_source', update_source)
                wf_search.add_task(task_update_source, predecessors=task_function)

                wf_fit = create_single_fit_workflow()
                wf_search.insert_workflow(wf_fit, predecessors=task_update_source)

                model_tasks += wf_fit.output_tasks
                model_features[model_name] = tuple(list(trans_previous.keys()) + [feat])

                candidate_count += 1
                no_of_trans += 1
        if no_of_trans == 0:
            break

    return wf_search, model_tasks, model_features


def _find_possible_trans(wf, task, features):
    funcs = features.all_funcs()
    trans_previous = {
        task.name: task.function
        for task in wf.get_upstream_tasks(task)
        if task.function in funcs.values()
    }

    trans_possible = {
        feat: func
        for feat, func in funcs.items()
        if _is_allowed(feat, func, trans_previous, features)
    }

    return trans_previous, trans_possible


def _is_allowed(feat_current, func_current, trans_previous, features):
    if func_current in trans_previous.values():
        return False
    if any(func in features.get_funcs_same_type(feat_current) for func in trans_previous.values()):
        return False
    not_supported_combo = {'ABSORPTION(ZO)': 'TRANSITS(1)'}
    for key, value in not_supported_combo.items():
        if any(feat_current == key and feat == value for feat in trans_previous.keys()):
            return False
    return True


def copy(name, model):
    model_copy = copy_model(model, name)
    return model_copy


def update_source(model):
    model.update_source(nofiles=True)
    return model
