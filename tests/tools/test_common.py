from pharmpy.deps import pandas as pd
from pharmpy.modeling import set_name
from pharmpy.tools.common import add_parent_column, concat_summaries, flatten_list
from pharmpy.workflows import ModelEntry


def test_flatten_list():
    assert flatten_list([['x'], 'y', [['z']]]) == ['x', 'y', 'z']


def test_concat_summaries():
    df1 = pd.DataFrame({'model': ['x', 'y'], 'col1': [0, 1], 'col2': ['a', 'b']}).set_index('model')
    df2 = pd.DataFrame({'model': ['y', 'z'], 'col1': [2, 3], 'col2': ['c', 'd']}).set_index('model')

    df_concat = concat_summaries([df1, df2], keys=[1, 2])
    assert list(df_concat.index.values) == [(1, 'x'), (1, 'y'), (2, 'y'), (2, 'z')]
    assert df_concat['col1'].values.tolist() == [0, 1, 2, 3]
    assert df_concat['col2'].values.tolist() == ['a', 'b', 'c', 'd']


def test_add_parent_column(load_model_for_test, pheno_path):
    model_ref = load_model_for_test(pheno_path)
    me_ref = ModelEntry.create(model=model_ref)
    models = [set_name(model_ref, f'model{i}') for i in range(1, 6)]
    mes = [me_ref]
    model_prev = model_ref
    for model in models:
        mes.append(ModelEntry.create(model, parent=model_prev))
        model_prev = model

    df_data = {'model': [me.model.name for me in mes], 'col1': [i for i in range(len(mes))]}
    df = pd.DataFrame(df_data).set_index('model')

    df_new = add_parent_column(df, mes)
    for me in mes:
        parent_from_df = df_new.loc[me.model.name, 'parent_model']
        if me.parent is not None:
            assert parent_from_df == me.parent.name
        else:
            assert parent_from_df == ''
