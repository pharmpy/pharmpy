import pandas as pd

from pharmpy.modeling import summarize_individuals_count_table


def test_summarize_individuals_count_table():
    df = pd.DataFrame(
        {
            'model': ['start_model'] * 3 + ['candidate1'] * 3 + ['candidate2'] * 3,
            'ID': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'parent_model': ['start_model'] * 6 + ['candidate1'] * 3,
            'outlier_count': [0, 0, 0, 1, 0, 2, 0, 0, 1],
            'ofv': [-1, -2, -3, -4, -5, -6, -7, -8, -9],
            'dofv_vs_parent': [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'predicted_dofv': [1.0, 1.5, 2.5, 3.0, 3.0, 4.0, 4.0, 2.0, 1.0],
            'predicted_residual': [1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    df.set_index(['model', 'ID'], inplace=True)

    res = summarize_individuals_count_table(df=df)
    assert list(res['inf_selection']) == [0, 0, 0]
    assert list(res['inf_params']) == [0, 1, 1]
    assert list(res['out_obs']) == [0, 2, 1]
    assert list(res['out_ind']) == [0, 2, 0]
    assert list(res['inf_outlier']) == [0, 1, 0]
