from pharmpy.deps import pandas as pd
from pharmpy.tools.bootstrap.results import calculate_results
from pharmpy.workflows.results import ModelfitResults, read_results


def test_bootstrap():
    orig = ModelfitResults(ofv=110, parameter_estimates=pd.Series({'TVCL': 0.75, 'TVV': 2.25}))
    res1 = ModelfitResults(ofv=100, parameter_estimates=pd.Series({'TVCL': 1.0, 'TVV': 2.0}))
    res2 = ModelfitResults(ofv=120, parameter_estimates=pd.Series({'TVCL': 1.5, 'TVV': 3.0}))

    class MyModel:
        name = "noname"
        description = "nodescr"

    res1_mod = MyModel()
    res2_mod = MyModel()

    boot = calculate_results([res1_mod, res2_mod], [res1, res2], original_results=orig)
    correct_statistics = pd.DataFrame(
        {
            'mean': [1.25, 2.5],
            'median': [1.25, 2.50],
            'bias': [0.5, 0.25],
            'stderr': [0.353553, 0.707107],
            'RSE': [0.2828424, 0.2828424],
        },
        index=['TVCL', 'TVV'],
    )

    pd.testing.assert_frame_equal(boot.parameter_statistics, correct_statistics)

    # json round trip
    json = boot.to_json()
    obj = read_results(json)
    pd.testing.assert_frame_equal(obj.parameter_statistics, correct_statistics)


def test_read_results(testdata):
    read_results(testdata / 'results/bootstrap_results.json')
