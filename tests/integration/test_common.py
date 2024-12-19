from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import read_results, run_modelsearch


def test_serialization(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        res = run_modelsearch(
            model=start_modelres[0],
            results=start_modelres[1],
            search_space='ABSORPTION(ZO)',
            algorithm='exhaustive',
        )

        rundir = tmp_path / 'modelsearch1'

        res_serialized = read_results(rundir / 'results.json')

        # This is a temporary workaround since floats are serialized with 15 precision, should be 17 for correct
        # round-trip
        cols_float = res.summary_tool.select_dtypes(include=['float64']).columns
        cols_round = {col: 15 for col in cols_float}
        summary_tool_rounded = res.summary_tool.round(cols_round)
        summary_tool_serialized_rounded = res_serialized.summary_tool.round(cols_round)

        assert summary_tool_rounded.equals(summary_tool_serialized_rounded)
