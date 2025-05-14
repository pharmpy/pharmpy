import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools.context import open_context
from pharmpy.tools.run import run_tool


@pytest.mark.parametrize(
    'tool_name, kwargs',
    [
        (
            'modelsearch',
            {
                'search_space': 'ABSORPTION([FO,ZO])',
                'algorithm': 'exhaustive',
            },
        ),
    ],
)
def test_resume_finished(
    tmp_path,
    start_modelres_dummy,
    tool_name,
    kwargs,
):
    with chdir(tmp_path):
        ctx = open_context('ctx1', tmp_path)

        kwargs.update(
            {
                'model': start_modelres_dummy[0],
                'results': start_modelres_dummy[1],
                'esttool': 'dummy',
                'context': ctx,
            }
        )

        res = run_tool(tool_name, **kwargs)

        res_resume = run_tool(tool_name, **kwargs)

        assert res.summary_tool.equals(res_resume.summary_tool)
        assert res.final_model == res_resume.final_model
