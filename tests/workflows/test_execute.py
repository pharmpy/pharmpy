import warnings
from dataclasses import dataclass, replace
from typing import Optional

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Results
from pharmpy.modeling import set_bolus_absorption
from pharmpy.results import ModelfitResults
from pharmpy.tools import read_results
from pharmpy.workflows import Task, ToolDatabase, Workflow, execute_workflow, local_dask

# All workflow tests are run by the same xdist test worker
# This is to limit the number of sporadic failures on GHA on Windows
# The failures seem like races in dask distributed because the tmp-dir takes to long to create
# on C: (main drive is D:).


def ignore_scratch_warning():
    warnings.filterwarnings(
        "ignore",
        message=".*creating scratch directories is taking a surprisingly long time",
        category=UserWarning,
    )


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_constant(tmp_path):
    a = lambda: 1  # noqa E731
    t1 = Task('t1', a)
    wf = Workflow([t1], name='test-workflow')

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res == a()


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_unary(tmp_path):
    a = lambda: 2  # noqa E731
    f = lambda x: x**2  # noqa E731
    t1 = Task('t1', a)
    t2 = Task('t2', f)
    wf = Workflow([t1], name='test-workflow')
    wf.add_task(t2, predecessors=[t1])

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res == f(a())


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_binary(tmp_path):
    a = lambda: 1  # noqa E731
    b = lambda: 2  # noqa E731
    f = lambda x, y: x + y  # noqa E731
    t1 = Task('t1', a)
    t2 = Task('t2', b)
    t3 = Task('t3', f)
    wf = Workflow([t1, t2], name='test-workflow')
    wf.add_task(t3, predecessors=[t1, t2])

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res == f(a(), b())


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_map_reduce(tmp_path):
    n = 10
    f = lambda x: x**2  # noqa E731
    layer_init = list(map(lambda i: Task(f'x{i}', lambda: i), range(n)))
    layer_map = list(map(lambda i: Task(f'f(x{i})', f), range(n)))
    layer_reduce = [Task('reduce', lambda *y: sum(y))]
    wf = Workflow(layer_init, name='test-workflow')
    wf.insert_workflow(Workflow(layer_map))
    wf.insert_workflow(Workflow(layer_reduce))

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res == sum(map(f, range(n)))


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_set_bolus_absorption(load_model_for_test, testdata, tmp_path):
    model1 = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    model2 = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    advan1_before = model1.model_code

    t1 = Task('init', lambda x: x, model2)
    t2 = Task('update', set_bolus_absorption)
    t3 = Task('postprocess', lambda x: x)
    wf = Workflow([t1], name='test-workflow')
    wf.insert_workflow(Workflow([t2]))
    wf.insert_workflow(Workflow([t3]))

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res.model_code == advan1_before


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_fit_mock(load_model_for_test, testdata, tmp_path):
    models = (
        load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod'),
        load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod'),
    )
    indices = range(len(models))
    ofvs = [(-17 + x) ** 2 - x + 3 for x in indices]

    def fit(ofv, m):
        m = m.replace(modelfit_results=ModelfitResults(ofv=ofv))
        return m

    init = map(lambda i: Task(f'init_{i}', lambda x: x, models[i]), indices)
    process = map(lambda i: Task(f'fit{i}', fit, ofvs[i]), indices)
    wf = Workflow(init, name='test-workflow')
    wf.insert_workflow(Workflow(process))
    gather = Task('gather', lambda *x: x)
    wf.insert_workflow(Workflow([gather]))

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            execute_workflow(wf)

    # FIXME: These cannot be updated in place
    # for orig, fitted, ofv in zip(models, res, ofvs):
    #    assert orig.modelfit_results.ofv == ofv
    #    assert fitted.modelfit_results.ofv == ofv
    #    assert orig.modelfit_results == fitted.modelfit_results


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_results(tmp_path):
    ofv = 3
    mfr = ModelfitResults(ofv=ofv)

    wf = Workflow([Task('result', lambda: mfr)], name='test-workflow')

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res.ofv == ofv
    assert not hasattr(res, 'tool_database')


@dataclass(frozen=True)
class MyResults(Results):
    ofv: Optional[float] = None
    tool_database: Optional[ToolDatabase] = None


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_results_with_tool_database(tmp_path):
    ofv = 3
    mfr = MyResults(ofv=ofv)

    wf = Workflow([Task('result', lambda: mfr)], name='test-workflow')

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)
        assert res.tool_database is not None

    assert res.ofv == ofv


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_results_with_report(testdata, tmp_path):
    mfr = replace(read_results(testdata / 'frem' / 'results.json'), tool_database=None)

    wf = Workflow([Task('result', lambda: mfr)], name='test-workflow')

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)
        html = res.tool_database.path / 'results.html'
        assert html.is_file()
        assert html.stat().st_size > 500000


@pytest.mark.xdist_group(name="workflow")
def test_local_dispatcher():
    wf = Workflow([Task('results', lambda x: x, 'input')])
    res = local_dask.run(wf)
    assert res == 'input'
