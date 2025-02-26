import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import set_instantaneous_absorption
from pharmpy.tools import read_results
from pharmpy.workflows import (
    Results,
    Task,
    Workflow,
    WorkflowBuilder,
    execute_subtool,
    execute_workflow,
)
from pharmpy.workflows.contexts import NullContext
from pharmpy.workflows.dispatchers.local_dask import LocalDaskDispatcher
from pharmpy.workflows.dispatchers.local_serial import LocalSerialDispatcher
from pharmpy.workflows.results import ModelfitResults

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
    wb = WorkflowBuilder(tasks=[t1], name='test-workflow')
    wf = Workflow(wb)

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
    wb = WorkflowBuilder(tasks=[t1], name='test-workflow')
    wb.add_task(t2, predecessors=[t1])
    wf = Workflow(wb)

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
    wb = WorkflowBuilder(tasks=[t1, t2], name='test-workflow')
    wb.add_task(t3, predecessors=[t1, t2])
    wf = Workflow(wb)

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
    wb = WorkflowBuilder(tasks=layer_init, name='test-workflow')
    wb.insert_workflow(WorkflowBuilder(tasks=layer_map))
    wb.insert_workflow(WorkflowBuilder(tasks=layer_reduce))
    wf = Workflow(wb)

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res == sum(map(f, range(n)))


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_set_instantaneous_absorption(load_model_for_test, testdata, tmp_path):
    model1 = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    model2 = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    advan1_before = model1.code

    t1 = Task('init', lambda x: x, model2)
    t2 = Task('update', set_instantaneous_absorption)
    t3 = Task('postprocess', lambda x: x)
    wb = WorkflowBuilder(tasks=[t1], name='test-workflow')
    wb.insert_workflow(WorkflowBuilder(tasks=[t2]))
    wb.insert_workflow(WorkflowBuilder(tasks=[t3]))
    wf = Workflow(wb)

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res.code == advan1_before


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_fit_mock(load_model_for_test, testdata, tmp_path):
    models = (
        load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod'),
        load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod'),
    )
    indices = range(len(models))
    ofvs = [(-17 + x) ** 2 - x + 3 for x in indices]

    def fit(ofv, m):
        res = ModelfitResults(ofv=ofv)
        return res

    init = map(lambda i: Task(f'init_{i}', lambda x: x, models[i]), indices)
    process = map(lambda i: Task(f'fit{i}', fit, ofvs[i]), indices)
    wb = WorkflowBuilder(tasks=init, name='test-workflow')
    wb.insert_workflow(WorkflowBuilder(tasks=process))
    gather = Task('gather', lambda *x: x)
    wb.insert_workflow(WorkflowBuilder(tasks=[gather]))
    wf = Workflow(wb)

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    for modelres, ofv in zip(res, ofvs):
        assert modelres.ofv == ofv


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_results(tmp_path):
    ofv = 3
    mfr = ModelfitResults(ofv=ofv)

    wb = WorkflowBuilder(tasks=[Task('result', lambda: mfr)], name='test-workflow')
    wf = Workflow(wb)

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res.ofv == ofv
    assert not hasattr(res, 'tool_database')


@dataclass(frozen=True)
class MyResults(Results):
    ofv: Optional[float] = None


@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_results_with_tool_database(tmp_path):
    ofv = 3
    mfr = MyResults(ofv=ofv)

    wb = WorkflowBuilder(tasks=[Task('result', lambda: mfr)], name='test-workflow')
    wf = Workflow(wb)

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            res = execute_workflow(wf)

    assert res.ofv == ofv


@pytest.mark.parametrize(
    'path',
    [
        Path('frem') / 'results.json',
        Path('results') / 'modelsearch_results.json',
    ],
)
@pytest.mark.xdist_group(name="workflow")
def test_execute_workflow_results_with_report(testdata, tmp_path, path):
    mfr = read_results(testdata / path)

    wb = WorkflowBuilder(tasks=[Task('result', lambda: mfr)], name='test-workflow')
    wf = Workflow(wb)

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            execute_workflow(wf)
        html = Path.cwd() / 'test-workflow' / 'results.html'
        assert html.is_file()
        assert html.stat().st_size > 500000


@pytest.mark.xdist_group(name="workflow")
def test_local_dispatcher():
    wb = WorkflowBuilder(tasks=[Task('results', lambda x: x, 'input')])
    wf = Workflow(wb)
    ctx = NullContext()
    res = LocalDaskDispatcher().run(wf, ctx)
    assert res == 'input'


@pytest.mark.xdist_group(name="workflow")
def test_serial_dispatcher_n_in_to_1_out(tmp_path):
    a = lambda: 1  # noqa E731
    b = lambda: 2  # noqa E731
    f = lambda x, y: x + y  # noqa E731
    t1 = Task('t1', a)
    t2 = Task('t2', b)
    t3 = Task('t3', f)
    wb = WorkflowBuilder(tasks=[t1, t2], name='test-workflow')
    wb.add_task(t3, predecessors=[t1, t2])
    wf = Workflow(wb)

    ctx = NullContext()
    res = LocalSerialDispatcher().run(wf, ctx)

    assert res == f(a(), b())


@pytest.mark.xdist_group(name="workflow")
def test_serial_dispatcher_1_in_to_1_out(tmp_path):
    start = lambda: 1  # noqa E731
    a = lambda x: x  # noqa E731
    b = lambda y: y + 1  # noqa E731
    f = lambda x, y: x + y  # noqa E731
    t0 = Task('t0', start)
    t1 = Task('t1', a)
    t2 = Task('t2', b)
    t3 = Task('t3', f)
    wb = WorkflowBuilder(tasks=[t0], name='test-workflow')
    wb.add_task(t1, predecessors=[t0])
    wb.add_task(t2, predecessors=[t0])
    wb.add_task(t3, predecessors=[t1, t2])
    wf = Workflow(wb)

    ctx = NullContext()
    res = LocalSerialDispatcher().run(wf, ctx)

    assert res == 3

    c = lambda z: z + 2  # noqa E731
    g = lambda x, y: x + y  # noqa E731
    t4 = Task('t4', c)
    t5 = Task('t5', g)
    wb.add_task(t4, predecessors=[t0])
    wb.add_task(t5, predecessors=[t3, t4])
    wf = Workflow(wb)

    res = LocalSerialDispatcher().run(wf, ctx)

    assert res == 6


@pytest.mark.xdist_group(name="workflow")
def test_execute_with_subtool(tmp_path):
    a = lambda: 1  # noqa E731
    t1 = Task('t1', a)
    wb = WorkflowBuilder(tasks=[t1], name='test-tool')

    def _run_subtool(context, x):
        g = lambda: x + 1  # noqa E731
        t = Task('t', g)
        wb = WorkflowBuilder(tasks=[t], name='test-subtool')
        wf = Workflow(wb)
        return execute_subtool(wf, context)

    t2 = Task('t2', _run_subtool)
    wb.add_task(t2, predecessors=[t1])
    wf = Workflow(wb)

    with chdir(tmp_path):
        with warnings.catch_warnings():
            ignore_scratch_warning()
            ctx = NullContext(dispatcher=LocalDaskDispatcher())
            res = execute_workflow(wf, context=ctx)
            assert res == 2
