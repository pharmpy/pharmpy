import networkx as nx
import pytest

from pharmpy import Model
from pharmpy.tools.modelsearch.algorithms import (
    create_workflow_transform,
    exhaustive,
    exhaustive_stepwise,
)
from pharmpy.tools.modelsearch.mfl import ModelFeatures
from pharmpy.tools.modelsearch.rankfuncs import aic, bic, ofv
from pharmpy.tools.workflows import Task, Workflow


class DummyResults:
    def __init__(self, ofv=None, aic=None, bic=None, parameter_estimates=None):
        self.ofv = ofv
        self.aic = aic
        self.bic = bic
        self.parameter_estimates = parameter_estimates


class DummyModel:
    def __init__(self, name, **kwargs):
        self.name = name
        self.modelfit_results = DummyResults(**kwargs)


@pytest.fixture
def wf_run():
    def run(model):
        return model

    return Workflow(Task('run', run))


def test_ofv():
    run1 = DummyModel("run1", ofv=0)
    run2 = DummyModel("run2", ofv=-1)
    run3 = DummyModel("run3", ofv=-14)
    res = ofv(run1, [run2, run3])
    assert [run3] == res

    run4 = DummyModel("run4", ofv=2)
    run5 = DummyModel("run5", ofv=-2)
    res = ofv(run1, [run2, run3, run4, run5], cutoff=2)
    assert [run3, run5] == res


def test_aic():
    run1 = DummyModel("run1", aic=0)
    run2 = DummyModel("run2", aic=-1)
    run3 = DummyModel("run3", aic=-14)
    res = aic(run1, [run2, run3])
    assert [run3] == res


def test_bic():
    run1 = DummyModel("run1", bic=0)
    run2 = DummyModel("run2", bic=-1)
    run3 = DummyModel("run3", bic=-14)
    res = bic(run1, [run2, run3])
    assert [run3] == res


def test_exhaustive(testdata):
    base = Model(testdata / 'nonmem' / 'pheno.mod')

    def do_nothing(model):
        return model

    trans = 'ABSORPTION(FO)'
    res = exhaustive(base, trans, do_nothing, ofv)
    assert len(res) == 1

    res = exhaustive(base, trans, do_nothing, ofv)
    assert len(res) == 1
    assert list(res['dofv']) == [0.0]


@pytest.mark.parametrize(
    'res, mfl, task_names_ref',
    [
        (
            (1, True),
            'ELIMINATION(ZO)\nPERIPHERALS(2)',
            [
                'start_model',
                'update_inits',
                'copy',
                'copy',
                'ELIMINATION(ZO)',
                'PERIPHERALS(2)',
                'run',
            ],
        ),
        (
            (1, True),
            'ELIMINATION(ZO)\nPERIPHERALS([2,3])',
            [
                'start_model',
                'update_inits',
                'copy',
                'copy',
                'copy',
                'ELIMINATION(ZO)',
                'PERIPHERALS(2)',
                'PERIPHERALS(3)',
                'run',
            ],
        ),
        (
            (1, True),
            'ELIMINATION(ZO)\nPERIPHERALS(2)\nABSORPTION(ZO)',
            [
                'start_model',
                'update_inits',
                'copy',
                'copy',
                'copy',
                'ABSORPTION(ZO)',
                'ELIMINATION(ZO)',
                'PERIPHERALS(2)',
                'run',
            ],
        ),
        (
            (None, None),
            'ELIMINATION(ZO)\nPERIPHERALS(2)',
            [
                'start_model',
                'run',
                'update_inits',
                'copy',
                'copy',
                'ELIMINATION(ZO)',
                'PERIPHERALS(2)',
                'run',
            ],
        ),
    ],
)
def test_exhaustive_stepwise(wf_run, res, mfl, task_names_ref):
    base_model = DummyModel('run1', ofv=res[0], parameter_estimates=res[1])
    mfl = ModelFeatures(mfl)
    wf_search = exhaustive_stepwise(base_model, mfl, wf_run)
    start_node = wf_search.get_input()[0]
    start_node_successors = list(wf_search.tasks.successors(start_node))
    assert len(start_node_successors) == 1
    bfs_node_names = [task.name for task in nx.bfs_tree(wf_search.tasks, start_node)]
    assert bfs_node_names[: len(task_names_ref)] == task_names_ref


def test_create_workflow_transform(wf_run):
    def transform_nothing(model):
        return model

    wf = create_workflow_transform('transform', transform_nothing, wf_run)
    assert len(wf.tasks.nodes) == 2
