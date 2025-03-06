import os

import pytest

from pharmpy.workflows.dispatchers.slurm_helpers import (
    get_slurm_corelist,
    get_slurm_nodelist,
    is_running_on_slurm,
)


def test_is_running_on_slurm():
    os.environ.pop('SLURM_JOB_ID', None)
    assert not is_running_on_slurm()
    os.environ['SLURM_JOB_ID'] = "323232"
    assert is_running_on_slurm()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("b23", ["b23"]),
        (
            "compute-b24-[1-3,5-9],compute-b25-[1,4,8]",
            [
                "compute-b24-1",
                "compute-b24-2",
                "compute-b24-3",
                "compute-b24-5",
                "compute-b24-6",
                "compute-b24-7",
                "compute-b24-8",
                "compute-b24-9",
                "compute-b25-1",
                "compute-b25-4",
                "compute-b25-8",
            ],
        ),
    ],
)
def test_get_slurm_nodelist(value, expected):
    os.environ['SLURM_JOB_NODELIST'] = value
    assert get_slurm_nodelist() == expected


@pytest.mark.parametrize(
    "tasks_per_node,cpus_per_task,expected",
    [
        ("1", "1", [1]),
        ("2", None, [2]),
        ("2(x3)", "1", [2, 2, 2]),
    ],
)
def test_get_slurm_corelist(tasks_per_node, cpus_per_task, expected):
    if tasks_per_node is None:
        os.environ.pop('SLURM_TASKS_PER_NODE', None)
    else:
        os.environ['SLURM_TASKS_PER_NODE'] = tasks_per_node
    if cpus_per_task is None:
        os.environ.pop('SLURM_CPUS_PER_TASK', None)
    else:
        os.environ['SLURM_CPUS_PER_TASK'] = cpus_per_task
    assert get_slurm_corelist() == expected
