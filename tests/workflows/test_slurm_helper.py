import os

import pytest

from pharmpy.workflows.dispatchers.slurm_helpers import (
    get_slurm_corelist,
    get_slurm_nodelist,
    get_slurm_nodename,
    is_running_on_slurm,
)


def test_is_running_on_slurm():
    os.environ.pop('SLURM_JOB_ID', None)
    assert not is_running_on_slurm()
    os.environ['SLURM_JOB_ID'] = "323232"
    assert is_running_on_slurm()


def test_get_slurm_nodename():
    os.environ.pop('SLURMD_NODENAME', None)
    assert not get_slurm_nodename()
    os.environ['SLURMD_NODENAME'] = 'nodename'
    assert get_slurm_nodename() == 'nodename'


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
    "cpus_per_node,expected",
    [
        ("1", [1]),
        ("2(x3)", [2, 2, 2]),
    ],
)
def test_get_slurm_corelist(cpus_per_node, expected):
    os.environ['SLURM_JOB_CPUS_PER_NODE'] = cpus_per_node
    assert get_slurm_corelist() == expected
