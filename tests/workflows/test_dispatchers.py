import os

import pytest

from pharmpy.workflows.dispatchers import Dispatcher
from pharmpy.workflows.dispatchers.local_dask import LocalDaskDispatcher
from pharmpy.workflows.dispatchers.local_serial import LocalSerialDispatcher


@pytest.mark.parametrize(
    'name, ref',
    [('local_dask', 'local_dask'), ('LOCAL_DASK', 'local_dask')],
)
def test_canonicalize_dispatcher_name(name, ref):
    assert Dispatcher.canonicalize_dispatcher_name(name) == ref


def test_canonicalize_dispatcher_name_raises():
    with pytest.raises(ValueError):
        Dispatcher.canonicalize_dispatcher_name('x')


@pytest.mark.parametrize(
    'name, dispatcher_type',
    [
        ('local_dask', LocalDaskDispatcher),
        ('LOCAL_DASK', LocalDaskDispatcher),
        ('local_serial', LocalSerialDispatcher),
    ],
)
def test_select_dispatcher(name, dispatcher_type):
    assert isinstance(Dispatcher.select_dispatcher(name), dispatcher_type)


@pytest.mark.parametrize(
    'name, ncores, on_slurm, run_on_multiple_nodes, ncores_ref',
    [
        ('local_serial', None, False, False, os.cpu_count()),
        ('local_dask', None, False, False, os.cpu_count()),
        ('local_serial', 1, False, False, 1),
        ('local_dask', 1, False, False, 1),
        ('local_serial', 2, False, False, 2),
        ('local_dask', 2, False, False, 2),
        ('local_serial', 1, True, False, 1),
        ('local_dask', 1, True, False, 1),
        ('local_serial', None, True, True, 2),
        ('local_dask', None, True, True, os.cpu_count()),
    ],
)
def test_canonicalize_ncores(
    monkeypatch, name, ncores, on_slurm, run_on_multiple_nodes, ncores_ref
):
    dispatcher = Dispatcher.select_dispatcher(name)
    if on_slurm:
        monkeypatch.setenv('SLURM_JOB_ID', '323232')
    if run_on_multiple_nodes:
        monkeypatch.setenv('SLURM_JOB_NODELIST', 'node1,node2')
        monkeypatch.setenv('SLURM_JOB_CPUS_PER_NODE', '1,1')
    assert dispatcher.canonicalize_ncores(ncores) == ncores_ref


@pytest.mark.parametrize(
    'name, ncores, on_slurm',
    [
        ('local_dask', 2, True),
    ],
)
def test_canonicalize_ncores_raises(monkeypatch, name, ncores, on_slurm):
    dispatcher = Dispatcher.select_dispatcher(name)
    if on_slurm:
        monkeypatch.setenv('SLURM_JOB_ID', "323232")
    with pytest.raises(ValueError):
        dispatcher.canonicalize_ncores(ncores)
