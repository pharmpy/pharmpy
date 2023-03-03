import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from itertools import chain, groupby
from multiprocessing import get_context
from pathlib import Path
from queue import Queue
from random import random
from threading import Barrier

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.internals.fs.lock import (
    AcquiringProcessLevelLockWouldBlockError,
    AcquiringThreadLevelLockWouldBlockError,
    RecursiveDeadlockError,
    path_lock,
    process_level_path_lock,
    thread_level_lock,
)

mp = get_context(method='spawn')


class SimpleThreadManager:
    @staticmethod
    def Barrier(*args, **kwargs):
        return Barrier(*args, **kwargs)

    @staticmethod
    def Queue(*args, **kwargs):
        return Queue(*args, **kwargs)


@contextmanager
def processes(n: int):
    with ProcessPoolExecutor(max_workers=n, mp_context=mp) as executor:
        with mp.Manager() as m:
            yield executor, m


@contextmanager
def threads(n: int):
    with ThreadPoolExecutor(max_workers=n) as executor:
        yield executor, SimpleThreadManager


@contextmanager
def lock(directory):
    with chdir(directory):
        lock = Path('lock')
        lock.touch()
        path = str(lock)
        yield path


def lock_first(is_locked, is_done, path, shared):
    with path_lock(path, shared=shared, blocking=False):
        is_locked.wait()
        is_done.wait()


def lock_rest(is_locked, is_done, path, shared, blocking):
    is_locked.wait()
    with path_lock(path, shared=shared, blocking=blocking):
        if is_done is not None and shared:
            is_done.wait()


@pytest.mark.parametrize(
    'parallelization, exception',
    (
        (threads, AcquiringThreadLevelLockWouldBlockError),
        (processes, AcquiringProcessLevelLockWouldBlockError),
    ),
)
@pytest.mark.parametrize(
    'shared',
    (
        [False, True],
        [False, False],
        [False, True, False],
        [False, True, True],
        [False, False, True],
        [False, False, False],
        [True, False],
        [True, True, False],
        [True, False, False],
    ),
    ids=repr,
)
def test_non_blocking(tmp_path, parallelization, exception, shared):
    if os.name == 'nt' and 'processes' in repr(parallelization):
        pytest.skip("TODO Processes-based tests randomly fail on Windows.")

    assert len(shared) >= 2
    assert not shared[0] or not shared[-1]

    with lock(tmp_path) as path:
        n = len(shared)
        nb = 1 + (
            1
            if not shared[0] or (os.name == 'nt' and 'processes' in repr(parallelization))
            else sum(map(lambda s: 1 if s else 0, shared))
        )

        with parallelization(n) as [executor, m]:
            is_locked = m.Barrier(n)
            is_done = m.Barrier(nb)

            tasks = []

            tasks.append(executor.submit(lock_first, is_locked, is_done, path, shared[0]))
            for s in shared[1:-1]:
                tasks.append(
                    executor.submit(
                        lock_rest, is_locked, is_done if nb >= 3 else None, path, s, True
                    )
                )
            tasks.append(executor.submit(lock_rest, is_locked, None, path, shared[-1], False))

            with pytest.raises(exception):
                tasks[-1].result()

            is_done.wait()

            for task in tasks[:-1]:
                task.result()


def locked_threads(parameters, is_done):
    n = len(parameters)
    with threads(n) as (executor, _):
        tasks = []
        for params in parameters:
            tasks.append(executor.submit(*params))
        for task in as_completed(tasks):
            try:
                task.result()
            except:  # noqa E722
                is_done.wait()
                raise


@pytest.mark.parametrize('n_blocking', (0, 1, 2, 3, 4, 5))
def test_non_blocking_processes_and_threads(tmp_path, n_blocking):
    if os.name == 'nt':
        pytest.skip("TODO Processes-based tests randomly fail on Windows.")

    with lock(tmp_path) as path:
        with processes(2) as (executor, m):
            is_locked = m.Barrier(2 + n_blocking)
            is_done = m.Barrier(2)

            t1 = executor.submit(lock_first, is_locked, is_done, path, False)

            t2 = executor.submit(
                locked_threads,
                ([(lock_rest, is_locked, None, path, True, True)] * n_blocking)
                + [(lock_rest, is_locked, None, path, True, False)],
                is_done,
            )

            with pytest.raises(AcquiringProcessLevelLockWouldBlockError):
                t2.result()

            t1.result()


def lock_shared(are_locked, q, path, i):
    with path_lock(path, shared=True):
        are_locked.wait()
        time.sleep(1)
        q.put(i)


def lock_exclusive(are_locked, q, path, i):
    are_locked.wait()
    with path_lock(path, shared=False):
        q.put(i)


@pytest.mark.parametrize('parallelization', (threads, processes))
def test_many_shared_one_exclusive_blocking(tmp_path, parallelization):
    if os.name == 'nt' and 'processes' in repr(parallelization):
        pytest.skip("TODO Processes-based tests randomly fail on Windows.")

    with lock(tmp_path) as path:
        n = 10

        with parallelization(n) as [executor, m]:
            results_queue = m.Queue()

            # NOTE We run the test twice to reuse workers to catch errors where
            # some workers are left in a locked state.
            for _ in range(2):
                are_locked = m.Barrier(n)

                for i in range(1, n):
                    executor.submit(lock_shared, are_locked, results_queue, path, i)
                last = executor.submit(lock_exclusive, are_locked, results_queue, path, 0)

                last.result()

                results = []
                for i in range(n):
                    results.append(results_queue.get())

                assert sorted(results) == sorted(range(n))
                assert results[-1] == 0


@pytest.mark.parametrize('acquire_lock', (path_lock, thread_level_lock, process_level_path_lock))
@pytest.mark.parametrize('shared', (True, False))
def test_non_reentrant_dead_lock(tmp_path, acquire_lock, shared):
    with lock(tmp_path) as path:
        with acquire_lock(path, shared=shared):
            with pytest.raises(RecursiveDeadlockError):
                with acquire_lock(path, shared=shared):
                    pass


@pytest.mark.parametrize('shared', (True, False))
def test_reentrant(tmp_path, shared):
    with lock(tmp_path) as path:
        with path_lock(path, shared=shared, reentrant=False):
            with path_lock(path, shared=shared, reentrant=True):
                pass

        with path_lock(path, shared=shared, reentrant=True):
            with path_lock(path, shared=shared, reentrant=True):
                pass

        with path_lock(path, shared=shared, reentrant=False):
            with path_lock(path, shared=shared, reentrant=True):
                with path_lock(path, shared=shared, reentrant=True):
                    pass


@pytest.mark.parametrize(
    'shared',
    (
        [True, False],
        [False, True],
        [True, False, True],
        [False, True, False],
        [True, True, False],
        [False, False, True],
    ),
    ids=repr,
)
def test_reentrant_mixed(tmp_path, shared):
    with lock(tmp_path) as path:

        def rec(types):
            if types:
                shared, *rest = types
                with path_lock(path, shared=shared, reentrant=True):
                    rec(rest)

        rec(shared)


def exclusive_thread_write(path, i):
    with path_lock(path, shared=False) as fd:
        try:
            with open(fd, closefd=False) as fp:
                fp.seek(0)
                done = json.load(fp)
        except json.decoder.JSONDecodeError:
            done = []

        done.append(i)
        with open(fd, 'w', closefd=False) as fp:
            fp.seek(0)
            fp.truncate(0)
            json.dump(done, fp)
            fp.seek(0)


def many_exclusive_threads_and_processes_rw_process(path, indices):
    with ThreadPoolExecutor(max_workers=len(indices)) as executor:
        executor.map(exclusive_thread_write, [path] * len(indices), indices)


def test_many_exclusive_threads_and_processes_rw(tmp_path):
    if os.name == 'nt':
        pytest.skip("TODO Processes-based tests randomly fail on Windows.")

    with lock(tmp_path) as path:
        m = 10
        n = m**2
        items = list(range(n))
        partition = list(
            map(
                lambda g: list(map(lambda t: t[1], g[1])),
                groupby(enumerate(items), lambda t: t[0] // m),
            )
        )

        assert len(partition) == m
        assert sorted(chain(*partition)) == items

        with mp.Pool(processes=len(partition)) as pool:
            pool.starmap(
                many_exclusive_threads_and_processes_rw_process,
                map(lambda part: (path, part), partition),
            )

        with path_lock(path, shared=False) as fd:
            with open(fd, closefd=False) as fp:
                fp.seek(0)
                results = json.load(fp)
                fp.seek(0)

        assert sorted(results) == sorted(range(n))


@pytest.mark.parametrize('parallelization, n', ((threads, 200), (processes, 20)))
def test_many_exclusive(tmp_path, parallelization, n):
    if os.name == 'nt' and 'processes' in repr(parallelization):
        pytest.skip("TODO Processes-based tests randomly fail on Windows.")

    with lock(tmp_path) as path:
        items = list(range(n))

        with parallelization(n) as [executor, _]:
            executor.map(
                exclusive_thread_write,
                [path] * n,
                items,
            )

        with path_lock(path, shared=False) as fd:
            with open(fd, closefd=False) as fp:
                fp.seek(0)
                results = json.load(fp)
                fp.seek(0)

        assert sorted(results) == sorted(range(n))


def lock_shared_chained(path, first_is_locked, recvp, send, recvn, results, n, i):
    if i > 0:
        j = recvp.get()  # Wait for previous thread to be locked
        assert j == i - 1
    with path_lock(path, shared=True):
        if i == 0:
            first_is_locked.wait()  # Allow exclusive lock attempt
        results.put(i)
        send.put(i)  # Notify next thread that we are locked
        send.put(i)  # Notify previous thread that we are locked
        if i < n - 2:
            j = recvn.get()  # Wait for next thread to be locked
            assert j == i + 1


@pytest.mark.parametrize('parallelization', (threads, processes))
def test_chained_shared_one_exclusive_blocking(tmp_path, parallelization):
    if os.name == 'nt' and 'processes' in repr(parallelization):
        pytest.skip("TODO Processes-based tests randomly fail on Windows.")

    with lock(tmp_path) as path:
        n = 10

        with parallelization(n) as [executor, m]:
            results_queue = m.Queue()

            # NOTE We run the test twice to reuse workers to catch errors where
            # some workers are left in a locked state.
            for _ in range(2):
                first_is_locked = m.Barrier(2)
                queues = [m.Queue() for _ in range(n + 1)]

                for i in range(n - 1):
                    executor.submit(
                        lock_shared_chained,
                        path,
                        first_is_locked,
                        queues[i],
                        queues[i + 1],
                        queues[i + 2],
                        results_queue,
                        n,
                        i,
                    )
                last = executor.submit(lock_exclusive, first_is_locked, results_queue, path, n - 1)

                last.result()

                results = []
                for i in range(n):
                    results.append(results_queue.get())

                assert results == sorted(range(n))


def sync_read(path, filename, before, after, q, i):
    before.wait()
    time.sleep(0.01 * random())
    with path_lock(path, shared=True):
        after.wait()
        with open(filename) as fp:
            contents = json.load(fp)
    q.put({'type': 'read', 'id': i, 'contents': contents})


def sync_write(path, filename, q, i):
    with path_lock(path, shared=False):
        with open(filename) as fp:
            contents = json.load(fp)

        contents['id'] = i
        contents['counter'] += 1

        with open(filename, 'w') as fp:
            json.dump(contents, fp)

        with open(filename) as fp:
            contents = json.load(fp)

        contents['copy'] += 1

        with open(filename, 'w') as fp:
            json.dump(contents, fp)

    q.put({'type': 'write', 'id': i, 'contents': contents})


@pytest.mark.parametrize('parallelization', (threads, processes))
def test_synchronized_reads_blocking(tmp_path, parallelization):
    if os.name == 'nt' and 'processes' in repr(parallelization):
        pytest.skip("TODO Processes-based tests randomly fail on Windows.")

    with lock(tmp_path) as path:
        filename = 'rw'

        with open(filename, 'w') as fp:
            json.dump({'id': -1, 'counter': 0, 'copy': 0}, fp)

        n = 1000
        k = 7
        p = 10
        max_write = p - k

        with parallelization(p) as [executor, m]:
            q = m.Queue()
            i = 0
            g = 0
            w = 0
            messages = []
            group = {}
            while i < n:
                c = min(n - i, k)
                if w >= max_write or random() < 1 / (c + 1):
                    before = m.Barrier(c)
                    after = m.Barrier(c)
                    for _ in range(c):
                        executor.submit(sync_read, path, filename, before, after, q, i)
                        group[i] = g
                        i += 1
                    g += 1
                else:
                    executor.submit(sync_write, path, filename, q, i)
                    i += 1
                    w += 1

                while (
                    not q.empty() or len(messages) < i - k
                ):  # NOTE We empty the queue as much as possible
                    messages.append(q.get())
                    if messages[-1]['type'] == 'write':
                        w -= 1

            while len(messages) < n:
                messages.append(q.get())

        values = {}
        for message in messages:
            t = message['type']
            if t == 'read':
                i = message['id']
                g = group[i]
                counter = message['contents']['counter']
                # NOTE counter is the same for the whole group
                assert values.setdefault(g, counter) == counter
                # NOTE copy is identical to counter
                assert message['contents']['copy'] == counter
            else:
                assert t == 'write'
                # NOTE check nobody else wrote to file while we held the lock
                assert message['id'] == message['contents']['id']
                assert message['contents']['copy'] == message['contents']['counter']
