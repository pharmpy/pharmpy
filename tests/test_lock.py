import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import chain, groupby
from multiprocessing import Manager, Pipe, Pool, Process
from pathlib import Path
from queue import Queue
from threading import Barrier

import pytest

from pharmpy.lock import (
    AcquiringProcessLevelLockWouldBlockError,
    AcquiringThreadLevelLockWouldBlockError,
    RecursiveDeadlockError,
    path_lock,
)
from pharmpy.utils import TemporaryDirectoryChanger


@contextmanager
def lock(directory):
    with TemporaryDirectoryChanger(directory):
        lock = Path('lock')
        lock.touch()
        path = str(lock)
        yield path


def test_exclusive_threads_non_blocking(tmp_path):
    with lock(tmp_path) as path:

        is_done = Barrier(2)
        is_locked = Barrier(2)

        def thread_lock_first():
            with path_lock(path, shared=False, blocking=False):
                is_locked.wait()
                is_done.wait()

        def thread_lock_last():
            is_locked.wait()
            with path_lock(path, shared=False, blocking=False):
                pass

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(thread_lock_first)
            last = executor.submit(thread_lock_last)

            with pytest.raises(AcquiringThreadLevelLockWouldBlockError):
                last.result()

            is_done.wait()


def process_lock_first(is_locked, is_done, path):
    with path_lock(path, shared=False, blocking=False):
        is_locked.wait()
        is_done.wait()


def process_lock_last(is_locked, conn, path):
    is_locked.wait()
    try:
        with path_lock(path, shared=False, blocking=False):
            pass
        conn.send(None)
    except Exception as e:
        conn.send(e)


def test_exclusive_processes_non_blocking(tmp_path):
    with lock(tmp_path) as path:

        m = Manager()
        is_done = m.Barrier(2)
        is_locked = m.Barrier(2)
        parent_last_conn, last_conn = Pipe()
        Process(target=process_lock_first, args=(is_locked, is_done, path)).start()
        last = Process(target=process_lock_last, args=(is_locked, last_conn, path))
        last.start()

        e = parent_last_conn.recv()

        assert isinstance(e, AcquiringProcessLevelLockWouldBlockError)

        is_done.wait()


def test_many_shared_one_exclusive_threads_blocking(tmp_path):
    with lock(tmp_path) as path:

        n = 10

        are_locked = Barrier(n)

        def thread_lock_shared(results, i):
            with path_lock(path, shared=True):
                are_locked.wait()
                time.sleep(1)
                results.append(i)

        def thread_lock_exclusive(results, i):
            are_locked.wait()
            with path_lock(path, shared=False):
                results.append(i)

        with ThreadPoolExecutor(max_workers=n) as executor:
            # NOTE We run the test twice to reuse workers to catch errors where
            # some workers are left in a locked state.
            for j in range(2):
                are_locked.reset()
                results = []

                for i in range(1, n):
                    executor.submit(thread_lock_shared, results, i)
                last = executor.submit(thread_lock_exclusive, results, 0)

                last.result()

                assert sorted(results) == sorted(range(n))
                assert results[-1] == 0


def process_lock_shared(are_locked, q, path, i):
    with path_lock(path, shared=True):
        are_locked.wait()
        time.sleep(1)
        q.put(i)


def process_lock_exclusive(are_locked, q, path, i):
    are_locked.wait()
    with path_lock(path, shared=False):
        q.put(i)


@pytest.mark.skipif(os.name == 'nt', reason="Windows shared process locks are not implemented.")
def test_many_shared_one_exclusive_processes_blocking(tmp_path):
    with lock(tmp_path) as path:

        n = 10

        with Pool(processes=n) as pool:

            m = Manager()
            results_queue = m.Queue()

            # NOTE We run the test twice to reuse workers to catch errors where
            # some workers are left in a locked state.
            for j in range(2):

                are_locked = m.Barrier(n)

                for i in range(1, n):
                    pool.apply_async(process_lock_shared, [are_locked, results_queue, path, i])
                last = pool.apply_async(
                    process_lock_exclusive, [are_locked, results_queue, path, 0]
                )

                last.wait()

                results = []
                for i in range(n):
                    results.append(results_queue.get())

                assert sorted(results) == sorted(range(n))
                assert results[-1] == 0


def test_non_reentrant_dead_lock(tmp_path):
    with lock(tmp_path) as path:
        with path_lock(path):
            with pytest.raises(RecursiveDeadlockError):
                with path_lock(path):
                    pass


def test_reentrant(tmp_path):
    with lock(tmp_path) as path:
        with path_lock(path):
            with path_lock(path, reentrant=True):
                pass

        with path_lock(path, reentrant=True):
            with path_lock(path, reentrant=True):
                pass

        with path_lock(path):
            with path_lock(path, reentrant=True):
                with path_lock(path, reentrant=True):
                    pass


def many_exclusive_threads_and_processes_rw_process(path, indices):
    def thread_write(i):
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

    with ThreadPoolExecutor(max_workers=len(indices)) as executor:
        executor.map(thread_write, indices)


def test_many_exclusive_threads_and_processes_rw(tmp_path):
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

        with Pool(processes=len(partition)) as pool:
            pool.starmap(
                many_exclusive_threads_and_processes_rw_process,
                map(lambda part: (path, part), partition),
            )

        with open(path) as fp:
            results = json.load(fp)

        assert sorted(results) == sorted(range(n))


def test_chained_shared_one_exclusive_threads_blocking(tmp_path):
    with lock(tmp_path) as path:

        n = 10

        first_is_locked = Barrier(2)

        def thread_lock_shared(recvp, send, recvn, results, i):
            if i > 0:
                j = recvp.get()  # Wait for previous thread to be locked
                assert j == i - 1
            with path_lock(path, shared=True):
                if i == 0:
                    first_is_locked.wait()  # Allow exclusive lock attempt
                results.append(i)
                send.put(i)  # Notify next thread that we are locked
                send.put(i)  # Notify previous thread that we are locked
                if i < n - 2:
                    j = recvn.get()  # Wait for next thread to be locked
                    assert j == i + 1

        def thread_lock_exclusive(results, i):
            first_is_locked.wait()
            with path_lock(path, shared=False):
                results.append(i)

        with ThreadPoolExecutor(max_workers=n) as executor:
            # NOTE We run the test twice to reuse workers to catch errors where
            # some workers are left in a locked state.
            for j in range(2):
                first_is_locked.reset()
                results = []

                queues = [Queue() for i in range(n + 1)]

                for i in range(n - 1):
                    executor.submit(
                        thread_lock_shared, queues[i], queues[i + 1], queues[i + 2], results, i
                    )
                last = executor.submit(thread_lock_exclusive, results, n - 1)

                last.result()

                assert results == sorted(range(n))
