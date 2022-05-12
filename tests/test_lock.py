import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from pathlib import Path
from queue import Queue

import pytest

from pharmpy.lock import (
    AcquiringProcessLevelLockWouldBlockError,
    AcquiringThreadLevelLockWouldBlockError,
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

        first_is_locked = Queue()
        is_done = Queue()
        last_attempt_locking = Queue()

        def thread_lock_first():
            with path_lock(path, shared=False, blocking=False):
                first_is_locked.put(0)
                is_done.get()

        def thread_lock_last():
            last_attempt_locking.get()
            with path_lock(path, shared=False, blocking=False):
                pass

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(thread_lock_first)
            last = executor.submit(thread_lock_last)

            first_is_locked.get()
            last_attempt_locking.put(0)

            with pytest.raises(AcquiringThreadLevelLockWouldBlockError):
                last.result()

            is_done.put(0)


def process_lock_first(conn, path):
    with path_lock(path, shared=False, blocking=False):
        conn.send('')
        conn.recv()


def process_lock_last(conn, path):
    conn.recv()
    try:
        with path_lock(path, shared=False, blocking=False):
            pass
        conn.send(None)
    except Exception as e:
        conn.send(e)


def test_exclusive_processes_non_blocking(tmp_path):
    with lock(tmp_path) as path:

        parent_first_conn, first_conn = Pipe()
        parent_last_conn, last_conn = Pipe()
        Process(target=process_lock_first, args=(first_conn, path)).start()
        last = Process(target=process_lock_last, args=(last_conn, path))
        last.start()

        parent_first_conn.recv()
        parent_last_conn.send('')
        e = parent_last_conn.recv()

        assert isinstance(e, AcquiringProcessLevelLockWouldBlockError)

        parent_first_conn.send('')
