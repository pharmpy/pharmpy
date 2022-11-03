import os
from contextlib import AbstractContextManager


class chdir(AbstractContextManager):
    """Non thread-safe context manager to change the current working directory.
    (copied from contextlib's implementation in Python 3.11)
    """

    def __init__(self, path):
        self.path = path
        self._old_cwd = []

    def __enter__(self):
        self._old_cwd.append(os.getcwd())
        os.chdir(self.path)

    def __exit__(self, *excinfo):
        os.chdir(self._old_cwd.pop())
