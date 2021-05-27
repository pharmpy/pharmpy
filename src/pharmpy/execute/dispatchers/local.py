import tempfile

from pharmpy.utils import TemporaryDirectoryChanger

from ..dispatcher import ExecutionDispatcher
from .local_run import run


class LocalDispatcher(ExecutionDispatcher):
    def run(self, workflow, database):
        with tempfile.TemporaryDirectory() as tempdirname:
            with TemporaryDirectoryChanger(tempdirname):
                results = run(workflow.as_dict())
            return results
