from pharmpy.utils import TemporaryDirectory, TemporaryDirectoryChanger

from ..dispatcher import ExecutionDispatcher
from .local_run import run


class LocalDispatcher(ExecutionDispatcher):
    def run(self, workflow, database):
        with TemporaryDirectory() as tempdirname:
            with TemporaryDirectoryChanger(tempdirname):
                results = run(workflow.as_dict())
        return results
