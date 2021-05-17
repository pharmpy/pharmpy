import shutil
import tempfile
from pathlib import Path

from pharmpy.utils import TemporaryDirectoryChanger

from ..dispatcher import ExecutionDispatcher
from .local_run import run


class LocalDispatcher(ExecutionDispatcher):
    def run(self, workflow, database):
        with tempfile.TemporaryDirectory() as tempdirname:
            temppath = Path(tempdirname)
            for source, dest in workflow.infiles:
                dest_path = temppath / dest
                dest_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest_path)
            with TemporaryDirectoryChanger(temppath):
                results = run(workflow.as_dict())
            for model, files in zip(workflow.models, workflow.outfiles):
                for f in files:
                    database.store_local_file(model, temppath / f)
            return results
