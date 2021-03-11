import shutil
import tempfile
from pathlib import Path

from pharmpy.utils import TemporaryDirectoryChanger

from ..dispatcher import ExecutionDispatcher
from .local_run import run


class LocalDispatcher(ExecutionDispatcher):
    def run(self, job, database):
        with tempfile.TemporaryDirectory() as tempdirname:
            temppath = Path(tempdirname)
            for source, dest in job.infiles:
                dest_path = temppath / dest
                dest_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest_path)
            with TemporaryDirectoryChanger(temppath):
                results = run(job.workflow)
            for model, files in zip(job.models, job.outfiles):
                for f in files:
                    database.store_local_file(model, temppath / f)
            return results
