import subprocess
from pathlib import Path

from toil.job import Job


class CommandLineJob(Job):
    def copy_file(self, file_store, file_id):
        path = file_store.readGlobalFile(file_id, userPath=Path(file_id).name)
        return Path(path)

    def call(self, command, file_store):
        output = subprocess.check_output(command).decode()
        file_store.logToMaster(output)


class GatherFiles(Job):
    """Gather files after a batch job
    """
    def __init__(self, rvs):
        super().__init__()
        self.rvs = rvs

    def run(self, file_store):
        result_files = [item for sublist in self.rvs for item in sublist]
        self.result_files = result_files
        return result_files


class BatchModelfit(Job):
    """Runs a list of models

       root -> [NonmemJob ...] -> GatherFiles
    """
    def __init__(self, workflow, models):
        super().__init__()
        self.models = models
        self.files = []
        for model in models:
            files = import_files(workflow, [model.input.path])
            self.files.append(files)

    def run(self, file_store):
        rvs = []
        for model, files in zip(self.models, self.files):
            job = NonmemJob(model, files)
            child = self.addChild(job)
            rvs.append(child.rv())

        return self.addFollowOn(GatherFiles(rvs)).rv()


class NonmemJob(CommandLineJob):
    """One nmfe execution
    """
    def __init__(self, model, files):
        super().__init__()
        self.model = model
        self.additional_files = files

    def run(self, file_store):
        path = self.model.write()
        for file_id in self.additional_files:
            self.copy_file(file_store, file_id)
        self.call(['nmfe74', path.name, path.with_suffix('.lst').name], file_store)
        result_files = [
            file_store.writeGlobalFile(path.with_suffix('.lst').name),
            file_store.writeGlobalFile(path.with_suffix('.ext').name)
        ]
        return result_files


def import_file(workflow, path):
    path = Path(path).resolve()
    url = "file://" + str(path)
    file_id = workflow.importFile(url)
    return file_id


def import_files(workflow, paths):
    file_ids = [import_file(workflow, path) for path in paths]
    return file_ids


def export_files(workflow, file_ids):
    dir_path = Path('.').resolve()
    for file_id in file_ids:
        path = dir_path / Path(file_id).name
        url = 'file://' + str(path)
        workflow.exportFile(file_id, url)
