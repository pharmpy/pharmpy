import subprocess
from pathlib import Path

from toil.job import Job


class InputModel:
    """A wrapper for a model that is input to a specific workflow

       Handles all files needed for the model and the model object
    """
    def __init__(self, workflow, model):
        self.model = model
        self.files = import_files(workflow, [model.input.path])


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
    def __init__(self, models):
        super().__init__()
        self.input_models = models
        #self.files = []
        #for model in models:
        #    files = import_files(workflow, [model.input.path])
        #    self.files.append(files)

    def run(self, file_store):
        rvs = []
        for input_model in self.input_models:
            job = NonmemJob(input_model)
            child = self.addChild(job)
            rvs.append(child.rv())

        return self.addFollowOn(GatherFiles(rvs)).rv()


class NonmemJob(CommandLineJob):
    """One nmfe execution
    """
    def __init__(self, model):
        super().__init__()
        self.input_model = model

    def run(self, file_store):
        path = self.input_model.model.write()
        for file_id in self.input_model.files:
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
