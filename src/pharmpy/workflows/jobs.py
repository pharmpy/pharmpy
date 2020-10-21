import subprocess
from pathlib import Path

from toil.job import Job

from pharmpy import Model


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
    """Gather files after a batch job"""

    def __init__(self, rvs):
        super().__init__()
        self.rvs = rvs

    def run(self, file_store):
        return self.rvs


class BatchModelfit(Job):
    """Runs a list of models

    root -> [NonmemJob ...] -> GatherFiles
    """

    def __init__(self, models):
        super().__init__()
        self.input_models = models

    def run(self, file_store):
        rvs = []
        for input_model in self.input_models:
            job = NonmemJob(input_model.model, input_model.files)
            child = self.addChild(job)
            rvs.append(child.rv())

        return self.addFollowOn(GatherFiles(rvs)).rv()


class NonmemJob(CommandLineJob):
    """One nmfe execution"""

    def __init__(self, model, additional_files):
        super().__init__()
        self.model = model
        self.additional_files = additional_files

    def run(self, file_store):
        path = self.model.write()
        for file_id in self.additional_files:
            self.copy_file(file_store, file_id)
        self.call(['nmfe74', path.name, path.with_suffix('.lst').name], file_store)
        model = Model(path)  # New model for now. Paths need updating...
        model.modelfit_results.parameter_estimates  # To read it in
        result_files = [
            file_store.writeGlobalFile(str(path)) for path in model.modelfit_results.tool_files
        ]
        model.modelfit_results.tool_files = result_files
        return model


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
