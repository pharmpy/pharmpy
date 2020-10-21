from toil.common import Toil
from toil.job import Job

from .jobs import BatchModelfit, InputModel, export_files

toil_options = Job.Runner.getDefaultOptions("file:jobStore")
toil_options.clean = "always"


class ModelfitWorkflow:
    """Workflow for a simple modelfit

    NONMEM only for now, but will become the agnostic workflow
    Can handle multiple models
    Copies up lst and ext files
    """

    def __init__(self, models):
        self._models = models

    def start(self):
        with Toil(toil_options) as workflow:
            input_models = [InputModel(workflow, model) for model in self._models]
            job = BatchModelfit(input_models)
            result_models = workflow.start(job)
            for model in result_models:
                export_files(workflow, model.modelfit_results.tool_files)
