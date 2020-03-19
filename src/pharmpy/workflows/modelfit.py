from toil.common import Toil
from toil.job import Job

from .jobs import BatchModelfit, export_files

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
            job = BatchModelfit(workflow, self._models)
            result_files = workflow.start(job)
            export_files(workflow, result_files)
