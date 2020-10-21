from toil.common import Toil
from toil.job import Job

from pharmpy.data.iterators import Resample
from pharmpy.methods.bootstrap.results import BootstrapResults

from .jobs import InputModel, NonmemJob

toil_options = Job.Runner.getDefaultOptions("file:jobStore")
toil_options.clean = "always"


class BootstrapPrepare(Job):
    """Prepare all bootstrap models and datasets"""

    def __init__(self, input_model, resamples=10):
        super().__init__()
        self.input_model = input_model
        self.resamples = resamples

    def run(self, file_store):
        resampler = Resample(
            self.input_model.model, group='ID', resamples=self.resamples, replace=True
        )
        rvs = []
        for model, _ in resampler:
            job = NonmemJob(model, [])  # No extra files since dataset is recreated
            child = self.addChild(job)
            rvs.append(child.rv())
        return self.addFollowOn(BootstrapPostprocess(self.input_model.model, rvs)).rv()


class BootstrapPostprocess(Job):
    def __init__(self, original_model, rvs):
        super().__init__()
        self.original_model = original_model
        self.rvs = rvs

    def run(self, file_store):
        res = BootstrapResults(self.rvs, original_model=self.original_model)
        return res


class BootstrapWorkflow:
    """Workflow to perform a bootstrap"""

    def __init__(self, model):
        self._model = model

    def start(self):
        with Toil(toil_options) as workflow:
            input_model = InputModel(workflow, self._model)
            job = BootstrapPrepare(input_model)
            res = workflow.start(job)
            print(res.parameter_estimates)
