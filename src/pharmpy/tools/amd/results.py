from pharmpy.model import Results


class AMDResults(Results):
    def __init__(
        self,
        final_model=None,
        summary_tool=None,
        summary_models=None,
        summary_individuals_count=None,
        summary_errors=None,
    ):
        self.final_model = final_model
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals_count = summary_individuals_count
        self.summary_errors = summary_errors
