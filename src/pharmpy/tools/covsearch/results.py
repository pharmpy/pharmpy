from typing import Any, Union

import pharmpy.results

DataFrame = Any  # NOTE should be pd.DataFrame but we wan lazy loading


class COVSearchResults(pharmpy.results.Results):
    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        summary_errors=None,
        best_model=None,
        input_model=None,
        models=None,
        steps: Union[None, DataFrame] = None,
        ofv_summary: Union[None, DataFrame] = None,
        candidate_summary: Union[None, DataFrame] = None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.summary_errors = summary_errors
        self.best_model = best_model
        self.input_model = input_model
        self.models = models
        self.steps = steps
        self.candidate_summary = candidate_summary
        self.ofv_summary = ofv_summary
