from typing import Union

import pandas as pd

import pharmpy.results


class COVSearchResults(pharmpy.results.Results):
    def __init__(
        self,
        summary_tool=None,
        summary_models=None,
        summary_individuals=None,
        summary_individuals_count=None,
        summary_errors=None,
        final_model_name=None,
        models=None,
        steps: Union[None, pd.DataFrame] = None,
        ofv_summary: Union[None, pd.DataFrame] = None,
        candidate_summary: Union[None, pd.DataFrame] = None,
        tool_database=None,
    ):
        self.summary_tool = summary_tool
        self.summary_models = summary_models
        self.summary_individuals = summary_individuals
        self.summary_individuals_count = summary_individuals_count
        self.summary_errors = summary_errors
        self.final_model_name = final_model_name
        self.models = models
        self.steps = steps
        self.candidate_summary = candidate_summary
        self.ofv_summary = ofv_summary
        self.tool_database = tool_database
