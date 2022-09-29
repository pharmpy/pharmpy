from .results import COVSearchResults
from .tool import create_workflow, validate_input

results_object = COVSearchResults

__all__ = ('create_workflow', 'COVSearchResults', 'validate_input')
