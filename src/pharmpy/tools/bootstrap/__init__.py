from .results import BootstrapResults
from .tool import create_workflow, validate_input

results_class = BootstrapResults

__all__ = ['create_workflow', 'validate_input', 'BootstrapResults']
