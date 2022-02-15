"""SCM
"""

from .psn_wrapper import run_scm
from .results import SCMResults

results_class = SCMResults

__all__ = ['SCMResults', 'run_scm']
