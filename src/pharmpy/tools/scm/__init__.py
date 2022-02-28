"""SCM
"""

from .psn_wrapper import have_scm, run_scm
from .results import SCMResults

results_class = SCMResults

__all__ = ['SCMResults', 'have_scm', 'run_scm']
