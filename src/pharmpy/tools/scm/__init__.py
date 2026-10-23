# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""SCM"""

from .psn_wrapper import have_scm, run_scm
from .results import SCMResults

results_class = SCMResults

__all__ = ['SCMResults', 'have_scm', 'run_scm']
