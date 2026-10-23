# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .results import QAResults
from .tool import create_workflow

results_class = QAResults

__all__ = ['QAResults', 'create_workflow']
