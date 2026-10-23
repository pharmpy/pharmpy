# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .results import COVSearchResults
from .tool import create_workflow, validate_input

results_object = COVSearchResults

__all__ = ('COVSearchResults', 'create_workflow', 'validate_input')
