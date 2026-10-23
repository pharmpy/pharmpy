# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .tool import StructSearchResults, create_workflow, validate_input

results_object = StructSearchResults

__all__ = ('StructSearchResults', 'create_workflow', 'validate_input')
