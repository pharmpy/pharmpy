# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .results import CrossvalResults, psn_crossval_results

results_class = CrossvalResults

__all__ = ['CrossvalResults', 'psn_crossval_results']
