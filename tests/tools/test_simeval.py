# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from pharmpy.tools.simeval.results import psn_simeval_results


def test_psn_simeval_results(testdata):
    res = psn_simeval_results(testdata / 'psn' / 'simeval_dir1')
    assert len(res.sampled_iofv) == 59
    assert len(res.iofv_summary) == 59
