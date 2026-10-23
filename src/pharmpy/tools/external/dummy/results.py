# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from pathlib import Path

from pharmpy.tools import read_results


def parse_modelfit_results(model, path):
    # Path to model file or results file
    if path is None:
        return None

    path = Path(path)
    path = path.with_name(f'{model.name}_results.json')

    res = read_results(path)
    return res
