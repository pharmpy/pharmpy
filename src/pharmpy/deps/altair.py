# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import altair
from altair.utils.execeval import eval_block


@altair.theme.register('pharmpy', enable=True)
def pharmpy_theme():
    return altair.theme.ThemeConfig(
        {
            'config': {
                'axis': {
                    'labelFontSize': 11,
                    'titleFontSize': 13,
                },
                'legend': {
                    'labelFontSize': 12,
                    'titleFontSize': 13,
                },
            }
        }
    )


altair.data_transformers.disable_max_rows()

__all__ = ('altair', 'eval_block')
