# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def name_mangle(s: str) -> str:
    """
    Changes the format of parameter name to avoid using parenthesis

    Parameters
    ----------
    s : str
        Parameter name to be changed

    Returns
    -------
    str
        Parameter name with parenthesis removed

    Example
    -------
    name_mangle("ETA(1)")
    -> "ETA1"

    """
    return s.replace('(', '').replace(')', '').replace(',', '_')
