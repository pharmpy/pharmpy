# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys

from .cli import main


def run():
    args = sys.argv[1:]
    main(args)


if __name__ == "__main__":
    run()
