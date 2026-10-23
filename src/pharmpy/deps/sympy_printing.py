# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import TYPE_CHECKING

from .lazy import LazyImport

if TYPE_CHECKING:
    import sympy.printing.pretty.pretty as pretty  # noqa: PLR0402
    from sympy.printing import codeprinter, fortran, str
else:
    str = LazyImport('str', globals(), 'sympy.printing.str')
    fortran = LazyImport('fortran', globals(), 'sympy.printing.fortran')
    codeprinter = LazyImport('codeprinter', globals(), 'sympy.printing.codeprinter')
    pretty = LazyImport('pretty', globals(), 'sympy.printing.pretty.pretty')
