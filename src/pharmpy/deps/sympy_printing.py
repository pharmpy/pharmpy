from typing import TYPE_CHECKING

from pharmpy.internals.module.lazy import LazyImport

if TYPE_CHECKING:
    from sympy.printing import codeprinter, fortran, str
    from sympy.printing.pretty import pretty
else:
    str = LazyImport('str', globals(), 'sympy.printing.str')
    fortran = LazyImport('fortran', globals(), 'sympy.printing.fortran')
    codeprinter = LazyImport('codeprinter', globals(), 'sympy.printing.codeprinter')
    pretty = LazyImport('pretty', globals(), 'sympy.printing.pretty.pretty')
