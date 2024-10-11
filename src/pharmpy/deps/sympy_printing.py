from typing import TYPE_CHECKING

from pharmpy.internals.module.lazy import LazyImport

if TYPE_CHECKING:
    import sympy.printing.codeprinter as codeprinter
    import sympy.printing.fortran as fortran
    import sympy.printing.pretty.pretty as pretty
    import sympy.printing.str as str
else:
    str = LazyImport('str', globals(), 'sympy.printing.str')
    fortran = LazyImport('fortran', globals(), 'sympy.printing.fortran')
    codeprinter = LazyImport('codeprinter', globals(), 'sympy.printing.codeprinter')
    pretty = LazyImport('pretty', globals(), 'sympy.printing.pretty.pretty')
