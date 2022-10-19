from pharmpy.internals.module.lazy import LazyImport

str = LazyImport('str', globals(), 'sympy.printing.str')
fortran = LazyImport('fortran', globals(), 'sympy.printing.fortran')
codeprinter = LazyImport('codeprinter', globals(), 'sympy.printing.codeprinter')
