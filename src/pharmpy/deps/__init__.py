from typing import TYPE_CHECKING

from pharmpy.internals.module.lazy import LazyImport

if TYPE_CHECKING:
    import altair
    import networkx
    import numpy
    import pandas
    import pint
    import symengine
    import sympy
    import sympy.physics.units as sympy_units
    import sympy.stats as sympy_stats
else:
    altair = LazyImport('altair', globals(), 'pharmpy.deps.altair', 'altair')
    networkx = LazyImport('networkx', globals(), 'networkx')
    numpy = LazyImport('numpy', globals(), 'numpy')
    pandas = LazyImport('pandas', globals(), 'pandas')
    pint = LazyImport('pint', globals(), 'pint')
    symengine = LazyImport('symengine', globals(), 'symengine')
    sympy = LazyImport('sympy', globals(), 'sympy')
    sympy_stats = LazyImport('sympy_stats', globals(), 'sympy.stats')
    sympy_units = LazyImport('sympy_units', globals(), 'sympy.physics.units')

__all__ = (
    'altair',
    'networkx',
    'numpy',
    'pandas',
    'pint',
    'symengine',
    'sympy',
    'sympy_stats',
    'sympy_units',
)
