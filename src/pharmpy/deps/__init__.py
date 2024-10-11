from typing import TYPE_CHECKING

from pharmpy.internals.module.lazy import LazyImport

if TYPE_CHECKING:
    import altair
    import networkx
    import numpy
    import pandas
    import symengine
    import sympy
    import sympy.stats as sympy_stats
else:
    altair = LazyImport('altair', globals(), 'pharmpy.deps.altair', 'altair')
    networkx = LazyImport('networkx', globals(), 'networkx')
    numpy = LazyImport('numpy', globals(), 'numpy')
    pandas = LazyImport('pandas', globals(), 'pandas')
    symengine = LazyImport('symengine', globals(), 'symengine')
    sympy = LazyImport('sympy', globals(), 'sympy')
    sympy_stats = LazyImport('sympy_stats', globals(), 'sympy.stats')

__all__ = (
    'altair',
    'networkx',
    'numpy',
    'pandas',
    'symengine',
    'sympy',
    'sympy_stats',
)
