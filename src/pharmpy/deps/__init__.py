from ..lazyload import LazyLoader

altair = LazyLoader('altair', globals(), 'pharmpy.deps.altair', 'altair')
networkx = LazyLoader('networkx', globals(), 'networkx')
numpy = LazyLoader('numpy', globals(), 'numpy')
pandas = LazyLoader('pandas', globals(), 'pandas')
symengine = LazyLoader('symengine', globals(), 'symengine')
sympy = LazyLoader('sympy', globals(), 'sympy')
sympy_stats = LazyLoader('sympy_stats', globals(), 'sympy.stats')

__all__ = (
    'altair',
    'networkx',
    'numpy',
    'pandas',
    'symengine',
    'sympy',
    'sympy_stats',
)
