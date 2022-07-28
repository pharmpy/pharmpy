from .lazyload import LazyLoader

altair = LazyLoader('altair', globals(), 'altair')
networkx = LazyLoader('networkx', globals(), 'networkx')
numpy = LazyLoader('numpy', globals(), 'numpy')
pandas = LazyLoader('pandas', globals(), 'pandas')
scipy = LazyLoader('scipy', globals(), 'scipy')
symengine = LazyLoader('symengine', globals(), 'symengine')
sympy = LazyLoader('sympy', globals(), 'sympy')
