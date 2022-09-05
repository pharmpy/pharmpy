from ..lazyload import LazyLoader

stats = LazyLoader('stats', globals(), 'scipy.stats')
linalg = LazyLoader('linalg', globals(), 'scipy.linalg')
