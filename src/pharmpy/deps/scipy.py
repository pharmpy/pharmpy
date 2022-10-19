from pharmpy.internals.module.lazy import LazyImport

stats = LazyImport('stats', globals(), 'scipy.stats')
linalg = LazyImport('linalg', globals(), 'scipy.linalg')
