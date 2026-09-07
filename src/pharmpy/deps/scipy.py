from typing import TYPE_CHECKING

from pharmpy.internals.module.lazy import LazyImport

if TYPE_CHECKING:
    from scipy import linalg, stats
else:
    stats = LazyImport('stats', globals(), 'scipy.stats')
    linalg = LazyImport('linalg', globals(), 'scipy.linalg')
