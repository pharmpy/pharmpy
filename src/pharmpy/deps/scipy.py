# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import TYPE_CHECKING

from .lazy import LazyImport

if TYPE_CHECKING:
    from scipy import linalg, stats
else:
    stats = LazyImport('stats', globals(), 'scipy.stats')
    linalg = LazyImport('linalg', globals(), 'scipy.linalg')
