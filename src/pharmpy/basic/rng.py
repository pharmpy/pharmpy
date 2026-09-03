from __future__ import annotations

import os
from typing import Optional, Union

from pharmpy.deps import numpy as np


class Seed:
    """A random seed"""

    def __init__(self, obj: Optional[Union[int, float, Seed]] = None):
        if obj is None:
            seed = int(os.urandom(16).hex(), 16)
        elif isinstance(obj, Seed):
            seed = obj._value
        else:
            if isinstance(obj, float):
                if int(obj) != obj:
                    raise ValueError("Seed must be an integer")
            try:
                seed = int(obj)
            except ValueError:
                raise ValueError("Seed must be an integer")
        self._value = seed

    @property
    def value(self) -> int:
        return self._value

    def __int__(self):
        return self._value

    def __repr__(self):
        return f"Seed({self._value})"


class RandomNumberGenerator:
    """A random number generator"""

    def __init__(self, obj: Optional[Union[int, list, float, Seed, RandomNumberGenerator]] = None):
        if isinstance(obj, RandomNumberGenerator):
            self._backend = obj._backend
        elif isinstance(obj, list):
            self._backend = np.random.default_rng(obj)
        else:
            seed = Seed(obj)
            self._backend = np.random.default_rng(seed.value)

    def to_numpy(self):
        return self._backend
