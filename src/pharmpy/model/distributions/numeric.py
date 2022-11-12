from __future__ import annotations

from abc import abstractmethod
from typing import Union

from pharmpy.deps import numpy as np
from pharmpy.internals.immutable import Immutable


class NumericDistribution(Immutable):
    @abstractmethod
    def sample(self, rng, size: int) -> np.ndarray:
        pass


class ConstantDistribution(NumericDistribution):
    def __init__(self, value: Union[int, float]):
        self._value = float(value)

    def sample(self, _, size: int) -> np.ndarray:
        return np.full(size, self._value)


class NormalDistribution(NumericDistribution):
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def sample(self, rng, size: int) -> np.ndarray:
        return rng.normal(self._mean, self._std, size=size)


class MultivariateNormalDistribution(NumericDistribution):
    def __init__(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma

    def sample(self, rng, size: int) -> np.ndarray:
        return rng.multivariate_normal(self._mu, self._sigma, size=size)
