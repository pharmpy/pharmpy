from __future__ import annotations

from typing import TYPE_CHECKING, Union

from .model_feature import ModelFeature

if TYPE_CHECKING:
    from ..model_features import ModelFeatures


class Allometry(ModelFeature):
    def __init__(self, covariate: str, reference: float):
        self._covariate = covariate
        self._reference = reference

    @classmethod
    def create(cls, covariate: str, reference: Union[int, float] = 70.0) -> Allometry:
        if not isinstance(covariate, str):
            raise TypeError(f'Type of `covariate` must be a string: got {type(covariate)}')
        if not isinstance(reference, float) and not isinstance(reference, int):
            raise TypeError(f'Type of `reference` must be a float or an int: got {type(reference)}')
        return cls(covariate=covariate.upper(), reference=float(reference))

    def replace(self, **kwargs):
        covariate = kwargs.get('covariate', self.covariate)
        reference = kwargs.get('reference', self.reference)
        return Allometry.create(covariate=covariate, reference=reference)

    @property
    def covariate(self) -> str:
        return self._covariate

    @property
    def reference(self) -> float:
        return self._reference

    @property
    def args(self) -> tuple[str, float]:
        return self.covariate, self.reference

    def __repr__(self) -> str:
        ref = int(self.reference) if self.reference.is_integer() else self.reference
        return f'ALLOMETRY({self.covariate},{ref})'

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Allometry):
            return False
        return self.covariate == other.covariate and self.reference == other.reference

    @staticmethod
    def repr_many(mf: ModelFeatures) -> str:
        features = tuple(feat for feat in mf.features if isinstance(feat, Allometry))
        assert len(features) == len(mf.features)
        assert len(features) == 1
        return repr(features[0])
