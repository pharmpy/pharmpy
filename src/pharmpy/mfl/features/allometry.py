from .model_feature import ModelFeature


class Allometry(ModelFeature):
    def __init__(self, covariate, reference):
        self._covariate = covariate
        self._reference = reference

    @classmethod
    def create(cls, covariate, reference=70.0):
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
    def covariate(self):
        return self._covariate

    @property
    def reference(self):
        return self._reference

    @property
    def args(self):
        return self.covariate, self.reference

    def __repr__(self):
        ref = int(self.reference) if self.reference.is_integer() else self.reference
        return f'ALLOMETRY({self.covariate},{ref})'

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Allometry):
            return False
        return self.covariate == other.covariate and self.reference == other.reference

    @staticmethod
    def repr_many(features):
        assert len(features) == 1
        return repr(features[0])
