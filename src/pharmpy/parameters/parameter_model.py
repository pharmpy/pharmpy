# -*- encoding: utf-8 -*-


class ParameterModel:
    """Generic template for :attr:`Model.parameters`, the parameters of the model."""

    def __init__(self, model):
        self.model = model

    @property
    def inits(self):
        """All initial estimates for the parameters."""
        raise NotImplementedError
