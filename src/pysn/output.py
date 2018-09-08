# -*- encoding: utf-8 -*-


class ModelOutput:
    """Generic template for :attr:`Model.output`, manager of all output of model execution."""

    def __init__(self, model):
        self.model = model
