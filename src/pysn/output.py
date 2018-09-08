# -*- encoding: utf-8 -*-


# from generic import ParameterModel


def _noexec_exception(model, details):
    return AttributeError("Tried to access output (%s) on model with no prior execution: %s" %
                          (details, model))


class ModelOutput:
    """Generic template for :attr:`Model.output`, manager of all output of model execution."""

    def __init__(self, model):
        self.model = model

    @property
    def estimation(self):
        if not self.model.executed:
            raise _noexec_exception(self.model, 'estimation')
