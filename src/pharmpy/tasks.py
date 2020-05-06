"""Model specific tasks. Can be built into a model
"""


class Estimation:
    """A generic parameter estimation step
    """
    def __init__(self, evaluate=False, tool_options=None):
        self.evaluate = evaluate
        if tool_options is None:
            self.tool_options = []
        else:
            self.tool_options = tool_options


class FOCE:
    pass


class Uncertainty:
    """A generic parameter uncertainty step
    """
    pass
