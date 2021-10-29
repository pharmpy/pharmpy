# The estimation steps in a model


class EstimationMethod:
    def __init__(
        self,
        method,
        interaction=False,
        cov=False,
        evaluation=False,
        maxeval=None,
        tool_options=None,
    ):
        method = self._canonicalize_and_check_method(method)
        self._method = method
        self.interaction = interaction
        self.cov = cov
        self.evaluation = evaluation
        self.maxeval = maxeval
        self.tool_options = tool_options

    def _canonicalize_and_check_method(self, method):
        method = method.upper()
        supported = list_supported_est()
        if method not in supported:
            raise ValueError(f'Estimation method: {method} not recognized. Use any of {supported}.')
        return method

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        method = self._canonicalize_and_check_method(value)
        self._method = method

    @property
    def tool_options(self):
        return self._tool_options

    @tool_options.setter
    def tool_options(self, options):
        self._tool_options = options

    def append_tool_options(self, options):
        if not self.tool_options:
            self.tool_options = options
        else:
            tool_options_new = {**self.tool_options, **options}
            self.tool_options = tool_options_new

    def __eq__(self, other):
        return (
            self.method == other.method
            and self.interaction == other.interaction
            and self.cov == other.cov
            and self.evaluation == other.evaluation
            and self.maxeval == other.maxeval
            and self.tool_options == other.tool_options
        )

    def __repr__(self):
        return (
            f'EstimationMethod("{self.method}", interaction={self.interaction}, '
            f'cov={self.cov}, evaluation={self.evaluation}, maxeval={self.maxeval}, '
            f'tool_options={self.tool_options})'
        )


def list_supported_est():
    return ['FO', 'FOCE', 'ITS', 'LAPLACE', 'IMPMAP', 'IMP', 'SAEM', 'BAYES']
