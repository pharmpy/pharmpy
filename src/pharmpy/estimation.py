import copy


class EstimationMethod:
    def __init__(
        self,
        method,
        interaction=False,
        cov=False,
        evaluation=False,
        maximum_evaluations=None,
        laplace=False,
        tool_options=None,
    ):
        method = self._canonicalize_and_check_method(method)
        self.method = method
        self.interaction = interaction
        self.cov = cov
        self.evaluation = evaluation
        self.maximum_evaluations = maximum_evaluations
        self.laplace = laplace
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
    def maximum_evaluations(self):
        return self._maximum_evaluations

    @maximum_evaluations.setter
    def maximum_evaluations(self, value):
        if value is not None and value < 1:
            raise ValueError(
                'Number of maximum evaluations must be more than one, use '
                'evaluation=True or tool_options for special cases (e.g. 0 and -1'
                'in NONMEM)'
            )
        self._maximum_evaluations = value

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
            and self.maximum_evaluations == other.maximum_evaluations
            and self.laplace == other.laplace
            and self.tool_options == other.tool_options
        )

    def __repr__(self):
        return (
            f'EstimationMethod("{self.method}", interaction={self.interaction}, '
            f'cov={self.cov}, evaluation={self.evaluation}, '
            f'maximum_evaluations={self.maximum_evaluations}, laplace={self.laplace}, '
            f'tool_options={self.tool_options})'
        )

    def copy(self):
        return copy.deepcopy(self)


def list_supported_est():
    return ['FO', 'FOCE', 'ITS', 'IMPMAP', 'IMP', 'SAEM', 'BAYES']
