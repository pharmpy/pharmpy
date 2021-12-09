import copy
from collections.abc import MutableSequence

import pandas as pd


class EstimationSteps(MutableSequence):
    def __init__(self, steps=None):
        if isinstance(steps, EstimationSteps):
            self._steps = copy.deepcopy(steps._steps)
        elif steps is None:
            self._steps = []
        else:
            self._steps = list(steps)

    def __getitem__(self, i):
        return self._steps[i]

    def __setitem__(self, i, value):
        self._steps[i] = value

    def __delitem__(self, i):
        del self._steps[i]

    def __len__(self):
        return len(self._steps)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for s1, s2 in zip(self, other):
            if s1 != s2:
                return False
        return True

    def insert(self, i, value):
        self._steps.insert(i, value)

    def copy(self):
        return copy.deepcopy(self)

    def to_dataframe(self):
        method = [s.method for s in self._steps]
        interaction = [s.interaction for s in self._steps]
        cov = [s.cov for s in self._steps]
        evaluation = [s.evaluation for s in self._steps]
        maximum_evaluations = [s.maximum_evaluations for s in self._steps]
        laplace = [s.laplace for s in self._steps]
        tool_options = [s.tool_options for s in self._steps]
        df = pd.DataFrame(
            {
                'method': method,
                'interaction': interaction,
                'cov': cov,
                'evaluation': evaluation,
                'maximum_evaluations': maximum_evaluations,
                'laplace': laplace,
                'tool_options': tool_options,
            }
        )
        return df

    def __repr__(self):
        if len(self) == 0:
            return "EstimationSteps()"
        return self.to_dataframe().to_string()

    def _repr_html_(self):
        if len(self) == 0:
            return "EstimationSteps()"
        else:
            return self.to_dataframe().to_html()


class EstimationStep:

    supported_methods = ['FO', 'FOCE', 'ITS', 'IMPMAP', 'IMP', 'SAEM', 'BAYES']

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
        if method not in self.supported_methods:
            raise ValueError(
                f'EstimationStep: {method} not recognized. Use any of {self.supported_methods}.'
            )
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
            f'EstimationStep("{self.method}", interaction={self.interaction}, '
            f'cov={self.cov}, evaluation={self.evaluation}, '
            f'maximum_evaluations={self.maximum_evaluations}, laplace={self.laplace}, '
            f'tool_options={self.tool_options})'
        )

    def copy(self):
        return copy.deepcopy(self)
