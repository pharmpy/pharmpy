# The estimation steps in a model

from collections import namedtuple


class EstimationMethod:
    def __init__(self, method, interaction=False, cov=False, options=None):
        method = self._canonicalize_and_check_method(method)
        self._method = method
        self.interaction = interaction
        self.cov = cov
        self.options = options

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
    def options(self):
        return self._options

    @options.setter
    def options(self, options):
        if isinstance(options, dict):
            self._options = self._options_dict_to_tuple(options)
        else:
            self._options = options

    def append_options(self, options):
        if self.options:
            options_old = self.options
        else:
            options_old = []

        if isinstance(options, dict):
            options_new = options_old + self._options_dict_to_tuple(options)
        else:
            options_new = options_old + options
        self.options = options_new

    @staticmethod
    def _options_dict_to_tuple(options):
        Option = namedtuple('Option', ['key', 'value'])
        options_new = []
        for key, value in options.items():
            options_new += [Option(key, value)]
        return options_new

    def __eq__(self, other):
        return (
            self._method == other._method
            and self.interaction == other.interaction
            and self.cov == other.cov
            and self.options == other.options
        )

    def __repr__(self):
        return (
            f'EstimationMethod("{self._method}", interaction={self.interaction}, cov={self.cov}, '
            f'options={self.options})'
        )


def list_supported_est():
    return ['FO', 'FOCE', 'ITS', 'LAPLACE', 'IMPMAP', 'IMP', 'SAEM', 'BAYES']
