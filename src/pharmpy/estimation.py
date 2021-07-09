# The estimation steps in a model


class EstimationMethod:
    def __init__(self, method, interaction=False, cov=False, other_options=[]):
        method = self._canonicalize_and_check_method(method)
        self._method = method
        self.interaction = interaction
        self.cov = cov
        self.other_options = other_options

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

    def __eq__(self, other):
        return (
            self._method == other._method
            and self.interaction == other.interaction
            and self.cov == other.cov
            and self.other_options == other.other_options
        )

    def __repr__(self):
        return (
            f'EstimationMethod("{self._method}", interaction={self.interaction}, cov={self.cov}, '
            f'other_options={self.other_options})'
        )


def list_supported_est():
    return ['FO', 'FOCE', 'ITS', 'LAPLACE', 'IMPMAP', 'IMP', 'SAEM']
