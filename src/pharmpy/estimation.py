# The estimation steps in a model


class EstimationMethod:
    def __init__(self, method, cov=False):
        method = self._canonicalize_and_check_method(method)
        self._method = method
        self.cov = cov

    def _canonicalize_and_check_method(self, method):
        method = method.upper()
        supported = ['FO', 'FOI', 'FOCE', 'FOCEI']
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
        return self._method == other._method and self.cov == other.cov

    def __repr__(self):
        return f'EstimationMethod("{self._method}", cov={self.cov})'
