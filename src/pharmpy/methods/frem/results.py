from pharmpy.parameter_sampling import sample_from_covariance_matrix
from pharmpy.results import Results


class FREMResults(Results):
    def __init__(self, frem_model):
        self.frem_model = frem_model
        parvecs = sample_from_covariance_matrix(frem_model, n=100)
        self.parvecs = parvecs
        # FIXME: Get to many samples here. Why?
        # FIXME: Also a bit slow?
