class CovariateEffect:
    def __init__(self, template):
        self.template = template

    def apply(self, parameter, covariate, median, mean):
        pass

    @classmethod
    def exponential(cls):
        template = None
        return cls(template)
