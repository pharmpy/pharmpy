class CovariateEffect:
    def __init__(self, template, data):
        self.template = template
        self.data = data

    def apply(self, parameter, covariate, median, mean):
        pass

    @classmethod
    def exponential(cls):
        template = None
        data = None
        return cls(template, data)

    def get_statistics(self, statistic_type, column):
        return statistic_type(self.data[column])
