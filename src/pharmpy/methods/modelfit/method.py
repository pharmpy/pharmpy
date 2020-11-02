import pharmpy.methods

from .run import run


class Modelfit(pharmpy.methods.Method):
    def __init__(self, models, **kwargs):
        self.models = models
        super().__init__(**kwargs)

    def run(self):
        run(self.models, self.rundir.path)
        res = self.models[0].modelfit_results
        res.to_json(path=self.rundir.path / 'results.json')
        res.to_csv(path=self.rundir.path / 'results.csv')
