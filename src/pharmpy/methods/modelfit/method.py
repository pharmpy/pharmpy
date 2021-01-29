import pharmpy.methods

from .run import run


class Modelfit(pharmpy.methods.Method):
    def __init__(self, models, **kwargs):
        self.models = models
        super().__init__(**kwargs)

    def run(self):
        fit_models = run(self.models, self.rundir.path)
        for i in range(len(fit_models)):
            self.models[i].modelfit_results = fit_models[i].modelfit_results
        res = self.models[0].modelfit_results
        res.to_json(path=self.rundir.path / 'results.json')
        res.to_csv(path=self.rundir.path / 'results.csv')
