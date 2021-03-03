import pharmpy.execute as execute
import pharmpy.methods
import pharmpy.methods.modelfit as modelfit
import pharmpy.modeling as modeling
import pharmpy.results
import pharmpy.search.algorithms as algorithms
import pharmpy.search.rankfuncs as rankfuncs


class ModelSearch(pharmpy.methods.Method):
    def __init__(self, base_model, **kwargs):
        self.base_model = base_model
        super().__init__(**kwargs)

    def fit(self, models):
        db = execute.LocalDirectoryDatabase(self.rundir.path / 'models')
        modelfit_run = modelfit.Modelfit(models, database=db, path=self.rundir.path)
        modelfit_run.run()

    def run(self):
        df = algorithms.exhaustive(
            self.base_model,
            [modeling.add_peripheral_compartment, modeling.first_order_absorption],
            self.fit,
            rankfuncs.ofv,
        )
        print(df)


class ModelSearchResults(pharmpy.results.Results):
    def __init__(self, models=None):
        self.models = models
