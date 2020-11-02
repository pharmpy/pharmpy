import pharmpy.methods
import pharmpy.methods.modelfit as modelfit
from pharmpy.data.iterators import Resample


class Bootstrap(pharmpy.methods.Method):
    def __init__(self, model):
        self.model = model
        super().__init__()

    def run(self):
        resample = Resample(self.model, self.model.dataset.pharmpy.id_label, name_pattern='bs_{}')
        models = []
        for (remod, groups) in resample:
            remod.write(path=self.rundir.models_path)  # FIXME: Automatic save to models?
            models.append(remod)
        modelfit_run = modelfit.Modelfit(models, path=self.rundir.path)
        modelfit_run.run()

        # run(self.models, self.rundir.path)
        # res = self.models[0].modelfit_results
        # res.to_json(path=self.rundir.path / 'results.json')
        # res.to_csv(path=self.rundir.path / 'results.csv')
