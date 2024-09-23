import pytest

from pharmpy.tools.external.dummy.run import create_dummy_modelfit_results
from pharmpy.workflows import ModelEntry


@pytest.fixture(scope='session')
def model_entry_factory():
    def _model_entry_factory(models, ref_val=None):
        model_entries = []
        seed = 0
        for model in models:
            while True:
                res = create_dummy_modelfit_results(model, seed)
                if ref_val and res.ofv > ref_val:
                    seed += 1
                else:
                    break
            model_entries.append(ModelEntry.create(model, modelfit_results=res))
            seed += 1
        return model_entries

    return _model_entry_factory
