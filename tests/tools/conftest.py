import pytest

from pharmpy.tools.external.dummy.run import create_dummy_modelfit_results
from pharmpy.workflows import ModelEntry


@pytest.fixture(scope='session')
def model_entry_factory():
    def _model_entry_factory(models, ref_val=None, parent=None):
        model_entries = []
        for i, model in enumerate(models):
            res = create_dummy_modelfit_results(model, ref=ref_val + i if ref_val else ref_val)
            model_entries.append(ModelEntry.create(model, modelfit_results=res, parent=parent))
        return model_entries

    return _model_entry_factory
