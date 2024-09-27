import pytest

from pharmpy.modeling import create_rng
from pharmpy.tools.external.dummy.run import create_dummy_modelfit_results
from pharmpy.workflows import ModelEntry
from pharmpy.workflows.results import ModelfitResults


@pytest.fixture(scope='session')
def model_entry_factory():
    def _model_entry_factory(models, ref_val=None):
        model_entries = []
        for i, model in enumerate(models):
            res = create_dummy_modelfit_results(model, seed=i)
            if ref_val:
                res_attrs = res.__dict__
                rng = create_rng(i)
                res_attrs['ofv'] = rng.uniform(2 * ref_val, ref_val)
                res = ModelfitResults(**res_attrs)
            model_entries.append(ModelEntry.create(model, modelfit_results=res))
        return model_entries

    return _model_entry_factory
