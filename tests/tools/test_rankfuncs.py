from pharmpy.model import Parameters
from pharmpy.modeling import load_example_model
from pharmpy.tools.rankfuncs import aic, ofv

pheno = load_example_model("pheno")


class DummyResults:
    def __init__(self, ofv=None, parameter_estimates=None):
        self.ofv = ofv
        self.parameter_estimates = parameter_estimates

    def __bool__(self):
        return bool(self.ofv) and bool(self.parameter_estimates)


class DummyModel:
    def __init__(self, name, parameters=None, **kwargs):
        self.name = name
        self.parameters = parameters
        self.dataset = pheno.dataset
        self.datainfo = pheno.datainfo
        if 'no_modelfit_results' in kwargs.keys():
            self.modelfit_results = None
        else:
            self.modelfit_results = DummyResults(**kwargs)


def test_ofv():
    run1 = DummyModel("run1", ofv=0)
    run2 = DummyModel("run2", ofv=-1)
    run3 = DummyModel("run3", ofv=-14)
    res, _ = ofv(run1, [run2, run3])
    assert [run3] == res

    run4 = DummyModel("run4", ofv=2)
    run5 = DummyModel("run5", ofv=-2)
    res, _ = ofv(run1, [run2, run3, run4, run5], cutoff=2)
    assert [run3, run5] == res

    run6 = DummyModel("run6", no_modelfit_results=True)
    res, _ = ofv(run6, [run1, run2, run3])
    assert [] == res

    run7 = DummyModel("run7", no_modelfit_results=True)
    run8 = DummyModel("run8", no_modelfit_results=True)
    res, _ = ofv(run6, [run7, run8])
    assert [] == res


def test_aic():
    run1 = DummyModel("run1", ofv=0, parameters=Parameters())
    run2 = DummyModel("run2", ofv=-1, parameters=Parameters())
    run3 = DummyModel("run3", ofv=-14, parameters=Parameters())
    res, _ = aic(run1, [run2, run3], cutoff=3.84)
    assert [run3] == res
    res, _ = aic(run1, [run2, run3])
    assert [run3, run2] == res
