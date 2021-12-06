from pharmpy.tools.rankfuncs import aic, bic, ofv


class DummyResults:
    def __init__(self, ofv=None, aic=None, bic=None, parameter_estimates=None):
        self.ofv = ofv
        self.aic = aic
        self.bic = bic
        self.parameter_estimates = parameter_estimates

    def __bool__(self):
        return bool(self.ofv) and bool(self.parameter_estimates)


class DummyModel:
    def __init__(self, name, **kwargs):
        self.name = name
        if 'no_modelfit_results' in kwargs.keys():
            self.modelfit_results = None
        else:
            self.modelfit_results = DummyResults(**kwargs)


def test_ofv():
    run1 = DummyModel("run1", ofv=0)
    run2 = DummyModel("run2", ofv=-1)
    run3 = DummyModel("run3", ofv=-14)
    res = ofv(run1, [run2, run3])
    assert [run3] == res

    run4 = DummyModel("run4", ofv=2)
    run5 = DummyModel("run5", ofv=-2)
    res = ofv(run1, [run2, run3, run4, run5], cutoff=2)
    assert [run3, run5] == res

    run6 = DummyModel("run6", no_modelfit_results=True)
    res = ofv(run6, [run1, run2, run3])
    assert [] == res

    run7 = DummyModel("run7", no_modelfit_results=True)
    run8 = DummyModel("run8", no_modelfit_results=True)
    res = ofv(run6, [run7, run8])
    assert [] == res


def test_aic():
    run1 = DummyModel("run1", aic=0)
    run2 = DummyModel("run2", aic=-1)
    run3 = DummyModel("run3", aic=-14)
    res = aic(run1, [run2, run3])
    assert [run3] == res


def test_bic():
    run1 = DummyModel("run1", bic=0)
    run2 = DummyModel("run2", bic=-1)
    run3 = DummyModel("run3", bic=-14)
    res = bic(run1, [run2, run3])
    assert [run3] == res
