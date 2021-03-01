from pharmpy.search.algorithms import exhaustive
from pharmpy.search.rankfuncs import ofv


class DummyResults:
    def __init__(self, ofv=None):
        self.ofv = ofv


class DummyModel:
    def __init__(self, name, ofv=None):
        self.name = name
        self.modelfit_results = DummyResults(ofv=ofv)

    def copy(self):
        return DummyModel(self.name, ofv=self.modelfit_results.ofv)


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


def test_exhaustive():
    base = DummyModel("run1", ofv=0)

    def do_nothing(model):
        return model

    trans = [do_nothing]
    res = exhaustive(base, trans, do_nothing, ofv)
    assert res == []

    def set_ofv(models):
        for i, model in enumerate(models):
            model.modelfit_results.ofv = -4 - i * 2

    res = exhaustive(base, trans, set_ofv, ofv)
    assert len(res) == 1
    assert res[0].modelfit_results.ofv == -4
