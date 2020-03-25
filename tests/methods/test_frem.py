from pharmpy import Model
from pharmpy.methods.frem.results import FREMResults


def test_frem_results(testdata):
    model = Model(testdata / 'nonmem' / 'model_4.mod')
    res = FREMResults(model)
    print(res)
