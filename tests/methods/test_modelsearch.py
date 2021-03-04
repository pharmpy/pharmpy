from pharmpy import Model
import pharmpy.methods.modelsearch as ms


def test_modelfit(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    tool = ms.ModelSearch(model, 'stepwise', ['add_peripheral_compartment()'])
    assert tool
