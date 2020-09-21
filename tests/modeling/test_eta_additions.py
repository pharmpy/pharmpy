from pharmpy import Model
from pharmpy.modeling import add_etas


def test_apply(pheno_path):
    model = Model(pheno_path)
    add_etas(model, 'S1', 'exp', '+')
    model.update_source()
    print(model)
