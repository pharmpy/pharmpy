from pharmpy.modeling import set_initial_estimates
from pharmpy.workflows.hashing import DatasetHash, ModelHash


def test_hash(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    h = ModelHash(model)
    assert str(h) == "Q1IeIf2PVAiya2ghqLshv7JSRZD-CG_Q2T_FMPvD6Zk"
    d = DatasetHash(model.dataset)
    assert str(d) == h.dataset_hash

    # name should not change hash
    m2 = model.replace(name="SomeOtherName")
    assert str(h) == str(ModelHash(m2))

    # description should not change hash
    m3 = model.replace(description="MyNewDescr")
    assert str(h) == str(ModelHash(m3))

    # changing init of a parameter should change hash
    m4 = set_initial_estimates(model, {'IVV': 0.99})
    assert str(h) != str(ModelHash(m4))
