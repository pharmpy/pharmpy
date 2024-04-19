from pharmpy.workflows.hashing import DatasetHash, ModelHash


def test_hash(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    h = ModelHash(model)
    assert str(h) == "Q1IeIf2PVAiya2ghqLshv7JSRZD-CG_Q2T_FMPvD6Zk"
    d = DatasetHash(model.dataset)
    assert str(d) == h.dataset_hash
