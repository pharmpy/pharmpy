from pharmpy.workflows.hashing import DatasetHash, ModelHash


def test_hash(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    h = ModelHash(model)
    assert str(h) == "Ov1wUMumvRAxL0WKygUhXacZ2pnRKmbr-mNpGRUuaBI"
    d = DatasetHash(model.dataset)
    assert str(d) == h.dataset_hash
