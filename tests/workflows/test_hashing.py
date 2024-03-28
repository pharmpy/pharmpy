from pharmpy.workflows.hashing import DatasetHash, ModelHash


def test_hash(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    h = ModelHash(model)
    assert str(h) == "UxuftgoIjoP2-6kip0sfmaLxbTGKDbhwEZvuU31MP_A"
    d = DatasetHash(model.dataset)
    assert str(d) == h.dataset_hash
