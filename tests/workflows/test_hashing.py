from pharmpy.workflows.hashing import DatasetHash, ModelHash


def test_hash(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    h = ModelHash(model)
    assert str(h) == "AU1L1vFndOkq14kD6i66EkOoYIx2Wo0dnX37aPxtdJg"
    d = DatasetHash(model.dataset)
    assert str(d) == h.dataset_hash
