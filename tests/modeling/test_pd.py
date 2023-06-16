from pharmpy.modeling import set_direct_effect


def test_set_direct_effect(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    model = set_direct_effect(model, 'step')
    # print(print_model_code(model))
    print(model.statements)
    assert False
