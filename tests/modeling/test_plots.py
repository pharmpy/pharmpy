from pharmpy.modeling import plot_individual_predictions, plot_iofv_vs_iofv


def test_plot_iofv_vs_iofv(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    iofv = model.modelfit_results.individual_ofv
    assert plot_iofv_vs_iofv(iofv, iofv, "run1", "run1")


def test_plot_individual_predictions(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    plot = plot_individual_predictions(model)
    assert plot
    plot = plot_individual_predictions(model, predictions=['PRED'], individuals=[1, 2, 5])
    assert plot
