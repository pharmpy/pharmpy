from pharmpy.modeling import (
    plot_individual_predictions,
    plot_iofv_vs_iofv,
    plot_transformed_eta_distributions,
)


def test_plot_iofv_vs_iofv(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    iofv = model.modelfit_results.individual_ofv
    assert plot_iofv_vs_iofv(iofv, iofv, "run1", "run1")


def test_plot_individual_predictions(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    plot = plot_individual_predictions(model, model.modelfit_results.predictions)
    assert plot
    plot = plot_individual_predictions(
        model, model.modelfit_results.predictions[['PRED']], individuals=[1, 2, 5]
    )
    assert plot


def test_plot_transformed_eta_distributions(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'qa' / 'boxcox.mod')
    pe = model.modelfit_results.parameter_estimates
    ie = model.modelfit_results.individual_estimates
    plot = plot_transformed_eta_distributions(model, pe, ie)
    assert plot
