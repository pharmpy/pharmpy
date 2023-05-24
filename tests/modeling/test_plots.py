from pharmpy.modeling import (
    plot_individual_predictions,
    plot_iofv_vs_iofv,
    plot_transformed_eta_distributions,
)
from pharmpy.tools import read_modelfit_results


def test_plot_iofv_vs_iofv(testdata):
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    iofv = res.individual_ofv
    assert plot_iofv_vs_iofv(iofv, iofv, "run1", "run1")


def test_plot_individual_predictions(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    plot = plot_individual_predictions(model, res.predictions)
    assert plot
    plot = plot_individual_predictions(model, res.predictions[['PRED']], individuals=[1, 2, 5])
    assert plot


def test_plot_transformed_eta_distributions(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'qa' / 'boxcox.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'qa' / 'boxcox.mod')
    pe = res.parameter_estimates
    ie = res.individual_estimates
    plot = plot_transformed_eta_distributions(model, pe, ie)
    assert plot
