from pharmpy import Model
from pharmpy.modeling import load_example_model, plot_individual_predictions, plot_iofv_vs_iofv


def test_plot_iofv_vs_iofv():
    model = load_example_model('pheno')
    assert plot_iofv_vs_iofv(model, model)


def test_plot_individual_predictions(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real.mod')
    plot = plot_individual_predictions(model)
    assert plot
    plot = plot_individual_predictions(model, predictions=['PRED'], individuals=[1, 2, 5])
    assert plot
