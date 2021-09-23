from pharmpy.modeling import load_example_model, plot_iofv_vs_iofv


def test_plot_iofv_vs_iofv():
    model = load_example_model('pheno')
    assert plot_iofv_vs_iofv(model, model)
