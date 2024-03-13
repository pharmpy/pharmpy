from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools.linearize.tool import create_linearized_model


def test_create_linearized_model(tmp_path, load_model_for_test, testdata):
    with chdir(tmp_path):
        path = testdata / 'nonmem' / 'pheno_real.mod'
        model = load_model_for_test(path)
        linear_results = create_linearized_model(model)
        assert len(linear_results.final_model.statements) == 9
