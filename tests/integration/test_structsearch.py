from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import fit, run_structsearch
from pharmpy.tools.structsearch.pkpd import create_pk_model


def test_pkpd(tmp_path, load_model_for_test, testdata):
    with chdir(tmp_path):
        model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
        pk_model = create_pk_model(model)  # NOTE: this step needs to be removed later
        pk_res = fit(pk_model)
        res = run_structsearch(type='pkpd', route='iv', results=pk_res, model=model)

        no_of_models = 12
        assert len(res.summary_models) == no_of_models + 1
        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.models) == no_of_models

        rundir = tmp_path / 'structsearch_dir1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()
