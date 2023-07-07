from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import fit, run_structsearch


def test_pkpd(tmp_path, load_model_for_test, testdata):
    with chdir(tmp_path):
        model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
        start_res = fit(model)
        res = run_structsearch(type='pkpd', route='iv', results=start_res, model=model)

        assert len(res.summary_models) > 1
        assert len(res.summary_tool) > 1
        assert len(res.models) > 1
        assert res is not None

        rundir = tmp_path / 'structsearch_dir1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()
