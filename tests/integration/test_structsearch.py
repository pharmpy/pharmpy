from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import convert_model, create_basic_pk_model  # , filter_dataset

# from pharmpy.tools import fit, run_structsearch


def test_pkpd(tmp_path, load_model_for_test, testdata):
    with chdir(tmp_path):
        model = create_basic_pk_model('iv', dataset_path=testdata / "nonmem" / "pheno_pd.csv")
        model = convert_model(model, 'nonmem')
    #    pk_model = filter_dataset(model, "DVID != 2")  # NOTE: This step needs to be removed later
    #    pk_res = fit(pk_model)
    #    res = run_structsearch(type='pkpd', search_space="DIRECTEFFECT(*)",results=pk_res, model=model)

    #    no_of_models = 9
    #    assert len(res.summary_models) == no_of_models + 1
    #    assert len(res.summary_tool) == no_of_models + 1
    #    assert len(res.models) == no_of_models

    #    rundir = tmp_path / 'structsearch_dir1'
    #    assert rundir.is_dir()
    #    assert (rundir / 'results.json').exists()
    #    assert (rundir / 'results.csv').exists()
    #    assert (rundir / 'metadata.json').exists()
