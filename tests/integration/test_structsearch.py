import os
import sys

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import convert_model, create_basic_pk_model, filter_dataset
from pharmpy.tools import fit, run_structsearch


def test_pkpd(tmp_path, load_model_for_test, testdata):
    if os.name == 'nt' or sys.platform == 'darwin':  # needs to be fixed. Issue #1967
        pytest.skip("TODO Fails on Windows and Mac, temporarily skipping.")
    with chdir(tmp_path):
        model = create_basic_pk_model('iv', dataset_path=testdata / "nonmem" / "pheno_pd.csv")
        model = convert_model(model, 'nonmem')
        pk_model = filter_dataset(model, "DVID != 2")
        pk_res = fit(pk_model)
        res = run_structsearch(
            type='pkpd',
            search_space="DIRECTEFFECT(*)",
            results=pk_res,
            model=model,
            b_init=0.1,
            emax_init=0.1,
            ec50_init=0.1,
            met_init=0.1,
        )

        no_of_models = 3
        assert len(res.summary_models) == no_of_models + 2
        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.models) == no_of_models + 1

        rundir = tmp_path / 'structsearch1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_drug_metabolite(tmp_path, load_model_for_test, testdata):
    with chdir(tmp_path):
        model = create_basic_pk_model('oral', dataset_path=testdata / "nonmem" / "pheno_pd.csv")
        model = convert_model(model, 'nonmem')
        modelres = fit(model)

        res = run_structsearch(
            model=model,
            results=modelres,
            type="drug_metabolite",
            search_space="METABOLITE(PSC);PERIPHERALS(0..1,MET)",
        )

        no_of_models = 2
        assert len(res.summary_models) == no_of_models + 1
        assert len(res.summary_tool) == no_of_models
        assert len(res.models) == no_of_models

        rundir = tmp_path / 'structsearch1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()
