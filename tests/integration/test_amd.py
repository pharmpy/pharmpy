from pathlib import Path

from pharmpy.modeling import run_iiv
from pharmpy.utils import TemporaryDirectoryChanger

# def test_amd(tmp_path, testdata):
#     with TemporaryDirectoryChanger(tmp_path):
#         shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
#         dipath = Path(pharmpy.modeling.__file__).parent / 'example_models' / 'pheno.datainfo'
#         shutil.copy2(dipath, tmp_path)
#         res = run_amd(tmp_path / 'pheno.dta', mfl='LAGTIME();PERIPHERALS(1)')
#         assert res


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


def test_iivsearch(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_iiv(start_model)

        assert len(res.summary_tool) == 2
        assert len(res.summary_tool[0]) == 8
        assert len(res.summary_tool[1]) == 5
        assert len(res.summary_models) == 12
        assert len(res.models) == 11
        rundir1 = tmp_path / 'iivsearch_dir1'
        assert rundir1.is_dir()
        assert _model_count(rundir1) == 7
        rundir2 = tmp_path / 'iivsearch_dir2'
        assert rundir2.is_dir()
        assert _model_count(rundir2) == 4

        run_iiv(start_model, path=tmp_path / 'test_path')

        rundir1 = tmp_path / 'test_path' / 'iiv1'
        assert rundir1.is_dir()
        rundir2 = tmp_path / 'test_path' / 'iiv2'
        assert rundir2.is_dir()
