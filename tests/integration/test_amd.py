import shutil
from pathlib import Path

from pharmpy import Model
from pharmpy.modeling import run_iiv
from pharmpy.utils import TemporaryDirectoryChanger

# def test_amd(tmp_path, testdata):
#     with TemporaryDirectoryChanger(tmp_path):
#         shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
#         dipath = Path(pharmpy.modeling.__file__).parent / 'example_models' / 'pheno.datainfo'
#         shutil.copy2(dipath, tmp_path)
#
#         res = run_amd(tmp_path / 'pheno.dta', mfl='LAGTIME();PERIPHERALS(1)')
#         assert res


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


def test_iiv(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mox_simulated_normal.csv'

        res = run_iiv(model_start)

        assert len(res.summary_tool) == 2
        assert len(res.summary_tool[0]) == 8
        assert len(res.summary_tool[1]) == 5
        assert len(res.summary_models) == 12
        assert len(res.models) == 11
        rundir1 = tmp_path / 'iiv_dir1'
        assert rundir1.is_dir()
        assert _model_count(rundir1) == 8
        rundir2 = tmp_path / 'iiv_dir2'
        assert rundir2.is_dir()
        assert _model_count(rundir2) == 4

        run_iiv(model_start, path=tmp_path / 'test_path')

        rundir1 = tmp_path / 'test_path' / 'iiv1'
        assert rundir1.is_dir()
        rundir2 = tmp_path / 'test_path' / 'iiv2'
        assert rundir2.is_dir()
