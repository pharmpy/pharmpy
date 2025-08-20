import pytest

from pharmpy.tools import export_model_files
from pharmpy.workflows import (
    LocalDirectoryContext,
    ModelEntry,
)


def test_export_model_files(tmp_path, load_model_for_test, testdata):
    ctx = LocalDirectoryContext(name='mycontext', ref=tmp_path)
    model1 = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    ctx.store_model_entry(model1)

    for path in (testdata / 'nonmem').glob('pheno.*'):
        ctx.model_database.store_local_file(model1, path)

    model2 = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    ctx.store_model_entry(model2)

    moxo_parent_dir = testdata / 'nonmem' / 'models'
    for path in moxo_parent_dir.glob('mox2.*'):
        ctx.model_database.store_local_file(model2, path)
    ctx.model_database.store_local_file(model2, moxo_parent_dir / 'mytab_mox2')

    me1 = ModelEntry(model=model1)
    ctx.store_input_model_entry(me1)
    me2 = ModelEntry(model=model2)
    ctx.store_final_model_entry(me2)

    export_model_files(ctx, destination_path=tmp_path)
    assert (tmp_path / 'pheno.ctl').exists()
    assert (tmp_path / 'mox2.ctl').exists()
    assert (tmp_path / 'mox2_mytab_mox2').exists()

    export_model_files(ctx, destination_path=tmp_path / 'export')
    assert (tmp_path / 'export' / 'pheno.ctl').exists()
    assert (tmp_path / 'export' / 'mox2.ctl').exists()
    assert (tmp_path / 'export' / 'mox2_mytab_mox2').exists()

    with pytest.raises(ValueError):
        export_model_files(ctx, destination_path=tmp_path / 'export')

    with pytest.raises(ValueError):
        export_model_files(ctx, destination_path=tmp_path / 'pheno.ctl')

    with pytest.raises(FileNotFoundError):
        export_model_files(ctx, destination_path=tmp_path / 'x' / 'y')

    with pytest.raises(ValueError):
        export_model_files(ctx, destination_path=tmp_path)

    export_model_files(ctx, destination_path=tmp_path, force=True)
    assert (tmp_path / 'pheno.ctl').exists()
    assert (tmp_path / 'mox2.ctl').exists()
    assert (tmp_path / 'mox2_mytab_mox2').exists()
