from pharmpy.workflows import LocalDirectoryContext, LocalDirectoryProject


def test_init(tmp_path):
    proj = LocalDirectoryProject('myproject', tmp_path)
    assert proj.path == tmp_path / 'myproject'
    assert (proj.path / '.modeldb').is_dir()
    assert repr(proj) == f'<Local directory project at {tmp_path / "myproject"}>'

    proj2 = LocalDirectoryProject('myproject', tmp_path)
    assert proj2.path == proj.path


def test_init_already_existing_dir(tmp_path):
    proj_path = tmp_path / 'myproject'
    assert not proj_path.is_dir()
    proj_path.mkdir()
    assert not (proj_path / '.modeldb').is_dir()
    proj = LocalDirectoryProject('myproject', tmp_path)
    assert proj.path == tmp_path / 'myproject'
    assert (proj_path / '.modeldb').is_dir()


def test_store_model_existing(tmp_path, load_example_model_for_test):
    proj = LocalDirectoryProject('myproject', tmp_path)
    ctx = LocalDirectoryContext(
        name='mycontext', ref=proj.get_context_ref(None), model_database=proj.model_database
    )
    model = load_example_model_for_test("pheno")
    ctx.store_model_entry(model)
    me1 = ctx.retrieve_model_entry("pheno")
    ctx.store_model_entry(model)
    me2 = ctx.retrieve_model_entry("pheno")
    assert me1.model == me2.model


def test_get_context_ref(tmp_path):
    proj = LocalDirectoryProject('myproject', tmp_path)
    assert proj.get_context_ref(None) == str(tmp_path / 'myproject')
    ref = 'path/to/ctx'
    assert proj.get_context_ref(ref) == str(tmp_path / 'myproject' / ref)
