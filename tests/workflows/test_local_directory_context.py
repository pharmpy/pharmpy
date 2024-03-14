from pharmpy.workflows import LocalDirectoryContext


def test_init(tmp_path):
    path = tmp_path / 'mycontext'
    ctx = LocalDirectoryContext(path=path)
    assert ctx.path == path
    assert (ctx.path / 'models').is_dir()
    assert (ctx.path / '.modeldb').is_dir()
    assert (ctx.path / 'log.csv').is_file()
    assert (ctx.path / '.annotations').is_file()
    assert ctx.context_path == 'mycontext'

    subctx = LocalDirectoryContext(name="mysubcontext", parent=ctx)
    assert (subctx.path / 'models').is_dir()
    assert not (subctx.path / '.modeldb').is_dir()
    assert not (subctx.path / 'log.csv').is_file()
    assert (subctx.path / '.annotations').is_file()
    assert subctx.context_path == 'mycontext/mysubcontext'

    existing_ctx = LocalDirectoryContext(path=path)
    assert existing_ctx.path == ctx.path
