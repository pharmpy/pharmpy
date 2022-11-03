from pharmpy.internals.module.lazy import LazyImport


def test_dir():
    import pharmpy.internals.module.lazy

    module = LazyImport('x', {}, 'pharmpy.internals.module.lazy')
    assert dir(pharmpy.internals.module.lazy) == dir(module)


def test_getattr():
    module = LazyImport('x', {}, 'pharmpy.internals.module.lazy')
    assert getattr(module, 'LazyImport') is LazyImport


def test_submodule():
    import os

    module = LazyImport('x', {}, 'os', attr='path')
    assert dir(module) == dir(os.path)
