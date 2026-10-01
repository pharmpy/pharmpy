from pharmpy.deps.lazy import LazyImport


def test_dir():
    import pharmpy.deps.lazy

    module = LazyImport('x', {}, 'pharmpy.deps.lazy')
    assert dir(pharmpy.deps.lazy) == dir(module)


def test_getattr():
    module = LazyImport('x', {}, 'pharmpy.deps.lazy')
    assert module.LazyImport is LazyImport


def test_submodule():
    import os

    module = LazyImport('x', {}, 'os', attr='path')
    assert dir(module) == dir(os.path)
