from pharmpy.lazyload import LazyLoader


def test_dir():
    import pharmpy.lazyload

    lazyload = LazyLoader('x', {}, 'pharmpy.lazyload')
    assert dir(pharmpy.lazyload) == dir(lazyload)


def test_getattr():
    lazyload = LazyLoader('x', {}, 'pharmpy.lazyload')
    assert getattr(lazyload, 'LazyLoader') is LazyLoader


def test_submodule():
    import os

    loader = LazyLoader('x', {}, 'os', attr='path')
    assert dir(loader) == dir(os.path)
