from pharmpy.internals.module.lazy import LazyImport

box = LazyImport('box', globals(), 'rich.box')
console = LazyImport('console', globals(), 'rich.console')
table = LazyImport('table', globals(), 'rich.table')
