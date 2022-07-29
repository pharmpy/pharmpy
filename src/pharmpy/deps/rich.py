from ..lazyload import LazyLoader

box = LazyLoader('box', globals(), 'rich.box')
console = LazyLoader('console', globals(), 'rich.console')
table = LazyLoader('table', globals(), 'rich.table')
