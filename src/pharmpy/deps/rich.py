from typing import TYPE_CHECKING

from pharmpy.internals.module.lazy import LazyImport

if TYPE_CHECKING:
    from rich import box, columns, console, markup, panel, table, text
else:
    box = LazyImport('box', globals(), 'rich.box')
    columns = LazyImport('columns', globals(), 'rich.columns')
    console = LazyImport('console', globals(), 'rich.console')
    panel = LazyImport('panel', globals(), 'rich.panel')
    table = LazyImport('table', globals(), 'rich.table')
    text = LazyImport('text', globals(), 'rich.text')
    markup = LazyImport('markup', globals(), 'rich.markup')
