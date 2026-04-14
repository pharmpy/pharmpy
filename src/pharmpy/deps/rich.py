from typing import TYPE_CHECKING

from pharmpy.internals.module.lazy import LazyImport

if TYPE_CHECKING:
    import rich.box as box
    import rich.columns as columns
    import rich.console as console
    import rich.markup as markup
    import rich.panel as panel
    import rich.table as table
    import rich.text as text
else:
    box = LazyImport('box', globals(), 'rich.box')
    columns = LazyImport('columns', globals(), 'rich.columns')
    console = LazyImport('console', globals(), 'rich.console')
    panel = LazyImport('panel', globals(), 'rich.panel')
    table = LazyImport('table', globals(), 'rich.table')
    text = LazyImport('text', globals(), 'rich.text')
    markup = LazyImport('markup', globals(), 'rich.markup')
