from typing import TYPE_CHECKING

from pharmpy.internals.module.lazy import LazyImport

if TYPE_CHECKING:
    import rich.box as box
    import rich.console as console
    import rich.markup as markup
    import rich.table as table
    import rich.text as text
else:
    box = LazyImport('box', globals(), 'rich.box')
    console = LazyImport('console', globals(), 'rich.console')
    table = LazyImport('table', globals(), 'rich.table')
    text = LazyImport('text', globals(), 'rich.text')
    markup = LazyImport('markup', globals(), 'rich.markup')
