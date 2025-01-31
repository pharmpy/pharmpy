from pharmpy.deps import rich

from .baseclass import Broadcaster


class TerminalBroadcaster(Broadcaster):
    def broadcast_message(self, severity, ctxpath, date, message):
        grid = rich.table.Table.grid(expand=True)
        grid.add_column(ratio=3, overflow="fold")
        grid.add_column(ratio=2)
        grid.add_column(ratio=1)
        grid.add_column(ratio=5)

        datestr = date.strftime("%Y-%m-%d %H:%M:%S")
        sevstr = rich.text.Text(severity.upper())
        if severity in ("error", "critical"):
            sevstr.stylize("red")
        elif severity == "info":
            sevstr.stylize("green")
        elif severity == "warning":
            sevstr.stylize("yellow")
        grid.add_row(
            rich.markup.escape(ctxpath),
            rich.markup.escape(datestr),
            sevstr,
            rich.markup.escape(message),
        )

        console = rich.console.Console()
        console.print(grid)
