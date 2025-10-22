from datetime import datetime
from typing import Literal

from .baseclass import Broadcaster


class NullBroadcaster(Broadcaster):
    def broadcast_message(
        self,
        severity: Literal["critical", "error", "warning", "info", "trace"],
        ctxpath: str,
        date: datetime,
        message: str,
    ) -> None:
        pass
