# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
