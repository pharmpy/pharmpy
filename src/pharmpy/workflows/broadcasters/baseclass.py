from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

BROADCASTERS = ('terminal', 'null')
DEFAULT_BROADCASTER = 'terminal'


class Broadcaster(ABC):
    @staticmethod
    def canonicalize_broadcaster_name(name: Optional[str]) -> str:
        if name is None:
            canon_name = DEFAULT_BROADCASTER
        else:
            canon_name = name.lower()
        if canon_name not in BROADCASTERS:
            raise ValueError(f"Unknown broadcaster {name}")
        return canon_name

    @staticmethod
    def select_broadcaster(name: str) -> Broadcaster:
        """Create a new broadcaster given a broadcaster name"""
        if name == 'null':
            from pharmpy.workflows.broadcasters.null import NullBroadcaster

            broadcaster = NullBroadcaster
        else:  # 'terminal'
            from pharmpy.workflows.broadcasters.terminal import TerminalBroadcaster

            broadcaster = TerminalBroadcaster
        return broadcaster

    @abstractmethod
    def broadcast_message(severity, ctxpath, date, message) -> None:
        pass
