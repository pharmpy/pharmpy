from collections.abc import Callable
from typing import Optional

BROADCASTERS = ('terminal', 'null')
DEFAULT_BROADCASTER = 'terminal'


def canonicalize_broadcaster_name(name: Optional[str]) -> str:
    if name is None:
        canon_name = DEFAULT_BROADCASTER
    else:
        canon_name = name.lower()
    if canon_name not in BROADCASTERS:
        raise ValueError(f"Unknown broadcaster {name}")
    return canon_name


def select_broadcaster(name: str) -> Callable:
    if name == 'null':
        from pharmpy.workflows.broadcasters.null import broadcast_message
    else:  # 'terminal'
        from pharmpy.workflows.broadcasters.terminal import broadcast_message
    return broadcast_message
