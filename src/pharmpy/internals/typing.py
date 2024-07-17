from __future__ import annotations

import typing

# This is to be able to use the typing.Self which was available only from Python 3.11
# Remove this when Python 3.10 is no longer supported

if typing.TYPE_CHECKING:
    Self = typing.Self
else:
    Self = typing.Any
