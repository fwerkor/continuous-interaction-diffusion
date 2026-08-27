from __future__ import annotations

import sys

if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042
        def __str__(self) -> str:
            return self.value
