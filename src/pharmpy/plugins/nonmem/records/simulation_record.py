"""
NONMEM simulation record class.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import cast

from pharmpy.internals.parse.generic import eval_token

from .record import ReplaceableRecord, with_parsed_and_generated


@with_parsed_and_generated
@dataclass(frozen=True)
class SimulationRecord(ReplaceableRecord):
    @cached_property
    def nsubs(self):
        """Number of subproblems"""
        n = cast(int, eval_token(self.tree.subtree('nsubs').leaf('INT')))
        # NOTE According to NONMEM documentation 0 means 1 here
        return 1 if n == 0 else n
