from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .option_record import OptionRecord
from .record import with_parsed_and_generated


@with_parsed_and_generated
@dataclass(frozen=True)
class EtasRecord(OptionRecord):
    @cached_property
    def path(self):
        file_option = self.option_pairs['FILE']
        assert file_option is not None
        return Path(file_option)

    def replace_path(self, value):
        return self.set_option('FILE', str(value))
