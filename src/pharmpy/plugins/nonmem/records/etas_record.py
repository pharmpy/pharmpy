from dataclasses import dataclass
from pathlib import Path

from .option_record import OptionRecord


@dataclass(frozen=True)
class EtasRecord(OptionRecord):
    @property
    def path(self):
        file_option = self.option_pairs['FILE']
        assert file_option is not None
        return Path(file_option)

    def with_path(self, value):
        return self.set_option('FILE', str(value))
