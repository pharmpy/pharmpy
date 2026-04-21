from typing import Self


class Record:
    """
    Top level class for records.

    Create objects only by using the factory function create_record.
    """

    def __init__(self, name, raw_name, root):
        self.name = name
        self.raw_name = raw_name
        self._root = root

    def set_name(self, name):
        return self.__class__(name, '$' + name, self._root)

    def replace(self, root):
        return self.__class__(self.name, self.raw_name, root)

    @classmethod
    def create(cls, name: str, content: str) -> Self:
        from .factory import known_records

        parser = known_records[name][1]
        root = parser(content).root
        record = cls(name, '$' + name, root)
        return record

    @property
    def root(self):
        """Root of the parse tree"""
        return self._root

    def __str__(self):
        assert self.raw_name is not None
        return self.raw_name + str(self.root)
