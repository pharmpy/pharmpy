from typing import Optional

from pharmpy.internals.parse import AttrTree


class Record:
    """
    Top level class for records.

    Create objects only by using the factory function create_record.
    """

    name: Optional[str] = None
    raw_name: Optional[str] = None

    def __init__(self, content, parser_class):
        self._root: AttrTree = parser_class(content).root

    @property
    def root(self):
        """Root of the parse tree"""
        return self._root

    @root.setter
    def root(self, root_new):
        self._root = root_new

    def __str__(self):
        assert self.raw_name is not None
        return self.raw_name + str(self.root)
