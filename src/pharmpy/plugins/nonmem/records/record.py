class Record:
    """
    Top level class for records.

    Create objects only by using the factory function create_record.
    """

    name = None

    def __init__(self, content, parser_class):
        self._content = content
        self._parser_class = parser_class
        self._root = None

    @property
    def root(self):
        """Root of the parse tree
        only parse on demand
        """
        if self._root is None:
            try:
                parser = self._parser_class(self._content)
            except AttributeError:
                assert self._root is not None
                return self._root

            self._root = parser.root

            try:
                del self._parser_class
            except AttributeError:
                pass
            try:
                del self._content
            except AttributeError:
                pass

        return self._root

    @root.setter
    def root(self, root_new):
        self._root = root_new

    def __str__(self):
        return self.raw_name + str(self.root)
