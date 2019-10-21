class Record:
    """
    Top level class for records.

    Create objects only by using the factory function create_record.
    """

    name = None

    def __str__(self):
        try:
            return self.raw_name
        except AttributeError:
            return ''
