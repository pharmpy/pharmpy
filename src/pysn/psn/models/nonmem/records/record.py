from . import records_list

class Record:
    """ Top level class for records
        Create objects only by using the factory function create_record
    """
    def __str__(self):
        return "$" + self.raw_name
