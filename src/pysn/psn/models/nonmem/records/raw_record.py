from . import record

class RawRecord(record.Record):
    """ A record that just keeps it contents unparsed. I.e. $PROBLEM or unknown records
    """
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return super().__str__() + self.content
