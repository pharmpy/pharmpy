import pharmpy.workflows as workflows


class Tool:
    def __init__(self, dispatcher=None, database=None, path=None):
        toolname = type(self).__name__.lower()
        if dispatcher is None:
            self.dispatcher = workflows.default_dispatcher
        else:
            self.dispatcher = dispatcher
        if database is None:
            self.database = workflows.default_tool_database(toolname=toolname, path=path)
        else:
            self.database = database
