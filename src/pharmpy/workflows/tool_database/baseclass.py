class ToolDatabase:
    """Database of results, metadata and run files for one run of a tool

    This corresponds to the run directory of PsN

     Attribute: model_database
    """

    def __init__(self, toolname):
        self.toolname = toolname
