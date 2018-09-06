class RunDirectory:
    """ Base class for the execution directory.
        This directory will contain files from a job.
        Cleaning will be handled here. 
        Subclasses can be used by different tools
        Generation can be done by the engine class to get specific directories
    """
    def path(self):
        """ The path of the directory
            This can be implemented directly here
        """
        raise NotImplementedError

    def clean(self, level):
        """ Clean out non-needed files after a run
        """
        raise NotImplementedError
