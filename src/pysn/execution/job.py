class Job:
    """ Base class for an Engine job that is running on an Environment
    """
    def has_finished(self):
        """ Check if a job has finished or not
        """
        raise NotImplementedError
