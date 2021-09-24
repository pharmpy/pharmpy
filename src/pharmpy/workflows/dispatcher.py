# The Dispatcher is the link between the local machine and the execution system.
# The run method should:
# 1. Create and cd into a temp directory
# 2. Copy the infiles from the local machine to the temp directory
# 3. Execute the workflow
# 4. Put the outfiles into the database


class ExecutionDispatcher:
    def run(self, workflow, database, infiles=None, outfiles=None):
        raise NotImplementedError()
