from ..broadcasters.null import NullBroadcaster
from .baseclass import Context


class NullContext(Context):
    """Dummy context

    No operation does anything. This context can be used if no storing of files
    is desirable for example for testing.
    """

    def __init__(self, *args, **kwargs):
        self._broadcaster = NullBroadcaster()

    def __repr__(self) -> str:
        return "<NullContext>"

    @property
    def context_path(self) -> str:
        return ""

    def store_local_file(self, source_path):
        pass

    def store_results(self, res):
        pass

    def store_metadata(self, metadata):
        pass

    def retrieve_metadata(self):
        return {}

    def retrieve_results(self):
        pass

    def read_metadata(self):
        pass

    def exists(name, ref):
        pass

    def store_key(self, *args, **kwargs):
        pass

    def retrieve_key(self, *args):
        pass

    def list_all_names(self):
        pass

    def list_all_subcontexts(self):
        pass

    def store_annotation(self, name, annotation):
        pass

    def retrieve_annotation(self, name):
        pass

    def store_message(self, severity, ctxpath, date, message):
        pass

    def retrieve_log(self, *args, **kwargs):
        pass

    def retrieve_common_options(self):
        pass

    def retrieve_dispatching_options(self):
        # Add new options as needed for testing
        return {'ncores': 1}

    def get_parent_context(self):
        pass

    def get_top_level_context(self):
        pass

    def get_subcontext(self, name):
        pass

    def create_subcontext(self, name):
        pass

    def finalize(self):
        pass
