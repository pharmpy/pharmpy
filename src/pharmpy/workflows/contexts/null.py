from typing import Optional

from pharmpy.deps import pandas as pd

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

    def retrieve_results(self):  # pyright: ignore [reportIncompatibleMethodOverride]
        pass

    def read_metadata(self):
        pass

    @staticmethod
    def exists(name: str, ref: Optional[str] = None) -> bool:
        return True

    def store_key(self, *args, **kwargs):
        pass

    def retrieve_key(self, name):  # pyright: ignore [reportIncompatibleMethodOverride]
        pass

    def list_all_names(self):
        return []

    def list_all_subcontexts(self):
        return []

    def store_annotation(self, name, annotation):
        pass

    def retrieve_annotation(self, name):
        return ""

    def store_message(self, severity, ctxpath, date, message):
        pass

    def retrieve_log(self, *args, **kwargs):
        return pd.DataFrame()

    def retrieve_common_options(self):
        return {}

    def retrieve_dispatching_options(self):
        # Add new options as needed for testing
        return {'ncores': 1}

    def get_parent_context(self) -> Context:
        return self

    def get_top_level_context(self) -> Context:
        return self

    def get_subcontext(self, name) -> Context:
        return self

    def create_subcontext(self, name) -> Context:
        return self

    def finalize(self):
        pass

    def abort_workflow(self, message):
        pass
