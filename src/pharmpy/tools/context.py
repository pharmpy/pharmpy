from pharmpy.workflows.context import LocalDirectoryContext


def create_context(name: str, path=None):
    """Create a new context

    Currently a local filesystem context
    """
    ref = str(path) if path is not None else None
    ctx = LocalDirectoryContext(name=name, ref=ref)
    return ctx
