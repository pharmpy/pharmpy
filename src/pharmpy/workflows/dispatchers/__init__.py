import pharmpy.config as config


class DispatcherConfiguration(config.Configuration):
    module = 'pharmpy.workflows.dispatchers'
    dask_dispatcher = config.ConfigItem(
        None,
        'Which type of dask scheduler to use (supports threaded and distributed).',
        str,
    )


conf = DispatcherConfiguration()
