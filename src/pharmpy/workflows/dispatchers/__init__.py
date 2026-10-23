# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from pharmpy import config

from .baseclass import Dispatcher


class DispatcherConfiguration(config.Configuration):
    module = 'pharmpy.workflows.dispatchers'
    dask_dispatcher = config.ConfigItem(
        None,
        'Which type of dask scheduler to use (supports threaded and distributed).',
        str,
    )


conf = DispatcherConfiguration()

__all__ = ('Dispatcher',)
